"""Local end-to-end smoke harness for development phases."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import httpx
import yaml

from .config_paths import PROJECT_ROOT


DEFAULT_LOCAL_SMOKE_HOST = "127.0.0.1"
DEFAULT_LOCAL_SMOKE_PORT = 18081
DEFAULT_LOCAL_SMOKE_UPSTREAM_URL = "http://localhost:11434"
DEFAULT_LOCAL_SMOKE_MODEL = "gemma3:1b"
DEFAULT_LOCAL_SMOKE_PROVIDER_NAME = "local-openai-smoke"
DEFAULT_LOCAL_SMOKE_PROMPT = "Reply with the single word smoke."
DEFAULT_LOCAL_SMOKE_TIMEOUT_SECONDS = 30.0
LOCAL_SMOKE_CHAT_PATH = "/v1/chat/completions"
LOCAL_SMOKE_LOGS_PATH = "/api/logs"
LOCAL_SMOKE_STATS_PATH = "/api/stats"
LOCAL_SMOKE_STATS_REQUIRED_KEYS = ("total_requests", "completed_requests", "pending_requests", "service_types")
DEFAULT_LOCAL_SMOKE_CONFIG_PATH = PROJECT_ROOT / "config" / "routes.local-smoke.yaml"


def build_local_smoke_config(
    upstream_url: str = DEFAULT_LOCAL_SMOKE_UPSTREAM_URL,
    model_name: str = DEFAULT_LOCAL_SMOKE_MODEL,
) -> dict[str, Any]:
    return {
        "routes": [],
        "providers": [
            {
                "name": DEFAULT_LOCAL_SMOKE_PROVIDER_NAME,
                "type": "openai",
                "enabled": True,
                "priority": 0,
                "url": upstream_url,
                "api_key": None,
                "timeout": 30.0,
                "static_models": [model_name],
                "metadata": {"description": "Local OpenAI-compatible smoke upstream"},
            }
        ],
        "strategy": {"type": "simple", "config": {"model_map": {}}},
        "access_control": {"type": "none", "config": {}},
    }


def build_local_smoke_request(
    model_name: str = DEFAULT_LOCAL_SMOKE_MODEL,
    prompt: str = DEFAULT_LOCAL_SMOKE_PROMPT,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 32,
    }


def extract_assistant_text(response_payload: Mapping[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Smoke response missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise ValueError("Smoke response choice is not an object")

    message = first_choice.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

    text = first_choice.get("text")
    if isinstance(text, str) and text.strip():
        return text

    raise ValueError("Smoke response missing assistant text")


def find_chat_completion_log(logs: list[Mapping[str, Any]], model_name: str) -> Optional[Mapping[str, Any]]:
    for log_entry in logs:
        if not isinstance(log_entry, Mapping):
            continue
        if log_entry.get("path") != LOCAL_SMOKE_CHAT_PATH:
            continue
        if model_name not in {log_entry.get("original_model"), log_entry.get("mapped_model")}:
            continue
        return log_entry

    return None


def build_local_smoke_env(
    base_env: Optional[Mapping[str, str]],
    *,
    routes_config: Path,
    blob_storage_path: Path,
    host: str = DEFAULT_LOCAL_SMOKE_HOST,
    port: int = DEFAULT_LOCAL_SMOKE_PORT,
) -> dict[str, str]:
    env = dict(base_env or {})
    env["APP_ENV"] = "dev"
    env["ENABLE_LOGGING"] = "true"
    env["ROUTES_CONFIG"] = str(routes_config)
    env["BLOB_STORAGE_PATH"] = str(blob_storage_path)
    env["LISTEN_HOST"] = host
    env["LISTEN_PORT"] = str(port)
    return env


def build_local_smoke_command(routes_config: Path, *, host: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "smolrouter",
        "-C",
        str(routes_config),
        "--host",
        host,
        "--port",
        str(port),
    ]


def _render_local_smoke_config(
    output_path: Path,
    *,
    upstream_url: str,
    model_name: str,
) -> None:
    config = build_local_smoke_config(upstream_url=upstream_url, model_name=model_name)
    output_path.write_text(yaml.safe_dump(config, sort_keys=False))


def _stop_process(process: Any, output_path: Path, timeout_seconds: float = 5.0) -> str:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout_seconds)

    if not output_path.exists():
        return ""

    return output_path.read_text(encoding="utf-8", errors="replace")


def _is_ready_stats_payload(payload: Any) -> bool:
    return isinstance(payload, Mapping) and all(key in payload for key in LOCAL_SMOKE_STATS_REQUIRED_KEYS)


def _tail_lines(output: str, max_lines: int = 20) -> str:
    if not output.strip():
        return "<no process output captured>"

    lines = output.strip().splitlines()
    return "\n".join(lines[-max_lines:])


def _wait_for_app_ready(base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Optional[BaseException] = None

    with httpx.Client(timeout=1.0) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(f"{base_url}{LOCAL_SMOKE_STATS_PATH}")
                if response.status_code != 200:
                    last_error = RuntimeError(f"Unexpected readiness status: {response.status_code}")
                    time.sleep(0.25)
                    continue

                stats_payload = response.json()
                if _is_ready_stats_payload(stats_payload):
                    return
                last_error = RuntimeError(f"Unexpected readiness payload: {stats_payload!r}")
            except httpx.HTTPError as exc:
                last_error = exc
            except ValueError as exc:
                last_error = exc
            time.sleep(0.25)

    raise TimeoutError(f"Timed out waiting for local smoke app at {base_url}") from last_error


def _wait_for_log_entry(base_url: str, model_name: str, timeout_seconds: float) -> Mapping[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: Any = None

    with httpx.Client(timeout=2.0) as client:
        while time.monotonic() < deadline:
            response = client.get(f"{base_url}{LOCAL_SMOKE_LOGS_PATH}", params={"limit": 10})
            response.raise_for_status()
            last_payload = response.json()
            if isinstance(last_payload, list):
                log_entry = find_chat_completion_log(last_payload, model_name)
                if log_entry is not None:
                    return log_entry
            time.sleep(0.25)

    raise RuntimeError(f"Smoke request did not appear in {LOCAL_SMOKE_LOGS_PATH}: {last_payload!r}")


def run_local_smoke(
    *,
    upstream_url: str = DEFAULT_LOCAL_SMOKE_UPSTREAM_URL,
    model_name: str = DEFAULT_LOCAL_SMOKE_MODEL,
    prompt: str = DEFAULT_LOCAL_SMOKE_PROMPT,
    host: str = DEFAULT_LOCAL_SMOKE_HOST,
    port: int = DEFAULT_LOCAL_SMOKE_PORT,
    timeout_seconds: float = DEFAULT_LOCAL_SMOKE_TIMEOUT_SECONDS,
) -> str:
    base_url = f"http://{host}:{port}"

    with tempfile.TemporaryDirectory(prefix="smolrouter-local-smoke-") as temp_dir:
        temp_root = Path(temp_dir)
        config_path = temp_root / DEFAULT_LOCAL_SMOKE_CONFIG_PATH.name
        blob_storage_path = temp_root / "blob_storage"
        output_path = temp_root / "smoke.log"
        blob_storage_path.mkdir(parents=True, exist_ok=True)

        _render_local_smoke_config(config_path, upstream_url=upstream_url, model_name=model_name)
        env = build_local_smoke_env(
            None,
            routes_config=config_path,
            blob_storage_path=blob_storage_path,
            host=host,
            port=port,
        )
        command = build_local_smoke_command(config_path, host=host, port=port)
        with output_path.open("w", encoding="utf-8") as process_output:
            process: Optional[subprocess.Popen[str]] = None

            try:
                process = subprocess.Popen(command, stdout=process_output, stderr=subprocess.STDOUT, text=True, env=env)
                _wait_for_app_ready(base_url, timeout_seconds)

                with httpx.Client(timeout=timeout_seconds) as client:
                    response = client.post(
                        f"{base_url}{LOCAL_SMOKE_CHAT_PATH}",
                        json=build_local_smoke_request(model_name=model_name, prompt=prompt),
                        headers={"Content-Type": "application/json", "Authorization": "Bearer local-smoke-key"},
                    )

                    if response.status_code != 200:
                        raise RuntimeError(
                            f"Smoke request failed with status {response.status_code}: {response.text.strip()}"
                        )

                    assistant_text = extract_assistant_text(response.json())

                log_entry = _wait_for_log_entry(base_url, model_name, timeout_seconds)
                if log_entry.get("status_code") != 200:
                    raise RuntimeError(f"Smoke log entry did not complete successfully: {log_entry!r}")

                _stop_process(process, output_path)
                return assistant_text
            except Exception as exc:
                output_text = _stop_process(process, output_path) if process is not None else ""
                output_tail = _tail_lines(output_text)
                raise RuntimeError(f"Local smoke failed: {exc}\n\nServer output tail:\n{output_tail}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a quick local SmolRouter smoke flow")
    parser.add_argument("--upstream-url", default=os.getenv("LOCAL_SMOKE_UPSTREAM_URL", DEFAULT_LOCAL_SMOKE_UPSTREAM_URL))
    parser.add_argument("--model", default=os.getenv("LOCAL_SMOKE_MODEL", DEFAULT_LOCAL_SMOKE_MODEL))
    parser.add_argument("--prompt", default=os.getenv("LOCAL_SMOKE_PROMPT", DEFAULT_LOCAL_SMOKE_PROMPT))
    parser.add_argument("--host", default=os.getenv("LOCAL_SMOKE_HOST", DEFAULT_LOCAL_SMOKE_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("LOCAL_SMOKE_PORT", str(DEFAULT_LOCAL_SMOKE_PORT))))
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("LOCAL_SMOKE_TIMEOUT_SECONDS", str(DEFAULT_LOCAL_SMOKE_TIMEOUT_SECONDS))),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    assistant_text = run_local_smoke(
        upstream_url=args.upstream_url,
        model_name=args.model,
        prompt=args.prompt,
        host=args.host,
        port=args.port,
        timeout_seconds=args.timeout,
    )
    print(f"Local smoke passed via {args.upstream_url} using {args.model}: {assistant_text.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())