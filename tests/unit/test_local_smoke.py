from pathlib import Path
import subprocess

import pytest
import yaml

import smolrouter.local_smoke as local_smoke
from smolrouter.local_smoke import (
    DEFAULT_LOCAL_SMOKE_CONFIG_PATH,
    DEFAULT_LOCAL_SMOKE_MODEL,
    DEFAULT_LOCAL_SMOKE_UPSTREAM_URL,
    LOCAL_SMOKE_CHAT_PATH,
    LOCAL_SMOKE_STATS_PATH,
    _is_ready_stats_payload,
    _stop_process,
    _wait_for_app_ready,
    _wait_for_log_entry,
    build_local_smoke_command,
    build_local_smoke_config,
    build_local_smoke_env,
    build_local_smoke_request,
    extract_assistant_text,
    find_chat_completion_log,
    run_local_smoke,
)


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, get_response=None, post_response=None):
        self.get_response = get_response
        self.post_response = post_response
        self.get_calls = []
        self.post_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, params=None):
        self.get_calls.append((url, params))
        return self.get_response

    def post(self, url, json=None, headers=None):
        self.post_calls.append((url, json, headers))
        return self.post_response


class _FakeProcess:
    def __init__(self, *, timeout_on_wait=False):
        self.timeout_on_wait = timeout_on_wait
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.timeout_on_wait and self.wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd="smoke", timeout=timeout)
        return 0


def test_build_local_smoke_config_matches_checked_in_template():
    rendered = build_local_smoke_config()
    checked_in = yaml.safe_load(DEFAULT_LOCAL_SMOKE_CONFIG_PATH.read_text())

    assert rendered == checked_in


def test_build_local_smoke_config_uses_override_values():
    config = build_local_smoke_config(upstream_url="http://localhost:8000", model_name="gemma-3-1b-it")

    provider = config["providers"][0]
    assert provider["url"] == "http://localhost:8000"
    assert provider["static_models"] == ["gemma-3-1b-it"]


def test_build_local_smoke_request_uses_openai_chat_shape():
    payload = build_local_smoke_request(model_name=DEFAULT_LOCAL_SMOKE_MODEL, prompt="say smoke")

    assert payload == {
        "model": DEFAULT_LOCAL_SMOKE_MODEL,
        "messages": [{"role": "user", "content": "say smoke"}],
        "temperature": 0,
        "max_tokens": 32,
    }


def test_extract_assistant_text_prefers_message_content():
    text = extract_assistant_text({"choices": [{"message": {"content": "smoke"}, "text": "ignored"}]})

    assert text == "smoke"


def test_extract_assistant_text_accepts_text_fallback():
    text = extract_assistant_text({"choices": [{"text": "fallback smoke"}]})

    assert text == "fallback smoke"


def test_extract_assistant_text_rejects_missing_content():
    with pytest.raises(ValueError, match="assistant text"):
        extract_assistant_text({"choices": [{"message": {"content": "   "}}]})


def test_find_chat_completion_log_matches_model_and_path():
    logs = [
        {"path": "/api/stats", "mapped_model": DEFAULT_LOCAL_SMOKE_MODEL},
        {"path": LOCAL_SMOKE_CHAT_PATH, "mapped_model": DEFAULT_LOCAL_SMOKE_MODEL, "status_code": 200},
    ]

    assert find_chat_completion_log(logs, DEFAULT_LOCAL_SMOKE_MODEL) == logs[1]


def test_find_chat_completion_log_returns_none_when_missing():
    logs = [{"path": LOCAL_SMOKE_CHAT_PATH, "mapped_model": "not-gemma", "status_code": 200}]

    assert find_chat_completion_log(logs, DEFAULT_LOCAL_SMOKE_MODEL) is None


def test_build_local_smoke_env_sets_required_runtime_values(tmp_path):
    routes_config = tmp_path / "routes.yaml"
    blob_storage_path = tmp_path / "blob_storage"
    env = build_local_smoke_env(
        {"KEEP_ME": "1", "APP_ENV": "test", "ENABLE_LOGGING": "false"},
        routes_config=routes_config,
        blob_storage_path=blob_storage_path,
        port=19090,
    )

    assert env["KEEP_ME"] == "1"
    assert env["APP_ENV"] == "dev"
    assert env["ENABLE_LOGGING"] == "true"
    assert env["ROUTES_CONFIG"] == str(routes_config)
    assert env["BLOB_STORAGE_PATH"] == str(blob_storage_path)
    assert env["LISTEN_HOST"] == "127.0.0.1"
    assert env["LISTEN_PORT"] == "19090"


def test_is_ready_stats_payload_requires_expected_keys():
    assert _is_ready_stats_payload(
        {
            "total_requests": 1,
            "completed_requests": 1,
            "pending_requests": 0,
            "service_types": {},
        }
    )
    assert not _is_ready_stats_payload({"total_requests": 1})


def test_wait_for_app_ready_accepts_expected_stats_shape(monkeypatch):
    client = _FakeClient(
        get_response=_FakeResponse(
            200,
            {
                "total_requests": 1,
                "completed_requests": 1,
                "pending_requests": 0,
                "service_types": {},
            },
        )
    )
    monkeypatch.setattr(local_smoke.httpx, "Client", lambda timeout: client)
    monkeypatch.setattr(local_smoke.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(local_smoke.time, "sleep", lambda _: None)

    _wait_for_app_ready("http://127.0.0.1:18081", 1.0)

    assert client.get_calls == [(f"http://127.0.0.1:18081{LOCAL_SMOKE_STATS_PATH}", None)]


def test_wait_for_app_ready_rejects_non_200_status(monkeypatch):
    client = _FakeClient(get_response=_FakeResponse(404, {"detail": "missing"}))
    monotonic_values = iter([0.0, 0.1, 0.9, 1.1])
    monkeypatch.setattr(local_smoke.httpx, "Client", lambda timeout: client)
    monkeypatch.setattr(local_smoke.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(local_smoke.time, "sleep", lambda _: None)

    with pytest.raises(TimeoutError, match="Timed out waiting for local smoke app"):
        _wait_for_app_ready("http://127.0.0.1:18081", 1.0)


def test_wait_for_log_entry_finds_matching_log(monkeypatch):
    matching_log = {"path": LOCAL_SMOKE_CHAT_PATH, "mapped_model": DEFAULT_LOCAL_SMOKE_MODEL, "status_code": 200}
    client = _FakeClient(get_response=_FakeResponse(200, [matching_log]))
    monkeypatch.setattr(local_smoke.httpx, "Client", lambda timeout: client)
    monkeypatch.setattr(local_smoke.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(local_smoke.time, "sleep", lambda _: None)

    assert _wait_for_log_entry("http://127.0.0.1:18081", DEFAULT_LOCAL_SMOKE_MODEL, 1.0) == matching_log


def test_stop_process_terminates_and_returns_output(tmp_path):
    output_path = tmp_path / "smoke.log"
    output_path.write_text("line 1\nline 2\n")
    process = _FakeProcess()

    output = _stop_process(process, output_path)

    assert process.terminated is True
    assert process.killed is False
    assert process.wait_calls == 1
    assert "line 2" in output


def test_stop_process_kills_when_wait_times_out(tmp_path):
    output_path = tmp_path / "smoke.log"
    output_path.write_text("boot\nready\n")
    process = _FakeProcess(timeout_on_wait=True)

    output = _stop_process(process, output_path)

    assert process.terminated is True
    assert process.killed is True
    assert process.wait_calls == 2
    assert "ready" in output


def test_run_local_smoke_uses_sanitized_env_and_pipe_free_output(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("ENABLE_LOGGING", "false")

    captured = {}

    def fake_popen(command, stdout, stderr, text, env):
        captured["command"] = command
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["text"] = text
        captured["env"] = env
        return _FakeProcess()

    client = _FakeClient(
        post_response=_FakeResponse(
            200,
            {
                "choices": [
                    {
                        "message": {"content": "smoke"},
                    }
                ]
            },
        )
    )

    monkeypatch.setattr(local_smoke.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(local_smoke.httpx, "Client", lambda timeout: client)
    monkeypatch.setattr(local_smoke, "_wait_for_app_ready", lambda base_url, timeout_seconds: None)
    monkeypatch.setattr(
        local_smoke,
        "_wait_for_log_entry",
        lambda base_url, model_name, timeout_seconds: {"status_code": 200, "mapped_model": model_name},
    )

    result = run_local_smoke(upstream_url="http://127.0.0.1:18082")

    assert result == "smoke"
    assert captured["env"]["APP_ENV"] == "dev"
    assert captured["env"]["ENABLE_LOGGING"] == "true"
    assert captured["stdout"] != subprocess.PIPE
    assert captured["stderr"] == subprocess.STDOUT


def test_run_local_smoke_reports_process_output_on_failure(monkeypatch, tmp_path):
    captured = {}

    def fake_popen(command, stdout, stderr, text, env):
        captured["command"] = command
        stdout.write("boot\nready? no\n")
        stdout.flush()
        return _FakeProcess()

    def fail_ready(base_url, timeout_seconds):
        raise RuntimeError("not ready")

    monkeypatch.setattr(local_smoke.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(local_smoke, "_wait_for_app_ready", fail_ready)

    with pytest.raises(RuntimeError, match="Server output tail") as exc_info:
        run_local_smoke(upstream_url="http://127.0.0.1:18082")

    assert captured["command"][1:3] == ["-m", "smolrouter"]
    assert "ready? no" in str(exc_info.value)


def test_build_local_smoke_command_uses_package_entrypoint(tmp_path):
    command = build_local_smoke_command(tmp_path / "routes.yaml", host="127.0.0.1", port=18081)

    assert command[1:3] == ["-m", "smolrouter"]
    assert command[3:] == ["-C", str(tmp_path / "routes.yaml"), "--host", "127.0.0.1", "--port", "18081"]


def test_smoke_defaults_point_at_local_openai_server():
    assert DEFAULT_LOCAL_SMOKE_UPSTREAM_URL == "http://localhost:11434"
    assert DEFAULT_LOCAL_SMOKE_MODEL == "gemma3:1b"
    assert Path(DEFAULT_LOCAL_SMOKE_CONFIG_PATH).name == "routes.local-smoke.yaml"