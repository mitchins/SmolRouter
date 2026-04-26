from pathlib import Path

import pytest
import yaml

from smolrouter.local_smoke import (
    DEFAULT_LOCAL_SMOKE_CONFIG_PATH,
    DEFAULT_LOCAL_SMOKE_MODEL,
    DEFAULT_LOCAL_SMOKE_UPSTREAM_URL,
    LOCAL_SMOKE_CHAT_PATH,
    build_local_smoke_command,
    build_local_smoke_config,
    build_local_smoke_env,
    build_local_smoke_request,
    extract_assistant_text,
    find_chat_completion_log,
)


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
        {"KEEP_ME": "1"},
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


def test_build_local_smoke_command_uses_package_entrypoint(tmp_path):
    command = build_local_smoke_command(tmp_path / "routes.yaml", host="127.0.0.1", port=18081)

    assert command[1:3] == ["-m", "smolrouter"]
    assert command[3:] == ["-C", str(tmp_path / "routes.yaml"), "--host", "127.0.0.1", "--port", "18081"]


def test_smoke_defaults_point_at_local_openai_server():
    assert DEFAULT_LOCAL_SMOKE_UPSTREAM_URL == "http://localhost:11434"
    assert DEFAULT_LOCAL_SMOKE_MODEL == "gemma3:1b"
    assert Path(DEFAULT_LOCAL_SMOKE_CONFIG_PATH).name == "routes.local-smoke.yaml"