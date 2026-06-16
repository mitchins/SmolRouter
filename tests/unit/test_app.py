import pytest
import httpx
import asyncio
import logging
import uuid
from types import SimpleNamespace
import smolrouter.database as database
from pydantic import BaseModel
import smolrouter.app as app_module
from smolrouter.app import (
    app,
    complete_request_log,
    _normalize_openai_model_name,
    _normalize_openai_request_payload,
    find_route,
    rewrite_model,
    strip_think_chain_from_text,
    strip_json_markdown_from_text,
    MODEL_MAP,
    validate_url,
    INVALID_JSON_REQUEST_ERROR,
)
import json
import respx
from logging.handlers import RotatingFileHandler
from pathlib import Path
from fastapi import HTTPException
from unittest.mock import AsyncMock, Mock, patch
from smolrouter.database import get_error_summary

from smolrouter.google_genai_provider import GoogleGenAIConfig, GoogleGenAIProvider
from smolrouter.interfaces import ProxyConfig
from smolrouter.request_metadata import RequestMetadata, apply_request_metadata
from smolrouter.task_utils import create_logged_task


def load_mock_json(filename):
    with open(f"tests/mocks/{filename}", "r") as f:
        return json.load(f)


@pytest.fixture
def mock_openai_upstream():
    with respx.mock as respx_mock:
        # Mock cloud OpenAI models at localhost:8000
        respx_mock.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=load_mock_json("openai_chat_completion_non_streaming.json"))
        )
        respx_mock.post("http://localhost:8000/v1/completions").mock(
            return_value=httpx.Response(200, json=load_mock_json("openai_completion_non_streaming.json"))
        )
        respx_mock.get("http://localhost:8000/v1/models").mock(
            return_value=httpx.Response(200, json=load_mock_json("openai_list_models.json"))
        )

        # Mock local LM Studio models at localhost:11434
        respx_mock.get("http://localhost:11434/v1/models").mock(
            return_value=httpx.Response(200, json=load_mock_json("lm_studio_list_models.json"))
        )
        yield respx_mock


@pytest.fixture
def mock_ollama_upstream():
    with respx.mock as respx_mock:
        respx_mock.get("http://localhost:11434/api/tags").mock(
            return_value=httpx.Response(200, json=load_mock_json("ollama_list_models.json"))
        )
        yield respx_mock


@pytest.mark.asyncio
async def test_openai_chat_completions_non_streaming(async_client, mock_openai_upstream, disable_logging):
    response = await async_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}], "stream": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert "Hello, this is a test." in data["choices"][0]["message"]["content"]
    assert "<think>" not in data["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_openai_completions_non_streaming(async_client, mock_openai_upstream, disable_logging):
    response = await async_client.post(
        "/v1/completions", json={"model": "text-davinci-003", "prompt": "Hello", "stream": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert "This is a test." in data["choices"][0]["text"]
    assert "<think>" not in data["choices"][0]["text"]


@pytest.mark.asyncio
async def test_openai_invalid_json_returns_400(async_client, disable_logging):
    response = await async_client.post(
        "/v1/chat/completions",
        content="{invalid json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json() == {"error": "Invalid JSON in request body"}


@pytest.mark.asyncio
async def test_openai_streaming_falls_back_to_non_streaming_provider_architecture(async_client, disable_logging, monkeypatch):
    fake_container = Mock()
    fake_container.route_streaming_request = AsyncMock(side_effect=RuntimeError("streaming unsupported"))
    fake_container.route_request = AsyncMock(
        return_value=({"choices": [{"message": {"content": "fallback response"}}]}, 200, "provider:test", None)
    )

    monkeypatch.setattr(app_module, "container", fake_container)
    monkeypatch.setattr(app_module, "_is_legacy_proxy_mode", lambda: False)

    response = await async_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": True},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "fallback response"
    fake_container.route_streaming_request.assert_awaited_once()
    fake_container.route_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_ollama_generate_non_streaming(async_client, mock_openai_upstream, disable_logging):
    response = await async_client.post(
        "/api/generate", json={"model": "llama2", "prompt": "Tell me a joke.", "stream": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert "Hello, this is a test." in data["response"]
    assert "<think>" not in data["response"]
    assert data["done"] is True
    assert data["model"] == "llama2"


@pytest.mark.asyncio
async def test_ollama_chat_non_streaming(async_client, mock_openai_upstream, disable_logging):
    response = await async_client.post(
        "/api/chat", json={"model": "mistral", "messages": [{"role": "user", "content": "Hello"}], "stream": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert "Hello, this is a test." in data["response"]
    assert "<think>" not in data["response"]
    assert data["done"] is True
    assert data["model"] == "mistral"


@pytest.mark.asyncio
async def test_ollama_invalid_json_returns_400(async_client, disable_logging):
    response = await async_client.post(
        "/api/chat",
        content="{invalid json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json() == {"error": "Invalid JSON in request body"}


@pytest.mark.asyncio
async def test_unhandled_route_exception_returns_500(async_client, disable_logging):
    route = f"/__test__/boom-{uuid.uuid4().hex}"

    async def boom_route():
        raise RuntimeError("route explosion")

    app_module.app.add_api_route(route, boom_route, methods=["GET"], include_in_schema=False)
    response = await async_client.get(route)

    assert response.status_code == 500
    assert response.json()["error"] == "internal_server_error"


@pytest.mark.asyncio
async def test_exception_middleware_emits_structured_error_log(async_client, disable_logging, caplog):
    route = f"/__test__/boom-structured-{uuid.uuid4().hex}"

    async def boom_route():
        raise ValueError("bad-value")

    app_module.app.add_api_route(route, boom_route, methods=["GET"], include_in_schema=False)

    with caplog.at_level(logging.ERROR):
        response = await async_client.get(route)

    assert response.status_code == 500
    error_records = [record for record in caplog.records if record.levelname == "ERROR"]
    assert error_records, "Expected ERROR logs from unhandled exception middleware"
    message = error_records[0].getMessage()
    assert "Unhandled exception request_id=" in message
    assert "method=GET" in message
    assert f"path={route}" in message
    assert "status_code=500" in message
    assert "exception_class=ValueError" in message
    assert error_records[0].exc_info is not None


@pytest.mark.asyncio
async def test_invalid_json_does_not_emit_error_log(async_client, disable_logging, caplog):
    with caplog.at_level(logging.ERROR):
        response = await async_client.post(
            "/v1/chat/completions",
            content="{invalid json",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert not any(record.levelname == "ERROR" for record in caplog.records)


@pytest.mark.asyncio
async def test_schema_invalid_request_returns_422_without_exception_signature(async_client, isolated_db, caplog):
    class _SchemaRequest(BaseModel):
        model: str
        max_tokens: int

    route = f"/__test__/validation-{uuid.uuid4().hex}"

    async def _validated_route(payload: _SchemaRequest):
        return {"ok": True}

    app_module.app.add_api_route(route, _validated_route, methods=["POST"], include_in_schema=False)

    before_summary = await get_error_summary()

    with caplog.at_level(logging.ERROR):
        response = await async_client.post(route, json={"model": 123})

    assert response.status_code == 422
    assert not any(record.levelname == "ERROR" for record in caplog.records)

    after_summary = await get_error_summary()
    assert after_summary["total_exceptions"] == before_summary["total_exceptions"]
    assert after_summary["signature_count"] == before_summary["signature_count"]
    assert after_summary["count_by_signature"] == before_summary["count_by_signature"]


@pytest.mark.asyncio
async def test_proxy_request_finalizes_log_on_client_cancel(monkeypatch, isolated_db):
    """A client disconnect mid-request must finalize the log (status 499) instead
    of leaving it 'pending' forever, and the CancelledError must still propagate.

    Regression for the orphaned-inflight pile: CancelledError is a BaseException,
    so it bypasses `except Exception`; only a finally finalizes the entry.
    """
    from starlette.requests import Request

    monkeypatch.setattr(app_module, "ENABLE_LOGGING", True)

    async def _cancel(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(app_module, "_route_openai_request", _cancel)

    completion_statuses = []
    real_complete = app_module.complete_request_log

    def _spy(log_entry, start_time, response_data, **kwargs):
        completion_statuses.append(response_data.get("status_code"))
        return real_complete(log_entry, start_time, response_data, **kwargs)

    monkeypatch.setattr(app_module, "complete_request_log", _spy)

    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()

    async def _receive():
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"content-type", b"application/json"), (b"authorization", b"Bearer test-key")],
        "client": ("127.0.0.1", 12345),
        "query_string": b"",
    }
    request = Request(scope, _receive)

    with pytest.raises(asyncio.CancelledError):
        await app_module.proxy_request("/v1/chat/completions", request)

    # Finalized exactly once, as a client-close (499) - not orphaned, not double-logged.
    assert completion_statuses == [app_module.CLIENT_CLOSED_REQUEST_STATUS]


@pytest.mark.asyncio
async def test_proxy_ollama_request_finalizes_log_on_client_cancel(monkeypatch, isolated_db):
    """Ollama requests should finalize as client-closed too.

    The Ollama path previously lacked a finally block, so CancelledError could
    orphan the request log even though the OpenAI path was already fixed.
    """
    from starlette.requests import Request

    monkeypatch.setattr(app_module, "ENABLE_LOGGING", True)

    async def _cancel(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(app_module, "_proxy_ollama_non_streaming", _cancel)

    completion_statuses = []
    real_complete = app_module.complete_request_log

    def _spy(log_entry, start_time, response_data, **kwargs):
        completion_statuses.append(response_data.get("status_code"))
        return real_complete(log_entry, start_time, response_data, **kwargs)

    monkeypatch.setattr(app_module, "complete_request_log", _spy)

    body = json.dumps({"model": "m", "prompt": "hi", "stream": False}).encode()

    async def _receive():
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/generate",
        "headers": [(b"content-type", b"application/json"), (b"authorization", b"Bearer test-key")],
        "client": ("127.0.0.1", 12345),
        "query_string": b"",
    }
    request = Request(scope, _receive)

    with pytest.raises(asyncio.CancelledError):
        await app_module.proxy_ollama_request("/api/generate", request)

    assert completion_statuses == [app_module.CLIENT_CLOSED_REQUEST_STATUS]


def test_initialize_blob_storage_strict_fails_loud_on_bad_precondition(monkeypatch):
    """A bad/unwritable storage path with logging enabled must fail LOUDLY, not
    silently disable logging (which orphaned every request as 'pending')."""
    monkeypatch.setattr(app_module, "ENABLE_LOGGING", True)

    def _boom():
        raise OSError("Read-only file system: '/app/blob_storage'")

    monkeypatch.setattr(app_module, "init_blob_storage", _boom)

    with pytest.raises(OSError):
        app_module._initialize_blob_storage_strict()

    # It must NOT silently flip logging off.
    assert app_module.ENABLE_LOGGING is True


def test_initialize_blob_storage_strict_skips_when_operator_disabled(monkeypatch):
    """Running without logging is the explicit footgun: ENABLE_LOGGING=false."""
    monkeypatch.setattr(app_module, "ENABLE_LOGGING", False)
    called = []
    monkeypatch.setattr(app_module, "init_blob_storage", lambda: called.append(1))

    app_module._initialize_blob_storage_strict()  # must not raise, must not init

    assert called == []


@pytest.mark.asyncio
async def test_completion_persists_status_even_when_blob_store_fails(isolated_db, monkeypatch):
    """Body archival is best-effort: a blob failure must not lose the completion
    accounting (status_code/completed_at), or the request stays 'pending'."""
    from datetime import datetime
    from starlette.requests import Request
    from smolrouter.redis_backend import RedisRequestLog
    import smolrouter.storage as storage_module

    monkeypatch.setattr(app_module, "ENABLE_LOGGING", True)

    class _BoomStorage:
        def store(self, *_args, **_kwargs):
            raise OSError("read-only file system")

    monkeypatch.setattr(storage_module, "get_blob_storage", lambda: _BoomStorage())

    async def _receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1),
        "query_string": b"",
    }
    log_entry = await app_module.start_request_log(
        Request(scope, _receive), "openai", "http://up", "m", "m", None, b'{"prompt":"hi"}'
    )

    log_entry.status_code = 200
    log_entry.duration_ms = 5
    log_entry.response_size = 3
    log_entry.completed_at = datetime.now()
    log_entry.set_response_body(b'{"ok":1}')

    # Deterministic persistence (no background task race).
    await log_entry.save_async()

    rec = await RedisRequestLog.get_by_id(log_entry.request_id)
    assert rec is not None
    assert rec.status_code == 200
    assert rec.completed_at is not None  # not orphaned despite blob failure


def test_resolve_route_pattern_falls_back_to_path():
    request = Mock()
    request.scope = {}
    request.url = SimpleNamespace(path="/fallback")
    assert app_module._resolve_route_pattern(request) == "/fallback"


def test_resolve_route_pattern_prefers_route_path():
    route = SimpleNamespace(path="/route/{id}")
    request = Mock()
    request.scope = {"route": route}
    request.url = SimpleNamespace(path="/fallback")
    assert app_module._resolve_route_pattern(request) == "/route/{id}"


def test_resolve_route_pattern_handles_missing_route_path():
    route = SimpleNamespace(path="")
    request = Mock()
    request.scope = {"route": route}
    request.url = SimpleNamespace(path="/fallback")
    assert app_module._resolve_route_pattern(request) == "/fallback"


def test_configure_error_file_logging_adds_rotating_handler(tmp_path, monkeypatch):
    log_file = tmp_path / "error.log"
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setenv("ERROR_LOG_FILE", str(log_file))
    monkeypatch.setenv("ERROR_LOG_MAX_BYTES", "1024")
    monkeypatch.setenv("ERROR_LOG_BACKUP_COUNT", "3")

    root_logger = logging.getLogger()
    added_handlers = []
    try:
        app_module.configure_error_file_logging()
        for handler in root_logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                if Path(handler.baseFilename) == log_file:
                    added_handlers.append(handler)

        assert added_handlers, "Expected rotating error handler on configured ERROR_LOG_FILE"
        assert str(log_file) in str(added_handlers[0].baseFilename)
        assert log_file.parent.exists()
    finally:
        for handler in added_handlers:
            root_logger.removeHandler(handler)
            handler.close()


@pytest.mark.asyncio
async def test_error_signature_patch_invalid_json_without_exception_signature(async_client, isolated_db, caplog):
    signature = uuid.uuid4().hex
    route = f"/api/errors/{signature}"

    before_summary = await get_error_summary()

    with caplog.at_level(logging.ERROR):
        response = await async_client.patch(
            route,
            content="{invalid json",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json() == {"error": INVALID_JSON_REQUEST_ERROR}
    assert not any(record.levelname == "ERROR" for record in caplog.records)

    after_summary = await get_error_summary()
    assert after_summary["total_exceptions"] == before_summary["total_exceptions"]
    assert after_summary["signature_count"] == before_summary["signature_count"]
    assert after_summary["count_by_signature"] == before_summary["count_by_signature"]


@pytest.mark.asyncio
async def test_api_error_endpoints_return_aggregated_exception_data(isolated_db):
    assert await app_module.api_error_summary() == {
        "total_exceptions": 0,
        "signature_count": 0,
        "count_by_signature": {},
        "count_by_exception_class": {},
        "count_by_route": {},
        "status_code_counts": {},
        "signatures": [],
    }

    detail = await database.record_exception_event(
        request_id="req-soak",
        exception=RuntimeError("critical failure"),
        route="/api/fail",
        request_path="/api/fail",
        method="POST",
        source_ip="127.0.0.1",
        status_code=500,
        user_agent="pytest",
    )
    signature = detail["signature"]

    summary = await app_module.api_error_summary()
    assert summary["total_exceptions"] == 1
    assert summary["signature_count"] == 1
    assert signature in summary["signatures"][0]["signature"]

    recent = await app_module.api_error_recent(limit=10)
    assert len(recent["events"]) >= 1
    assert recent["events"][0]["signature"] == signature

    signature_detail = await app_module.api_error_signature(signature)
    assert signature_detail["signature"] == signature
    assert signature_detail["exception_class"] == "RuntimeError"


@pytest.mark.asyncio
async def test_api_update_error_signature_validates_payload_and_updates_state(isolated_db):
    detail = await database.record_exception_event(
        request_id="req-soak2",
        exception=ValueError("bad config"),
        route="/api/fail",
        request_path="/api/fail",
        method="POST",
        source_ip="127.0.0.1",
        status_code=500,
        user_agent="pytest",
    )
    signature = detail["signature"]

    class _JsonRequest:
        def __init__(self, payload):
            self.payload = payload

        async def json(self):
            if isinstance(self.payload, BaseException):
                raise self.payload
            return self.payload

    response = await app_module.api_update_error_signature(signature, _JsonRequest({"state": "expected"}))
    assert response["state"] == "expected"
    assert response["notes"] == ""

    response_no_body = await app_module.api_update_error_signature(signature, _JsonRequest({"other": "value"}))
    assert response_no_body.status_code == 400
    assert json.loads(response_no_body.body) == {
        "error": "Request body must include at least one of 'state' or 'notes'"
    }

    response_invalid_state = await app_module.api_update_error_signature(
        signature,
        _JsonRequest({"state": "not-a-valid-state"}),
    )
    assert response_invalid_state.status_code == 400
    assert json.loads(response_invalid_state.body)["error"] == "Invalid state"

    with pytest.raises(HTTPException):
        await app_module.api_error_signature("missing")

    response_bad_json = await app_module.api_update_error_signature(
        signature,
        _JsonRequest(ValueError("not-json")),
    )
    assert response_bad_json.status_code == 400
    assert json.loads(response_bad_json.body)["error"] == INVALID_JSON_REQUEST_ERROR



@pytest.mark.asyncio
async def test_ollama_chat_streaming_transforms_openai_sse(async_client, disable_logging):
    class FakeStreamResponse:
        def __init__(self, chunks):
            self.status_code = 200
            self.headers = {"content-type": "text/event-stream"}
            self._chunks = chunks

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_bytes(self):
            for chunk in self._chunks:
                yield chunk

    with patch("httpx.AsyncClient") as mock_client:
        mock_http_client = mock_client.return_value.__aenter__.return_value
        mock_http_client.stream = Mock(
            return_value=FakeStreamResponse(
                [
                    b'data: {"choices": [{"delta": {"content": "Hello"}}], "created": "now"}\n\n',
                    b'data: [DONE]\n\n',
                ]
            )
        )

        response = await async_client.post(
            "/api/chat",
            json={"model": "mistral", "messages": [{"role": "user", "content": "Hello"}], "stream": True},
        )

    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert lines[0]["response"] == "Hello"
    assert lines[0]["done"] is False
    assert lines[-1]["done"] is True


@pytest.mark.asyncio
async def test_system_dashboard_shows_google_proxy_pool(async_client, disable_logging):
    provider = GoogleGenAIProvider(
        GoogleGenAIConfig(
            name="test-google",
            type="google-genai",
            enabled=True,
            api_keys=["test-key"],
            proxy_pool_enabled=True,
            proxy_pool=[None, ProxyConfig(https_proxy="http://127.0.0.1:8888")],
            per_model_proxy={"gemma-3-4b-it": ProxyConfig(https_proxy="http://127.0.0.1:8899")},
        )
    )
    fake_container = Mock()
    fake_container.get_providers.return_value = [provider]
    fake_security = Mock()
    fake_security.check_webui_access.return_value = None

    with (
        patch("smolrouter.app.container", fake_container),
        patch("smolrouter.app.get_webui_security", return_value=fake_security),
    ):
        response = await async_client.get("/system")

    assert response.status_code == 200
    content = response.text
    assert "Proxy Configuration" in content
    assert "test-google" in content
    assert "round-robin pool enabled" in content
    assert "http://127.0.0.1:8888" in content
    assert "http://127.0.0.1:8899" in content
    assert "No proxy configurations in use - all requests go direct" not in content


@pytest.mark.asyncio
async def test_google_genai_stats_returns_frontend_compatible_shape(async_client, disable_logging, monkeypatch):
    fake_provider = Mock()
    fake_provider.get_provider_type.return_value = "google-genai"
    fake_provider.get_provider_id.return_value = "google-main"
    fake_provider.get_api_key_stats = AsyncMock(
        return_value={
            "_rate_limiter": {"tokens": 1},
            "key-1": {
                "total_requests_today": 3,
                "daily_limit_per_model": 5,
                "models": {
                    "gemini-2.5-pro": {
                        "requests_today": 3,
                        "tokens_today": 120,
                        "status": "available",
                        "quota_exhausted": False,
                        "daily_limit_per_model": 5,
                    }
                },
            },
            "key-2": {
                "total_requests_today": 0,
                "daily_limit_per_model": 5,
                "models": {},
            },
        }
    )
    fake_container = Mock()
    fake_container.get_providers.return_value = [fake_provider]

    monkeypatch.setattr(app_module, "container", fake_container)

    response = await async_client.get("/api/google-genai/stats")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total_providers"] == 1
    assert payload["summary"]["total_keys"] == 2
    assert payload["providers"]["google-main"]["summary"]["total_keys"] == 2
    assert "_rate_limiter" not in payload["providers"]["google-main"]["api_keys"]
    assert payload["providers"]["google-main"]["api_keys"]["key-1"]["models"]["gemini-2.5-pro"]["requests_today"] == 3


@pytest.mark.asyncio
async def test_app_lifespan_allows_sync_proxy_monitor_shutdown(monkeypatch):
    stop_calls = []
    fake_provider = Mock()
    fake_provider.stop_proxy_health_monitor = Mock(side_effect=lambda: stop_calls.append("stopped"))
    fake_container = Mock()
    fake_container.get_providers.return_value = [fake_provider]

    monkeypatch.setattr(app_module, "container", fake_container)
    monkeypatch.setattr(app_module, "ENABLE_LOGGING", False)
    monkeypatch.setattr(app_module, "init_new_architecture", AsyncMock(return_value=None))

    with patch("smolrouter.database.RedisApiKeyQuota.initialize_lua_script", AsyncMock(return_value=None)):
        async with app_module.app_lifespan(app):
            await asyncio.sleep(0)

    assert stop_calls == ["stopped"]


@pytest.mark.asyncio
async def test_app_lifespan_drains_background_tasks(monkeypatch):
    from smolrouter import task_utils

    monkeypatch.setattr(app_module, "container", None)
    monkeypatch.setattr(app_module, "ENABLE_LOGGING", False)
    monkeypatch.setattr(app_module, "_initialize_lua_scripting", AsyncMock(return_value=None))
    monkeypatch.setattr(app_module, "init_new_architecture", AsyncMock(return_value=None))

    started = asyncio.Event()
    release = asyncio.Event()

    async def _work():
        started.set()
        await release.wait()

    task = create_logged_task(_work(), task_name="lifespan-drain-test")
    assert task is not None
    await started.wait()
    assert task in task_utils._background_tasks

    async def _release_later():
        await asyncio.sleep(0.01)
        release.set()

    release_task = asyncio.create_task(_release_later())

    async with app_module.app_lifespan(app):
        await asyncio.sleep(0)

    await release_task
    await asyncio.sleep(0)

    assert task not in task_utils._background_tasks


def test_rewrite_model_exact_match():
    original_model_map = MODEL_MAP.copy()
    MODEL_MAP.update({"old-model": "new-model"})
    assert rewrite_model("old-model") == "new-model"
    MODEL_MAP.clear()
    MODEL_MAP.update(original_model_map)


def test_rewrite_model_regex_match():
    original_model_map = MODEL_MAP.copy()
    MODEL_MAP.update({"/old-(.*)/": "new-\\1"})
    assert rewrite_model("old-variant") == "new-variant"
    MODEL_MAP.clear()
    MODEL_MAP.update(original_model_map)


def test_rewrite_model_no_match():
    original_model_map = MODEL_MAP.copy()
    MODEL_MAP.update({"old-model": "new-model"})
    assert rewrite_model("unmapped-model") == "unmapped-model"
    MODEL_MAP.clear()
    MODEL_MAP.update(original_model_map)


def test_normalize_openai_model_name_strips_provider_tag():
    assert _normalize_openai_model_name("gpt-5.4-nano-2026-03-17 [openai-main]") == "gpt-5.4-nano-2026-03-17"
    assert _normalize_openai_model_name("gpt-5-mini") == "gpt-5-mini"


def test_normalize_openai_request_payload_remaps_gpt5_max_tokens():
    payload = {
        "model": "gpt-5.4-nano-2026-03-17 [openai-main]",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 500,
    }
    _normalize_openai_request_payload(payload)

    assert payload["model"] == "gpt-5.4-nano-2026-03-17"
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 500


def test_normalize_openai_request_payload_does_not_remap_non_gpt5():
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 500,
    }
    _normalize_openai_request_payload(payload)

    assert payload["model"] == "gpt-4"
    assert payload["max_tokens"] == 500
    assert "max_completion_tokens" not in payload


def test_normalize_openai_request_payload_prefers_existing_max_completion_tokens():
    payload = {
        "model": "gpt-5-nano",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        "max_completion_tokens": 200,
    }
    _normalize_openai_request_payload(payload)

    assert payload["max_completion_tokens"] == 200
    assert "max_tokens" not in payload


def test_normalize_openai_request_payload_ignores_non_dict_payload():
    payload = ["not", "a", "dict"]
    _normalize_openai_request_payload(payload)

    assert payload == ["not", "a", "dict"]


def test_strip_think_chain_from_text():
    text_with_think = "Hello <think>internal thought</think> world."
    assert strip_think_chain_from_text(text_with_think) == "Hello  world."

    text_with_punctuation = "Hello <think>internal thought</think> , world !"
    assert strip_think_chain_from_text(text_with_punctuation) == "Hello, world!"

    text_with_multiple_think = "First <think>1</think> second <think>2</think>."
    assert strip_think_chain_from_text(text_with_multiple_think) == "First  second."

    text_no_think = "Just a regular sentence."
    assert strip_think_chain_from_text(text_no_think) == "Just a regular sentence."

    text_only_think = "<think>only think</think>"
    assert strip_think_chain_from_text(text_only_think) == ""

    text_think_with_newlines = "Hello <think>\ninternal\nthought\n</think> world."
    assert strip_think_chain_from_text(text_think_with_newlines) == "Hello  world."


def test_model_mapping_with_environment_variables():
    """Test that model mapping works with environment variable configuration"""
    # This tests the core model mapping logic without mocking the environment
    original_model_map = MODEL_MAP.copy()

    # Test exact mapping
    MODEL_MAP.update({"gpt-4": "claude-3-opus", "gpt-3.5-turbo": "claude-3-sonnet"})
    assert rewrite_model("gpt-4") == "claude-3-opus"
    assert rewrite_model("gpt-3.5-turbo") == "claude-3-sonnet"
    assert rewrite_model("unmapped") == "unmapped"

    # Test regex mapping
    MODEL_MAP.update({"/gpt-(.*)/": "claude-3-\\1"})
    assert rewrite_model("gpt-4o") == "claude-3-4o"

    MODEL_MAP.clear()
    MODEL_MAP.update(original_model_map)


def test_think_chain_stripping_edge_cases():
    """Test edge cases for think chain stripping"""
    # Test nested think tags (should not happen but test anyway)
    nested = "Text <think>outer <think>inner</think> more</think> end."
    result = strip_think_chain_from_text(nested)
    assert "<think>" not in result
    assert "Text" in result and "end." in result

    # Test empty content after stripping
    only_think = "<think>all content</think>"
    assert strip_think_chain_from_text(only_think) == ""

    # Test malformed tags
    malformed = "Text <think>unclosed tag"
    # Unclosed tags are removed from start tag to end
    assert strip_think_chain_from_text(malformed) == "Text "


def test_url_validation():
    """Test URL validation and normalization"""
    # Test normal URLs
    assert validate_url("http://localhost:8000", "TEST") == "http://localhost:8000"
    assert validate_url("https://api.openai.com", "TEST") == "https://api.openai.com"

    # Test URLs without protocol
    assert validate_url("localhost:8000", "TEST") == "http://localhost:8000"

    # Test double protocol (the original issue that caused the user's problem)
    assert validate_url("http://http://localhost:8000", "TEST") == "http://localhost:8000"
    assert validate_url("https://https://api.openai.com", "TEST") == "https://api.openai.com"

    # Test invalid URLs
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_url("", "TEST")

    with pytest.raises(ValueError, match="must use http or https"):
        validate_url("ftp://example.com", "TEST")


def test_timeout_configuration():
    """Test that timeout configuration works with environment variables"""
    import os
    from unittest.mock import patch

    # Test default timeout
    with patch.dict(os.environ, {}, clear=True):
        # Reimport to get fresh config
        import importlib
        from smolrouter import app

        importlib.reload(app)
        assert app.REQUEST_TIMEOUT == pytest.approx(3000.0)

    # Test custom timeout
    with patch.dict(os.environ, {"REQUEST_TIMEOUT": "45.5"}, clear=True):
        importlib.reload(app)
        assert app.REQUEST_TIMEOUT == pytest.approx(45.5)
        assert isinstance(app.REQUEST_TIMEOUT, float)


def test_strip_json_markdown_from_text():
    """Test JSON markdown scrubbing functionality"""
    # Basic JSON block
    text_with_json = """Here is the response:
```json
{
  "commit_type": "refactor",
  "description": "Updated code"
}
```
That's the result."""

    expected = """Here is the response:
{ "commit_type": "refactor", "description": "Updated code" }
That's the result."""

    assert strip_json_markdown_from_text(text_with_json) == expected

    # Multiple JSON blocks
    multiple_json = """First:
```json
{"status": "success"}
```
Second:
```json
{"error": null}
```
Done."""

    expected_multiple = """First:
{"status": "success"}
Second:
{"error": null}
Done."""

    assert strip_json_markdown_from_text(multiple_json) == expected_multiple

    # No JSON blocks
    no_json = "Just regular text with no JSON blocks here."
    assert strip_json_markdown_from_text(no_json) == no_json

    # Complex nested JSON
    complex_json = """Result:
```json
{
  "items": [
    {"id": 1, "name": "test"},
    {"id": 2, "name": "demo"}
  ],
  "count": 2
}
```
Finished."""

    expected_complex = """Result:
{ "items": [ {"id": 1, "name": "test"}, {"id": 2, "name": "demo"} ], "count": 2 }
Finished."""

    assert strip_json_markdown_from_text(complex_json) == expected_complex


def test_strip_json_markdown_preserves_unclosed_code_blocks():
    text_with_unclosed_block = 'Before ```json\n{\n  "status": "pending"\n}'

    assert strip_json_markdown_from_text(text_with_unclosed_block) == text_with_unclosed_block


@pytest.mark.parametrize(
    ("routes", "source_host", "model", "expected"),
    [
        (
            [
                {
                    "match": {"source_host": "127.0.0.1", "model": "gpt-4"},
                    "route": {"upstream": "http://primary-upstream", "model": "gpt-4o"},
                }
            ],
            "127.0.0.1",
            "gpt-4",
            ("http://primary-upstream", "gpt-4o"),
        ),
        (
            [{"match": {"model": "/^gpt-4/"}, "route": {"upstream": "http://regex-upstream"}}],
            "10.0.0.8",
            "gpt-4.1-mini",
            ("http://regex-upstream", None),
        ),
        (
            [{"match": {"source_host": "127.0.0.1"}, "route": {"upstream": "http://other-upstream"}}],
            "10.0.0.8",
            "gpt-4",
            ("http://fallback-upstream", None),
        ),
    ],
)
def test_find_route(monkeypatch, routes, source_host, model, expected):
    monkeypatch.setattr(app_module, "ROUTES_CONFIG_DATA", {"routes": routes})
    monkeypatch.setattr(app_module, "DEFAULT_UPSTREAM", "http://fallback-upstream")

    assert find_route(source_host, model) == expected


class DummyLogEntry:
    def __init__(self):
        self.request_id = "req-123"
        self.upstream_url = "http://upstream"
        self.request_body = None
        self.response_body = None
        self.saved = False

    def set_request_body(self, request_body):
        self.request_body = request_body

    def set_response_body(self, response_body):
        self.response_body = response_body

    def save(self):
        self.saved = True


def test_complete_request_log_uses_usage_metrics(monkeypatch):
    log_entry = DummyLogEntry()
    scheduled_coroutines = []
    completed_lb_calls = []

    def fake_create_logged_task(coro, *_args, **_kwargs):
        scheduled_coroutines.append(coro)
        coro.close()

    monkeypatch.setattr(app_module, "ENABLE_LOGGING", True)
    monkeypatch.setattr(app_module.time, "time", lambda: 102.5)
    monkeypatch.setattr(app_module, "extract_tokens_from_openai_response", lambda response_data: (11, 5, 16))
    monkeypatch.setattr(app_module, "broadcast_request_event", AsyncMock(return_value=None))
    monkeypatch.setattr(app_module, "_complete_lb_request", lambda lb_instance, start_time, success: completed_lb_calls.append((lb_instance, success)))
    monkeypatch.setattr(app_module, "create_logged_task", fake_create_logged_task)

    complete_request_log(
        log_entry,
        100.0,
        {"status_code": 200, "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}},
        request_body=b'{"messages": [{"role": "user", "content": "hello"}]}',
        response_body=b'{"choices": [{"message": {"content": "hi"}}]}',
        metadata=SimpleNamespace(lb_instance="lb-a"),
    )

    assert log_entry.saved is True
    assert log_entry.duration_ms == 2500
    assert log_entry.status_code == 200
    assert log_entry.prompt_tokens == 11
    assert log_entry.completion_tokens == 5
    assert log_entry.total_tokens == 16
    assert log_entry.request_body is not None
    assert log_entry.response_body is not None
    assert log_entry.completed_at is not None
    assert len(scheduled_coroutines) == 1
    assert completed_lb_calls == [("lb-a", True)]


def test_complete_request_log_estimates_tokens_without_usage(monkeypatch):
    log_entry = DummyLogEntry()

    monkeypatch.setattr(app_module, "ENABLE_LOGGING", True)
    monkeypatch.setattr(app_module.time, "time", lambda: 201.0)
    monkeypatch.setattr(app_module, "estimate_tokens_from_request", lambda request_data: 7)
    monkeypatch.setattr(app_module, "estimate_token_count", lambda text: 3 if text == "hello world" else 0)
    monkeypatch.setattr(app_module, "broadcast_request_event", AsyncMock(return_value=None))
    monkeypatch.setattr(app_module, "create_logged_task", lambda coro, *_args, **_kwargs: coro.close())

    complete_request_log(
        log_entry,
        200.0,
        {"status_code": 201},
        request_body=b'{"prompt": "hello"}',
        response_body=b'{"choices": [{"message": {"content": "hello world"}}]}',
    )

    assert log_entry.saved is True
    assert log_entry.prompt_tokens == 7
    assert log_entry.completion_tokens == 3
    assert log_entry.total_tokens == 10


def test_complete_request_log_without_logging_still_completes_lb_request(monkeypatch):
    completed_lb_calls = []

    monkeypatch.setattr(app_module, "ENABLE_LOGGING", False)
    monkeypatch.setattr(app_module, "_complete_lb_request", lambda lb_instance, start_time, success: completed_lb_calls.append((lb_instance, success)))

    complete_request_log(
        None,
        50.0,
        {"status_code": 503},
        metadata=SimpleNamespace(lb_instance="lb-disabled"),
    )

    assert completed_lb_calls == [("lb-disabled", False)]


@pytest.mark.asyncio
async def test_create_logged_task_logs_exception(caplog):
    async def boom():
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR, logger="smolrouter.task_utils"):
        task = create_logged_task(boom(), task_name="background-exception")
        with pytest.raises(RuntimeError):
            await task

    assert any("Unhandled exception in background-exception" in rec.getMessage() for rec in caplog.records)


def test_request_metadata_to_dict_exposes_contract_fields():
    metadata = RequestMetadata(
        api_key_suffix="abcd1234",
        proxy_used="http://127.0.0.1:8888",
        provider_id="google-main",
        model_name="gemini-2.5-pro",
        api_key_index=2,
        api_key_total=5,
        api_key_verified=True,
        proxy_verified=True,
        observation_id="obs-123",
    )

    assert metadata.to_dict() == {
        "api_key_suffix": "abcd1234",
        "proxy_used": "http://127.0.0.1:8888",
        "provider_id": "google-main",
        "model_name": "gemini-2.5-pro",
        "api_key_index": 2,
        "api_key_total": 5,
        "api_key_verified": True,
        "proxy_verified": True,
        "observation_id": "obs-123",
    }


def test_apply_request_metadata_populates_target_and_handles_missing_inputs():
    target = SimpleNamespace()
    metadata = RequestMetadata(
        api_key_suffix="abcd1234",
        proxy_used="http://127.0.0.1:8888",
        provider_id="google-main",
        api_key_index=2,
        api_key_total=5,
    )

    apply_request_metadata(target, metadata)

    assert target.api_key_suffix == "abcd1234"
    assert target.proxy_used == "http://127.0.0.1:8888"
    assert target.provider_id == "google-main"
    assert target.api_key_index == 2
    assert target.api_key_total == 5

    apply_request_metadata(None, metadata)
    apply_request_metadata(target, None)


def test_update_log_entry_provider_metadata_applies_metadata_fields():
    log_entry = SimpleNamespace(upstream_url=None)
    metadata = RequestMetadata(
        api_key_suffix="abcd1234",
        proxy_used="http://127.0.0.1:8888",
        provider_id="google-main",
        api_key_index=2,
        api_key_total=5,
    )

    app_module._update_log_entry_provider_metadata(log_entry, "provider:google-main", metadata)

    assert log_entry.upstream_url == "provider:google-main"
    assert log_entry.api_key_suffix == "abcd1234"
    assert log_entry.proxy_used == "http://127.0.0.1:8888"
    assert log_entry.provider_id == "google-main"
    assert log_entry.api_key_index == 2
    assert log_entry.api_key_total == 5


def test_serialize_request_log_provider_metadata_normalizes_contract_values():
    log_entry = SimpleNamespace(
        api_key_suffix="abcd1234",
        proxy_used="http://127.0.0.1:8888",
        provider_id="",
        api_key_index="2",
        api_key_total="5",
    )

    assert app_module._serialize_request_log_provider_metadata(log_entry) == {
        "api_key_suffix": "abcd1234",
        "proxy_used": "http://127.0.0.1:8888",
        "provider_id": None,
        "api_key_index": 2,
        "api_key_total": 5,
    }


def test_serialize_request_log_reuses_summary_metadata():
    timestamp = app_module.datetime.now()
    log_entry = SimpleNamespace(
        id="req-123",
        timestamp=timestamp,
        source_ip="127.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        provider_id="google-main",
        original_model="gpt-4o",
        mapped_model="gemini-2.5-pro",
        duration_ms=None,
        request_size=12,
        response_size=24,
        status_code="pending",
        completed_at=None,
        upstream_url="provider:google-main",
        api_key_suffix="abcd1234",
        proxy_used="http://127.0.0.1:8888",
        api_key_index=2,
        api_key_total=5,
        is_duplicate=False,
        duplicate_count=0,
    )

    payload = app_module._serialize_request_log(log_entry)

    assert payload["status_code"] is None
    assert isinstance(payload["duration_ms"], int)
    assert payload["provider_id"] == "google-main"
    assert payload["api_key_suffix"] == "abcd1234"
    assert payload["api_key_index"] == 2
    assert payload["api_key_total"] == 5


def test_serialize_request_log_normalizes_empty_provider_id():
    log_entry = SimpleNamespace(
        id="req-124",
        timestamp=app_module.datetime.now(),
        source_ip="127.0.0.1",
        path="/v1/chat/completions",
        service_type="openai",
        provider_id="",
        original_model="gpt-4o",
        mapped_model="gemini-2.5-pro",
        duration_ms=None,
        request_size=12,
        response_size=24,
        status_code="pending",
        completed_at=None,
    )

    payload = app_module._serialize_request_log(log_entry)

    assert payload["provider_id"] is None


def test_serialize_request_detail_response_reuses_summary_and_provider_metadata():
    timestamp = app_module.datetime.now()
    log_entry = SimpleNamespace(
        id="req-789",
        timestamp=timestamp,
        source_ip="127.0.0.1",
        path="/v1/chat/completions",
        service_type="openai",
        original_model="gpt-4o",
        mapped_model="gemini-2.5-pro",
        duration_ms=321,
        request_size=12,
        response_size=24,
        upstream_url="provider:google-main",
        status_code=200,
        error_message=None,
        api_key_suffix="abcd1234",
        proxy_used="http://127.0.0.1:8888",
        provider_id="google-main",
        api_key_index=2,
        api_key_total=5,
        request_body=b'{"model": "gpt-4o"}',
        response_body=b'{"choices": []}',
    )

    payload = app_module._serialize_request_detail_response(
        log_entry,
        {"is_duplicate": False, "duplicate_count": 0, "request_body_hash": None, "duplicates": []},
    )

    assert payload["upstream_url"] == "provider:google-main"
    assert payload["provider_id"] == "google-main"
    assert payload["api_key_suffix"] == "abcd1234"
    assert payload["api_key_index"] == 2
    assert payload["api_key_total"] == 5
    assert payload["request_body"] == '{"model": "gpt-4o"}'
    assert payload["response_body"] == '{"choices": []}'


def test_serialize_performance_point_uses_summary_fields():
    log_entry = SimpleNamespace(
        id="req-456",
        timestamp=app_module.datetime.now(),
        original_model="gpt-4",
        mapped_model="llama3-70b",
        service_type="openai",
        path="/v1/chat/completions",
        status_code=200,
        duration_ms=1800,
        request_size=64,
        response_size=128,
        prompt_tokens=120,
        completion_tokens=30,
        total_tokens=150,
    )

    payload = app_module._serialize_performance_point(log_entry)

    assert payload["model"] == "llama3-70b"
    assert payload["mapped_model"] == "llama3-70b"
    assert payload["status_code"] == 200
    assert payload["duration_ms"] == 1800
    assert payload["request_size"] == 64
    assert payload["response_size"] == 128


def test_json_markdown_environment_variable():
    """Test that JSON markdown scrubbing can be controlled via environment variable"""
    import os
    from unittest.mock import patch

    # Test disabled by default
    with patch.dict(os.environ, {}, clear=True):
        import importlib
        from smolrouter import app

        importlib.reload(app)
        assert app.STRIP_JSON_MARKDOWN is False

    # Test enabled
    with patch.dict(os.environ, {"STRIP_JSON_MARKDOWN": "true"}, clear=True):
        importlib.reload(app)
        assert app.STRIP_JSON_MARKDOWN is True

    # Test various true values
    for true_value in ["1", "TRUE", "yes", "Yes"]:
        with patch.dict(os.environ, {"STRIP_JSON_MARKDOWN": true_value}, clear=True):
            importlib.reload(app)
            assert app.STRIP_JSON_MARKDOWN is True
