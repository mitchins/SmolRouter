import pytest
import httpx
import asyncio
from types import SimpleNamespace
import smolrouter.app as app_module
from smolrouter.app import (
    app,
    complete_request_log,
    find_route,
    rewrite_model,
    strip_think_chain_from_text,
    strip_json_markdown_from_text,
    MODEL_MAP,
    validate_url,
)
import json
import respx
from unittest.mock import AsyncMock, Mock, patch

from smolrouter.google_genai_provider import GoogleGenAIConfig, GoogleGenAIProvider
from smolrouter.interfaces import ProxyConfig
from smolrouter.request_metadata import RequestMetadata, apply_request_metadata


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

    def fake_create_task(coro):
        scheduled_coroutines.append(coro)
        coro.close()

    monkeypatch.setattr(app_module, "ENABLE_LOGGING", True)
    monkeypatch.setattr(app_module.time, "time", lambda: 102.5)
    monkeypatch.setattr(app_module, "extract_tokens_from_openai_response", lambda response_data: (11, 5, 16))
    monkeypatch.setattr(app_module, "broadcast_request_event", AsyncMock(return_value=None))
    monkeypatch.setattr(app_module, "_complete_lb_request", lambda lb_instance, start_time, success: completed_lb_calls.append((lb_instance, success)))
    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

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
    monkeypatch.setattr(asyncio, "create_task", lambda coro: coro.close())

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
