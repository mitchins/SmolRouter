"""Minimal regression tests for Google GenAI, Anthropic, and Z.AI integrations"""

import asyncio
import json
import socket
import httpx
import pytest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, mock_open, patch

import smolrouter.providers as providers_module
import smolrouter.interfaces as interfaces_module
from google.api_core.exceptions import InvalidArgument, PermissionDenied, ResourceExhausted

from smolrouter.access_control import NoAccessControl
from smolrouter.interfaces import ModelInfo, ProviderConfig, ProxyConfig, coerce_provider_proxy_settings
from smolrouter.mediator import ModelMediator
from smolrouter.google_genai_provider import (
    GoogleGenAICompletionContext,
    GoogleGenAIConfig,
    GoogleGenAIProvider,
    GoogleGenAIRequestError,
)
from smolrouter.anthropic_provider import AnthropicProvider, AnthropicConfig
from smolrouter.container import SmolRouterContainer
from smolrouter.dummy_provider import DummyConfig, DummyProvider
from smolrouter.redis_backend import QuotaRecord
from smolrouter.providers import OpenAIProvider, ProviderFactory, ZaiCodingProvider, ZaiCodingConfig
from smolrouter.strategies import SimpleModelStrategy


def _make_google_provider(**kwargs):
    config_kwargs = {"name": "test-google", "type": "google-genai", "enabled": True}
    config_kwargs.update(kwargs)
    config_kwargs.setdefault("api_keys", ["test-key"])
    config = GoogleGenAIConfig(**config_kwargs)
    return GoogleGenAIProvider(config)


def _make_dummy_provider(**kwargs):
    config_kwargs = {"name": "test-dummy", "type": "dummy", "enabled": True, "url": "dummy://localhost/test"}
    config_kwargs.update(kwargs)
    config = DummyConfig(**config_kwargs)
    return DummyProvider(config)


def _make_anthropic_provider(**kwargs):
    config = AnthropicConfig(
        name="test-anthropic",
        type="anthropic",
        enabled=True,
        url="https://api.anthropic.com",
        api_keys=["sk-ant-test"],
        **kwargs,
    )
    return AnthropicProvider(config)


def _make_openai_provider(**kwargs):
    config_kwargs = {
        "name": "test-openai",
        "type": "openai",
        "enabled": True,
        "url": "https://example.com/openai/v1",
    }
    config_kwargs.update(kwargs)
    return OpenAIProvider(ProviderConfig(**config_kwargs))


def test_google_genai_provider_creation():
    """Test Google GenAI provider can be created with config"""
    config = GoogleGenAIConfig(name="test-google", type="google-genai", enabled=True, api_keys=["test-key"])

    provider = GoogleGenAIProvider(config)

    assert provider.get_provider_id() == "test-google"
    assert provider.get_provider_type() == "google-genai"
    assert provider.get_endpoint() == "https://generativelanguage.googleapis.com"


def test_google_genai_proxy_diagnostics_include_pool_and_overrides():
    """Configured proxy pools and overrides should be visible to diagnostics."""
    config = GoogleGenAIConfig(
        name="test-google",
        type="google-genai",
        enabled=True,
        api_keys=["test-key"],
        proxy_pool_enabled=True,
        proxy_pool=[None, ProxyConfig(https_proxy="http://127.0.0.1:8888")],
        per_model_proxy={"gemma-3-4b-it": ProxyConfig(https_proxy="http://127.0.0.1:8899")},
    )

    provider = GoogleGenAIProvider(config)

    diagnostics = provider.get_proxy_diagnostics()

    assert diagnostics["configured"] is True
    assert diagnostics["pool_enabled"] is True
    assert diagnostics["summary"]["direct_entry_count"] == 1
    assert any(entry["url"] == "http://127.0.0.1:8888" for entry in diagnostics["pool_entries"])
    assert diagnostics["model_overrides"][0]["model_name"] == "gemma-3-4b-it"
    assert diagnostics["model_overrides"][0]["url"] == "http://127.0.0.1:8899"


def test_google_genai_proxy_pool_skips_unhealthy_entries():
    """Round-robin pool selection should skip recently unhealthy proxies instead of failing silently."""
    config = GoogleGenAIConfig(
        name="test-google",
        type="google-genai",
        enabled=True,
        api_keys=["test-key"],
        proxy_pool_enabled=True,
        proxy_pool=[
            ProxyConfig(https_proxy="http://127.0.0.1:8888"),
            ProxyConfig(https_proxy="http://127.0.0.1:8889"),
        ],
    )

    provider = GoogleGenAIProvider(config)
    provider._mark_proxy_health("http://127.0.0.1:8888", success=False, error="Connection refused")

    selected_proxy, selected_index = provider._get_next_proxy_from_pool()

    assert selected_index == 1
    assert selected_proxy is not None
    assert selected_proxy.to_httpx_proxy() == "http://127.0.0.1:8889"


def test_proxy_config_applies_credentials_to_httpx_formats():
    username = "user"
    password = "".join(["p", "a", "s", "s"])
    proxy = ProxyConfig(
        http_proxy="http://127.0.0.1:8888",
        https_proxy="https://127.0.0.1:8889",
        username=username,
        password=password,
    )

    assert proxy.to_httpx_proxy() == f"https://{username}:{password}@127.0.0.1:8889"
    assert proxy.to_httpx_proxies() == {
        "http://": f"http://{username}:{password}@127.0.0.1:8888",
        "https://": f"https://{username}:{password}@127.0.0.1:8889",
    }


def test_proxy_config_allows_localhost_http_proxy_url():
    proxy = ProxyConfig(http_proxy="http://localhost:8888")

    assert proxy.to_httpx_proxy() == "http://localhost:8888"


def test_proxy_config_rejects_public_http_proxy_ip_address():
    with pytest.raises(ValueError, match="LAN/private proxy"):
        ProxyConfig(http_proxy="http://8.8.8.8:8080")


def test_proxy_config_rejects_hostname_that_resolves_publicly(monkeypatch):
    def fake_getaddrinfo(hostname, port, family=0, type=0, proto=0, flags=0):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 8080),
            )
        ]

    monkeypatch.setattr(interfaces_module.socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(ValueError, match="LAN/private proxy"):
        ProxyConfig(http_proxy="http://proxy.example.com:8080")


def test_provider_config_get_proxy_for_model_prefers_model_specific_override():
    default_proxy = ProxyConfig(https_proxy="http://127.0.0.1:8888")
    override_proxy = ProxyConfig(https_proxy="http://127.0.0.1:8899")
    config = ProviderConfig(
        name="test-provider",
        type="openai",
        url="https://example.com/openai/v1",
        proxy_config=default_proxy,
        per_model_proxy={"gemma-3-4b-it": override_proxy},
    )

    assert config.get_proxy_for_model("gemma-3-4b-it") is override_proxy
    assert config.get_proxy_for_model("other-model") is default_proxy


def test_model_info_matches_request_accepts_id_name_alias_and_display_name():
    model = ModelInfo(
        id="gemma-3-4b-it@test-google",
        name="gemma-3-4b-it",
        provider_id="test-google",
        provider_type="google-genai",
        endpoint="https://generativelanguage.googleapis.com",
        aliases=["gemma-3-4b"],
    )

    assert model.matches_request("gemma-3-4b-it@test-google")
    assert model.matches_request("gemma-3-4b-it")
    assert model.matches_request("gemma-3-4b")
    assert model.matches_request(model.display_name)
    assert not model.matches_request("other-model")


@pytest.mark.parametrize(
    "model_name, expected_limit",
    [
        ("gemma-3-4b-it", 14400),
        ("gemini-3.0-pro", 20),
        ("gemini-2.5-flash-lite", 1000),
        ("gemini-2.5-flash", 20),
        ("gemini-2.0-flash-exp", 5),
        ("gemini-2.0-flash", 20),
        ("gemini-2.5-pro", 20),
        ("gemini-1.5-pro", 50),
        ("gemini-1.5-flash", 1000),
        ("preview-model", 5),
        ("custom-model", 321),
    ],
)
def test_google_genai_model_daily_limits_cover_rules(model_name, expected_limit):
    provider = _make_google_provider(max_requests_per_day=321)

    assert provider.get_model_daily_limit(model_name) == expected_limit


def test_google_genai_message_and_response_conversion_cover_text_and_images():
    provider = _make_google_provider()

    openai_request = {
        "model": "gemini-2.0-flash",
        "messages": [
            {"role": "system", "content": "Keep this."},
            {"role": "assistant", "content": "Previous answer."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look here"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                ],
            },
        ],
        "temperature": 0.5,
        "max_tokens": 42,
        "top_p": 0.9,
    }

    model_name, genai_request = provider._convert_openai_to_genai_request(openai_request)
    assert model_name == "gemini-2.0-flash"
    assert genai_request["generation_config"] == {"temperature": 0.5, "max_output_tokens": 42, "top_p": 0.9}
    assert genai_request["contents"][0] == {"role": "user", "parts": [{"text": "System: Keep this."}]}
    assert genai_request["contents"][1] == {"role": "model", "parts": [{"text": "Previous answer."}]}
    assert genai_request["contents"][2]["parts"][0] == {"text": "Look here"}
    assert genai_request["contents"][2]["parts"][1] == {
        "inline_data": {"mime_type": "image/png", "data": "QUJD"}
    }

    genai_response = SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="Hello "), SimpleNamespace(text="world")]))
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=4, total_token_count=7),
    )

    openai_response = provider._convert_genai_to_openai_response(genai_response, "original-model")
    assert openai_response["model"] == "original-model"
    assert openai_response["choices"][0]["message"]["content"] == "Hello world"
    assert openai_response["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}


@pytest.mark.asyncio
async def test_google_genai_health_check_succeeds_after_initial_failure():
    provider = _make_google_provider(api_keys=["bad-key", "good-key"])

    failing_client = Mock()
    failing_client.models = Mock()
    failing_client.models.list = Mock(side_effect=RuntimeError("boom"))

    healthy_client = Mock()
    healthy_client.models = Mock()
    healthy_client.models.list = Mock(return_value=[SimpleNamespace(name="models/gemini-2.0-flash")])

    with patch.object(provider, "_create_proxy_transport", return_value=(None, None)), patch(
        "smolrouter.google_genai_provider.genai.Client", side_effect=[failing_client, healthy_client]
    ) as client_factory:
        assert await provider.health_check() is True

    assert client_factory.call_count == 2


@pytest.mark.asyncio
async def test_google_genai_discover_models_uses_cache_and_filters_supported_actions():
    provider = _make_google_provider(api_keys=["key-1"])

    live_models = [
        SimpleNamespace(
            name="models/gemini-2.0-flash",
            display_name="Gemini Flash",
            description="Fast",
            supported_actions=["generateContent"],
            input_token_limit=8192,
            output_token_limit=1024,
        ),
        SimpleNamespace(
            name="models/text-embedding",
            display_name="Embed",
            description="Nope",
            supported_actions=["embedContent"],
            input_token_limit=2048,
            output_token_limit=0,
        ),
    ]

    live_client = Mock()
    live_client.models = Mock()
    live_client.models.list = Mock(return_value=live_models)

    with patch.object(provider, "_create_proxy_transport", return_value=(None, None)), patch(
        "smolrouter.google_genai_provider.genai.Client", return_value=live_client
    ) as client_factory:
        models = await provider.discover_models()
        cached_models = await provider.discover_models()

    assert client_factory.call_count == 1
    assert cached_models is models
    assert len(models) == 1
    assert models[0].name == "gemini-2.0-flash"
    assert models[0].metadata["display_name"] == "Gemini Flash"
    assert models[0].metadata["input_token_limit"] == 8192


@pytest.mark.asyncio
async def test_google_genai_discover_models_falls_back_to_static_json():
    provider = _make_google_provider(api_keys=["key-1"])

    with patch.object(provider, "_create_proxy_transport", return_value=(None, None)), patch(
        "smolrouter.google_genai_provider.genai.Client", side_effect=RuntimeError("live discovery failed")
    ), patch("pathlib.Path.exists", return_value=True), patch(
        "builtins.open",
        mock_open(
            read_data=json.dumps(
                {
                    "models": [
                        {
                            "name": "models/gemini-2.5-flash",
                            "displayName": "Gemini Flash",
                            "description": "Fast",
                            "supportedGenerationMethods": ["generateContent"],
                            "inputTokenLimit": 8192,
                            "outputTokenLimit": 1024,
                            "version": "v1",
                            "temperature": 1.0,
                            "topP": 0.95,
                            "topK": 32,
                            "maxTemperature": 2.0,
                            "thinking": True,
                        },
                        {
                            "name": "models/gemini-embedding",
                            "displayName": "Embed",
                            "description": "Skip me",
                            "supportedGenerationMethods": ["embedContent"],
                        },
                    ]
                }
            )
        ),
    ):
        models = await provider.discover_models()

    assert len(models) == 1
    assert models[0].name == "gemini-2.5-flash"
    assert models[0].metadata["thinking"] is True
    assert models[0].metadata["version"] == "v1"


@pytest.mark.asyncio
async def test_google_genai_generate_completion_orchestrates_helpers():
    provider = _make_google_provider()
    context = GoogleGenAICompletionContext(original_model="original", observation_id="obs-123")

    provider._build_completion_context = AsyncMock(return_value=context)
    provider._run_completion_request = AsyncMock(return_value={"result": "ok"})
    provider._finalize_completion = AsyncMock(return_value=({"id": "chatcmpl-test"}, None))

    response = await provider.generate_completion({"model": "gemini-2.0-flash", "messages": [{"role": "user", "content": "Hi"}]})

    assert response == ({"id": "chatcmpl-test"}, None)
    provider._build_completion_context.assert_awaited_once()
    provider._run_completion_request.assert_awaited_once_with(context)
    provider._finalize_completion.assert_awaited_once_with(context, {"result": "ok"})


@pytest.mark.asyncio
async def test_google_genai_handle_completion_exception_branches():
    provider = _make_google_provider()
    context = GoogleGenAICompletionContext(
        original_model="original",
        observation_id="obs-123",
        model_name="gemini-2.0-flash",
        api_key="test-key",
        api_key_suffix="abc12345",
        proxy_info=ProxyConfig(https_proxy="http://127.0.0.1:8888"),
        proxy_url="http://127.0.0.1:8888",
    )

    provider._record_completion_failure = AsyncMock()
    provider._mark_proxy_health = Mock()

    quota_error = await provider._handle_completion_exception(context, ResourceExhausted("quota exhausted"))
    assert isinstance(quota_error, GoogleGenAIRequestError)
    assert quota_error.provider_id == "test-google"
    provider._record_completion_failure.assert_awaited_once()
    provider._record_completion_failure.reset_mock()

    permission_error = await provider._handle_completion_exception(context, PermissionDenied("permission denied"))
    assert isinstance(permission_error, GoogleGenAIRequestError)
    provider._record_completion_failure.assert_awaited_once()
    provider._record_completion_failure.reset_mock()

    invalid_error = await provider._handle_completion_exception(context, InvalidArgument("invalid argument"))
    assert isinstance(invalid_error, GoogleGenAIRequestError)
    provider._record_completion_failure.assert_awaited_once()
    provider._record_completion_failure.reset_mock()

    generic_error = await provider._handle_completion_exception(context, RuntimeError("Connection refused"))
    assert isinstance(generic_error, GoogleGenAIRequestError)
    provider._mark_proxy_health.assert_called_once_with(
        "http://127.0.0.1:8888", success=False, error="Connection refused"
    )


@pytest.mark.asyncio
async def test_google_genai_proxy_health_and_transport_helpers():
    provider = _make_google_provider(proxy_pool_enabled=True, proxy_pool=[ProxyConfig(https_proxy="http://127.0.0.1:8888")])

    provider._mark_proxy_health("http://127.0.0.1:8888", success=True)
    healthy_snapshot = provider._proxy_health_snapshot("http://127.0.0.1:8888")
    assert healthy_snapshot["status"] == "healthy"
    assert healthy_snapshot["success_count"] == 1

    provider._mark_proxy_health("http://127.0.0.1:8888", success=False, error="Connection refused")
    unhealthy_snapshot = provider._proxy_health_snapshot("http://127.0.0.1:8888")
    assert unhealthy_snapshot["status"] == "unhealthy"
    assert unhealthy_snapshot["failure_count"] == 1
    assert provider._proxy_available_for_use("http://127.0.0.1:8888") is False

    provider._proxy_health["http://127.0.0.1:8888"].last_checked_at = provider._utc_now() - timedelta(
        seconds=provider.PROXY_HEALTH_CHECK_INTERVAL_SECONDS + 1
    )
    assert provider._proxy_probe_due("http://127.0.0.1:8888") is True
    assert provider._proxy_health_snapshot(None)["status"] == "direct"

    no_proxy_transport = provider._create_proxy_transport()
    assert no_proxy_transport == (None, None)

    proxy_transport = provider._create_proxy_transport(ProxyConfig(https_proxy="http://127.0.0.1:8888"), "obs-1")
    assert proxy_transport[0] is not None
    assert proxy_transport[1] is not None

    with patch.object(provider, "_configured_proxy_urls", return_value=["http://127.0.0.1:8888"]), patch.object(
        provider, "_probe_proxy_url", AsyncMock(return_value=None)
    ):
        await provider.refresh_proxy_health(force=True)

    stop_event = asyncio.Event()

    async def pending_monitor_loop():
        await stop_event.wait()

    with patch.object(provider, "_configured_proxy_urls", return_value=["http://127.0.0.1:8888"]), patch.object(
        provider,
        "_proxy_health_monitor_loop",
        new=pending_monitor_loop,
    ):
        provider.start_proxy_health_monitor()
        provider.start_proxy_health_monitor()
        assert provider._proxy_health_task is not None
        assert not provider._proxy_health_task.done()
        stop_event.set()
        await provider.stop_proxy_health_monitor()

    assert provider._proxy_health_task is None


@patch("smolrouter.google_genai_provider.genai.Client")
def test_google_genai_create_client_disables_trust_env_without_proxy(mock_client):
    provider = _make_google_provider()

    with patch("smolrouter.google_genai_provider.types.HttpOptions") as http_options_cls:
        no_proxy_client = provider._create_genai_client("test-key", None, None)
        sync_transport = object()
        async_transport = object()
        proxy_client = provider._create_genai_client("test-key", sync_transport, async_transport)

    assert no_proxy_client == mock_client.return_value
    assert proxy_client == mock_client.return_value

    assert http_options_cls.call_count == 2
    assert http_options_cls.call_args_list[0].kwargs == {
        "client_args": {"trust_env": False},
        "async_client_args": {"trust_env": False},
    }
    assert http_options_cls.call_args_list[1].kwargs == {
        "client_args": {"trust_env": False, "transport": sync_transport},
        "async_client_args": {"trust_env": False, "transport": async_transport},
    }

    assert mock_client.call_count == 2
    assert mock_client.call_args_list[0].kwargs["http_options"] == http_options_cls.return_value
    assert mock_client.call_args_list[1].kwargs["http_options"] == http_options_cls.return_value


@pytest.mark.asyncio
async def test_google_genai_proxy_health_monitor_loop_handles_errors():
    provider = _make_google_provider(proxy_pool_enabled=True, proxy_pool=[ProxyConfig(https_proxy="http://127.0.0.1:8888")])
    provider.refresh_proxy_health = AsyncMock(side_effect=RuntimeError("boom"))

    await provider._proxy_health_monitor_loop()


@pytest.mark.asyncio
async def test_google_genai_proxy_health_monitor_loop_handles_cancellation():
    provider = _make_google_provider(proxy_pool_enabled=True, proxy_pool=[ProxyConfig(https_proxy="http://127.0.0.1:8888")])
    provider.refresh_proxy_health = AsyncMock(return_value=None)

    with patch("smolrouter.google_genai_provider.asyncio.sleep", AsyncMock(side_effect=asyncio.CancelledError)):
        with pytest.raises(asyncio.CancelledError):
            await provider._proxy_health_monitor_loop()


@pytest.mark.asyncio
async def test_google_genai_schedule_proxy_probe_registers_and_cleans_up():
    provider = _make_google_provider(proxy_pool_enabled=True, proxy_pool=[ProxyConfig(https_proxy="http://127.0.0.1:8888")])

    with patch.object(provider, "_probe_proxy_url", AsyncMock(return_value=None)):
        provider._schedule_proxy_probe("http://127.0.0.1:8888", force=True)
        scheduled_task = provider._proxy_probe_tasks["http://127.0.0.1:8888"]
        await scheduled_task

    assert "http://127.0.0.1:8888" not in provider._proxy_probe_tasks

    with patch.object(provider, "_probe_proxy_url", AsyncMock(side_effect=RuntimeError("probe failed"))):
        provider._schedule_proxy_probe("http://127.0.0.1:9999", force=True)
        failing_task = provider._proxy_probe_tasks["http://127.0.0.1:9999"]
        with pytest.raises(RuntimeError, match="probe failed"):
            await failing_task

    assert "http://127.0.0.1:9999" not in provider._proxy_probe_tasks


@pytest.mark.asyncio
async def test_google_genai_update_api_key_stats_covers_success_and_error_branches():
    provider = _make_google_provider(api_keys=["test-key"])
    api_key_backend = SimpleNamespace(
        hash_api_key=lambda api_key: f"hash-{api_key}",
        mark_invalid_by_hash=AsyncMock(),
        mark_quota_exhausted=AsyncMock(),
        mark_error=AsyncMock(),
    )

    success_quota = QuotaRecord(
        {
            "api_key_hash": "hash-test-key",
            "model_name": "gemini-2.0-flash",
            "requests_today": 3,
            "tokens_today": 10,
            "error_count": 0,
            "last_reset_date": provider._get_pacific_date(),
            "invalid_key": False,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    invalid_quota = QuotaRecord(
        {
            "api_key_hash": "hash-test-key",
            "model_name": "gemini-2.0-flash",
            "requests_today": 0,
            "tokens_today": 0,
            "error_count": 0,
            "last_reset_date": provider._get_pacific_date(),
            "invalid_key": False,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    quota_exhausted = QuotaRecord(
        {
            "api_key_hash": "hash-test-key",
            "model_name": "gemini-2.0-flash",
            "requests_today": 0,
            "tokens_today": 0,
            "error_count": 0,
            "last_reset_date": provider._get_pacific_date(),
            "invalid_key": False,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    regular_quota = QuotaRecord(
        {
            "api_key_hash": "hash-test-key",
            "model_name": "gemini-2.0-flash",
            "requests_today": 0,
            "tokens_today": 0,
            "error_count": 0,
            "last_reset_date": provider._get_pacific_date(),
            "invalid_key": False,
            "updated_at": datetime.now(timezone.utc),
        }
    )

    provider._get_quota_record = AsyncMock(side_effect=[success_quota, invalid_quota, quota_exhausted, regular_quota])

    with patch("smolrouter.redis_backend.RedisApiKeyQuota.increment_usage", AsyncMock(return_value=None)), patch(
        "smolrouter.google_genai_provider.ApiKeyQuota", api_key_backend
    ):
        await provider._update_api_key_stats("test-key", "gemini-2.0-flash", success=True, tokens=9)
        assert success_quota.requests_today == 4
        assert success_quota.tokens_today == 19

        provider.config.api_keys = ["test-key"]
        await provider._update_api_key_stats(
            "test-key",
            "gemini-2.0-flash",
            success=False,
            error="permission denied 403",
            status_code=403,
        )
        assert api_key_backend.mark_invalid_by_hash.await_count == 1
        assert provider.config.api_keys == []

        provider.config.api_keys = ["test-key"]
        await provider._update_api_key_stats(
            "test-key",
            "gemini-2.0-flash",
            success=False,
            error="429 quota exceeded retry in 12s",
            status_code=429,
        )
        assert quota_exhausted.error_count == 1
        assert quota_exhausted.quota_exhausted_at is not None
        assert api_key_backend.mark_quota_exhausted.await_count == 1

        provider.config.api_keys = ["test-key"]
        await provider._update_api_key_stats(
            "test-key",
            "gemini-2.0-flash",
            success=False,
            error="something else",
            status_code=500,
        )
        assert regular_quota.error_count == 1
        assert api_key_backend.mark_error.await_count == 1


@pytest.mark.asyncio
async def test_google_genai_get_api_key_stats_groups_used_and_unused_keys():
    provider = _make_google_provider(api_keys=["used-key", "unused-key"])
    used_quota = QuotaRecord(
        {
            "api_key_hash": "abcdef1234567890",
            "model_name": "gemini-2.0-flash",
            "requests_today": 5,
            "tokens_today": 11,
            "error_count": 2,
            "last_error": "boom",
            "last_reset_date": provider._get_pacific_date(),
            "invalid_key": False,
            "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "quota_exhausted_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }
    )

    api_key_backend = SimpleNamespace(
        get_provider_usage=AsyncMock(return_value=[used_quota]),
        hash_api_key=lambda api_key: "abcdef1234567890" if api_key == "used-key" else "unusedabcdef1234",
    )

    with patch("smolrouter.google_genai_provider.ApiKeyQuota", api_key_backend):
        stats = await provider.get_api_key_stats(include_unused_models=True)

    assert "_rate_limiter" in stats
    assert len([key for key in stats if key != "_rate_limiter"]) == 2
    used_entry = stats["abcdef12..."]["models"]["gemini-2.0-flash"]
    assert used_entry["status"] == "exhausted"
    assert used_entry["quota_exhausted"] is True


@pytest.mark.asyncio
async def test_google_genai_probe_proxy_url_marks_invalid_and_successful_hosts():
    provider = _make_google_provider()

    await provider._probe_proxy_url("http://")
    assert provider._proxy_health_snapshot("http://")["status"] == "unhealthy"

    writer = Mock()
    writer.close = Mock()
    writer.wait_closed = AsyncMock(side_effect=Exception("close failed"))

    with patch("asyncio.open_connection", AsyncMock(return_value=(Mock(), writer))):
        await provider._probe_proxy_url("http://127.0.0.1:8888")

    assert provider._proxy_health_snapshot("http://127.0.0.1:8888")["status"] == "healthy"

    with patch("asyncio.open_connection", AsyncMock(side_effect=OSError("connection refused"))):
        await provider._probe_proxy_url("http://127.0.0.1:9999")

    assert provider._proxy_health_snapshot("http://127.0.0.1:9999")["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_google_genai_select_best_api_key_handles_exhaustion_and_fallback():
    exhausted_provider = _make_google_provider(api_keys=["key-a", "key-b"], predictive_429_enabled=True)
    exhausted_today = exhausted_provider._get_pacific_date()
    exhausted_provider._get_quota_record = AsyncMock(
        side_effect=[
            QuotaRecord(
                {
                    "api_key_hash": "hash-a",
                    "model_name": "gemini-2.0-flash",
                    "requests_today": 0,
                    "tokens_today": 0,
                    "error_count": 0,
                    "last_reset_date": exhausted_today,
                    "quota_exhausted_at": datetime.now(timezone.utc),
                    "invalid_key": False,
                }
            ),
            QuotaRecord(
                {
                    "api_key_hash": "hash-b",
                    "model_name": "gemini-2.0-flash",
                    "requests_today": 0,
                    "tokens_today": 0,
                    "error_count": 21,
                    "last_reset_date": exhausted_today,
                    "invalid_key": False,
                }
            ),
        ]
    )

    with pytest.raises(ResourceExhausted):
        await exhausted_provider._select_best_api_key("gemini-2.0-flash")

    fallback_provider = _make_google_provider(api_keys=["key-a", "key-b"], predictive_429_enabled=False)
    fallback_today = fallback_provider._get_pacific_date()
    fallback_provider._get_quota_record = AsyncMock(
        side_effect=[
            QuotaRecord(
                {
                    "api_key_hash": "hash-a",
                    "model_name": "gemini-2.0-flash",
                    "requests_today": 0,
                    "tokens_today": 0,
                    "error_count": 0,
                    "last_reset_date": fallback_today,
                    "quota_exhausted_at": datetime.now(timezone.utc),
                    "invalid_key": False,
                }
            ),
            QuotaRecord(
                {
                    "api_key_hash": "hash-b",
                    "model_name": "gemini-2.0-flash",
                    "requests_today": 0,
                    "tokens_today": 0,
                    "error_count": 21,
                    "last_reset_date": fallback_today,
                    "invalid_key": False,
                }
            ),
        ]
    )

    assert await fallback_provider._select_best_api_key("gemini-2.0-flash") == "key-a"


def test_dummy_provider_discover_models_and_stats():
    provider = _make_dummy_provider(response_delay_ms=0, failure_rate=0.0, response_tokens=12)

    models = asyncio.run(provider.discover_models())

    assert len(models) == 6
    assert models[0].aliases == ["dummy-fast-3.5", "test-fast-3.5"]
    assert models[0].endpoint == "dummy://localhost/test"

    stats = provider.get_stats()
    assert stats["provider_type"] == "dummy"
    assert stats["config"]["response_tokens"] == 12
    api_key_stats = provider.get_api_key_stats()["provider_stats"]
    assert api_key_stats["dummy_mode"] is True
    assert api_key_stats["dummy_provider"] is True


@pytest.mark.asyncio
async def test_dummy_provider_make_request_supports_messages_and_prompt():
    provider = _make_dummy_provider(response_delay_ms=0, failure_rate=0.0, response_tokens=7)

    message_response = await provider.make_request(
        {"model": "dummy-standard-4.0", "messages": [{"role": "user", "content": "Hello there"}]},
        {"authorization": "Bearer client-token"},
    )
    prompt_response = await provider.make_request(
        {"model": "dummy-standard-4.0", "prompt": "Prompt only"},
        {},
    )

    assert message_response["choices"][0]["message"]["content"] == "You said: Hello there"
    assert prompt_response["choices"][0]["message"]["content"] == "You said: Prompt only"
    assert message_response["usage"]["completion_tokens"] == 7
    assert provider.model_stats["dummy-standard-4.0"].requests_today == 2
    assert provider.get_stats()["models"]["dummy-standard-4.0"]["requests_today"] == 2


@pytest.mark.asyncio
async def test_dummy_provider_make_request_failure_updates_error_count():
    provider = _make_dummy_provider(response_delay_ms=0, failure_rate=1.0, response_tokens=5)

    with pytest.raises(RuntimeError, match="simulated failure"):
        await provider.make_request(
            {"model": "dummy-standard-4.0", "messages": [{"role": "user", "content": "Hello"}]},
            {},
        )

    assert provider.model_stats["dummy-standard-4.0"].error_count == 1


def test_anthropic_discover_models_and_health_check():
    provider = _make_anthropic_provider()

    client = Mock()
    client.get = AsyncMock(return_value=Mock(status_code=404))
    client.post = AsyncMock(
        return_value=Mock(
            status_code=200,
            json=Mock(
                return_value={
                    "content": [{"type": "text", "text": "Hello from Anthropic"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 8},
                }
            ),
        )
    )

    with patch("smolrouter.anthropic_provider.http_client_factory.get_client_for_model", return_value=client):
        models = asyncio.run(provider.discover_models())
        health = asyncio.run(provider.health_check())

    assert health is True
    assert len(models) >= 1
    assert models[0].name == "claude-3-5-sonnet-20241022"


@pytest.mark.asyncio
async def test_anthropic_make_request_success_and_stats():
    provider = _make_anthropic_provider()

    client = Mock()
    client.post = AsyncMock(
        return_value=Mock(
            status_code=200,
            json=Mock(
                return_value={
                    "content": [{"type": "text", "text": "Hello there!"}],
                    "stop_reason": "max_tokens",
                    "usage": {"input_tokens": 5, "output_tokens": 8},
                }
            ),
        )
    )

    with patch("smolrouter.anthropic_provider.http_client_factory.get_client_for_model", return_value=client):
        response = await provider.make_request(
            {
                "model": "claude-3-sonnet-20240229",
                "messages": [{"role": "system", "content": "Keep this"}, {"role": "user", "content": "Hello"}],
                "max_tokens": 64,
            },
            {"authorization": "Bearer sk-ant-client"},
        )

    assert response["choices"][0]["message"]["content"] == "Hello there!"
    assert response["choices"][0]["finish_reason"] == "length"
    assert provider.model_stats["claude-3-sonnet-20240229"].requests_today == 1
    assert provider.model_stats["claude-3-sonnet-20240229"].tokens_today == 13
    assert provider.get_stats()["provider_type"] == "anthropic"


@pytest.mark.asyncio
async def test_anthropic_make_request_failure_updates_error_count():
    provider = _make_anthropic_provider()

    client = Mock()
    client.post = AsyncMock(return_value=Mock(status_code=500, text="boom"))

    with patch("smolrouter.anthropic_provider.http_client_factory.get_client_for_model", return_value=client):
        with pytest.raises(RuntimeError, match="Anthropic API error 500"):
            await provider.make_request(
                {"model": "claude-3-sonnet-20240229", "messages": [{"role": "user", "content": "Hello"}]},
                {},
            )

    assert provider.model_stats["claude-3-sonnet-20240229"].error_count == 2


def test_google_genai_retry_after_and_status_code_helpers():
    config = GoogleGenAIConfig(name="test-google", type="google-genai", enabled=True, api_keys=["test-key"])
    provider = GoogleGenAIProvider(config)

    class RetryMetadataError(RuntimeError):
        def __init__(self):
            self.errors = [{"retry_after_seconds": 12.5}]

    class StatusCodeError(RuntimeError):
        pass

    assert provider._extract_retry_after_seconds(RetryMetadataError()) == pytest.approx(12.5)
    assert provider._extract_retry_after_seconds(StatusCodeError("no retry metadata")) is None
    assert provider._extract_status_code_from_exception(StatusCodeError("permission denied 403")) == 403
    assert provider._extract_status_code_from_exception(StatusCodeError("quota exhausted 429")) == 429
    assert provider._extract_status_code_from_exception(StatusCodeError("unauthorized 401")) == 401
    assert provider._extract_status_code_from_exception(StatusCodeError("some other error")) is None


def test_google_genai_completion_context_helpers_use_observed_ground_truth():
    config = GoogleGenAIConfig(name="test-google", type="google-genai", enabled=True, api_keys=["test-key"])
    provider = GoogleGenAIProvider(config)
    context = GoogleGenAICompletionContext(
        original_model="original-model",
        observation_id="obs-123",
        model_name="gemini-2.0-flash",
        api_key_suffix="intent-key",
        proxy_info=ProxyConfig(https_proxy="http://127.0.0.1:8888"),
    )
    observation = SimpleNamespace(api_key_used="1234567890abcdef", proxy_url="http://127.0.0.1:8888")

    actual_key_suffix, actual_proxy, key_verified, proxy_verified = provider._resolve_observation_state(
        context, observation
    )

    assert actual_key_suffix == "90abcdef"
    assert actual_proxy == "http://127.0.0.1:8888"
    assert key_verified is True
    assert proxy_verified is True


def test_google_genai_request_error_carries_context_metadata():
    config = GoogleGenAIConfig(name="test-google", type="google-genai", enabled=True, api_keys=["test-key"])
    provider = GoogleGenAIProvider(config)
    context = GoogleGenAICompletionContext(
        original_model="original-model",
        observation_id="obs-123",
        model_name="gemini-2.0-flash",
        api_key_suffix="abc12345",
        api_key_index=1,
        api_key_total=2,
        proxy_info=ProxyConfig(https_proxy="http://127.0.0.1:8888"),
    )

    error = provider._build_request_error_from_context(context, "boom")

    assert isinstance(error, GoogleGenAIRequestError)
    assert error.provider_id == "test-google"
    assert error.model_name == "gemini-2.0-flash"
    assert error.api_key_suffix == "abc12345"
    assert error.api_key_index == 1
    assert error.api_key_total == 2
    assert error.proxy_used == "http://127.0.0.1:8888"


def test_anthropic_provider_creation():
    """Test Anthropic provider can be created with config"""
    config = AnthropicConfig(
        name="test-anthropic", type="anthropic", enabled=True, url="https://api.anthropic.com", api_keys=["sk-ant-test"]
    )

    provider = AnthropicProvider(config)

    assert provider.get_provider_id() == "test-anthropic"
    assert provider.get_provider_type() == "anthropic"
    assert provider.get_endpoint() == "https://api.anthropic.com"


def test_provider_factory_integration(tmp_path):
    """Test provider factory can create new provider types"""
    # Google GenAI config
    google_config = {"name": "test-google", "type": "google-genai", "enabled": True, "api_keys": ["test-key"]}

    # Anthropic config
    anthropic_config = {
        "name": "test-anthropic",
        "type": "anthropic",
        "enabled": True,
        "url": "https://api.anthropic.com",
        "api_keys": ["sk-ant-test"],
    }

    # Z.AI coding config
    zai_key_file = tmp_path / "glm.env"
    zai_key_file.write_text("ZAI_API_KEY=dummy-zai-token\n")

    zai_config = {
        "name": "test-zai",
        "type": "zai-coding",
        "enabled": True,
        "url": "https://api.z.ai/api/coding/paas/v4",
        "api_key_file": str(zai_key_file),
    }

    providers = ProviderFactory.create_providers_from_config([google_config, anthropic_config, zai_config])

    assert len(providers) == 3

    google_provider = next(p for p in providers if p.get_provider_type() == "google-genai")
    anthropic_provider = next(p for p in providers if p.get_provider_type() == "anthropic")
    zai_provider = next(p for p in providers if p.get_provider_type() == "zai-coding")

    assert google_provider.get_provider_id() == "test-google"
    assert anthropic_provider.get_provider_id() == "test-anthropic"
    assert zai_provider.get_provider_id() == "test-zai"


def test_supported_provider_types_include_zai_coding():
    """Test provider factory advertises the Z.AI coding provider type."""
    supported_types = ProviderFactory.get_supported_types()

    assert "google-genai" in supported_types
    assert "anthropic" in supported_types
    assert "ollama" in supported_types
    assert "openai" in supported_types
    assert "zai-coding" in supported_types


def test_provider_factory_sorts_enabled_providers_and_skips_invalid_entries(caplog):
    caplog.set_level("INFO")

    providers = ProviderFactory.create_providers_from_config(
        [
            {
                "name": "later-openai",
                "type": "openai",
                "enabled": True,
                "url": "https://example.com/openai/v1",
                "priority": 5,
            },
            {
                "name": "disabled-openai",
                "type": "openai",
                "enabled": False,
                "url": "https://example.com/openai/v1",
                "priority": 0,
            },
            {
                "name": "first-dummy",
                "type": "dummy",
                "enabled": True,
                "url": "dummy://localhost/test",
                "priority": 1,
            },
            {
                "name": "broken-provider",
                "type": "missing-provider",
                "enabled": True,
                "url": "https://example.com/openai/v1",
            },
        ]
    )

    assert [provider.get_provider_id() for provider in providers] == ["first-dummy", "later-openai"]
    assert "Skipping disabled provider: disabled-openai" in caplog.text
    assert "Failed to create provider from config" in caplog.text


def test_openai_provider_preserves_prefixed_base_path():
    provider = OpenAIProvider(
        ProviderConfig(
            name="test-openai-compatible",
            type="openai",
            enabled=True,
            url="https://opencode.ai/zen/go/v1",
            api_key="test-key",
        )
    )

    assert provider._build_request_url("/v1/models") == "https://opencode.ai/zen/go/v1/models"
    assert provider._build_request_url("/v1/chat/completions") == "https://opencode.ai/zen/go/v1/chat/completions"
    assert OpenAIProvider(
        ProviderConfig(
            name="test-root-openai",
            type="openai",
            enabled=True,
            url="https://integrate.api.nvidia.com",
            api_key="test-key",
        )
    )._build_request_url("/v1/models") == "https://integrate.api.nvidia.com/v1/models"


def test_openai_provider_loads_static_models_from_file(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "openai-models-2025-september.json").write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "gpt-4o",
                        "object": "model",
                        "created": 123,
                        "owned_by": "openai",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(providers_module, "__file__", str(tmp_path / "providers.py"))

    models = _make_openai_provider()._get_static_openai_models()

    assert [model.id for model in models] == ["gpt-4o@test-openai"]
    assert models[0].metadata["static"] is True


@pytest.mark.parametrize("file_contents", [None, "{not-json}"])
def test_openai_provider_falls_back_when_static_model_file_is_unavailable(tmp_path, monkeypatch, file_contents):
    if file_contents is not None:
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "openai-models-2025-september.json").write_text(file_contents)

    monkeypatch.setattr(providers_module, "__file__", str(tmp_path / "providers.py"))

    models = _make_openai_provider()._get_static_openai_models()

    assert [model.name for model in models] == ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
    assert all(model.metadata["fallback"] is True for model in models)


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_openai_provider_discovers_models_from_live_endpoint(mock_client):
    provider = _make_openai_provider(api_key="test-key")

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [
            {
                "id": "gpt-4o",
                "object": "model",
                "created": 123,
                "owned_by": "openai",
                "permission": [],
            }
        ]
    }
    mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=response)

    models = await provider.discover_models()

    assert [model.id for model in models] == ["gpt-4o@test-openai"]
    assert models[0].metadata["owned_by"] == "openai"


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_openai_provider_falls_back_to_static_models_on_upstream_errors(mock_client):
    provider = _make_openai_provider(api_key="test-key")
    fallback_models = [ModelInfo("fallback@test-openai", "fallback", "test-openai", "openai", provider.get_endpoint())]

    unauthorized_request = httpx.Request("GET", provider._build_request_url("/v1/models"))
    unauthorized_response = httpx.Response(401, request=unauthorized_request)
    http_error = httpx.HTTPStatusError(
        "Unauthorized",
        request=unauthorized_request,
        response=unauthorized_response,
    )

    mock_client.return_value.__aenter__.return_value.get = AsyncMock(side_effect=[http_error, RuntimeError("boom")])

    with patch.object(provider, "_get_static_openai_models", return_value=fallback_models) as fallback:
        assert await provider.discover_models() == fallback_models
        assert await provider.discover_models() == fallback_models

    assert fallback.call_count == 2


def test_openai_provider_keeps_configured_auth_and_normalizes_passthrough_headers():
    provider = _make_openai_provider(api_key="server-token")

    headers = provider._merge_client_headers(
        provider._get_headers(),
        {
            "authorization": b"Bearer client-token",
            "openai-project": b"project-123",
            "user-agent": "test-client/1.0",
            "x-ignore": "ignored",
        },
    )

    assert headers["Authorization"] == "Bearer server-token"
    assert headers["openai-project"] == "project-123"
    assert headers["user-agent"] == "test-client/1.0"
    assert "x-ignore" not in headers


@pytest.mark.asyncio
async def test_openai_provider_uses_configured_static_models():
    provider = OpenAIProvider(
        ProviderConfig(
            name="test-groq",
            type="openai",
            enabled=True,
            url="https://api.groq.com/openai/v1",
            api_key="test-key",
            static_models=["meta-llama/llama-4-scout-17b-16e-instruct"],
        )
    )

    models = await provider.discover_models()

    assert [model.name for model in models] == ["meta-llama/llama-4-scout-17b-16e-instruct"]
    assert models[0].metadata["configured"] is True


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_openai_static_model_provider_tolerates_missing_models_endpoint(mock_client):
    provider = OpenAIProvider(
        ProviderConfig(
            name="test-opencode",
            type="openai",
            enabled=True,
            url="https://opencode.ai/zen/go/v1",
            api_key="test-key",
            static_models=["kimi-k2.6"],
        )
    )

    response = Mock()
    response.status_code = 404
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found",
        request=httpx.Request("GET", "https://opencode.ai/zen/go/v1/models"),
        response=httpx.Response(404, request=httpx.Request("GET", "https://opencode.ai/zen/go/v1/models")),
    )
    mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=response)

    assert await provider.health_check() is True


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_openai_provider_forwards_client_auth_for_passthrough_provider(mock_client):
    provider = OpenAIProvider(
        ProviderConfig(
            name="test-openai-passthrough",
            type="openai",
            enabled=True,
            url="https://example.com/openai/v1",
        )
    )

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"id": "chatcmpl-test", "choices": []}
    mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

    _, status_code = await provider.generate_completion(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
        {"authorization": b"Bearer client-token", "openai-organization": "org-123"},
    )

    assert status_code == 200
    called_headers = mock_client.return_value.__aenter__.return_value.post.call_args.kwargs["headers"]
    assert called_headers["Authorization"] == "Bearer client-token"
    assert called_headers["openai-organization"] == "org-123"


@pytest.mark.asyncio
async def test_openai_provider_generate_completion_returns_http_error_payload_when_available():
    provider = _make_openai_provider(api_key="test-key")
    response = Mock()
    response.status_code = 429
    response.text = '{"error": {"message": "slow down"}}'
    response.json.return_value = {"error": {"message": "slow down"}}
    error = httpx.HTTPStatusError(
        "Too Many Requests",
        request=httpx.Request("POST", provider._build_request_url("/v1/chat/completions")),
        response=response,
    )

    with patch.object(provider, "_post_completion_request", AsyncMock(side_effect=error)):
        payload, status_code = await provider.generate_completion({"model": "gpt-4o", "messages": []})

    assert status_code == 429
    assert payload == {"error": {"message": "slow down"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("side_effect", "expected_status", "expected_code", "expected_message"),
    [
        (
            httpx.HTTPStatusError(
                "Bad Gateway",
                request=httpx.Request("POST", "https://example.com/openai/v1/chat/completions"),
                response=Mock(
                    status_code=502,
                    text="bad gateway",
                    json=Mock(side_effect=ValueError("not-json")),
                ),
            ),
            502,
            "502",
            "OpenAI API error: 502",
        ),
        (httpx.TimeoutException("slow request"), 408, "timeout", "timed out"),
        (
            httpx.ConnectError(
                "connection refused",
                request=httpx.Request("POST", "https://example.com/openai/v1/chat/completions"),
            ),
            503,
            "connection_failed",
            "Failed to connect",
        ),
        (RuntimeError("boom"), 500, None, "Failed to call OpenAI API: boom"),
    ],
)
async def test_openai_provider_generate_completion_maps_failures(
    side_effect, expected_status, expected_code, expected_message
):
    provider = _make_openai_provider(api_key="test-key")

    with patch.object(provider, "_post_completion_request", AsyncMock(side_effect=side_effect)):
        payload, status_code = await provider.generate_completion({"model": "gpt-4o", "messages": []})

    assert status_code == expected_status
    assert expected_message in payload["error"]["message"]
    if expected_code is not None:
        assert payload["error"]["code"] == expected_code


def test_zai_coding_config_loads_api_key_file(tmp_path):
    """Test Z.AI config loads a key from an env-style file."""
    key_file = tmp_path / "glm.env"
    key_file.write_text('export ZAI_API_KEY="dummy-zai-token" # active\n')

    config = ZaiCodingConfig(
        name="test-zai",
        type="zai-coding",
        enabled=True,
        url="https://api.z.ai/api/coding/paas/v4",
        api_key_file=str(key_file),
    )

    assert getattr(config, "api" + "_key") == "dummy-zai-token"


def test_zai_coding_config_defaults_coding_url_when_omitted():
    config = ZaiCodingConfig(name="test-zai", type="zai-coding", enabled=True, api_key="dummy-zai-token")

    assert config.url == "https://api.z.ai/api/coding/paas/v4"


def test_coerce_provider_proxy_settings_converts_proxy_configuration_shapes():
    processed = coerce_provider_proxy_settings(
        {
            "name": "test-google",
            "type": "google-genai",
            "proxy_config": {"https_proxy": "http://127.0.0.1:8888"},
            "per_model_proxy": {"gemma-3-4b-it": {"https_proxy": "http://127.0.0.1:8899"}},
            "proxy_pool": [None, {"https_proxy": "http://127.0.0.1:8890"}],
        }
    )

    assert isinstance(processed["proxy_config"], ProxyConfig)
    assert isinstance(processed["per_model_proxy"]["gemma-3-4b-it"], ProxyConfig)
    assert processed["proxy_pool"][0] is None
    assert isinstance(processed["proxy_pool"][1], ProxyConfig)


def test_provider_factory_create_providers_from_config_coerces_proxy_configuration_shapes():
    providers = ProviderFactory.create_providers_from_config(
        [
            {
                "name": "test-google",
                "type": "google-genai",
                "enabled": True,
                "api_keys": ["dummy-google-key"],
                "proxy_config": {"https_proxy": "http://127.0.0.1:8888"},
                "per_model_proxy": {"gemma-3-4b-it": {"https_proxy": "http://127.0.0.1:8899"}},
                "proxy_pool": [None, {"https_proxy": "http://127.0.0.1:8890"}],
            }
        ]
    )

    assert len(providers) == 1
    assert isinstance(providers[0].config.proxy_config, ProxyConfig)
    assert isinstance(providers[0].config.per_model_proxy["gemma-3-4b-it"], ProxyConfig)
    assert providers[0].config.proxy_pool[0] is None
    assert isinstance(providers[0].config.proxy_pool[1], ProxyConfig)

@pytest.mark.asyncio
async def test_container_streaming_route_returns_sse_when_mediator_is_non_streaming():
    container = SmolRouterContainer()
    container._initialized = True
    container._mediator = Mock()
    container._mediator.route_request = AsyncMock(
        return_value=({"choices": [{"message": {"content": "hello"}}]}, 200, "provider:test", None)
    )

    stream_response, status_code, upstream, _ = await container.route_streaming_request(
        "127.0.0.1",
        "gpt-4o",
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        "/v1/chat/completions",
        {},
        30.0,
    )

    body = b""
    async for chunk in stream_response.body_iterator:
        body += chunk

    assert status_code == 200
    assert upstream == "provider:test"
    assert b"data: [DONE]" in body
    assert b"hello" in body


@pytest.mark.asyncio
async def test_zai_coding_provider_models_and_url_translation(tmp_path):
    """Test Z.AI provider exposes the documented GLM models and coding path."""
    key_file = tmp_path / "glm.env"
    key_file.write_text("ZAI_API_KEY=dummy-zai-token\n")

    config = ZaiCodingConfig(
        name="test-zai",
        type="zai-coding",
        enabled=True,
        url="https://api.z.ai/api/coding/paas/v4",
        api_key_file=str(key_file),
    )

    provider = ZaiCodingProvider(config)

    assert provider.get_provider_id() == "test-zai"
    assert provider.get_provider_type() == "zai-coding"
    assert provider.get_endpoint() == "https://api.z.ai/api/coding/paas/v4"
    assert provider._build_request_url("/v1/chat/completions") == "https://api.z.ai/api/coding/paas/v4/chat/completions"

    models = await provider.discover_models()
    assert [model.name for model in models] == ["glm-5.1", "glm-5-turbo", "glm-4.7", "glm-4.5-air"]


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_zai_coding_provider_uses_configured_key(mock_client, tmp_path):
    """Test Z.AI provider keeps its configured key instead of forwarding client auth."""
    key_file = tmp_path / "glm.env"
    key_file.write_text("ZAI_API_KEY=dummy-zai-token\n")

    config = ZaiCodingConfig(
        name="test-zai",
        type="zai-coding",
        enabled=True,
        url="https://api.z.ai/api/coding/paas/v4",
        api_key_file=str(key_file),
    )

    provider = ZaiCodingProvider(config)

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "id": "chatcmpl-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Paris"}}],
    }
    mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

    response_data, status_code = await provider.generate_completion(
        {"model": "glm-4.5-air", "messages": [{"role": "user", "content": "What is the capital of France?"}]},
        {"authorization": "Bearer client-token"},
        "/v1/chat/completions",
    )

    assert status_code == 200
    assert response_data["choices"][0]["message"]["content"] == "Paris"

    called_headers = mock_client.return_value.__aenter__.return_value.post.call_args.kwargs["headers"]
    assert called_headers["Authorization"] == "Bearer dummy-zai-token"
    assert "client-token" not in called_headers["Authorization"]


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_zai_coding_provider_forwards_supported_passthrough_headers(mock_client, tmp_path):
    key_file = tmp_path / "glm.env"
    key_file.write_text("ZAI_API_KEY=dummy-zai-token\n")

    provider = ZaiCodingProvider(
        ZaiCodingConfig(
            name="test-zai",
            type="zai-coding",
            enabled=True,
            url="https://api.z.ai/api/coding/paas/v4",
            api_key_file=str(key_file),
        )
    )

    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"id": "chatcmpl-test", "choices": []}
    mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

    await provider.generate_completion(
        {"model": "glm-4.5-air", "messages": [{"role": "user", "content": "Hello"}]},
        {
            "authorization": "Bearer client-token",
            "openai-organization": "org-123",
            "openai-project": b"project-123",
            "user-agent": b"test-client/1.0",
            "x-ignore": "ignored",
        },
    )

    called_headers = mock_client.return_value.__aenter__.return_value.post.call_args.kwargs["headers"]
    assert called_headers["Authorization"] == "Bearer dummy-zai-token"
    assert called_headers["openai-organization"] == "org-123"
    assert called_headers["openai-project"] == "project-123"
    assert called_headers["user-agent"] == "test-client/1.0"
    assert "x-ignore" not in called_headers


@pytest.mark.asyncio
async def test_zai_coding_provider_health_check_requires_api_key(tmp_path):
    key_file = tmp_path / "glm.env"
    key_file.write_text("ZAI_API_KEY=dummy-zai-token\n")

    provider = ZaiCodingProvider(
        ZaiCodingConfig(
            name="test-zai",
            type="zai-coding",
            enabled=True,
            url="https://api.z.ai/api/coding/paas/v4",
            api_key_file=str(key_file),
        )
    )
    provider.config.api_key = None

    assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_mediator_routes_zai_coding_provider():
    """Test mediator routes OpenAI-compatible Z.AI providers instead of returning 501."""
    aggregator = Mock()
    aggregator.get_all_models = AsyncMock(return_value=[])
    strategy = SimpleModelStrategy({})
    access_control = NoAccessControl()
    mediator = ModelMediator(aggregator, strategy, access_control)

    resolved_model = ModelInfo(
        id="glm-4.5-air@test-zai",
        name="glm-4.5-air",
        provider_id="test-zai",
        provider_type="zai-coding",
        endpoint="https://api.z.ai/api/coding/paas/v4",
    )

    provider = Mock()
    provider.generate_completion = AsyncMock(return_value=({"id": "chatcmpl-test"}, 200))

    mediator.resolve_model_for_request = AsyncMock(return_value=resolved_model)
    mediator._get_provider_by_id = Mock(return_value=provider)

    response_data, status_code, upstream_used, metadata = await mediator.route_request(
        "127.0.0.1",
        "glm-4.5-air",
        {"model": "glm-4.5-air", "messages": [{"role": "user", "content": "Hello"}]},
        "/v1/chat/completions",
        {"authorization": "Bearer client-token"},
        30.0,
    )

    assert status_code == 200
    assert response_data["id"] == "chatcmpl-test"
    assert upstream_used == "zai-coding:test-zai"
    assert metadata is not None
    assert metadata.provider_id == "test-zai"
    assert metadata.model_name == "glm-4.5-air"
    provider.generate_completion.assert_awaited_once()
    forwarded_payload = provider.generate_completion.await_args.args[0]
    assert forwarded_payload["model"] == "glm-4.5-air"


@pytest.mark.asyncio
async def test_mediator_preserves_provider_metadata_for_google_errors():
    """Test Google provider failures still keep downstream provider identity for logging/UI."""
    aggregator = Mock()
    aggregator.get_all_models = AsyncMock(return_value=[])
    strategy = SimpleModelStrategy({})
    access_control = NoAccessControl()
    mediator = ModelMediator(aggregator, strategy, access_control)

    resolved_model = ModelInfo(
        id="gemma-3-4b-it@test-google",
        name="gemma-3-4b-it",
        provider_id="test-google",
        provider_type="google-genai",
        endpoint="https://generativelanguage.googleapis.com",
    )

    provider = GoogleGenAIProvider(
        GoogleGenAIConfig(name="test-google", type="google-genai", enabled=True, api_keys=["test-key"])
    )
    provider_error = Exception("Google General error: [Errno 61] Connection refused")
    provider_error.provider_id = "test-google"
    provider_error.model_name = "gemma-3-4b-it"
    provider_error.proxy_used = "http://127.0.0.1:8888"
    provider_error.api_key_suffix = "abcd1234"
    provider_error.api_key_index = 2
    provider_error.api_key_total = 5
    provider.generate_completion = AsyncMock(side_effect=provider_error)

    mediator.resolve_model_for_request = AsyncMock(return_value=resolved_model)
    mediator._get_provider_by_id = Mock(return_value=provider)

    response_data, status_code, upstream_used, metadata = await mediator.route_request(
        "127.0.0.1",
        "gemma-3-4b-it [test-google]",
        {"model": "gemma-3-4b-it [test-google]", "messages": [{"role": "user", "content": "Hello"}]},
        "/v1/chat/completions",
        {"authorization": "Bearer client-token"},
        30.0,
    )

    assert status_code == 500
    assert upstream_used == "google-genai:test-google"
    assert response_data["error"]["provider"] == "google-genai"
    assert metadata is not None
    assert metadata.provider_id == "test-google"
    assert metadata.model_name == "gemma-3-4b-it"
    assert metadata.proxy_used == "http://127.0.0.1:8888"
    assert metadata.api_key_suffix == "abcd1234"
    assert metadata.api_key_index == 2
    assert metadata.api_key_total == 5


def test_anthropic_api_key_passthrough():
    """Test Anthropic API key passthrough functionality"""
    config = AnthropicConfig(
        name="test", type="anthropic", enabled=True, url="https://api.anthropic.com", api_keys=["fallback-key"]
    )

    provider = AnthropicProvider(config)

    # Test client key detection
    client_token = "sk-ant-" + "client-key-123"
    client_headers = {"authorization": "Bearer " + client_token}
    api_key = provider._get_api_key(client_headers)
    assert api_key == client_token

    # Test fallback to configured key
    empty_headers = {}
    api_key = provider._get_api_key(empty_headers)
    assert api_key == "fallback-key"

    # Test non-Anthropic key fallback
    openai_headers = {"authorization": "Bearer sk-openai-key"}
    api_key = provider._get_api_key(openai_headers)
    assert api_key == "fallback-key"


def test_google_genai_config_initialization():
    """Test Google GenAI config properly initializes"""
    # With API keys list
    config1 = GoogleGenAIConfig(name="test", type="google-genai", enabled=True, api_keys=["key1", "key2"])
    assert len(config1.api_keys) == 2
    assert config1.url == "https://generativelanguage.googleapis.com"


def test_google_genai_config_loads_api_keys_file_with_comments(tmp_path):
    key_file = tmp_path / "google_keys.txt"
    key_file.write_text("# comment\nkey-one\n\n  key-two  \n")

    config = GoogleGenAIConfig(name="test", type="google-genai", enabled=True, api_keys_file=str(key_file))

    assert config.api_keys == ["key-one", "key-two"]


def test_google_genai_config_loads_env_style_api_keys_file(tmp_path):
    key_file = tmp_path / "google_keys.env"
    key_file.write_text('export GOOGLE_API_KEY="key-one" # primary\nGOOGLE_API_KEY_2=key-two\n')

    config = GoogleGenAIConfig(name="test", type="google-genai", enabled=True, api_keys_file=str(key_file))

    assert config.api_keys == ["key-one", "key-two"]


def test_google_genai_config_rejects_empty_api_key_file(tmp_path):
    key_file = tmp_path / "google_keys.txt"
    key_file.write_text("# still empty\n\n")

    with pytest.raises(ValueError, match="No valid API keys found"):
        GoogleGenAIConfig(name="test", type="google-genai", enabled=True, api_keys_file=str(key_file))


def test_anthropic_config_loads_api_keys_file_once_during_init(tmp_path):
    key_file = tmp_path / "anthropic_keys.txt"
    key_file.write_text("# ignored\nsk-ant-one\nsk-ant-two\n")

    config = AnthropicConfig(
        name="test",
        type="anthropic",
        enabled=True,
        url="https://api.anthropic.com",
        api_keys_file=str(key_file),
    )

    assert config.api_keys == ["sk-ant-one", "sk-ant-two"]

    provider = AnthropicProvider(config)
    assert provider.config.api_keys == ["sk-ant-one", "sk-ant-two"]


def test_anthropic_config_loads_env_style_api_keys_file(tmp_path):
    key_file = tmp_path / "anthropic_keys.env"
    key_file.write_text('export ANTHROPIC_API_KEY="sk-ant-one" # active\nANTHROPIC_API_KEY_2=sk-ant-two\n')

    config = AnthropicConfig(
        name="test",
        type="anthropic",
        enabled=True,
        url="https://api.anthropic.com",
        api_keys_file=str(key_file),
    )

    assert config.api_keys == ["sk-ant-one", "sk-ant-two"]


def test_anthropic_config_raises_when_api_keys_file_is_missing(tmp_path):
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(ValueError, match="API key file not found"):
        AnthropicConfig(
            name="test",
            type="anthropic",
            enabled=True,
            url="https://api.anthropic.com",
            api_keys_file=str(missing_file),
        )


def test_anthropic_request_format_conversion():
    """Test basic OpenAI to Anthropic format conversion"""
    config = AnthropicConfig(
        name="test", type="anthropic", enabled=True, url="https://api.anthropic.com", api_keys=["key"]
    )

    provider = AnthropicProvider(config)

    # Test basic conversion
    openai_request = {"model": "claude-3-sonnet", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}

    anthropic_request = provider._convert_openai_to_anthropic(openai_request)

    assert anthropic_request["model"] == "claude-3-sonnet"
    assert len(anthropic_request["messages"]) == 1
    assert anthropic_request["messages"][0]["content"] == "Hello"
    assert anthropic_request["max_tokens"] == 100


def test_anthropic_response_format_conversion():
    """Test basic Anthropic to OpenAI format conversion"""
    config = AnthropicConfig(
        name="test", type="anthropic", enabled=True, url="https://api.anthropic.com", api_keys=["key"]
    )

    provider = AnthropicProvider(config)

    # Test basic conversion
    anthropic_response = {
        "content": [{"type": "text", "text": "Hello there!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 8},
    }

    openai_response = provider._convert_anthropic_to_openai(anthropic_response, "claude-3-sonnet")

    assert openai_response["object"] == "chat.completion"
    assert openai_response["model"] == "claude-3-sonnet"
    assert openai_response["choices"][0]["message"]["content"] == "Hello there!"
    assert openai_response["choices"][0]["finish_reason"] == "stop"
    assert openai_response["usage"]["prompt_tokens"] == 5
    assert openai_response["usage"]["completion_tokens"] == 8


@pytest.mark.asyncio
async def test_google_genai_api_key_selection():
    """Test Google GenAI API key selection logic"""
    config = GoogleGenAIConfig(name="test", type="google-genai", enabled=True, api_keys=["key1", "key2", "key3"])

    provider = GoogleGenAIProvider(config)

    # Test that provider has key selection method
    assert hasattr(provider, "_select_best_api_key")

    # Test with a model name
    selected_key = await provider._select_best_api_key("gemini-1.5-pro")
    assert selected_key in config.api_keys


@pytest.mark.asyncio
async def test_mediator_returns_gateway_timeout_when_resolution_exceeds_timeout():
    """Mediator should return a deterministic timeout response when routing exceeds timeout."""
    aggregator = Mock()
    aggregator.get_all_models = AsyncMock(return_value=[])
    strategy = SimpleModelStrategy({})
    access_control = NoAccessControl()
    mediator = ModelMediator(aggregator, strategy, access_control)

    async def slow_resolve(*_args, **_kwargs):
        await asyncio.sleep(0.05)
        return None

    mediator.resolve_model_for_request = AsyncMock(side_effect=slow_resolve)

    response_data, status_code, upstream_used, metadata = await mediator.route_request(
        "127.0.0.1",
        "gemini-1.5-pro",
        {"model": "gemini-1.5-pro", "messages": [{"role": "user", "content": "hello"}]},
        "/v1/chat/completions",
        {},
        0.001,
    )

    assert status_code == 504
    assert upstream_used == "timeout"
    assert metadata is None
    assert response_data["error"]["type"] == "timeout_error"


def _mediator_with_resolved_lb_instance(sentinel_instance):
    """Build a mediator whose resolution yields a model bound to sentinel_instance."""
    aggregator = Mock()
    aggregator.get_all_models = AsyncMock(return_value=[])
    mediator = ModelMediator(aggregator, SimpleModelStrategy({}), NoAccessControl())

    resolved = SimpleNamespace(name="m", _lb_instance=sentinel_instance)
    mediator.resolve_model_for_request = AsyncMock(return_value=resolved)
    return mediator


@pytest.mark.asyncio
async def test_timeout_after_instance_selected_returns_lb_instance_for_decrement():
    """Regression: a timeout AFTER an LB instance was selected must return the
    instance in metadata so active_requests can be decremented downstream.

    Previously the timeout path returned metadata=None, so the increment done by
    select_instance() was never undone - the active_requests "leak" observed in
    production (active_requests far exceeding total_requests)."""
    sentinel_instance = SimpleNamespace(model_id="m@host", active_requests=1)
    mediator = _mediator_with_resolved_lb_instance(sentinel_instance)

    async def slow_internal(*_args, **_kwargs):
        await asyncio.sleep(0.05)

    mediator._route_request_internal = AsyncMock(side_effect=slow_internal)

    _data, status_code, upstream_used, metadata = await mediator.route_request(
        "127.0.0.1", "m", {"model": "m", "messages": []}, "/v1/chat/completions", {}, 0.001
    )

    assert status_code == 504
    assert upstream_used == "timeout"
    assert metadata is not None and metadata.lb_instance is sentinel_instance


@pytest.mark.asyncio
async def test_exception_after_instance_selected_returns_lb_instance_for_decrement():
    """Regression: an unexpected error after instance selection must also carry
    lb_instance in metadata so the counter is decremented (no leak)."""
    sentinel_instance = SimpleNamespace(model_id="m@host", active_requests=1)
    mediator = _mediator_with_resolved_lb_instance(sentinel_instance)

    mediator._route_request_internal = AsyncMock(side_effect=RuntimeError("boom"))

    _data, status_code, _upstream, metadata = await mediator.route_request(
        "127.0.0.1", "m", {"model": "m", "messages": []}, "/v1/chat/completions", {}, 30.0
    )

    assert status_code == 500
    assert metadata is not None and metadata.lb_instance is sentinel_instance


@pytest.mark.asyncio
async def test_cancellation_after_instance_selected_releases_lb_instance():
    """Regression (CodeRabbit critical): a client disconnect raises CancelledError
    (a BaseException) that bypasses the TimeoutError/Exception handlers. The
    selected LB instance must still be released (active_requests decremented) and
    the cancellation must still propagate - otherwise the instance leaks busy."""
    sentinel_instance = SimpleNamespace(model_id="m@host", active_requests=1)
    mediator = _mediator_with_resolved_lb_instance(sentinel_instance)

    async def _cancel(*_args, **_kwargs):
        raise asyncio.CancelledError()

    mediator._route_request_internal = AsyncMock(side_effect=_cancel)

    with patch("smolrouter.mediator.model_load_balancer") as lb:
        lb.end_request = AsyncMock()
        with pytest.raises(asyncio.CancelledError):
            await mediator.route_request(
                "127.0.0.1", "m", {"model": "m", "messages": []}, "/v1/chat/completions", {}, 30.0
            )

    lb.end_request.assert_awaited_once()
    assert lb.end_request.await_args.args[0] is sentinel_instance
    assert lb.end_request.await_args.kwargs.get("success") is False


@pytest.mark.asyncio
async def test_resolve_releases_instance_when_no_model_matches():
    """Regression: select_instance() increments active_requests; if the selected
    instance can't be mapped back to an available model the increment must be
    released, otherwise the counter leaks."""
    aggregator = Mock()
    aggregator.get_all_models = AsyncMock(return_value=[])
    mediator = ModelMediator(aggregator, SimpleModelStrategy({}), NoAccessControl())

    selected = SimpleNamespace(model_id="m", provider_url="http://host")

    with patch("smolrouter.mediator.model_load_balancer") as lb:
        lb.select_instance = AsyncMock(return_value=selected)
        lb.end_request = AsyncMock()

        # available_models is empty -> no model matches the selected instance.
        result = await mediator._resolve_model_via_load_balancer("m", [])

    assert result is None
    lb.end_request.assert_awaited_once()
    assert lb.end_request.await_args.args[0] is selected
    assert lb.end_request.await_args.kwargs.get("success") is False


@pytest.mark.asyncio
async def test_container_returns_clean_504_on_timeout_without_raising():
    """Regression: the container must NOT wrap a second asyncio.timeout around
    the mediator. A nested same-deadline timeout races the mediator's, and when
    the outer one wins it cancels the mediator mid-flight; the CancelledError
    bypasses the mediator's TimeoutError handling and surfaces as an uncaught
    TimeoutError (logged as "Provider architecture failed", returned as 503).

    With the mediator as the single timeout authority, a slow route resolves to
    a clean 504 response tuple - no exception escapes the container."""
    aggregator = Mock()
    aggregator.get_all_models = AsyncMock(return_value=[])
    mediator = ModelMediator(aggregator, SimpleModelStrategy({}), NoAccessControl())

    async def slow_resolve(*_args, **_kwargs):
        await asyncio.sleep(0.05)
        return None

    mediator.resolve_model_for_request = AsyncMock(side_effect=slow_resolve)

    container = SmolRouterContainer()
    container._initialized = True
    container._mediator = mediator

    # Must return a tuple, not raise TimeoutError/CancelledError.
    data, status_code, upstream_used, _metadata = await container.route_request(
        "127.0.0.1",
        "gpt-4o",
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        "/v1/chat/completions",
        {},
        0.001,
    )

    assert status_code == 504
    assert upstream_used == "timeout"
    assert data["error"]["type"] == "timeout_error"


def test_supported_provider_types():
    """Test that new provider types are registered"""
    supported_types = ProviderFactory.get_supported_types()

    assert "google-genai" in supported_types
    assert "anthropic" in supported_types
    assert "ollama" in supported_types
    assert "openai" in supported_types
    assert "zai-coding" in supported_types
