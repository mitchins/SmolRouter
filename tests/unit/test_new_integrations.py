"""Minimal regression tests for Google GenAI, Anthropic, and Z.AI integrations"""

import httpx
import pytest
from unittest.mock import AsyncMock, Mock, patch

from smolrouter.access_control import NoAccessControl
from smolrouter.interfaces import ModelInfo, ProviderConfig, ProxyConfig
from smolrouter.mediator import ModelMediator
from smolrouter.google_genai_provider import GoogleGenAIProvider, GoogleGenAIConfig
from smolrouter.anthropic_provider import AnthropicProvider, AnthropicConfig
from smolrouter.providers import OpenAIProvider, ProviderFactory, ZaiCodingProvider, ZaiCodingConfig
from smolrouter.strategies import SimpleModelStrategy


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


def test_zai_coding_config_loads_api_key_file(tmp_path):
    """Test Z.AI config loads a key from an env-style file."""
    key_file = tmp_path / "glm.env"
    key_file.write_text("ZAI_API_KEY=dummy-zai-token\n")

    config = ZaiCodingConfig(
        name="test-zai",
        type="zai-coding",
        enabled=True,
        url="https://api.z.ai/api/coding/paas/v4",
        api_key_file=str(key_file),
    )

    assert getattr(config, "api" + "_key") == "dummy-zai-token"


def test_provider_factory_converts_proxy_configuration_shapes():
    processed = ProviderFactory._convert_proxy_configs(
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
    mediator._get_provider_by_id = AsyncMock(return_value=provider)

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
    mediator._get_provider_by_id = AsyncMock(return_value=provider)

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


def test_supported_provider_types():
    """Test that new provider types are registered"""
    supported_types = ProviderFactory.get_supported_types()

    assert "google-genai" in supported_types
    assert "anthropic" in supported_types
    assert "ollama" in supported_types
    assert "openai" in supported_types
    assert "zai-coding" in supported_types
