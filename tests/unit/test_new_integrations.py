"""Minimal regression tests for Google GenAI and Anthropic integrations"""

import pytest
from smolrouter.google_genai_provider import GoogleGenAIProvider, GoogleGenAIConfig
from smolrouter.anthropic_provider import AnthropicProvider, AnthropicConfig
from smolrouter.providers import ProviderFactory


def test_google_genai_provider_creation():
    """Test Google GenAI provider can be created with config"""
    config = GoogleGenAIConfig(
        name="test-google",
        type="google-genai",
        enabled=True,
        api_keys=["test-key"]
    )

    provider = GoogleGenAIProvider(config)

    assert provider.get_provider_id() == "test-google"
    assert provider.get_provider_type() == "google-genai"
    assert provider.get_endpoint() == "https://generativelanguage.googleapis.com"


def test_anthropic_provider_creation():
    """Test Anthropic provider can be created with config"""
    config = AnthropicConfig(
        name="test-anthropic",
        type="anthropic",
        enabled=True,
        url="https://api.anthropic.com",
        api_keys=["sk-ant-test"]
    )

    provider = AnthropicProvider(config)

    assert provider.get_provider_id() == "test-anthropic"
    assert provider.get_provider_type() == "anthropic"
    assert provider.get_endpoint() == "https://api.anthropic.com"




def test_provider_factory_integration():
    """Test provider factory can create new provider types"""
    # Google GenAI config
    google_config = {
        "name": "test-google",
        "type": "google-genai",
        "enabled": True,
        "api_keys": ["test-key"]
    }

    # Anthropic config
    anthropic_config = {
        "name": "test-anthropic",
        "type": "anthropic",
        "enabled": True,
        "url": "https://api.anthropic.com",
        "api_keys": ["sk-ant-test"]
    }

    providers = ProviderFactory.create_providers_from_config([google_config, anthropic_config])

    assert len(providers) == 2

    google_provider = next(p for p in providers if p.get_provider_type() == "google-genai")
    anthropic_provider = next(p for p in providers if p.get_provider_type() == "anthropic")

    assert google_provider.get_provider_id() == "test-google"
    assert anthropic_provider.get_provider_id() == "test-anthropic"


def test_anthropic_api_key_passthrough():
    """Test Anthropic API key passthrough functionality"""
    config = AnthropicConfig(
        name="test",
        type="anthropic",
        enabled=True,
        url="https://api.anthropic.com",
        api_keys=["fallback-key"]
    )

    provider = AnthropicProvider(config)

    # Test client key detection
    client_headers = {"authorization": "Bearer sk-ant-client-key-123"}
    api_key = provider._get_api_key(client_headers)
    assert api_key == "sk-ant-client-key-123"

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
    config1 = GoogleGenAIConfig(
        name="test",
        type="google-genai",
        enabled=True,
        api_keys=["key1", "key2"]
    )
    assert len(config1.api_keys) == 2


def test_anthropic_request_format_conversion():
    """Test basic OpenAI to Anthropic format conversion"""
    config = AnthropicConfig(
        name="test",
        type="anthropic",
        enabled=True,
        url="https://api.anthropic.com",
        api_keys=["key"]
    )

    provider = AnthropicProvider(config)

    # Test basic conversion
    openai_request = {
        "model": "claude-3-sonnet",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 100
    }

    anthropic_request = provider._convert_openai_to_anthropic(openai_request)

    assert anthropic_request["model"] == "claude-3-sonnet"
    assert len(anthropic_request["messages"]) == 1
    assert anthropic_request["messages"][0]["content"] == "Hello"
    assert anthropic_request["max_tokens"] == 100


def test_anthropic_response_format_conversion():
    """Test basic Anthropic to OpenAI format conversion"""
    config = AnthropicConfig(
        name="test",
        type="anthropic",
        enabled=True,
        url="https://api.anthropic.com",
        api_keys=["key"]
    )

    provider = AnthropicProvider(config)

    # Test basic conversion
    anthropic_response = {
        "content": [{"type": "text", "text": "Hello there!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 8}
    }

    openai_response = provider._convert_anthropic_to_openai(anthropic_response, "claude-3-sonnet")

    assert openai_response["object"] == "chat.completion"
    assert openai_response["model"] == "claude-3-sonnet"
    assert openai_response["choices"][0]["message"]["content"] == "Hello there!"
    assert openai_response["choices"][0]["finish_reason"] == "stop"
    assert openai_response["usage"]["prompt_tokens"] == 5
    assert openai_response["usage"]["completion_tokens"] == 8


def test_google_genai_api_key_selection():
    """Test Google GenAI API key selection logic"""
    config = GoogleGenAIConfig(
        name="test",
        type="google-genai",
        enabled=True,
        api_keys=["key1", "key2", "key3"]
    )

    provider = GoogleGenAIProvider(config)

    # Test that provider has key selection method
    assert hasattr(provider, '_select_best_api_key')

    # Test with a model name
    selected_key = provider._select_best_api_key("gemini-1.5-pro")
    assert selected_key in config.api_keys


def test_supported_provider_types():
    """Test that new provider types are registered"""
    supported_types = ProviderFactory.get_supported_types()

    assert "google-genai" in supported_types
    assert "anthropic" in supported_types
    assert "ollama" in supported_types
    assert "openai" in supported_types