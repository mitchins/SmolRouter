#!/usr/bin/env python3
"""
Test script to create providers with proxy configuration for testing
"""

from smolrouter.google_genai_provider import GoogleGenAIProvider, GoogleGenAIConfig
from smolrouter.interfaces import ProxyConfig


def create_test_providers():
    """Create test providers with proxy configuration"""

    # Create proxy config for test
    proxy_config = ProxyConfig(
        https_proxy="http://localhost:8080"  # Our test proxy
    )

    # Configure per-model proxy - gemini-2.0-flash uses our test proxy
    per_model_proxy = {"gemini-2.0-flash": proxy_config}

    # Create Google GenAI config with proxy
    config = GoogleGenAIConfig(
        name="google-test",
        endpoint="https://generativelanguage.googleapis.com",
        api_keys=["GOOGLE-API-KEY-GOES-HERE"],
        priority=1,
        per_model_proxy=per_model_proxy,
    )

    # Create provider
    provider = GoogleGenAIProvider(config)

    print("✅ Created GoogleGenAI provider with proxy config:")
    print(f"   - Provider: {provider.get_provider_id()}")
    print(f"   - Per-model proxy: {per_model_proxy}")
    print(f"   - Test proxy URL: {proxy_config.https_proxy}")

    return [provider]


if __name__ == "__main__":
    providers = create_test_providers()
    print(f"Created {len(providers)} providers for testing")
