#!/usr/bin/env python3
"""
Comprehensive unit tests for the new SmolRouter SOLID architecture.

Tests cover all major components:
- Model providers
- Caching system  
- Strategies and resolution
- Access control
- Model mediator
- Dependency injection container
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from smolrouter.interfaces import (
    ModelInfo, ClientContext, ProviderConfig, IModelProvider, 
    IModelStrategy, IAccessControl, IModelCache
)
from smolrouter.providers import (
    OllamaProvider, OpenAIProvider, ProviderFactory
)
from smolrouter.caching import (
    InMemoryModelCache, NoOpModelCache, ModelAggregator, CacheEntry
)
from smolrouter.strategies import (
    SmartModelStrategy, SimpleModelStrategy, AliasRule
)
from smolrouter.access_control import (
    NoAccessControl, IPBasedAccessControl, AuthBasedAccessControl
)
from smolrouter.mediator import ModelMediator
from smolrouter.container import SmolRouterContainer, SmolRouterConfig


class TestModelInfo:
    """Test ModelInfo data class"""
    
    def test_model_info_creation(self):
        model = ModelInfo(
            id="llama3-70b@fast-kitten",
            name="llama3-70b",
            provider_id="fast-kitten",
            provider_type="ollama",
            endpoint="http://localhost:11434",
            aliases=["llama3", "llama"],
            metadata={"size": 40000000000}
        )
        
        assert model.id == "llama3-70b@fast-kitten"
        assert model.display_name == "llama3-70b [fast-kitten]"
        assert model.aliases == ["llama3", "llama"]
        assert model.metadata["size"] == 40000000000
    
    def test_model_matches_request(self):
        model = ModelInfo(
            id="llama3-70b@fast-kitten",
            name="llama3-70b", 
            provider_id="fast-kitten",
            provider_type="ollama",
            endpoint="http://localhost:11434",
            aliases=["llama3", "llama"]
        )
        
        # Test exact matches
        assert model.matches_request("llama3-70b@fast-kitten")
        assert model.matches_request("llama3-70b")
        assert model.matches_request("llama3")
        assert model.matches_request("llama")
        assert model.matches_request("llama3-70b [fast-kitten]")
        
        # Test non-matches
        assert not model.matches_request("gpt-4")
        assert not model.matches_request("llama3-70b@other-provider")


class TestClientContext:
    """Test ClientContext data class"""
    
    def test_client_context_creation(self):
        client = ClientContext(
            ip="192.168.1.100",
            auth_payload={"sub": "user123", "roles": ["admin"]},
            user_agent="test-client/1.0",
            headers={"authorization": "Bearer token123"}
        )
        
        assert client.ip == "192.168.1.100"
        assert client.user_id == "user123"
        assert client.headers["authorization"] == "Bearer token123"
    
    def test_user_id_extraction(self):
        # Test different auth payload formats
        contexts = [
            ClientContext("127.0.0.1", {"sub": "user1"}),
            ClientContext("127.0.0.1", {"user": "user2"}),  
            ClientContext("127.0.0.1", {"username": "user3"}),
            ClientContext("127.0.0.1", None)
        ]
        
        assert contexts[0].user_id == "user1"
        assert contexts[1].user_id == "user2"
        assert contexts[2].user_id == "user3"
        assert contexts[3].user_id is None


class TestProviders:
    """Test model provider implementations"""
    
    def test_provider_config_validation(self):
        # Valid config
        config = ProviderConfig(name="test", type="ollama", url="http://localhost:11434")
        assert config.name == "test"
        assert config.enabled is True
        assert config.priority == 0
        
        # Test metadata initialization
        config2 = ProviderConfig(name="test2", type="openai", url="http://localhost:8000")
        assert config2.metadata == {}
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_ollama_provider_health_check(self, mock_client):
        """Test Ollama provider health checking"""
        config = ProviderConfig(name="test-ollama", type="ollama", url="http://localhost:11434")
        provider = OllamaProvider(config)
        
        # Mock successful health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        
        is_healthy = await provider.health_check()
        assert is_healthy is True
        
        # Mock failed health check
        mock_client.return_value.__aenter__.return_value.get.side_effect = Exception("Connection failed")
        is_healthy = await provider.health_check()
        assert is_healthy is False
    
    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_ollama_provider_discover_models(self, mock_client):
        """Test Ollama model discovery"""
        config = ProviderConfig(name="test-ollama", type="ollama", url="http://localhost:11434")
        provider = OllamaProvider(config)
        
        # Mock successful model discovery
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llama3:8b",
                    "size": 4000000000,
                    "modified_at": "2024-01-01T00:00:00Z",
                    "digest": "sha256:abc123"
                },
                {
                    "name": "codellama:34b", 
                    "size": 20000000000,
                    "modified_at": "2024-01-02T00:00:00Z",
                    "digest": "sha256:def456"
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        
        models = await provider.discover_models()
        
        assert len(models) == 2
        assert models[0].name == "llama3:8b"
        assert models[0].id == "llama3:8b@test-ollama"
        assert models[0].provider_type == "ollama"
        assert "llama3-8b" in models[0].aliases  # Test normalization
        assert models[0].metadata["size"] == 4000000000
    
    def test_provider_factory(self):
        """Test provider factory"""
        # Test Ollama provider creation
        ollama_config = ProviderConfig(name="ollama-test", type="ollama", url="http://localhost:11434")
        ollama_provider = ProviderFactory.create_provider(ollama_config)
        assert isinstance(ollama_provider, OllamaProvider)
        
        # Test OpenAI provider creation
        openai_config = ProviderConfig(name="openai-test", type="openai", url="http://localhost:8000")
        openai_provider = ProviderFactory.create_provider(openai_config)
        assert isinstance(openai_provider, OpenAIProvider)
        
        # Test invalid type
        with pytest.raises(ValueError, match="Unknown provider type"):
            invalid_config = ProviderConfig(name="invalid", type="invalid", url="http://localhost:8000")
            ProviderFactory.create_provider(invalid_config)


class TestCaching:
    """Test caching implementations"""
    
    @pytest.mark.asyncio
    async def test_noop_cache(self):
        """Test no-op cache implementation"""
        cache = NoOpModelCache()
        
        # Should always return None/False
        assert await cache.get_cached_models("test") is None
        assert await cache.is_cache_valid("test") is False
        
        # Should not raise errors
        await cache.cache_models("test", [], 300)
        await cache.invalidate_cache("test")
    
    @pytest.mark.asyncio
    async def test_inmemory_cache_basic_operations(self):
        """Test basic in-memory cache operations"""
        cache = InMemoryModelCache(default_ttl=1)  # 1 second TTL for testing
        
        models = [
            ModelInfo("test1@provider", "test1", "provider", "ollama", "http://localhost:11434"),
            ModelInfo("test2@provider", "test2", "provider", "ollama", "http://localhost:11434")
        ]
        
        # Test cache miss
        assert await cache.get_cached_models("provider") is None
        assert not await cache.is_cache_valid("provider")
        
        # Test cache hit
        await cache.cache_models("provider", models, 5)
        cached = await cache.get_cached_models("provider")
        assert len(cached) == 2
        assert cached[0].id == "test1@provider"
        assert await cache.is_cache_valid("provider")
        
        # Test cache expiration
        await cache.cache_models("provider", models, 0.1)  # Very short TTL
        await asyncio.sleep(0.2)
        assert await cache.get_cached_models("provider") is None
        assert not await cache.is_cache_valid("provider")
    
    @pytest.mark.asyncio
    async def test_inmemory_cache_invalidation(self):
        """Test cache invalidation"""
        cache = InMemoryModelCache()
        
        models = [ModelInfo("test@provider", "test", "provider", "ollama", "http://localhost:11434")]
        
        # Cache some data
        await cache.cache_models("provider1", models, 300)
        await cache.cache_models("provider2", models, 300) 
        
        # Test specific provider invalidation
        await cache.invalidate_cache("provider1")
        assert await cache.get_cached_models("provider1") is None
        assert await cache.get_cached_models("provider2") is not None
        
        # Test full cache invalidation
        await cache.invalidate_cache()
        assert await cache.get_cached_models("provider2") is None
    
    @pytest.mark.asyncio
    async def test_cache_entry(self):
        """Test cache entry functionality"""
        entry = CacheEntry(data=["test"], cached_at=time.time(), ttl_seconds=1)
        
        assert not entry.is_expired()
        assert entry.access_count == 0
        
        entry.touch()
        assert entry.access_count == 1
        
        # Test expiration
        entry.cached_at = time.time() - 2  # 2 seconds ago with 1 second TTL
        assert entry.is_expired()
    
    @patch('smolrouter.providers.OllamaProvider.discover_models')
    @patch('smolrouter.providers.OllamaProvider.health_check')
    @pytest.mark.asyncio
    async def test_model_aggregator(self, mock_health, mock_discover):
        """Test model aggregator functionality"""
        # Create mock providers
        config1 = ProviderConfig(name="provider1", type="ollama", url="http://localhost:11434")
        config2 = ProviderConfig(name="provider2", type="openai", url="http://localhost:8000")
        
        provider1 = OllamaProvider(config1)
        provider2 = OpenAIProvider(config2)
        
        # Mock responses
        models1 = [ModelInfo("model1@provider1", "model1", "provider1", "ollama", "http://localhost:11434")]
        models2 = [ModelInfo("model2@provider2", "model2", "provider2", "openai", "http://localhost:8000")]
        
        mock_health.return_value = True
        mock_discover.side_effect = [models1, models2]
        
        # Create aggregator
        cache = InMemoryModelCache()
        aggregator = ModelAggregator([provider1, provider2], cache, default_cache_ttl=300)
        
        # Test aggregation
        all_models = await aggregator.get_all_models()
        assert len(all_models) == 2
        assert all_models[0].id == "model1@provider1"
        assert all_models[1].id == "model2@provider2"
        
        # Test caching (second call should use cache)
        mock_discover.reset_mock()
        all_models2 = await aggregator.get_all_models()
        assert len(all_models2) == 2
        mock_discover.assert_not_called()  # Should use cache
        
        # Test provider-specific queries
        provider1_models = await aggregator.get_models_by_provider("provider1")
        assert len(provider1_models) == 1
        assert provider1_models[0].id == "model1@provider1"


class TestStrategies:
    """Test model resolution strategies"""
    
    def test_alias_rule(self):
        """Test alias rule functionality"""
        # Test exact match rule
        rule1 = AliasRule(pattern="gpt-4", target="llama3-70b", priority=0)
        assert rule1.matches("gpt-4")
        assert not rule1.matches("gpt-3.5")
        assert rule1.apply("gpt-4") == "llama3-70b"
        
        # Test regex rule
        rule2 = AliasRule(pattern="/gpt-.*/", target="llama3-8b", priority=0)
        assert rule2.matches("gpt-4")
        assert rule2.matches("gpt-3.5-turbo")
        assert not rule2.matches("claude-3")
        
        # Test regex substitution
        rule3 = AliasRule(pattern="/gpt-(.*)/", target="llama3-\\1", priority=0)
        assert rule3.apply("gpt-4") == "llama3-4"
    
    @pytest.mark.asyncio
    async def test_simple_strategy(self):
        """Test simple model strategy"""
        aliases = {"gpt-4": "llama3-70b", "gpt-3.5-turbo": "llama3-8b"}
        strategy = SimpleModelStrategy(aliases)
        
        models = [
            ModelInfo("llama3-70b@provider", "llama3-70b", "provider", "ollama", "http://localhost:11434"),
            ModelInfo("llama3-8b@provider", "llama3-8b", "provider", "ollama", "http://localhost:11434")
        ]
        
        # Test alias resolution
        resolved = await strategy.resolve_model_request("gpt-4", models)
        assert resolved is not None
        assert resolved.name == "llama3-70b"
        
        # Test direct match
        resolved = await strategy.resolve_model_request("llama3-8b", models)
        assert resolved is not None
        assert resolved.name == "llama3-8b"
        
        # Test no match
        resolved = await strategy.resolve_model_request("unknown-model", models)
        assert resolved is None
    
    @pytest.mark.asyncio
    async def test_smart_strategy(self):
        """Test smart model strategy"""
        config = {
            'model_map': {'gpt-4': 'llama3-70b'},
            'servers': {'provider1': 'http://localhost:11434', 'provider2': 'http://localhost:8000'},
            'provider_priorities': {'provider1': 0, 'provider2': 1}
        }
        
        strategy = SmartModelStrategy(config)
        
        models = [
            ModelInfo("llama3-70b@provider2", "llama3-70b", "provider2", "openai", "http://localhost:8000"),
            ModelInfo("llama3-70b@provider1", "llama3-70b", "provider1", "ollama", "http://localhost:11434")
        ]
        
        # Test alias with priority
        resolved = await strategy.resolve_model_request("gpt-4", models)
        assert resolved is not None
        assert resolved.provider_id == "provider1"  # Higher priority
        
        # Test fully qualified name
        resolved = await strategy.resolve_model_request("llama3-70b [provider2]", models)
        assert resolved is not None
        assert resolved.provider_id == "provider2"
    
    def test_smart_strategy_fq_parsing(self):
        """Test fully qualified name parsing"""
        strategy = SmartModelStrategy()
        
        # Test valid FQ names
        assert strategy._parse_fully_qualified_name("llama3-70b [fast-kitten]") == ("llama3-70b", "fast-kitten")
        assert strategy._parse_fully_qualified_name("model [provider-1]") == ("model", "provider-1")
        
        # Test invalid formats
        assert strategy._parse_fully_qualified_name("llama3-70b") is None
        assert strategy._parse_fully_qualified_name("llama3-70b [") is None


class TestAccessControl:
    """Test access control implementations"""
    
    @pytest.mark.asyncio
    async def test_no_access_control(self):
        """Test no-op access control"""
        access_control = NoAccessControl()
        client = ClientContext("192.168.1.100")
        
        models = [
            ModelInfo("model1@provider", "model1", "provider", "ollama", "http://localhost:11434"),
            ModelInfo("model2@provider", "model2", "provider", "openai", "http://localhost:8000")
        ]
        
        # Should allow all models
        filtered = await access_control.filter_models(models, client)
        assert len(filtered) == 2
        
        # Should allow access to any model
        assert await access_control.can_access_model(models[0], client)
    
    @pytest.mark.asyncio
    async def test_ip_based_access_control(self):
        """Test IP-based access control"""
        rules = {
            "192.168.1.100": ["llama*", "/.*-8b/"],
            "192.168.1.101": {"allow": ["gpt-*"], "deny": ["gpt-4"]}
        }
        
        access_control = IPBasedAccessControl(rules)
        
        models = [
            ModelInfo("llama3-70b@provider", "llama3-70b", "provider", "ollama", "http://localhost:11434"),
            ModelInfo("llama3-8b@provider", "llama3-8b", "provider", "ollama", "http://localhost:11434"),
            ModelInfo("gpt-4@provider", "gpt-4", "provider", "openai", "http://localhost:8000"),
            ModelInfo("gpt-3.5@provider", "gpt-3.5", "provider", "openai", "http://localhost:8000")
        ]
        
        # Test first client (allows llama* and *-8b patterns)
        client1 = ClientContext("192.168.1.100")
        filtered1 = await access_control.filter_models(models, client1)
        assert len(filtered1) == 2  # Both llama models match
        
        # Test second client (allows gpt-* except gpt-4)
        client2 = ClientContext("192.168.1.101")
        filtered2 = await access_control.filter_models(models, client2)
        assert len(filtered2) == 1  # Only gpt-3.5
        assert filtered2[0].name == "gpt-3.5"
        
        # Test unknown client (no restrictions)
        client3 = ClientContext("10.0.0.1")
        filtered3 = await access_control.filter_models(models, client3)
        assert len(filtered3) == 4  # All models allowed
    
    @pytest.mark.asyncio
    async def test_auth_based_access_control(self):
        """Test authentication-based access control"""
        rules = {
            "default": {"allowed_models": ["llama*"]},
            "roles": {
                "admin": {"allow_all": True},
                "user": {"allowed_models": ["gpt-3.5*"]}
            },
            "users": {
                "special_user": {"allowed_models": ["gpt-4"]}
            }
        }
        
        access_control = AuthBasedAccessControl(rules)
        
        models = [
            ModelInfo("llama3-70b@provider", "llama3-70b", "provider", "ollama", "http://localhost:11434"),
            ModelInfo("gpt-4@provider", "gpt-4", "provider", "openai", "http://localhost:8000"),
            ModelInfo("gpt-3.5@provider", "gpt-3.5", "provider", "openai", "http://localhost:8000")
        ]
        
        # Test admin role (should see all)
        admin_client = ClientContext("127.0.0.1", {"sub": "admin_user", "roles": ["admin"]})
        admin_filtered = await access_control.filter_models(models, admin_client)
        assert len(admin_filtered) == 3
        
        # Test user role (should see gpt-3.5 only)
        user_client = ClientContext("127.0.0.1", {"sub": "regular_user", "roles": ["user"]})
        user_filtered = await access_control.filter_models(models, user_client)
        assert len(user_filtered) == 1
        assert user_filtered[0].name == "gpt-3.5"
        
        # Test specific user (should see gpt-4 only)
        special_client = ClientContext("127.0.0.1", {"sub": "special_user"})
        special_filtered = await access_control.filter_models(models, special_client)
        assert len(special_filtered) == 1
        assert special_filtered[0].name == "gpt-4"


class TestMediator:
    """Test model mediator orchestration"""
    
    @pytest.mark.asyncio
    async def test_mediator_integration(self):
        """Test mediator with all components"""
        # Create mock aggregator
        aggregator = Mock()
        models = [
            ModelInfo("llama3-70b@provider1", "llama3-70b", "provider1", "ollama", "http://localhost:11434"),
            ModelInfo("gpt-4@provider2", "gpt-4", "provider2", "openai", "http://localhost:8000")
        ]
        aggregator.get_all_models = AsyncMock(return_value=models)
        
        # Create strategy
        strategy = SimpleModelStrategy({"gpt-4": "llama3-70b"})
        
        # Create access control
        access_control = NoAccessControl()
        
        # Create mediator
        mediator = ModelMediator(aggregator, strategy, access_control)
        
        # Test getting available models
        client = ClientContext("192.168.1.100")
        available = await mediator.get_available_models(client)
        assert len(available) == 2
        
        # Test model resolution
        resolved = await mediator.resolve_model_for_request("gpt-4", client)
        assert resolved is not None
        assert resolved.name == "llama3-70b"  # Should resolve via alias
        
        # Test non-existent model
        not_found = await mediator.resolve_model_for_request("unknown", client)
        assert not_found is None


class TestContainer:
    """Test dependency injection container"""
    
    def test_container_config_creation(self):
        """Test container configuration"""
        providers_config = [
            {"name": "test1", "type": "ollama", "url": "http://localhost:11434", "enabled": True},
            {"name": "test2", "type": "openai", "url": "http://localhost:8000", "enabled": False}
        ]
        
        config = SmolRouterConfig(providers=providers_config)
        assert len(config.providers) == 2
        assert config.cache_enabled is True
        assert config.cache_ttl == 300
    
    @pytest.mark.asyncio
    async def test_container_initialization(self):
        """Test container initialization"""
        providers_config = [
            {"name": "test1", "type": "ollama", "url": "http://localhost:11434", "enabled": True}
        ]
        
        config = SmolRouterConfig(providers=providers_config)
        container = SmolRouterContainer(config)
        
        await container.initialize()
        
        # Test that components are created
        mediator = await container.get_mediator()
        assert mediator is not None
        
        providers = container.get_providers()
        assert len(providers) == 1
        assert providers[0].get_provider_id() == "test1"
        
        # Test client context creation
        client = container.create_client_context("192.168.1.100")
        assert client.ip == "192.168.1.100"
        
        # Cleanup
        await container.close()


# Integration test
@pytest.mark.asyncio
async def test_end_to_end_integration():
    """Test end-to-end integration with real components"""
    # Create a realistic configuration
    providers_config = [
        {"name": "ollama-local", "type": "ollama", "url": "http://localhost:11434", "priority": 0},
        {"name": "openai-api", "type": "openai", "url": "http://localhost:8000", "priority": 1}
    ]
    
    config = SmolRouterConfig(
        providers=providers_config,
        cache_ttl=60,
        strategy={
            "model_map": {"gpt-4": "llama3-70b"},
            "provider_priorities": {"ollama-local": 0, "openai-api": 1}
        },
        access_control={"type": "none"}
    )
    
    container = SmolRouterContainer(config)
    
    try:
        await container.initialize()
        
        # Test health check
        health = await container.health_check()
        assert health["initialized"] is True
        assert health["provider_count"] == 2
        
        # Test mediator access
        mediator = await container.get_mediator()
        assert mediator is not None
        
        # Test client operations (will fail due to mock servers, but architecture should handle gracefully)
        client = container.create_client_context("127.0.0.1")
        models = await mediator.get_available_models(client)
        assert isinstance(models, list)  # Should return empty list, not error
        
    finally:
        await container.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])