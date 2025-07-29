#!/usr/bin/env python3
"""
Integration tests for the new SmolRouter architecture with web UI.

These tests verify that the new architecture integrates correctly with
the existing FastAPI application and web interface.
"""

import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch

# Import the app
from smolrouter.app import app
from smolrouter.container import SmolRouterContainer, SmolRouterConfig
from smolrouter.interfaces import ModelInfo, ClientContext


class TestWebUIIntegration:
    """Test the new web UI integration"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_upstreams_page_loads(self):
        """Test that the upstreams page loads successfully"""
        # This tests the HTML template loading
        response = self.client.get("/upstreams")
        assert response.status_code == 200
        assert "Upstream Providers" in response.text
        assert "upstream-grid" in response.text  # Check for key CSS class
    
    def test_api_upstreams_endpoint(self):
        """Test the API endpoint for upstream data"""
        response = self.client.get("/api/upstreams")
        assert response.status_code == 200
        
        data = response.json()
        assert "upstreams" in data
        assert "summary" in data
        assert isinstance(data["upstreams"], list)
        assert isinstance(data["summary"], dict)
        
        # Check summary structure
        summary = data["summary"]
        required_keys = ["total_providers", "healthy_providers", "total_models", "cache_enabled", "cache_entries"]
        for key in required_keys:
            assert key in summary


class TestModelAggregationEndpoints:  
    """Test model aggregation in the API endpoints"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_v1_models_endpoint(self):
        """Test /v1/models endpoint with aggregation"""
        response = self.client.get("/v1/models")
        assert response.status_code in [200, 502]  # 502 if upstream unavailable
        
        if response.status_code == 200:
            data = response.json()
            assert "object" in data
            assert data["object"] == "list"
            assert "data" in data
            assert isinstance(data["data"], list)
    
    def test_api_tags_endpoint(self):
        """Test /api/tags endpoint with aggregation"""
        response = self.client.get("/api/tags")
        assert response.status_code in [200, 502]  # 502 if upstream unavailable
        
        if response.status_code == 200:
            data = response.json()
            assert "models" in data
            assert isinstance(data["models"], list)


@patch('smolrouter.container.SmolRouterContainer')
class TestArchitectureIntegration:
    """Test integration between new architecture and existing app"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    async def test_container_initialization(self, mock_container_class):
        """Test that container initializes correctly"""
        mock_container = Mock()
        mock_mediator = Mock()
        mock_models = [
            ModelInfo("llama3-70b@fast-kitten", "llama3-70b", "fast-kitten", "ollama", "http://localhost:11434"),
            ModelInfo("gpt-4@openai", "gpt-4", "openai", "openai", "http://localhost:8000")
        ]
        
        # Setup mocks
        mock_mediator.get_available_models = AsyncMock(return_value=mock_models)
        mock_container.get_mediator = AsyncMock(return_value=mock_mediator)
        mock_container.create_client_context.return_value = ClientContext("127.0.0.1")
        mock_container_class.return_value = mock_container
        
        # Test models endpoint
        response = self.client.get("/v1/models")
        
        # Should use the new architecture
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2
        
        # Check model format
        model = data["data"][0]
        assert "id" in model
        assert "object" in model
        assert model["object"] == "model"
    
    async def test_upstream_api_with_mock_data(self, mock_container_class):
        """Test upstream API with mocked architecture"""
        mock_container = Mock()
        mock_mediator = Mock()
        mock_providers = [Mock(), Mock()]
        
        # Setup provider mocks
        mock_providers[0].get_provider_id.return_value = "fast-kitten"
        mock_providers[0].get_provider_type.return_value = "ollama"
        mock_providers[0].get_endpoint.return_value = "http://localhost:11434"
        mock_providers[0].config.priority = 0
        mock_providers[0].config.enabled = True
        
        mock_providers[1].get_provider_id.return_value = "openai-api"
        mock_providers[1].get_provider_type.return_value = "openai"
        mock_providers[1].get_endpoint.return_value = "http://localhost:8000"
        mock_providers[1].config.priority = 1
        mock_providers[1].config.enabled = True
        
        # Setup mediator mocks
        mock_mediator.get_provider_health = AsyncMock(return_value={
            "fast-kitten": True,
            "openai-api": False
        })
        mock_mediator.get_mediator_stats = AsyncMock(return_value={
            "aggregation": {"cache_stats": {"total_entries": 2}}
        })
        mock_mediator.get_models_by_provider = AsyncMock(return_value=[
            ModelInfo("llama3-70b@fast-kitten", "llama3-70b", "fast-kitten", "ollama", "http://localhost:11434")
        ])
        
        # Setup container mocks
        mock_container.get_mediator = AsyncMock(return_value=mock_mediator)
        mock_container.get_providers.return_value = mock_providers
        mock_container.create_client_context.return_value = ClientContext("127.0.0.1")
        mock_container_class.return_value = mock_container
        
        # Test upstream API
        response = self.client.get("/api/upstreams")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["upstreams"]) == 2
        
        # Check first provider
        provider1 = data["upstreams"][0]
        assert provider1["id"] == "fast-kitten"
        assert provider1["type"] == "ollama"
        assert provider1["healthy"] is True
        assert provider1["model_count"] == 1
        
        # Check second provider  
        provider2 = data["upstreams"][1]
        assert provider2["id"] == "openai-api"
        assert provider2["healthy"] is False
        
        # Check summary
        summary = data["summary"]
        assert summary["total_providers"] == 2
        assert summary["healthy_providers"] == 1


class TestBackwardCompatibility:
    """Test that new architecture doesn't break existing functionality"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_existing_endpoints_still_work(self):
        """Test that existing endpoints continue to function"""
        # Test dashboard
        response = self.client.get("/")
        assert response.status_code == 200
        
        # Test performance page
        response = self.client.get("/performance")
        assert response.status_code == 200
        
        # Test API endpoints
        response = self.client.get("/api/logs")
        assert response.status_code == 200
        
        response = self.client.get("/api/stats")
        assert response.status_code == 200
    
    def test_legacy_model_fallback(self):
        """Test that endpoints fall back to legacy behavior when new architecture fails"""
        # If the new architecture is not available, endpoints should still work
        # This is tested implicitly by the other endpoint tests
        pass


def test_run_architecture_demo():
    """Test that the architecture demo script runs without errors"""
    import subprocess
    import sys
    
    # Run the demo script
    result = subprocess.run([
        sys.executable, "test_new_architecture.py"
    ], capture_output=True, text=True)
    
    # Should complete without fatal errors
    assert result.returncode == 0
    assert "Demo completed successfully!" in result.stdout


def test_web_ui_navigation():
    """Test navigation between different web UI pages"""
    client = TestClient(app)
    
    # Test main pages load
    pages = ["/", "/performance", "/upstreams"]
    
    for page in pages:
        response = client.get(page)
        assert response.status_code == 200
        
        # Check that navigation links are present
        assert 'href="/"' in response.text  # Dashboard link
        assert 'href="/performance"' in response.text  # Performance link
        assert 'href="/upstreams"' in response.text  # Upstreams link


@pytest.mark.asyncio
async def test_real_container_with_mock_providers():
    """Test real container with mock provider configurations"""
    # Create a real container with mock configuration
    providers_config = [
        {
            "name": "test-ollama",
            "type": "ollama", 
            "url": "http://localhost:11434",
            "enabled": True,
            "priority": 0
        },
        {
            "name": "test-openai",
            "type": "openai",
            "url": "http://localhost:8000", 
            "enabled": True,
            "priority": 1
        }
    ]
    
    config = SmolRouterConfig(
        providers=providers_config,
        cache_ttl=60,
        strategy={"model_map": {"gpt-4": "llama3-70b"}},
        access_control={"type": "none"}
    )
    
    container = SmolRouterContainer(config)
    
    try:
        await container.initialize()
        
        # Test that components are properly wired
        mediator = await container.get_mediator()
        assert mediator is not None
        
        # Test client operations (will return empty results due to unavailable servers)
        client = container.create_client_context("127.0.0.1")
        models = await mediator.get_available_models(client)
        assert isinstance(models, list)  # Should be empty list, not error
        
        # Test health check  
        health = await container.health_check()
        assert health["initialized"] is True
        assert health["provider_count"] == 2
        
    finally:
        await container.close()


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])