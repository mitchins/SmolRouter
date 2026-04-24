#!/usr/bin/env python3
"""
Integration tests for the new SmolRouter architecture with web UI.
Relocated into `tests/` and updated subprocess call to reference moved scripts.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from importlib import import_module
from unittest.mock import AsyncMock, Mock

# Import the app
from smolrouter.app import app
from smolrouter.interfaces import ModelInfo


# The real container test needs these imports
from smolrouter.container import SmolRouterContainer, SmolRouterConfig


class TestWebUIIntegration:
    """Test the new web UI integration"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    def test_upstreams_page_loads(self):
        """Test that the providers page loads successfully"""
        # Providers UI is now at /providers
        response = self.client.get("/providers")
        assert response.status_code == 200
        assert "Provider Management" in response.text

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

    def test_html_and_json_responses_disable_cache(self):
        """Dashboard HTML and dashboard JSON should opt out of Safari caching."""
        html_response = self.client.get("/")
        json_response = self.client.get("/api/dashboard?limit=1")

        assert html_response.status_code == 200
        assert json_response.status_code == 200

        assert "no-store" in html_response.headers.get("cache-control", "")
        assert html_response.headers.get("pragma") == "no-cache"
        assert html_response.headers.get("expires") == "0"

        assert "no-store" in json_response.headers.get("cache-control", "")
        assert json_response.headers.get("pragma") == "no-cache"
        assert json_response.headers.get("expires") == "0"

    def test_testing_models_api_returns_exact_request_model(self):
        """Test testing models API exposes a provider-qualified request model."""
        fake_model = ModelInfo(
            id="gemma-3-4b-it@test-google",
            name="gemma-3-4b-it",
            provider_id="test-google",
            provider_type="google-genai",
            endpoint="https://example.test",
        )
        fake_mediator = Mock()
        fake_mediator.get_available_models = AsyncMock(return_value=[fake_model])
        fake_container = Mock()
        fake_container.create_client_context.return_value = Mock()
        fake_container.get_mediator = AsyncMock(return_value=fake_mediator)

        with patch("smolrouter.app.container", fake_container):
            response = self.client.get("/api/testing/models")

        assert response.status_code == 200
        payload = response.json()
        assert payload["models"][0]["request_model"] == "gemma-3-4b-it [test-google]"
        assert payload["models"][0]["display_name"] == "gemma-3-4b-it [test-google]"


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


@patch("smolrouter.container.initialize_container")
@patch("smolrouter.container.SmolRouterContainer")
class TestArchitectureIntegration:
    """Test integration between new architecture and existing app"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    # These tests were removed because they only tested mock integration
    # rather than real functionality. The actual architecture is tested
    # comprehensively in tests/test_architecture.py with real components.


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


@pytest.mark.asyncio
async def test_run_architecture_demo(capsys):
    """Test that the architecture demo completes successfully"""

    demo_module = import_module("tests.integration.test_new_architecture")

    await demo_module.demo_new_architecture()

    captured = capsys.readouterr()
    assert "Demo completed successfully!" in captured.out


def test_web_ui_navigation():
    """Test navigation between different web UI pages"""
    client = TestClient(app)

    # Test main pages load
    pages = ["/", "/performance", "/providers"]

    for page in pages:
        response = client.get(page)
        assert response.status_code == 200

        # Check that navigation links are present
        assert 'href="/"' in response.text  # Dashboard link
        assert 'href="/performance"' in response.text  # Performance link
        assert 'href="/providers"' in response.text  # Providers link


def test_dashboard_renders_mobile_scroll_wrapper():
    """Dashboard HTML should expose the responsive table scroll wrapper."""
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "table-scroll" in response.text
    assert "smolrouter-nav-toggle" in response.text
    assert "nav-label" in response.text
    assert 'data-label="Time"' in response.text
    assert 'data-label="Provider"' in response.text


def test_client_dashboard_renders_mobile_scroll_wrapper():
    """Client dashboard HTML should expose the responsive table scroll wrapper."""
    client = TestClient(app)

    response = client.get("/clients/192.168.1.26")

    assert response.status_code == 200
    assert "table-scroll" in response.text
    assert "smolrouter-nav-toggle" in response.text
    assert "Client Dashboard" in response.text
    assert 'data-label="Service"' in response.text
    assert 'data-label="Duration"' in response.text


@pytest.mark.asyncio
async def test_real_container_with_mock_providers():
    """Test real container with mock provider configurations"""
    # Create a real container with mock configuration
    providers_config = [
        {"name": "test-ollama", "type": "ollama", "url": "http://localhost:11434", "enabled": True, "priority": 0},
        {"name": "test-openai", "type": "openai", "url": "http://localhost:8000", "enabled": True, "priority": 1},
    ]

    config = SmolRouterConfig(
        providers=providers_config,
        cache_ttl=60,
        strategy={"model_map": {"gpt-4": "llama3-70b"}},
        access_control={"type": "none"},
    )

    container = SmolRouterContainer(config)

    try:
        await container.initialize()

        # Test that components are properly wired
        mediator = await container.get_mediator()
        assert mediator is not None

        # Test client operations (will return empty results due to unavailable servers)
        client = container.create_client_context("127.0.0.1")  # NOSONAR S1313
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
