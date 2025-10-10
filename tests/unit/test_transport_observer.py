"""
Tests for transport observer (ground truth verification system).
"""

import pytest
import httpx
from smolrouter.transport_observer import (
    TransportObserver,
    ObservingHTTPTransport,
    ObservingAsyncHTTPTransport,
    get_observer,
)


def test_observer_initialization():
    """Test that observer initializes correctly"""
    observer = TransportObserver()
    assert observer.observations == {}
    assert observer._request_counter == 0


def test_observer_singleton():
    """Test that get_observer returns the same instance"""
    obs1 = get_observer()
    obs2 = get_observer()
    assert obs1 is obs2


def test_sync_transport_observes_api_key():
    """Test that sync transport captures API key from headers"""
    observer = TransportObserver()
    obs_id = "test_sync_key"

    # Create a base transport and wrap it
    base_transport = httpx.HTTPTransport()
    observing_transport = ObservingHTTPTransport(obs_id, base_transport)
    observing_transport.observer = observer  # Use our test observer

    # Create a mock request with API key header
    test_key = "test_key_12345678"  # pragma: allowlist secret
    request = httpx.Request(method="GET", url="https://example.com/test", headers={"x-goog-api-key": test_key})

    # Mock the wrapped transport's handle_request
    def mock_handle_request(req):
        return httpx.Response(200, content=b'{"result": "ok"}')

    base_transport.handle_request = mock_handle_request

    # Handle the request
    observing_transport.handle_request(request)

    # Verify observation was captured
    assert obs_id in observer.observations
    observation = observer.observations[obs_id]

    assert observation.api_key_used == test_key
    assert observation.api_key_header_name == "x-goog-api-key"  # pragma: allowlist secret
    assert observation.method == "GET"
    assert observation.url == "https://example.com/test"
    assert observation.status_code == 200
    assert observation.response_received is True


@pytest.mark.asyncio
async def test_async_transport_observes_api_key():
    """Test that async transport captures API key from headers"""
    observer = TransportObserver()
    obs_id = "test_async_key"

    # Create a base transport and wrap it
    base_transport = httpx.AsyncHTTPTransport()
    observing_transport = ObservingAsyncHTTPTransport(obs_id, base_transport)
    observing_transport.observer = observer  # Use our test observer

    # Create a mock request with API key header
    request = httpx.Request(
        method="POST", url="https://api.example.com/v1/generate", headers={"x-goog-api-key": "async_key_87654321"}
    )

    # Mock the wrapped transport's handle_async_request
    async def mock_handle_async_request(req):
        return httpx.Response(200, content=b'{"result": "success"}')

    base_transport.handle_async_request = mock_handle_async_request

    # Handle the request
    await observing_transport.handle_async_request(request)

    # Verify observation was captured
    assert obs_id in observer.observations
    observation = observer.observations[obs_id]

    assert observation.api_key_used == "async_key_87654321"  # pragma: allowlist secret
    assert observation.api_key_header_name == "x-goog-api-key"  # pragma: allowlist secret
    assert observation.method == "POST"
    assert observation.url == "https://api.example.com/v1/generate"
    assert observation.status_code == 200
    assert observation.response_received is True


def test_observer_verify_api_key_match():
    """Test API key verification when keys match"""
    observer = TransportObserver()
    obs_id = "test_verify_match"

    # Create a base transport and wrap it
    base_transport = httpx.HTTPTransport()
    observing_transport = ObservingHTTPTransport(obs_id, base_transport)
    observing_transport.observer = observer

    # Create request with known API key
    request = httpx.Request(
        method="GET", url="https://example.com/test", headers={"x-goog-api-key": "my_secret_key_abc12345"}
    )

    def mock_handle_request(req):
        return httpx.Response(200)

    base_transport.handle_request = mock_handle_request
    observing_transport.handle_request(request)

    # Verify with correct suffix
    result = observer.verify_api_key(obs_id, "abc12345")
    assert result is True


def test_observer_verify_api_key_mismatch():
    """Test API key verification when keys don't match"""
    observer = TransportObserver()
    obs_id = "test_verify_mismatch"

    # Create a base transport and wrap it
    base_transport = httpx.HTTPTransport()
    observing_transport = ObservingHTTPTransport(obs_id, base_transport)
    observing_transport.observer = observer

    # Create request with known API key
    request = httpx.Request(
        method="GET", url="https://example.com/test", headers={"x-goog-api-key": "my_secret_key_abc12345"}
    )

    def mock_handle_request(req):
        return httpx.Response(200)

    base_transport.handle_request = mock_handle_request
    observing_transport.handle_request(request)

    # Verify with WRONG suffix
    result = observer.verify_api_key(obs_id, "xyz99999")
    assert result is False


def test_observer_bearer_token_extraction():
    """Test that Bearer tokens are extracted correctly"""
    observer = TransportObserver()
    obs_id = "test_bearer"

    base_transport = httpx.HTTPTransport()
    observing_transport = ObservingHTTPTransport(obs_id, base_transport)
    observing_transport.observer = observer

    # Create request with Bearer token
    request = httpx.Request(
        method="GET", url="https://example.com/test", headers={"authorization": "Bearer sk-test123456789"}
    )

    def mock_handle_request(req):
        return httpx.Response(200)

    base_transport.handle_request = mock_handle_request
    observing_transport.handle_request(request)

    # Verify Bearer prefix was stripped
    observation = observer.observations[obs_id]
    assert observation.api_key_used == "sk-test123456789"  # pragma: allowlist secret
    assert observation.api_key_header_name == "authorization"  # pragma: allowlist secret


def test_observer_cleanup_old_observations():
    """Test that old observations are cleaned up"""
    from datetime import datetime, timedelta

    observer = TransportObserver()

    # Add some observations with different ages
    now = datetime.now()
    observer.observations["old1"] = type("obj", (object,), {"timestamp": now - timedelta(hours=2)})()
    observer.observations["old2"] = type("obj", (object,), {"timestamp": now - timedelta(hours=3)})()
    observer.observations["recent"] = type("obj", (object,), {"timestamp": now})()

    # Cleanup observations older than 1 hour
    observer.cleanup_old_observations(max_age_seconds=3600)

    # Only recent should remain
    assert "old1" not in observer.observations
    assert "old2" not in observer.observations
    assert "recent" in observer.observations


def test_observer_proxy_verification_no_proxy():
    """Test proxy verification when no proxy is used"""
    observer = TransportObserver()
    obs_id = "test_no_proxy"

    base_transport = httpx.HTTPTransport()
    observing_transport = ObservingHTTPTransport(obs_id, base_transport)
    observing_transport.observer = observer

    request = httpx.Request(method="GET", url="https://example.com/test")

    def mock_handle_request(req):
        return httpx.Response(200)

    base_transport.handle_request = mock_handle_request
    observing_transport.handle_request(request)

    # Verify no proxy (both expected and actual are None)
    result = observer.verify_proxy(obs_id, None)
    assert result is True


def test_observation_missing():
    """Test behavior when observation doesn't exist"""
    observer = TransportObserver()

    # Try to verify non-existent observation
    result = observer.verify_api_key("nonexistent", "somekey")
    assert result is None

    result = observer.verify_proxy("nonexistent", None)
    assert result is None

    observation = observer.get_observation("nonexistent")
    assert observation is None
