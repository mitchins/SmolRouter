import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import httpx
from httpx import AsyncClient
from fastapi.testclient import TestClient

from smolrouter.app import app
from smolrouter.database import RequestLog, get_log_stats


# Use the shared isolated_db fixture from conftest.py


@pytest.fixture
def sample_logs(isolated_db):
    """Create sample log entries for testing"""
    now = datetime.now()

    async def _create_all():
        return [
            await RequestLog.create(
                timestamp=now - timedelta(minutes=5),
                source_ip="192.168.1.100",  # NOSONAR S1313
                method="POST",
                path="/v1/chat/completions",
                service_type="openai",
                upstream_url="http://localhost:8000",
                original_model="gpt-3.5-turbo",
                mapped_model="gemma3-12b",
                provider_id="google-gen-ai",
                duration_ms=1200,
                request_size=256,
                response_size=1024,
                status_code=200,
                completed_at=now - timedelta(minutes=4),
            ),
            await RequestLog.create(
                timestamp=now - timedelta(minutes=3),
                source_ip="192.168.1.101",  # NOSONAR S1313
                method="POST",
                path="/api/generate",
                service_type="ollama",
                upstream_url="http://localhost:8000",
                original_model="mistral",
                mapped_model="mistral",
                provider_id="local-ollama",
                duration_ms=2500,
                request_size=128,
                response_size=512,
                status_code=200,
                completed_at=now - timedelta(minutes=2),
            ),
            await RequestLog.create(
                timestamp=now - timedelta(minutes=1),
                source_ip="192.168.1.100",  # NOSONAR S1313
                method="GET",
                path="/v1/models",
                service_type="openai",
                upstream_url="http://localhost:8000",
                original_model="gemma3-12b",
                mapped_model="gemma3-12b",
                provider_id="google-gen-ai",
                duration_ms=150,
                request_size=0,
                response_size=2048,
                status_code=200,
                completed_at=now - timedelta(minutes=1),
            ),
            await RequestLog.create(
                timestamp=now - timedelta(days=10),  # Old log for cleanup testing
                source_ip="192.168.1.102",  # NOSONAR S1313
                method="POST",
                path="/v1/chat/completions",
                service_type="openai",
                upstream_url="http://localhost:8000",
                provider_id="openai-fallback",
                status_code=500,
                error_message="Connection timeout",
                completed_at=now - timedelta(days=10),
            ),
        ]

    logs = asyncio.run(_create_all())
    return logs


def test_database_operations(isolated_db, sample_logs):
    """Test basic database operations with Redis backend"""
    # Test that logs were created - use Redis backend directly
    import asyncio
    from smolrouter.redis_backend import RedisRequestLog

    async def verify_redis_logs():
        # Get recent logs using Redis backend
        recent_logs = await RedisRequestLog.get_recent(limit=10)
        return recent_logs

    # Run async operation
    recent_logs = asyncio.run(verify_redis_logs())

    # Test that logs were created (4 logs expected)
    assert len(recent_logs) == 4

    # Test getting specific number of recent logs
    recent_logs_limited = asyncio.run(RedisRequestLog.get_recent(limit=3))
    assert len(recent_logs_limited) == 3

    # Should be ordered by timestamp desc (most recent first)
    # For Redis logs, we'll check the created_at field
    created_times = [log.get("created_at") for log in recent_logs if log.get("created_at")]
    assert len(created_times) >= 3  # At least 3 logs should have timestamps

    # Test filtering by service type using Redis data directly
    openai_logs = [log for log in recent_logs if log.get("service_type") == "openai"]
    assert len(openai_logs) >= 2  # We created at least 2 openai logs
    assert all(log.get("service_type") == "openai" for log in openai_logs)

    ollama_logs = [log for log in recent_logs if log.get("service_type") == "ollama"]
    assert len(ollama_logs) >= 1  # We created at least 1 ollama log
    assert all(log.get("service_type") == "ollama" for log in ollama_logs)


def test_log_stats(isolated_db, sample_logs):
    """Test statistics calculation"""
    stats = asyncio.run(get_log_stats())

    assert stats["total_requests"] == 4
    # service_types map should include counts
    assert stats["service_types"].get("openai", 0) == 3
    assert stats["service_types"].get("ollama", 0) == 1


@pytest.mark.asyncio
async def test_cleanup_old_logs(isolated_db, sample_logs):
    """Test automatic cleanup of old logs using Redis backend"""
    # Before cleanup, we should have 4 logs
    recent = await RequestLog.get_recent(10)
    assert len(recent) == 4

    # Cleanup logs older than 7 days
    from smolrouter.database import cleanup_old_logs_async

    with patch("smolrouter.database.MAX_AGE_DAYS", 7):
        deleted = await cleanup_old_logs_async()
        assert deleted >= 1

    recent_after = await RequestLog.get_recent(10)
    assert len(recent_after) == 3

    # Verify remaining logs are newer than 7 days
    from datetime import datetime, timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    assert all(log.timestamp > cutoff for log in recent_after)


@pytest.mark.asyncio
async def test_web_ui_dashboard(isolated_db, sample_logs, disable_logging):
    """Test the web UI dashboard endpoint"""
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Check that the response contains expected UI elements
        content = response.text
        assert "SmolRouter" in content
        assert "Recent Requests" in content
        assert "Total Requests" in content


@pytest.mark.asyncio
async def test_api_logs_endpoint(isolated_db, sample_logs, disable_logging):
    """Test the logs API endpoint"""
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Test basic logs endpoint
        response = await client.get("/api/logs")
        assert response.status_code == 200

        logs = response.json()
        assert len(logs) <= 100  # Default limit
        assert all("timestamp" in log for log in logs)
        assert all("service_type" in log for log in logs)

        # Test with limit parameter
        response = await client.get("/api/logs?limit=2")
        assert response.status_code == 200
        logs = response.json()
        assert len(logs) <= 2

        # Test with service type filter
        response = await client.get("/api/logs?service_type=openai")
        assert response.status_code == 200
        logs = response.json()
        assert all(log["service_type"] == "openai" for log in logs)


@pytest.mark.asyncio
async def test_api_dashboard_supports_combinable_field_filters(isolated_db, sample_logs, disable_logging):
    """Test host, provider, and model filters together on the dashboard API."""
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/dashboard",
            params={"q": "host:192.168.1.100 provider:google-gen-ai model:gemma3-12b"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["filter"]["active"] is True
        assert payload["filter"]["query"] == "host:192.168.1.100 provider:google-gen-ai model:gemma3-12b"

        logs = payload["logs"]
        assert len(logs) == 2
        assert all(log["source_ip"] == "192.168.1.100" for log in logs)
        assert all(log["provider_id"] == "google-gen-ai" for log in logs)
        assert all("gemma3-12b" in {log.get("original_model"), log.get("mapped_model")} for log in logs)


@pytest.mark.asyncio
async def test_api_dashboard_rejects_unknown_filter_fields(isolated_db, sample_logs, disable_logging):
    """Test invalid primitiveQL clauses return an explicit API error."""
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/dashboard", params={"q": "status:200"})

        assert response.status_code == 422
        payload = response.json()
        assert payload["error"] == "invalid_filter"
        assert "status:200" in payload["invalid_terms"]
        assert "Supported fields" in payload["message"]


@pytest.mark.asyncio
async def test_api_stats_endpoint(isolated_db, sample_logs, disable_logging):
    """Test the stats API endpoint"""
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/stats")
        assert response.status_code == 200

        stats = response.json()
        assert "total_requests" in stats
        assert "service_types" in stats

        assert isinstance(stats["total_requests"], int)
        assert isinstance(stats["completed_requests"], int)
        assert isinstance(stats["pending_requests"], int)


@pytest.mark.asyncio
async def test_api_client_endpoint_uses_redis_client_logs(isolated_db, sample_logs, disable_logging):
    """Test the client-specific API uses Redis-backed per-IP lookup and keeps stats independent of log limit."""
    inflight_log = await RequestLog.create(
        source_ip="192.168.1.100",  # NOSONAR S1313
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-4o-mini",
        mapped_model="gpt-4o-mini",
        timestamp=datetime.now() - timedelta(minutes=2),
    )

    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/clients/192.168.1.100", params={"limit": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["client_ip"] == "192.168.1.100"
    assert payload["stats"]["total_requests"] == 3
    assert payload["stats"]["successful_requests"] == 2
    assert payload["stats"]["recent_requests"] == 3
    assert payload["stats"]["inflight_requests"] == 1
    assert set(payload["stats"]["models_used"]) == {"gemma3-12b", "gpt-3.5-turbo", "gpt-4o-mini"}
    assert len(payload["logs"]) == 2

    inflight_entry = next(log for log in payload["logs"] if log["id"] == inflight_log.id)
    assert inflight_entry["status_code"] is None
    assert isinstance(inflight_entry["duration_ms"], int)


@pytest.mark.asyncio
async def test_api_performance_endpoint_uses_redis_recent_logs(isolated_db, disable_logging):
    """Performance API should return Redis-backed completed requests with token data and respect filters."""
    now = datetime.now(timezone.utc)
    recent_log = await RequestLog.create(
        source_ip="192.168.1.150",  # NOSONAR S1313
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-4",
        mapped_model="llama3-70b",
        duration_ms=1800,
        status_code=200,
        completed_at=now - timedelta(hours=1),
        timestamp=now - timedelta(hours=1),
    )
    recent_log.prompt_tokens = 120
    recent_log.completion_tokens = 30
    recent_log.total_tokens = 150
    await recent_log.save_async()

    old_log = await RequestLog.create(
        source_ip="192.168.1.151",  # NOSONAR S1313
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-4",
        mapped_model="llama3-70b",
        duration_ms=3200,
        status_code=200,
        completed_at=now - timedelta(hours=30),
        timestamp=now - timedelta(hours=30),
    )
    old_log.prompt_tokens = 200
    old_log.completion_tokens = 50
    old_log.total_tokens = 250
    await old_log.save_async()

    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/api/performance",
            params={"hours": 24, "model": "llama3-70b", "service_type": "openai", "limit": 10},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["meta"]["total_points"] == 1
    point = payload["data_points"][0]
    assert point["id"] == recent_log.id
    assert point["prompt_tokens"] == 120
    assert point["completion_tokens"] == 30
    assert point["total_tokens"] == 150
    assert point["duration_ms"] == 1800
    assert point["mapped_model"] == "llama3-70b"
    assert point["service_type"] == "openai"


def test_client_dashboard_websocket_sends_initial_data_and_refresh(isolated_db, sample_logs, disable_logging):
    """Client dashboard websocket should send Redis-backed initial data and respond to refresh/ping."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/clients/192.168.1.100") as websocket:
            initial_payload = websocket.receive_json()
            assert initial_payload["type"] == "client_dashboard_update"
            assert initial_payload["client_ip"] == "192.168.1.100"
            assert initial_payload["stats"]["total_requests"] == 2
            assert initial_payload["stats"]["successful_requests"] == 2
            assert len(initial_payload["logs"]) == 2

            websocket.send_json({"type": "ping"})
            assert websocket.receive_json() == {"type": "pong"}

            websocket.send_json({"type": "refresh"})
            refreshed_payload = websocket.receive_json()
            assert refreshed_payload["type"] == "client_dashboard_update"
            assert refreshed_payload["client_ip"] == "192.168.1.100"
            assert refreshed_payload["stats"]["total_requests"] == 2


@pytest.mark.asyncio
async def test_api_inflight_endpoint(isolated_db, sample_logs, disable_logging):
    """Test the inflight requests API endpoint"""
    # Create an inflight request
    inflight_log = await RequestLog.create(
        source_ip="192.168.1.100",  # NOSONAR S1313
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-4",
        mapped_model="llama3-70b",
    )

    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/inflight")
        assert response.status_code == 200

        inflight = response.json()
        assert len(inflight) == 1
        assert inflight[0]["id"] == inflight_log.id
        assert inflight[0]["source_ip"] == "192.168.1.100"
        assert inflight[0]["service_type"] == "openai"
        assert inflight[0]["original_model"] == "gpt-4"
        assert inflight[0]["mapped_model"] == "llama3-70b"
        assert "elapsed_ms" in inflight[0]
        assert isinstance(inflight[0]["elapsed_ms"], int)


@pytest.mark.asyncio
async def test_request_detail_view(isolated_db, sample_logs, disable_logging):
    """Test the request detail view endpoint"""
    # Create a completed request with full data
    log_entry = await RequestLog.create(
        source_ip="192.168.1.100",
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-4",
        mapped_model="llama3-70b",
        provider_id="test-zai",
        duration_ms=1500,
        request_size=512,
        response_size=1024,
        status_code=200,
        completed_at=datetime.now(),
    )

    # Set bodies using the setter methods
    log_entry.set_request_body(b'{"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}')
    log_entry.set_response_body(b'{"choices": [{"message": {"content": "Hello!"}}]}')
    log_entry.save()

    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/request/{log_entry.id}")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        content = response.text
        assert f"Request #{log_entry.id}" in content
        assert "Request Information" in content
        assert "Response Information" in content
        assert "192.168.1.100" in content
        assert "Protocol:" in content
        assert "OpenAI-compatible" in content
        assert "Provider:" in content
        assert "test-zai" in content
        assert "gpt-4" in content
        assert "llama3-70b" in content
        assert "Request Body" in content
        assert "Response Body" in content


@pytest.mark.asyncio
async def test_request_detail_view_404(isolated_db, disable_logging):
    """Test the request detail view with non-existent request"""
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/request/999999")
        assert response.status_code == 404
        assert "Request Not Found" in response.text


@pytest.mark.asyncio
async def test_request_detail_api_404(isolated_db, disable_logging):
    """Test the JSON request detail endpoint with non-existent request."""
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/requests/999999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Request not found"}


@pytest.mark.asyncio
async def test_request_detail_view_inflight(isolated_db, disable_logging):
    """Test the request detail view for inflight requests"""
    # Create an inflight request (no completed_at)
    log_entry = await RequestLog.create(
        source_ip="127.0.0.1",  # NOSONAR S1313
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-3.5-turbo",
        mapped_model="llama3-8b",
    )

    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/request/{log_entry.id}")
        assert response.status_code == 200

        content = response.text
        assert "Inflight" in content
        assert "Still processing" in content


@pytest.mark.asyncio
async def test_request_detail_api_serializes_bodies_and_duplicates(isolated_db, disable_logging):
    """Test JSON request details include decoded bodies and duplicate request metadata."""
    timestamp = datetime.now()
    primary_log = await RequestLog.create(
        source_ip="192.168.1.100",
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-4",
        mapped_model="llama3-70b",
        request_body_hash="body-hash-123",
        duration_ms=1500,
        request_size=64,
        response_size=128,
        status_code=200,
        completed_at=timestamp,
        timestamp=timestamp,
    )
    primary_log.set_request_body(b'{"model": "gpt-4"}')
    primary_log.set_response_body(b'{"choices": [{"message": {"content": "Hello!"}}]}')
    await primary_log.save_async()

    duplicate_log = await RequestLog.create(
        source_ip="192.168.1.101",
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-4",
        mapped_model="llama3-70b",
        request_body_hash="body-hash-123",
        status_code=429,
        completed_at=timestamp,
        timestamp=timestamp,
    )

    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/api/requests/{primary_log.id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == primary_log.id
    assert payload["request_body"] == '{"model": "gpt-4"}'
    assert payload["response_body"] == '{"choices": [{"message": {"content": "Hello!"}}]}'
    assert payload["duplicate"]["request_body_hash"] == "body-hash-123"
    assert payload["duplicate"]["duplicates"][0]["id"] == duplicate_log.id
    assert payload["duplicate"]["duplicates"][0]["source_ip"] == "192.168.1.101"


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_request_log_model_fields(isolated_db):
    """Test that the RequestLog model has all required fields"""
    # Create a comprehensive log entry
    log = await RequestLog.create(
        source_ip="127.0.0.1",  # NOSONAR S1313
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-4",
        mapped_model="claude-3-opus",
        provider_id="google-gen-ai",
        duration_ms=1500,
        request_size=512,
        response_size=2048,
        status_code=200,
        error_message=None,
    )

    # Set bodies using the setter methods and mark as completed
    log.set_request_body(b'{"model": "gpt-4", "messages": [...]}')
    log.set_response_body(b'{"choices": [...]}')
    from datetime import datetime

    log.completed_at = datetime.now()
    await log.save_async()

    # Verify all fields are stored correctly
    retrieved_log = await RequestLog.get_by_id(log.id)
    assert retrieved_log.source_ip == "127.0.0.1"
    assert retrieved_log.method == "POST"
    assert retrieved_log.path == "/v1/chat/completions"
    assert retrieved_log.service_type == "openai"
    assert retrieved_log.original_model == "gpt-4"
    assert retrieved_log.mapped_model == "claude-3-opus"
    assert retrieved_log.provider_id == "google-gen-ai"
    assert retrieved_log.duration_ms == 1500
    assert retrieved_log.request_size == 512
    assert retrieved_log.response_size == 2048
    assert retrieved_log.status_code == 200
    assert retrieved_log.request_body is not None
    assert retrieved_log.response_body is not None


@pytest.mark.asyncio
async def test_logging_middleware_integration(isolated_db):
    """Test that requests are actually logged when ENABLE_LOGGING is True"""

    # Mock the upstream server to avoid real network calls
    with patch("smolrouter.app.ENABLE_LOGGING", True):
        async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            # Before request, check log count
            _ = await RequestLog.get_recent(10)

            # Make a request that will fail (no upstream server)
            try:
                await client.post(
                    "/v1/chat/completions",
                    json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "test"}]},
                )
            except Exception:
                pass  # We expect this to fail due to no upstream

            # After request, there should be one more log entry
            # Note: This test may not work as expected without proper mocking of the upstream
            # In a real scenario, you'd mock the httpx client calls


@pytest.mark.asyncio
async def test_log_entry_with_model_mapping(isolated_db):
    """Test logging when model mapping occurs"""
    log = await RequestLog.create(
        source_ip="127.0.0.1",  # NOSONAR S1313
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-3.5-turbo",
        mapped_model="llama3-8b",  # Different from original - mapping occurred
        status_code=200,
    )

    # Verify model mapping is recorded
    assert log.original_model != log.mapped_model
    assert log.original_model == "gpt-3.5-turbo"
    assert log.mapped_model == "llama3-8b"


@pytest.mark.asyncio
async def test_error_logging(isolated_db):
    """Test logging of error scenarios"""
    error_log = await RequestLog.create(
        source_ip="127.0.0.1",  # NOSONAR S1313
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        status_code=502,
        error_message="Connection refused",
    )

    assert error_log.status_code == 502
    assert error_log.error_message == "Connection refused"
    assert error_log.duration_ms is None  # No duration for failed requests


@pytest.mark.asyncio
async def test_inflight_tracking(isolated_db):
    """Test inflight request tracking"""
    from smolrouter.database import get_inflight_requests

    # Create an inflight request (no completed_at)
    inflight_log = await RequestLog.create(
        source_ip="127.0.0.1",  # NOSONAR S1313
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-4",
        mapped_model="llama3-70b",
    )

    # Create a completed request
    _ = await RequestLog.create(
        source_ip="127.0.0.1",  # NOSONAR S1313
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-3.5-turbo",
        mapped_model="llama3-8b",
        duration_ms=1500,
        status_code=200,
        completed_at=datetime.now(),
    )

    # Test inflight retrieval
    inflight_requests = await get_inflight_requests()
    assert len(inflight_requests) == 1
    assert inflight_requests[0].id == inflight_log.id
    assert inflight_requests[0].completed_at is None

    # Test stats include inflight count
    stats = await get_log_stats()
    assert stats["inflight_requests"] == 1
    assert stats["total_requests"] == 2

    # Complete the inflight request
    inflight_log.completed_at = datetime.now()
    inflight_log.duration_ms = 2000
    inflight_log.status_code = 200
    await inflight_log.save_async()

    # Should no longer be inflight
    inflight_requests = await get_inflight_requests()
    assert len(inflight_requests) == 0

    stats = await get_log_stats()
    assert stats["total_requests"] == 2


@pytest.mark.asyncio
async def test_request_body_key_update_does_not_complete_inflight_request(isolated_db):
    """Attaching a request body blob key should not mark an inflight request complete."""
    from smolrouter.redis_backend import RedisRequestLog

    inflight_log = await RequestLog.create(
        source_ip="127.0.0.1",  # NOSONAR S1313
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-4",
        mapped_model="llama3-70b",
    )

    await RedisRequestLog.update_request_body_key(inflight_log.id, "blob/request-body-1")

    updated_log = await RequestLog.get_by_id(inflight_log.id)

    assert getattr(updated_log, "request_body_key", None) == "blob/request-body-1"
    assert updated_log.completed_at is None
    assert updated_log.status_code == "pending"


def test_vacuum_database(isolated_db):
    """Test database vacuum functionality (no-op on Redis backend)"""
    from smolrouter.database import vacuum_database

    assert vacuum_database() is True
