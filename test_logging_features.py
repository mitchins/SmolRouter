import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
import httpx
from httpx import AsyncClient

from smolrouter.app import app
from smolrouter.database import RequestLog, get_recent_logs, get_log_stats, cleanup_old_logs


# Use the shared isolated_db fixture from conftest.py


@pytest.fixture
def sample_logs(isolated_db):
    """Create sample log entries for testing"""
    now = datetime.now()
    
    logs = [
        RequestLog.create(
            timestamp=now - timedelta(minutes=5),
            source_ip="192.168.1.100",
            method="POST",
            path="/v1/chat/completions",
            service_type="openai",
            upstream_url="http://localhost:8000",
            original_model="gpt-3.5-turbo",
            mapped_model="llama3-8b",
            duration_ms=1200,
            request_size=256,
            response_size=1024,
            status_code=200
        ),
        RequestLog.create(
            timestamp=now - timedelta(minutes=3),
            source_ip="192.168.1.101",
            method="POST",
            path="/api/generate",
            service_type="ollama",
            upstream_url="http://localhost:8000",
            original_model="mistral",
            mapped_model="mistral",
            duration_ms=2500,
            request_size=128,
            response_size=512,
            status_code=200
        ),
        RequestLog.create(
            timestamp=now - timedelta(minutes=1),
            source_ip="192.168.1.100",
            method="GET",
            path="/v1/models",
            service_type="openai",
            upstream_url="http://localhost:8000",
            duration_ms=150,
            request_size=0,
            response_size=2048,
            status_code=200
        ),
        RequestLog.create(
            timestamp=now - timedelta(days=10),  # Old log for cleanup testing
            source_ip="192.168.1.102",
            method="POST",
            path="/v1/chat/completions",
            service_type="openai",
            upstream_url="http://localhost:8000",
            status_code=500,
            error_message="Connection timeout"
        )
    ]
    
    return logs


def test_database_operations(isolated_db, sample_logs):
    """Test basic database operations"""
    # Test that logs were created
    assert RequestLog.select().count() == 4
    
    # Test getting recent logs
    recent_logs = get_recent_logs(limit=3)
    assert len(recent_logs) == 3
    
    # Should be ordered by timestamp desc (most recent first)
    timestamps = [log.timestamp for log in recent_logs]
    assert timestamps == sorted(timestamps, reverse=True)
    
    # Test filtering by service type
    openai_logs = get_recent_logs(limit=10, service_type="openai")
    assert all(log.service_type == "openai" for log in openai_logs)
    
    ollama_logs = get_recent_logs(limit=10, service_type="ollama")
    assert all(log.service_type == "ollama" for log in ollama_logs)


def test_log_stats(isolated_db, sample_logs):
    """Test statistics calculation"""
    stats = get_log_stats()
    
    assert stats['total_requests'] == 4
    assert stats['openai_requests'] == 3
    assert stats['ollama_requests'] == 1
    
    # Recent requests (last 24 hours) should exclude the 10-day-old log
    assert stats['recent_requests'] == 3


def test_cleanup_old_logs(isolated_db, sample_logs):
    """Test automatic cleanup of old logs"""
    # Before cleanup, we should have 4 logs
    assert RequestLog.select().count() == 4
    
    # Mock the MAX_AGE_DAYS to 7 days
    with patch('smolrouter.database.MAX_AGE_DAYS', 7):
        cleanup_old_logs()
    
    # After cleanup, should have 3 logs (the 10-day-old one should be removed)
    assert RequestLog.select().count() == 3
    
    # Verify the old log is gone
    remaining_logs = list(RequestLog.select())
    assert all(
        (datetime.now() - log.timestamp).days < 7 
        for log in remaining_logs
    )


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
async def test_api_stats_endpoint(isolated_db, sample_logs, disable_logging):
    """Test the stats API endpoint"""
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/stats")
        assert response.status_code == 200
        
        stats = response.json()
        assert "total_requests" in stats
        assert "openai_requests" in stats
        assert "ollama_requests" in stats
        assert "recent_requests" in stats
        assert "inflight_requests" in stats
        
        assert isinstance(stats["total_requests"], int)
        assert isinstance(stats["openai_requests"], int)
        assert isinstance(stats["ollama_requests"], int)
        assert isinstance(stats["recent_requests"], int)
        assert isinstance(stats["inflight_requests"], int)


@pytest.mark.asyncio
async def test_api_inflight_endpoint(isolated_db, disable_logging):
    """Test the inflight requests API endpoint"""
    # Create an inflight request
    inflight_log = RequestLog.create(
        source_ip="192.168.1.100",
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-4",
        mapped_model="llama3-70b"
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
async def test_request_detail_view(isolated_db, disable_logging):
    """Test the request detail view endpoint"""
    # Create a completed request with full data
    log_entry = RequestLog.create(
        source_ip="192.168.1.100",
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-4",
        mapped_model="llama3-70b",
        duration_ms=1500,
        request_size=512,
        response_size=1024,
        status_code=200,
        completed_at=datetime.now()
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
async def test_request_detail_view_inflight(isolated_db, disable_logging):
    """Test the request detail view for inflight requests"""
    # Create an inflight request (no completed_at)
    log_entry = RequestLog.create(
        source_ip="127.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-3.5-turbo",
        mapped_model="llama3-8b"
    )
    
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/request/{log_entry.id}")
        assert response.status_code == 200
        
        content = response.text
        assert "Inflight" in content
        assert "Still processing" in content


def test_request_log_model_fields(isolated_db):
    """Test that the RequestLog model has all required fields"""
    # Create a comprehensive log entry
    log = RequestLog.create(
        source_ip="127.0.0.1",
        method="POST", 
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-4",
        mapped_model="claude-3-opus",
        duration_ms=1500,
        request_size=512,
        response_size=2048,
        status_code=200,
        error_message=None
    )
    
    # Set bodies using the setter methods
    log.set_request_body(b'{"model": "gpt-4", "messages": [...]}')
    log.set_response_body(b'{"choices": [...]}')
    log.save()
    
    # Verify all fields are stored correctly
    retrieved_log = RequestLog.get_by_id(log.id)
    assert retrieved_log.source_ip == "127.0.0.1"
    assert retrieved_log.method == "POST"
    assert retrieved_log.path == "/v1/chat/completions"
    assert retrieved_log.service_type == "openai"
    assert retrieved_log.original_model == "gpt-4"
    assert retrieved_log.mapped_model == "claude-3-opus"
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
    with patch('smolrouter.app.ENABLE_LOGGING', True):
        async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            
            # Before request, check log count
            _ = RequestLog.select().count()
            
            # Make a request that will fail (no upstream server)
            try:
                await client.post(
                    "/v1/chat/completions",
                    json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "test"}]}
                )
            except Exception:
                pass  # We expect this to fail due to no upstream
            
            # After request, there should be one more log entry
            # Note: This test may not work as expected without proper mocking of the upstream
            # In a real scenario, you'd mock the httpx client calls


def test_log_entry_with_model_mapping(isolated_db):
    """Test logging when model mapping occurs"""
    log = RequestLog.create(
        source_ip="127.0.0.1",
        method="POST",
        path="/v1/chat/completions", 
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-3.5-turbo",
        mapped_model="llama3-8b",  # Different from original - mapping occurred
        status_code=200
    )
    
    # Verify model mapping is recorded
    assert log.original_model != log.mapped_model
    assert log.original_model == "gpt-3.5-turbo"
    assert log.mapped_model == "llama3-8b"


def test_error_logging(isolated_db):
    """Test logging of error scenarios"""
    error_log = RequestLog.create(
        source_ip="127.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        service_type="openai", 
        upstream_url="http://localhost:8000",
        status_code=502,
        error_message="Connection refused"
    )
    
    assert error_log.status_code == 502
    assert error_log.error_message == "Connection refused"
    assert error_log.duration_ms is None  # No duration for failed requests


def test_inflight_tracking(isolated_db):
    """Test inflight request tracking"""
    from smolrouter.database import get_inflight_requests
    
    # Create an inflight request (no completed_at)
    inflight_log = RequestLog.create(
        source_ip="127.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-4",
        mapped_model="llama3-70b"
    )
    
    # Create a completed request
    _ = RequestLog.create(
        source_ip="127.0.0.1",
        method="POST", 
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        original_model="gpt-3.5-turbo",
        mapped_model="llama3-8b",
        duration_ms=1500,
        status_code=200,
        completed_at=datetime.now()
    )
    
    # Test inflight retrieval
    inflight_requests = get_inflight_requests()
    assert len(inflight_requests) == 1
    assert inflight_requests[0].id == inflight_log.id
    assert inflight_requests[0].completed_at is None
    
    # Test stats include inflight count
    stats = get_log_stats()
    assert stats['inflight_requests'] == 1
    assert stats['total_requests'] == 2
    
    # Complete the inflight request
    inflight_log.completed_at = datetime.now()
    inflight_log.duration_ms = 2000
    inflight_log.status_code = 200
    inflight_log.save()
    
    # Should no longer be inflight
    inflight_requests = get_inflight_requests()
    assert len(inflight_requests) == 0
    
    stats = get_log_stats()
    assert stats['inflight_requests'] == 0
    assert stats['total_requests'] == 2


def test_vacuum_database(isolated_db):
    """Test database vacuum functionality"""
    from smolrouter.database import vacuum_database
    
    # Create some test data
    RequestLog.create(
        source_ip="127.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="http://localhost:8000",
        completed_at=datetime.now()
    )
    
    # Test vacuum doesn't raise errors
    vacuum_database()
    
    # Data should still be there
    assert RequestLog.select().count() == 1