"""
Tests for quota increment functionality - ensuring no stubs/no-ops break tracking.
"""

import pytest
from smolrouter.redis_backend import RedisApiKeyQuota


@pytest.mark.asyncio
async def test_increment_usage_actually_increments():
    """Test that increment_usage actually updates Redis, not a stub/no-op"""
    api_key = "test_key_quota_increment_123"  # pragma: allowlist secret
    provider_id = "google"
    model_name = "gemini-2.5-flash"

    # First increment
    result1 = await RedisApiKeyQuota.increment_usage(
        api_key=api_key,
        provider_id=provider_id,
        model_name=model_name,
        request_count=1,
        token_count=500,
    )

    assert result1["requests_today"] == 1, "First request should increment to 1"
    assert result1["tokens_today"] == 500, "First request should have 500 tokens"

    # Second increment - CRITICAL: must accumulate, not stay at 0
    result2 = await RedisApiKeyQuota.increment_usage(
        api_key=api_key,
        provider_id=provider_id,
        model_name=model_name,
        request_count=1,
        token_count=300,
    )

    assert result2["requests_today"] == 2, "Second request should increment to 2, not stay at 0 or 1"
    assert result2["tokens_today"] == 800, "Tokens should accumulate (500 + 300 = 800)"


@pytest.mark.asyncio
async def test_multiple_increments_accumulate():
    """Test that multiple increments for same key accumulate correctly"""
    api_key = "key_accumulate_test"  # pragma: allowlist secret
    provider_id = "google"
    model_name = "gemini-2.5-flash"

    # Multiple increments
    await RedisApiKeyQuota.increment_usage(api_key, provider_id, model_name, 1, 100)
    await RedisApiKeyQuota.increment_usage(api_key, provider_id, model_name, 1, 200)
    await RedisApiKeyQuota.increment_usage(api_key, provider_id, model_name, 1, 300)

    # Fetch the quota record
    quota, _created = await RedisApiKeyQuota.get_or_create_quota(api_key, provider_id, model_name)

    # Verify accumulation (not stubbed to 0)
    assert quota.requests_today == 3, "Should accumulate 3 requests, not stay at 0"
    assert quota.tokens_today == 600, "Should accumulate 600 tokens (100+200+300)"


@pytest.mark.asyncio
async def test_mark_request_success_updates_local_object():
    """Test that mark_request_success is not a no-op - it updates the object"""
    from smolrouter.redis_backend import QuotaRecord

    # Create a quota record
    quota = QuotaRecord(
        {
            "provider_id": "google",
            "key_hash": "test_hash",
            "model_name": "gemini-2.5-flash",
            "requests_today": 5,
            "tokens_today": 1000,
            "last_reset": "2025-10-02",
        }
    )

    # Call mark_request_success
    quota.mark_request_success(tokens=250)

    # Verify it's NOT a no-op - the object state should change
    assert quota.requests_today == 6, "requests_today should increment from 5 to 6"
    assert quota.tokens_today == 1250, "tokens_today should increment from 1000 to 1250"


@pytest.mark.asyncio
async def test_mark_request_failure_updates_local_object():
    """Test that mark_request_failure is not a no-op - it updates the object"""
    from smolrouter.redis_backend import QuotaRecord

    # Create a quota record
    quota = QuotaRecord(
        {
            "provider_id": "google",
            "key_hash": "test_hash",
            "model_name": "gemini-2.5-flash",
            "requests_today": 5,
            "tokens_today": 1000,
            "error_count": 2,
            "last_reset": "2025-10-02",
        }
    )

    # Call mark_request_failure
    quota.mark_request_failure(error="Test error")

    # Verify it's NOT a no-op - error count should increment
    assert quota.error_count == 3, "error_count should increment from 2 to 3"


@pytest.mark.asyncio
async def test_circuit_breaker_raises_instead_of_silent_fallback():
    """Test that circuit breaker raises errors instead of returning fake data"""
    from smolrouter.redis_backend import _circuit_breaker

    # Force circuit breaker open by recording failures
    for _ in range(10):
        _circuit_breaker.record_failure()

    # Circuit breaker should now be OPEN
    assert not _circuit_breaker.should_attempt(), "Circuit breaker should be OPEN after failures"

    # Attempting increment should RAISE, not return fake data
    with pytest.raises(Exception, match="circuit breaker OPEN"):
        await RedisApiKeyQuota.increment_usage(
            api_key="test_key",  # pragma: allowlist secret
            provider_id="google",
            model_name="gemini-2.5-flash",
            request_count=1,
            token_count=100,
        )

    # Reset circuit breaker for other tests
    _circuit_breaker.failure_count = 0
    _circuit_breaker.last_failure_time = 0
