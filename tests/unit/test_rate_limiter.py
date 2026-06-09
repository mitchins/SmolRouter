"""Unit tests for smolrouter.rate_limiter.GoogleGenAIRequestFunnel.

Exercises the disabled no-op path, concurrency limiting, the rolling-window
limiter, timestamp pruning, and the stats snapshot.
"""

import asyncio

import pytest

from smolrouter.rate_limiter import GoogleGenAIRequestFunnel


# --------------------------------------------------------------------------
# Disabled funnel
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_funnel_is_noop():
    funnel = GoogleGenAIRequestFunnel(enabled=False)
    # Should not raise and should not track anything
    await funnel.acquire_slot()
    await funnel.release_slot()
    assert funnel.stats == {"enabled": False}


# --------------------------------------------------------------------------
# Stats / basic acquire-release accounting
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_updates_stats():
    funnel = GoogleGenAIRequestFunnel(
        max_concurrent=2, max_requests_per_window=12, window_minutes=4, enabled=True
    )
    await funnel.acquire_slot()
    stats = funnel.stats
    assert stats["enabled"] is True
    assert stats["active_requests"] == 1
    assert stats["total_requests"] == 1
    assert stats["window_usage"] == "1/12"
    assert stats["window_remaining_seconds"] > 0


@pytest.mark.asyncio
async def test_release_decrements_active_count():
    funnel = GoogleGenAIRequestFunnel(max_concurrent=2, enabled=True)
    await funnel.acquire_slot()
    await funnel.release_slot()
    assert funnel.stats["active_requests"] == 0
    # Timestamp remains recorded for the rolling window
    assert funnel.stats["total_requests"] == 1


@pytest.mark.asyncio
async def test_release_with_no_active_is_safe(caplog):
    funnel = GoogleGenAIRequestFunnel(max_concurrent=2, enabled=True)
    # Release without an acquire should warn but not raise or go negative
    await funnel.release_slot()
    assert funnel.stats["active_requests"] == 0


@pytest.mark.asyncio
async def test_stats_window_remaining_zero_when_empty():
    funnel = GoogleGenAIRequestFunnel(max_concurrent=2, enabled=True)
    # No requests yet -> no timestamps -> remaining is 0
    assert funnel.stats["window_remaining_seconds"] == 0


# --------------------------------------------------------------------------
# Concurrency limiting (semaphore)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_limit_blocks_extra_acquirer():
    funnel = GoogleGenAIRequestFunnel(
        max_concurrent=1, max_requests_per_window=100, enabled=True
    )
    await funnel.acquire_slot()  # fills the single concurrent slot

    # The second acquire must block on the semaphore
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(funnel.acquire_slot(), timeout=0.2)

    # After releasing, a new acquire succeeds promptly
    await funnel.release_slot()
    await asyncio.wait_for(funnel.acquire_slot(), timeout=0.5)
    assert funnel.stats["active_requests"] == 1


# --------------------------------------------------------------------------
# Rolling-window limiting
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_window_limit_blocks_when_full():
    funnel = GoogleGenAIRequestFunnel(
        max_concurrent=10, max_requests_per_window=2, window_minutes=4, enabled=True
    )
    await funnel.acquire_slot()
    await funnel.acquire_slot()
    assert funnel.stats["window_usage"] == "2/2"

    # Third request cannot get a window slot -> blocks in _wait_for_window_slot
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(funnel.acquire_slot(), timeout=0.3)


@pytest.mark.asyncio
async def test_window_prunes_expired_timestamps(monkeypatch):
    funnel = GoogleGenAIRequestFunnel(
        max_concurrent=10, max_requests_per_window=1, window_minutes=4, enabled=True
    )

    base = 1000.0
    monkeypatch.setattr("smolrouter.rate_limiter.time.time", lambda: base)
    await funnel.acquire_slot()
    assert funnel.stats["window_usage"] == "1/1"

    # Advance time beyond the window so the old timestamp is pruned
    later = base + funnel._window_seconds + 10
    monkeypatch.setattr("smolrouter.rate_limiter.time.time", lambda: later)

    # A new acquire should now succeed without blocking
    await asyncio.wait_for(funnel.acquire_slot(), timeout=0.5)
    assert funnel.stats["window_usage"] == "1/1"
