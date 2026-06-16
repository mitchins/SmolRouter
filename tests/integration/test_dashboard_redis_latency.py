"""
Production-like dashboard latency tests against a REAL networked Redis.

Why this exists
---------------
The dashboard N+1 (one awaited ``hgetall`` per record) is *invisible* with
FakeRedis because FakeRedis is in-memory: every command is effectively free, so
wall-clock latency does not grow with the number of round-trips. The
deterministic CI guard for the N+1 therefore lives in
``tests/unit/test_dashboard_performance.py`` and asserts on **round-trip count**.

These tests are the complementary half: they run against a real Redis over a
socket, where latency == round_trips x RTT, and assert that dashboard reads stay
fast and *flat* as stored volume grows - i.e. they reproduce the actual reported
symptom and prove the fix in wall-clock terms.

They are opt-in so CI stays hermetic and docker-free:

    # one-off:
    docker run -d --rm -p 6399:6379 redis:7-alpine
    SMOLROUTER_TEST_REDIS_URL=redis://localhost:6399/0 \
        pytest tests/integration/test_dashboard_redis_latency.py -v -s

Without ``SMOLROUTER_TEST_REDIS_URL`` set, every test here is skipped.
"""

import asyncio
import os
import time

import pytest
import pytest_asyncio

REAL_REDIS_URL = os.getenv("SMOLROUTER_TEST_REDIS_URL")
REAL_REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNS", "2048"))

pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(
        not REAL_REDIS_URL,
        reason="set SMOLROUTER_TEST_REDIS_URL to a real redis to run latency tests",
    ),
]


@pytest_asyncio.fixture
async def real_redis_backend(monkeypatch):
    """Point the redis backend at a real Redis for the duration of a test."""
    import redis.asyncio as redis_async

    import smolrouter.redis_backend as redis_backend
    import smolrouter.database as database  # noqa: F401 - ensures module import

    client = redis_async.from_url(
        REAL_REDIS_URL,
        decode_responses=True,
        max_connections=REAL_REDIS_MAX_CONNECTIONS,
    )
    await client.flushall()

    # Backend code resolves the client via redis_backend.get_redis().
    monkeypatch.setattr(redis_backend, "get_redis", lambda: client)
    # get_error_summary uses database.redis_client directly; point it at real Redis too.
    monkeypatch.setattr(database, "redis_client", client)

    try:
        yield client
    finally:
        await client.flushall()
        await client.aclose()


async def _seed(n: int) -> None:
    from smolrouter.redis_backend import RedisRequestLog

    for i in range(n):
        rid = await RedisRequestLog.create(
            source_ip=f"10.0.{i // 256}.{i % 256}",
            method="POST",
            path="/v1/chat/completions",
            service_type="openai",
            upstream_url="https://api.openai.com/v1/chat/completions",
            original_model="gpt-oss-20b",
            mapped_model="gpt-oss-20b",
        )
        await RedisRequestLog.update_completion(
            request_id=rid, status_code=200, response_size=500, error_message=None
        )


@pytest.mark.asyncio
async def test_get_recent_latency_is_flat_across_volume(real_redis_backend):
    """get_recent latency must not scale with stored volume (N+1 would)."""
    from smolrouter.redis_backend import RedisRequestLog

    await _seed(100)
    t0 = time.perf_counter()
    await RedisRequestLog.get_recent(100)
    small = time.perf_counter() - t0

    await _seed(900)  # 1000 total
    t0 = time.perf_counter()
    logs = await RedisRequestLog.get_recent(100)
    large = time.perf_counter() - t0

    assert len(logs) == 100
    print(f"\nget_recent: 100={small * 1000:.1f}ms  (after 1000 total)={large * 1000:.1f}ms")
    # Batched: ~2 round-trips regardless of size. Allow generous headroom for
    # payload size growth, but a per-record N+1 would make this ~10x, not <4x.
    assert large < small * 4 + 0.05, f"latency scaled with volume: {small:.3f}s -> {large:.3f}s"


@pytest.mark.asyncio
async def test_stats_latency_bounded(real_redis_backend):
    """/api/stats (get_log_stats) must complete quickly even with a full sample."""
    from smolrouter.database import get_log_stats

    await _seed(1000)
    t0 = time.perf_counter()
    stats = await get_log_stats()
    elapsed = time.perf_counter() - t0

    assert stats["total_requests"] == 1000
    print(f"\nget_log_stats(1000 logs): {elapsed * 1000:.1f}ms")
    # Pre-fix this was ~2000 sequential round-trips and took seconds against a
    # networked Redis. Batched it is a handful of round-trips; 1s is a very
    # loose ceiling that the N+1 would blow past.
    assert elapsed < 1.0, f"get_log_stats too slow: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_peak_concurrent_dashboard_reads(real_redis_backend):
    """A burst of concurrent dashboard readers must not collapse."""
    from smolrouter.database import get_log_stats

    await _seed(1000)
    viewers = 25

    t0 = time.perf_counter()
    results = await asyncio.gather(*[get_log_stats() for _ in range(viewers)])
    elapsed = time.perf_counter() - t0

    assert all(r["total_requests"] == 1000 for r in results)
    print(f"\n{viewers} concurrent get_log_stats: {elapsed * 1000:.1f}ms total")
    # 25 viewers x N+1 against networked redis would be tens of thousands of
    # round-trips. Batched, this stays comfortably bounded.
    assert elapsed < 3.0, f"peak dashboard load too slow: {elapsed:.3f}s"


async def _seed_pending(n: int) -> None:
    """Seed n never-completed requests (the worst-case orphan/inflight pile)."""
    from smolrouter.redis_backend import RedisRequestLog

    for i in range(n):
        await RedisRequestLog.create(
            source_ip=f"10.1.{i // 256}.{i % 256}",
            method="POST",
            path="/v1/chat/completions",
            service_type="openai",
            upstream_url="https://api.openai.com/v1/chat/completions",
            original_model="glm-4.5-air",
            mapped_model="glm-4.5-air",
        )


@pytest.mark.asyncio
async def test_stats_under_250ms_worst_case(real_redis_backend):
    """The literal bar: get_log_stats must be <=250ms even with a large mixed
    dataset (completed + orphan pile). O(1) counters make this volume-independent."""
    from smolrouter.database import get_log_stats

    await _seed(1500)            # completed
    await _seed_pending(1500)    # orphan/inflight pile

    t0 = time.perf_counter()
    stats = await get_log_stats()
    elapsed = time.perf_counter() - t0

    print(f"\nget_log_stats(3000 stored): {elapsed * 1000:.1f}ms")
    assert stats["total_requests"] == 3000
    assert elapsed < 0.25, f"get_log_stats took {elapsed * 1000:.0f}ms (>250ms)"


@pytest.mark.asyncio
async def test_dashboard_under_250ms_worst_case(real_redis_backend):
    """/api/dashboard must be <=250ms with a large dataset (it only deserializes
    the page; stats/inflight are O(1))."""
    import smolrouter.app as app_module

    await _seed(1500)
    await _seed_pending(1500)

    t0 = time.perf_counter()
    result = await app_module.api_dashboard(limit=100)
    elapsed = time.perf_counter() - t0

    print(f"\napi_dashboard(3000 stored): {elapsed * 1000:.1f}ms")
    assert result["stats"]["total_requests"] == 3000
    assert elapsed < 0.25, f"api_dashboard took {elapsed * 1000:.0f}ms (>250ms)"
