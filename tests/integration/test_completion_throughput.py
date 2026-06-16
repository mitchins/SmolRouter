"""
Throughput ceiling for the request create+complete path against a REAL Redis.

Answers "up to what TPS does the system complete every request without
orphaning?" - i.e. where the 'truly sustained' residual (completion writes the
retry can't absorb -> stuck in the inflight set) actually starts to bite.

Drives create+complete cycles through the real Redis backend at increasing
concurrency, reports achieved TPS, and asserts no orphans (inflight set drains,
completed == total). Gated on SMOLROUTER_TEST_REDIS_URL:

    docker run -d --rm -p 6399:6379 redis:7-alpine
    SMOLROUTER_TEST_REDIS_URL=redis://localhost:6399/0 \
        pytest tests/integration/test_completion_throughput.py -v -s
"""

import asyncio
import os
import time

import pytest
import pytest_asyncio

REAL_REDIS_URL = os.getenv("SMOLROUTER_TEST_REDIS_URL")

pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(not REAL_REDIS_URL, reason="set SMOLROUTER_TEST_REDIS_URL to a real redis"),
]


@pytest_asyncio.fixture
async def real_redis(monkeypatch):
    import redis.asyncio as redis_async
    import smolrouter.redis_backend as redis_backend
    import smolrouter.database as database

    client = redis_async.from_url(REAL_REDIS_URL, decode_responses=True)
    await client.flushall()
    monkeypatch.setattr(redis_backend, "get_redis", lambda: client)
    monkeypatch.setattr(database, "redis_client", client)
    try:
        yield client
    finally:
        await client.flushall()
        await client.aclose()


async def _run_level(client, n: int, concurrency: int) -> tuple[float, dict]:
    from smolrouter.redis_backend import RedisRequestLog

    await client.flushall()
    sem = asyncio.Semaphore(concurrency)

    async def cycle(i: int):
        async with sem:
            rid = await RedisRequestLog.create(
                source_ip=f"10.0.{i % 256}.1",
                method="POST",
                path="/v1/chat/completions",
                service_type="openai",
                upstream_url="https://api.example/v1/chat/completions",
                original_model="glm-4.5-air",
                mapped_model="glm-4.5-air",
            )
            await RedisRequestLog.update_completion(request_id=rid, status_code=200, response_size=20)

    t0 = time.perf_counter()
    await asyncio.gather(*[cycle(i) for i in range(n)])
    elapsed = time.perf_counter() - t0
    counters = await RedisRequestLog.get_stats_counters()
    return n / elapsed, counters


@pytest.mark.asyncio
async def test_completion_throughput_sweep_no_orphans(real_redis):
    """Sweep concurrency; report achieved TPS and assert zero orphans at each level."""
    print("\nconcurrency |   TPS   | completed | inflight(orphans)")
    for concurrency in (100, 500, 1000, 2000):
        n = concurrency * 5
        tps, counters = await _run_level(real_redis, n, concurrency)
        print(f"  {concurrency:>6}    | {tps:>7.0f} | {counters['completed']:>9} | {counters['inflight']:>4}")
        assert counters["completed"] == n, f"lost completions at concurrency {concurrency}"
        assert counters["inflight"] == 0, f"orphans at concurrency {concurrency}: {counters['inflight']}"


@pytest.mark.asyncio
async def test_no_orphans_while_dashboard_polled_under_load(real_redis):
    """The original vicious cycle: dashboard polling blocked the event loop (~2.3s
    deserialization) and starved completion writes -> orphans. With O(1) stats the
    dashboard must stay fast AND completions must not orphan under sustained load."""
    import smolrouter.app as app_module
    from smolrouter.redis_backend import RedisRequestLog

    n = 5000
    concurrency = 500
    sem = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()

    async def cycle(i: int):
        async with sem:
            rid = await RedisRequestLog.create(
                source_ip=f"10.0.{i % 256}.1", method="POST", path="/v1/chat/completions",
                service_type="openai", upstream_url="u", original_model="glm", mapped_model="glm",
            )
            await RedisRequestLog.update_completion(request_id=rid, status_code=200, response_size=20)

    async def poll_dashboard():
        polls, worst_ms = 0, 0.0
        while not stop.is_set():
            t0 = time.perf_counter()
            await app_module.api_dashboard(limit=100)
            worst_ms = max(worst_ms, (time.perf_counter() - t0) * 1000)
            polls += 1
            await asyncio.sleep(0.02)
        return polls, worst_ms

    poller = asyncio.create_task(poll_dashboard())
    await asyncio.gather(*[cycle(i) for i in range(n)])
    stop.set()
    polls, worst_ms = await poller

    counters = await RedisRequestLog.get_stats_counters()
    print(f"\nunder load: {polls} dashboard polls, worst {worst_ms:.1f}ms, inflight(orphans)={counters['inflight']}")
    assert counters["inflight"] == 0, f"orphans under load: {counters['inflight']}"
    assert worst_ms < 250, f"dashboard poll hit {worst_ms:.0f}ms under load"
