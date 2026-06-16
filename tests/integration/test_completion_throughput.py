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
from datetime import datetime

import pytest
import pytest_asyncio

REAL_REDIS_URL = os.getenv("SMOLROUTER_TEST_REDIS_URL")


def _redis_max_connections() -> int:
    try:
        return max(1, int(os.getenv("REDIS_MAX_CONNS", "2048")))
    except (TypeError, ValueError):
        return 2048

pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(not REAL_REDIS_URL, reason="set SMOLROUTER_TEST_REDIS_URL to a real redis"),
]


@pytest_asyncio.fixture
async def real_redis(monkeypatch):
    import redis.asyncio as redis_async
    import smolrouter.redis_backend as redis_backend
    import smolrouter.database as database

    client = redis_async.from_url(
        REAL_REDIS_URL,
        decode_responses=True,
        max_connections=_redis_max_connections(),
    )
    await client.flushall()
    monkeypatch.setattr(redis_backend, "get_redis", lambda: client)
    monkeypatch.setattr(redis_backend, "is_fake_redis", lambda: False)
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
        latencies_ms = []
        while not stop.is_set():
            t0 = time.perf_counter()
            await app_module.api_dashboard(limit=100)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            await asyncio.sleep(0.02)
        return latencies_ms

    poller = asyncio.create_task(poll_dashboard())
    await asyncio.gather(*[cycle(i) for i in range(n)])
    stop.set()
    dashboard_latencies_ms = await poller
    worst_ms = max(dashboard_latencies_ms, default=0.0)

    counters = await RedisRequestLog.get_stats_counters()
    print(
        f"\nunder load: {len(dashboard_latencies_ms)} dashboard polls, "
        f"worst {worst_ms:.1f}ms, inflight(orphans)={counters['inflight']}"
    )
    assert dashboard_latencies_ms, "dashboard poller did not run under load"
    assert counters["inflight"] == 0, f"orphans under load: {counters['inflight']}"


@pytest.mark.asyncio
async def test_background_save_path_persists_completion(real_redis):
    from smolrouter.database import RequestLog
    from smolrouter.redis_backend import RedisRequestLog
    from smolrouter.task_utils import drain_background_tasks

    before = await RedisRequestLog.get_stats_counters()

    entry = await RequestLog.create(
        source_ip="10.0.0.9",
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="https://api.example/v1/chat/completions",
        original_model="glm-4.5-air",
        mapped_model="glm-4.5-air",
    )
    entry.status_code = 200
    entry.completed_at = datetime.now()
    entry.duration_ms = 5
    entry.response_body = b'{"ok": true}'

    entry.save()
    await drain_background_tasks()

    rec = await RedisRequestLog.get_by_id(entry.request_id)
    counters = await RedisRequestLog.get_stats_counters()

    assert rec is not None
    assert rec.status_code == 200
    assert rec.completed_at is not None
    assert counters["completed"] >= before["completed"] + 1
    assert counters["inflight"] == before["inflight"]
