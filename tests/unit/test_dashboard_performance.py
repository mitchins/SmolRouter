"""
Dashboard / read-path performance regression tests.

Background
----------
The router exhibited severe dashboard latency under load: ``/api/stats`` took
~8s when idle and timed out (>15s) under soak, and the request-list queries
crawled. Root cause was an N+1 in ``RequestLog.get_recent`` - it issued one
awaited ``hgetall`` per record after a ``zrevrange`` - so a 1000-record stats
sample cost ~1001 sequential Redis round-trips, and ``get_log_stats`` did that
*twice* (directly and via ``get_inflight_requests``).

These tests lock in the fix by asserting on **Redis round-trips**, which is the
backend-independent invariant. FakeRedis is in-memory so the bug is invisible in
wall-clock terms here, but the round-trip count makes the regression impossible
to reintroduce silently. See ``tests/redis_roundtrip.py`` for the harness.

The SOAK and PEAK tests additionally assert that dashboard cost stays *flat* as
the stored log volume grows - the defining property the original code lacked.
"""

import asyncio
import os
from unittest.mock import patch

import pytest
import pytest_asyncio

import smolrouter.database as database
from smolrouter.database import get_log_stats, get_recent_logs
from smolrouter.redis_backend import RedisRequestLog
from smolrouter.redis_config import redis_client, is_fake_redis
from tests.redis_roundtrip import count_round_trips


@pytest.fixture(autouse=True)
def ensure_fakeredis():
    """Force FakeRedis so these tests are deterministic and CI-friendly."""
    with patch.dict(os.environ, {"APP_ENV": "test"}):
        original = os.environ.pop("REDIS_URL", None)
        yield
        if original is not None:
            os.environ["REDIS_URL"] = original


@pytest_asyncio.fixture
async def fresh_redis():
    await redis_client.flushall()
    yield redis_client
    await redis_client.flushall()


async def _seed_requests(n: int, *, complete: bool = True) -> None:
    """Create ``n`` request logs (optionally completed)."""
    for i in range(n):
        request_id = await RedisRequestLog.create(
            source_ip=f"10.0.{i // 256}.{i % 256}",
            method="POST",
            path="/v1/chat/completions",
            service_type="openai",
            upstream_url="https://api.openai.com/v1/chat/completions",
            original_model="gpt-oss-20b",
            mapped_model="gpt-oss-20b",
        )
        if complete:
            await RedisRequestLog.update_completion(
                request_id=request_id, status_code=200, response_size=500, error_message=None
            )


class TestGetRecentRoundTrips:
    """get_recent must batch reads: O(1) round-trips, not O(N)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("n", [10, 100, 500])
    async def test_get_recent_is_constant_round_trips(self, fresh_redis, n):
        assert is_fake_redis()
        await _seed_requests(n)

        with count_round_trips() as counter:
            logs = await RedisRequestLog.get_recent(n)

        assert len(logs) == n
        # zrevrange (1) + a single batched pipeline (1) == 2, independent of n.
        # The pre-fix N+1 produced n + 1 round-trips.
        assert counter.round_trips <= 3, (
            f"get_recent({n}) made {counter.round_trips} round-trips "
            f"({dict(counter.calls)}); expected O(1). N+1 regression?"
        )

    @pytest.mark.asyncio
    async def test_round_trips_do_not_grow_with_volume(self, fresh_redis):
        """The whole point: cost is flat as stored volume increases."""
        assert is_fake_redis()

        await _seed_requests(50)
        with count_round_trips() as small:
            await RedisRequestLog.get_recent(50)

        await _seed_requests(450)  # 500 total now
        with count_round_trips() as large:
            await RedisRequestLog.get_recent(500)

        assert large.round_trips == small.round_trips, (
            f"round-trips grew with volume: {small.round_trips} -> {large.round_trips}"
        )


class TestStatsRoundTrips:
    """get_log_stats must not re-fetch the recent sample twice."""

    @pytest.mark.asyncio
    async def test_stats_does_not_double_fetch(self, fresh_redis):
        assert is_fake_redis()
        await _seed_requests(300)

        with count_round_trips() as counter:
            stats = await get_log_stats()

        assert stats["total_requests"] == 300
        # One get_recent (zrevrange + 1 pipeline = 2) for the sample, reused for
        # inflight, plus a bounded number of error-summary lookups. A second
        # get_recent(1000) (the old behaviour) would batch-read 300 hashes
        # again. Keep a tight ceiling that the double-fetch would breach.
        assert counter.round_trips <= 8, (
            f"get_log_stats made {counter.round_trips} round-trips "
            f"({dict(counter.calls)}); a double get_recent regressed?"
        )

    @pytest.mark.asyncio
    async def test_stats_round_trips_flat_under_growth(self, fresh_redis):
        assert is_fake_redis()

        await _seed_requests(100)
        with count_round_trips() as small:
            await get_log_stats()

        await _seed_requests(900)  # 1000 total
        with count_round_trips() as large:
            await get_log_stats()

        # Round-trips must stay flat even though the sample grew 10x.
        assert large.round_trips <= small.round_trips + 1, (
            f"stats round-trips grew with volume: {small.round_trips} -> {large.round_trips}"
        )


@pytest.mark.performance
class TestDashboardSoak:
    """SOAK: sustained write load with periodic dashboard reads.

    Models production: the hot path keeps logging requests while operators (and
    the auto-refreshing UI) poll the dashboard. Asserts the dashboard read cost
    stays bounded and flat across the soak instead of degrading as the log set
    grows - the exact failure mode that was reported.
    """

    @pytest.mark.asyncio
    async def test_dashboard_reads_stay_bounded_during_soak(self, fresh_redis):
        assert is_fake_redis()

        rounds = 10
        writes_per_round = 200
        read_costs = []

        for _ in range(rounds):
            await _seed_requests(writes_per_round)
            with count_round_trips() as counter:
                logs = await get_recent_logs(limit=100)
                stats = await get_log_stats()
            assert len(logs) == 100
            assert stats["total_requests"] >= writes_per_round
            read_costs.append(counter.round_trips)

        first, last = read_costs[0], read_costs[-1]
        # 2000 logs accumulated by the end; a per-record read path would make
        # `last` an order of magnitude larger than `first`.
        assert last <= first + 1, f"dashboard cost degraded over soak: {read_costs}"
        assert max(read_costs) <= 12, f"dashboard read cost too high: {read_costs}"


@pytest.mark.performance
class TestDashboardPeak:
    """PEAK: many concurrent dashboard reads against a large log set.

    Asserts the dashboard survives a burst of simultaneous viewers without the
    aggregate round-trip count exploding (N+1 multiplied by concurrency was the
    pathological case that exhausted the Redis pool and stalled the UI).
    """

    @pytest.mark.asyncio
    async def test_concurrent_dashboard_reads(self, fresh_redis):
        assert is_fake_redis()
        await _seed_requests(1000)

        concurrent_viewers = 25

        with count_round_trips() as counter:
            results = await asyncio.gather(
                *[get_log_stats() for _ in range(concurrent_viewers)]
            )

        assert len(results) == concurrent_viewers
        assert all(r["total_requests"] == 1000 for r in results)

        # Per-viewer cost must be O(1) in stored volume. With ~1000 logs, the
        # old N+1 path would be ~2002 round-trips *per viewer*; batched it is a
        # small constant. Bound the per-viewer average tightly.
        per_viewer = counter.round_trips / concurrent_viewers
        assert per_viewer <= 8, (
            f"per-viewer dashboard cost {per_viewer:.1f} round-trips "
            f"(total {counter.round_trips}); N+1 under concurrency?"
        )
