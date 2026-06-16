"""
Worst-case performance guards for the dashboard / stats hot paths.

The dashboard polls these on a few-TPS box with Redis on the SAME host, yet
observed ~9.8s. Root cause: stats/inflight deserialize a large record sample in
Python on every poll (event-loop-blocking), and the error summary is N+1 over
signatures. Redis being local means the cost is Python work, not round-trips.

The robust way to guarantee "≤250ms on the real box" without flaky wall-clock
assertions (fakeredis won't reproduce the real-Redis+LAN latency) is to assert
the *work is bounded* regardless of how much data is stored:
  - stats / inflight must be O(1) (no per-record deserialization)
  - the dashboard must deserialize only the page it returns
  - the error summary must not scale its Redis round-trips with signature count

These are RED against the current implementation and GREEN once stats/inflight
are served from counters/sets and the error summary is batched. Real wall-clock
≤250ms proofs live in tests/integration/test_dashboard_redis_latency.py (gated
on a real Redis).
"""

import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import pytest_asyncio

import smolrouter.app as app_module
import smolrouter.database as database
from smolrouter.database import (
    get_error_summary,
    get_inflight_requests,
    get_log_stats,
    get_recent_logs,
    record_exception_event,
)
from smolrouter.redis_backend import RedisRequestLog
from smolrouter.redis_config import redis_client, is_fake_redis
from tests.redis_roundtrip import RoundTripCounter

# Bound for "O(1)" work: a handful of reads, never a per-record scan.
O1_RECORDS = 50
# Bound for the error summary's Redis round-trips, independent of signature count.
O1_ROUNDTRIPS = 15
DEFAULT_PAGE = 100


@pytest.fixture(autouse=True)
def ensure_fakeredis():
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


async def _seed(n: int, state: str = "completed") -> None:
    """Seed n request logs in a given terminal/non-terminal state."""
    for i in range(n):
        rid = await RedisRequestLog.create(
            source_ip=f"10.0.{i // 256}.{i % 256}",
            method="POST",
            path="/v1/chat/completions",
            service_type="openai" if i % 2 == 0 else "ollama",
            upstream_url="https://api.example/v1/chat/completions",
            original_model="glm-4.5-air",
            mapped_model="glm-4.5-air",
        )
        if state == "completed":
            await RedisRequestLog.update_completion(request_id=rid, status_code=200, response_size=100)
        elif state == "failed":
            await RedisRequestLog.update_completion(request_id=rid, status_code=500, error_message="upstream boom")
        # "pending" / "inflight" / "orphan": leave uncompleted (status pending)


async def _seed_mixed(per_state: int) -> None:
    await _seed(per_state, "completed")
    await _seed(per_state, "failed")
    await _seed(per_state, "pending")


async def _seed_signatures(n: int) -> None:
    for i in range(n):
        try:
            raise ValueError(f"distinct failure {i}")  # distinct message+route -> distinct signature
        except ValueError as exc:
            await record_exception_event(
                request_id=f"req-{i}", exception=exc, route=f"/route/{i}", request_path=f"/p/{i}"
            )


@contextmanager
def count_records_deserialized():
    """Count request records deserialized via get_recent (the dominant CPU cost)."""
    state = {"records": 0, "calls": 0}
    original = RedisRequestLog.get_recent

    async def wrapped(limit: int = 100):
        records = await original(limit)
        state["records"] += len(records)
        state["calls"] += 1
        return records

    with patch.object(RedisRequestLog, "get_recent", staticmethod(wrapped)):
        yield state


@contextmanager
def count_db_roundtrips():
    """Count Redis round-trips made through database.redis_client (error path)."""
    counter = RoundTripCounter(database.redis_client)
    with patch.object(database, "redis_client", counter):
        yield counter


@pytest.mark.performance
class TestStatsWorstCase:
    """get_log_stats must be O(1) regardless of stored volume or request state."""

    @pytest.mark.asyncio
    async def test_stats_o1_with_many_completed(self, fresh_redis):
        await _seed(800, "completed")
        with count_records_deserialized() as c:
            stats = await get_log_stats()
        assert stats["completed_requests"] >= 1
        assert c["records"] <= O1_RECORDS, f"stats deserialized {c['records']} records with 800 completed"

    @pytest.mark.asyncio
    async def test_stats_o1_with_many_orphans(self, fresh_redis):
        await _seed(800, "pending")  # the production orphan pile
        with count_records_deserialized() as c:
            await get_log_stats()
        assert c["records"] <= O1_RECORDS, f"stats deserialized {c['records']} records with 800 orphans"

    @pytest.mark.asyncio
    async def test_stats_o1_with_many_failed(self, fresh_redis):
        await _seed(800, "failed")
        with count_records_deserialized() as c:
            await get_log_stats()
        assert c["records"] <= O1_RECORDS, f"stats deserialized {c['records']} records with 800 failed"

    @pytest.mark.asyncio
    async def test_stats_o1_with_huge_mixed(self, fresh_redis):
        await _seed_mixed(500)  # 1500 total, mixed states (exceeds the 1000 sample cap)
        with count_records_deserialized() as c:
            await get_log_stats()
        assert c["records"] <= O1_RECORDS, f"stats deserialized {c['records']} records with 1500 mixed"

    @pytest.mark.asyncio
    async def test_stats_work_constant_as_volume_grows(self, fresh_redis):
        await _seed(100, "completed")
        with count_records_deserialized() as small:
            await get_log_stats()
        await _seed(1400, "completed")  # 1500 total
        with count_records_deserialized() as large:
            await get_log_stats()
        assert large["records"] <= small["records"] + 5, (
            f"stats work grew with volume: {small['records']} -> {large['records']}"
        )


@pytest.mark.performance
class TestInflightWorstCase:
    """Inflight count must be O(1), not a scan of all pending records."""

    @pytest.mark.asyncio
    async def test_inflight_o1_with_many_inflight(self, fresh_redis):
        await _seed(800, "inflight")
        with count_records_deserialized() as c:
            await get_inflight_requests()
        assert c["records"] <= O1_RECORDS, f"inflight deserialized {c['records']} records with 800 inflight"


@pytest.mark.performance
class TestDashboardWorstCase:
    """api_dashboard must deserialize only the page it returns, not the stats sample."""

    @pytest.mark.asyncio
    async def test_dashboard_page_bounded_with_many_completed(self, fresh_redis):
        await _seed(1200, "completed")
        with count_records_deserialized() as c:
            await app_module.api_dashboard(limit=DEFAULT_PAGE)
        assert c["records"] <= DEFAULT_PAGE + O1_RECORDS, f"dashboard deserialized {c['records']} records"

    @pytest.mark.asyncio
    async def test_dashboard_page_bounded_with_many_orphans(self, fresh_redis):
        await _seed(1200, "pending")
        with count_records_deserialized() as c:
            await app_module.api_dashboard(limit=DEFAULT_PAGE)
        assert c["records"] <= DEFAULT_PAGE + O1_RECORDS, f"dashboard deserialized {c['records']} records"

    @pytest.mark.asyncio
    async def test_dashboard_page_bounded_with_huge_mixed(self, fresh_redis):
        await _seed_mixed(500)
        with count_records_deserialized() as c:
            await app_module.api_dashboard(limit=DEFAULT_PAGE)
        assert c["records"] <= DEFAULT_PAGE + O1_RECORDS, f"dashboard deserialized {c['records']} records"


@pytest.mark.performance
class TestErrorSummaryWorstCase:
    """get_error_summary must not scale Redis round-trips with signature count (N+1)."""

    @pytest.mark.asyncio
    async def test_error_summary_roundtrips_bounded_with_many_signatures(self, fresh_redis):
        await _seed_signatures(150)
        with count_db_roundtrips() as counter:
            summary = await get_error_summary()
        assert summary.get("signature_count", 0) >= 1
        assert counter.round_trips <= O1_ROUNDTRIPS, (
            f"error summary made {counter.round_trips} round-trips for 150 signatures (N+1)"
        )


@pytest.mark.performance
class TestLogsPageContract:
    """The log list (used by the dashboard) must fetch only the page, never the full set."""

    @pytest.mark.asyncio
    async def test_recent_logs_fetch_is_page_bounded(self, fresh_redis):
        await _seed(1200, "completed")
        with count_records_deserialized() as c:
            logs = await get_recent_logs(limit=DEFAULT_PAGE)
        assert len(logs) == DEFAULT_PAGE
        assert c["records"] <= DEFAULT_PAGE + 5, f"log page deserialized {c['records']} records"
