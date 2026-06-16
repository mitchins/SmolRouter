"""
Defect: the request-completion write is fire-and-forget and swallows errors with
no retry, so a *transient* Redis failure (pool contention under load, the
condition that produced the orphan pile) leaves the request permanently
"pending" AND leaves it in the inflight set (the SREM never fires) - silently.

These prove the defect (RED) and lock the fix (GREEN): a transient failure on the
completion write must not orphan the request, and completion accounting must not
wait on blob archival. Deterministic - the fire-and-forget body
(_run_completion_update) is awaited directly unless the test is explicitly
verifying the background scheduler.
"""

import asyncio
import os
from datetime import datetime
from unittest.mock import patch

import pytest

from smolrouter.database import RequestLog
from smolrouter.redis_backend import RedisRequestLog
from smolrouter.task_utils import drain_background_tasks


@pytest.fixture(autouse=True)
def ensure_fakeredis():
    with patch.dict(os.environ, {"APP_ENV": "test"}):
        original = os.environ.pop("REDIS_URL", None)
        yield
        if original is not None:
            os.environ["REDIS_URL"] = original


async def _pending_entry():
    entry = await RequestLog.create(
        source_ip="1.2.3.4",
        method="POST",
        path="/v1/chat/completions",
        service_type="openai",
        upstream_url="https://api.example/v1/chat/completions",
    )
    entry.status_code = 200
    entry.completed_at = datetime.now()
    entry.duration_ms = 5
    return entry


def _flaky_update_completion(fail_times: int, calls: dict):
    real_update = RedisRequestLog.update_completion

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise ConnectionError("max number of clients reached")  # transient pool contention
        return await real_update(*args, **kwargs)

    return staticmethod(flaky)


@pytest.mark.asyncio
async def test_completion_survives_transient_redis_failure(isolated_db, monkeypatch):
    entry = await _pending_entry()
    calls = {"n": 0}
    monkeypatch.setattr(RedisRequestLog, "update_completion", _flaky_update_completion(1, calls))

    await entry._run_completion_update()  # the fire-and-forget body, awaited

    assert calls["n"] >= 2, "completion was not retried after a transient failure"
    rec = await RedisRequestLog.get_by_id(entry.request_id)
    assert rec is not None
    assert rec.status_code == 200, "DEFECT: transient failure orphaned the request (status not persisted)"


@pytest.mark.asyncio
async def test_inflight_set_drains_after_transient_failure(isolated_db, monkeypatch):
    entry = await _pending_entry()

    before = await RedisRequestLog.get_stats_counters()
    assert before["inflight"] == 1  # SADD on create

    calls = {"n": 0}
    monkeypatch.setattr(RedisRequestLog, "update_completion", _flaky_update_completion(1, calls))

    await entry._run_completion_update()

    after = await RedisRequestLog.get_stats_counters()
    assert after["inflight"] == 0, "DEFECT: orphaned completion left the request in the inflight set"


@pytest.mark.asyncio
async def test_completion_persists_before_body_archival(isolated_db, monkeypatch):
    entry = await _pending_entry()
    entry.request_body = b'{"messages":[{"role":"user","content":"hi"}]}'

    release_archival = asyncio.Event()

    async def blocked_request_body_store(self, _blob_storage):
        await release_archival.wait()
        self.request_body_key = "blob/request-body-1"
        return self.request_body_key

    async def no_response_body(self, _blob_storage):
        return None

    monkeypatch.setattr(type(entry), "_store_request_body_if_needed", blocked_request_body_store)
    monkeypatch.setattr(type(entry), "_store_response_body_if_present", no_response_body)

    await asyncio.wait_for(entry._run_completion_update(), timeout=0.2)

    rec = await RedisRequestLog.get_by_id(entry.request_id)
    assert rec is not None
    assert rec.status_code == 200, "DEFECT: completion waited on blob archival before persisting status"
    counters = await RedisRequestLog.get_stats_counters()
    assert counters["inflight"] == 0, "DEFECT: completion accounting waited on blob archival"

    release_archival.set()
    await drain_background_tasks()
