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
import errno
import itertools
import os
from datetime import datetime
from unittest.mock import patch

import pytest

import smolrouter.database as database_module
import smolrouter.storage as storage_module
from smolrouter.database import RequestLog
from smolrouter.redis_backend import RedisRequestLog
from smolrouter.storage import FilesystemBlobStorage
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


@pytest.mark.asyncio
async def test_save_async_persists_request_body_key_when_response_archival_fails(isolated_db, monkeypatch):
    entry = await _pending_entry()
    entry.request_body = b'{"messages":[{"role":"user","content":"hi"}]}'
    entry.response_body = b'{"choices":[{"message":{"content":"hello"}}]}'

    async def request_body_store(self, _blob_storage):
        self.request_body_key = "blob/request-body-1"
        return self.request_body_key

    async def failing_response_store(self, _blob_storage):
        raise OSError("disk full")

    monkeypatch.setattr(type(entry), "_store_request_body_if_needed", request_body_store)
    monkeypatch.setattr(type(entry), "_store_response_body_if_present", failing_response_store)

    await entry.save_async()

    rec = await RedisRequestLog.get_by_id(entry.request_id)
    assert rec is not None
    assert getattr(rec, "request_body_key", None) == "blob/request-body-1"
    assert getattr(rec, "response_body_key", None) in (None, "")
    assert rec.status_code == 200


@pytest.mark.asyncio
async def test_save_async_records_explicit_disk_full_blob_failure(isolated_db, monkeypatch):
    entry = await _pending_entry()
    entry.request_body = b'{"messages":[{"role":"user","content":"hi"}]}'

    async def failing_request_body_store(self, _blob_storage):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(type(entry), "_store_request_body_if_needed", failing_request_body_store)

    await entry.save_async()

    rec = await RedisRequestLog.get_by_id(entry.request_id)
    assert rec is not None
    assert getattr(rec, "request_body_key", None) in (None, "")
    assert getattr(rec, "request_body_status", None) == "write_failed"
    assert "insufficient disk space" in (getattr(rec, "request_body_error", "") or "").lower()


@pytest.mark.asyncio
async def test_save_async_retries_transient_body_storage_result_persistence(isolated_db, monkeypatch):
    entry = await _pending_entry()
    entry.request_body = b'{"messages":[{"role":"user","content":"hi"}]}'

    calls = {"n": 0}
    real_update = RedisRequestLog.update_body_storage_result

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("temporary redis outage")
        return await real_update(*args, **kwargs)

    monkeypatch.setattr(RedisRequestLog, "update_body_storage_result", staticmethod(flaky))

    await entry.save_async()

    rec = await RedisRequestLog.get_by_id(entry.request_id)
    assert calls["n"] >= 2
    assert rec is not None
    assert getattr(rec, "request_body_key", None)
    assert getattr(rec, "request_body_status", None) == "available"


@pytest.mark.asyncio
async def test_background_small_file_siege_preserves_recent_request_bodies_under_1mb_cap(
    isolated_db, monkeypatch, tmp_path
):
    monkeypatch.setattr(storage_module, "MAX_TOTAL_STORAGE_SIZE", 1_000_000)
    monkeypatch.setattr(storage_module, "MAX_BLOB_SIZE", 1_000_000)
    monkeypatch.setattr(storage_module, "KEEP_RECENT_HOURS", 1)
    monkeypatch.setattr(storage_module, "WATERMARK_FRACTION", 0.8)

    blob_storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    monkeypatch.setattr(storage_module, "get_blob_storage", lambda: blob_storage)
    key_counter = itertools.count(1)
    monkeypatch.setattr(blob_storage, "_generate_key", lambda: f"2000000000000-{next(key_counter):08d}")

    request_payload = b"q" * 16384
    total_entries = 80

    before = await RedisRequestLog.get_stats_counters()

    async def create_and_schedule(i: int):
        entry = await RequestLog.create(
            source_ip=f"10.0.0.{i}",
            method="POST",
            path="/v1/chat/completions",
            service_type="openai",
            upstream_url="https://api.example/v1/chat/completions",
        )
        entry.status_code = 200
        entry.completed_at = datetime.now()
        entry.duration_ms = 5
        entry.request_body = request_payload + f":req:{i:03d}".encode()
        entry.save()
        return entry

    entries = await asyncio.gather(*[create_and_schedule(i) for i in range(total_entries)])
    await drain_background_tasks()

    after = await RedisRequestLog.get_stats_counters()
    assert after["completed"] >= before["completed"] + total_entries
    assert after["inflight"] == 0
    assert blob_storage._total_size_bytes() <= storage_module.MAX_TOTAL_STORAGE_SIZE

    all_records = [await RedisRequestLog.get_by_id(entry.request_id) for entry in entries]
    assert all(record is not None for record in all_records)

    records_by_write_order = sorted(
        (record for record in all_records if record is not None and getattr(record, "request_body_key", None)),
        key=lambda record: getattr(record, "request_body_key"),
    )

    assert len(records_by_write_order) == total_entries

    oldest_written = records_by_write_order[:10]
    newest_written = records_by_write_order[-10:]

    assert all(record.status_code == 200 for record in newest_written)
    assert all(record.completed_at is not None for record in newest_written)
    assert all(record.request_body_status == "available" for record in newest_written)
    assert any(record.request_body_status == "not_found" for record in oldest_written)
    assert all(record.request_body_status != "write_failed" for record in oldest_written + newest_written)


def test_save_without_event_loop_archives_bodies(isolated_db, monkeypatch):
    entry = asyncio.run(_pending_entry())
    entry.request_body = b'{"messages":[{"role":"user","content":"hi"}]}'
    entry.response_body = b'{"choices":[{"message":{"content":"hello"}}]}'

    async def request_body_store(self, _blob_storage):
        self.request_body_key = "blob/request-body-inline"
        return self.request_body_key

    async def response_body_store(self, _blob_storage):
        self.response_body_key = "blob/response-body-inline"
        return self.response_body_key

    monkeypatch.setattr(type(entry), "_store_request_body_if_needed", request_body_store)
    monkeypatch.setattr(type(entry), "_store_response_body_if_present", response_body_store)
    def unscheduled_task(coro, *args, **kwargs):
        coro.close()
        return None

    monkeypatch.setattr(database_module, "create_logged_task", unscheduled_task)

    entry.save()

    rec = asyncio.run(RedisRequestLog.get_by_id(entry.request_id))
    assert rec is not None
    assert getattr(rec, "request_body_key", None) == "blob/request-body-inline"
    assert getattr(rec, "response_body_key", None) == "blob/response-body-inline"
    assert rec.status_code == 200
