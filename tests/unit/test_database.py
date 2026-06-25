import asyncio
from datetime import datetime, timedelta, timezone
import smolrouter.database as database
from unittest.mock import AsyncMock

from smolrouter.database import RequestLogEntry, estimate_tokens_from_request
import pytest


def test_request_log_entry_save_schedules_completion_update(monkeypatch):
    entry = RequestLogEntry("req-1", completed_at=datetime.now())
    scheduled = []

    async def fake_run_completion_update(self, *args, **kwargs):
        await asyncio.sleep(0)
        return None

    def fake_create_logged_task(coro, *_args, **_kwargs):
        scheduled.append(coro)
        coro.close()

    monkeypatch.setattr(RequestLogEntry, "_run_completion_update", fake_run_completion_update)
    monkeypatch.setattr(database, "create_logged_task", fake_create_logged_task)

    entry.save()

    assert len(scheduled) == 1


def test_request_log_entry_save_falls_back_to_asyncio_run(monkeypatch):
    entry = RequestLogEntry("req-2", completed_at=datetime.now())
    fallback_calls = []

    async def fake_run_completion_update(self, *args, **kwargs):
        assert kwargs == {"run_archival_inline": True}
        await asyncio.sleep(0)
        return None

    def fake_create_logged_task(coro, *_args, **_kwargs):
        coro.close()
        return None

    def fake_asyncio_run(coro):
        fallback_calls.append(coro)
        coro.close()

    monkeypatch.setattr(RequestLogEntry, "_run_completion_update", fake_run_completion_update)
    monkeypatch.setattr(database, "create_logged_task", fake_create_logged_task)
    monkeypatch.setattr(asyncio, "run", fake_asyncio_run)

    entry.save()

    assert len(fallback_calls) == 1


@pytest.mark.asyncio
async def test_record_exception_event_aggregates_signatures_and_summary(isolated_db):
    signature_payload = ValueError("failure 0x7f00 user@example.com")
    first = await database.record_exception_event(
        request_id="req-1",
        exception=signature_payload,
        route="/api/test",
        request_path="/api/test",
        method="POST",
        source_ip="127.0.0.1",
        status_code=500,
        user_agent="pytest",
    )

    assert first is not None
    assert int(first["count"]) == 1
    assert first["route"] == "/api/test"
    signature = first["signature"]

    second = await database.record_exception_event(
        request_id="req-2",
        exception=ValueError("failure 0x7f00 user@example.com"),
        route="/api/test",
        request_path="/api/test",
        method="POST",
        source_ip="127.0.0.1",
        status_code=500,
        user_agent="pytest",
    )

    assert second is not None
    assert second["signature"] == signature
    assert int(second["count"]) == 2

    summary = await database.get_error_summary()
    assert summary["signature_count"] == 1
    assert summary["count_by_exception_class"]["ValueError"] == 2
    assert summary["count_by_route"]["/api/test"] == 2
    assert summary["count_by_signature"][signature] == 2

    recent = await database.get_error_recent_events(limit=5)
    assert any(event.get("signature") == signature for event in recent)

    updated = await database.set_exception_signature_state(
        signature=signature,
        state="known",
        notes="expected in soak",
    )
    assert updated is not None
    assert updated["state"] == "known"
    assert updated["notes"] == "expected in soak"


@pytest.mark.asyncio
async def test_record_exception_event_falls_back_to_non_atomic_on_eval_failure(monkeypatch, isolated_db):
    async def fail_eval(*_args, **_kwargs):
        raise RuntimeError("atomic aggregation unavailable")

    monkeypatch.setattr(database.redis_client, "eval", AsyncMock(side_effect=fail_eval))

    summary = await database.record_exception_event(
        request_id="req-fallback",
        exception=KeyError("missing-key"),
        route="/api/fallback",
        request_path="/api/fallback",
        method="GET",
        source_ip="10.0.0.2",
        status_code=500,
        user_agent="pytest-fallback",
    )

    assert summary is not None
    assert summary["signature"] is not None
    assert int(summary["count"]) == 1
    assert summary["route"] == "/api/fallback"


def test_normalize_exception_message_trims_and_masks_identifiers():
    message = "Error code 404 in 0x7f00 with user=test@example.com and value 12345 in '/a/b' and token deadbeefcafebabe and 'abc'"
    normalized = database._normalize_exception_message(message)

    assert "<num>" in normalized
    assert "<ptr>" in normalized
    assert "<email>" in normalized
    assert "<hash>" in normalized


def test_build_exception_signature_is_deterministic():
    exc = RuntimeError("route failed 404")
    sig_a, metadata_a = database._build_exception_signature(exc, route="/api/items", message="route failed 404")
    sig_b, _ = database._build_exception_signature(RuntimeError("route failed 404"), route="/api/items", message="route failed 404")

    assert sig_a == sig_b
    assert metadata_a["exception_class"] == "RuntimeError"
    assert metadata_a["route"] == "/api/items"
    assert metadata_a["top_frame"]


def test_to_str_handles_bytes_none_and_objects():
    assert database._to_str(b"hello") == "hello"
    assert database._to_str(bytearray(b"world")) == "world"
    assert database._to_str(None) is None
    assert database._to_str(123) == "123"


def test_extract_exception_top_frame_prefers_application_frame():
    def explode():
        raise RuntimeError("boom")

    try:
        explode()
    except RuntimeError as exc:
        top_frame = database._extract_exception_top_frame(exc)

    assert top_frame.endswith(":explode")
    assert top_frame != database.UNKNOWN_VALUE


def test_extract_exception_top_frame_returns_unknown_without_traceback():
    assert database._extract_exception_top_frame(ValueError("boom")) == database.UNKNOWN_VALUE


@pytest.mark.asyncio
async def test_cleanup_old_logs_async_removes_stale_error_artifacts_and_orphans(isolated_db):
    old_ts = (datetime.now(timezone.utc) - timedelta(days=2)).timestamp()
    old_request_id = "req-old"
    old_source_ip = "10.0.0.1"

    await database.redis_client.hset(
        f"request:{old_request_id}",
        mapping={
            "source_ip": old_source_ip,
            "identity_kind": "facade_key",
            "identity_subject_id": "project-a",
        },
    )
    await database.redis_client.zadd("requests:by_time", {old_request_id: old_ts})
    await database.redis_client.sadd(f"requests:by_ip:{old_source_ip}", old_request_id)
    await database.redis_client.sadd(database.INFLIGHT_SET_KEY, old_request_id)
    await database.redis_client.zadd("requests:by_identity:facade_key:project-a", {old_request_id: old_ts})

    signature = "sig-old"
    signature_key = f"{database.ERROR_SIGNATURE_KEY_PREFIX}{signature}"
    signature_request_ids_key = f"{signature_key}{database.ERROR_SIGNATURE_REQUEST_IDS_KEY}"
    event_key = f"{database.ERROR_EVENT_KEY_PREFIX}event-old"
    signature_events_key = f"{database.ERROR_EVENTS_BY_SIGNATURE_PREFIX}{signature}"

    await database.redis_client.hset(
        signature_key,
        mapping={
            "signature": signature,
            "count": "1",
            "state": "unknown",
            "notes": "",
        },
    )
    await database.redis_client.lpush(signature_request_ids_key, old_request_id)
    await database.redis_client.sadd(database.ERROR_SIGNATURE_SET_KEY, signature)
    await database.redis_client.hset(
        event_key,
        mapping={
            "signature": signature,
            "request_id": old_request_id,
            "timestamp": old_ts,
            "status_code": "500",
            "exception_class": "RuntimeError",
            "route": "/api/test",
            "top_frame": "test_database.py:1:explode",
            "message": "boom",
            "stack_trace": "trace",
        },
    )
    await database.redis_client.zadd(database.ERROR_EVENT_INDEX_KEY, {event_key: old_ts})
    await database.redis_client.zadd(signature_events_key, {"event-old": old_ts})

    orphan_signature = "sig-orphan"
    await database.redis_client.sadd(database.ERROR_SIGNATURE_SET_KEY, orphan_signature)

    deleted = await database.cleanup_old_logs_async(max_age_days=1)

    assert deleted == 2
    assert not await database.redis_client.exists(f"request:{old_request_id}")
    assert not await database.redis_client.sismember(database.INFLIGHT_SET_KEY, old_request_id)
    assert not await database.redis_client.exists(event_key)
    assert not await database.redis_client.exists(signature_key)
    assert not await database.redis_client.exists(signature_request_ids_key)
    assert not await database.redis_client.exists(signature_events_key)
    assert not await database.redis_client.exists(f"{database.ERROR_SIGNATURE_KEY_PREFIX}{orphan_signature}")
    assert await database.redis_client.zcard("requests:by_identity:facade_key:project-a") == 0

def test_estimate_tokens_from_request_counts_chat_messages():
    request_data = {
        "messages": [
            {"content": "abcdefgh"},
            {"content": [{"type": "text", "text": "abcd"}, {"type": "image_url", "image_url": "ignore"}]},
        ]
    }

    assert estimate_tokens_from_request(request_data) == 3


def test_estimate_tokens_from_request_counts_prompt_lists():
    request_data = {"prompt": ["abcd", 12345]}

    assert estimate_tokens_from_request(request_data) == 2
