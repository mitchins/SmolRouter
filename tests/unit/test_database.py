import asyncio
from datetime import datetime
import smolrouter.database as database
from unittest.mock import AsyncMock

from smolrouter.database import RequestLogEntry, estimate_tokens_from_request
import pytest


def test_request_log_entry_save_schedules_completion_update(monkeypatch):
    entry = RequestLogEntry("req-1", completed_at=datetime.now())
    scheduled = []

    async def fake_run_completion_update(self):
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

    async def fake_run_completion_update(self):
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
