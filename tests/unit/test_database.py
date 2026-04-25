import asyncio
from datetime import datetime

from smolrouter.database import RequestLogEntry, estimate_tokens_from_request


def test_request_log_entry_save_schedules_completion_update(monkeypatch):
    entry = RequestLogEntry("req-1", completed_at=datetime.now())
    scheduled = []

    async def fake_run_completion_update(self):
        await asyncio.sleep(0)
        return None

    def fake_create_task(coro):
        scheduled.append(coro)
        coro.close()

    monkeypatch.setattr(RequestLogEntry, "_run_completion_update", fake_run_completion_update)
    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

    entry.save()

    assert len(scheduled) == 1


def test_request_log_entry_save_falls_back_to_asyncio_run(monkeypatch):
    entry = RequestLogEntry("req-2", completed_at=datetime.now())
    fallback_calls = []

    async def fake_run_completion_update(self):
        await asyncio.sleep(0)
        return None

    def fake_create_task(coro):
        coro.close()
        raise RuntimeError("no running event loop")

    def fake_asyncio_run(coro):
        fallback_calls.append(coro)
        coro.close()

    monkeypatch.setattr(RequestLogEntry, "_run_completion_update", fake_run_completion_update)
    monkeypatch.setattr(asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(asyncio, "run", fake_asyncio_run)

    entry.save()

    assert len(fallback_calls) == 1


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