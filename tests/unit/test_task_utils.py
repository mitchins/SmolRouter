import asyncio
import logging

import pytest

from smolrouter.task_utils import create_logged_task


@pytest.mark.asyncio
async def test_create_logged_task_success_invokes_done_callback(caplog):
    callback_events = []

    async def success_task():
        await asyncio.sleep(0)
        return "done"

    def on_done(task: asyncio.Task):
        callback_events.append(task.result())

    with caplog.at_level(logging.ERROR):
        task = create_logged_task(success_task(), task_name="task-utils-success", done_callback=on_done)
        assert task is not None
        assert await task == "done"

    assert callback_events == ["done"]
    assert not any(record.levelname == "ERROR" for record in caplog.records)


@pytest.mark.asyncio
async def test_create_logged_task_exception_is_logged_with_traceback(caplog):
    async def failing_task():
        raise RuntimeError("boom")

    with caplog.at_level(logging.ERROR):
        task = create_logged_task(failing_task(), task_name="task-utils-fail")
        assert task is not None
        try:
            await task
        except RuntimeError:
            pass

    assert any(record.message == "Unhandled exception in task-utils-fail" for record in caplog.records)
    assert any(record.exc_info is not None for record in caplog.records)


@pytest.mark.asyncio
async def test_create_logged_task_done_callback_error_is_logged(caplog):
    async def success_task():
        return "ok"

    def broken_callback(task: asyncio.Task):
        raise ValueError("callback failed")

    with caplog.at_level(logging.ERROR):
        task = create_logged_task(success_task(), task_name="task-utils-callback", done_callback=broken_callback)
        assert task is not None
        assert await task == "ok"

    assert any("Background task completion callback failed for task-utils-callback" in record.message for record in caplog.records)


def test_create_logged_task_no_event_loop_logs_and_closes(caplog):
    async def noop():
        return True

    def fake_create_task(_coro):
        raise RuntimeError("no running event loop")

    with caplog.at_level(logging.WARNING):
        task = create_logged_task(noop(), task_name="task-utils-no-loop", create_task_fn=fake_create_task)
        assert task is None

    assert any("Unable to schedule background task task-utils-no-loop: no running event loop" in record.message for record in caplog.records)
