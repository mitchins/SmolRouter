import asyncio
from asyncio import Task
from typing import Any, Awaitable, Callable, Optional
import logging


logger = logging.getLogger(__name__)

# asyncio only keeps weak references to tasks, so a fire-and-forget task can be
# garbage-collected mid-flight and silently never finish. Hold a strong
# reference here until the task completes, then drop it.
_background_tasks: "set[Task[Any]]" = set()
# Long-lived service loops (health/cache-cleanup/janitor). They never finish on
# their own, so shutdown must CANCEL them, not wait for them.
_service_tasks: "set[Task[Any]]" = set()
BACKGROUND_TASK_DRAIN_TIMEOUT_SECONDS = 5.0


def _close_unscheduled_coro(coro: Awaitable[Any], task_name: str) -> None:
    """Best-effort close of a coroutine we never managed to schedule."""
    if not hasattr(coro, "close"):
        return
    try:
        coro.close()
    except Exception:
        logger.debug("Failed to close unscheduled coroutine for %s", task_name, exc_info=True)


def _finalize_task(
    task: Task[Any],
    task_name: str,
    done_callback: Optional[Callable[[Task[Any]], None]],
) -> None:
    """Drop the strong ref, surface any unhandled exception, then run the callback."""
    _background_tasks.discard(task)
    _service_tasks.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Unhandled exception in %s", task_name)
    finally:
        if done_callback is not None:
            try:
                done_callback(task)
            except Exception:
                logger.exception("Background task completion callback failed for %s", task_name)


def create_logged_task(
    coro: Awaitable[Any],
    *,
    task_name: str = "background-task",
    create_task_fn: Optional[Callable[[Awaitable[Any]], Task[Any]]] = None,
    done_callback: Optional[Callable[[Task[Any]], None]] = None,
    service: bool = False,
) -> Optional[Task[Any]]:
    """
    Schedule a background task and attach a done callback that logs any unhandled
    exceptions with full traceback.

    service=True marks a long-lived loop that shutdown should cancel rather than
    wait for (see drain_background_tasks).
    """
    create_task = create_task_fn or asyncio.create_task
    try:
        task = create_task(coro)
    except RuntimeError:
        # No running loop available (e.g. during import-time bootstrapping).
        logger.warning("Unable to schedule background task %s: no running event loop", task_name)
        _close_unscheduled_coro(coro, task_name)
        return None

    _background_tasks.add(task)
    if service:
        _service_tasks.add(task)

    task.add_done_callback(lambda done: _finalize_task(done, task_name, done_callback))
    return task


async def _cancel_tasks(tasks: "set[Task[Any]]", *, wait: bool = True) -> None:
    pending = {task for task in tasks if not task.done()}
    if not pending:
        return
    for task in pending:
        task.cancel()
    if wait:
        await asyncio.gather(*pending, return_exceptions=True)


async def _wait_for_tasks_with_timeout(
    tasks: "set[Task[Any]]", *, timeout_seconds: float
) -> "set[Task[Any]]":
    if not tasks:
        return set()

    try:
        async with asyncio.timeout(timeout_seconds):
            _done, still_pending = await asyncio.wait(tasks)
            return still_pending
    except TimeoutError:
        return {task for task in tasks if not task.done()}


async def _cancel_service_tasks() -> None:
    await _cancel_tasks(set(_service_tasks), wait=False)


async def drain_background_tasks() -> None:
    """Flush fire-and-forget work on shutdown.

    Short-lived tasks (e.g. completion writes) are awaited up to a fixed
    timeout so in-flight work finishes cleanly; anything still pending is then
    cancelled so shutdown cannot hang. Long-lived service loops are cancelled
    outright - they never finish on their own, so waiting for them would just
    burn the timeout.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + BACKGROUND_TASK_DRAIN_TIMEOUT_SECONDS
    still_pending: "set[Task[Any]]" = set()

    while True:
        drainable = {t for t in _background_tasks if not t.done() and t not in _service_tasks}
        if not drainable:
            break

        remaining = deadline - loop.time()
        if remaining <= 0:
            still_pending = drainable
            break

        still_pending = await _wait_for_tasks_with_timeout(drainable, timeout_seconds=remaining)
        if still_pending:
            break

    if still_pending:
        logger.warning(
            "Cancelling %d background task(s) after waiting %.1fs for shutdown",
            len(still_pending),
            BACKGROUND_TASK_DRAIN_TIMEOUT_SECONDS,
        )
        await _cancel_tasks(still_pending, wait=False)

    await _cancel_service_tasks()
