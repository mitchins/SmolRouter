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
        try:
            if hasattr(coro, "close"):
                coro.close()
        except Exception:
            logger.debug("Failed to close unscheduled coroutine for %s", task_name, exc_info=True)
        return None

    _background_tasks.add(task)
    if service:
        _service_tasks.add(task)

    def _on_done(done: Task[Any]) -> None:
        _background_tasks.discard(done)
        _service_tasks.discard(done)
        try:
            done.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Unhandled exception in %s", task_name)
        finally:
            if done_callback is not None:
                try:
                    done_callback(done)
                except Exception:
                    logger.exception("Background task completion callback failed for %s", task_name)

    task.add_done_callback(_on_done)
    return task


async def _cancel_tasks(tasks: "set[Task[Any]]") -> None:
    pending = {task for task in tasks if not task.done()}
    if not pending:
        return
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


async def drain_background_tasks(timeout: float = 5.0) -> None:
    """Flush fire-and-forget work on shutdown.

    Short-lived tasks (e.g. completion writes) are awaited up to `timeout` so
    in-flight work finishes cleanly; anything still pending is then cancelled so
    shutdown cannot hang. Long-lived service loops are cancelled outright - they
    never finish on their own, so waiting for them would just burn the timeout.
    """
    drainable = {t for t in _background_tasks if not t.done() and t not in _service_tasks}
    if drainable:
        _done, still_pending = await asyncio.wait(drainable, timeout=timeout)
        if still_pending:
            logger.warning(
                "Cancelling %d background task(s) after waiting %.1fs for shutdown",
                len(still_pending),
                timeout,
            )
            await _cancel_tasks(still_pending)

    await _cancel_tasks(set(_service_tasks))
