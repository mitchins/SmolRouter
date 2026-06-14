import asyncio
from asyncio import Task
from typing import Any, Awaitable, Callable, Optional
import logging


logger = logging.getLogger(__name__)

# asyncio only keeps weak references to tasks, so a fire-and-forget task can be
# garbage-collected mid-flight and silently never finish. Hold a strong
# reference here until the task completes, then drop it.
_background_tasks: "set[Task[Any]]" = set()


def create_logged_task(
    coro: Awaitable[Any],
    *,
    task_name: str = "background-task",
    create_task_fn: Optional[Callable[[Awaitable[Any]], Task[Any]]] = None,
    done_callback: Optional[Callable[[Task[Any]], None]] = None,
) -> Optional[Task[Any]]:
    """
    Schedule a background task and attach a done callback that logs any unhandled
    exceptions with full traceback.
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

    def _on_done(done: Task[Any]) -> None:
        _background_tasks.discard(done)
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
