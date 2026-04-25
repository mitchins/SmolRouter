"""
Redis-only database backend for SmolRouter.

This module provides high-performance database operations using Redis,
eliminating SQLite complexity and tech debt.
"""

import os
import logging
import asyncio
from datetime import datetime

from .redis_backend import (
    RequestLog as RedisRequestLog,
    ApiKeyQuota as RedisApiKeyQuota,
    init_redis_db,
    close_redis_db,
    get_redis_stats,
)
from .redis_config import redis_client

logger = logging.getLogger("smolrouter.database")

# Configuration
MAX_AGE_DAYS = int(os.getenv("MAX_LOG_AGE_DAYS", "7"))  # Auto-purge logs older than N days (0 = disabled)
JSON_BLOB_CONTENT_TYPE = "application/json"

# Global cleanup task
_cleanup_task = None


class RequestLogEntry:
    """Compatible request log entry object for app.py integration"""

    def __init__(self, request_id: str, **kwargs):
        self.id = request_id
        self.request_id = request_id
        self.original_model = kwargs.get("original_model")
        self.mapped_model = kwargs.get("mapped_model")
        self.upstream_url = kwargs.get("upstream_url")
        self.service_type = kwargs.get("service_type")
        self.source_ip = kwargs.get("source_ip")
        self.method = kwargs.get("method")
        self.path = kwargs.get("path")
        self.duration_ms = kwargs.get("duration_ms")
        self.status_code = kwargs.get("status_code")
        self.error_message = kwargs.get("error_message")
        self.timestamp = kwargs.get("timestamp") or datetime.now()
        self.request_body = None
        self.response_body = None
        self.user_agent = kwargs.get("user_agent", "Unknown")
        self.request_size = kwargs.get("request_size", 0)
        self.response_size = kwargs.get("response_size", 0)
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None
        self.completed_at = kwargs.get("completed_at")
        self.auth_user = kwargs.get("auth_user")
        self.request_body_key = None
        self.response_body_key = None
        self.api_key_suffix = kwargs.get("api_key_suffix")  # pragma: allowlist secret
        self.proxy_used = kwargs.get("proxy_used")
        self.provider_id = kwargs.get("provider_id")  # Downstream provider that handled request
        self.api_key_index = kwargs.get("api_key_index")  # Position in key pool (1-based)
        self.api_key_total = kwargs.get("api_key_total")  # Total keys in pool
        # Duplicate detection fields (optional)
        self.request_body_hash = kwargs.get("request_body_hash")
        self.is_duplicate = kwargs.get("is_duplicate", False)
        self.duplicate_count = kwargs.get("duplicate_count", 0)

    def _has_completion_data(self) -> bool:
        return bool(getattr(self, "completed_at", None))

    async def _store_request_body_if_needed(self, blob_storage) -> str | None:
        request_body_key = getattr(self, "request_body_key", None)
        request_body_bytes = getattr(self, "request_body", None)

        if request_body_key or not request_body_bytes:
            return request_body_key

        request_body_key = await asyncio.to_thread(
            blob_storage.store,
            request_body_bytes,
            content_type=JSON_BLOB_CONTENT_TYPE,
        )
        self.request_body_key = request_body_key
        return request_body_key

    async def _store_response_body_if_present(self, blob_storage) -> str | None:
        response_body_bytes = getattr(self, "response_body", None)
        if not response_body_bytes:
            return None

        response_body_key = await asyncio.to_thread(
            blob_storage.store,
            response_body_bytes,
            content_type=JSON_BLOB_CONTENT_TYPE,
        )
        self.response_body_key = response_body_key
        return response_body_key

    def _completion_update_kwargs(self, request_body_key: str | None, response_body_key: str | None) -> dict:
        return {
            "request_id": self.request_id,
            "status_code": getattr(self, "status_code", 200),
            "response_size": getattr(self, "response_size", 0),
            "error_message": getattr(self, "error_message", None),
            "duration_ms": getattr(self, "duration_ms", None),
            "prompt_tokens": getattr(self, "prompt_tokens", None),
            "completion_tokens": getattr(self, "completion_tokens", None),
            "total_tokens": getattr(self, "total_tokens", None),
            "request_body_key": request_body_key,
            "response_body_key": response_body_key,
            "api_key_suffix": getattr(self, "api_key_suffix", None),
            "proxy_used": getattr(self, "proxy_used", None),
            "provider_id": getattr(self, "provider_id", None),
            "api_key_index": getattr(self, "api_key_index", None),
            "api_key_total": getattr(self, "api_key_total", None),
        }

    async def _store_completion_update(self) -> None:
        from .storage import get_blob_storage
        from .redis_backend import RedisRequestLog

        blob_storage = get_blob_storage()
        request_body_key = await self._store_request_body_if_needed(blob_storage)
        response_body_key = await self._store_response_body_if_present(blob_storage)
        await RedisRequestLog.update_completion(
            **self._completion_update_kwargs(request_body_key, response_body_key)
        )

    async def _run_completion_update(self) -> None:
        try:
            await self._store_completion_update()
        except Exception as e:
            logger.error(f"Failed to store blobs/update completion asynchronously: {e}")

    def _schedule_completion_update(self) -> None:
        try:
            asyncio.create_task(self._run_completion_update())
        except RuntimeError:
            # No event loop running - run synchronously (tests/CLI)
            try:
                asyncio.run(self._run_completion_update())
            except Exception as e:
                logger.error(f"Failed to run async store/update: {e}")

    def save(self):
        """Update Redis with completion data when request is finished"""
        if not self._has_completion_data():
            return

        self._schedule_completion_update()

    async def save_async(self):
        """Async version of save() that awaits blob storage and Redis updates.

        Useful in tests to avoid races with background tasks.
        """
        if not self._has_completion_data():
            return

        await self._store_completion_update()

    def set_request_body(self, body):
        """Store request body for logging"""
        self.request_body = body

    def set_response_body(self, body):
        """Store response body for logging"""
        self.response_body = body


class RequestLog:
    """Redis-based request logging interface"""

    @staticmethod
    async def create(**kwargs):
        """Create request log entry - returns RequestLogEntry object for compatibility"""
        # Map parameters to Redis backend
        redis_kwargs = {}
        param_mapping = {
            "source_ip": "source_ip",
            "method": "method",
            "path": "path",
            "service_type": "service_type",
            "upstream_url": "upstream_url",
            "original_model": "original_model",
            "mapped_model": "mapped_model",
            "provider_id": "provider_id",
            "request_id": "request_id",
            "user_agent": "user_agent",
            "auth_user": "auth_user",
            "request_size": "request_size",
            "request_body_hash": "request_body_hash",
            # Optional completion/test fields
            "duration_ms": "duration_ms",
            "response_size": "response_size",
            "status_code": "status_code",
            "error_message": "error_message",
            "completed_at": "completed_at",
            "timestamp": "timestamp",
        }

        for param_key, redis_key in param_mapping.items():
            if param_key in kwargs:
                redis_kwargs[redis_key] = kwargs[param_key]

        # Create in Redis
        request_id = await RedisRequestLog.create(**redis_kwargs)

        # Return compatible object
        # Remove request_id from kwargs to avoid duplicate parameter error
        entry_kwargs = {k: v for k, v in kwargs.items() if k != "request_id"}
        entry = RequestLogEntry(request_id, **entry_kwargs)
        return entry

    @staticmethod
    async def get_by_id(request_id: str):
        """Get request by ID"""
        return await RedisRequestLog.get_by_id(request_id)

    @staticmethod
    async def get_recent(limit: int = 100):
        """Get recent requests"""
        return await RedisRequestLog.get_recent(limit)

    @staticmethod
    def select():
        """Compatibility method - returns recent requests"""
        # This is a sync method for compatibility, but Redis is async
        # In practice, this should be replaced with async calls
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't run async in running loop - return empty for now
                logger.warning("RequestLog.select() called in running async loop - returning empty")
                return []
            else:
                return asyncio.run(RedisRequestLog.get_recent(100))
        except RuntimeError:
            return asyncio.run(RedisRequestLog.get_recent(100))


class ApiKeyQuota:
    """Redis-based API key quota tracking interface"""

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Create hash of API key for identification"""
        return RedisApiKeyQuota.hash_api_key(api_key)

    @staticmethod
    async def get_or_create(**kwargs):
        """Get or create quota entry"""
        api_key = kwargs.get("api_key")
        provider_id = kwargs.get("provider_id")
        model_name = kwargs.get("model_name")

        if not all([api_key, provider_id, model_name]):
            raise ValueError("api_key, provider_id, and model_name are required")

        return await RedisApiKeyQuota.get_or_create_quota(api_key, provider_id, model_name)

    @staticmethod
    async def get_or_create_quota(api_key: str, provider_id: str, model_name: str):
        """Get or create quota entry (alias for compatibility)"""
        return await RedisApiKeyQuota.get_or_create_quota(api_key, provider_id, model_name)

    @staticmethod
    async def increment_usage(
        api_key: str, provider_id: str, model_name: str, request_count: int = 1, token_count: int = 0
    ):
        """Increment usage counters"""
        return await RedisApiKeyQuota.increment_usage(api_key, provider_id, model_name, request_count, token_count)

    @staticmethod
    async def get_provider_usage(provider_id: str):
        """Get all quota entries for a provider"""
        return await RedisApiKeyQuota.get_provider_usage(provider_id)

    @staticmethod
    def select():
        """Compatibility method - returns all quotas"""
        # This is a sync method for compatibility, but Redis is async
        # In practice, this should be replaced with async calls
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't run async in running loop - return empty for now
                logger.warning("ApiKeyQuota.select() called in running async loop - returning empty")
                return []
            else:
                # Get usage for all providers - simplified for compatibility
                return []
        except RuntimeError:
            return []

    @staticmethod
    async def mark_invalid_by_hash(api_key_hash: str, provider_name: str) -> int:
        """Mark all quota entries for a given key hash as invalid.

        Args:
            api_key_hash: The hashed API key
            provider_name: The provider name

        Returns:
            Number of quota entries marked as invalid
        """
        return await RedisApiKeyQuota.mark_invalid(api_key_hash, provider_name)

    @staticmethod
    async def mark_quota_exhausted(api_key: str, provider_id: str, model_name: str, error: str = None) -> None:
        """Mark a quota entry as exhausted and persist to Redis.

        Args:
            api_key: The API key
            provider_id: The provider ID
            model_name: The model name
            error: Optional error message to store
        """
        return await RedisApiKeyQuota.mark_quota_exhausted(api_key, provider_id, model_name, error)

    @staticmethod
    async def mark_error(api_key: str, provider_id: str, model_name: str, error: str = None) -> None:
        """Mark an error for a quota entry.

        Args:
            api_key: The API key
            provider_id: The provider ID
            model_name: The model name
            error: Error message
        """
        return await RedisApiKeyQuota.mark_error(api_key, provider_id, model_name, error)


# Async database initialization
async def init_database():
    """Initialize Redis database"""
    logger.info("Initializing Redis-only database backend")
    await init_redis_db()

    # Start background cleanup if enabled
    if MAX_AGE_DAYS > 0:
        start_background_cleanup()

    logger.info("Redis database initialization complete")


async def close_database():
    """Close Redis database connections"""
    logger.info("Closing Redis database connections")

    # Stop background cleanup
    stop_background_cleanup()

    await close_redis_db()
    logger.info("Redis database connections closed")


def get_database_stats():
    """Get database statistics"""
    stats = get_redis_stats()
    stats.update(
        {
            "max_log_age_days": MAX_AGE_DAYS,
            "cleanup_enabled": MAX_AGE_DAYS > 0,
        }
    )
    return stats


async def cleanup_old_logs_async(max_age_days: int | None = None) -> int:
    """Delete request logs older than max_age_days from Redis indices and hashes.

    Returns number of deleted entries.
    """
    try:
        client = redis_client
        from datetime import timezone, timedelta

        days = max_age_days if max_age_days is not None else MAX_AGE_DAYS
        if days <= 0:
            return 0

        cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
        # Get old request IDs
        old_ids = await client.zrangebyscore("requests:by_time", 0, cutoff_ts)
        deleted = 0
        for request_id in old_ids:
            key = f"request:{request_id}"
            data = await client.hgetall(key)
            # Remove from IP index if present
            source_ip = data.get("source_ip") if data else None
            if source_ip:
                await client.srem(f"requests:by_ip:{source_ip}", request_id)
            # Delete hash and sorted set entry
            await client.delete(key)
            await client.zrem("requests:by_time", request_id)
            deleted += 1
        return deleted
    except Exception as e:
        logger.error(f"Error during cleanup_old_logs_async: {e}")
        return 0


def cleanup_old_logs(max_age_days: int | None = None) -> int:
    """Sync wrapper for cleanup_old_logs_async for non-async contexts.

    If called within a running event loop, raises RuntimeError to avoid
    unsafe nested event loop usage. Callers in async code should use
    `await cleanup_old_logs_async(...)` instead.
    """
    try:
        asyncio.get_running_loop()
        # If we got here, there is a running event loop
        raise RuntimeError("cleanup_old_logs() called inside a running event loop; use cleanup_old_logs_async()")
    except RuntimeError:
        # No running loop: safe to use asyncio.run
        return asyncio.run(cleanup_old_logs_async(max_age_days))


def vacuum_database() -> bool:
    """No-op for Redis backend; provided for compatibility with legacy tests."""
    return True


# Background cleanup functionality (simplified - Redis TTL handles most of this)
async def background_cleanup_task():
    """Background task to clean up old Redis entries"""
    if MAX_AGE_DAYS <= 0:
        return

    while True:
        try:
            # Sleep for 24 hours
            await asyncio.sleep(24 * 3600)

            # Redis TTL should handle most cleanup, but we can do additional maintenance here
            logger.info(f"Background cleanup cycle (MAX_AGE_DAYS={MAX_AGE_DAYS})")

        except asyncio.CancelledError:
            logger.info("Background cleanup task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in background cleanup: {e}")
            # Continue running despite errors


def start_background_cleanup():
    """Start the background cleanup task"""
    global _cleanup_task
    if MAX_AGE_DAYS > 0:
        try:
            _cleanup_task = asyncio.create_task(background_cleanup_task())
            logger.info(
                f"Started background cleanup task (will run every 24h, purging logs older than {MAX_AGE_DAYS} days)"
            )
        except RuntimeError:
            # No event loop running, this is fine - task will start when FastAPI starts
            logger.info(f"Background cleanup will start with event loop (purging logs older than {MAX_AGE_DAYS} days)")
    else:
        logger.info("Background cleanup disabled (MAX_LOG_AGE_DAYS=0)")


def stop_background_cleanup():
    """Stop the background cleanup task"""
    global _cleanup_task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        logger.info("Stopped background cleanup task")


# Legacy function for app.py compatibility
async def start_request_log(**kwargs):
    """Start request logging (legacy compatibility)"""
    return await RequestLog.create(**kwargs)


# Dashboard and API compatibility functions
async def get_recent_logs(limit: int = 100, service_type: str = None):
    """Get recent logs for dashboard - Redis backend"""
    logs = await RequestLog.get_recent(limit)

    # Filter by service_type if specified
    if service_type:
        logs = [log for log in logs if getattr(log, "service_type", None) == service_type]

    # LogRecord objects from redis_backend already have the right format
    return logs


async def get_log_stats():
    """Get logging statistics"""
    try:
        recent_logs = await RequestLog.get_recent(1000)  # Get larger sample for stats
        total_requests = len(recent_logs)

        # Calculate basic stats
        completed_requests = len([log for log in recent_logs if getattr(log, "status_code", "0") != "0"])
        pending_requests = total_requests - completed_requests

        # Service type breakdown
        service_types = {}
        for log in recent_logs:
            service_type = getattr(log, "service_type", "unknown")
            service_types[service_type] = service_types.get(service_type, 0) + 1

        # Inflight count
        inflight = await get_inflight_requests()

        return {
            "total_requests": total_requests,
            "completed_requests": completed_requests,
            "pending_requests": pending_requests,
            "service_types": service_types,
            "inflight_requests": len(inflight),
        }
    except Exception as e:
        logger.error(f"Error getting log stats: {e}")
        return {"total_requests": 0, "completed_requests": 0, "pending_requests": 0, "service_types": {}}


async def get_inflight_requests():
    """Get in-flight (pending) requests from the last 60 minutes"""
    try:
        from datetime import datetime, timedelta, timezone

        all_recent = await RequestLog.get_recent(1000)
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=60)

        # Filter for pending requests (status_code = "pending" or empty completed_at) within 60 min window
        inflight = [
            log
            for log in all_recent
            if (getattr(log, "status_code", "pending") == "pending" or not getattr(log, "completed_at", None))
            and getattr(log, "timestamp", datetime.now()) >= cutoff_time
        ]
        return inflight
    except Exception as e:
        logger.error(f"Error getting inflight requests: {e}")
        return []


# Token estimation functions (utility functions)
def estimate_token_count(text: str) -> int:
    """Estimate token count for text (rough approximation)"""
    if not text:
        return 0
    # Rough approximation: 1 token ≈ 4 characters for English text
    return max(1, len(text) // 4)


def _estimate_tokens_from_message_content(content) -> int:
    if isinstance(content, str):
        return estimate_token_count(content)

    if not isinstance(content, list):
        return 0

    total_tokens = 0
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            total_tokens += estimate_token_count(item.get("text", ""))
    return total_tokens


def _estimate_tokens_from_messages(messages) -> int:
    if not isinstance(messages, list):
        return 0

    total_tokens = 0
    for message in messages:
        if isinstance(message, dict):
            total_tokens += _estimate_tokens_from_message_content(message.get("content", ""))
    return total_tokens


def _estimate_tokens_from_prompt(prompt) -> int:
    if isinstance(prompt, str):
        return estimate_token_count(prompt)

    if not isinstance(prompt, list):
        return 0

    return sum(estimate_token_count(str(item)) for item in prompt)


def estimate_tokens_from_request(request_data: dict) -> int:
    """Extract token count from request data"""
    if not request_data:
        return 0

    if "messages" in request_data:  # Chat completions
        return _estimate_tokens_from_messages(request_data.get("messages", []))

    if "prompt" in request_data:  # Legacy completions
        return _estimate_tokens_from_prompt(request_data.get("prompt", ""))

    return 0


def extract_tokens_from_openai_response(response_data: dict) -> tuple:
    """Extract token counts from OpenAI response usage data"""
    usage = response_data.get("usage", {})

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    return prompt_tokens, completion_tokens, total_tokens


# Export main classes for compatibility
__all__ = [
    "RequestLog",
    "ApiKeyQuota",
    "RequestLogEntry",
    "init_database",
    "close_database",
    "get_database_stats",
    "start_request_log",
    "get_recent_logs",
    "get_log_stats",
    "get_inflight_requests",
    "estimate_token_count",
    "estimate_tokens_from_request",
    "extract_tokens_from_openai_response",
]
