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

logger = logging.getLogger("smolrouter.database")

# Configuration
MAX_AGE_DAYS = int(os.getenv("MAX_LOG_AGE_DAYS", "7"))  # Auto-purge logs older than N days (0 = disabled)

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
        self.duration_ms = None
        self.status_code = None
        self.error_message = None
        self.timestamp = datetime.now()
        self.request_body = None
        self.response_body = None
        self.user_agent = kwargs.get("user_agent", "Unknown")
        self.request_size = 0
        self.response_size = 0
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None
        self.completed_at = None
        self.auth_user = kwargs.get("auth_user")
        self.request_body_key = None
        self.response_body_key = None

    def save(self):
        """Update Redis with completion data when request is finished"""
        if hasattr(self, "completed_at") and self.completed_at:
            import asyncio
            from .storage import get_blob_storage

            # Store response body in blob storage (request body was stored at request start)
            request_body_key = getattr(self, "request_body_key", None)  # Use existing key if already stored
            response_body_key = None

            blob_storage = get_blob_storage()

            # Only store request body if it wasn't already stored at request start
            if not request_body_key and hasattr(self, "request_body") and self.request_body:
                request_body_key = blob_storage.store(self.request_body, content_type="application/json")
                self.request_body_key = request_body_key

            # Always store response body on completion
            if hasattr(self, "response_body") and self.response_body:
                response_body_key = blob_storage.store(self.response_body, content_type="application/json")
                self.response_body_key = response_body_key

            try:
                # Create async task to update completion data
                asyncio.create_task(
                    RedisRequestLog.update_completion(
                        request_id=self.request_id,
                        status_code=getattr(self, "status_code", 200),
                        response_size=getattr(self, "response_size", 0),
                        error_message=getattr(self, "error_message", None),
                        duration_ms=getattr(self, "duration_ms", None),
                        prompt_tokens=getattr(self, "prompt_tokens", None),
                        completion_tokens=getattr(self, "completion_tokens", None),
                        total_tokens=getattr(self, "total_tokens", None),
                        request_body_key=request_body_key,
                        response_body_key=response_body_key,
                    )
                )
            except RuntimeError:
                # No event loop running - run directly
                import asyncio

                asyncio.run(
                    RedisRequestLog.update_completion(
                        request_id=self.request_id,
                        status_code=getattr(self, "status_code", 200),
                        response_size=getattr(self, "response_size", 0),
                        error_message=getattr(self, "error_message", None),
                        duration_ms=getattr(self, "duration_ms", None),
                        prompt_tokens=getattr(self, "prompt_tokens", None),
                        completion_tokens=getattr(self, "completion_tokens", None),
                        total_tokens=getattr(self, "total_tokens", None),
                        request_body_key=request_body_key,
                        response_body_key=response_body_key,
                    )
                )

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
            "request_id": "request_id",
            "user_agent": "user_agent",
            "auth_user": "auth_user",
            "request_size": "request_size",
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
            break
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

        return {
            "total_requests": total_requests,
            "completed_requests": completed_requests,
            "pending_requests": pending_requests,
            "service_types": service_types,
        }
    except Exception as e:
        logger.error(f"Error getting log stats: {e}")
        return {"total_requests": 0, "completed_requests": 0, "pending_requests": 0, "service_types": {}}


async def get_inflight_requests():
    """Get in-flight (pending) requests from the last 60 minutes"""
    try:
        from datetime import datetime, timedelta

        all_recent = await RequestLog.get_recent(1000)
        cutoff_time = datetime.now() - timedelta(minutes=60)

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


def estimate_tokens_from_request(request_data: dict) -> int:
    """Extract token count from request data"""
    if not request_data:
        return 0

    total_tokens = 0

    # Handle different request formats
    if "messages" in request_data:  # Chat completions
        for message in request_data.get("messages", []):
            content = message.get("content", "")
            if isinstance(content, str):
                total_tokens += estimate_token_count(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total_tokens += estimate_token_count(item.get("text", ""))

    elif "prompt" in request_data:  # Legacy completions
        prompt = request_data.get("prompt", "")
        if isinstance(prompt, str):
            total_tokens += estimate_token_count(prompt)
        elif isinstance(prompt, list):
            for p in prompt:
                total_tokens += estimate_token_count(str(p))

    return total_tokens


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
