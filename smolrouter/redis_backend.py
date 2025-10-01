"""
Redis-based backend for high-performance database operations.

Provides the same interface as the existing database layer but uses Redis
for high-throughput concurrent operations, replacing SQLite bottlenecks.
"""

import logging
import os
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from redis.exceptions import ConnectionError, TimeoutError

from .redis_config import redis_client, is_fake_redis, get_redis_status


class LogRecord:
    """Object wrapper for Redis log data to provide attribute access"""

    def __init__(self, data: Dict[str, Any]):
        from datetime import datetime, timezone

        for key, value in data.items():
            setattr(self, key, value)

        # Convert string values to proper types (Redis stores everything as strings)
        # These conversions are critical for template comparisons
        if hasattr(self, "duration_ms") and self.duration_ms:
            try:
                self.duration_ms = int(self.duration_ms) if self.duration_ms not in ("", "None", None) else None
            except (ValueError, TypeError):
                self.duration_ms = None
        else:
            self.duration_ms = None

        if hasattr(self, "status_code") and self.status_code:
            if self.status_code == "pending":
                self.status_code = "pending"  # Keep as string for pending status
            else:
                try:
                    self.status_code = (
                        int(self.status_code) if self.status_code not in ("", "0", "None", None) else None
                    )
                except (ValueError, TypeError):
                    self.status_code = None
        else:
            self.status_code = None

        if hasattr(self, "request_size") and self.request_size:
            try:
                self.request_size = int(self.request_size) if self.request_size not in ("", "None", None) else 0
            except (ValueError, TypeError):
                self.request_size = 0
        else:
            self.request_size = 0

        if hasattr(self, "response_size") and self.response_size:
            try:
                self.response_size = int(self.response_size) if self.response_size not in ("", "None", None) else 0
            except (ValueError, TypeError):
                self.response_size = 0
        else:
            self.response_size = 0

        if hasattr(self, "prompt_tokens") and self.prompt_tokens:
            try:
                self.prompt_tokens = int(self.prompt_tokens) if self.prompt_tokens not in ("", "None", None) else None
            except (ValueError, TypeError):
                self.prompt_tokens = None

        if hasattr(self, "completion_tokens") and self.completion_tokens:
            try:
                self.completion_tokens = (
                    int(self.completion_tokens) if self.completion_tokens not in ("", "None", None) else None
                )
            except (ValueError, TypeError):
                self.completion_tokens = None

        if hasattr(self, "total_tokens") and self.total_tokens:
            try:
                self.total_tokens = int(self.total_tokens) if self.total_tokens not in ("", "None", None) else None
            except (ValueError, TypeError):
                self.total_tokens = None

        # Add compatibility attributes that app.py expects
        self.id = getattr(self, "request_id", None)

        # Parse timestamp from created_at if available - make timezone-aware
        if hasattr(self, "created_at") and self.created_at:
            try:
                self.timestamp = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
                # Ensure it's timezone-aware
                if self.timestamp.tzinfo is None:
                    self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                self.timestamp = datetime.now(timezone.utc)
        else:
            self.timestamp = datetime.now(timezone.utc)

        # Parse completed_at if available - make timezone-aware
        if hasattr(self, "completed_at") and self.completed_at and self.completed_at != "":
            try:
                self.completed_at = datetime.fromisoformat(self.completed_at.replace("Z", "+00:00"))
                # Ensure it's timezone-aware
                if self.completed_at.tzinfo is None:
                    self.completed_at = self.completed_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                self.completed_at = None
        else:
            self.completed_at = None

        # Ensure other expected attributes exist
        if not hasattr(self, "upstream_url"):
            self.upstream_url = ""
        if not hasattr(self, "error_message"):
            self.error_message = None

        # Retrieve request/response bodies from blob storage if keys are present
        from .storage import get_blob_storage

        blob_storage = get_blob_storage()

        if hasattr(self, "request_body_key") and self.request_body_key:
            self.request_body = blob_storage.retrieve(self.request_body_key)
        else:
            self.request_body = None

        if hasattr(self, "response_body_key") and self.response_body_key:
            self.response_body = blob_storage.retrieve(self.response_body_key)
        else:
            self.response_body = None

    def items(self):
        """Provide dict-like items() method for backward compatibility"""
        return self.__dict__.items()

    def get(self, key, default=None):
        """Provide dict-like get() method for backward compatibility"""
        return getattr(self, key, default)


class QuotaRecord:
    """Object wrapper for Redis quota data to provide attribute access"""

    def __init__(self, data: Dict[str, Any]):
        from datetime import datetime, timezone

        for key, value in data.items():
            setattr(self, key, value)

        # Ensure required attributes exist with defaults for Google GenAI provider
        # Also ensure they're the right type (Redis returns strings)
        self.requests_today = int(getattr(self, "requests_today", 0))
        self.tokens_today = int(getattr(self, "tokens_today", 0))
        self.error_count = int(getattr(self, "error_count", 0))

        # Ensure string attributes exist
        if not hasattr(self, "last_error"):
            self.last_error = None
        if not hasattr(self, "last_reset_date"):
            self.last_reset_date = None
        if not hasattr(self, "api_key_hash"):
            self.api_key_hash = data.get("key_hash", "")
        if not hasattr(self, "model_name"):
            self.model_name = data.get("model_name", "")

        # Handle boolean conversion from Redis string
        if hasattr(self, "invalid_key"):
            if isinstance(self.invalid_key, str):
                self.invalid_key = self.invalid_key.lower() == "true"
            elif isinstance(self.invalid_key, bool):
                pass  # Already boolean
            else:
                self.invalid_key = False
        else:
            self.invalid_key = False

        # Parse updated_at timestamp if available - make timezone-aware
        if hasattr(self, "updated_at") and self.updated_at:
            try:
                if isinstance(self.updated_at, str):
                    self.updated_at = datetime.fromisoformat(self.updated_at.replace("Z", "+00:00"))
                    if self.updated_at.tzinfo is None:
                        self.updated_at = self.updated_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError, AttributeError):
                self.updated_at = None
        else:
            self.updated_at = None

        # Parse quota_exhausted_at timestamp if available - make timezone-aware
        if hasattr(self, "quota_exhausted_at") and self.quota_exhausted_at:
            try:
                if isinstance(self.quota_exhausted_at, str) and self.quota_exhausted_at != "":
                    self.quota_exhausted_at = datetime.fromisoformat(self.quota_exhausted_at.replace("Z", "+00:00"))
                    if self.quota_exhausted_at.tzinfo is None:
                        self.quota_exhausted_at = self.quota_exhausted_at.replace(tzinfo=timezone.utc)
                elif not isinstance(self.quota_exhausted_at, datetime):
                    self.quota_exhausted_at = None
            except (ValueError, TypeError, AttributeError):
                self.quota_exhausted_at = None
        else:
            self.quota_exhausted_at = None

    def mark_request_success(self, tokens: int = 0):
        """Mark a successful request (updates will be persisted via increment_usage)"""
        # This is a no-op on the object - actual persistence happens via Redis atomic operations
        pass

    def mark_request_failure(self, error: str = None, quota_exhausted: bool = False):
        """Mark a failed request (updates will be persisted separately)"""
        # This is a no-op on the object - actual persistence happens via Redis atomic operations
        pass

    def items(self):
        """Provide dict-like items() method for backward compatibility"""
        return self.__dict__.items()

    def get(self, key, default=None):
        """Provide dict-like get() method for backward compatibility"""
        return getattr(self, key, default)


logger = logging.getLogger(__name__)


# Circuit breaker for Redis failures (production hardening)
class RedisCircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=30):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def should_attempt(self) -> bool:
        """Check if we should attempt Redis operation"""
        now = datetime.now().timestamp()

        if self.state == "OPEN":
            if now - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
                return True
            return False

        return True

    def record_success(self):
        """Record successful Redis operation"""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        """Record failed Redis operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now().timestamp()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Redis circuit breaker OPEN after {self.failure_count} failures")


# Global circuit breaker instance
_circuit_breaker = RedisCircuitBreaker()


# Atomic Lua script for quota operations (inspired by GPT-5 guidance)
QUOTA_UPDATE_SCRIPT = """
local quota_key = KEYS[1]
local req_count = tonumber(ARGV[1])
local token_count = tonumber(ARGV[2])
local today = ARGV[3]
local timestamp = ARGV[4]

-- Check if we need daily reset
local last_reset = redis.call('HGET', quota_key, 'last_reset')
if last_reset ~= today then
    redis.call('HSET', quota_key, 'requests_today', 0)
    redis.call('HSET', quota_key, 'tokens_today', 0)
    redis.call('HSET', quota_key, 'last_reset', today)
end

-- Atomically increment counters
redis.call('HINCRBY', quota_key, 'requests_today', req_count)
redis.call('HINCRBY', quota_key, 'tokens_today', token_count)
redis.call('HSET', quota_key, 'updated_at', timestamp)

-- Return updated values
return {
    redis.call('HGET', quota_key, 'requests_today'),
    redis.call('HGET', quota_key, 'tokens_today'),
    redis.call('HGET', quota_key, 'last_reset')
}
"""

# Precompute script SHA for EVALSHA optimization
QUOTA_UPDATE_SHA = hashlib.sha1(QUOTA_UPDATE_SCRIPT.encode()).hexdigest()


async def get_redis():
    """Get the global Redis client"""
    return redis_client


class RedisRequestLog:
    """Redis-based request logging with same interface as RequestLog"""

    @staticmethod
    async def create(
        source_ip: str,
        method: str,
        path: str,
        service_type: str = "unknown",
        upstream_url: str = "unknown",
        original_model: Optional[str] = None,
        mapped_model: Optional[str] = None,
        request_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        auth_user: Optional[str] = None,
        request_size: Optional[int] = None,
    ) -> str:
        """Create request log entry - returns request_id"""
        client = await get_redis()

        # Generate request ID if not provided
        if request_id is None:
            request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"

        # Create request data (Redis doesn't handle None values)
        request_data = {
            "request_id": request_id,
            "source_ip": source_ip,
            "method": method,
            "path": path,
            "service_type": service_type,
            "upstream_url": upstream_url,
            "original_model": original_model or "",
            "mapped_model": mapped_model or "",
            "user_agent": user_agent or "",
            "auth_user": auth_user or "",
            "request_size": str(request_size or 0),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": "",  # Empty string instead of None
            "status_code": "pending",  # Mark as pending immediately
            "response_size": "0",
            "error_message": "",
        }

        # Store in Redis hash
        await client.hset(f"request:{request_id}", mapping=request_data)

        # Add to request index for queries
        await client.zadd("requests:by_time", {request_id: datetime.now().timestamp()})

        # Add to IP index
        await client.sadd(f"requests:by_ip:{source_ip}", request_id)

        logger.debug(f"Created Redis request log: {request_id}")
        return request_id

    @staticmethod
    async def update_completion(
        request_id: str,
        status_code: int,
        response_size: Optional[int] = None,
        error_message: Optional[str] = None,
        duration_ms: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        request_body_key: Optional[str] = None,
        response_body_key: Optional[str] = None,
    ):
        """Update request with completion data"""
        client = await get_redis()

        update_data = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status_code": status_code,
            "response_size": response_size or 0,
            "error_message": error_message or "",
            "duration_ms": duration_ms or 0,
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": total_tokens or 0,
        }

        # Store blob keys if provided (bodies are stored in blob storage, not Redis)
        if request_body_key:
            update_data["request_body_key"] = request_body_key

        if response_body_key:
            update_data["response_body_key"] = response_body_key

        await client.hset(f"request:{request_id}", mapping=update_data)
        logger.debug(f"Updated Redis request completion: {request_id}")

    @staticmethod
    async def get_by_id(request_id: str) -> Optional[LogRecord]:
        """Get request by ID"""
        client = await get_redis()
        data = await client.hgetall(f"request:{request_id}")
        return LogRecord(dict(data)) if data else None

    @staticmethod
    async def get_recent(limit: int = 100) -> List[LogRecord]:
        """Get recent requests"""
        client = await get_redis()

        # Get recent request IDs from sorted set
        request_ids = await client.zrevrange("requests:by_time", 0, limit - 1)

        # Get request data
        requests = []
        for request_id in request_ids:
            data = await client.hgetall(f"request:{request_id}")
            if data:
                requests.append(LogRecord(dict(data)))

        return requests


class RedisApiKeyQuota:
    """Redis-based API key quota tracking with atomic counters"""

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Create hash of API key for identification"""
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]

    @staticmethod
    async def get_or_create_quota(
        api_key: str,
        provider_id: str,
        model_name: str,
    ) -> tuple["QuotaRecord", bool]:
        """Get or create quota entry - returns (quota_record, was_created)"""
        client = await get_redis()

        key_hash = RedisApiKeyQuota.hash_api_key(api_key)
        quota_key = f"quota:{provider_id}:{key_hash}:{model_name}"

        # Check if quota exists
        existing = await client.hgetall(quota_key)

        if existing:
            # Convert Redis data back to proper types
            quota_data = dict(existing)
            quota_data["requests_today"] = int(quota_data.get("requests_today", 0))
            quota_data["tokens_today"] = int(quota_data.get("tokens_today", 0))
            quota_data["error_count"] = int(quota_data.get("error_count", 0))
            quota_data["last_reset"] = quota_data.get("last_reset", "")
            quota_data["last_reset_date"] = quota_data.get("last_reset", "")  # Alias for compatibility
            quota_data["invalid_key"] = quota_data.get("invalid_key", "false").lower() == "true"
            return QuotaRecord(quota_data), False

        # Create new quota entry
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        quota_data = {
            "provider_id": provider_id,
            "key_hash": key_hash,
            "model_name": model_name,
            "requests_today": 0,
            "tokens_today": 0,
            "error_count": 0,
            "last_reset": today,
            "last_reset_date": today,  # Alias for compatibility
            "invalid_key": "false",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        await client.hset(quota_key, mapping=quota_data)

        # Add to indices
        await client.sadd(f"quotas:by_provider:{provider_id}", quota_key)
        await client.sadd(f"quotas:by_key:{key_hash}", quota_key)

        logger.debug(f"Created Redis quota entry: {quota_key}")
        return QuotaRecord(quota_data), True

    @staticmethod
    async def increment_usage(
        api_key: str,
        provider_id: str,
        model_name: str,
        request_count: int = 1,
        token_count: int = 0,
    ) -> Dict[str, Any]:
        """Atomically increment usage counters using Lua script with circuit breaker"""
        return await _safe_increment_usage(api_key, provider_id, model_name, request_count, token_count)

    @staticmethod
    async def _unsafe_increment_usage(
        api_key: str,
        provider_id: str,
        model_name: str,
        request_count: int = 1,
        token_count: int = 0,
    ) -> Dict[str, Any]:
        """Internal method without circuit breaker protection"""
        client = await get_redis()

        key_hash = RedisApiKeyQuota.hash_api_key(api_key)
        quota_key = f"quota:{provider_id}:{key_hash}:{model_name}"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # Try EVALSHA first for better performance (real Redis)
            result = await client.evalsha(
                QUOTA_UPDATE_SHA,
                1,
                quota_key,
                str(request_count),
                str(token_count),
                today,
                timestamp,
            )
            # Parse result from Lua script
            requests_today, tokens_today, last_reset = result
        except Exception as e:
            # Fallback to EVAL for NOSCRIPT error
            if "NOSCRIPT" in str(e):
                result = await client.eval(
                    QUOTA_UPDATE_SCRIPT,
                    1,
                    quota_key,
                    str(request_count),
                    str(token_count),
                    today,
                    timestamp,
                )
                requests_today, tokens_today, last_reset = result
            # Fallback for FakeRedis (no Lua script support)
            elif "unknown command" in str(e).lower():
                # Check if we need daily reset
                last_reset = await client.hget(quota_key, "last_reset")
                if last_reset != today:
                    # Reset daily counters
                    await client.hset(
                        quota_key,
                        mapping={
                            "requests_today": 0,
                            "tokens_today": 0,
                            "last_reset": today,
                        },
                    )

                # Atomically increment counters using pipeline
                pipe = client.pipeline()
                pipe.hincrby(quota_key, "requests_today", request_count)
                pipe.hincrby(quota_key, "tokens_today", token_count)
                pipe.hset(quota_key, "updated_at", timestamp)
                results = await pipe.execute()

                requests_today = results[0]
                tokens_today = results[1]
                last_reset = today
            else:
                raise

        # Get full quota data for return
        updated_data = await client.hgetall(quota_key)
        quota_data = dict(updated_data)
        quota_data["requests_today"] = int(requests_today)
        quota_data["tokens_today"] = int(tokens_today)

        return quota_data

    @staticmethod
    async def get_provider_usage(provider_id: str) -> List["QuotaRecord"]:
        """Get all quota entries for a provider"""
        client = await get_redis()

        quota_keys = await client.smembers(f"quotas:by_provider:{provider_id}")

        quotas = []
        for quota_key in quota_keys:
            data = await client.hgetall(quota_key)
            if data:
                quota_data = dict(data)
                quota_data["requests_today"] = int(quota_data.get("requests_today", 0))
                quota_data["tokens_today"] = int(quota_data.get("tokens_today", 0))
                quota_data["error_count"] = int(quota_data.get("error_count", 0))
                quota_data["last_reset_date"] = quota_data.get("last_reset", "")
                quota_data["invalid_key"] = quota_data.get("invalid_key", "false").lower() == "true"
                quotas.append(QuotaRecord(quota_data))

        return quotas


# Production-hardened safe wrapper for quota operations
async def _safe_increment_usage(
    api_key: str,
    provider_id: str,
    model_name: str,
    request_count: int = 1,
    token_count: int = 0,
) -> Dict[str, Any]:
    """Safe quota increment with circuit breaker protection"""

    # Check circuit breaker state
    if not _circuit_breaker.should_attempt():
        logger.warning("Redis circuit breaker OPEN - skipping quota increment")
        # Return empty quota data to avoid blocking request
        return {
            "provider_id": provider_id,
            "key_hash": RedisApiKeyQuota.hash_api_key(api_key),
            "model_name": model_name,
            "requests_today": 0,
            "tokens_today": 0,
            "last_reset": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }

    try:
        # Attempt Redis operation
        result = await RedisApiKeyQuota._unsafe_increment_usage(
            api_key, provider_id, model_name, request_count, token_count
        )
        _circuit_breaker.record_success()
        return result

    except (ConnectionError, TimeoutError) as e:
        # Record failure but don't block the request
        _circuit_breaker.record_failure()
        logger.warning(f"Redis quota increment failed (circuit breaker): {e}")

        # Return fallback quota data
        return {
            "provider_id": provider_id,
            "key_hash": RedisApiKeyQuota.hash_api_key(api_key),
            "model_name": model_name,
            "requests_today": 0,  # Conservative fallback
            "tokens_today": 0,
            "last_reset": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }

    except Exception as e:
        # Unexpected errors - log but don't trip circuit breaker
        logger.error(f"Unexpected Redis error: {e}")
        return {
            "provider_id": provider_id,
            "key_hash": RedisApiKeyQuota.hash_api_key(api_key),
            "model_name": model_name,
            "requests_today": 0,
            "tokens_today": 0,
            "last_reset": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }


# Maintain compatibility with existing code
RequestLog = RedisRequestLog
ApiKeyQuota = RedisApiKeyQuota


async def init_redis_db():
    """Initialize Redis database connection"""
    from .redis_config import redis_startup_check

    client = await get_redis()

    # Use the centralized startup check
    await redis_startup_check(client, is_fake_redis())

    if not is_fake_redis():
        logger.info("Redis database initialized successfully")
    else:
        logger.info("FakeRedis initialized for development")


async def close_redis_db():
    """Close Redis database connection"""
    if hasattr(redis_client, "close"):
        await redis_client.close()
    logger.info("Redis database connection closed")


def get_redis_stats() -> Dict[str, Any]:
    """Get Redis connection statistics"""
    config_status = get_redis_status()
    return {
        "backend": "redis",
        "using_fake": is_fake_redis(),
        "redis_url": config_status["redis_url"],
        "environment": config_status["environment"],
        "max_connections": config_status["max_connections"],
        "connection_active": redis_client is not None,
    }
