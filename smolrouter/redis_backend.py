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


UTC_OFFSET_SUFFIX = "+00:00"
REDIS_REQUESTS_BY_TIME_KEY = "requests:by_time"
REDIS_EMPTY_VALUES = ("", "None", None)
REDIS_STATUS_NONE_VALUES = ("", "0", "None", None)


def _normalize_int(value: Any, default: Optional[int] = None, none_values: tuple[Any, ...] = REDIS_EMPTY_VALUES) -> Optional[int]:
    if value in none_values:
        return default

    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _normalize_status_code(value: Any) -> str | int | None:
    if value == "pending":
        return "pending"

    return _normalize_int(value, default=None, none_values=REDIS_STATUS_NONE_VALUES)


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).lower() in ("1", "true", "yes")


def _normalize_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None

    try:
        normalized = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", UTC_OFFSET_SUFFIX))
    except (ValueError, TypeError, AttributeError):
        return None

    if normalized.tzinfo is None:
        return normalized.replace(tzinfo=timezone.utc)
    return normalized


def _load_blob_body(blob_storage: Any, key: Any) -> Any:
    if not key:
        return None
    return blob_storage.retrieve(key)


class LogRecord:
    """Object wrapper for Redis log data to provide attribute access"""

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            setattr(self, key, value)

        self.duration_ms = _normalize_int(getattr(self, "duration_ms", None))
        self.status_code = _normalize_status_code(getattr(self, "status_code", None))
        self.request_size = _normalize_int(getattr(self, "request_size", None), default=0) or 0
        self.response_size = _normalize_int(getattr(self, "response_size", None), default=0) or 0
        self.prompt_tokens = _normalize_int(getattr(self, "prompt_tokens", None))
        self.completion_tokens = _normalize_int(getattr(self, "completion_tokens", None))
        self.total_tokens = _normalize_int(getattr(self, "total_tokens", None))

        self.id = getattr(self, "request_id", None)

        self.timestamp = _normalize_datetime(getattr(self, "created_at", None)) or datetime.now(timezone.utc)
        self.completed_at = _normalize_datetime(getattr(self, "completed_at", None))
        self.upstream_url = getattr(self, "upstream_url", "")
        self.error_message = getattr(self, "error_message", None)
        self.duplicate_count = _normalize_int(getattr(self, "duplicate_count", None), default=0) or 0
        self.is_duplicate = _normalize_bool(getattr(self, "is_duplicate", False), default=False)

        # Retrieve request/response bodies from blob storage if keys are present
        from .storage import get_blob_storage

        blob_storage = get_blob_storage()
        self.request_body = _load_blob_body(blob_storage, getattr(self, "request_body_key", None))
        self.response_body = _load_blob_body(blob_storage, getattr(self, "response_body_key", None))

    def items(self):
        """Provide dict-like items() method for backward compatibility"""
        return self.__dict__.items()

    def get(self, key, default=None):
        """Provide dict-like get() method for backward compatibility"""
        return getattr(self, key, default)


class QuotaRecord:
    """Object wrapper for Redis quota data to provide attribute access"""

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            setattr(self, key, value)

        self.requests_today = _normalize_int(getattr(self, "requests_today", 0), default=0) or 0
        self.tokens_today = _normalize_int(getattr(self, "tokens_today", 0), default=0) or 0
        self.error_count = _normalize_int(getattr(self, "error_count", 0), default=0) or 0
        self.last_error = getattr(self, "last_error", None)
        self.last_reset_date = getattr(self, "last_reset_date", None)
        self.api_key_hash = getattr(self, "api_key_hash", data.get("key_hash", ""))
        self.model_name = getattr(self, "model_name", data.get("model_name", ""))
        self.invalid_key = _normalize_bool(getattr(self, "invalid_key", False), default=False)
        self.updated_at = _normalize_datetime(getattr(self, "updated_at", None))
        self.quota_exhausted_at = _normalize_datetime(getattr(self, "quota_exhausted_at", None))

    def mark_request_success(self, tokens: int = 0):
        """Update local object state after successful request

        WARNING: This ONLY updates the in-memory object. You MUST call
        RedisApiKeyQuota.increment_usage() separately to persist to Redis.
        """
        self.requests_today = getattr(self, "requests_today", 0) + 1
        self.tokens_today = getattr(self, "tokens_today", 0) + tokens

    def mark_request_failure(self, error: str = None, quota_exhausted: bool = False):
        """Update local object state after failed request

        WARNING: This ONLY updates the in-memory object. Persistence must be handled separately.
        """
        self.error_count = getattr(self, "error_count", 0) + 1
        if error:
            self.last_error = error
        if quota_exhausted:
            self.quota_exhausted_at = datetime.now(timezone.utc)

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
    -- Clear quota exhaustion status on daily reset
    redis.call('HDEL', quota_key, 'quota_exhausted_at')
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

# Precompute script SHA for EVALSHA optimization. Redis script SHAs are protocol identifiers,
# not security hashes, so mark this accordingly for security scanners.
QUOTA_UPDATE_SHA = hashlib.sha1(QUOTA_UPDATE_SCRIPT.encode(), usedforsecurity=False).hexdigest()


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
        provider_id: Optional[str] = None,
        request_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        auth_user: Optional[str] = None,
        request_size: Optional[int] = None,
        request_body_hash: Optional[str] = None,
        # Optional completion fields (for tests and backfills)
        duration_ms: Optional[int] = None,
        response_size: Optional[int] = None,
        status_code: Optional[any] = None,
        error_message: Optional[str] = None,
        completed_at: Optional[datetime] = None,
        timestamp: Optional[datetime] = None,
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
            "provider_id": provider_id or "",
            "user_agent": user_agent or "",
            "auth_user": auth_user or "",
            "request_size": str(request_size or 0),
            "created_at": (timestamp or datetime.now(timezone.utc)).isoformat(),
            "completed_at": completed_at.isoformat() if completed_at else "",
            "status_code": status_code if status_code is not None else "pending",
            "response_size": str(response_size or 0),
            "duration_ms": str(duration_ms or 0),
            "error_message": error_message or "",
        }

        # Store in Redis hash
        await client.hset(f"request:{request_id}", mapping=request_data)

        # Add to request index for queries
        ts = (timestamp or datetime.now(timezone.utc)).timestamp()
        await client.zadd(REDIS_REQUESTS_BY_TIME_KEY, {request_id: ts})

        # Add to IP index
        await client.sadd(f"requests:by_ip:{source_ip}", request_id)

        # Index by request body hash and set duplicate flags
        if request_body_hash:
            set_key = f"requests:by_body:{request_body_hash}"
            try:
                existing_count = await client.scard(set_key)
            except Exception:
                existing_count = 0
            is_dup = existing_count > 0
            await client.hset(
                f"request:{request_id}",
                mapping={
                    "request_body_hash": request_body_hash,
                    "is_duplicate": "true" if is_dup else "false",
                    # Number of other matching requests (before adding this one)
                    "duplicate_count": existing_count,
                },
            )
            await client.sadd(set_key, request_id)

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
        api_key_suffix: Optional[str] = None,
        proxy_used: Optional[str] = None,
        provider_id: Optional[str] = None,
        api_key_index: Optional[int] = None,
        api_key_total: Optional[int] = None,
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

        # Store metadata if provided
        if api_key_suffix:
            update_data["api_key_suffix"] = api_key_suffix

        if proxy_used:
            update_data["proxy_used"] = proxy_used

        if provider_id:
            update_data["provider_id"] = provider_id

        if api_key_index is not None:
            update_data["api_key_index"] = api_key_index

        if api_key_total is not None:
            update_data["api_key_total"] = api_key_total

        await client.hset(f"request:{request_id}", mapping=update_data)
        logger.debug(f"Updated Redis request completion: {request_id}")

    @staticmethod
    async def update_request_body_key(request_id: str, request_body_key: str) -> None:
        """Attach a stored request body blob key without marking the request complete."""
        client = await get_redis()
        await client.hset(
            f"request:{request_id}",
            mapping={
                "request_body_key": request_body_key,
            },
        )
        logger.debug(f"Updated Redis request body key: {request_id}")

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
        request_ids = await client.zrevrange(REDIS_REQUESTS_BY_TIME_KEY, 0, limit - 1)

        # Get request data
        requests = []
        for request_id in request_ids:
            data = await client.hgetall(f"request:{request_id}")
            if data:
                requests.append(LogRecord(dict(data)))

        return requests

    @staticmethod
    async def get_by_source_ip(source_ip: str, limit: Optional[int] = None) -> List[LogRecord]:
        """Get requests for a specific source IP ordered by recency."""
        client = await get_redis()
        request_ids = [str(request_id) for request_id in await client.smembers(f"requests:by_ip:{source_ip}")]

        requests = []
        for request_id in request_ids:
            data = await client.hgetall(f"request:{request_id}")
            if data:
                requests.append(LogRecord(dict(data)))

        requests.sort(
            key=lambda log: getattr(getattr(log, "timestamp", None), "timestamp", lambda: 0)(),
            reverse=True,
        )

        if limit is None:
            return requests

        return requests[:limit]

    @staticmethod
    async def get_duplicate_request_ids(body_hash: str) -> List[str]:
        """Return all request IDs that share the given request body hash."""
        client = await get_redis()
        set_key = f"requests:by_body:{body_hash}"
        try:
            ids = await client.smembers(set_key)
            # Normalize to list of strings
            return [str(x) for x in ids]
        except Exception:
            return []

    @staticmethod
    async def get_recent_duplicate_request_ids(body_hash: str, limit: int = 10, max_scan: int = 1000) -> List[str]:
        """Return up to `limit` duplicate request IDs ordered by recency.

        Intersects the duplicate set with the global recency index.
        """
        client = await get_redis()
        set_key = f"requests:by_body:{body_hash}"
        try:
            dup_ids = {str(x) for x in await client.smembers(set_key)}
            if not dup_ids:
                return []
            recent_ids = [str(x) for x in await client.zrevrange(REDIS_REQUESTS_BY_TIME_KEY, 0, max_scan - 1)]
            ordered = [rid for rid in recent_ids if rid in dup_ids]
            return ordered[:limit]
        except Exception:
            return []


class RedisApiKeyQuota:
    """Redis-based API key quota tracking with atomic counters"""

    _script_sha: Optional[str] = None  # Class variable to store loaded script SHA
    _script_initialized: bool = False
    _lua_disabled: bool = False  # When true, use pipeline fallback (e.g., FakeRedis/tests)

    @classmethod
    async def initialize_lua_script(cls):
        """Load Lua script on startup. MUST succeed or raise exception.

        This is called during application startup and WILL CRASH THE SERVER
        if the script cannot be loaded. No silent failures, no fallbacks.
        """
        if cls._script_initialized:
            logger.warning("Lua script already initialized, skipping re-initialization")
            return

        # Allow explicit opt-out or FakeRedis fallback without crashing
        if os.getenv("REDIS_DISABLE_LUA", "false").lower() in ("1", "true", "yes") or is_fake_redis():
            cls._lua_disabled = True
            logger.warning("Lua disabled (REDIS_DISABLE_LUA or FakeRedis detected) - using pipeline fallback")
            return

        redis_client = await get_redis()

        try:
            cls._script_sha = await redis_client.script_load(QUOTA_UPDATE_SCRIPT)
            cls._script_initialized = True
            logger.info(f"✅ Lua script loaded successfully: {cls._script_sha}")
        except Exception as e:
            logger.critical(f"❌ FATAL: Failed to load Lua script into Redis: {e}")
            logger.critical("⚠️  Server CANNOT start without functional quota tracking")
            raise RuntimeError(f"Cannot start - Lua script loading failed: {e}") from e

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
        # Use Pacific timezone for Google API quota resets (midnight Pacific)
        import pytz

        pacific_tz = pytz.timezone("US/Pacific")
        today = datetime.now(pacific_tz).strftime("%Y-%m-%d")
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

    @classmethod
    async def _unsafe_increment_usage(
        cls,
        api_key: str,
        provider_id: str,
        model_name: str,
        request_count: int = 1,
        token_count: int = 0,
    ) -> Dict[str, Any]:
        """Internal method without circuit breaker protection

        RAISES if Lua script not initialized - no fallbacks, no excuses.
        """
        if not cls._script_initialized or cls._script_sha is None:
            # In tests/FakeRedis or when explicitly disabled, use pipeline fallback
            if (
                cls._lua_disabled
                or is_fake_redis()
                or os.getenv("REDIS_DISABLE_LUA", "false").lower() in ("1", "true", "yes")
            ):
                # Defer to fallback path implemented below in the exception handler
                pass
            else:
                error_msg = "❌ FATAL: Lua script not initialized - call initialize_lua_script() during startup"
                logger.critical(error_msg)
                raise RuntimeError(error_msg)

        client = await get_redis()

        key_hash = RedisApiKeyQuota.hash_api_key(api_key)
        quota_key = f"quota:{provider_id}:{key_hash}:{model_name}"
        # Use Pacific timezone for Google API quota resets (midnight Pacific)
        import pytz

        pacific_tz = pytz.timezone("US/Pacific")
        today = datetime.now(pacific_tz).strftime("%Y-%m-%d")
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # Use pre-loaded script SHA - NO FALLBACK
            result = await client.evalsha(
                cls._script_sha,
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
            # FakeRedis/compat fallback ONLY (for tests or when Lua disabled)
            if ("unknown command" in str(e).lower() or "noscript" in str(e).lower() or cls._lua_disabled) and (
                is_fake_redis() or cls._lua_disabled
            ):
                logger.warning("⚠️  FakeRedis detected - using pipeline fallback for tests only")
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
                    # Clear quota exhaustion status on daily reset
                    await client.hdel(quota_key, "quota_exhausted_at")

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
                # NO OTHER FALLBACKS - if script fails in production, CRASH LOUDLY
                logger.critical(f"❌ FATAL: Lua script execution failed: {e}")
                logger.critical(f"⚠️  Script SHA: {cls._script_sha}")
                logger.critical(f"⚠️  Quota key: {quota_key}")
                raise RuntimeError(f"Quota tracking BROKEN - Lua script failed: {e}") from e

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

    @staticmethod
    async def mark_invalid(api_key_hash: str, provider_name: str) -> int:
        """Mark all quota entries for a given key hash as invalid.

        Args:
            api_key_hash: The hashed API key
            provider_name: The provider name

        Returns:
            Number of quota entries marked as invalid
        """
        client = await get_redis()

        # Get all quota keys for this key hash
        quota_keys = await client.smembers(f"quotas:by_key:{api_key_hash}")

        count = 0
        for quota_key in quota_keys:
            # quota_key format: "quota:{provider_id}:{key_hash}:{model_name}"
            # We need to match keys that have this provider_name in the right position
            parts = quota_key.split(":")
            if len(parts) >= 4 and parts[0] == "quota" and parts[1] == provider_name:
                # This quota key matches both the key_hash (already filtered) and provider_name
                await client.hset(quota_key, "invalid_key", "true")
                await client.hset(quota_key, "updated_at", datetime.now(timezone.utc).isoformat())
                count += 1

        logger.debug(f"Marked {count} quota entries as invalid for key_hash={api_key_hash}, provider={provider_name}")
        return count

    @staticmethod
    async def mark_quota_exhausted(api_key: str, provider_id: str, model_name: str, error: str = None) -> None:
        """Mark a quota entry as exhausted and persist to Redis.

        Args:
            api_key: The API key
            provider_id: The provider ID
            model_name: The model name
            error: Optional error message to store
        """
        client = await get_redis()

        key_hash = RedisApiKeyQuota.hash_api_key(api_key)
        quota_key = f"quota:{provider_id}:{key_hash}:{model_name}"

        # Update fields in Redis
        updates = {
            "quota_exhausted_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if error:
            updates["last_error"] = error
            # Increment error count
            await client.hincrby(quota_key, "error_count", 1)

        await client.hset(quota_key, mapping=updates)

        logger.debug(f"Marked quota as exhausted: {quota_key}")

    @staticmethod
    async def mark_error(api_key: str, provider_id: str, model_name: str, error: str = None) -> None:
        """Mark an error for a quota entry and persist to Redis.

        Args:
            api_key: The API key
            provider_id: The provider ID
            model_name: The model name
            error: Error message to store
        """
        client = await get_redis()

        key_hash = RedisApiKeyQuota.hash_api_key(api_key)
        quota_key = f"quota:{provider_id}:{key_hash}:{model_name}"

        # Update fields in Redis
        updates = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        if error:
            updates["last_error"] = error

        # Increment error count
        await client.hincrby(quota_key, "error_count", 1)
        await client.hset(quota_key, mapping=updates)

        logger.debug(f"Marked error for quota: {quota_key}, error={error[:100] if error else 'N/A'}")


# Production-hardened safe wrapper for quota operations
async def _safe_increment_usage(
    api_key: str,
    provider_id: str,
    model_name: str,
    request_count: int = 1,
    token_count: int = 0,
) -> Dict[str, Any]:
    """Safe quota increment with circuit breaker protection

    RAISES exceptions if Redis is unavailable - quota tracking is CRITICAL,
    not optional. Callers must handle failures appropriately.
    """

    # Check circuit breaker state
    if not _circuit_breaker.should_attempt():
        error_msg = "Redis circuit breaker OPEN - quota tracking unavailable"
        logger.error(error_msg)
        raise ConnectionError(error_msg)

    try:
        # Attempt Redis operation
        result = await RedisApiKeyQuota._unsafe_increment_usage(
            api_key, provider_id, model_name, request_count, token_count
        )
        _circuit_breaker.record_success()
        return result

    except (ConnectionError, TimeoutError) as e:
        # Record failure and re-raise - this is a critical error
        _circuit_breaker.record_failure()
        logger.error(f"Redis quota increment FAILED: {e}")
        raise

    except Exception as e:
        # Unexpected errors - log and re-raise
        logger.error(f"Unexpected Redis error during quota increment: {e}")
        raise


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
