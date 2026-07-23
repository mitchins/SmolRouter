"""
Redis-based backend for high-performance database operations.

Provides the same interface as the existing database layer but uses Redis
for high-throughput concurrent operations, replacing SQLite bottlenecks.
"""

import logging
import os
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Mapping, TypedDict, Unpack
from zoneinfo import ZoneInfo

from redis.exceptions import ConnectionError, TimeoutError

from .redis_config import redis_client, is_fake_redis, get_redis_status


UTC_OFFSET_SUFFIX = "+00:00"
PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
REDIS_REQUESTS_BY_TIME_KEY = "requests:by_time"
REDIS_REQUEST_IDENTITY_KEY_PREFIX = "requests:by_identity"
# O(1) dashboard stats: maintained on create/complete instead of scanning records.
STATS_TOTAL_KEY = "stats:requests:total"
STATS_COMPLETED_KEY = "stats:requests:completed"
STATS_FAILED_KEY = "stats:requests:failed"
STATS_SERVICE_TYPES_KEY = "stats:requests:service_types"
INFLIGHT_SET_KEY = "stats:requests:inflight"
REDIS_EMPTY_VALUES = ("", "None", None)
REDIS_STATUS_NONE_VALUES = ("", "0", "None", None)
_BODY_STORAGE_UPDATE_UNSET = object()
GOOGLE_INVALID_KEY_RECOVERY_AUDIT_LIMIT = 100
_COMPLETE_ONCE_LUA_SCRIPT = """
local inflight_key = KEYS[1]
local completed_key = KEYS[2]
local failed_key = KEYS[3]
local request_id = ARGV[1]
local status_code = tonumber(ARGV[2])

local removed = redis.call("SREM", inflight_key, request_id)
if removed == 1 then
    redis.call("INCR", completed_key)
    if status_code and status_code >= 400 then
        redis.call("INCR", failed_key)
    end
end

return removed
"""
_GET_RECENT_LUA_SCRIPT = """
local ids = redis.call("ZRANGE", KEYS[1], 0, tonumber(ARGV[1]) - 1, "REV")
local records = {}

for _, request_id in ipairs(ids) do
    local data = redis.call("HGETALL", "request:" .. request_id)
    if #data > 0 then
        table.insert(records, data)
    end
end

return records
"""
GOOGLE_ROTARY_SELECTOR_LUA_SCRIPT = """
local counter_key = KEYS[1]
local provider_id = ARGV[1]
local model_name = ARGV[2]
local today = ARGV[3]
local now_iso = ARGV[4]
-- Reserved for future predictive heuristics. Rotary selection intentionally
-- does not preempt keys by request count; availability wins until a key is
-- actually cooling down, exhausted for the day, or invalid.
local model_limit = tonumber(ARGV[5]) or 0
local key_count = #ARGV - 5

if key_count <= 0 then
    return {
        "status", "no_keys",
        "selected_index", "",
        "selected_key_hash", "",
        "invalid_count", "0",
        "cooling_down_count", "0",
        "exhausted_count", "0",
        "soonest_cooldown_until", ""
    }
end

local counter = tonumber(redis.call("GET", counter_key) or "0")
local invalid_key_set = "google_invalid_keys:" .. provider_id
local invalid_count = 0
local cooling_down_count = 0
local exhausted_count = 0
local soonest_cooldown_until = ""

for offset = 0, key_count - 1 do
    local idx = (counter + offset) % key_count
    local key_hash = ARGV[6 + idx]
    local quota_key = "quota:" .. provider_id .. ":" .. key_hash .. ":" .. model_name

    if redis.call("SISMEMBER", invalid_key_set, key_hash) == 1 then
        invalid_count = invalid_count + 1
    else
    local invalid_key = redis.call("HGET", quota_key, "invalid_key")
    if invalid_key ~= "true" and invalid_key ~= "1" then
        local last_reset = redis.call("HGET", quota_key, "last_reset")
        if not last_reset or last_reset == "" then
            last_reset = redis.call("HGET", quota_key, "last_reset_date")
        end

        local quota_exhausted_at = redis.call("HGET", quota_key, "quota_exhausted_at")
        local cooldown_until = redis.call("HGET", quota_key, "quota_cooldown_until")
        local reset_is_today = last_reset == today
        local quota_exhausted_today = reset_is_today and quota_exhausted_at and quota_exhausted_at ~= ""
        local is_cooling_down = cooldown_until and cooldown_until ~= "" and cooldown_until > now_iso

        if is_cooling_down then
            cooling_down_count = cooling_down_count + 1
            if soonest_cooldown_until == "" or cooldown_until < soonest_cooldown_until then
                soonest_cooldown_until = cooldown_until
            end
        elseif quota_exhausted_today then
            exhausted_count = exhausted_count + 1
        else
            redis.call("SET", counter_key, tostring(idx + 1))
            return {
                "status", "ok",
                "selected_index", tostring(idx),
                "selected_key_hash", key_hash,
                "invalid_count", tostring(invalid_count),
                "cooling_down_count", tostring(cooling_down_count),
                "exhausted_count", tostring(exhausted_count),
                "soonest_cooldown_until", soonest_cooldown_until
            }
        end
    else
        invalid_count = invalid_count + 1
    end
    end
end

local status = "none_available"
if invalid_count == key_count then
    status = "all_invalid"
end

return {
    "status", status,
    "selected_index", "",
    "selected_key_hash", "",
    "invalid_count", tostring(invalid_count),
    "cooling_down_count", tostring(cooling_down_count),
    "exhausted_count", tostring(exhausted_count),
    "soonest_cooldown_until", soonest_cooldown_until
}
"""
REQUEST_LOG_CREATE_OPTION_KEYS = frozenset(
    {
        "service_type",
        "upstream_url",
        "original_model",
        "mapped_model",
        "provider_id",
        "request_id",
        "user_agent",
        "auth_user",
        "identity_kind",
        "identity_subject_id",
        "identity_display_name",
        "request_size",
        "request_body_hash",
        "duration_ms",
        "response_size",
        "status_code",
        "error_message",
        "completed_at",
        "timestamp",
    }
)
REQUEST_LOG_COMPLETION_OPTION_KEYS = frozenset(
    {
        "response_size",
        "error_message",
        "duration_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "upstream_url",
        "request_body_key",
        "response_body_key",
        "api_key_suffix",
        "proxy_used",
        "provider_id",
        "api_key_index",
        "api_key_total",
    }
)
REQUEST_LOG_COMPLETION_TRUTHY_FIELDS = (
    "upstream_url",
    "request_body_key",
    "response_body_key",
    "api_key_suffix",
    "proxy_used",
    "provider_id",
)
REQUEST_LOG_COMPLETION_NUMERIC_FIELDS = ("api_key_index", "api_key_total")


def _current_pacific_date() -> str:
    return datetime.now(PACIFIC_TZ).strftime("%Y-%m-%d")


def _seconds_until_pacific_midnight(now_utc: Optional[datetime] = None) -> int:
    now = now_utc or datetime.now(timezone.utc)
    now_pacific = now.astimezone(PACIFIC_TZ)
    tomorrow_pacific = (now_pacific + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return int((tomorrow_pacific - now_pacific).total_seconds())


def _to_pacific_datetime(value: datetime, assume_utc: bool = False) -> datetime:
    if value.tzinfo is None:
        if assume_utc:
            return value.replace(tzinfo=timezone.utc).astimezone(PACIFIC_TZ)
        return value.replace(tzinfo=PACIFIC_TZ)

    return value.astimezone(PACIFIC_TZ)


class RequestLogCreateOptions(TypedDict, total=False):
    service_type: str
    upstream_url: str
    original_model: Optional[str]
    mapped_model: Optional[str]
    provider_id: Optional[str]
    request_id: Optional[str]
    user_agent: Optional[str]
    auth_user: Optional[str]
    identity_kind: Optional[str]
    identity_subject_id: Optional[str]
    identity_display_name: Optional[str]
    request_size: Optional[int]
    request_body_hash: Optional[str]
    duration_ms: Optional[int]
    response_size: Optional[int]
    status_code: Optional[Any]
    error_message: Optional[str]
    completed_at: Optional[datetime]
    timestamp: Optional[datetime]


class RequestLogCompletionOptions(TypedDict, total=False):
    response_size: Optional[int]
    error_message: Optional[str]
    duration_ms: Optional[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    upstream_url: Optional[str]
    request_body_key: Optional[str]
    response_body_key: Optional[str]
    api_key_suffix: Optional[str]
    proxy_used: Optional[str]
    provider_id: Optional[str]
    api_key_index: Optional[int]
    api_key_total: Optional[int]


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


def _load_blob_body(blob_storage: Any, key: Any) -> tuple[Any, str]:
    if not key:
        return None, "not_stored"

    if not blob_storage:
        return None, "storage_error"

    try:
        data = blob_storage.retrieve(key)
    except Exception:
        logger.exception("Failed to retrieve blob %s", key)
        return None, "storage_error"

    if data is None:
        return None, "not_found"

    return data, "available"


def _resolve_body_status(load_status: str, archival_status: Any) -> str:
    archival_status_text = str(archival_status) if archival_status not in REDIS_EMPTY_VALUES else ""
    if load_status == "not_stored" and archival_status_text:
        return archival_status_text
    return load_status


def _prepare_quota_record_data(data: Dict[str, Any]) -> Dict[str, Any]:
    quota_data = dict(data)
    last_reset = quota_data.get("last_reset", "")

    if last_reset:
        quota_data["last_reset_date"] = last_reset

    if not quota_data.get("api_key_hash") and quota_data.get("key_hash"):
        quota_data["api_key_hash"] = quota_data["key_hash"]

    quota_data["invalid_key"] = _normalize_bool(quota_data.get("invalid_key", False), default=False)
    return quota_data


def _is_lua_fallback_enabled(lua_disabled: bool) -> bool:
    return lua_disabled or is_fake_redis() or os.getenv("REDIS_DISABLE_LUA", "false").lower() in ("1", "true", "yes")


def _should_use_increment_fallback(error: Exception, lua_disabled: bool) -> bool:
    error_text = str(error).lower()
    return ("unknown command" in error_text or "noscript" in error_text or lua_disabled) and (
        is_fake_redis() or lua_disabled
    )


def _validate_option_names(option_fields: Mapping[str, Any], allowed: frozenset[str], operation: str) -> None:
    unexpected = sorted(set(option_fields) - allowed)
    if unexpected:
        raise TypeError(f"Unexpected {operation} fields: {', '.join(unexpected)}")


def _resolve_request_id(request_id: Optional[str]) -> str:
    if request_id is not None:
        return request_id

    return f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"


def _resolve_created_at(timestamp: Optional[datetime]) -> datetime:
    return timestamp or datetime.now(timezone.utc)


def _build_request_log_data(
    request_id: str,
    source_ip: str,
    method: str,
    path: str,
    created_at: datetime,
    option_fields: RequestLogCreateOptions,
) -> Dict[str, Any]:
    completed_at = option_fields.get("completed_at")
    status_code = option_fields.get("status_code")

    return {
        "request_id": request_id,
        "source_ip": source_ip,
        "method": method,
        "path": path,
        "service_type": option_fields.get("service_type", "unknown"),
        "upstream_url": option_fields.get("upstream_url", "unknown"),
        "original_model": option_fields.get("original_model") or "",
        "mapped_model": option_fields.get("mapped_model") or "",
        "provider_id": option_fields.get("provider_id") or "",
        "user_agent": option_fields.get("user_agent") or "",
        "auth_user": option_fields.get("auth_user") or "",
        "identity_kind": option_fields.get("identity_kind") or "",
        "identity_subject_id": option_fields.get("identity_subject_id") or "",
        "identity_display_name": option_fields.get("identity_display_name") or "",
        "request_size": str(option_fields.get("request_size") or 0),
        "created_at": created_at.isoformat(),
        "completed_at": completed_at.isoformat() if completed_at else "",
        "status_code": status_code if status_code is not None else "pending",
        "response_size": str(option_fields.get("response_size") or 0),
        "duration_ms": str(option_fields.get("duration_ms") or 0),
        "error_message": option_fields.get("error_message") or "",
    }


def _queue_store_request_log(
    pipe: Any,
    request_id: str,
    source_ip: str,
    created_at: datetime,
    request_data: Dict[str, Any],
) -> None:
    pipe.hset(f"request:{request_id}", mapping=request_data)
    pipe.zadd(REDIS_REQUESTS_BY_TIME_KEY, {request_id: created_at.timestamp()})
    pipe.sadd(f"requests:by_ip:{source_ip}", request_id)


def _identity_index_key(identity_kind: str, identity_subject_id: str) -> str:
    return f"{REDIS_REQUEST_IDENTITY_KEY_PREFIX}:{identity_kind}:{identity_subject_id}"


def _request_hash_key(request_id: str) -> str:
    return f"request:{request_id}"


def _queue_store_identity_index(
    pipe: Any,
    request_id: str,
    identity_kind: str,
    identity_subject_id: str,
    created_at: datetime,
) -> None:
    key = _identity_index_key(identity_kind, identity_subject_id)
    pipe.zadd(key, {request_id: created_at.timestamp()})


def _status_is_terminal(status_code: Any) -> bool:
    """A request is terminal (completed) when it has a real HTTP status, not
    'pending'/0/None."""
    if status_code is None:
        return False
    text = str(status_code)
    if text in ("pending", "0", "", "None"):
        return False
    try:
        int(text)
        return True
    except ValueError:
        return False


def _status_is_failure(status_code: Any) -> bool:
    return _status_is_terminal(status_code) and int(str(status_code)) >= 400


def _queue_create_stats(pipe: Any, request_id: str, request_data: Dict[str, Any]) -> None:
    """Maintain O(1) dashboard counters on create (total, per-service, inflight set).

    A request created already-terminal (status_code supplied at create) is counted
    as completed and is NOT added to the inflight set."""
    service_type = request_data.get("service_type") or "unknown"
    status_code = request_data.get("status_code")
    pipe.incr(STATS_TOTAL_KEY)
    pipe.hincrby(STATS_SERVICE_TYPES_KEY, service_type, 1)
    if _status_is_terminal(status_code):
        pipe.incr(STATS_COMPLETED_KEY)
        if _status_is_failure(status_code):
            pipe.incr(STATS_FAILED_KEY)
    else:
        pipe.sadd(INFLIGHT_SET_KEY, request_id)


async def _increment_completion_stats(client: Any, request_id: str, status_code: Any) -> None:
    """Maintain O(1) dashboard counters on completion (completed/failed, remove from inflight)."""
    status_code_value = int(status_code) if _status_is_terminal(status_code) else 0

    if _is_lua_fallback_enabled(False):
        removed = await client.srem(INFLIGHT_SET_KEY, request_id)
        if removed == 1:
            pipe = client.pipeline(transaction=False)
            pipe.incr(STATS_COMPLETED_KEY)
            if _status_is_failure(status_code_value):
                pipe.incr(STATS_FAILED_KEY)
            await pipe.execute()
        return

    await client.eval(
        _COMPLETE_ONCE_LUA_SCRIPT,
        3,
        INFLIGHT_SET_KEY,
        STATS_COMPLETED_KEY,
        STATS_FAILED_KEY,
        request_id,
        str(status_code_value),
    )


async def _check_and_queue_duplicate_request_body(
    client: Any,
    pipe: Any,
    request_id: str,
    request_body_hash: Optional[str],
) -> None:
    """Check duplicate count immediately, then queue duplicate index writes."""
    if not request_body_hash:
        return

    set_key = f"requests:by_body:{request_body_hash}"
    try:
        existing_count = await client.scard(set_key)
    except Exception:
        logger.debug("Failed to get duplicate count for %s", set_key)
        existing_count = 0

    pipe.hset(
        f"request:{request_id}",
        mapping={
            "request_body_hash": request_body_hash,
            "is_duplicate": "true" if existing_count > 0 else "false",
            "duplicate_count": existing_count,
        },
    )
    pipe.sadd(set_key, request_id)


def _build_completion_update_data(status_code: int, option_fields: RequestLogCompletionOptions) -> Dict[str, Any]:
    update_data = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status_code": status_code,
        "response_size": option_fields.get("response_size") or 0,
        "error_message": option_fields.get("error_message") or "",
        "duration_ms": option_fields.get("duration_ms") or 0,
        "prompt_tokens": option_fields.get("prompt_tokens") or 0,
        "completion_tokens": option_fields.get("completion_tokens") or 0,
        "total_tokens": option_fields.get("total_tokens") or 0,
    }

    for field_name in REQUEST_LOG_COMPLETION_TRUTHY_FIELDS:
        value = option_fields.get(field_name)
        if value:
            update_data[field_name] = value

    for field_name in REQUEST_LOG_COMPLETION_NUMERIC_FIELDS:
        value = option_fields.get(field_name)
        if value is not None:
            update_data[field_name] = value

    return update_data


def _flat_pairs_to_dict(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data

    pairs = iter(data)
    return {str(key): value for key, value in zip(pairs, pairs)}


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
        stored_request_body_status = getattr(self, "request_body_status", None)
        stored_response_body_status = getattr(self, "response_body_status", None)
        self.request_body, request_body_status = _load_blob_body(
            blob_storage, getattr(self, "request_body_key", None)
        )
        self.request_body_status = _resolve_body_status(request_body_status, stored_request_body_status)
        self.response_body, response_body_status = _load_blob_body(
            blob_storage, getattr(self, "response_body_key", None)
        )
        self.response_body_status = _resolve_body_status(response_body_status, stored_response_body_status)

    def items(self):
        """Provide dict-like items() method for backward compatibility"""
        return self.__dict__.items()

    def get(self, key, default=None):
        """Provide dict-like get() method for backward compatibility"""
        return getattr(self, key, default)


class QuotaRecord:
    """Object wrapper for Redis quota data to provide attribute access"""

    def __init__(self, data: Dict[str, Any]):
        normalized_data = _prepare_quota_record_data(data)

        for key, value in normalized_data.items():
            setattr(self, key, value)

        self.requests_today = _normalize_int(getattr(self, "requests_today", 0), default=0) or 0
        self.tokens_today = _normalize_int(getattr(self, "tokens_today", 0), default=0) or 0
        self.error_count = _normalize_int(getattr(self, "error_count", 0), default=0) or 0
        self.last_error = getattr(self, "last_error", None)
        self.last_reset_date = getattr(self, "last_reset_date", normalized_data.get("last_reset", ""))
        self.api_key_hash = getattr(self, "api_key_hash", normalized_data.get("key_hash", ""))
        self.model_name = getattr(self, "model_name", normalized_data.get("model_name", ""))
        self.invalid_key = _normalize_bool(getattr(self, "invalid_key", False), default=False)
        self.updated_at = _normalize_datetime(getattr(self, "updated_at", None))
        self.quota_exhausted_at = _normalize_datetime(getattr(self, "quota_exhausted_at", None))
        self.quota_cooldown_until = _normalize_datetime(getattr(self, "quota_cooldown_until", None))

    def mark_request_success(self, tokens: int = 0):
        """Update local object state after successful request

        WARNING: This ONLY updates the in-memory object. You MUST call
        RedisApiKeyQuota.increment_usage() separately to persist to Redis.
        """
        self.requests_today = getattr(self, "requests_today", 0) + 1
        self.tokens_today = getattr(self, "tokens_today", 0) + tokens

    def mark_request_failure(self, error: Optional[str] = None, quota_exhausted: bool = False):
        """Update local object state after failed request

        WARNING: This ONLY updates the in-memory object. Persistence must be handled separately.
        """
        self.error_count = getattr(self, "error_count", 0) + 1
        if error:
            self.last_error = error
        if quota_exhausted:
            self.quota_exhausted_at = datetime.now(timezone.utc)

    def mark_rate_limited(self, cooldown_until: datetime, error: Optional[str] = None):
        """Update local object state after a transient (per-minute) rate limit.

        Unlike mark_request_failure(quota_exhausted=True), this does NOT bench the
        key for the whole Pacific day. It records a short cooldown window after
        which the key returns to the selection pool.

        WARNING: This ONLY updates the in-memory object. Persistence must be handled separately.
        """
        self.quota_cooldown_until = cooldown_until
        if error:
            self.last_error = error

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
    redis.call('HSET', quota_key, 'last_reset_date', today)
    -- Clear quota exhaustion status on daily reset
    redis.call('HDEL', quota_key, 'quota_exhausted_at')
    redis.call('HDEL', quota_key, 'quota_cooldown_until')
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

def get_redis():
    """Get the global Redis client"""
    return redis_client


class RedisRequestLog:
    """Redis-based request logging with same interface as RequestLog"""

    @staticmethod
    async def create(
        source_ip: str,
        method: str,
        path: str,
        **option_fields: Unpack[RequestLogCreateOptions],
    ) -> str:
        """Create request log entry - returns request_id"""
        _validate_option_names(option_fields, REQUEST_LOG_CREATE_OPTION_KEYS, "request log create")

        client = get_redis()
        request_id = _resolve_request_id(option_fields.get("request_id"))
        created_at = _resolve_created_at(option_fields.get("timestamp"))
        request_data = _build_request_log_data(request_id, source_ip, method, path, created_at, option_fields)

        pipe = client.pipeline(transaction=False)
        _queue_store_request_log(pipe, request_id, source_ip, created_at, request_data)
        identity_kind = option_fields.get("identity_kind")
        identity_subject_id = option_fields.get("identity_subject_id")
        if identity_kind and identity_subject_id:
            _queue_store_identity_index(
                pipe,
                request_id=request_id,
                identity_kind=str(identity_kind),
                identity_subject_id=str(identity_subject_id),
                created_at=created_at,
            )
        await _check_and_queue_duplicate_request_body(client, pipe, request_id, option_fields.get("request_body_hash"))
        _queue_create_stats(pipe, request_id, request_data)
        await pipe.execute()

        logger.debug(f"Created Redis request log: {request_id}")
        return request_id

    @staticmethod
    async def update_completion(
        request_id: str,
        status_code: int,
        **option_fields: Unpack[RequestLogCompletionOptions],
    ) -> None:
        """Update request with completion data"""
        _validate_option_names(option_fields, REQUEST_LOG_COMPLETION_OPTION_KEYS, "request completion")

        client = get_redis()
        update_data = _build_completion_update_data(status_code, option_fields)
        if _is_lua_fallback_enabled(False):
            await client.hset(f"request:{request_id}", mapping=update_data)
            await _increment_completion_stats(client, request_id, status_code)
        else:
            status_code_value = int(status_code) if _status_is_terminal(status_code) else 0
            pipe = client.pipeline(transaction=False)
            pipe.hset(f"request:{request_id}", mapping=update_data)
            pipe.eval(
                _COMPLETE_ONCE_LUA_SCRIPT,
                3,
                INFLIGHT_SET_KEY,
                STATS_COMPLETED_KEY,
                STATS_FAILED_KEY,
                request_id,
                str(status_code_value),
            )
            await pipe.execute()
        logger.debug(f"Updated Redis request completion: {request_id}")

    @staticmethod
    async def get_stats_counters() -> Dict[str, Any]:
        """O(1) dashboard counters: a single pipeline of reads, no record scan."""
        client = get_redis()
        pipe = client.pipeline(transaction=False)
        pipe.get(STATS_TOTAL_KEY)
        pipe.get(STATS_COMPLETED_KEY)
        pipe.get(STATS_FAILED_KEY)
        pipe.hgetall(STATS_SERVICE_TYPES_KEY)
        pipe.scard(INFLIGHT_SET_KEY)
        total, completed, failed, service_types, inflight = await pipe.execute()
        return {
            "total": int(total or 0),
            "completed": int(completed or 0),
            "failed": int(failed or 0),
            "service_types": {str(k): int(v) for k, v in (service_types or {}).items()},
            "inflight": int(inflight or 0),
        }

    @staticmethod
    async def update_request_body_key(request_id: str, request_body_key: str) -> None:
        """Attach a stored request body blob key without marking the request complete."""
        await RedisRequestLog.update_body_keys(request_id, request_body_key=request_body_key)
        logger.debug(f"Updated Redis request body key: {request_id}")

    @staticmethod
    async def update_response_body_key(request_id: str, response_body_key: str) -> None:
        """Attach a stored response body blob key without changing completion counters."""
        await RedisRequestLog.update_body_keys(request_id, response_body_key=response_body_key)
        logger.debug(f"Updated Redis response body key: {request_id}")

    @staticmethod
    async def update_body_keys(
        request_id: str,
        request_body_key: Optional[str] = None,
        response_body_key: Optional[str] = None,
    ) -> None:
        """Attach stored request/response body blob keys without changing terminal state."""
        client = get_redis()
        mapping = {}
        if request_body_key:
            mapping["request_body_key"] = request_body_key
        if response_body_key:
            mapping["response_body_key"] = response_body_key
        if not mapping:
            return
        await client.hset(f"request:{request_id}", mapping=mapping)
        logger.debug(f"Updated Redis body keys: {request_id}")

    @staticmethod
    async def update_body_storage_result(
        request_id: str,
        request_body_key: Any = _BODY_STORAGE_UPDATE_UNSET,
        response_body_key: Any = _BODY_STORAGE_UPDATE_UNSET,
        request_body_status: Any = _BODY_STORAGE_UPDATE_UNSET,
        request_body_error: Any = _BODY_STORAGE_UPDATE_UNSET,
        response_body_status: Any = _BODY_STORAGE_UPDATE_UNSET,
        response_body_error: Any = _BODY_STORAGE_UPDATE_UNSET,
    ) -> None:
        """Persist blob keys and any archival failure details."""
        client = get_redis()
        mapping = {}
        clear_fields: list[str] = []

        def _apply(field_name: str, value: Any) -> None:
            if value is _BODY_STORAGE_UPDATE_UNSET:
                return
            if value in REDIS_EMPTY_VALUES:
                clear_fields.append(field_name)
                return
            mapping[field_name] = value

        _apply("request_body_key", request_body_key)
        _apply("response_body_key", response_body_key)
        _apply("request_body_status", request_body_status)
        _apply("request_body_error", request_body_error)
        _apply("response_body_status", response_body_status)
        _apply("response_body_error", response_body_error)

        if mapping:
            await client.hset(f"request:{request_id}", mapping=mapping)
        if clear_fields:
            await client.hdel(f"request:{request_id}", *clear_fields)
        logger.debug("Updated Redis body storage result: %s", request_id)

    @staticmethod
    async def get_by_id(request_id: str) -> Optional[LogRecord]:
        """Get request by ID"""
        client = get_redis()
        data = await client.hgetall(_request_hash_key(request_id))
        return LogRecord(dict(data)) if data else None

    @staticmethod
    async def get_recent(limit: int = 100) -> List[LogRecord]:
        """Get recent requests"""
        if limit <= 0:
            return []

        client = get_redis()

        if not _is_lua_fallback_enabled(False):
            try:
                results = await client.eval(_GET_RECENT_LUA_SCRIPT, 1, REDIS_REQUESTS_BY_TIME_KEY, str(limit))
                return [LogRecord(_flat_pairs_to_dict(data)) for data in results if data]
            except Exception:
                logger.warning("Redis Lua get_recent failed; falling back to pipeline path")

        # Get recent request IDs from sorted set
        request_ids = await client.zrevrange(REDIS_REQUESTS_BY_TIME_KEY, 0, limit - 1)
        if not request_ids:
            return []

        # Batch the per-request hash reads into a single pipeline round-trip.
        # Issuing one hgetall per id (N+1) is the dominant cost of the dashboard
        # under load: at limit=1000 that is 1000 sequential round-trips.
        pipe = client.pipeline(transaction=False)
        for request_id in request_ids:
            pipe.hgetall(f"request:{request_id}")
        results = await pipe.execute()

        return [LogRecord(_flat_pairs_to_dict(data)) for data in results if data]

    @staticmethod
    async def get_by_source_ip(source_ip: str, limit: Optional[int] = None) -> List[LogRecord]:
        """Get requests for a specific source IP ordered by recency."""
        client = get_redis()
        set_key = f"requests:by_ip:{source_ip}"

        if limit is not None and limit <= 0:
            return []

        if limit is not None:
            request_ids = []
            page_size = max(limit * 10, 100)
            start = 0
            while len(request_ids) < limit:
                recent_ids = [
                    str(request_id)
                    for request_id in await client.zrange(
                        REDIS_REQUESTS_BY_TIME_KEY,
                        start,
                        start + page_size - 1,
                        desc=True,
                    )
                ]
                if not recent_ids:
                    break

                pipe = client.pipeline(transaction=False)
                for request_id in recent_ids:
                    pipe.sismember(set_key, request_id)
                matches = await pipe.execute()
                request_ids.extend(
                    request_id for request_id, is_member in zip(recent_ids, matches) if is_member
                )
                request_ids = request_ids[:limit]
                start += page_size
        else:
            request_ids = [str(request_id) for request_id in await client.smembers(set_key)]

        if not request_ids:
            return []

        # Batch the per-request hash reads into a single pipeline round-trip
        # instead of one hgetall per id (N+1).
        pipe = client.pipeline(transaction=False)
        for request_id in request_ids:
            pipe.hgetall(f"request:{request_id}")
        results = await pipe.execute()

        requests = [LogRecord(dict(data)) for data in results if data]

        requests.sort(key=RedisRequestLog._request_log_timestamp, reverse=True)

        if limit is None:
            return requests

        return requests[:limit]

    @staticmethod
    def _normalize_identity_query_limit(limit: int | None) -> int:
        if limit is None:
            return 100
        return limit

    @staticmethod
    async def _fetch_identity_log_batch(client, request_ids: List[str]) -> tuple[list[LogRecord], list[str]]:
        pipe = client.pipeline(transaction=False)
        for request_id in request_ids:
            pipe.hgetall(_request_hash_key(request_id))
        results = await pipe.execute()

        logs: list[LogRecord] = []
        stale_request_ids: list[str] = []
        for request_id, data in zip(request_ids, results):
            if not data:
                stale_request_ids.append(request_id)
                continue
            logs.append(LogRecord(_flat_pairs_to_dict(data)))

        return logs, stale_request_ids

    @staticmethod
    def _request_log_timestamp(log: LogRecord):
        return getattr(getattr(log, "timestamp", None), "timestamp", lambda: 0)()

    @staticmethod
    async def get_by_identity(
        identity_kind: str,
        identity_subject_id: str,
        limit: int | None = None,
    ) -> List[LogRecord]:
        """Get requests for a specific identity kind and subject ordered by recency."""
        client = get_redis()

        if not identity_kind or not identity_subject_id:
            return []

        limit = RedisRequestLog._normalize_identity_query_limit(limit)
        if limit <= 0:
            return []

        index_key = _identity_index_key(identity_kind, identity_subject_id)
        page_size = max(limit * 2, 25)
        start = 0
        logs: list[LogRecord] = []

        while len(logs) < limit:
            request_ids = [str(request_id) for request_id in await client.zrevrange(index_key, start, start + page_size - 1)]
            if not request_ids:
                break

            batch_logs, stale_request_ids = await RedisRequestLog._fetch_identity_log_batch(client, request_ids)
            logs.extend(batch_logs)

            if stale_request_ids:
                await client.zrem(index_key, *stale_request_ids)

            if len(request_ids) < page_size:
                break
            start += len(request_ids) - len(stale_request_ids)

        logs.sort(key=RedisRequestLog._request_log_timestamp, reverse=True)
        return logs[:limit]

    @staticmethod
    async def get_duplicate_request_ids(body_hash: str) -> List[str]:
        """Return all request IDs that share the given request body hash."""
        client = get_redis()
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
        client = get_redis()
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
    _google_selector_script_sha: Optional[str] = None
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

        redis_client = get_redis()

        try:
            cls._script_sha = await redis_client.script_load(QUOTA_UPDATE_SCRIPT)
            cls._google_selector_script_sha = await redis_client.script_load(GOOGLE_ROTARY_SELECTOR_LUA_SCRIPT)
            cls._script_initialized = True
            logger.info(
                "✅ Lua scripts loaded successfully: quota=%s selector=%s",
                cls._script_sha,
                cls._google_selector_script_sha,
            )
        except Exception as e:
            logger.exception("❌ FATAL: Failed to load Lua script into Redis")
            logger.critical("⚠️  Server CANNOT start without functional quota tracking")
            raise RuntimeError(f"Cannot start - Lua script loading failed: {e}") from e

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Create hash of API key for identification"""
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]

    @staticmethod
    def google_rotary_counter_key(provider_id: str, model_name: str) -> str:
        return f"google_rr:{provider_id}:{model_name}"

    @staticmethod
    def google_invalid_keys_key(provider_id: str) -> str:
        return f"google_invalid_keys:{provider_id}"

    @staticmethod
    def google_invalid_key_metadata_key(provider_id: str, api_key_hash: str) -> str:
        return f"google_invalid_key_metadata:{provider_id}:{api_key_hash}"

    @staticmethod
    async def get_or_create_quota(
        api_key: str,
        provider_id: str,
        model_name: str,
    ) -> tuple["QuotaRecord", bool]:
        """Get or create quota entry - returns (quota_record, was_created)"""
        client = get_redis()

        key_hash = RedisApiKeyQuota.hash_api_key(api_key)
        quota_key = f"quota:{provider_id}:{key_hash}:{model_name}"

        # Check if quota exists
        existing = await client.hgetall(quota_key)

        if existing:
            return QuotaRecord(existing), False

        # Create new quota entry
        # Use Pacific timezone for Google API quota resets (midnight Pacific)
        today = _current_pacific_date()
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
    async def _increment_usage_with_pipeline_fallback(
        client: Any,
        quota_key: str,
        request_count: int,
        token_count: int,
        today: str,
        timestamp: str,
    ) -> tuple[int, int, str]:
        logger.warning("⚠️  FakeRedis detected - using pipeline fallback for tests only")

        last_reset = await client.hget(quota_key, "last_reset")
        if last_reset != today:
            await client.hset(
                quota_key,
                mapping={
                    "requests_today": 0,
                    "tokens_today": 0,
                    "last_reset": today,
                    "last_reset_date": today,
                },
            )
            await client.hdel(quota_key, "quota_exhausted_at")
            await client.hdel(quota_key, "quota_cooldown_until")

        pipe = client.pipeline()
        pipe.hincrby(quota_key, "requests_today", request_count)
        pipe.hincrby(quota_key, "tokens_today", token_count)
        pipe.hset(quota_key, "updated_at", timestamp)
        results = await pipe.execute()

        return results[0], results[1], today

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
            if _is_lua_fallback_enabled(cls._lua_disabled):
                # Defer to fallback path implemented below in the exception handler
                pass
            else:
                error_msg = "❌ FATAL: Lua script not initialized - call initialize_lua_script() during startup"
                logger.critical(error_msg)
                raise RuntimeError(error_msg)

        client = get_redis()

        key_hash = RedisApiKeyQuota.hash_api_key(api_key)
        quota_key = f"quota:{provider_id}:{key_hash}:{model_name}"
        # Use Pacific timezone for Google API quota resets (midnight Pacific)
        today = _current_pacific_date()
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
            requests_today, tokens_today, _ = result
        except Exception as e:
            # FakeRedis/compat fallback ONLY (for tests or when Lua disabled)
            if _should_use_increment_fallback(e, cls._lua_disabled):
                requests_today, tokens_today, _ = await cls._increment_usage_with_pipeline_fallback(
                    client,
                    quota_key,
                    request_count,
                    token_count,
                    today,
                    timestamp,
                )
            else:
                # NO OTHER FALLBACKS - if script fails in production, CRASH LOUDLY
                logger.exception("❌ FATAL: Lua script execution failed")
                logger.critical("⚠️  Script SHA: %s", cls._script_sha)
                logger.critical("⚠️  Quota key: %s", quota_key)
                raise RuntimeError(f"Quota tracking BROKEN - Lua script failed: {e}") from e

        # Get full quota data for return
        updated_data = await client.hgetall(quota_key)
        quota_data = dict(updated_data)
        quota_data["requests_today"] = int(requests_today)
        quota_data["tokens_today"] = int(tokens_today)

        return quota_data

    @classmethod
    async def _select_google_api_key_with_pipeline_fallback(
        cls,
        client: Any,
        provider_id: str,
        model_name: str,
        key_hashes: List[str],
        today: str,
        now_iso: str,
        model_limit: int,
    ) -> Dict[str, Any]:
        # Keep the fallback signature aligned with the Lua selector even though
        # rotary selection no longer uses predictive request-count exclusion.
        _ = model_limit
        counter_key = cls.google_rotary_counter_key(provider_id, model_name)
        invalid_key_set = cls.google_invalid_keys_key(provider_id)
        counter_raw = await client.get(counter_key)
        counter = int(counter_raw or 0)
        now_dt = _normalize_datetime(now_iso) or datetime.now(timezone.utc)
        invalid_key_hashes = {str(key_hash) for key_hash in (await client.smembers(invalid_key_set))}
        counts = {"invalid": 0, "cooling_down": 0, "exhausted": 0}
        soonest_cooldown_until: Optional[datetime] = None

        for offset in range(len(key_hashes)):
            idx = (counter + offset) % len(key_hashes)
            key_hash = key_hashes[idx]
            availability, cooldown_until = await cls._get_google_api_key_fallback_availability(
                client, provider_id, model_name, key_hash, invalid_key_hashes, today, now_dt
            )

            if availability != "available":
                counts[availability] += 1
                if cooldown_until is not None:
                    soonest_cooldown_until = cls._earlier_cooldown(soonest_cooldown_until, cooldown_until)
                continue

            await client.set(counter_key, idx + 1)
            return cls._google_api_key_selection_result("ok", idx, key_hash, counts, soonest_cooldown_until, 0)

        retry_after_seconds = (
            max(int((soonest_cooldown_until - now_dt).total_seconds()), 0) if soonest_cooldown_until else _seconds_until_pacific_midnight(now_dt)
        )
        status = "all_invalid" if counts["invalid"] == len(key_hashes) else "none_available"
        return cls._google_api_key_selection_result(status, None, "", counts, soonest_cooldown_until, retry_after_seconds)

    @staticmethod
    async def _get_google_api_key_fallback_availability(
        client: Any,
        provider_id: str,
        model_name: str,
        key_hash: str,
        invalid_key_hashes: set[str],
        today: str,
        now_dt: datetime,
    ) -> tuple[str, Optional[datetime]]:
        """Classify one fallback key using the same order as the Lua selector."""
        if key_hash in invalid_key_hashes:
            return "invalid", None

        quota_data = await client.hgetall(f"quota:{provider_id}:{key_hash}:{model_name}")
        if not quota_data:
            return "available", None

        quota = QuotaRecord(quota_data)
        if quota.invalid_key:
            return "invalid", None
        if quota.quota_cooldown_until and quota.quota_cooldown_until > now_dt:
            return "cooling_down", quota.quota_cooldown_until
        if getattr(quota, "last_reset", quota.last_reset_date) == today and quota.quota_exhausted_at:
            return "exhausted", None
        return "available", None

    @staticmethod
    def _earlier_cooldown(current: Optional[datetime], candidate: datetime) -> datetime:
        if current is None or candidate < current:
            return candidate
        return current

    @staticmethod
    def _google_api_key_selection_result(
        status: str,
        selected_index: Optional[int],
        selected_key_hash: str,
        counts: Mapping[str, int],
        soonest_cooldown_until: Optional[datetime],
        retry_after_seconds: int,
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "selected_index": selected_index,
            "selected_key_hash": selected_key_hash,
            "invalid_count": counts["invalid"],
            "cooling_down_count": counts["cooling_down"],
            "exhausted_count": counts["exhausted"],
            "soonest_cooldown_until": soonest_cooldown_until.isoformat() if soonest_cooldown_until else "",
            "retry_after_seconds": retry_after_seconds,
        }

    @classmethod
    async def select_google_api_key(
        cls,
        provider_id: str,
        model_name: str,
        api_keys: List[str],
        model_limit: int,
    ) -> Dict[str, Any]:
        """Atomically select the next eligible Google API key in serial rotary order.

        ``model_limit`` is intentionally threaded through for selector parity, but
        rotary selection does not preemptively exclude keys by request count.
        """
        if not api_keys:
            return cls._google_api_key_selection_result(
                "no_keys", None, "", {"invalid": 0, "cooling_down": 0, "exhausted": 0}, None, 0
            )

        client = get_redis()
        today = _current_pacific_date()
        now_iso = datetime.now(timezone.utc).isoformat()
        now_dt = _normalize_datetime(now_iso) or datetime.now(timezone.utc)
        key_hashes = [cls.hash_api_key(api_key) for api_key in api_keys]
        counter_key = cls.google_rotary_counter_key(provider_id, model_name)

        selection = await cls._run_google_api_key_selector(
            client, counter_key, provider_id, model_name, key_hashes, today, now_iso, model_limit
        )
        return cls._format_google_api_key_selection(selection, now_dt)

    @classmethod
    async def _run_google_api_key_selector(
        cls,
        client: Any,
        counter_key: str,
        provider_id: str,
        model_name: str,
        key_hashes: List[str],
        today: str,
        now_iso: str,
        model_limit: int,
    ) -> Dict[str, Any]:
        if not cls._script_initialized or cls._google_selector_script_sha is None:
            if _is_lua_fallback_enabled(cls._lua_disabled):
                return await cls._select_google_api_key_with_pipeline_fallback(
                    client, provider_id, model_name, key_hashes, today, now_iso, model_limit
                )
            error_msg = "❌ FATAL: Google selector script not initialized - call initialize_lua_script() during startup"
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        try:
            result = await client.evalsha(
                cls._google_selector_script_sha,
                1,
                counter_key,
                provider_id,
                model_name,
                today,
                now_iso,
                str(model_limit),
                *key_hashes,
            )
        except Exception as error:
            if _should_use_increment_fallback(error, cls._lua_disabled):
                return await cls._select_google_api_key_with_pipeline_fallback(
                    client, provider_id, model_name, key_hashes, today, now_iso, model_limit
                )
            logger.exception("❌ FATAL: Google selector Lua script execution failed")
            raise RuntimeError(f"Google selector BROKEN - Lua script failed: {error}") from error
        return _flat_pairs_to_dict(result)

    @staticmethod
    def _format_google_api_key_selection(selection: Mapping[str, Any], now_dt: datetime) -> Dict[str, Any]:
        selected_index_raw = selection.get("selected_index")
        selected_index = int(selected_index_raw) if selected_index_raw not in REDIS_EMPTY_VALUES else None
        soonest_cooldown_until = str(selection.get("soonest_cooldown_until") or "")
        retry_after_seconds = RedisApiKeyQuota._google_api_key_retry_after(selection, soonest_cooldown_until, now_dt)

        return {
            "status": str(selection.get("status") or "none_available"),
            "selected_index": selected_index,
            "selected_key_hash": str(selection.get("selected_key_hash") or ""),
            "invalid_count": int(selection.get("invalid_count") or 0),
            "cooling_down_count": int(selection.get("cooling_down_count") or 0),
            "exhausted_count": int(selection.get("exhausted_count") or 0),
            "soonest_cooldown_until": soonest_cooldown_until,
            "retry_after_seconds": retry_after_seconds,
        }

    @staticmethod
    def _google_api_key_retry_after(selection: Mapping[str, Any], cooldown_until: str, now_dt: datetime) -> int:
        if selection.get("status") == "ok":
            return 0
        cooldown_dt = _normalize_datetime(cooldown_until) if cooldown_until else None
        if cooldown_dt is not None:
            retry_after_seconds = max(int((cooldown_dt - now_dt).total_seconds()), 0)
            if retry_after_seconds:
                return retry_after_seconds
        return _seconds_until_pacific_midnight(now_dt)

    @staticmethod
    async def get_provider_usage(provider_id: str) -> List["QuotaRecord"]:
        """Get all quota entries for a provider"""
        client = get_redis()

        quota_keys = await client.smembers(f"quotas:by_provider:{provider_id}")

        quotas = []
        for quota_key in quota_keys:
            data = await client.hgetall(quota_key)
            if data:
                quotas.append(QuotaRecord(data))

        return quotas

    @staticmethod
    async def mark_invalid(
        api_key_hash: str,
        provider_name: str,
        *,
        reason: str = "operator",
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> int:
        """Atomically mark one provider/key as invalid with bounded audit metadata.

        Args:
            api_key_hash: The hashed API key
            provider_name: The provider name

        Returns:
            Number of quota entries marked as invalid
        """
        client = get_redis()

        quota_keys = await client.smembers(f"quotas:by_key:{api_key_hash}")
        provider_quota_keys = []
        for quota_key in quota_keys:
            if await client.hget(quota_key, "provider_id") == provider_name:
                provider_quota_keys.append(quota_key)

        now = datetime.now(timezone.utc).isoformat()
        metadata_key = RedisApiKeyQuota.google_invalid_key_metadata_key(provider_name, api_key_hash)
        pipe = client.pipeline(transaction=True)
        pipe.sadd(RedisApiKeyQuota.google_invalid_keys_key(provider_name), api_key_hash)
        pipe.hsetnx(metadata_key, "first_reason", reason)
        pipe.hsetnx(metadata_key, "first_status_code", str(status_code or ""))
        pipe.hsetnx(metadata_key, "first_request_id", request_id or "")
        pipe.hsetnx(metadata_key, "first_recorded_at", now)
        pipe.hincrby(metadata_key, "occurrence_count", 1)
        pipe.hset(
            metadata_key,
            mapping={
                "latest_reason": reason,
                "latest_status_code": str(status_code or ""),
                "latest_request_id": request_id or "",
                "latest_recorded_at": now,
            },
        )
        for quota_key in provider_quota_keys:
            pipe.hset(
                quota_key,
                mapping={
                    "invalid_key": "true",
                    "invalid_reason": reason,
                    "invalid_status_code": str(status_code or ""),
                    "invalidated_at": now,
                    "updated_at": now,
                },
            )
        await pipe.execute()

        logger.debug(
            "Marked %s quota entries invalid for key_hash=%s provider=%s reason=%s",
            len(provider_quota_keys),
            api_key_hash,
            provider_name,
            reason,
        )
        return len(provider_quota_keys)

    @staticmethod
    async def recover_invalid(api_key_hash: str, provider_name: str, *, actor: str, reason: str) -> int:
        """Atomically restore a key's eligibility while retaining quota state."""
        client = get_redis()
        quota_keys = await client.smembers(f"quotas:by_key:{api_key_hash}")
        provider_quota_keys = []
        for quota_key in quota_keys:
            if await client.hget(quota_key, "provider_id") == provider_name:
                provider_quota_keys.append(quota_key)

        metadata_key = RedisApiKeyQuota.google_invalid_key_metadata_key(provider_name, api_key_hash)
        prior_metadata = await client.hgetall(metadata_key)
        audit_key = f"google_invalid_key_recovery_audit:{provider_name}"
        now = datetime.now(timezone.utc).isoformat()
        pipe = client.pipeline(transaction=True)
        pipe.srem(RedisApiKeyQuota.google_invalid_keys_key(provider_name), api_key_hash)
        pipe.delete(metadata_key)
        for quota_key in provider_quota_keys:
            pipe.hdel(quota_key, "invalid_key", "invalid_reason", "invalid_status_code", "invalidated_at")
        pipe.lpush(
            audit_key,
            json.dumps(
                {
                    "actor": actor,
                    "reason": reason,
                    "api_key_hash": api_key_hash,
                    "recovered_at": now,
                    "prior_metadata": prior_metadata,
                },
                sort_keys=True,
            ),
        )
        pipe.ltrim(audit_key, 0, GOOGLE_INVALID_KEY_RECOVERY_AUDIT_LIMIT - 1)
        await pipe.execute()
        return len(provider_quota_keys)

    @staticmethod
    async def mark_quota_exhausted(
        api_key: str, provider_id: str, model_name: str, error: Optional[str] = None
    ) -> None:
        """Mark a quota entry as exhausted and persist to Redis.

        Args:
            api_key: The API key
            provider_id: The provider ID
            model_name: The model name
            error: Optional error message to store
        """
        client = get_redis()

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
    async def mark_quota_cooldown(
        api_key: str,
        provider_id: str,
        model_name: str,
        cooldown_until: datetime,
        error: Optional[str] = None,
    ) -> None:
        """Mark a quota entry as rate-limited (transient cooldown) and persist to Redis.

        Used for per-minute (RPM) 429s. The key returns to the selection pool once
        ``cooldown_until`` passes, rather than being benched until midnight Pacific.
        Unlike mark_quota_exhausted, this does NOT inflate ``error_count`` toward the
        error-prone ban threshold, since transient rate limits are expected.

        Args:
            api_key: The API key
            provider_id: The provider ID
            model_name: The model name
            cooldown_until: UTC timestamp until which the key should be skipped
            error: Optional error message to store
        """
        client = get_redis()

        key_hash = RedisApiKeyQuota.hash_api_key(api_key)
        quota_key = f"quota:{provider_id}:{key_hash}:{model_name}"

        updates = {
            "quota_cooldown_until": cooldown_until.isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if error:
            updates["last_error"] = error

        await client.hset(quota_key, mapping=updates)

        logger.debug(f"Marked quota cooldown until {cooldown_until.isoformat()}: {quota_key}")

    @staticmethod
    async def mark_error(api_key: str, provider_id: str, model_name: str, error: Optional[str] = None) -> None:
        """Mark an error for a quota entry and persist to Redis.

        Args:
            api_key: The API key
            provider_id: The provider ID
            model_name: The model name
            error: Error message to store
        """
        client = get_redis()

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

    except (ConnectionError, TimeoutError):
        # Record failure and re-raise - this is a critical error
        _circuit_breaker.record_failure()
        logger.exception("Redis quota increment FAILED")
        raise

    except Exception:
        # Unexpected errors - log and re-raise
        logger.exception("Unexpected Redis error during quota increment")
        raise


# Maintain compatibility with existing code
RequestLog = RedisRequestLog
ApiKeyQuota = RedisApiKeyQuota


async def init_redis_db():
    """Initialize Redis database connection"""
    from .redis_config import redis_startup_check

    client = get_redis()

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
