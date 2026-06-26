"""
Redis-only database backend for SmolRouter.

This module provides high-performance database operations using Redis,
eliminating SQLite complexity and tech debt.
"""

import os
import logging
import asyncio
import json
import re
import traceback
import uuid
import hashlib
from typing import Any
from datetime import datetime, timezone

from .redis_backend import (
    RequestLog as RedisRequestLog,
    ApiKeyQuota as RedisApiKeyQuota,
    INFLIGHT_SET_KEY,
    REDIS_REQUEST_IDENTITY_KEY_PREFIX,
    init_redis_db,
    close_redis_db,
    get_redis_stats,
)
from .redis_config import redis_client
from .task_utils import create_logged_task

logger = logging.getLogger("smolrouter.database")

# Configuration
MAX_AGE_DAYS = int(os.getenv("MAX_LOG_AGE_DAYS", "7"))  # Auto-purge logs older than N days (0 = disabled)
JSON_BLOB_CONTENT_TYPE = "application/json"
ERROR_SIGNATURE_STATES = ("unknown", "known", "expected", "ignored", "needs_investigation", "fixed")

ERROR_SIGNATURE_KEY_PREFIX = "errors:signature:"
ERROR_EVENT_KEY_PREFIX = "errors:event:"
ERROR_SIGNATURE_SET_KEY = "errors:signatures"
ERROR_EVENT_INDEX_KEY = "errors:events"
ERROR_EVENTS_BY_SIGNATURE_PREFIX = "errors:signature_events:"
ERROR_SIGNATURE_REQUEST_IDS_KEY = ":request_ids"
MAX_EXCEPTION_MESSAGE_LENGTH = 1000
UNKNOWN_VALUE = "<unknown>"

_EXCEPTION_SIGNATURE_AGGREGATE_LUA_SCRIPT = """
local event_id = ARGV[1]
local now = ARGV[2]
local now_ts = tonumber(ARGV[3])
local status_code_key = ARGV[4]
local route = ARGV[5]
local method = ARGV[6]
local source_ip = ARGV[7]
local request_path = ARGV[8]
local user_agent = ARGV[9]
local exception_class = ARGV[10]
local top_frame = ARGV[11]
local normalized_message = ARGV[12]
local latest_error_message = ARGV[13]
local latest_stack_trace = ARGV[14]
local request_id = ARGV[15]
local max_age_days = tonumber(ARGV[16])
local signature_default_state = ARGV[17]
local signature_default_notes = ARGV[18]

local signature = string.sub(KEYS[2], string.len("errors:signature:") + 1)

local existing_count = tonumber(redis.call("HGET", KEYS[2], "count") or "0")
local first_seen = redis.call("HGET", KEYS[2], "first_seen")
if first_seen == false or first_seen == nil or first_seen == "" then
    first_seen = now
end

local raw_status_codes = redis.call("HGET", KEYS[2], "status_codes") or "{}"
local status_codes = cjson.decode(raw_status_codes)
if type(status_codes) ~= "table" then
    status_codes = {}
end

local existing_count_for_status = tonumber(status_codes[status_code_key])
if existing_count_for_status == nil then
    existing_count_for_status = 0
end
status_codes[status_code_key] = existing_count_for_status + 1

local existing_request_ids = redis.call("LRANGE", KEYS[3], 0, -1)
local request_ids = {}

if request_id ~= "" then
    table.insert(request_ids, request_id)
end

for _, request_id_raw in ipairs(existing_request_ids) do
    if request_id_raw ~= request_id and #request_ids < 20 then
        table.insert(request_ids, request_id_raw)
    end
end

if #request_ids > 20 then
    while #request_ids > 20 do
        table.remove(request_ids)
    end
end

local state = redis.call("HGET", KEYS[2], "state")
if state == false or state == nil or state == "" then
    state = signature_default_state
end

local notes = redis.call("HGET", KEYS[2], "notes")
if notes == false or notes == nil or notes == "" then
    notes = signature_default_notes
end

local count = existing_count + 1

redis.call("DEL", KEYS[3])
for _, request_id_from_list in ipairs(request_ids) do
    redis.call("RPUSH", KEYS[3], request_id_from_list)
end
redis.call("LTRIM", KEYS[3], 0, 19)

redis.call("HSET", KEYS[2],
    "signature", signature,
    "exception_class", exception_class,
    "route", route,
    "top_frame", top_frame,
    "normalized_message", normalized_message,
    "count", tostring(count),
    "first_seen", first_seen,
    "last_seen", now,
    "latest_stack_trace", latest_stack_trace,
    "latest_request_id", request_id,
    "latest_status_code", status_code_key,
    "latest_error_message", latest_error_message,
    "status_codes", cjson.encode(status_codes),
    "source_ip", source_ip,
    "request_path", request_path,
    "method", method,
    "user_agent", user_agent,
    "state", state,
    "notes", notes,
    "affected_request_ids", cjson.encode(request_ids)
)

redis.call("SADD", KEYS[6], signature)
redis.call("HSET", KEYS[1],
    "signature", signature,
    "request_id", request_id,
    "timestamp", now,
    "status_code", status_code_key,
    "exception_class", exception_class,
    "route", route,
    "top_frame", top_frame,
    "message", latest_error_message,
    "stack_trace", latest_stack_trace,
    "request_path", request_path,
    "method", method,
    "source_ip", source_ip,
    "user_agent", user_agent
)
redis.call("ZADD", KEYS[4], now_ts, KEYS[1])
redis.call("ZADD", KEYS[5], now_ts, event_id)

if max_age_days and max_age_days > 0 then
    local ttl = math.floor(max_age_days * 24 * 60 * 60)
    if ttl > 0 then
        redis.call("EXPIRE", KEYS[1], ttl)
        redis.call("EXPIRE", KEYS[2], ttl)
        redis.call("EXPIRE", KEYS[3], ttl)
        redis.call("EXPIRE", KEYS[5], ttl)
    end
end

return tostring(count)
"""

# Global cleanup task
_cleanup_task = None


def _to_str(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


def _safe_json_loads(value, default=None):
    if value is None:
        return default
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")

    if isinstance(value, str):
        if not value:
            return default
        try:
            return json.loads(value)
        except Exception:
            return default

    return default


def _normalize_exception_message(message: str | None) -> str:
    if not message:
        return ""

    normalized = _to_str(message)
    if not normalized:
        return ""

    normalized = normalized[:MAX_EXCEPTION_MESSAGE_LENGTH].strip()
    # Reduce noise from changing identifiers
    normalized = re.sub(r"\b[0-9a-fA-F]{16,64}\b", "<hash>", normalized)
    normalized = re.sub(r"\b\d+\b", "<num>", normalized)
    normalized = re.sub(r"0x[0-9a-fA-F]+", "<ptr>", normalized)
    normalized = re.sub(r"\b[A-Za-z0-9_\-]+@[A-Za-z0-9.\-]+", "<email>", normalized)
    normalized = re.sub(r"'[^']*'", "'<str>'", normalized)
    normalized = re.sub(r'"[^"]*"', '"<str>"', normalized)
    return normalized[:1000]


def _extract_exception_top_frame(exception: BaseException) -> str:
    try:
        tb = traceback.extract_tb(exception.__traceback__)
        if not tb:
            return UNKNOWN_VALUE

        for frame in reversed(tb):
            filename = _to_str(frame.filename) or UNKNOWN_VALUE
            if "/site-packages/" not in filename and "/fastapi/" not in filename and "/starlette/" not in filename:
                return f"{os.path.basename(filename)}:{frame.lineno}:{frame.name}"

        frame = tb[-1]
        return f"{os.path.basename(_to_str(frame.filename) or UNKNOWN_VALUE)}:{frame.lineno}:{frame.name}"
    except Exception:
        return UNKNOWN_VALUE


def _build_exception_signature(exception: BaseException, route: str | None, message: str | None) -> tuple[str, dict]:
    exception_class = type(exception).__name__
    route_pattern = route or UNKNOWN_VALUE
    top_frame = _extract_exception_top_frame(exception)
    normalized_message = _normalize_exception_message(message or str(exception))

    signature_payload = {
        "exception_class": exception_class,
        "route": route_pattern,
        "top_frame": top_frame,
        "normalized_message": normalized_message,
    }
    signature_hash = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return signature_hash, signature_payload


async def _load_exception_signature_state(
    exception_signature_key: str, status_code_key: str
) -> tuple[int, str | None, str | None, str | None, dict[str, int]]:
    existing_count_value = await redis_client.hget(exception_signature_key, "count")
    existing_count = int(existing_count_value) if existing_count_value not in (None, "", b"", "None", b"None") else 0

    first_seen = await redis_client.hget(exception_signature_key, "first_seen")
    existing_state = await redis_client.hget(exception_signature_key, "state")
    existing_notes = await redis_client.hget(exception_signature_key, "notes")

    raw_status_codes = await redis_client.hget(exception_signature_key, "status_codes")
    status_codes = _safe_json_loads(raw_status_codes, default={}) or {}
    if not isinstance(status_codes, dict):
        status_codes = {}
    status_codes[status_code_key] = int(status_codes.get(status_code_key, 0)) + 1

    return existing_count, first_seen, existing_state, existing_notes, status_codes


async def _refresh_signature_request_ids(signature_request_ids_key: str, request_id: str | None) -> list[str]:
    if request_id:
        await redis_client.lrem(signature_request_ids_key, 0, request_id)
        await redis_client.lpush(signature_request_ids_key, request_id)
        await redis_client.ltrim(signature_request_ids_key, 0, 19)

    request_ids_raw = await redis_client.lrange(signature_request_ids_key, 0, 19)
    return [_to_str(item) for item in request_ids_raw if item]


async def _apply_exception_record_ttl(
    event_key: str, exception_signature_key: str, signature_request_ids_key: str, signature: str
) -> None:
    if MAX_AGE_DAYS <= 0:
        return

    ttl = MAX_AGE_DAYS * 24 * 60 * 60
    if ttl > 0:
        await redis_client.expire(event_key, ttl)
        await redis_client.expire(exception_signature_key, ttl)
        await redis_client.expire(signature_request_ids_key, ttl)
        await redis_client.expire(f"{ERROR_EVENTS_BY_SIGNATURE_PREFIX}{signature}", ttl)


async def _record_exception_event_non_atomic(
    *,
    event_id: str,
    now: str,
    now_ts: float,
    exception: BaseException,
    route: str | None,
    request_path: str | None,
    method: str | None,
    source_ip: str | None,
    user_agent: str | None,
    request_id: str | None,
    status_code: int,
    stack_text: str,
) -> dict[str, str]:
    signature, signature_metadata = _build_exception_signature(
        exception=exception, route=route, message=str(exception)
    )
    exception_signature_key = f"{ERROR_SIGNATURE_KEY_PREFIX}{signature}"
    signature_request_ids_key = f"{exception_signature_key}{ERROR_SIGNATURE_REQUEST_IDS_KEY}"

    event_key = f"{ERROR_EVENT_KEY_PREFIX}{event_id}"
    status_code_key = str(int(status_code))
    existing_count, first_seen, existing_state, existing_notes, status_codes = await _load_exception_signature_state(
        exception_signature_key, status_code_key
    )
    request_ids = await _refresh_signature_request_ids(signature_request_ids_key, request_id)

    signature_data = {
        "id": event_id,
        "signature": signature,
        "exception_class": signature_metadata["exception_class"],
        "route": route or request_path or UNKNOWN_VALUE,
        "top_frame": signature_metadata["top_frame"],
        "normalized_message": signature_metadata["normalized_message"],
        "count": str(existing_count + 1),
        "first_seen": _to_str(first_seen) or now,
        "last_seen": now,
        "latest_stack_trace": stack_text,
        "latest_request_id": request_id or "",
        "latest_status_code": status_code_key,
        "latest_error_message": str(exception),
        "status_codes": json.dumps(status_codes),
        "source_ip": source_ip or "",
        "request_path": request_path or "",
        "method": method or "",
        "user_agent": user_agent or "",
        "state": _to_str(existing_state) or ERROR_SIGNATURE_STATES[0],
        "notes": _to_str(existing_notes) or "",
        "affected_request_ids": json.dumps(request_ids),
    }

    await redis_client.hset(exception_signature_key, mapping=signature_data)
    await redis_client.sadd(ERROR_SIGNATURE_SET_KEY, signature)

    await redis_client.hset(
        event_key,
        mapping={
            "id": event_id,
            "signature": signature,
            "request_id": request_id or "",
            "timestamp": now,
            "status_code": status_code_key,
            "exception_class": signature_metadata["exception_class"],
            "route": route or request_path or UNKNOWN_VALUE,
            "top_frame": signature_metadata["top_frame"],
            "message": str(exception),
            "stack_trace": stack_text,
        },
    )
    await redis_client.zadd(ERROR_EVENT_INDEX_KEY, {event_key: now_ts})
    await redis_client.zadd(f"{ERROR_EVENTS_BY_SIGNATURE_PREFIX}{signature}", {event_id: now_ts})
    await _apply_exception_record_ttl(event_key, exception_signature_key, signature_request_ids_key, signature)

    return signature_data


async def record_exception_event(
    *,
    request_id: str | None,
    exception: BaseException,
    route: str | None,
    request_path: str | None,
    method: str | None = None,
    source_ip: str | None = None,
    status_code: int = 500,
    user_agent: str | None = None,
) -> dict[str, str] | None:
    try:
        event_id = str(uuid.uuid4())
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        now_ts = now_dt.timestamp()

        signature, signature_metadata = _build_exception_signature(
            exception=exception, route=route, message=str(exception)
        )

        stack = traceback.format_exception(type(exception), exception, exception.__traceback__)
        stack_text = "".join(stack)
        exception_class = signature_metadata["exception_class"]
        resolved_route = route or request_path or UNKNOWN_VALUE
        status_code_key = str(int(status_code))

        exception_signature_key = f"{ERROR_SIGNATURE_KEY_PREFIX}{signature}"
        signature_request_ids_key = f"{ERROR_SIGNATURE_KEY_PREFIX}{signature}{ERROR_SIGNATURE_REQUEST_IDS_KEY}"
        event_key = f"{ERROR_EVENT_KEY_PREFIX}{event_id}"
        try:
            await redis_client.eval(
                _EXCEPTION_SIGNATURE_AGGREGATE_LUA_SCRIPT,
                6,
                event_key,
                exception_signature_key,
                signature_request_ids_key,
                ERROR_EVENT_INDEX_KEY,
                f"{ERROR_EVENTS_BY_SIGNATURE_PREFIX}{signature}",
                ERROR_SIGNATURE_SET_KEY,
                event_id,
                now,
                str(now_ts),
                status_code_key,
                resolved_route,
                method or "",
                source_ip or "",
                request_path or "",
                user_agent or "",
                exception_class,
                signature_metadata["top_frame"],
                signature_metadata["normalized_message"],
                str(exception),
                stack_text,
                request_id or "",
                str(MAX_AGE_DAYS),
                ERROR_SIGNATURE_STATES[0],
                "",
            )
            return await get_exception_signature_detail(signature)
        except Exception:
            logger.exception("Failed atomic exception aggregation for %s", signature)
            signature_data = await _record_exception_event_non_atomic(
                event_id=event_id,
                now=now,
                now_ts=now_ts,
                exception=exception,
                route=route,
                request_path=request_path,
                method=method,
                source_ip=source_ip,
                user_agent=user_agent,
                request_id=request_id,
                status_code=status_code,
                stack_text=stack_text,
            )
            return signature_data
    except Exception:
        logger.exception("Failed to record exception event")
        return None


async def _collect_exception_signature_summary(
    signature_id: str,
    class_counts: dict[str, int],
    route_counts: dict[str, int],
    status_code_counts: dict[str, int],
) -> dict | None:
    summary = await redis_client.hgetall(f"{ERROR_SIGNATURE_KEY_PREFIX}{signature_id}")
    return _summarize_signature_hash(signature_id, summary, class_counts, route_counts, status_code_counts)


def _summarize_signature_hash(
    signature_id: str,
    summary: Any,
    class_counts: dict[str, int],
    route_counts: dict[str, int],
    status_code_counts: dict[str, int],
) -> dict | None:
    """Process one already-fetched signature hash (no Redis I/O)."""
    if not summary:
        return None

    summary_dict = { _to_str(k): _to_str(v) for k, v in summary.items() }
    count = int(summary_dict.get("count", 0) or 0)
    exception_class = summary_dict.get("exception_class", "Unknown")
    route = summary_dict.get("route", UNKNOWN_VALUE)

    class_counts[exception_class] = class_counts.get(exception_class, 0) + count
    route_counts[route] = route_counts.get(route, 0) + count

    status_codes = _safe_json_loads(summary_dict.get("status_codes", "{}"), default={})
    if isinstance(status_codes, dict):
        for status_code, status_count in status_codes.items():
            status_code_key = str(status_code)
            status_code_counts[status_code_key] = status_code_counts.get(status_code_key, 0) + int(status_count)

    affected_request_ids = _safe_json_loads(summary_dict.get("affected_request_ids", "[]"), default=[])

    return {
        "signature": signature_id,
        "exception_class": exception_class,
        "route": route,
        "count": count,
        "first_seen": summary_dict.get("first_seen"),
        "last_seen": summary_dict.get("last_seen"),
        "top_frame": summary_dict.get("top_frame"),
        "normalized_message": summary_dict.get("normalized_message"),
        "status_codes": status_codes if isinstance(status_codes, dict) else {},
        "state": summary_dict.get("state", ERROR_SIGNATURE_STATES[0]),
        "notes": summary_dict.get("notes", ""),
        "latest_request_id": summary_dict.get("latest_request_id"),
        "latest_stack_trace": summary_dict.get("latest_stack_trace"),
        "affected_request_ids": affected_request_ids if isinstance(affected_request_ids, list) else [],
    }


async def get_error_summary(limit_signatures: int = 50) -> dict:
    try:
        signature_ids = [s for s in (_to_str(s) for s in await redis_client.smembers(ERROR_SIGNATURE_SET_KEY)) if s]

        class_counts: dict[str, int] = {}
        route_counts: dict[str, int] = {}
        summaries: list[dict] = []
        status_code_counts: dict[str, int] = {}

        if signature_ids:
            # Batch all per-signature reads into one pipeline (was N+1 over
            # signatures - O(signatures) round-trips on every dashboard poll).
            pipe = redis_client.pipeline(transaction=False)
            for signature_id in signature_ids:
                pipe.hgetall(f"{ERROR_SIGNATURE_KEY_PREFIX}{signature_id}")
            raw_summaries = await pipe.execute()

            for signature_id, raw in zip(signature_ids, raw_summaries):
                summary = _summarize_signature_hash(
                    signature_id, raw, class_counts, route_counts, status_code_counts
                )
                if summary is not None:
                    summaries.append(summary)

        summaries.sort(key=lambda item: item.get("count", 0), reverse=True)
        if limit_signatures > 0:
            signatures = summaries[:limit_signatures]
        else:
            signatures = summaries

        total_exceptions = sum(item.get("count", 0) for item in summaries)
        return {
            "total_exceptions": total_exceptions,
            "signature_count": len(summaries),
            "count_by_signature": {item["signature"]: item["count"] for item in summaries},
            "count_by_exception_class": class_counts,
            "count_by_route": route_counts,
            "status_code_counts": status_code_counts,
            "signatures": signatures,
        }
    except Exception:
        logger.exception("Failed to get error summary")
        return {
            "total_exceptions": 0,
            "signature_count": 0,
            "count_by_signature": {},
            "count_by_exception_class": {},
            "count_by_route": {},
            "status_code_counts": {},
            "signatures": [],
        }


async def get_error_recent_events(limit: int = 100):
    try:
        recent_ids = list(await redis_client.zrevrange(ERROR_EVENT_INDEX_KEY, 0, max(limit - 1, 0)))
        events = []

        for event_id in recent_ids:
            if isinstance(event_id, bytes):
                event_id = _to_str(event_id)
            event_key = _to_str(event_id)
            if event_key is None:
                continue
            if event_key.startswith(ERROR_EVENT_KEY_PREFIX):
                data = await redis_client.hgetall(event_key)
                if data:
                    event = { _to_str(k): _to_str(v) for k, v in data.items() }
                    event["id"] = event.get("id") or event_key.replace(ERROR_EVENT_KEY_PREFIX, "")
                    events.append(event)

        return events
    except Exception:
        logger.exception("Failed to fetch recent error events")
        return []


async def get_exception_signature_detail(signature: str):
    try:
        summary_data = await redis_client.hgetall(f"{ERROR_SIGNATURE_KEY_PREFIX}{signature}")
        if not summary_data:
            return None

        summary = { _to_str(k): _to_str(v) for k, v in summary_data.items() }
        status_codes = _safe_json_loads(summary.get("status_codes", "{}"), default={})

        events = []
        event_ids = await redis_client.zrevrange(f"{ERROR_EVENTS_BY_SIGNATURE_PREFIX}{signature}", 0, 49)
        for event_id in event_ids:
            event_id_text = _to_str(event_id)
            if not event_id_text:
                continue

            event_data = await redis_client.hgetall(f"{ERROR_EVENT_KEY_PREFIX}{event_id_text}")
            if event_data:
                event_payload = { _to_str(k): _to_str(v) for k, v in event_data.items() }
                event_payload["id"] = event_payload.get("id") or event_id_text
                events.append(event_payload)

        affected_request_ids = _safe_json_loads(summary.get("affected_request_ids", "[]"), default=[])

        return {
            "signature": signature,
            "exception_class": summary.get("exception_class", "Unknown"),
            "route": summary.get("route", UNKNOWN_VALUE),
            "top_frame": summary.get("top_frame", ""),
            "normalized_message": summary.get("normalized_message", ""),
            "count": int(summary.get("count", 0) or 0),
            "first_seen": summary.get("first_seen"),
            "last_seen": summary.get("last_seen"),
            "latest_stack_trace": summary.get("latest_stack_trace", ""),
            "status_codes": status_codes if isinstance(status_codes, dict) else {},
            "state": summary.get("state", ERROR_SIGNATURE_STATES[0]),
            "notes": summary.get("notes", ""),
            "affected_request_ids": affected_request_ids if isinstance(affected_request_ids, list) else [],
            "latest_request_id": summary.get("latest_request_id", ""),
            "latest_error_message": summary.get("latest_error_message", ""),
            "request_path": summary.get("request_path", ""),
            "method": summary.get("method", ""),
            "recent_events": events,
        }
    except Exception:
        logger.exception("Failed to get exception signature detail")
        return None


async def set_exception_signature_state(signature: str, state: str | None = None, notes: str | None = None):
    try:
        key = f"{ERROR_SIGNATURE_KEY_PREFIX}{signature}"
        existing = await redis_client.exists(key)
        if not existing:
            return None

        updates: dict[str, str] = {}
        if state is not None:
            if state not in ERROR_SIGNATURE_STATES:
                raise ValueError(f"Unsupported state '{state}'")
            updates["state"] = state
        if notes is not None:
            updates["notes"] = notes

        if updates:
            await redis_client.hset(key, mapping=updates)

        updated = await redis_client.hgetall(key)
        return { _to_str(k): _to_str(v) for k, v in updated.items() }
    except Exception:
        logger.exception("Failed to update exception signature state")
        return None


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
        self.identity_kind = kwargs.get("identity_kind")
        self.identity_subject_id = kwargs.get("identity_subject_id")
        self.identity_display_name = kwargs.get("identity_display_name")
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
        self._body_archival_scheduled = False

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
            "upstream_url": getattr(self, "upstream_url", None),
            "request_body_key": request_body_key,
            "response_body_key": response_body_key,
            "api_key_suffix": getattr(self, "api_key_suffix", None),
            "proxy_used": getattr(self, "proxy_used", None),
            "provider_id": getattr(self, "provider_id", None),
            "api_key_index": getattr(self, "api_key_index", None),
            "api_key_total": getattr(self, "api_key_total", None),
        }

    def _needs_body_archival(self) -> bool:
        return (
            (getattr(self, "request_body", None) and not getattr(self, "request_body_key", None))
            or (getattr(self, "response_body", None) and not getattr(self, "response_body_key", None))
        )

    async def _archive_bodies_after_completion(self) -> None:
        from .storage import get_blob_storage
        from .redis_backend import RedisRequestLog

        if not self._needs_body_archival():
            return

        request_body_key = None
        response_body_key = None
        try:
            blob_storage = get_blob_storage()
            try:
                request_body_key = await self._store_request_body_if_needed(blob_storage)
            except Exception:
                logger.exception("Request body storage failed for request %s", self.request_id)

            try:
                response_body_key = await self._store_response_body_if_present(blob_storage)
            except Exception:
                logger.exception("Response body storage failed for request %s", self.request_id)

            await RedisRequestLog.update_body_keys(
                request_id=self.request_id,
                request_body_key=request_body_key,
                response_body_key=response_body_key,
            )
        except Exception:
            logger.exception("Body storage failed for request %s after completion persisted", self.request_id)

    def _schedule_body_archival(self) -> None:
        if self._body_archival_scheduled or not self._needs_body_archival():
            return

        archival_task = create_logged_task(
            self._archive_bodies_after_completion(),
            task_name=f"request-body-archival:{self.request_id}",
            create_task_fn=asyncio.create_task,
        )
        if archival_task is not None:
            self._body_archival_scheduled = True

    async def _store_completion_update(self, *, run_archival_inline: bool = False) -> None:
        from .redis_backend import RedisRequestLog

        await RedisRequestLog.update_completion(
            **self._completion_update_kwargs(
                getattr(self, "request_body_key", None),
                getattr(self, "response_body_key", None),
            )
        )
        if run_archival_inline:
            await self._archive_bodies_after_completion()
        else:
            self._schedule_body_archival()

    async def _run_completion_update(self, attempts: int = 3, *, run_archival_inline: bool = False) -> None:
        # The completion write is fire-and-forget; a transient Redis failure
        # (pool contention under load) must NOT silently orphan the request (it
        # would also leave it in the inflight set forever). Retry with small
        # backoff, and on final failure log loudly instead of swallowing silently.
        for attempt in range(1, attempts + 1):
            try:
                await self._store_completion_update(run_archival_inline=run_archival_inline)
                return
            except Exception:
                if attempt >= attempts:
                    logger.exception(
                        "Failed to persist completion for %s after %s attempts (request left pending)",
                        self.request_id,
                        attempts,
                    )
                    return
                await asyncio.sleep(0.05 * attempt)

    def _schedule_completion_update(self) -> None:
        completion_update_task = create_logged_task(
            self._run_completion_update(),
            task_name=f"request-completion-update:{self.request_id}",
            create_task_fn=asyncio.create_task,
        )
        if completion_update_task is not None:
            return

        # No event loop running - run synchronously (tests/CLI)
        try:
            asyncio.run(self._run_completion_update(run_archival_inline=True))
        except Exception:
            logger.exception("Failed to run async store/update")

    def save(self):
        """Update Redis with completion data when request is finished"""
        if not self._has_completion_data():
            return

        self._schedule_completion_update()

    async def save_async(self):
        """Async version of save() that awaits completion persistence and archival.

        Useful in tests and sync-adjacent call sites that need deterministic bodies.
        """
        if not self._has_completion_data():
            return

        await self._store_completion_update(run_archival_inline=True)

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
            "identity_kind": "identity_kind",
            "identity_subject_id": "identity_subject_id",
            "identity_display_name": "identity_display_name",
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
    async def get_by_identity(identity_kind: str, identity_subject_id: str, limit: int | None = None):
        """Get requests for a specific identity ordered by recency."""
        return await RedisRequestLog.get_by_identity(identity_kind, identity_subject_id, limit=limit)

    @staticmethod
    async def get_by_source_ip(source_ip: str, limit: int | None = None):
        """Get requests for a specific client IP ordered by recency."""
        return await RedisRequestLog.get_by_source_ip(source_ip, limit)

    @staticmethod
    async def get_stats_counters():
        """O(1) dashboard counters (total/completed/failed/service_types/inflight)."""
        return await RedisRequestLog.get_stats_counters()

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
    async def mark_quota_cooldown(
        api_key: str,
        provider_id: str,
        model_name: str,
        cooldown_until: datetime,
        error: str = None,
    ) -> None:
        """Mark a quota entry as rate-limited with a transient cooldown.

        Args:
            api_key: The API key
            provider_id: The provider ID
            model_name: The model name
            cooldown_until: UTC timestamp until which the key should be skipped
            error: Optional error message to store
        """
        return await RedisApiKeyQuota.mark_quota_cooldown(
            api_key, provider_id, model_name, cooldown_until, error
        )

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
        deleted = await _cleanup_old_request_logs(client, cutoff_ts)
        error_deleted, signatures_touched = await _cleanup_old_error_events(client, cutoff_ts)
        deleted += error_deleted
        await _cleanup_orphaned_error_signatures(client, signatures_touched)
        return deleted
    except Exception:
        logger.exception("Error during cleanup_old_logs_async")
        return 0


async def _cleanup_old_request_logs(client, cutoff_ts: float) -> int:
    old_ids = await client.zrangebyscore("requests:by_time", 0, cutoff_ts)
    deleted = 0
    for request_id in old_ids:
        key = f"request:{request_id}"
        data = await client.hgetall(key)
        source_ip = data.get("source_ip") if data else None
        identity_kind = data.get("identity_kind") if data else None
        identity_subject_id = data.get("identity_subject_id") if data else None

        if source_ip:
            await client.srem(f"requests:by_ip:{source_ip}", request_id)

        if identity_kind and identity_subject_id:
            await client.zrem(f"{REDIS_REQUEST_IDENTITY_KEY_PREFIX}:{_to_str(identity_kind)}:{_to_str(identity_subject_id)}", request_id)

        await client.srem(INFLIGHT_SET_KEY, request_id)
        await client.delete(key)
        await client.zrem("requests:by_time", request_id)
        deleted += 1
    return deleted


async def _cleanup_old_error_events(client, cutoff_ts: float) -> tuple[int, set[str]]:
    old_error_event_keys = await client.zrangebyscore(ERROR_EVENT_INDEX_KEY, 0, cutoff_ts)
    signatures_touched: set[str] = set()
    deleted = 0
    for event_key in old_error_event_keys:
        error_event_key = _to_str(event_key)
        if not error_event_key:
            continue

        event_data = await client.hgetall(error_event_key)
        signature = _to_str(event_data.get("signature", "")) if event_data else None

        event_id = error_event_key
        if event_id.startswith(ERROR_EVENT_KEY_PREFIX):
            event_id = error_event_key.replace(ERROR_EVENT_KEY_PREFIX, "", 1)

        await client.delete(error_event_key)
        await client.zrem(ERROR_EVENT_INDEX_KEY, error_event_key)
        deleted += 1

        if signature:
            signatures_touched.add(signature)
            await client.zrem(f"{ERROR_EVENTS_BY_SIGNATURE_PREFIX}{signature}", event_id)

    return deleted, signatures_touched


async def _delete_error_signature_artifacts(client, signature: str) -> None:
    await client.delete(f"{ERROR_SIGNATURE_KEY_PREFIX}{signature}")
    await client.delete(f"{ERROR_EVENTS_BY_SIGNATURE_PREFIX}{signature}")
    await client.delete(f"{ERROR_SIGNATURE_KEY_PREFIX}{signature}{ERROR_SIGNATURE_REQUEST_IDS_KEY}")
    await client.srem(ERROR_SIGNATURE_SET_KEY, signature)


async def _cleanup_orphaned_error_signatures(client, signatures_touched: set[str]) -> None:
    stale_signatures = await client.smembers(ERROR_SIGNATURE_SET_KEY)
    for stale_signature in stale_signatures:
        stale_signature_text = _to_str(stale_signature)
        if not stale_signature_text:
            continue

        if not await client.exists(f"{ERROR_SIGNATURE_KEY_PREFIX}{stale_signature_text}"):
            signatures_touched.add(stale_signature_text)

    for signature in signatures_touched:
        signature_key = f"{ERROR_SIGNATURE_KEY_PREFIX}{signature}"
        if not await client.exists(signature_key):
            await _delete_error_signature_artifacts(client, signature)
            continue

        if await client.zcard(f"{ERROR_EVENTS_BY_SIGNATURE_PREFIX}{signature}") == 0:
            await _delete_error_signature_artifacts(client, signature)


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
            deleted = await cleanup_old_logs_async(MAX_AGE_DAYS)
            logger.info("Background cleanup cycle (MAX_LOG_AGE_DAYS=%s) removed=%s", MAX_AGE_DAYS, deleted)

        except asyncio.CancelledError:
            logger.info("Background cleanup task cancelled")
            raise
        except Exception:
            logger.exception("Error in background cleanup")
            # Continue running despite errors
            continue


def start_background_cleanup():
    """Start the background cleanup task"""
    global _cleanup_task
    if MAX_AGE_DAYS > 0:
        try:
            _cleanup_task = create_logged_task(
                background_cleanup_task(),
                task_name="background-request-log-cleanup",
                service=True,
                create_task_fn=asyncio.create_task,
            )
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
    """Get logging statistics.

    O(1): served from Redis counters/sets maintained on create/complete, not by
    scanning+deserializing a 1000-record sample on every call (which blocked the
    event loop and made the dashboard crawl). total_requests is now a true
    monotonic counter (requests since these counters were introduced), not the
    size of a recent sample.
    """
    try:
        counters = await RequestLog.get_stats_counters()
        total = counters["total"]
        completed = counters["completed"]

        error_summary = await get_error_summary(limit_signatures=10)

        return {
            "total_requests": total,
            "completed_requests": completed,
            "pending_requests": max(0, total - completed),
            "failed_requests": counters["failed"],
            "service_types": counters["service_types"],
            "inflight_requests": counters["inflight"],
            "exception_signatures": error_summary.get("signature_count", 0),
            "exceptions_total": error_summary.get("total_exceptions", 0),
        }
    except Exception:
        logger.exception("Error getting log stats")
        return {"total_requests": 0, "completed_requests": 0, "pending_requests": 0, "service_types": {}}


async def get_inflight_requests(recent_logs=None):
    """Get in-flight (pending) requests from the last 60 minutes.

    Accepts an optional pre-fetched recent-logs sample so callers that already
    hold one (e.g. get_log_stats) avoid a redundant get_recent(1000).
    """
    try:
        from datetime import datetime, timedelta, timezone

        all_recent = recent_logs if recent_logs is not None else await RequestLog.get_recent(1000)
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=60)

        # Filter for pending requests (status_code = "pending" or empty completed_at) within 60 min window
        inflight = [
            log
            for log in all_recent
            if (getattr(log, "status_code", "pending") == "pending" or not getattr(log, "completed_at", None))
            and getattr(log, "timestamp", datetime.now()) >= cutoff_time
        ]
        return inflight
    except Exception:
        logger.exception("Error getting inflight requests")
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
