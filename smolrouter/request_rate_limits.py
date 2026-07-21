"""Atomic Redis-backed request rate limiting for anonymous and project traffic."""

from __future__ import annotations

import hashlib
import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

from .redis_config import redis_client


logger = logging.getLogger("smolrouter.request_rate_limits")
_CONFIG_KEYS = frozenset({"enabled", "anonymous", "project_default"})
_POLICY_KEYS = frozenset({"windows"})
_WINDOW_KEYS = frozenset({"requests", "seconds"})
_BUCKET_CLASSES = frozenset({"anonymous", "project"})
_BUCKET_KEY_PREFIX = "smolrouter:request-rate-limit:bucket"
_STATS_KEY = "smolrouter:request-rate-limit:stats"


class RequestRateLimitConfigError(ValueError):
    """Raised when request rate-limit configuration violates its schema."""


@dataclass(frozen=True, slots=True)
class RateLimitWindow:
    requests: int
    seconds: int

    def to_dict(self) -> dict[str, int]:
        return {"requests": self.requests, "seconds": self.seconds}


@dataclass(frozen=True, slots=True)
class RequestRateLimitPolicy:
    windows: tuple[RateLimitWindow, RateLimitWindow]

    def to_dict(self) -> dict[str, list[dict[str, int]]]:
        return {"windows": [window.to_dict() for window in self.windows]}


DEFAULT_REQUEST_RATE_LIMIT_POLICY = RequestRateLimitPolicy(
    windows=(RateLimitWindow(requests=1, seconds=1), RateLimitWindow(requests=3, seconds=10))
)


@dataclass(frozen=True, slots=True)
class RequestRateLimitConfig:
    enabled: bool = True
    anonymous: RequestRateLimitPolicy = DEFAULT_REQUEST_RATE_LIMIT_POLICY
    project_default: RequestRateLimitPolicy = DEFAULT_REQUEST_RATE_LIMIT_POLICY


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    retry_after_ms: int
    violated_windows: tuple[RateLimitWindow, ...]
    counts: tuple[int, int]
    bucket_class: str
    policy: RequestRateLimitPolicy

    @property
    def retry_after_seconds(self) -> int:
        """Return an HTTP ``Retry-After`` value, rounded up to whole seconds."""

        if self.allowed:
            return 0
        return max(1, math.ceil(self.retry_after_ms / 1000))


def _require_mapping(raw: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise RequestRateLimitConfigError(f"{field_name} must be a mapping")
    return raw


def _reject_unknown_keys(raw: Mapping[str, Any], allowed: frozenset[str], field_name: str) -> None:
    unknown = sorted(set(raw) - allowed, key=repr)
    if unknown:
        rendered = ", ".join(str(key) for key in unknown)
        raise RequestRateLimitConfigError(f"{field_name} contains unknown keys: {rendered}")


def _parse_request_rate_limit_window(raw: Any, field_name: str) -> RateLimitWindow:
    window = _require_mapping(raw, field_name)
    _reject_unknown_keys(window, _WINDOW_KEYS, field_name)
    missing = sorted(_WINDOW_KEYS - set(window))
    if missing:
        raise RequestRateLimitConfigError(f"{field_name} is missing required keys: {', '.join(missing)}")

    requests = window["requests"]
    seconds = window["seconds"]
    if type(requests) is not int or requests <= 0:
        raise RequestRateLimitConfigError(f"{field_name}.requests must be a positive integer")
    if type(seconds) is not int or seconds <= 0:
        raise RequestRateLimitConfigError(f"{field_name}.seconds must be a positive integer")
    return RateLimitWindow(requests=requests, seconds=seconds)


def parse_request_rate_limit_policy(
    raw: Any,
    field_name: str,
    default: RequestRateLimitPolicy | None = None,
) -> RequestRateLimitPolicy:
    """Parse one strict, complete two-window rate-limit policy."""

    if raw is None:
        if default is not None:
            return default
        raise RequestRateLimitConfigError(f"{field_name} must be a mapping")

    policy = _require_mapping(raw, field_name)
    _reject_unknown_keys(policy, _POLICY_KEYS, field_name)
    if "windows" not in policy:
        raise RequestRateLimitConfigError(f"{field_name}.windows is required")

    windows_raw = policy["windows"]
    if type(windows_raw) is not list:
        raise RequestRateLimitConfigError(f"{field_name}.windows must be a list")
    if len(windows_raw) != 2:
        raise RequestRateLimitConfigError(f"{field_name}.windows must contain exactly 2 windows")

    windows = [
        _parse_request_rate_limit_window(window_raw, f"{field_name}.windows[{index}]")
        for index, window_raw in enumerate(windows_raw)
    ]

    short, long = windows
    if short.seconds >= long.seconds:
        raise RequestRateLimitConfigError(f"{field_name}.windows durations must be unique and strictly ascending")
    if short.requests > long.requests:
        raise RequestRateLimitConfigError(f"{field_name}.windows request capacities must be monotonic")

    return RequestRateLimitPolicy(windows=(short, long))


def parse_request_rate_limit_config(raw: Any) -> RequestRateLimitConfig:
    """Parse the strict top-level request rate-limit configuration."""

    if raw is None:
        raw = {}
    config = _require_mapping(raw, "request_rate_limits")
    _reject_unknown_keys(config, _CONFIG_KEYS, "request_rate_limits")

    enabled = config.get("enabled", True)
    if type(enabled) is not bool:
        raise RequestRateLimitConfigError("request_rate_limits.enabled must be a boolean")

    return RequestRateLimitConfig(
        enabled=enabled,
        anonymous=parse_request_rate_limit_policy(
            config["anonymous"] if "anonymous" in config else None,
            "request_rate_limits.anonymous",
            DEFAULT_REQUEST_RATE_LIMIT_POLICY if "anonymous" not in config else None,
        ),
        project_default=parse_request_rate_limit_policy(
            config["project_default"] if "project_default" in config else None,
            "request_rate_limits.project_default",
            DEFAULT_REQUEST_RATE_LIMIT_POLICY if "project_default" not in config else None,
        ),
    )


def request_rate_limit_bucket_key(
    bucket_class: str,
    *,
    identity_kind: str | None = None,
    identity_subject_id: str | None = None,
) -> str:
    """Build a stable bucket key without putting identity material into Redis keys."""

    if bucket_class not in _BUCKET_CLASSES:
        raise ValueError("bucket_class must be 'anonymous' or 'project'")

    if bucket_class == "anonymous":
        if identity_kind is not None or identity_subject_id is not None:
            raise ValueError("anonymous buckets do not accept identity fields")
        canonical = "anonymous:global"
    else:
        if not isinstance(identity_kind, str) or not identity_kind:
            raise ValueError("project buckets require a non-empty identity_kind")
        if not isinstance(identity_subject_id, str) or not identity_subject_id:
            raise ValueError("project buckets require a non-empty identity_subject_id")
        canonical = f"{identity_kind}:{identity_subject_id}"

    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{_BUCKET_KEY_PREFIX}:{bucket_class}:{digest}"


_CHECK_RATE_LIMIT_LUA = r"""
local bucket_key = KEYS[1]
local member = ARGV[1]
local short_limit = tonumber(ARGV[2])
local short_window_ms = tonumber(ARGV[3])
local long_limit = tonumber(ARGV[4])
local long_window_ms = tonumber(ARGV[5])
local supplied_now_ms = ARGV[6]

local now_ms
if supplied_now_ms ~= "" then
    now_ms = tonumber(supplied_now_ms)
else
    local redis_time = redis.call("TIME")
    now_ms = (tonumber(redis_time[1]) * 1000) + math.floor(tonumber(redis_time[2]) / 1000)
end

local short_cutoff = now_ms - short_window_ms
local long_cutoff = now_ms - long_window_ms
local short_count = tonumber(redis.call("ZCOUNT", bucket_key, "(" .. short_cutoff, "+inf"))
local long_count = tonumber(redis.call("ZCOUNT", bucket_key, "(" .. long_cutoff, "+inf"))
local short_violated = short_count >= short_limit
local long_violated = long_count >= long_limit

local retry_after_ms = 0

-- Redis returns WITHSCORES as {member, score} inside Lua. Newer FakeRedis
-- versions model the RESP3 shape as {{member, score}} instead. Accept both so
-- development and CI exercise the same admission semantics as real Redis.
local function ranked_event_score(event)
    if #event == 2 then
        return tonumber(event[2])
    end
    if #event == 1 and type(event[1]) == "table" and #event[1] >= 2 then
        return tonumber(event[1][2])
    end
    return nil
end

if short_violated then
    local rank = short_count - short_limit
    local event = redis.call("ZRANGEBYSCORE", bucket_key, "(" .. short_cutoff, "+inf", "WITHSCORES", "LIMIT", rank, 1)
    local score = ranked_event_score(event)
    if score then
        retry_after_ms = math.max(retry_after_ms, score + short_window_ms - now_ms)
    end
end

if long_violated then
    local rank = long_count - long_limit
    local event = redis.call("ZRANGEBYSCORE", bucket_key, "(" .. long_cutoff, "+inf", "WITHSCORES", "LIMIT", rank, 1)
    local score = ranked_event_score(event)
    if score then
        retry_after_ms = math.max(retry_after_ms, score + long_window_ms - now_ms)
    end
end

if short_violated or long_violated then
    return {0, retry_after_ms, short_count, long_count, short_violated and 1 or 0, long_violated and 1 or 0}
end

redis.call("ZREMRANGEBYSCORE", bucket_key, "-inf", long_cutoff)
redis.call("ZADD", bucket_key, now_ms, member)
redis.call("PEXPIRE", bucket_key, long_window_ms + 1000)
return {1, 0, short_count + 1, long_count + 1, 0, 0}
"""


class RedisRequestRateLimiter:
    """Apply a two-window rolling limit atomically using a Redis sorted set."""

    def __init__(self, client: Any = redis_client):
        self._client = client
        self._backend_error_count = 0

    @property
    def backend_error_count(self) -> int:
        """Return the bounded-label process-local count of authoritative backend failures."""

        return self._backend_error_count

    @property
    def stats(self) -> dict[str, int]:
        """Return process-local limiter outcomes that remain available during Redis outages."""

        return {"backend_error_count": self._backend_error_count}

    async def verify(self) -> None:
        """Exercise the production Lua/ZSET/TTL path using an ephemeral bucket."""

        await self._client.ping()
        verification_key = f"{_BUCKET_KEY_PREFIX}:verify:{uuid.uuid4().hex}"
        short, long = DEFAULT_REQUEST_RATE_LIMIT_POLICY.windows
        try:
            result = await self._client.eval(
                _CHECK_RATE_LIMIT_LUA,
                1,
                verification_key,
                uuid.uuid4().hex,
                short.requests,
                short.seconds * 1000,
                long.requests,
                long.seconds * 1000,
                "",
            )
            if not result or int(result[0]) != 1:
                raise RuntimeError("request rate limiter verification was not admitted")
            if await self._client.zcard(verification_key) != 1:
                raise RuntimeError("request rate limiter verification did not write its event")
            pttl = await self._client.pttl(verification_key)
            expected_pttl = long.seconds * 1000 + 1000
            if not 0 < pttl <= expected_pttl:
                raise RuntimeError("request rate limiter verification did not set a TTL")
        except BaseException:
            try:
                await self._client.delete(verification_key)
            except Exception:
                logger.warning("Request rate-limit verification cleanup failed after verification error")
            raise
        else:
            await self._client.delete(verification_key)

    async def check(
        self,
        bucket_class: str,
        policy: RequestRateLimitPolicy,
        identity_kind: str | None = None,
        identity_subject_id: str | None = None,
    ) -> RateLimitDecision:
        return await self._check(
            bucket_class,
            policy,
            identity_kind=identity_kind,
            identity_subject_id=identity_subject_id,
            now_ms=None,
        )

    async def _check_at_ms(
        self,
        bucket_class: str,
        policy: RequestRateLimitPolicy,
        now_ms: int,
        identity_kind: str | None = None,
        identity_subject_id: str | None = None,
    ) -> RateLimitDecision:
        """Deterministic test seam; production checks always use Redis TIME."""

        if not isinstance(now_ms, int) or isinstance(now_ms, bool) or now_ms < 0:
            raise ValueError("now_ms must be a non-negative integer")
        return await self._check(
            bucket_class,
            policy,
            identity_kind=identity_kind,
            identity_subject_id=identity_subject_id,
            now_ms=now_ms,
        )

    async def _check(
        self,
        bucket_class: str,
        policy: RequestRateLimitPolicy,
        *,
        identity_kind: str | None,
        identity_subject_id: str | None,
        now_ms: int | None,
    ) -> RateLimitDecision:
        if not isinstance(policy, RequestRateLimitPolicy):
            raise TypeError("policy must be a RequestRateLimitPolicy")

        bucket_key = request_rate_limit_bucket_key(
            bucket_class,
            identity_kind=identity_kind,
            identity_subject_id=identity_subject_id,
        )
        short, long = policy.windows
        try:
            result = await self._client.eval(
                _CHECK_RATE_LIMIT_LUA,
                1,
                bucket_key,
                uuid.uuid4().hex,
                short.requests,
                short.seconds * 1000,
                long.requests,
                long.seconds * 1000,
                "" if now_ms is None else now_ms,
            )
        except Exception:
            self._backend_error_count += 1
            logger.warning("Request rate-limit backend check failed")
            raise
        allowed = bool(int(result[0]))
        violations = tuple(
            window for window, violated in zip(policy.windows, result[4:6], strict=True) if bool(int(violated))
        )
        decision = RateLimitDecision(
            allowed=allowed,
            retry_after_ms=max(0, int(result[1])),
            violated_windows=violations,
            counts=(int(result[2]), int(result[3])),
            bucket_class=bucket_class,
            policy=policy,
        )

        stats_field = f"{'admitted' if allowed else 'rejected'}:{bucket_class}"
        try:
            await self._client.hincrby(_STATS_KEY, stats_field, 1)
        except Exception:
            # Admission is authoritative; bounded aggregate telemetry is best effort.
            logger.debug("Request rate-limit aggregate stats update failed", exc_info=True)
        return decision
