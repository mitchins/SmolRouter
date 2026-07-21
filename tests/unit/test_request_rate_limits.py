import asyncio
import os
from types import SimpleNamespace
import uuid

import fakeredis.aioredis
import pytest
import pytest_asyncio
import redis.asyncio as redis

from smolrouter.request_rate_limits import (
    DEFAULT_REQUEST_RATE_LIMIT_POLICY,
    RateLimitWindow,
    RedisRequestRateLimiter,
    RequestRateLimitConfig,
    RequestRateLimitConfigError,
    RequestRateLimitPolicy,
    parse_request_rate_limit_config,
    parse_request_rate_limit_policy,
    request_rate_limit_bucket_key,
)


@pytest_asyncio.fixture
async def fake_redis():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


def policy(short_requests=1, short_seconds=1, long_requests=3, long_seconds=10):
    return RequestRateLimitPolicy(
        windows=(
            RateLimitWindow(short_requests, short_seconds),
            RateLimitWindow(long_requests, long_seconds),
        )
    )


def test_default_configuration_is_enabled_with_safe_two_window_limits():
    expected = RequestRateLimitConfig(
        enabled=True,
        anonymous=DEFAULT_REQUEST_RATE_LIMIT_POLICY,
        project_default=DEFAULT_REQUEST_RATE_LIMIT_POLICY,
    )

    assert parse_request_rate_limit_config(None) == expected
    assert parse_request_rate_limit_config({}) == expected
    assert DEFAULT_REQUEST_RATE_LIMIT_POLICY.to_dict() == {
        "windows": [{"requests": 1, "seconds": 1}, {"requests": 3, "seconds": 10}]
    }


def test_complete_configuration_parses_without_coercion():
    parsed = parse_request_rate_limit_config(
        {
            "enabled": False,
            "anonymous": {"windows": [{"requests": 2, "seconds": 3}, {"requests": 8, "seconds": 20}]},
            "project_default": {"windows": [{"requests": 10, "seconds": 1}, {"requests": 100, "seconds": 60}]},
        }
    )

    assert parsed.enabled is False
    assert parsed.anonymous.windows == (RateLimitWindow(2, 3), RateLimitWindow(8, 20))
    assert parsed.project_default.windows == (RateLimitWindow(10, 1), RateLimitWindow(100, 60))


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ([], "must be a mapping"),
        ({"enabled": 1}, "must be a boolean"),
        ({"enabled": "true"}, "must be a boolean"),
        ({"surprise": True}, "unknown keys"),
        ({"anonymous": None}, "must be a mapping"),
        ({"project_default": None}, "must be a mapping"),
        ({"anonymous": []}, "must be a mapping"),
        ({"anonymous": {}}, "windows is required"),
        ({"anonymous": {"windows": ()}}, "must be a list"),
        ({"anonymous": {"windows": []}}, "exactly 2"),
        (
            {"anonymous": {"windows": [{"requests": 1, "seconds": 1}, {"requests": 2, "seconds": 2}], "x": 1}},
            "unknown keys",
        ),
        (
            {"anonymous": {"windows": [{"requests": 1, "seconds": 1, "x": 2}, {"requests": 2, "seconds": 2}]}},
            "unknown keys",
        ),
        (
            {"anonymous": {"windows": [{"requests": 1}, {"requests": 2, "seconds": 2}]}},
            "missing required keys",
        ),
        (
            {"anonymous": {"windows": [{"requests": True, "seconds": 1}, {"requests": 2, "seconds": 2}]}},
            "positive integer",
        ),
        (
            {"anonymous": {"windows": [{"requests": 1, "seconds": 1.0}, {"requests": 2, "seconds": 2}]}},
            "positive integer",
        ),
        (
            {"anonymous": {"windows": [{"requests": 0, "seconds": 1}, {"requests": 2, "seconds": 2}]}},
            "positive integer",
        ),
        (
            {"anonymous": {"windows": [{"requests": 1, "seconds": 2}, {"requests": 2, "seconds": 2}]}},
            "strictly ascending",
        ),
        (
            {"anonymous": {"windows": [{"requests": 3, "seconds": 1}, {"requests": 2, "seconds": 2}]}},
            "monotonic",
        ),
    ],
)
def test_configuration_schema_is_strict(raw, error):
    with pytest.raises(ValueError, match=error):
        parse_request_rate_limit_config(raw)


def test_policy_none_only_uses_an_explicit_default():
    with pytest.raises(RequestRateLimitConfigError, match="must be a mapping"):
        parse_request_rate_limit_policy(None, "project.rate_limit")

    assert (
        parse_request_rate_limit_policy(None, "project.rate_limit", DEFAULT_REQUEST_RATE_LIMIT_POLICY)
        is DEFAULT_REQUEST_RATE_LIMIT_POLICY
    )


def test_config_errors_have_a_dedicated_fail_fast_type():
    with pytest.raises(RequestRateLimitConfigError, match="enabled"):
        parse_request_rate_limit_config({"enabled": "yes"})


def test_bucket_keys_are_secret_free_stable_and_isolated():
    anonymous_key = request_rate_limit_bucket_key("anonymous")
    alpha_key = request_rate_limit_bucket_key(
        "project", identity_kind="facade_key", identity_subject_id="sk-secret-alpha"
    )
    alpha_again = request_rate_limit_bucket_key(
        "project", identity_kind="facade_key", identity_subject_id="sk-secret-alpha"
    )
    beta_key = request_rate_limit_bucket_key(
        "project", identity_kind="facade_key", identity_subject_id="sk-secret-beta"
    )

    assert anonymous_key != alpha_key
    assert alpha_key == alpha_again
    assert alpha_key != beta_key
    assert "facade_key" not in alpha_key
    assert "sk-secret-alpha" not in alpha_key
    assert len(alpha_key.rsplit(":", 1)[-1]) == 64


@pytest.mark.parametrize(
    ("bucket_class", "kind", "subject"),
    [
        ("other", None, None),
        ("anonymous", "kind", None),
        ("project", None, "subject"),
        ("project", "", "subject"),
        ("project", "kind", ""),
    ],
)
def test_bucket_identity_validation(bucket_class, kind, subject):
    with pytest.raises(ValueError):
        request_rate_limit_bucket_key(
            bucket_class,
            identity_kind=kind,
            identity_subject_id=subject,
        )


@pytest.mark.asyncio
async def test_real_lua_script_enforces_exclusive_window_boundaries(fake_redis):
    limiter = RedisRequestRateLimiter(fake_redis)
    limits = policy(1, 1, 3, 10)

    first = await limiter._check_at_ms("anonymous", limits, 1_000)
    denied = await limiter._check_at_ms("anonymous", limits, 1_999)
    boundary = await limiter._check_at_ms("anonymous", limits, 2_000)

    assert first.allowed is True
    assert denied.allowed is False
    assert denied.retry_after_ms == 1
    assert denied.retry_after_seconds == 1
    assert denied.violated_windows == (limits.windows[0],)
    assert boundary.allowed is True
    assert boundary.counts == (1, 2)


@pytest.mark.asyncio
async def test_long_window_boundary_is_exclusive(fake_redis):
    limiter = RedisRequestRateLimiter(fake_redis)
    limits = policy(2, 1, 3, 10)
    for timestamp in (1_000, 2_000, 3_000):
        assert (await limiter._check_at_ms("anonymous", limits, timestamp)).allowed

    denied = await limiter._check_at_ms("anonymous", limits, 10_999)
    boundary = await limiter._check_at_ms("anonymous", limits, 11_000)

    assert denied.allowed is False
    assert denied.retry_after_ms == 1
    assert denied.violated_windows == (limits.windows[1],)
    assert boundary.allowed is True


@pytest.mark.asyncio
async def test_concurrent_checks_never_over_admit(fake_redis):
    limiter = RedisRequestRateLimiter(fake_redis)
    limits = policy(3, 1, 3, 10)

    decisions = await asyncio.gather(*(limiter._check_at_ms("anonymous", limits, 5_000) for _ in range(30)))

    assert sum(decision.allowed for decision in decisions) == 3
    bucket = request_rate_limit_bucket_key("anonymous")
    assert await fake_redis.zcard(bucket) == 3


@pytest.mark.asyncio
async def test_denial_does_not_charge_or_change_bucket(fake_redis):
    limiter = RedisRequestRateLimiter(fake_redis)
    limits = policy(1, 1, 3, 10)
    await limiter._check_at_ms("anonymous", limits, 1_000)
    bucket = request_rate_limit_bucket_key("anonymous")
    before = await fake_redis.zrange(bucket, 0, -1, withscores=True)

    for timestamp in (1_100, 1_200, 1_300):
        assert not (await limiter._check_at_ms("anonymous", limits, timestamp)).allowed

    assert await fake_redis.zrange(bucket, 0, -1, withscores=True) == before


@pytest.mark.asyncio
async def test_uuid_members_prevent_same_millisecond_collisions(fake_redis, monkeypatch):
    ids = iter((SimpleNamespace(hex=f"request-{index}") for index in range(5)))
    monkeypatch.setattr("smolrouter.request_rate_limits.uuid.uuid4", lambda: next(ids))
    limiter = RedisRequestRateLimiter(fake_redis)
    limits = policy(5, 1, 5, 10)

    for _ in range(5):
        assert (await limiter._check_at_ms("anonymous", limits, 1_000)).allowed

    bucket = request_rate_limit_bucket_key("anonymous")
    assert await fake_redis.zcard(bucket) == 5


@pytest.mark.asyncio
async def test_admission_prunes_cardinality_and_sets_max_window_ttl(fake_redis):
    limiter = RedisRequestRateLimiter(fake_redis)
    limits = policy(2, 1, 3, 10)
    bucket = request_rate_limit_bucket_key("anonymous")
    for timestamp in (1_000, 2_000, 3_000):
        assert (await limiter._check_at_ms("anonymous", limits, timestamp)).allowed

    assert await fake_redis.zcard(bucket) == 3
    assert 10_900 <= await fake_redis.pttl(bucket) <= 11_000

    assert (await limiter._check_at_ms("anonymous", limits, 11_000)).allowed
    assert await fake_redis.zcard(bucket) == 3


@pytest.mark.asyncio
async def test_lowered_policy_uses_c_minus_l_event_rank_for_retry(fake_redis):
    limiter = RedisRequestRateLimiter(fake_redis)
    initial = policy(10, 1, 10, 10)
    for timestamp in (100, 200, 300, 400, 500):
        assert (await limiter._check_at_ms("anonymous", initial, timestamp)).allowed

    lowered = policy(2, 1, 10, 10)
    denied = await limiter._check_at_ms("anonymous", lowered, 600)

    assert denied.allowed is False
    assert denied.counts == (5, 5)
    assert denied.retry_after_ms == 800  # rank 5 - 2 => event at 400, expiring at 1400
    assert denied.retry_after_seconds == 1


@pytest.mark.asyncio
async def test_retry_is_maximum_wait_across_all_violated_windows(fake_redis):
    limiter = RedisRequestRateLimiter(fake_redis)
    initial = policy(10, 1, 10, 10)
    for timestamp in (100, 200, 300, 400, 500):
        assert (await limiter._check_at_ms("anonymous", initial, timestamp)).allowed

    lowered = policy(2, 1, 3, 10)
    denied = await limiter._check_at_ms("anonymous", lowered, 600)

    assert denied.violated_windows == lowered.windows
    assert denied.retry_after_ms == 9_700  # long rank 5 - 3 => event at 300
    assert denied.retry_after_seconds == 10


@pytest.mark.asyncio
async def test_project_buckets_are_isolated(fake_redis):
    limiter = RedisRequestRateLimiter(fake_redis)
    limits = policy(1, 1, 3, 10)

    alpha = await limiter._check_at_ms("project", limits, 1_000, "facade_key", "alpha")
    alpha_denied = await limiter._check_at_ms("project", limits, 1_000, "facade_key", "alpha")
    beta = await limiter._check_at_ms("project", limits, 1_000, "facade_key", "beta")

    assert alpha.allowed is True
    assert alpha_denied.allowed is False
    assert beta.allowed is True


@pytest.mark.asyncio
async def test_stats_use_only_bounded_fields_and_are_best_effort(fake_redis, monkeypatch, caplog):
    limiter = RedisRequestRateLimiter(fake_redis)
    limits = policy(1, 1, 3, 10)
    await limiter._check_at_ms("anonymous", limits, 1_000)
    await limiter._check_at_ms("anonymous", limits, 1_000)
    await limiter._check_at_ms("project", limits, 1_000, "kind", "one")
    await limiter._check_at_ms("project", limits, 1_000, "kind", "one")

    stats = await fake_redis.hgetall("smolrouter:request-rate-limit:stats")
    assert stats == {
        "admitted:anonymous": "1",
        "rejected:anonymous": "1",
        "admitted:project": "1",
        "rejected:project": "1",
    }

    async def broken_stats(*_args, **_kwargs):
        raise RuntimeError("telemetry unavailable")

    monkeypatch.setattr(fake_redis, "hincrby", broken_stats)
    with caplog.at_level("DEBUG", logger="smolrouter.request_rate_limits"):
        assert (await limiter._check_at_ms("anonymous", limits, 2_000)).allowed
    assert "aggregate stats update failed" in caplog.text


@pytest.mark.asyncio
async def test_backend_failure_is_counted_and_raised_without_logging_credentials(fake_redis, monkeypatch, caplog):
    limiter = RedisRequestRateLimiter(fake_redis)

    async def broken_eval(*_args, **_kwargs):
        raise RuntimeError("redis unavailable for credential sk-must-not-leak")

    monkeypatch.setattr(fake_redis, "eval", broken_eval)
    assert limiter.backend_error_count == 0
    assert limiter.stats == {"backend_error_count": 0}
    with caplog.at_level("WARNING", logger="smolrouter.request_rate_limits"):
        with pytest.raises(RuntimeError, match="redis unavailable"):
            await limiter.check("anonymous", DEFAULT_REQUEST_RATE_LIMIT_POLICY)

    assert limiter.backend_error_count == 1
    assert limiter.stats == {"backend_error_count": 1}
    assert "backend check failed" in caplog.text
    assert "sk-must-not-leak" not in caplog.text


@pytest.mark.asyncio
async def test_verify_exercises_production_lua_and_cleans_up(fake_redis):
    await RedisRequestRateLimiter(fake_redis).verify()
    assert await fake_redis.keys("*") == []


@pytest.mark.asyncio
async def test_verify_cleanup_does_not_mask_primary_error(fake_redis, monkeypatch):
    limiter = RedisRequestRateLimiter(fake_redis)

    async def broken_eval(*_args, **_kwargs):
        raise RuntimeError("primary verification failure")

    async def broken_delete(*_args, **_kwargs):
        raise ConnectionError("cleanup failure")

    monkeypatch.setattr(fake_redis, "eval", broken_eval)
    monkeypatch.setattr(fake_redis, "delete", broken_delete)

    with pytest.raises(RuntimeError, match="primary verification failure"):
        await limiter.verify()


class _RealRedisWithoutAggregateStats:
    """Delegate Redis semantics while keeping an optional shared test DB free of stats changes."""

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        return getattr(self._client, name)

    async def hincrby(self, *_args, **_kwargs):
        return 0


@pytest.mark.asyncio
async def test_semantic_suite_against_real_redis_when_explicitly_configured():
    redis_url = os.getenv("SMOLROUTER_TEST_REDIS_URL")
    if not redis_url:
        pytest.skip("set SMOLROUTER_TEST_REDIS_URL to exercise a disposable real Redis")

    raw_client = redis.from_url(redis_url, decode_responses=True)
    client = _RealRedisWithoutAggregateStats(raw_client)
    limiter = RedisRequestRateLimiter(client)
    suite_id = uuid.uuid4().hex
    identities = {
        name: ("real_redis_test", f"{suite_id}:{name}")
        for name in ("concurrency", "boundary", "lowered", "cardinality")
    }
    bucket_keys = [
        request_rate_limit_bucket_key("project", identity_kind=kind, identity_subject_id=subject)
        for kind, subject in identities.values()
    ]
    try:
        await limiter.verify()

        kind, subject = identities["concurrency"]
        concurrency_policy = policy(3, 1, 3, 10)
        decisions = await asyncio.gather(
            *(limiter._check_at_ms("project", concurrency_policy, 5_000, kind, subject) for _ in range(30))
        )
        assert sum(decision.allowed for decision in decisions) == 3
        concurrency_key = request_rate_limit_bucket_key("project", identity_kind=kind, identity_subject_id=subject)
        before_deny = await raw_client.zrange(concurrency_key, 0, -1, withscores=True)
        assert not (await limiter._check_at_ms("project", concurrency_policy, 5_001, kind, subject)).allowed
        assert await raw_client.zrange(concurrency_key, 0, -1, withscores=True) == before_deny

        kind, subject = identities["boundary"]
        boundary_policy = policy(1, 1, 3, 10)
        assert (await limiter._check_at_ms("project", boundary_policy, 1_000, kind, subject)).allowed
        just_inside = await limiter._check_at_ms("project", boundary_policy, 1_999, kind, subject)
        at_boundary = await limiter._check_at_ms("project", boundary_policy, 2_000, kind, subject)
        assert not just_inside.allowed and just_inside.retry_after_ms == 1
        assert at_boundary.allowed

        kind, subject = identities["lowered"]
        initial = policy(10, 1, 10, 10)
        for timestamp in (100, 200, 300, 400, 500):
            assert (await limiter._check_at_ms("project", initial, timestamp, kind, subject)).allowed
        lowered = await limiter._check_at_ms("project", policy(2, 1, 10, 10), 600, kind, subject)
        assert not lowered.allowed
        assert lowered.counts == (5, 5)
        assert lowered.retry_after_ms == 800

        kind, subject = identities["cardinality"]
        cardinality_policy = policy(2, 1, 3, 10)
        cardinality_key = request_rate_limit_bucket_key("project", identity_kind=kind, identity_subject_id=subject)
        for timestamp in (1_000, 2_000, 3_000):
            assert (await limiter._check_at_ms("project", cardinality_policy, timestamp, kind, subject)).allowed
        assert await raw_client.zcard(cardinality_key) == 3
        assert 10_900 <= await raw_client.pttl(cardinality_key) <= 11_000
        assert (await limiter._check_at_ms("project", cardinality_policy, 11_000, kind, subject)).allowed
        assert await raw_client.zcard(cardinality_key) == 3
    finally:
        await raw_client.delete(*bucket_keys)
        await raw_client.aclose()
