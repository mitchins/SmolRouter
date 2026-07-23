import json
from datetime import datetime, timedelta, timezone

import pytest

import smolrouter.redis_backend as redis_backend_module
import smolrouter.storage as storage_module
from smolrouter.redis_backend import LogRecord, QuotaRecord, RedisApiKeyQuota, RedisRequestLog, _flat_pairs_to_dict
from smolrouter.redis_config import redis_client


class FakeBlobStorage:
    def __init__(self, blobs):
        self.blobs = blobs

    def retrieve(self, key):
        return self.blobs.get(key)


def test_log_record_parses_fields_and_loads_blob_bodies(monkeypatch):
    monkeypatch.setattr(
        storage_module,
        "get_blob_storage",
        lambda: FakeBlobStorage({"req-body": b'{"input": true}', "resp-body": b'{"output": true}'}),
    )

    record = LogRecord(
        {
            "request_id": "req-1",
            "duration_ms": "123",
            "status_code": "200",
            "request_size": "45",
            "response_size": "67",
            "prompt_tokens": "10",
            "completion_tokens": "3",
            "total_tokens": "13",
            "created_at": "2026-04-25T01:23:45Z",
            "completed_at": "2026-04-25T01:24:00Z",
            "duplicate_count": "2",
            "is_duplicate": "true",
            "request_body_key": "req-body",
            "response_body_key": "resp-body",
        }
    )

    assert record.id == "req-1"
    assert record.duration_ms == 123
    assert record.status_code == 200
    assert record.request_size == 45
    assert record.response_size == 67
    assert record.prompt_tokens == 10
    assert record.completion_tokens == 3
    assert record.total_tokens == 13
    assert record.timestamp == datetime(2026, 4, 25, 1, 23, 45, tzinfo=timezone.utc)
    assert record.completed_at == datetime(2026, 4, 25, 1, 24, 0, tzinfo=timezone.utc)
    assert record.duplicate_count == 2
    assert record.is_duplicate is True
    assert record.request_body == b'{"input": true}'
    assert record.response_body == b'{"output": true}'


def test_log_record_marks_missing_blob_storage_as_storage_error(monkeypatch):
    monkeypatch.setattr(storage_module, "get_blob_storage", lambda: None)

    record = LogRecord(
        {
            "request_id": "req-1",
            "request_body_key": "req-body",
            "response_body_key": "resp-body",
        }
    )

    assert record.request_body is None
    assert record.response_body is None
    assert record.request_body_status == "storage_error"
    assert record.response_body_status == "storage_error"


def test_log_record_normalizes_pending_and_invalid_values(monkeypatch):
    monkeypatch.setattr(storage_module, "get_blob_storage", lambda: FakeBlobStorage({}))

    record = LogRecord(
        {
            "request_id": "req-2",
            "duration_ms": "not-a-number",
            "status_code": "pending",
            "request_size": "",
            "response_size": "bad",
            "prompt_tokens": "",
            "completion_tokens": "bad",
            "total_tokens": "None",
            "created_at": "not-a-date",
            "completed_at": "not-a-date",
            "duplicate_count": "oops",
            "is_duplicate": "no",
        }
    )

    assert record.id == "req-2"
    assert record.duration_ms is None
    assert record.status_code == "pending"
    assert record.request_size == 0
    assert record.response_size == 0
    assert record.prompt_tokens is None
    assert record.completion_tokens is None
    assert record.total_tokens is None
    assert isinstance(record.timestamp, datetime)
    assert record.timestamp.tzinfo is not None
    assert record.completed_at is None
    assert record.duplicate_count == 0
    assert record.is_duplicate is False
    assert record.request_body is None
    assert record.response_body is None


def test_flat_pairs_to_dict_ignores_unpaired_tail_value():
    assert _flat_pairs_to_dict(["request_id", "req-1", "orphan"]) == {"request_id": "req-1"}


def test_quota_record_parses_booleans_and_timestamps():
    quota = QuotaRecord(
        {
            "requests_today": "5",
            "tokens_today": "1000",
            "error_count": "2",
            "invalid_key": "true",
            "updated_at": "2026-04-25T03:00:00Z",
            "quota_exhausted_at": "2026-04-25T04:00:00Z",
            "key_hash": "hash-1",
            "model_name": "gemini-2.5-flash",
        }
    )

    assert quota.requests_today == 5
    assert quota.tokens_today == 1000
    assert quota.error_count == 2
    assert quota.invalid_key is True
    assert quota.api_key_hash == "hash-1"
    assert quota.model_name == "gemini-2.5-flash"
    assert quota.updated_at == datetime(2026, 4, 25, 3, 0, 0, tzinfo=timezone.utc)
    assert quota.quota_exhausted_at == datetime(2026, 4, 25, 4, 0, 0, tzinfo=timezone.utc)


def test_quota_record_hydrates_last_reset_and_key_hash_aliases():
    quota = QuotaRecord(
        {
            "requests_today": "1",
            "tokens_today": "2",
            "error_count": "0",
            "invalid_key": "false",
            "last_reset": "2026-04-25",
            "key_hash": "hash-alias",
            "model_name": "gemini-2.5-flash",
        }
    )

    assert quota.last_reset_date == "2026-04-25"
    assert quota.api_key_hash == "hash-alias"
    assert quota.invalid_key is False


def test_quota_record_failure_tracks_last_error_and_quota_exhaustion():
    quota = QuotaRecord({"requests_today": "1", "tokens_today": "10", "error_count": "0"})

    quota.mark_request_failure(error="quota hit", quota_exhausted=True)

    assert quota.error_count == 1
    assert quota.last_error == "quota hit"
    assert isinstance(quota.quota_exhausted_at, datetime)
    assert quota.quota_exhausted_at.tzinfo is not None


@pytest.mark.asyncio
async def test_request_log_create_rejects_unexpected_fields():
    await redis_client.flushall()

    with pytest.raises(TypeError, match="Unexpected request log create fields: unexpected_field"):
        await RedisRequestLog.create(
            source_ip="127.0.0.1",
            method="GET",
            path="/health",
            unexpected_field="boom",
        )


@pytest.mark.asyncio
async def test_request_log_update_completion_rejects_unexpected_fields():
    await redis_client.flushall()
    request_id = await RedisRequestLog.create(source_ip="127.0.0.1", method="GET", path="/health")

    with pytest.raises(TypeError, match="Unexpected request completion fields: unexpected_field"):
        await RedisRequestLog.update_completion(
            request_id=request_id,
            status_code=200,
            unexpected_field="boom",
        )


@pytest.mark.asyncio
async def test_request_log_update_completion_is_idempotent_for_retries():
    await redis_client.flushall()
    request_id = await RedisRequestLog.create(source_ip="127.0.0.1", method="POST", path="/v1/chat/completions")

    await RedisRequestLog.update_completion(
        request_id=request_id,
        status_code=200,
        response_size=12,
    )
    await RedisRequestLog.update_completion(
        request_id=request_id,
        status_code=200,
        response_size=24,
    )

    stats = await RedisRequestLog.get_stats_counters()
    assert stats["total"] == 1
    assert stats["completed"] == 1
    assert stats["failed"] == 0
    assert stats["inflight"] == 0


@pytest.mark.asyncio
async def test_request_log_round_trip_preserves_recency_and_duplicate_indexes(monkeypatch):
    monkeypatch.setattr(
        storage_module,
        "get_blob_storage",
        lambda: FakeBlobStorage({"req-blob": b'{"prompt": "hi"}', "resp-blob": b'{"ok": true}'}),
    )
    await redis_client.flushall()

    first_time = datetime(2026, 4, 25, 1, 0, 0, tzinfo=timezone.utc)
    second_time = first_time + timedelta(seconds=5)

    first_id = await RedisRequestLog.create(
        source_ip="192.168.1.100",
        method="POST",
        path="/v1/chat/completions",
        request_id="req-1",
        timestamp=first_time,
        request_body_hash="body-hash",
    )
    second_id = await RedisRequestLog.create(
        source_ip="192.168.1.100",
        method="POST",
        path="/v1/chat/completions",
        request_id="req-2",
        timestamp=second_time,
        request_body_hash="body-hash",
    )

    await RedisRequestLog.update_request_body_key(second_id, "req-blob")
    await RedisRequestLog.update_completion(
        request_id=second_id,
        status_code=201,
        response_size=12,
        response_body_key="resp-blob",
        provider_id="provider-a",
        api_key_index=1,
        api_key_total=3,
    )

    record = await RedisRequestLog.get_by_id(second_id)
    recent = await RedisRequestLog.get_recent(limit=2)
    by_source_ip = await RedisRequestLog.get_by_source_ip("192.168.1.100", limit=1)
    duplicate_ids = await RedisRequestLog.get_duplicate_request_ids("body-hash")
    recent_duplicate_ids = await RedisRequestLog.get_recent_duplicate_request_ids("body-hash", limit=1)

    assert record is not None
    assert record.request_body == b'{"prompt": "hi"}'
    assert record.response_body == b'{"ok": true}'
    assert record.provider_id == "provider-a"
    assert record.is_duplicate is True
    assert record.duplicate_count == 1
    assert [item.id for item in recent] == [second_id, first_id]
    assert [item.id for item in by_source_ip] == [second_id]
    assert set(duplicate_ids) == {first_id, second_id}
    assert recent_duplicate_ids == [second_id]


@pytest.mark.asyncio
async def test_request_log_get_by_source_ip_with_limit_avoids_full_set_fetch(monkeypatch):
    await redis_client.flushall()
    request_id = await RedisRequestLog.create(
        source_ip="192.168.1.100",
        method="GET",
        path="/health",
        request_id="req-source-ip-limited",
    )

    async def failing_smembers(_key):
        raise AssertionError("limited get_by_source_ip should not use SMEMBERS")

    monkeypatch.setattr(redis_client, "smembers", failing_smembers)

    records = await RedisRequestLog.get_by_source_ip("192.168.1.100", limit=1)

    assert [record.id for record in records] == [request_id]


@pytest.mark.asyncio
async def test_request_log_create_indexes_by_identity_and_get_by_identity_is_ordered():
    await redis_client.flushall()

    now = datetime(2026, 4, 25, 10, 0, 0, tzinfo=timezone.utc)
    later = now + timedelta(minutes=10)

    await RedisRequestLog.create(
        source_ip="127.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        request_id="req-project-a-1",
        timestamp=now,
        identity_kind="facade_key",
        identity_subject_id="project-a",
    )
    await RedisRequestLog.create(
        source_ip="127.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        request_id="req-other-project",
        timestamp=now + timedelta(seconds=1),
        identity_kind="facade_key",
        identity_subject_id="project-b",
    )
    await RedisRequestLog.create(
        source_ip="127.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        request_id="req-project-a-2",
        timestamp=later,
        identity_kind="facade_key",
        identity_subject_id="project-a",
    )

    records = await RedisRequestLog.get_by_identity("facade_key", "project-a", limit=1)
    assert [record.id for record in records] == ["req-project-a-2"]

    records = await RedisRequestLog.get_by_identity("facade_key", "project-a", limit=5)
    assert [record.id for record in records] == ["req-project-a-2", "req-project-a-1"]

    records_other = await RedisRequestLog.get_by_identity("facade_key", "project-b", limit=5)
    assert [record.id for record in records_other] == ["req-other-project"]


@pytest.mark.asyncio
async def test_request_log_get_by_identity_prunes_stale_members_and_backfills_results():
    await redis_client.flushall()

    now = datetime(2026, 4, 25, 10, 0, 0, tzinfo=timezone.utc)
    later = now + timedelta(minutes=10)
    latest = later + timedelta(minutes=10)

    await RedisRequestLog.create(
        source_ip="127.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        request_id="req-project-a-1",
        timestamp=now,
        identity_kind="facade_key",
        identity_subject_id="project-a",
    )
    await RedisRequestLog.create(
        source_ip="127.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        request_id="req-project-a-2",
        timestamp=later,
        identity_kind="facade_key",
        identity_subject_id="project-a",
    )
    await RedisRequestLog.create(
        source_ip="127.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        request_id="req-project-a-3",
        timestamp=latest,
        identity_kind="facade_key",
        identity_subject_id="project-a",
    )

    await redis_client.delete("request:req-project-a-3")

    records = await RedisRequestLog.get_by_identity("facade_key", "project-a", limit=2)

    assert [record.id for record in records] == ["req-project-a-2", "req-project-a-1"]
    assert await redis_client.zrevrange("requests:by_identity:facade_key:project-a", 0, -1) == [
        "req-project-a-2",
        "req-project-a-1",
    ]


@pytest.mark.asyncio
async def test_request_log_get_by_identity_does_not_skip_live_rows_after_pruning_stale_pages():
    await redis_client.flushall()

    base_time = datetime(2026, 4, 25, 10, 0, 0, tzinfo=timezone.utc)

    for offset in range(30):
        request_number = offset + 1
        request_id = f"req-project-a-{request_number:02d}"
        await RedisRequestLog.create(
            source_ip="127.0.0.1",
            method="POST",
            path="/v1/chat/completions",
            request_id=request_id,
            timestamp=base_time + timedelta(seconds=offset),
            identity_kind="facade_key",
            identity_subject_id="project-a",
        )

    for request_number in range(10, 31):
        await redis_client.delete(f"request:req-project-a-{request_number:02d}")

    records = await RedisRequestLog.get_by_identity("facade_key", "project-a", limit=5)

    assert [record.id for record in records] == [
        "req-project-a-09",
        "req-project-a-08",
        "req-project-a-07",
        "req-project-a-06",
        "req-project-a-05",
    ]
    assert await redis_client.zrevrange("requests:by_identity:facade_key:project-a", 0, -1) == [
        "req-project-a-09",
        "req-project-a-08",
        "req-project-a-07",
        "req-project-a-06",
        "req-project-a-05",
        "req-project-a-04",
        "req-project-a-03",
        "req-project-a-02",
        "req-project-a-01",
    ]


@pytest.mark.asyncio
async def test_request_log_get_recent_falls_back_when_lua_fails(monkeypatch):
    await redis_client.flushall()
    request_id = await RedisRequestLog.create(
        source_ip="192.168.1.100",
        method="GET",
        path="/health",
        request_id="req-lua-fallback",
    )
    original_eval = redis_client.eval

    async def failing_eval(*_args, **_kwargs):
        raise RuntimeError("lua unavailable")

    monkeypatch.setattr(redis_client, "eval", failing_eval)
    monkeypatch.setenv("REDIS_DISABLE_LUA", "false")

    recent = await RedisRequestLog.get_recent(limit=1)

    monkeypatch.setattr(redis_client, "eval", original_eval)
    assert [record.id for record in recent] == [request_id]


@pytest.mark.asyncio
async def test_request_log_create_falls_back_when_duplicate_count_fails(monkeypatch, caplog):
    await redis_client.flushall()

    async def failing_scard(_key):
        raise RuntimeError("scard unavailable")

    monkeypatch.setattr(redis_client, "scard", failing_scard)
    caplog.set_level("DEBUG", logger=redis_backend_module.logger.name)

    request_id = await RedisRequestLog.create(
        source_ip="192.168.1.100",
        method="POST",
        path="/v1/chat/completions",
        request_id="req-scard-failure",
        request_body_hash="body-hash",
    )

    record = await RedisRequestLog.get_by_id(request_id)

    assert record is not None
    assert record.is_duplicate is False
    assert record.duplicate_count == 0
    assert "Failed to get duplicate count for requests:by_body:body-hash" in caplog.text


@pytest.mark.asyncio
async def test_api_key_quota_round_trip_and_usage_markers(monkeypatch):
    await redis_client.flushall()
    monkeypatch.setattr(redis_backend_module, "_circuit_breaker", redis_backend_module.RedisCircuitBreaker())
    monkeypatch.setattr(RedisApiKeyQuota, "_script_initialized", False)
    monkeypatch.setattr(RedisApiKeyQuota, "_script_sha", None)
    monkeypatch.setattr(RedisApiKeyQuota, "_lua_disabled", True)

    quota_a, created_a = await RedisApiKeyQuota.get_or_create_quota("sk-test", "provider-a", "gemini-2.5-flash")
    quota_b, created_b = await RedisApiKeyQuota.get_or_create_quota("sk-test", "provider-b", "gemini-2.5-flash")
    updated = await RedisApiKeyQuota.increment_usage(
        "sk-test",
        "provider-a",
        "gemini-2.5-flash",
        request_count=2,
        token_count=50,
    )
    existing_a, was_created = await RedisApiKeyQuota.get_or_create_quota("sk-test", "provider-a", "gemini-2.5-flash")

    await RedisApiKeyQuota.mark_error("sk-test", "provider-a", "gemini-2.5-flash", "temporary failure")
    await RedisApiKeyQuota.mark_quota_exhausted("sk-test", "provider-a", "gemini-2.5-flash", "daily limit")
    marked_invalid = await RedisApiKeyQuota.mark_invalid(RedisApiKeyQuota.hash_api_key("sk-test"), "provider-a")

    provider_a_usage = await RedisApiKeyQuota.get_provider_usage("provider-a")
    provider_b_usage = await RedisApiKeyQuota.get_provider_usage("provider-b")

    assert created_a is True
    assert created_b is True
    assert quota_a.requests_today == 0
    assert quota_b.requests_today == 0
    assert updated["requests_today"] == 2
    assert updated["tokens_today"] == 50
    assert was_created is False
    assert existing_a.requests_today == 2
    assert existing_a.tokens_today == 50
    assert marked_invalid == 1
    assert len(provider_a_usage) == 1
    assert provider_a_usage[0].error_count == 2
    assert provider_a_usage[0].last_error == "daily limit"
    assert provider_a_usage[0].invalid_key is True
    assert provider_a_usage[0].quota_exhausted_at is not None
    assert await redis_client.sismember(
        RedisApiKeyQuota.google_invalid_keys_key("provider-a"),
        RedisApiKeyQuota.hash_api_key("sk-test"),
    )
    assert len(provider_b_usage) == 1
    assert provider_b_usage[0].invalid_key is False


@pytest.mark.asyncio
async def test_invalid_key_recovery_is_provider_scoped_and_preserves_quota_state(monkeypatch):
    await redis_client.flushall()
    monkeypatch.setattr(redis_backend_module, "_circuit_breaker", redis_backend_module.RedisCircuitBreaker())
    monkeypatch.setattr(RedisApiKeyQuota, "_script_initialized", False)
    monkeypatch.setattr(RedisApiKeyQuota, "_script_sha", None)
    monkeypatch.setattr(RedisApiKeyQuota, "_lua_disabled", True)

    await RedisApiKeyQuota.get_or_create_quota("sk-recover", "provider-a", "model-a")
    await RedisApiKeyQuota.get_or_create_quota("sk-recover", "provider-a", "model-b")
    await RedisApiKeyQuota.get_or_create_quota("sk-recover", "provider-b", "model-a")
    await RedisApiKeyQuota.increment_usage("sk-recover", "provider-a", "model-a", request_count=2)
    key_hash = RedisApiKeyQuota.hash_api_key("sk-recover")

    assert await RedisApiKeyQuota.mark_invalid(
        key_hash, "provider-a", reason="api_key_invalid", status_code=400, request_id="request-1"
    ) == 2
    assert await RedisApiKeyQuota.mark_invalid(
        key_hash, "provider-a", reason="api_key_revoked", status_code=401, request_id="request-2"
    ) == 2

    metadata_key = RedisApiKeyQuota.google_invalid_key_metadata_key("provider-a", key_hash)
    metadata = await redis_client.hgetall(metadata_key)
    assert metadata["first_reason"] == "api_key_invalid"
    assert metadata["first_status_code"] == "400"
    assert metadata["first_request_id"] == "request-1"
    assert metadata["occurrence_count"] == "2"
    assert metadata["latest_reason"] == "api_key_revoked"
    assert metadata["latest_status_code"] == "401"
    assert metadata["latest_request_id"] == "request-2"

    assert await RedisApiKeyQuota.recover_invalid(
        key_hash, "provider-a", actor="operator@example.test", reason="false positive"
    ) == 2

    provider_a_usage = await RedisApiKeyQuota.get_provider_usage("provider-a")
    provider_b_usage = await RedisApiKeyQuota.get_provider_usage("provider-b")
    assert {quota.model_name for quota in provider_a_usage if quota.invalid_key} == set()
    assert next(quota for quota in provider_a_usage if quota.model_name == "model-a").requests_today == 2
    assert provider_b_usage[0].invalid_key is False
    assert await redis_client.sismember(RedisApiKeyQuota.google_invalid_keys_key("provider-a"), key_hash) == 0
    assert await redis_client.exists(metadata_key) == 0

    audit = json.loads(await redis_client.lindex("google_invalid_key_recovery_audit:provider-a", 0))
    assert audit["actor"] == "operator@example.test"
    assert audit["reason"] == "false positive"
    assert audit["prior_metadata"]["occurrence_count"] == "2"


@pytest.mark.asyncio
async def test_mark_quota_cooldown_round_trips_and_clears_on_daily_reset(monkeypatch):
    await redis_client.flushall()
    monkeypatch.setattr(redis_backend_module, "_circuit_breaker", redis_backend_module.RedisCircuitBreaker())
    monkeypatch.setattr(RedisApiKeyQuota, "_script_initialized", False)
    monkeypatch.setattr(RedisApiKeyQuota, "_script_sha", None)
    monkeypatch.setattr(RedisApiKeyQuota, "_google_selector_script_sha", None)
    monkeypatch.setattr(RedisApiKeyQuota, "_lua_disabled", True)

    await RedisApiKeyQuota.get_or_create_quota("sk-cool", "provider-c", "gemini-3.1-flash-lite")

    cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=21)
    await RedisApiKeyQuota.mark_quota_cooldown(
        "sk-cool", "provider-c", "gemini-3.1-flash-lite", cooldown_until, "per-minute limit"
    )

    usage = await RedisApiKeyQuota.get_provider_usage("provider-c")
    assert len(usage) == 1
    assert usage[0].quota_cooldown_until is not None
    # Transient rate limit must NOT bench the key for the whole day.
    assert usage[0].quota_exhausted_at is None
    # Cooldown does not count toward the error-prone ban threshold.
    assert usage[0].error_count == 0

    # A request on a new Pacific day clears the cooldown via the daily-reset path.
    quota_key = "quota:provider-c:" + RedisApiKeyQuota.hash_api_key("sk-cool") + ":gemini-3.1-flash-lite"
    await redis_client.hset(quota_key, "last_reset", "1999-01-01")
    await RedisApiKeyQuota.increment_usage("sk-cool", "provider-c", "gemini-3.1-flash-lite")

    usage_after = await RedisApiKeyQuota.get_provider_usage("provider-c")
    assert usage_after[0].quota_cooldown_until is None


@pytest.mark.asyncio
async def test_select_google_api_key_uses_serial_rotary_order_per_model(monkeypatch):
    await redis_client.flushall()
    monkeypatch.setattr(RedisApiKeyQuota, "_script_initialized", False)
    monkeypatch.setattr(RedisApiKeyQuota, "_script_sha", None)
    monkeypatch.setattr(RedisApiKeyQuota, "_google_selector_script_sha", None)
    monkeypatch.setattr(RedisApiKeyQuota, "_lua_disabled", True)

    first = await RedisApiKeyQuota.select_google_api_key(
        provider_id="provider-rr",
        model_name="gemini-2.5-flash-lite",
        api_keys=["key-1", "key-2"],
        model_limit=1000,
    )
    second = await RedisApiKeyQuota.select_google_api_key(
        provider_id="provider-rr",
        model_name="gemini-2.5-flash-lite",
        api_keys=["key-1", "key-2"],
        model_limit=1000,
    )
    third = await RedisApiKeyQuota.select_google_api_key(
        provider_id="provider-rr",
        model_name="gemini-2.5-flash-lite",
        api_keys=["key-1", "key-2"],
        model_limit=1000,
    )
    other_model = await RedisApiKeyQuota.select_google_api_key(
        provider_id="provider-rr",
        model_name="gemini-2.0-flash",
        api_keys=["key-1", "key-2"],
        model_limit=1000,
    )

    assert first["status"] == "ok"
    assert first["selected_index"] == 0
    assert second["selected_index"] == 1
    assert third["selected_index"] == 0
    assert other_model["selected_index"] == 0


@pytest.mark.asyncio
async def test_select_google_api_key_skips_cooling_and_observed_exhausted_keys(monkeypatch):
    await redis_client.flushall()
    monkeypatch.setattr(RedisApiKeyQuota, "_script_initialized", False)
    monkeypatch.setattr(RedisApiKeyQuota, "_script_sha", None)
    monkeypatch.setattr(RedisApiKeyQuota, "_google_selector_script_sha", None)
    monkeypatch.setattr(RedisApiKeyQuota, "_lua_disabled", True)

    model_name = "gemini-2.5-flash-lite"
    provider_id = "provider-skip"
    await RedisApiKeyQuota.get_or_create_quota("key-1", provider_id, model_name)
    await RedisApiKeyQuota.get_or_create_quota("key-2", provider_id, model_name)

    cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=30)
    await RedisApiKeyQuota.mark_quota_cooldown("key-1", provider_id, model_name, cooldown_until, "per-minute limit")

    exhausted_key_hash = RedisApiKeyQuota.hash_api_key("key-2")
    exhausted_quota_key = f"quota:{provider_id}:{exhausted_key_hash}:{model_name}"
    await redis_client.hset(
        exhausted_quota_key,
        mapping={
            "quota_exhausted_at": datetime.now(timezone.utc).isoformat(),
            "last_reset": redis_backend_module._current_pacific_date(),
            "last_reset_date": redis_backend_module._current_pacific_date(),
        },
    )

    selection = await RedisApiKeyQuota.select_google_api_key(
        provider_id=provider_id,
        model_name=model_name,
        api_keys=["key-1", "key-2", "key-3"],
        model_limit=1000,
    )

    assert selection["status"] == "ok"
    assert selection["selected_index"] == 2


@pytest.mark.asyncio
async def test_select_google_api_key_does_not_preemptively_exclude_by_request_count(monkeypatch):
    await redis_client.flushall()
    monkeypatch.setattr(RedisApiKeyQuota, "_script_initialized", False)
    monkeypatch.setattr(RedisApiKeyQuota, "_script_sha", None)
    monkeypatch.setattr(RedisApiKeyQuota, "_google_selector_script_sha", None)
    monkeypatch.setattr(RedisApiKeyQuota, "_lua_disabled", True)

    provider_id = "provider-count"
    model_name = "gemini-2.5-flash-lite"
    await RedisApiKeyQuota.get_or_create_quota("key-1", provider_id, model_name)

    quota_key = f"quota:{provider_id}:{RedisApiKeyQuota.hash_api_key('key-1')}:{model_name}"
    await redis_client.hset(
        quota_key,
        mapping={
            "requests_today": "1000",
            "last_reset": redis_backend_module._current_pacific_date(),
            "last_reset_date": redis_backend_module._current_pacific_date(),
        },
    )

    selection = await RedisApiKeyQuota.select_google_api_key(
        provider_id=provider_id,
        model_name=model_name,
        api_keys=["key-1"],
        model_limit=1000,
    )

    assert selection["status"] == "ok"
    assert selection["selected_index"] == 0


@pytest.mark.asyncio
async def test_select_google_api_key_returns_retry_when_all_keys_cooling_down(monkeypatch):
    await redis_client.flushall()
    monkeypatch.setattr(RedisApiKeyQuota, "_script_initialized", False)
    monkeypatch.setattr(RedisApiKeyQuota, "_script_sha", None)
    monkeypatch.setattr(RedisApiKeyQuota, "_google_selector_script_sha", None)
    monkeypatch.setattr(RedisApiKeyQuota, "_lua_disabled", True)

    provider_id = "provider-cool"
    model_name = "gemini-3.1-flash-lite"
    await RedisApiKeyQuota.get_or_create_quota("key-1", provider_id, model_name)
    await RedisApiKeyQuota.get_or_create_quota("key-2", provider_id, model_name)

    await RedisApiKeyQuota.mark_quota_cooldown(
        "key-1",
        provider_id,
        model_name,
        datetime.now(timezone.utc) + timedelta(seconds=60),
        "per-minute limit",
    )
    await RedisApiKeyQuota.mark_quota_cooldown(
        "key-2",
        provider_id,
        model_name,
        datetime.now(timezone.utc) + timedelta(seconds=20),
        "per-minute limit",
    )

    selection = await RedisApiKeyQuota.select_google_api_key(
        provider_id=provider_id,
        model_name=model_name,
        api_keys=["key-1", "key-2"],
        model_limit=1000,
    )

    assert selection["status"] == "none_available"
    assert selection["cooling_down_count"] == 2
    assert selection["exhausted_count"] == 0
    assert 0 < selection["retry_after_seconds"] <= 25


@pytest.mark.asyncio
async def test_select_google_api_key_skips_provider_invalid_key_even_without_model_quota(monkeypatch):
    await redis_client.flushall()
    monkeypatch.setattr(RedisApiKeyQuota, "_script_initialized", False)
    monkeypatch.setattr(RedisApiKeyQuota, "_script_sha", None)
    monkeypatch.setattr(RedisApiKeyQuota, "_google_selector_script_sha", None)
    monkeypatch.setattr(RedisApiKeyQuota, "_lua_disabled", True)

    provider_id = "provider-invalid"
    invalid_hash = RedisApiKeyQuota.hash_api_key("key-1")
    await redis_client.sadd(RedisApiKeyQuota.google_invalid_keys_key(provider_id), invalid_hash)

    selection = await RedisApiKeyQuota.select_google_api_key(
        provider_id=provider_id,
        model_name="gemini-2.5-flash-lite",
        api_keys=["key-1", "key-2"],
        model_limit=1000,
    )

    assert selection["status"] == "ok"
    assert selection["selected_index"] == 1
    assert selection["invalid_count"] == 1
