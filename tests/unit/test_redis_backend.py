from datetime import datetime, timedelta, timezone

import pytest

import smolrouter.redis_backend as redis_backend_module
import smolrouter.storage as storage_module
from smolrouter.redis_backend import LogRecord, QuotaRecord, RedisApiKeyQuota, RedisRequestLog
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
    assert len(provider_b_usage) == 1
    assert provider_b_usage[0].invalid_key is False