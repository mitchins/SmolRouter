from datetime import datetime, timezone

import smolrouter.storage as storage_module
from smolrouter.redis_backend import LogRecord, QuotaRecord


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


def test_quota_record_failure_tracks_last_error_and_quota_exhaustion():
    quota = QuotaRecord({"requests_today": "1", "tokens_today": "10", "error_count": "0"})

    quota.mark_request_failure(error="quota hit", quota_exhausted=True)

    assert quota.error_count == 1
    assert quota.last_error == "quota hit"
    assert isinstance(quota.quota_exhausted_at, datetime)
    assert quota.quota_exhausted_at.tzinfo is not None