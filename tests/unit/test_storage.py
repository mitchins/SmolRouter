import asyncio

import pytest

import smolrouter.storage as storage_module
from smolrouter.storage import FilesystemBlobStorage


@pytest.mark.asyncio
async def test_filesystem_janitor_prunes_when_over_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "MAX_TOTAL_STORAGE_SIZE", 100)
    monkeypatch.setattr(storage_module, "WATERMARK_FRACTION", 0.5)
    monkeypatch.setattr(storage_module, "KEEP_RECENT_HOURS", 0)

    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    storage.store(b"a" * 60)
    storage.store(b"b" * 60)

    assert storage._total_size_bytes() > storage_module.MAX_TOTAL_STORAGE_SIZE

    await storage._run_janitor_once()

    assert storage._total_size_bytes() <= int(storage_module.MAX_TOTAL_STORAGE_SIZE * storage_module.WATERMARK_FRACTION)
    actual_size = sum(blob_file.stat().st_size for blob_file in storage.base_path.rglob(storage_module.BLOB_FILE_GLOB))
    assert storage._total_size_bytes() == actual_size


@pytest.mark.asyncio
async def test_janitor_loop_reraises_cancelled_error(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "JANITOR_INTERVAL_SEC", 0.01)

    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    task = asyncio.create_task(storage._janitor_loop())
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


def test_filesystem_store_uses_incremental_usage_counter_and_triggers_janitor(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "MAX_TOTAL_STORAGE_SIZE", 10)

    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    janitor_calls = []
    scan_calls = []

    real_dir_size_bytes = storage._dir_size_bytes

    def tracked_dir_size(path):
        scan_calls.append(path)
        return real_dir_size_bytes(path)

    monkeypatch.setattr(storage, "_dir_size_bytes", tracked_dir_size)
    monkeypatch.setattr(storage, "_trigger_janitor", lambda: janitor_calls.append(True))
    storage._adjust_usage_bytes(9)

    storage.store(b"12345")

    assert janitor_calls == [True]
    assert storage._total_size_bytes() == 14
    assert scan_calls == []


def test_filesystem_delete_decrements_usage_counter(tmp_path):
    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))

    key = storage.store(b"abc")

    assert storage._total_size_bytes() == 3
    assert storage.delete(key) is True
    assert storage._total_size_bytes() == 0
    assert storage.delete(key) is False
    assert storage._total_size_bytes() == 0


def test_filesystem_delete_returns_false_when_blob_was_removed_elsewhere(tmp_path):
    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    key = storage.store(b"abc")

    storage._get_blob_path(key).unlink()

    assert storage.delete(key) is False


def test_usage_lock_gracefully_skips_when_fcntl_is_unavailable(tmp_path, monkeypatch):
    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    monkeypatch.setattr(storage_module, "fcntl", None)

    with storage._usage_lock():
        storage._write_usage_bytes_locked(7)

    assert storage._read_usage_bytes() == 7


def test_usage_counter_rebuilds_once_then_store_stays_incremental(tmp_path, monkeypatch):
    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    storage.store(b"aa")
    storage._usage_file.write_text("not-a-number\n", encoding="utf-8")

    scan_calls = []
    real_dir_size_bytes = storage._dir_size_bytes

    def tracked_dir_size(path):
        scan_calls.append(path)
        return real_dir_size_bytes(path)

    monkeypatch.setattr(storage, "_dir_size_bytes", tracked_dir_size)

    assert storage._total_size_bytes() == 2
    assert len(scan_calls) == 1

    storage.store(b"bbb")

    assert len(scan_calls) == 1
    assert storage._total_size_bytes() == 5
