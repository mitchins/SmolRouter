import asyncio
import errno

import pytest

import smolrouter.storage as storage_module
from smolrouter.storage import FilesystemBlobStorage


@pytest.mark.asyncio
async def test_filesystem_janitor_prunes_when_over_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "MAX_TOTAL_STORAGE_SIZE", 100)
    monkeypatch.setattr(storage_module, "WATERMARK_FRACTION", 0.5)
    monkeypatch.setattr(storage_module, "KEEP_RECENT_HOURS", 0)

    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    first = storage.base_path / "2026" / "07" / "04" / "01" / "a.blob"
    second = storage.base_path / "2026" / "07" / "04" / "02" / "b.blob"
    first.parent.mkdir(parents=True, exist_ok=True)
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"a" * 60)
    second.write_bytes(b"b" * 60)
    storage._write_usage_bytes_locked(120)

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


def test_filesystem_store_uses_incremental_usage_counter_without_scanning_under_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "MAX_TOTAL_STORAGE_SIZE", 100)

    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    scan_calls = []

    real_dir_size_bytes = storage._dir_size_bytes

    def tracked_dir_size(path):
        scan_calls.append(path)
        return real_dir_size_bytes(path)

    monkeypatch.setattr(storage, "_dir_size_bytes", tracked_dir_size)
    storage.store(b"12345")

    assert storage._total_size_bytes() == 5
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


def test_cleanup_old_tolerates_disappearing_blob(tmp_path, monkeypatch):
    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))

    class _MissingBlob:
        def stat(self):
            raise FileNotFoundError

        def unlink(self):
            raise AssertionError("unlink should not be reached")

    class _FakeBasePath:
        def rglob(self, _pattern):
            return [_MissingBlob()]

        def iterdir(self):
            return []

    monkeypatch.setattr(storage, "base_path", _FakeBasePath())

    assert storage.cleanup_old(1) == 0


def test_start_janitor_without_running_loop_is_harmless(tmp_path, monkeypatch):
    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    monkeypatch.setattr(storage_module.asyncio, "get_running_loop", lambda: (_ for _ in ()).throw(RuntimeError()))

    storage.start_janitor()

    assert storage._janitor_task is None


def test_adjust_usage_underflow_resets_counter_to_zero(tmp_path):
    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    storage._write_usage_bytes_locked(3)

    assert storage._adjust_usage_bytes(-10) == 0
    assert storage._total_size_bytes() == 0


def test_cleanup_for_space_requests_janitor(tmp_path, monkeypatch):
    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    janitor_calls = []
    monkeypatch.setattr(storage, "_trigger_janitor", lambda: janitor_calls.append(True))

    storage._cleanup_for_space(5)

    assert janitor_calls == [True]


def test_trigger_janitor_wakes_registered_event_loop(tmp_path):
    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    called = []

    class _Loop:
        def call_soon_threadsafe(self, callback):
            called.append("wake")
            callback()

    event = asyncio.Event()
    storage._janitor_loop_ref = _Loop()
    storage._janitor_wakeup = event

    storage._trigger_janitor()

    assert called == ["wake"]
    assert event.is_set() is True


def test_stop_janitor_cancels_active_task(tmp_path):
    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))

    class _Task:
        def __init__(self):
            self.cancelled = False

        def done(self):
            return False

        def cancel(self):
            self.cancelled = True

    task = _Task()
    storage._janitor_task = task

    storage.stop_janitor()

    assert task.cancelled is True
    assert storage._janitor_task is None


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


@pytest.mark.asyncio
async def test_janitor_reconciles_stale_usage_counter_without_deleting_live_blobs(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "MAX_TOTAL_STORAGE_SIZE", 100)
    monkeypatch.setattr(storage_module, "WATERMARK_FRACTION", 0.5)

    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    key = storage.store(b"x" * 10)
    storage._write_usage_bytes_locked(5000)

    await storage._run_janitor_once()

    assert storage._total_size_bytes() == 10
    assert storage._get_blob_path(key).exists() is True


def test_store_prunes_older_blobs_before_writing_new_blob(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "MAX_TOTAL_STORAGE_SIZE", 10)

    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    old_path = storage.base_path / "2026" / "07" / "04" / "23" / "old.blob"
    old_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.write_bytes(b"abcdefgh")
    storage._write_usage_bytes_locked(8)
    monkeypatch.setattr(storage, "_generate_key", lambda: "1783236000000-newblob")

    key = storage.store(b"xyz")
    new_path = storage._get_blob_path(key)

    assert old_path.exists() is False
    assert new_path.exists() is True
    assert storage.retrieve(key) == b"xyz"
    assert storage._total_size_bytes() == 3


def test_store_raises_enospc_when_blob_cannot_fit_even_after_eviction(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "MAX_TOTAL_STORAGE_SIZE", 5)

    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))

    with pytest.raises(OSError) as exc_info:
        storage.store(b"123456")

    assert exc_info.value.errno == errno.ENOSPC
