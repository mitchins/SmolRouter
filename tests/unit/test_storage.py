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


@pytest.mark.asyncio
async def test_janitor_loop_reraises_cancelled_error(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module, "JANITOR_INTERVAL_SEC", 0.01)

    storage = FilesystemBlobStorage(str(tmp_path / "blob_storage"))
    task = asyncio.create_task(storage._janitor_loop())
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
