import os
import hashlib
import logging
import shutil
import asyncio
import time
import secrets
from pathlib import Path
from typing import Optional, Dict, List
from abc import ABC, abstractmethod

from .config_paths import resolve_blob_storage_path

logger = logging.getLogger("model-rerouter")

# Configuration constants
JSON_CONTENT_TYPE = "application/json"
BLOB_FILE_GLOB = "*.blob"
MAX_BLOB_SIZE = int(os.getenv("MAX_BLOB_SIZE", "10485760"))  # 10MB default
MAX_TOTAL_STORAGE_SIZE = int(os.getenv("MAX_TOTAL_STORAGE_SIZE", "1073741824"))  # 1GB default
JANITOR_INTERVAL_SEC = int(os.getenv("BLOB_JANITOR_INTERVAL_SEC", "300"))  # 5 minutes
# Target to reduce to when pruning (e.g., 0.8 -> prune to 80% of cap)
WATERMARK_FRACTION = float(os.getenv("BLOB_WATERMARK_FRACTION", "0.8"))
# Keep at least this many recent hourly buckets untouched when pruning
KEEP_RECENT_HOURS = int(os.getenv("BLOB_KEEP_RECENT_HOURS", "1"))


class BlobStorage(ABC):
    """Abstract base class for blob storage backends"""

    @abstractmethod
    def store(self, data: bytes, content_type: str = JSON_CONTENT_TYPE, record_id: Optional[int] = None) -> str:
        """Store data and return a reference key"""
        raise NotImplementedError()

    @abstractmethod
    def retrieve(self, key: str, record_id: Optional[int] = None) -> Optional[bytes]:
        """Retrieve data by key, return None if not found"""
        raise NotImplementedError()

    @abstractmethod
    def delete(self, key: str, record_id: Optional[int] = None) -> bool:
        """Delete data by key, return True if successful"""
        raise NotImplementedError()

    @abstractmethod
    def exists(self, key: str, record_id: Optional[int] = None) -> bool:
        """Check if key exists"""
        raise NotImplementedError()

    @abstractmethod
    def cleanup_old(self, max_age_days: int) -> int:
        """Remove old blobs, return count of deleted items"""
        raise NotImplementedError()


class FilesystemBlobStorage(BlobStorage):
    """Filesystem-based blob storage implementation"""

    def __init__(self, base_path: str = "blob_storage"):
        self.base_path = resolve_blob_storage_path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized filesystem blob storage at {self.base_path}")
        # Background janitor task handle (set by init/start functions)
        self._janitor_task: Optional[asyncio.Task] = None

    def _get_blob_path(self, key: str, record_id: Optional[int] = None) -> Path:
        """Get the filesystem path for a blob key using timestamp-bucket sharding.

        Key format: "<epoch_ms>-<rand_hex8>[-<suffix>]"
        Directory layout: base/YYYY/MM/DD/HH/<key>.blob (UTC time)
        """
        # record_id no longer affects sharding; kept for API compatibility
        try:
            epoch_ms_str = key.split("-", 1)[0]
            epoch_ms = int(epoch_ms_str)
        except Exception:
            # Fallback: put into unknown dir to avoid crashes
            return self.base_path / "unknown" / f"{key}.blob"

        t = time.gmtime(epoch_ms / 1000.0)
        year = f"{t.tm_year:04d}"
        month = f"{t.tm_mon:02d}"
        day = f"{t.tm_mday:02d}"
        hour = f"{t.tm_hour:02d}"
        return self.base_path / year / month / day / hour / f"{key}.blob"

    def _generate_key(self) -> str:
        """Generate a timestamp-aligned unique key.

        Format: epoch_ms-rand8hex. Content is not used to dedupe.
        """
        epoch_ms = int(time.time() * 1000)
        rand8 = secrets.token_hex(4)
        return f"{epoch_ms}-{rand8}"

    def store(self, data: bytes, content_type: str = JSON_CONTENT_TYPE, record_id: Optional[int] = None) -> str:
        """Store data and return SHA256 hash as key"""
        # content_type is accepted for API compatibility but not used here
        _ = content_type
        if not data:
            return ""

        # Check individual blob size limit
        if len(data) > MAX_BLOB_SIZE:
            logger.warning(f"Blob size {len(data)} bytes exceeds limit {MAX_BLOB_SIZE} bytes, truncating")
            data = data[:MAX_BLOB_SIZE]

        # Generate a unique key and path
        attempts = 0
        while True:
            key = self._generate_key()
            blob_path = self._get_blob_path(key, record_id)
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            if not blob_path.exists():
                break
            attempts += 1
            if attempts > 3:
                # Extremely unlikely collision; add hash suffix to guarantee uniqueness
                suffix = hashlib.sha256(data).hexdigest()[:8]
                key = f"{key}-{suffix}"
                blob_path = self._get_blob_path(key, record_id)
                blob_path.parent.mkdir(parents=True, exist_ok=True)
                break

        with open(blob_path, "wb") as f:
            f.write(data)
        logger.debug(f"Stored blob {key} ({len(data)} bytes) at {blob_path}")
        return key

    def retrieve(self, key: str, record_id: Optional[int] = None) -> Optional[bytes]:
        """Retrieve data by key"""
        if not key:
            return None

        blob_path = self._get_blob_path(key, record_id)

        try:
            if blob_path.exists():
                with open(blob_path, "rb") as f:
                    data = f.read()
                logger.debug(f"Retrieved blob {key} ({len(data)} bytes)")
                return data
        except Exception as e:
            logger.error(f"Failed to retrieve blob {key}: {e}")

        return None

    def delete(self, key: str, record_id: Optional[int] = None) -> bool:
        """Delete data by key"""
        if not key:
            return False

        blob_path = self._get_blob_path(key, record_id)

        try:
            if blob_path.exists():
                blob_path.unlink()
                logger.debug(f"Deleted blob {key}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete blob {key}: {e}")

        return False

    def exists(self, key: str, record_id: Optional[int] = None) -> bool:
        """Check if key exists"""
        if not key:
            return False
        return self._get_blob_path(key, record_id).exists()

    def cleanup_old(self, max_age_days: int) -> int:
        """Remove blobs older than max_age_days"""
        import time

        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        deleted_count = 0

        try:
            for blob_file in self.base_path.rglob(BLOB_FILE_GLOB):
                if blob_file.stat().st_mtime < cutoff_time:
                    try:
                        blob_file.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete old blob {blob_file}: {e}")

            # Clean up empty subdirectories
            for subdir in self.base_path.iterdir():
                if subdir.is_dir() and not any(subdir.iterdir()):
                    try:
                        subdir.rmdir()
                    except Exception:
                        pass  # Ignore errors removing empty dirs

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old blob files")

        except Exception as e:
            logger.error(f"Failed to cleanup old blobs: {e}")

        return deleted_count

    def _check_storage_limit(self, additional_bytes: int) -> bool:
        """Check if adding additional_bytes would exceed storage limit"""
        # No longer used in hot path; retained for potential external callers
        try:
            current_size = sum(f.stat().st_size for f in self.base_path.rglob(BLOB_FILE_GLOB))
            return (current_size + additional_bytes) <= MAX_TOTAL_STORAGE_SIZE
        except Exception as e:
            logger.error(f"Failed to check storage limit: {e}")
            return True  # Allow storage if we can't check

    def _cleanup_for_space(self):
        """Deprecated: cleanup now handled by background janitor."""
        logger.debug("_cleanup_for_space called but janitor is responsible for pruning; ignoring.")

    # ---------- Background Janitor (size-based pruning) ----------
    def _list_hour_buckets(self) -> List[Path]:
        """Return sorted list of hour-level bucket directories (oldest -> newest)."""
        buckets: List[Path] = []
        if not self.base_path.exists():
            return buckets
        try:
            for year_dir in sorted(p for p in self.base_path.iterdir() if p.is_dir() and p.name.isdigit()):
                for month_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
                    for day_dir in sorted(p for p in month_dir.iterdir() if p.is_dir()):
                        for hour_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
                            buckets.append(hour_dir)
        except Exception as e:
            logger.error(f"Failed to list hour buckets: {e}")
        return buckets

    def _bucket_epoch(self, bucket: Path) -> int:
        """Parse bucket path YYYY/MM/DD/HH to epoch seconds (UTC)."""
        try:
            year = int(bucket.parents[2].name)
            month = int(bucket.parents[1].name)
            day = int(bucket.parents[0].name)
            hour = int(bucket.name)
            return int(
                time.mktime(time.strptime(f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:00:00", "%Y-%m-%d %H:%M:%S"))
            )
        except Exception:
            return 0

    def _dir_size_bytes(self, path: Path) -> int:
        total = 0
        try:
            for f in path.rglob(BLOB_FILE_GLOB):
                try:
                    total += f.stat().st_size
                except Exception:
                    pass
        except Exception:
            pass
        return total

    def _total_size_bytes(self) -> int:
        return self._dir_size_bytes(self.base_path)

    def _get_prune_candidates(self, buckets: List[Path]) -> List[Path]:
        if KEEP_RECENT_HOURS > 0 and len(buckets) > KEEP_RECENT_HOURS:
            return buckets[:-KEEP_RECENT_HOURS]
        return buckets

    def _select_oldest_bucket(self, prune_candidates: List[Path], buckets: List[Path]) -> Optional[Path]:
        if prune_candidates:
            return prune_candidates[0]
        if buckets:
            return buckets[0]
        return None

    async def _prune_full_buckets(self, prune_candidates: List[Path], current_size: int, target: int) -> int:
        for bucket in prune_candidates:
            if current_size <= target:
                break
            try:
                bucket_size = await asyncio.to_thread(self._dir_size_bytes, bucket)
                await asyncio.to_thread(shutil.rmtree, bucket, True)
                current_size -= bucket_size
                logger.info(f"Deleted bucket {bucket} (freed {bucket_size} bytes)")
            except Exception as e:
                logger.error(f"Failed to delete bucket {bucket}: {e}")
        return current_size

    def _prune_bucket_files(self, bucket: Path, current_size: int, target: int) -> int:
        try:
            files = []
            for blob_file in bucket.rglob(BLOB_FILE_GLOB):
                try:
                    stats = blob_file.stat()
                    files.append((blob_file, stats.st_mtime, stats.st_size))
                except Exception:
                    continue

            files.sort(key=lambda entry: entry[1])
            for blob_file, _, file_size in files:
                if current_size <= target:
                    break
                try:
                    blob_file.unlink()
                    current_size -= file_size
                except Exception:
                    continue

            logger.info(f"Pruned files in {bucket} to reach target; now {current_size} bytes")
        except Exception as e:
            logger.error(f"Failed pruning files in {bucket}: {e}")
        return current_size

    async def _run_janitor_once(self):
        """One pruning cycle: if over cap, delete oldest buckets/files to watermark."""
        try:
            current_size = await asyncio.to_thread(self._total_size_bytes)
            cap = MAX_TOTAL_STORAGE_SIZE
            if current_size <= cap:
                return

            target = int(cap * WATERMARK_FRACTION)
            logger.warning(f"Blob storage over cap: current={current_size} > cap={cap}. Pruning to ≈{target} bytes.")

            buckets = await asyncio.to_thread(self._list_hour_buckets)
            prune_candidates = self._get_prune_candidates(buckets)
            current_size = await self._prune_full_buckets(prune_candidates, current_size, target)

            # If still over target, prune files in the oldest remaining bucket
            if current_size > target:
                oldest = self._select_oldest_bucket(prune_candidates, buckets)
                if oldest and oldest.exists():
                    current_size = self._prune_bucket_files(oldest, current_size, target)
        except Exception as e:
            logger.error(f"Janitor cycle failed: {e}")

    async def _janitor_loop(self):
        logger.info(
            f"Starting blob janitor: interval={JANITOR_INTERVAL_SEC}s, watermark={WATERMARK_FRACTION}, keep_recent_hours={KEEP_RECENT_HOURS}"
        )
        try:
            while True:
                await asyncio.sleep(JANITOR_INTERVAL_SEC)
                await self._run_janitor_once()
        except asyncio.CancelledError:
            logger.info("Blob janitor cancelled")
            raise
        except Exception as e:
            logger.error(f"Blob janitor error: {e}")

    def start_janitor(self):
        """Start background janitor if not already running."""
        try:
            loop = asyncio.get_running_loop()
            if self._janitor_task is None or self._janitor_task.done():
                self._janitor_task = loop.create_task(self._janitor_loop())
        except RuntimeError:
            # No loop yet; app startup will call again when loop exists
            logger.info("Janitor will start when event loop is available")

    def stop_janitor(self):
        if self._janitor_task and not self._janitor_task.done():
            self._janitor_task.cancel()
            self._janitor_task = None


class InMemoryBlobStorage(BlobStorage):
    """In-memory blob storage for testing/development"""

    def __init__(self):
        self._storage: Dict[str, bytes] = {}
        self._timestamps: Dict[str, float] = {}
        logger.info("Initialized in-memory blob storage")

    def _generate_key(self, data: bytes) -> str:
        """Generate a SHA256 hash key for the data"""
        return hashlib.sha256(data).hexdigest()

    def store(self, data: bytes, content_type: str = JSON_CONTENT_TYPE, record_id: Optional[int] = None) -> str:
        """Store data and return SHA256 hash as key"""
        # content_type is accepted for API compatibility but not used here
        _ = content_type
        if not data:
            return ""

        # Check individual blob size limit
        if len(data) > MAX_BLOB_SIZE:
            logger.warning(f"Blob size {len(data)} bytes exceeds limit {MAX_BLOB_SIZE} bytes, truncating")
            data = data[:MAX_BLOB_SIZE]

        key = self._generate_key(data)

        # Check total storage size limit
        current_size = sum(len(blob) for blob in self._storage.values())
        if current_size + len(data) > MAX_TOTAL_STORAGE_SIZE:
            logger.warning("Memory storage limit exceeded, cleaning up old blobs")
            self._cleanup_for_space()

        self._storage[key] = data
        import time

        self._timestamps[key] = time.time()

        logger.debug(f"Stored blob {key} in memory ({len(data)} bytes)")
        return key

    def retrieve(self, key: str, record_id: Optional[int] = None) -> Optional[bytes]:
        """Retrieve data by key"""
        if not key:
            return None

        data = self._storage.get(key)
        if data:
            logger.debug(f"Retrieved blob {key} from memory ({len(data)} bytes)")
        return data

    def delete(self, key: str, record_id: Optional[int] = None) -> bool:
        """Delete data by key"""
        if not key:
            return False

        if key in self._storage:
            del self._storage[key]
            self._timestamps.pop(key, None)
            logger.debug(f"Deleted blob {key} from memory")
            return True
        return False

    def exists(self, key: str, record_id: Optional[int] = None) -> bool:
        """Check if key exists"""
        return key in self._storage

    def cleanup_old(self, max_age_days: int) -> int:
        """Remove blobs older than max_age_days"""
        import time

        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        old_keys = [key for key, timestamp in self._timestamps.items() if timestamp < cutoff_time]

        for key in old_keys:
            self.delete(key)

        if old_keys:
            logger.info(f"Cleaned up {len(old_keys)} old blob entries from memory")

        return len(old_keys)

    def _cleanup_for_space(self, needed_bytes: int = 0):
        """Remove oldest blobs to make space for needed_bytes"""

        # Sort by timestamp (oldest first)
        items_by_age = sorted(self._timestamps.items(), key=lambda x: x[1])

        freed_space = 0
        deleted_count = 0

        for key, timestamp in items_by_age:
            if key in self._storage:
                blob_size = len(self._storage[key])
                self.delete(key)
                freed_space += blob_size
                deleted_count += 1

                # Check if we've freed enough space
                if freed_space >= needed_bytes:
                    break

        logger.info(f"Cleaned up {deleted_count} old memory blobs, freed {freed_space} bytes")


def create_blob_storage(storage_type: str = "filesystem", **kwargs) -> BlobStorage:
    """Factory function to create blob storage instances"""
    if storage_type == "filesystem":
        return FilesystemBlobStorage(**kwargs)
    elif storage_type == "memory":
        return InMemoryBlobStorage(**kwargs)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


# Global blob storage instance
_blob_storage: Optional[BlobStorage] = None


def get_blob_storage() -> BlobStorage:
    """Get the global blob storage instance"""
    global _blob_storage
    if _blob_storage is None:
        storage_type = os.getenv("BLOB_STORAGE_TYPE", "filesystem")
        storage_path = os.getenv("BLOB_STORAGE_PATH")
        _blob_storage = create_blob_storage(storage_type, base_path=storage_path)
    return _blob_storage


def init_blob_storage():
    """Initialize blob storage (called at startup)"""
    storage = get_blob_storage()
    logger.info(f"Blob storage initialized: {type(storage).__name__}")
    # Start janitor for filesystem storage
    if isinstance(storage, FilesystemBlobStorage):
        try:
            storage.start_janitor()
        except Exception as e:
            logger.error(f"Failed to start blob janitor: {e}")
    return storage
