import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict
from abc import ABC, abstractmethod

logger = logging.getLogger("model-rerouter")

# Configuration constants
MAX_BLOB_SIZE = int(os.getenv("MAX_BLOB_SIZE", "10485760"))  # 10MB default
MAX_TOTAL_STORAGE_SIZE = int(os.getenv("MAX_TOTAL_STORAGE_SIZE", "1073741824"))  # 1GB default


class BlobStorage(ABC):
    """Abstract base class for blob storage backends"""

    @abstractmethod
    def store(self, data: bytes, content_type: str = "application/json", record_id: Optional[int] = None) -> str:
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
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized filesystem blob storage at {self.base_path}")

    def _get_blob_path(self, key: str, record_id: Optional[int] = None) -> Path:
        """Get the filesystem path for a blob key with enhanced sharding"""
        # Enhanced sharding strategy for 1M+ records
        if record_id is not None:
            # Shard by record_id/10000 for predictable distribution
            shard = record_id // 10000
            subdir = f"shard_{shard:04d}"
        else:
            # Fallback to hash-based sharding (for manual/legacy keys)
            subdir = key[:2] if len(key) >= 2 else "00"

        return self.base_path / subdir / f"{key}.blob"

    def _generate_key(self, data: bytes) -> str:
        """Generate a SHA256 hash key for the data"""
        return hashlib.sha256(data).hexdigest()

    def store(self, data: bytes, content_type: str = "application/json", record_id: Optional[int] = None) -> str:
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
        blob_path = self._get_blob_path(key, record_id)

        # Create subdirectory if needed
        blob_path.parent.mkdir(parents=True, exist_ok=True)

        # Only write if file doesn't exist (deduplication)
        if not blob_path.exists():
            # Check total storage size limit before writing
            if not self._check_storage_limit(len(data)):
                logger.warning(f"Total storage limit exceeded, running cleanup before storing blob {key}")
                self._cleanup_for_space(len(data))

            with open(blob_path, 'wb') as f:
                f.write(data)
            logger.debug(f"Stored blob {key} ({len(data)} bytes)")
        else:
            logger.debug(f"Blob {key} already exists, skipping write")

        return key

    def retrieve(self, key: str, record_id: Optional[int] = None) -> Optional[bytes]:
        """Retrieve data by key"""
        if not key:
            return None

        blob_path = self._get_blob_path(key, record_id)

        try:
            if blob_path.exists():
                with open(blob_path, 'rb') as f:
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
            for blob_file in self.base_path.rglob("*.blob"):
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
        try:
            current_size = sum(f.stat().st_size for f in self.base_path.rglob("*.blob"))
            return (current_size + additional_bytes) <= MAX_TOTAL_STORAGE_SIZE
        except Exception as e:
            logger.error(f"Failed to check storage limit: {e}")
            return True  # Allow storage if we can't check
    
    def _cleanup_for_space(self, needed_bytes: int):
        """Remove oldest blobs to make space for needed_bytes"""
        try:
            # Get all blob files with their modification times
            blob_files = []
            for blob_file in self.base_path.rglob("*.blob"):
                try:
                    stat = blob_file.stat()
                    blob_files.append((blob_file, stat.st_mtime, stat.st_size))
                except Exception:
                    continue
            
            # Sort by modification time (oldest first)
            blob_files.sort(key=lambda x: x[1])
            
            freed_space = 0
            deleted_count = 0
            
            for blob_file, mtime, size in blob_files:
                try:
                    blob_file.unlink()
                    freed_space += size
                    deleted_count += 1
                    
                    # Check if we've freed enough space
                    if freed_space >= needed_bytes:
                        break
                        
                except Exception as e:
                    logger.error(f"Failed to delete blob {blob_file}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old blobs, freed {freed_space} bytes")
            
        except Exception as e:
            logger.error(f"Failed to cleanup for space: {e}")

class InMemoryBlobStorage(BlobStorage):
    """In-memory blob storage for testing/development"""
    
    def __init__(self):
        self._storage: Dict[str, bytes] = {}
        self._timestamps: Dict[str, float] = {}
        logger.info("Initialized in-memory blob storage")
    
    def _generate_key(self, data: bytes) -> str:
        """Generate a SHA256 hash key for the data"""
        return hashlib.sha256(data).hexdigest()
    
    def store(self, data: bytes, content_type: str = "application/json", record_id: Optional[int] = None) -> str:
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
            self._cleanup_for_space(len(data))

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
        
        old_keys = [
            key for key, timestamp in self._timestamps.items()
            if timestamp < cutoff_time
        ]
        
        for key in old_keys:
            self.delete(key)
        
        if old_keys:
            logger.info(f"Cleaned up {len(old_keys)} old blob entries from memory")
        
        return len(old_keys)
    
    def _cleanup_for_space(self, needed_bytes: int):
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
        storage_path = os.getenv("BLOB_STORAGE_PATH", "blob_storage")
        _blob_storage = create_blob_storage(storage_type, base_path=storage_path)
    return _blob_storage

def init_blob_storage():
    """Initialize blob storage (called at startup)"""
    storage = get_blob_storage()
    logger.info(f"Blob storage initialized: {type(storage).__name__}")
    return storage