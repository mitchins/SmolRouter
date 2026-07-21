"""Thread- and process-safe file-backed project and facade-key management."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml
from platformdirs import user_config_dir

from .config_paths import resolve_routes_config_path
from .facade_key_store import (
    generate_facade_key_secret,
    load_facade_key_secrets_from_path,
    reload_facade_key_secrets,
    resolve_facade_key_config_path,
)
from .facade_keys import FacadeKeyRegistry

if os.name == "nt":
    import msvcrt
else:
    import fcntl


PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
PROJECT_NOT_FOUND_MESSAGE = "Project not found"
_DIRECTORY_FSYNC_UNSUPPORTED_ERRNOS = {
    errno.EINVAL,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}
_PROCESS_LOCK = threading.RLock()


class ProjectKeyManagementError(RuntimeError):
    status_code = 500


class ProjectKeyValidationError(ProjectKeyManagementError):
    status_code = 422


class ProjectKeyNotFoundError(ProjectKeyManagementError):
    status_code = 404


class ProjectKeyConflictError(ProjectKeyManagementError):
    status_code = 409


def facade_key_id(secret: str) -> str:
    return f"sha256:{hashlib.sha256(secret.encode('utf-8')).hexdigest()}"


def _revision(path: Path) -> tuple[Any, ...]:
    try:
        stat = path.stat()
        return (True, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    except FileNotFoundError:
        return (False, None, None, None, None)


def _content_hash(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ProjectKeyManagementError(f"Managed file may not be a symlink: {path}")


def _read_mapping(path: Path) -> dict[str, Any]:
    _reject_symlink(path)
    if not path.is_file():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ProjectKeyManagementError(f"Unable to read managed configuration {path}: {exc}") from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ProjectKeyManagementError(f"Managed configuration must be a mapping: {path}")
    return {str(key): value for key, value in parsed.items()}


def _serialize(path: Path, payload: Mapping[str, Any]) -> bytes:
    if path.suffix.lower() == ".json":
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return yaml.safe_dump(dict(payload), sort_keys=True).encode()


def _atomic_replace(
    path: Path,
    payload: Mapping[str, Any],
    *,
    mode: int,
    expected_hash: str | None,
    preserve_mode: bool = False,
) -> None:
    _reject_symlink(path)
    if _content_hash(path) != expected_hash:
        raise ProjectKeyConflictError(f"Managed configuration changed while it was being updated: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = (path.stat().st_mode & 0o777) if preserve_mode and path.exists() else mode
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, existing_mode)
        else:
            os.chmod(tmp_name, existing_mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(_serialize(path, payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        os.chmod(path, existing_mode)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry where the platform supports directory fsync."""
    if os.name == "nt":
        return
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        if exc.errno in _DIRECTORY_FSYNC_UNSUPPORTED_ERRNOS:
            return
        raise
    try:
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            if exc.errno not in _DIRECTORY_FSYNC_UNSUPPORTED_ERRNOS:
                raise
    finally:
        os.close(directory_fd)


def _lock_file(lock_fd: int) -> None:
    if os.name != "nt":
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return
    if os.fstat(lock_fd).st_size == 0:
        os.write(lock_fd, b"\0")
        os.fsync(lock_fd)
    os.lseek(lock_fd, 0, os.SEEK_SET)
    msvcrt.locking(lock_fd, msvcrt.LK_LOCK, 1)


def _unlock_file(lock_fd: int) -> None:
    if os.name != "nt":
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        return
    os.lseek(lock_fd, 0, os.SEEK_SET)
    msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)


def _has_invalid_project_path(value: str) -> bool:
    segments = value.split("/")
    return (
        value.startswith("/")
        or value.endswith("/")
        or "//" in value
        or any(segment in {".", ".."} for segment in segments)
    )


def _is_valid_new_project_id(raw_value: str, value: str) -> bool:
    return raw_value == value and PROJECT_ID_RE.fullmatch(value) is not None and not _has_invalid_project_path(value)


def _validate_project_id(project_id: Any) -> str:
    raw_value = str(project_id or "")
    value = raw_value.strip()
    if not _is_valid_new_project_id(raw_value, value):
        raise ProjectKeyValidationError(
            "project_id must be 1-128 characters, start with an alphanumeric character, "
            "and contain only letters, numbers, '.', '_', '-', and well-formed '/' segments"
        )
    return value


def _normalize_display_name(display_name: Any) -> str:
    if not isinstance(display_name, str):
        raise ProjectKeyValidationError("display_name must be a non-empty string of at most 100 characters")
    normalized = display_name.strip()
    if not normalized or len(normalized) > 100:
        raise ProjectKeyValidationError("display_name must be a non-empty string of at most 100 characters")
    return normalized


def _normalize_tag(tag: Any) -> str:
    if not isinstance(tag, str):
        raise ProjectKeyValidationError("each tag must be a non-empty string of at most 40 characters")
    normalized = tag.strip()
    if not normalized or len(normalized) > 40:
        raise ProjectKeyValidationError("each tag must be a non-empty string of at most 40 characters")
    return normalized


def _normalize_tags(tags: Any) -> list[str]:
    if not isinstance(tags, list) or len(tags) > 20:
        raise ProjectKeyValidationError("tags must be a list containing at most 20 values")
    normalized = [_normalize_tag(tag) for tag in tags]
    if len(normalized) != len(set(normalized)):
        raise ProjectKeyValidationError("tags must be unique")
    return normalized


def _project_config(display_name: Any = None, tags: Any = None) -> dict[str, Any]:
    result: dict[str, Any] = {"enabled": True}
    if display_name is not None:
        result["display_name"] = _normalize_display_name(display_name)
    if tags is not None:
        result["tags"] = _normalize_tags(tags)
    return result


class ProjectKeyManager:
    """Own the authoritative route metadata, key file, locks, and registry snapshot."""

    def __init__(self, routes_path: Path | str | None = None, secrets_path: Path | str | None = None):
        self.routes_path = (
            Path(routes_path).expanduser().absolute()
            if routes_path is not None
            else resolve_routes_config_path(os.getenv("ROUTES_CONFIG"))
        )
        resolved_secrets = Path(secrets_path).expanduser().absolute() if secrets_path else resolve_facade_key_config_path()
        self.secrets_path = resolved_secrets or (Path(user_config_dir("smolrouter")) / "facade_keys.yaml").resolve()
        lock_seed = hashlib.sha256(f"{self.routes_path}\0{self.secrets_path}".encode()).hexdigest()
        self.lock_path = self.secrets_path.parent / f".project-keys-{lock_seed}.lock"
        self._snapshot: FacadeKeyRegistry | None = None
        self._snapshot_revision: tuple[Any, ...] | None = None
        self._snapshot_error: Exception | None = None

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with _PROCESS_LOCK:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            acquired = False
            try:
                _lock_file(lock_fd)
                acquired = True
                yield
            finally:
                if acquired:
                    _unlock_file(lock_fd)
                os.close(lock_fd)

    def revision(self) -> tuple[Any, ...]:
        return (_revision(self.routes_path), _revision(self.secrets_path))

    def _read_sources(self) -> tuple[dict[str, Any], dict[str, list[str]]]:
        routes = _read_mapping(self.routes_path)
        raw_projects = routes.get("facade_keys", {})
        if raw_projects is None:
            raw_projects = {}
            routes["facade_keys"] = raw_projects
        if not isinstance(raw_projects, dict):
            raise ProjectKeyManagementError("routes facade_keys must be a mapping")
        _reject_symlink(self.secrets_path)
        try:
            key_map = load_facade_key_secrets_from_path(self.secrets_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise ProjectKeyManagementError(f"Unable to read managed facade keys: {exc}") from exc
        for project_id, values in key_map.items():
            if len(values) != len(set(values)):
                raise ProjectKeyManagementError(
                    f"Facade key secrets contain duplicate values for project {project_id!r}"
                )
        return routes, key_map

    def _recover_after_delete_failure(
        self,
        project_id: str,
        old_keys: Mapping[str, list[str]],
        wrote_keys: bool,
        original_error: Exception,
    ) -> None:
        """Reconcile the snapshot with the exact disk state after a partial delete."""
        try:
            disk_routes, disk_keys, disk_revision = self._read_stable_sources()
            disk_projects = disk_routes.get("facade_keys", {})
            if wrote_keys and project_id in disk_projects:
                try:
                    _atomic_replace(
                        self.secrets_path,
                        old_keys,
                        mode=0o600,
                        expected_hash=_content_hash(self.secrets_path),
                    )
                except Exception:
                    # A configured project without keys is valid and fail-closed.
                    pass
                disk_routes, disk_keys, disk_revision = self._read_stable_sources()
            self._install_snapshot(disk_routes, disk_keys, disk_revision)
        except Exception as recovery_error:
            self._snapshot = None
            self._snapshot_revision = self.revision()
            self._snapshot_error = ProjectKeyManagementError(
                f"Project deletion recovery failed after {type(original_error).__name__}"
            )
            raise self._snapshot_error from recovery_error

    def _read_stable_sources(
        self,
        *,
        attempts: int = 3,
    ) -> tuple[dict[str, Any], dict[str, list[str]], tuple[Any, ...]]:
        for _attempt in range(attempts):
            before = self.revision()
            routes, keys = self._read_sources()
            after = self.revision()
            if before == after:
                return routes, keys, after
        raise ProjectKeyManagementError("Managed configuration changed repeatedly while it was being read")

    def _install_snapshot(
        self,
        routes: Mapping[str, Any],
        keys: Mapping[str, list[str]],
        revision: tuple[Any, ...],
    ) -> FacadeKeyRegistry:
        registry = FacadeKeyRegistry.from_sources(routes.get("facade_keys", {}), keys)
        self._snapshot = registry
        self._snapshot_revision = revision
        self._snapshot_error = None
        return registry

    def _reload_snapshot_from_disk(self) -> FacadeKeyRegistry:
        try:
            routes, keys, revision = self._read_stable_sources()
            return self._install_snapshot(routes, keys, revision)
        except Exception as exc:
            self._snapshot = None
            self._snapshot_revision = self.revision()
            self._snapshot_error = exc
            raise

    def get_registry(self) -> FacadeKeyRegistry:
        revision = self.revision()
        if self._snapshot is not None and self._snapshot_revision == revision and self._snapshot_error is None:
            return self._snapshot
        with self._locked():
            revision = self.revision()
            if self._snapshot is not None and self._snapshot_revision == revision and self._snapshot_error is None:
                return self._snapshot
            try:
                return self._reload_snapshot_from_disk()
            except Exception as exc:
                self._snapshot = None
                self._snapshot_revision = revision
                self._snapshot_error = exc
                raise

    def list_keys(self, project_id: str) -> list[dict[str, str]]:
        registry = self.get_registry()
        return [{"key_id": facade_key_id(value)} for value in registry.get_secrets(project_id)]

    def create_project(self, project_id: Any, display_name: Any = None, tags: Any = None) -> dict[str, Any]:
        project_id = _validate_project_id(project_id)
        config = _project_config(display_name, tags)
        with self._locked():
            routes_hash = _content_hash(self.routes_path)
            routes, keys = self._read_sources()
            projects = routes.setdefault("facade_keys", {})
            if project_id in projects:
                raise ProjectKeyConflictError("Project already exists")
            projects[project_id] = config
            FacadeKeyRegistry.from_sources(projects, keys)
            _atomic_replace(self.routes_path, routes, mode=0o600, expected_hash=routes_hash, preserve_mode=True)
            self._reload_snapshot_from_disk()
        return {"project_id": project_id, **config}

    def delete_project(self, project_id: Any) -> None:
        project_id = str(project_id or "").strip()
        with self._locked():
            routes_hash = _content_hash(self.routes_path)
            keys_hash = _content_hash(self.secrets_path)
            routes, keys = self._read_sources()
            projects = routes.get("facade_keys", {})
            if project_id not in projects:
                raise ProjectKeyNotFoundError(PROJECT_NOT_FOUND_MESSAGE)
            old_keys = copy.deepcopy(keys)
            projects.pop(project_id)
            keys.pop(project_id, None)
            FacadeKeyRegistry.from_sources(projects, keys)
            wrote_keys = keys != old_keys
            try:
                if wrote_keys:
                    _atomic_replace(self.secrets_path, keys, mode=0o600, expected_hash=keys_hash)
                _atomic_replace(self.routes_path, routes, mode=0o600, expected_hash=routes_hash, preserve_mode=True)
            except Exception as exc:
                self._recover_after_delete_failure(project_id, old_keys, wrote_keys, exc)
                raise
            reload_facade_key_secrets()
            self._reload_snapshot_from_disk()

    def create_key(self, project_id: Any, secret: str | None = None) -> tuple[str, str]:
        project_id = str(project_id or "").strip()
        with self._locked():
            routes_hash = _content_hash(self.routes_path)
            keys_hash = _content_hash(self.secrets_path)
            routes, keys = self._read_sources()
            projects = routes.get("facade_keys", {})
            if project_id not in projects:
                raise ProjectKeyNotFoundError(PROJECT_NOT_FOUND_MESSAGE)
            secret = secret or generate_facade_key_secret()
            if any(secret in existing for existing in keys.values()):
                raise ProjectKeyConflictError("Generated key already exists")
            keys.setdefault(project_id, []).append(secret)
            FacadeKeyRegistry.from_sources(projects, keys)
            if _content_hash(self.routes_path) != routes_hash:
                raise ProjectKeyConflictError("Routes configuration changed while it was being updated")
            _atomic_replace(self.secrets_path, keys, mode=0o600, expected_hash=keys_hash)
            reload_facade_key_secrets()
            self._reload_snapshot_from_disk()
            return secret, facade_key_id(secret)

    def revoke_key(self, project_id: Any, key_id: Any) -> None:
        project_id = str(project_id or "").strip()
        key_id = str(key_id or "").strip()
        with self._locked():
            routes_hash = _content_hash(self.routes_path)
            keys_hash = _content_hash(self.secrets_path)
            routes, keys = self._read_sources()
            projects = routes.get("facade_keys", {})
            if project_id not in projects:
                raise ProjectKeyNotFoundError(PROJECT_NOT_FOUND_MESSAGE)
            matches = [value for value in keys.get(project_id, []) if secrets.compare_digest(facade_key_id(value), key_id)]
            if not matches:
                raise ProjectKeyNotFoundError("Key not found")
            if len(matches) != 1:
                raise ProjectKeyConflictError("Key fingerprint is ambiguous")
            keys[project_id] = [value for value in keys[project_id] if value != matches[0]]
            if not keys[project_id]:
                keys.pop(project_id)
            FacadeKeyRegistry.from_sources(projects, keys)
            if _content_hash(self.routes_path) != routes_hash:
                raise ProjectKeyConflictError("Routes configuration changed while it was being updated")
            _atomic_replace(self.secrets_path, keys, mode=0o600, expected_hash=keys_hash)
            reload_facade_key_secrets()
            self._reload_snapshot_from_disk()
