"""Facade-key secret loading and persistence."""

from __future__ import annotations

import logging
import os
import secrets
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .secret_store import normalize_secret_values, resolve_config_file

FACADE_KEY_SECRET_ENV = "SMOLROUTER_FACADE_KEYS"
FACADE_KEY_PREFIX = "srk-"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FacadeKeyStoreData:
    facade_keys: Dict[str, List[str]]


_CACHED_FACADE_KEYS: Optional[FacadeKeyStoreData] = None


def resolve_facade_key_config_path(explicit_env_var: Optional[str] = None) -> Path | None:
    """Resolve facade-key secret file path."""
    return resolve_config_file("facade_keys.yaml", explicit_env_var or FACADE_KEY_SECRET_ENV)


def _read_yaml_mapping(path: Path) -> Dict[str, object]:
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Facade key secrets source must be a mapping, got {type(parsed).__name__}: {path}")
    return {str(key): value for key, value in parsed.items()}


def _parse_facade_key_map(raw: Dict[str, object], path: Path) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for key_id, raw_values in raw.items():
        if not isinstance(raw_values, (str, list, type(None))):
            raise ValueError(
                f"Facade key secret entry '{key_id}' must be a scalar string or list at {path}, "
                f"got {type(raw_values).__name__}"
            )
        values = normalize_secret_values(raw_values)
        if values:
            normalized[str(key_id).strip()] = values
    return normalized


def _load_dedicated_facade_keys(path: Path) -> Dict[str, List[str]]:
    raw = _read_yaml_mapping(path)
    parsed = _parse_facade_key_map(raw, path)
    return {key: values[:] for key, values in parsed.items() if key}


def load_facade_key_secrets_from_path(path: Path) -> Dict[str, List[str]]:
    """Load facade-key secrets from a specific dedicated file path."""
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        return {}
    loaded = _load_dedicated_facade_keys(candidate)
    return {key: values[:] for key, values in loaded.items()}


def _load_facade_key_secret_data() -> FacadeKeyStoreData:
    global _CACHED_FACADE_KEYS

    if _CACHED_FACADE_KEYS is not None:
        return FacadeKeyStoreData(
            facade_keys={key: values[:] for key, values in _CACHED_FACADE_KEYS.facade_keys.items()}
        )

    env_value = os.getenv(FACADE_KEY_SECRET_ENV)
    dedicated_path = resolve_facade_key_config_path()
    if env_value is not None and dedicated_path is not None and not dedicated_path.is_file():
        raise FileNotFoundError(
            f"Facade key secrets file not found at explicit override {FACADE_KEY_SECRET_ENV}={dedicated_path}"
        )

    if dedicated_path is not None and dedicated_path.is_file():
        loaded = _load_dedicated_facade_keys(dedicated_path)
        _CACHED_FACADE_KEYS = FacadeKeyStoreData(facade_keys=loaded)
        return FacadeKeyStoreData(facade_keys={key: values[:] for key, values in loaded.items()})

    _CACHED_FACADE_KEYS = FacadeKeyStoreData(facade_keys={})
    return FacadeKeyStoreData(facade_keys={})


def load_facade_key_secrets() -> Dict[str, List[str]]:
    """Load facade-key secrets from dedicated secrets file with legacy fallback."""
    return _load_facade_key_secret_data().facade_keys


def reload_facade_key_secrets() -> None:
    global _CACHED_FACADE_KEYS
    _CACHED_FACADE_KEYS = None


def get_facade_key_project_secrets(project_id: str) -> List[str]:
    """Load facade keys for a specific logical project id."""
    return list((load_facade_key_secrets() if project_id else {}).get(project_id, []))


def generate_facade_key_secret() -> str:
    """Generate a random facade secret with `srk-` prefix."""
    return f"{FACADE_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def _dedupe(values: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def append_facade_key_secret(
    project_id: str,
    secret: str,
    facade_key_map: Dict[str, List[str]],
) -> tuple[Dict[str, List[str]], bool]:
    """Return rotated secrets and whether the value was newly added."""
    normalized_project = str(project_id).strip()
    normalized_secret = str(secret).strip()
    if not normalized_project or not normalized_secret:
        raise ValueError("project_id and secret are required")

    updated = {key: values[:] for key, values in facade_key_map.items()}
    existing = updated.get(normalized_project, [])
    if normalized_secret in existing:
        return updated, False

    updated[normalized_project] = _dedupe(list(existing) + [normalized_secret])
    return updated, True


def _atomic_write_yaml_0600(path: Path, payload: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_dump(payload, sort_keys=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".facade-keys-", suffix=".tmp")
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.close(fd)  # nosec B018 - fd may already be closed by os.fdopen
        except OSError:
            pass
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.chmod(path, 0o600)


def save_facade_key_secrets(path: Path, facade_keys: Dict[str, List[str]]) -> None:
    if path is None:
        raise ValueError("Facade key destination path is required")
    payload: Dict[str, List[str]] = {
        key_id: _dedupe(list(values))
        for key_id, values in sorted(facade_keys.items())
        if key_id.strip()
    }
    _atomic_write_yaml_0600(path, payload)


def write_facade_key_secret(
    project_id: str,
    secret: str,
    path: Path,
) -> bool:
    existing = load_facade_key_secrets_from_path(path)
    updated, added = append_facade_key_secret(project_id=project_id, secret=secret, facade_key_map=existing)
    if not added:
        return False
    save_facade_key_secrets(path=path, facade_keys=updated)
    reload_facade_key_secrets()
    return True
