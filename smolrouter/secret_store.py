"""Centralized provider secret loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import os
import yaml
from platformdirs import site_config_dir, user_config_dir

APP = "smolrouter"


@dataclass(frozen=True)
class SecretStoreData:
    provider_keys: Dict[str, List[str]]


_CACHED_SECRETS: Optional[SecretStoreData] = None
_LAST_SECRETS_PATHS: List[Path] = []


def secrets_search_paths() -> List[str]:
    return [str(path) for path in _LAST_SECRETS_PATHS]


def _candidate_secret_paths(filename: str, env_var: Optional[str]) -> List[Path]:
    paths: List[Path] = []

    env_value = os.getenv(env_var) if env_var else None
    if env_value is not None:
        paths.append((Path(env_value).expanduser()).resolve())

    paths.append((Path.cwd() / filename).resolve())
    paths.append((Path(user_config_dir(APP)) / filename).resolve())
    paths.append((Path(site_config_dir(APP)) / filename).resolve())

    return paths


def resolve_config_file(filename: str, env_var: Optional[str]) -> Path | None:
    """
    Resolve the secrets file path.

    Resolution order:
      1. explicit env override (if present)
      2. <cwd>/<filename>
      3. user config dir
      4. site config dir

    The env override always wins, even when it points to a missing file.
    """

    global _LAST_SECRETS_PATHS

    candidate_paths = _candidate_secret_paths(filename, env_var)
    _LAST_SECRETS_PATHS = candidate_paths

    env_value = os.getenv(env_var) if env_var else None
    if env_value is not None:
        return Path(env_value).expanduser().resolve()

    # No env override -> candidate_paths is [cwd, user, site]; check all of them.
    # A [1:] slice here would skip the ./ dev path, the one that matters most in
    # development.
    for path in candidate_paths:
        if path.is_file():
            return path

    return None


def normalize_secret_values(values: object) -> List[str]:
    """Normalize provider secret definitions into non-empty strings."""
    if isinstance(values, str):
        values = [values]
    elif not isinstance(values, list):
        return []

    normalized: List[str] = []
    for value in values:
        if not isinstance(value, str):
            if value is None:
                continue
            value = str(value)
        candidate = value.strip()
        if candidate:
            normalized.append(candidate)

    return normalized


def _normalize_provider_values(values: object) -> List[str]:
    return normalize_secret_values(values)


def _copy_secret_mapping(mapping: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {name: keys[:] for name, keys in mapping.items()}


def _parse_secrets_payload(parsed: Dict[object, object]) -> SecretStoreData:
    provider_keys: Dict[str, List[str]] = {}
    for entry_name, values in parsed.items():
        normalized_name = str(entry_name)
        normalized = _normalize_provider_values(values)
        if not normalized:
            continue
        provider_keys[normalized_name] = normalized

    return SecretStoreData(provider_keys=provider_keys)


def _load_secret_store_data() -> SecretStoreData:
    """Load and cache provider secret material."""

    global _CACHED_SECRETS

    if _CACHED_SECRETS is not None:
        return SecretStoreData(
            provider_keys=_copy_secret_mapping(_CACHED_SECRETS.provider_keys),
        )

    env_value = os.getenv("SMOLROUTER_SECRETS")
    path = resolve_config_file("secrets.yaml", "SMOLROUTER_SECRETS")

    if env_value is not None and path is not None and not path.is_file():
        raise FileNotFoundError(f"Secrets file not found at explicit override SMOLROUTER_SECRETS={path}")

    if path is None or not path.is_file():
        _CACHED_SECRETS = SecretStoreData(provider_keys={})
        return SecretStoreData(provider_keys={})

    raw = path.read_text(encoding="utf-8")
    try:
        if not raw.strip():
            parsed = {}
        else:
            parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse secrets YAML at {path}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Secrets file must be a mapping, got {type(parsed).__name__}: {path}")

    if not parsed:
        _CACHED_SECRETS = SecretStoreData(provider_keys={})
        return SecretStoreData(provider_keys={})

    _CACHED_SECRETS = _parse_secrets_payload(parsed)
    return SecretStoreData(provider_keys=_copy_secret_mapping(_CACHED_SECRETS.provider_keys))


def load_secrets() -> Dict[str, List[str]]:
    """Load consolidated secrets file -> provider_name->[keys]."""
    return _load_secret_store_data().provider_keys


def load_facade_key_secrets() -> Dict[str, List[str]]:
    """Load facade-key secrets via the dedicated facade-key store."""
    from .facade_key_store import load_facade_key_secrets as _load_facade_key_secrets

    return _load_facade_key_secrets()


def reload_secrets() -> None:
    global _CACHED_SECRETS

    _CACHED_SECRETS = None
    from .facade_key_store import reload_facade_key_secrets

    reload_facade_key_secrets()


def get_keys(provider_name: str) -> List[str]:
    return list((load_secrets() if provider_name else {}).get(provider_name, []))


def get_facade_key_secrets(facade_key_id: str) -> List[str]:
    return list((load_facade_key_secrets() if facade_key_id else {}).get(facade_key_id, []))


def redact_secret(value: str | None) -> str:
    """Redact a secret for logging."""

    if not value:
        return "***"

    text = value.strip()
    if not text:
        return "***"

    if len(text) <= 5:
        return "***"

    return f"{text[:5]}…"
