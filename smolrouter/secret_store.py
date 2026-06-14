"""Centralized secrets loading for provider API keys.

Secrets are loaded from a YAML file with the following shape::

    provider_name:
      - key

Values may be a scalar string (treated as a single-item list) or a list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import os
import yaml
from platformdirs import site_config_dir, user_config_dir

APP = "smolrouter"

_CACHED_SECRETS: Optional[Dict[str, List[str]]] = None
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
        if path.exists():
            return path

    return None


def _normalize_provider_values(values: object) -> List[str]:
    if isinstance(values, str):
        values = [values]
    elif isinstance(values, list):
        pass
    else:
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


def load_secrets() -> Dict[str, List[str]]:
    """Load consolidated secrets file -> provider_name->[keys]."""

    global _CACHED_SECRETS

    if _CACHED_SECRETS is not None:
        return {name: keys[:] for name, keys in _CACHED_SECRETS.items()}

    env_value = os.getenv("SMOLROUTER_SECRETS")
    path = resolve_config_file("secrets.yaml", "SMOLROUTER_SECRETS")

    if env_value is not None and path is not None and not path.exists():
        raise FileNotFoundError(f"Secrets file not found at explicit override SMOLROUTER_SECRETS={path}")

    if path is None or not path.exists():
        _CACHED_SECRETS = {}
        return {}

    raw = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw) if raw else {}

    if not parsed:
        _CACHED_SECRETS = {}
        return {}

    if not isinstance(parsed, dict):
        raise ValueError(f"Secrets file must be a mapping, got {type(parsed).__name__}: {path}")

    secrets: Dict[str, List[str]] = {}
    for provider_name, values in parsed.items():
        normalized = _normalize_provider_values(values)
        if not normalized:
            continue
        secrets[str(provider_name)] = normalized

    _CACHED_SECRETS = secrets
    return {name: keys[:] for name, keys in secrets.items()}


def reload_secrets() -> None:
    global _CACHED_SECRETS

    _CACHED_SECRETS = None


def get_keys(provider_name: str) -> List[str]:
    return list((load_secrets() if provider_name else {}).get(provider_name, []))


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
