"""Helpers for deterministic configuration path resolution."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROUTES_CONFIG_PATH = PROJECT_ROOT / "config" / "routes.yaml"
DEFAULT_BLOB_STORAGE_PATH = Path.home() / ".smolrouter" / "blob_storage"


def resolve_routes_config_path(raw_path: Optional[str] = None) -> Path:
    """Resolve the routes config path deterministically.

    Rules:
    - Explicit paths are resolved relative to the current working directory.
    - Without an explicit path, use the repository-local default config path.
    """

    if raw_path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate.resolve()

    return DEFAULT_ROUTES_CONFIG_PATH.resolve()


def routes_config_base_dir(routes_config_path: Path) -> Path:
    """Return the base directory used for relative file references in routes config.

    When the config lives under a conventional `config/` folder, treat the parent of
    that folder as the base so entries like `config/google_api_keys.txt` keep working
    regardless of the shell's current directory.
    """

    config_path = Path(routes_config_path).resolve()
    if config_path.parent.name == "config":
        return config_path.parent.parent
    return config_path.parent


def _normalize_relative_path(raw_value: str, base_dir: Path) -> str:
    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def normalize_provider_file_references(config_data: Dict[str, Any], routes_config_path: Path) -> Dict[str, Any]:
    """Resolve provider file-backed settings against the resolved routes config root."""

    normalized = copy.deepcopy(config_data)
    providers = normalized.get("providers")
    if not isinstance(providers, list):
        return config_data

    base_dir = routes_config_base_dir(routes_config_path)
    changed = False

    for provider in providers:
        if not isinstance(provider, dict):
            continue

        for field_name in ("api_keys_file", "api_key_file"):
            raw_value = provider.get(field_name)
            if not isinstance(raw_value, str) or not raw_value:
                continue

            new_value = _normalize_relative_path(raw_value, base_dir)
            if provider[field_name] != new_value:
                changed = True
            provider[field_name] = new_value

    return normalized if changed else config_data


def resolve_blob_storage_path(raw_path: Optional[str] = None) -> Path:
    """Resolve the filesystem blob storage path deterministically.

    - Explicit paths are resolved relative to the current working directory.
    - Without an explicit path, use a user-state directory under the home folder.
    """

    if raw_path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate.resolve()

    return DEFAULT_BLOB_STORAGE_PATH.resolve()