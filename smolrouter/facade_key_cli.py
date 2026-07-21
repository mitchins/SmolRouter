#!/usr/bin/env python3
"""Compatibility wrapper for facade-key operator commands."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from platformdirs import user_config_dir
from typing import Sequence
import yaml

from .config_paths import resolve_routes_config_path
from .facade_key_store import (
    FACADE_KEY_PREFIX,
    append_facade_key_secret,
    generate_facade_key_secret,
    load_facade_key_secrets_from_path,
    resolve_facade_key_config_path,
)
from .project_key_manager import ProjectKeyManager

def configure_create_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--routes-config", default=os.getenv("ROUTES_CONFIG"))
    parser.add_argument("--facade-keys-file", default=None, help="Destination facade-key secret file")
    parser.add_argument("--project", help="Logical facade-key id from routes.yaml facade_keys metadata")
    parser.add_argument("--dry-run", action="store_true", help="Generate and validate without writing")
    return parser


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper for facade-key provisioning. "
            "Preferred usage: python -m smolrouter.manage_facade_keys create --project <id>"
        )
    )
    return configure_create_parser(parser)


def _load_routes_facade_config(routes_path: Path) -> dict:
    raw = routes_path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw) if raw else {}
    if not loaded:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Routes config must be a mapping, got {type(loaded).__name__}")
    facade_keys = loaded.get("facade_keys", {})
    if not isinstance(facade_keys, dict):
        return {}
    return {str(key).strip(): value for key, value in facade_keys.items() if str(key).strip()}


def _normalize_secret_paths(args_facade_key_path: str | None) -> Path:
    if args_facade_key_path:
        return Path(args_facade_key_path).expanduser().resolve()

    resolved = resolve_facade_key_config_path()
    if resolved is not None:
        return resolved

    return (Path(user_config_dir("smolrouter")) / "facade_keys.yaml").resolve()


def _resolve_routes_config_path(raw_path: str | None) -> Path:
    resolved = resolve_routes_config_path(raw_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Routes config not found: {resolved}")
    return resolved


def _print_known_projects(facade_key_configs: dict) -> None:
    if not facade_key_configs:
        print("No facade key metadata found in routes config.")
        return

    print("Known facade_key ids (routes.yaml facade_keys):")
    for key_id, metadata in sorted(facade_key_configs.items()):
        display_name = metadata.get("display_name") if isinstance(metadata, dict) else None
        if display_name:
            print(f"  - {key_id}: {display_name}")
        else:
            print(f"  - {key_id}")


def run_create(args: argparse.Namespace) -> int:
    routes_path = _resolve_routes_config_path(args.routes_config)
    facade_key_configs = _load_routes_facade_config(routes_path)
    project_id = (args.project or "").strip()

    if not project_id:
        print("No project id provided.")
        _print_known_projects(facade_key_configs)
        print("Pass --project <id> to create a facade-key secret.")
        return 1

    if project_id not in facade_key_configs:
        print(f"Unknown facade-key id: {project_id!r}")
        _print_known_projects(facade_key_configs)
        return 2

    secret_file_path = _normalize_secret_paths(args.facade_keys_file)
    generated = generate_facade_key_secret()
    if not generated.startswith(FACADE_KEY_PREFIX):
        print("Failed to generate a valid facade-key secret.")
        return 3

    if secret_file_path.is_file():
        existing = load_facade_key_secrets_from_path(secret_file_path)
    else:
        existing = {}
    updated, added = append_facade_key_secret(project_id, generated, existing)
    if not added:
        print(f"Secret already present for project {project_id}; no update required.")
        return 0

    if args.dry_run:
        print(f"dry-run: no write to {secret_file_path}")
        print(f"project_id={project_id}, generated_secret={generated}")
        return 0

    manager = ProjectKeyManager(routes_path=routes_path, secrets_path=secret_file_path)
    generated, _key_id = manager.create_key(project_id, generated)
    print(f"Generated facade key for project: {project_id}")
    print(f"secret (copy exactly once): {generated}")
    print(f"wrote facade-key secrets to: {secret_file_path}")
    print(f"secret_count: {len(updated.get(project_id, []))}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return run_create(args)


if __name__ == "__main__":
    raise SystemExit(main())
