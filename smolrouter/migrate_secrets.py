#!/usr/bin/env python3
"""Consolidate inline provider API keys from a routes config into secrets.yaml.

Safety properties:
- NEVER prints secret values. Output to the operator is provider names + key
  counts only. Diagnostics (missing files) print paths, never contents.
- Writes the output file with mode 0600, atomically (temp + rename).
- Non-destructive: does not modify the input config. It prints a checklist of
  which key fields to remove afterwards (so secrets live only in secrets.yaml).
- Idempotent/mergeable: existing provider entries in the output are preserved
  unless --force is given.

Usage:
    python -m smolrouter.migrate_secrets                         # -> user config dir
    python -m smolrouter.migrate_secrets --out ./secrets.yaml    # dev
    python -m smolrouter.migrate_secrets --dry-run               # report only
    python -m smolrouter.migrate_secrets --config /path/routes.yaml --force
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import yaml
from platformdirs import user_config_dir

from .config_loading import load_config_entries

APP = "smolrouter"
KEY_LIST_FIELDS = ("api_keys",)
KEY_SCALAR_FIELDS = ("api_key",)
KEY_FILE_FIELDS = ("api_keys_file", "api_key_file")
ALL_KEY_FIELDS = (*KEY_LIST_FIELDS, *KEY_SCALAR_FIELDS, *KEY_FILE_FIELDS)


def _dedupe(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def extract_keys(provider: dict) -> List[str]:
    """Collect keys from inline + file-backed fields. BYOK (api_key: null) -> none."""
    keys: List[str] = []

    for field in KEY_LIST_FIELDS:
        value = provider.get(field)
        if isinstance(value, list):
            keys += [str(k).strip() for k in value if k and str(k).strip()]

    for field in KEY_SCALAR_FIELDS:
        value = provider.get(field)
        if isinstance(value, str) and value.strip():
            keys.append(value.strip())

    for field in KEY_FILE_FIELDS:
        ref = provider.get(field)
        if isinstance(ref, str) and ref.strip():
            # Resolve + parse exactly like the runtime provider loaders: relative
            # to CWD (not the config dir), honoring env-style assignments and
            # inline comments. Anything else would write keys the store/provider
            # can't use, or miss config/*.txt refs entirely.
            try:
                keys += load_config_entries(ref, allow_assignments=True, strip_inline_comments=True)
            except Exception as exc:  # path only, never contents
                print(f"  ! {provider.get('name')}: {field} -> {ref}: {type(exc).__name__}; skipped", file=sys.stderr)

    return _dedupe(keys)


def _atomic_write_0600(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".secrets-", suffix=".tmp")
    try:
        if hasattr(os, "fchmod"):  # not available on Windows
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.close(fd)  # fd may already be closed by fdopen; ignore
        except OSError:
            pass
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.chmod(path, 0o600)


def build_secrets(config: dict) -> tuple[Dict[str, List[str]], list[tuple[str, list[str]]]]:
    """Extract {provider: [keys]} and a checklist of fields to remove. Assumes a
    valid 'providers' list (the caller validates and reports)."""
    discovered: Dict[str, List[str]] = {}
    checklist: list[tuple[str, list[str]]] = []
    for provider in config.get("providers") or []:
        if not isinstance(provider, dict):
            continue
        name = str(provider.get("name") or "").strip()
        if not name:
            continue
        keys = extract_keys(provider)
        if keys:
            discovered[name] = keys
            checklist.append((name, [f for f in ALL_KEY_FIELDS if provider.get(f)]))
    return discovered, checklist


def _load_existing(out_path: Path) -> Dict[str, List[str]]:
    if not out_path.is_file():
        return {}
    loaded = yaml.safe_load(out_path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _merge(existing, discovered, force):
    merged = dict(existing)
    added, overwritten, skipped = [], [], []
    for name, keys in discovered.items():
        if name in existing and not force:
            skipped.append(name)
            continue
        (overwritten if name in existing else added).append(name)
        merged[name] = keys
    return merged, added, overwritten, skipped


def _print_summary(config_path, out_path, discovered, skipped):
    print(f"config:  {config_path}")
    print(f"output:  {out_path}")
    print("discovered (provider: key count):")
    for name, keys in discovered.items():
        print(f"  {name}: {len(keys)}")
    if skipped:
        print(f"already present (kept; --force to overwrite): {', '.join(skipped)}")


def _print_write_result(out_path, merged, added, overwritten, checklist):
    print(f"wrote {len(merged)} provider entries to {out_path} (mode 0600)")
    if added:
        print(f"  added: {', '.join(added)}")
    if overwritten:
        print(f"  overwritten: {', '.join(overwritten)}")
    if checklist:
        print("\nNext: remove these key fields from your config (keep 'api_key: null' for BYOK):")
        for name, fields in checklist:
            print(f"  {name}: {', '.join(fields)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Consolidate provider API keys into secrets.yaml (no values printed).")
    parser.add_argument("--config", default=os.getenv("ROUTES_CONFIG", "config/routes.yaml"), help="Input routes config")
    parser.add_argument("--out", default=str(Path(user_config_dir(APP)) / "secrets.yaml"), help="Output secrets.yaml")
    parser.add_argument("--force", action="store_true", help="Overwrite provider entries already in the output")
    parser.add_argument("--dry-run", action="store_true", help="Report only; write nothing")
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser()
    if not config_path.is_file():
        parser.error(f"config not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config.get("providers"), list):
        print("error: no 'providers' list found in config", file=sys.stderr)
        return 1

    discovered, checklist = build_secrets(config)
    out_path = Path(args.out).expanduser()
    merged, added, overwritten, skipped = _merge(_load_existing(out_path), discovered, args.force)

    _print_summary(config_path, out_path, discovered, skipped)

    if args.dry_run:
        print("dry-run: nothing written.")
        return 0

    header = "# SECRETS - DO NOT COMMIT OR PRINT. Generated by smolrouter.migrate_secrets\n"
    _atomic_write_0600(out_path, header + yaml.safe_dump(merged, default_flow_style=False, sort_keys=True))
    _print_write_result(out_path, merged, added, overwritten, checklist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
