"""Tests for scripts/migrate_secrets.py using dummy (non-secret) values."""

import stat
from pathlib import Path

import yaml

from smolrouter import migrate_secrets as _migrate_secrets


def _load_module():
    return _migrate_secrets


def _write_config(tmp_path: Path) -> Path:
    (tmp_path / "keys.txt").write_text("DUMMYG1\nDUMMYG2\n# a comment\nDUMMYG3\n")
    config = {
        "providers": [
            {"name": "google", "type": "google-genai", "api_keys_file": "keys.txt"},
            {"name": "anthropic-test", "type": "anthropic", "api_keys": ["DUMMYANT1", "DUMMYANT2"]},
            {"name": "nvidia-nim", "type": "openai", "api_key": "DUMMYNV"},
            {"name": "openai-main", "type": "openai", "api_key": None},  # BYOK -> skipped
        ]
    }
    path = tmp_path / "routes.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_migration_consolidates_keys_without_leaking(tmp_path, capsys):
    mod = _load_module()
    config = _write_config(tmp_path)
    out = tmp_path / "secrets.yaml"

    rc = mod.main(["--config", str(config), "--out", str(out)])
    assert rc == 0

    data = yaml.safe_load(out.read_text())
    assert data == {
        "google": ["DUMMYG1", "DUMMYG2", "DUMMYG3"],  # file-backed, comment skipped
        "anthropic-test": ["DUMMYANT1", "DUMMYANT2"],
        "nvidia-nim": ["DUMMYNV"],
        # openai-main (api_key: null) is BYOK -> intentionally absent
    }
    assert "openai-main" not in data

    # 0600 perms
    assert stat.S_IMODE(out.stat().st_mode) == 0o600

    # The summary reports counts but NEVER values.
    output = capsys.readouterr().out
    assert "google: 3" in output
    for secret in ("DUMMYG1", "DUMMYG2", "DUMMYG3", "DUMMYANT1", "DUMMYNV"):
        assert secret not in output


def test_dry_run_writes_nothing(tmp_path, capsys):
    mod = _load_module()
    config = _write_config(tmp_path)
    out = tmp_path / "secrets.yaml"

    assert mod.main(["--config", str(config), "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()
    assert "dry-run" in capsys.readouterr().out


def test_merge_preserves_existing_without_force(tmp_path):
    mod = _load_module()
    config = _write_config(tmp_path)
    out = tmp_path / "secrets.yaml"
    out.write_text(yaml.safe_dump({"google": ["PRE-EXISTING"]}))

    mod.main(["--config", str(config), "--out", str(out)])
    data = yaml.safe_load(out.read_text())
    assert data["google"] == ["PRE-EXISTING"]  # kept
    assert data["nvidia-nim"] == ["DUMMYNV"]  # new added

    mod.main(["--config", str(config), "--out", str(out), "--force"])
    data = yaml.safe_load(out.read_text())
    assert data["google"] == ["DUMMYG1", "DUMMYG2", "DUMMYG3"]  # overwritten
