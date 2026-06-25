"""Unit tests for dedicated facade-key secret storage."""

import stat

import pytest

from smolrouter.facade_key_store import (
    append_facade_key_secret,
    load_facade_key_secrets,
    resolve_facade_key_config_path,
    reload_facade_key_secrets,
    save_facade_key_secrets,
)


@pytest.fixture(autouse=True)
def _reset_facade_key_cache():
    reload_facade_key_secrets()
    yield
    reload_facade_key_secrets()


@pytest.fixture
def configured_dirs(monkeypatch, tmp_path):
    cwd = tmp_path / "cwd"
    user = tmp_path / "user-config"
    site = tmp_path / "site-config"
    for path in (cwd, user, site):
        path.mkdir()

    monkeypatch.chdir(cwd)
    monkeypatch.setattr("smolrouter.secret_store.user_config_dir", lambda app: str(user))
    monkeypatch.setattr("smolrouter.secret_store.site_config_dir", lambda app: str(site))
    return cwd, user, site


def test_facade_key_path_prefers_explicit_env(monkeypatch, configured_dirs):
    cwd, user, site = configured_dirs

    (cwd / "facade_keys.yaml").write_text("{}")
    (user / "facade_keys.yaml").write_text("{}")
    (site / "facade_keys.yaml").write_text("{}")

    explicit = cwd / "facade-keys-explicit.yaml"
    explicit.write_text("{}")

    monkeypatch.setenv("SMOLROUTER_FACADE_KEYS", str(explicit))
    resolved = resolve_facade_key_config_path()
    assert resolved == explicit.resolve()

    from smolrouter.secret_store import secrets_search_paths

    assert secrets_search_paths() == [
        str(explicit.resolve()),
        str((cwd / "facade_keys.yaml").resolve()),
        str((user / "facade_keys.yaml").resolve()),
        str((site / "facade_keys.yaml").resolve()),
    ]


def test_load_facade_key_store_prefers_cwd_then_user_then_site(monkeypatch, configured_dirs):
    cwd, user, site = configured_dirs
    monkeypatch.delenv("SMOLROUTER_FACADE_KEYS", raising=False)

    cwd_file = cwd / "facade_keys.yaml"
    user_file = user / "facade_keys.yaml"
    site_file = site / "facade_keys.yaml"

    cwd_file.write_text("{}")
    user_file.write_text("{}")
    site_file.write_text("{}")
    assert resolve_facade_key_config_path() == cwd_file.resolve()

    cwd_file.unlink()
    assert resolve_facade_key_config_path() == user_file.resolve()

    user_file.unlink()
    assert resolve_facade_key_config_path() == site_file.resolve()


def test_facade_key_store_fallback_to_legacy_nested_facade_keys(monkeypatch, configured_dirs):
    cwd, _user, _site = configured_dirs

    (cwd / "secrets.yaml").write_text(
        "google: sk-google\n"
        "facade_keys:\n"
        "  project-a: srk-project-a\n"
        "  project-b:\n"
        "    - srk-project-b-1\n"
        "    - srk-project-b-2\n"
    )
    monkeypatch.delenv("SMOLROUTER_FACADE_KEYS", raising=False)
    reload_facade_key_secrets()

    assert load_facade_key_secrets() == {}


def test_append_facade_key_secret_rotates_without_clobber():
    original = {"project-a": ["srk-old-1"]}
    updated, changed = append_facade_key_secret("project-a", "srk-new", original)

    assert changed
    assert original["project-a"] == ["srk-old-1"]
    assert updated["project-a"] == ["srk-old-1", "srk-new"]


def test_save_facade_key_secrets_writes_atomic_0600(tmp_path, monkeypatch):
    path = tmp_path / "facade_keys.yaml"
    save_facade_key_secrets(path, {"project-a": ["srk-1", "srk-2"]})

    text = path.read_text(encoding="utf-8")
    assert "project-a" in text
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    monkeypatch.setenv("SMOLROUTER_FACADE_KEYS", str(path))
    reload_facade_key_secrets()
    loaded = load_facade_key_secrets()
    assert loaded == {"project-a": ["srk-1", "srk-2"]}
