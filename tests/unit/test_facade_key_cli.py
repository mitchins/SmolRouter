"""Unit tests for facade key provisioning CLI."""

import pytest

from smolrouter import facade_key_cli
from smolrouter import manage_facade_keys
from smolrouter.facade_key_store import (
    append_facade_key_secret,
    load_facade_key_secrets,
    save_facade_key_secrets,
    reload_facade_key_secrets,
)


@pytest.fixture
def routes_yaml(tmp_path):
    path = tmp_path / "routes.yaml"
    path.write_text(
        "facade_keys:\n"
        "  project-a:\n"
        "    display_name: Project A\n"
        "  project-b:\n"
        "    display_name: Project B\n"
    )
    return path


def test_cli_missing_project_emits_known_projects_and_no_write(tmp_path, routes_yaml, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMOLROUTER_FACADE_KEYS", raising=False)
    reload_facade_key_secrets()
    result = facade_key_cli.main(["--routes-config", str(routes_yaml), "--facade-keys-file", str(tmp_path / "out.yaml")])

    output = capsys.readouterr().out
    assert result == 1
    assert "No project id provided." in output
    assert "project-a" in output
    assert not (tmp_path / "out.yaml").exists()


def test_cli_unknown_project_rejected_with_helpful_message(tmp_path, routes_yaml, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMOLROUTER_FACADE_KEYS", raising=False)
    reload_facade_key_secrets()
    result = facade_key_cli.main(
        [
            "--routes-config",
            str(routes_yaml),
            "--facade-keys-file",
            str(tmp_path / "out.yaml"),
            "--project",
            "missing-project",
        ]
    )

    output = capsys.readouterr().out
    assert result == 2
    assert "Unknown facade-key id" in output
    assert "project-a" in output
    assert not (tmp_path / "out.yaml").exists()


def test_cli_dry_run_generates_secret_without_writing(tmp_path, routes_yaml, monkeypatch, capsys):
    monkeypatch.setattr(facade_key_cli, "generate_facade_key_secret", lambda: "srk-dry-run")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMOLROUTER_FACADE_KEYS", str(tmp_path / "facade_keys.yaml"))

    out_file = tmp_path / "facade_keys.yaml"
    existing = {"project-a": ["srk-old"]}
    save_facade_key_secrets(out_file, existing)
    reload_facade_key_secrets()

    result = facade_key_cli.main(
        [
            "--routes-config",
            str(routes_yaml),
            "--facade-keys-file",
            str(out_file),
            "--project",
            "project-a",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "dry-run: no write" in output
    assert "srk-dry-run" in output

    # unchanged after dry-run
    assert load_facade_key_secrets() == existing
    assert "srk-dry-run" not in out_file.read_text(encoding="utf-8")


def test_cli_append_supports_rotation_and_writes_0600(tmp_path, routes_yaml, monkeypatch, capsys):
    monkeypatch.setattr(facade_key_cli, "generate_facade_key_secret", lambda: "srk-new")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMOLROUTER_FACADE_KEYS", str(tmp_path / "facade_keys.yaml"))

    out_file = tmp_path / "facade_keys.yaml"
    save_facade_key_secrets(out_file, {"project-a": ["srk-old"]})
    initial = append_facade_key_secret("project-a", "srk-new", {"project-a": ["srk-old"]})[0]
    reload_facade_key_secrets()

    result = facade_key_cli.main(
        [
            "--routes-config",
            str(routes_yaml),
            "--project",
            "project-a",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Generated facade key for project" in output
    assert "secret (copy exactly once): srk-" in output
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8").find("srk-new") != -1
    assert load_facade_key_secrets() == initial
    assert oct(out_file.stat().st_mode)[-3:] == "600"


def test_cli_defaults_new_dedicated_file_to_user_config_when_unset(tmp_path, routes_yaml, monkeypatch, capsys):
    monkeypatch.setattr(facade_key_cli, "generate_facade_key_secret", lambda: "srk-migrated")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SMOLROUTER_FACADE_KEYS", raising=False)

    user_dir = tmp_path / "user-config"
    user_dir.mkdir()
    monkeypatch.setattr("smolrouter.secret_store.user_config_dir", lambda app: str(user_dir))
    monkeypatch.setattr("smolrouter.secret_store.site_config_dir", lambda app: str(tmp_path / "site-config"))
    monkeypatch.setattr(facade_key_cli, "user_config_dir", lambda app: str(user_dir))
    reload_facade_key_secrets()

    result = facade_key_cli.main(
        [
            "--routes-config",
            str(routes_yaml),
            "--project",
            "project-a",
        ]
    )

    output = capsys.readouterr().out
    migrated_file = user_dir / "facade_keys.yaml"
    assert result == 0
    assert migrated_file.exists()
    assert "wrote facade-key secrets to" in output
    assert load_facade_key_secrets() == {"project-a": ["srk-migrated"]}


def test_cli_rewrites_without_clobber_when_secret_already_exists(tmp_path, routes_yaml, monkeypatch, capsys):
    monkeypatch.setattr(facade_key_cli, "generate_facade_key_secret", lambda: "srk-old")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMOLROUTER_FACADE_KEYS", str(tmp_path / "facade_keys.yaml"))

    out_file = tmp_path / "facade_keys.yaml"
    save_facade_key_secrets(out_file, {"project-a": ["srk-old"]})
    reload_facade_key_secrets()

    result = facade_key_cli.main(
        [
            "--routes-config",
            str(routes_yaml),
            "--project",
            "project-a",
        ]
    )
    output = capsys.readouterr().out

    assert result == 0
    assert "already present for project" in output
    assert out_file.read_text(encoding="utf-8").count("srk-old") == 1
    assert load_facade_key_secrets() == {"project-a": ["srk-old"]}


def test_manage_facade_keys_create_subcommand_uses_operator_shape(tmp_path, routes_yaml, monkeypatch, capsys):
    monkeypatch.setattr(facade_key_cli, "generate_facade_key_secret", lambda: "srk-managed")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMOLROUTER_FACADE_KEYS", str(tmp_path / "facade_keys.yaml"))
    reload_facade_key_secrets()

    result = manage_facade_keys.main(
        [
            "create",
            "--routes-config",
            str(routes_yaml),
            "--project",
            "project-a",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Generated facade key for project" in output
    assert load_facade_key_secrets() == {"project-a": ["srk-managed"]}


def test_manage_facade_keys_requires_subcommand(capsys):
    result = manage_facade_keys.main([])
    output = capsys.readouterr().out
    assert result == 2
    assert "create" in output
