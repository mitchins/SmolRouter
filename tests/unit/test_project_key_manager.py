import errno
import hashlib
import json
import multiprocessing
import stat
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml
from fastapi import HTTPException

import smolrouter.app as app_module
import smolrouter.project_key_manager as manager_module
from smolrouter.database import RequestLog
from smolrouter.interfaces import ClientContext
from smolrouter.project_key_manager import (
    ProjectKeyConflictError,
    ProjectKeyManager,
    ProjectKeyManagementError,
    ProjectKeyValidationError,
    facade_key_id,
)


def _process_create_keys(routes_path, secrets_path, start):
    manager = ProjectKeyManager(routes_path, secrets_path)
    for number in range(start, start + 4):
        manager.create_key("project-a", f"srk-process-{number}")


def _manager(tmp_path, routes_text="facade_keys: {}\n"):
    routes = tmp_path / "routes.yaml"
    routes.write_text(routes_text, encoding="utf-8")
    routes.chmod(0o640)
    return ProjectKeyManager(routes, tmp_path / "facade_keys.yaml"), routes


def test_create_project_key_revoke_and_delete_are_immediately_visible(tmp_path):
    manager, routes = _manager(tmp_path)
    created = manager.create_project("team/project", "Project", ["team"])
    assert created["enabled"] is True
    assert stat.S_IMODE(routes.stat().st_mode) == 0o640

    secret, key_id = manager.create_key("team/project", "srk-test-secret")
    assert secret == "srk-test-secret"
    assert key_id == "sha256:" + hashlib.sha256(secret.encode()).hexdigest()
    assert manager.get_registry().resolve_secret(secret).identity.subject_id == "team/project"
    assert stat.S_IMODE(manager.secrets_path.stat().st_mode) == 0o600

    manager.revoke_key("team/project", key_id)
    assert manager.get_registry().resolve_secret(secret) is None
    manager.delete_project("team/project")
    assert "team/project" not in yaml.safe_load(routes.read_text())["facade_keys"]


@pytest.mark.parametrize("project_id", ["", "/bad", "bad/", "bad//id", "bad/../id", " bad", "a" * 129])
def test_new_project_ids_are_strictly_validated(tmp_path, project_id):
    manager, _ = _manager(tmp_path)
    with pytest.raises(ProjectKeyValidationError):
        manager.create_project(project_id)


@pytest.mark.parametrize(
    "project_id",
    ["a", "Project_01", "team/project", "team.alpha/project-2", "0/child.name"],
)
def test_new_project_ids_accept_supported_corpus(tmp_path, project_id):
    manager, _ = _manager(tmp_path)
    assert manager.create_project(project_id)["project_id"] == project_id


def test_project_config_preserves_omitted_tags_vs_explicit_empty_tags(tmp_path):
    manager, routes = _manager(tmp_path)
    manager.create_project("omitted")
    manager.create_project("empty", tags=[])
    projects = yaml.safe_load(routes.read_text(encoding="utf-8"))["facade_keys"]
    assert "tags" not in projects["omitted"]
    assert projects["empty"]["tags"] == []


def test_project_config_trims_display_name_and_tags_at_boundaries(tmp_path):
    manager, routes = _manager(tmp_path)
    manager.create_project("bounded", f" {'d' * 100} ", [f" {'t' * 40} ", *[f"tag-{i}" for i in range(19)]])
    config = yaml.safe_load(routes.read_text(encoding="utf-8"))["facade_keys"]["bounded"]
    assert config["display_name"] == "d" * 100
    assert config["tags"][0] == "t" * 40
    assert len(config["tags"]) == 20


@pytest.mark.parametrize(
    ("display_name", "tags", "message"),
    [
        (123, None, "display_name"),
        (" ", None, "display_name"),
        ("x" * 101, None, "display_name"),
        (None, "tag", "tags must be a list"),
        (None, ["tag"] * 21, "at most 20"),
        (None, [123], "each tag"),
        (None, [" "], "each tag"),
        (None, ["x" * 41], "each tag"),
        (None, [" tag ", "tag"], "unique"),
    ],
)
def test_project_config_rejects_invalid_types_bounds_and_trimmed_duplicates(
    tmp_path, display_name, tags, message
):
    manager, _ = _manager(tmp_path)
    with pytest.raises(ProjectKeyValidationError, match=message):
        manager.create_project("project-a", display_name=display_name, tags=tags)


@pytest.mark.parametrize("phase", ["open", "fsync"])
def test_directory_fsync_ignores_only_supported_platform_errors(tmp_path, phase):
    open_mock = Mock(return_value=42)
    fsync_mock = Mock()
    close_mock = Mock()
    if phase == "open":
        open_mock.side_effect = OSError(errno.ENOTSUP, "unsupported")
    else:
        fsync_mock.side_effect = OSError(errno.EOPNOTSUPP, "unsupported")
    with patch.object(manager_module.os, "open", open_mock), patch.object(
        manager_module.os, "fsync", fsync_mock
    ), patch.object(manager_module.os, "close", close_mock):
        manager_module._fsync_directory(tmp_path)
    if phase == "open":
        fsync_mock.assert_not_called()
        close_mock.assert_not_called()
    else:
        close_mock.assert_called_once_with(42)


@pytest.mark.parametrize("phase", ["open", "fsync"])
def test_directory_fsync_reraises_unexpected_errors(tmp_path, phase):
    open_mock = Mock(return_value=42)
    fsync_mock = Mock()
    close_mock = Mock()
    if phase == "open":
        open_mock.side_effect = OSError(errno.EACCES, "denied")
    else:
        fsync_mock.side_effect = OSError(errno.EIO, "failed")
    with patch.object(manager_module.os, "open", open_mock), patch.object(
        manager_module.os, "fsync", fsync_mock
    ), patch.object(manager_module.os, "close", close_mock), pytest.raises(OSError):
        manager_module._fsync_directory(tmp_path)
    if phase == "fsync":
        close_mock.assert_called_once_with(42)


def test_directory_fsync_on_windows_returns_before_open(tmp_path):
    open_mock = Mock(side_effect=AssertionError("directory open must not run on Windows"))
    with patch.object(manager_module.os, "name", "nt"), patch.object(manager_module.os, "open", open_mock):
        manager_module._fsync_directory(tmp_path)
    open_mock.assert_not_called()


def test_manager_rejects_symlinked_managed_file(tmp_path):
    target = tmp_path / "target.yaml"
    target.write_text("facade_keys: {}\n")
    routes = tmp_path / "routes.yaml"
    routes.symlink_to(target)
    manager = ProjectKeyManager(routes, tmp_path / "facade_keys.yaml")
    with pytest.raises(ProjectKeyManagementError, match="symlink"):
        manager.get_registry()


def test_manager_rejects_symlinked_secrets_file(tmp_path):
    manager, _ = _manager(tmp_path, "facade_keys:\n  project-a: {}\n")
    target = tmp_path / "target-keys.yaml"
    target.write_text("project-a: [srk-live]\n", encoding="utf-8")
    manager.secrets_path.symlink_to(target)
    with pytest.raises(ProjectKeyManagementError, match="symlink"):
        manager.get_registry()


@pytest.mark.parametrize(
    ("routes_name", "routes_text", "error"),
    [
        ("routes.yaml", "facade_keys: []\n", "facade_keys must be a mapping"),
        ("routes.yaml", "[broken", "Unable to read managed configuration"),
        ("routes.json", '{"facade_keys": []}\n', "facade_keys must be a mapping"),
        ("routes.json", "{broken", "Unable to read managed configuration"),
    ],
)
def test_manager_rejects_malformed_routes_sources(tmp_path, routes_name, routes_text, error):
    routes = tmp_path / routes_name
    routes.write_text(routes_text, encoding="utf-8")
    manager = ProjectKeyManager(routes, tmp_path / "facade_keys.yaml")
    with pytest.raises(ProjectKeyManagementError, match=error):
        manager.get_registry()


@pytest.mark.parametrize(
    ("routes_name", "routes_text"),
    [("routes.yaml", "facade_keys:\n"), ("routes.json", '{"facade_keys": null}\n')],
)
def test_manager_accepts_and_replaces_explicit_null_facade_key_sections(tmp_path, routes_name, routes_text):
    routes = tmp_path / routes_name
    routes.write_text(routes_text, encoding="utf-8")
    manager = ProjectKeyManager(routes, tmp_path / "facade_keys.yaml")
    assert manager.get_registry().key_ids() == ()
    manager.create_project("project-a")
    stored = json.loads(routes.read_text()) if routes.suffix == ".json" else yaml.safe_load(routes.read_text())
    assert stored["facade_keys"]["project-a"] == {"enabled": True}


def test_manager_accepts_truly_blank_yaml_for_read_and_create(tmp_path):
    manager, routes = _manager(tmp_path, "")
    assert manager.get_registry().key_ids() == ()
    manager.create_project("project-a")
    assert yaml.safe_load(routes.read_text())["facade_keys"]["project-a"] == {"enabled": True}


def test_json_routes_preserve_unrelated_fields(tmp_path):
    routes = tmp_path / "routes.json"
    routes.write_text('{"providers": [{"name": "local"}], "facade_keys": {}}\n', encoding="utf-8")
    manager = ProjectKeyManager(routes, tmp_path / "facade_keys.yaml")
    manager.create_project("project-a")
    stored = json.loads(routes.read_text(encoding="utf-8"))
    assert stored["providers"] == [{"name": "local"}]
    assert stored["facade_keys"]["project-a"] == {"enabled": True}


def test_existing_duplicate_secrets_fail_closed_instead_of_becoming_unrevokeable(tmp_path):
    manager, _ = _manager(tmp_path, "facade_keys:\n  project-a: {}\n")
    manager.secrets_path.write_text("project-a: [srk-dup, srk-dup]\n", encoding="utf-8")
    with pytest.raises(ProjectKeyManagementError, match="duplicate values"):
        manager.get_registry()


@pytest.mark.parametrize("content", ["[broken", "[]\n", "project-a: {nested: value}\n"])
def test_manager_rejects_malformed_secrets_sources(tmp_path, content):
    manager, _ = _manager(tmp_path, "facade_keys:\n  project-a: {}\n")
    manager.secrets_path.write_text(content, encoding="utf-8")
    with pytest.raises(ProjectKeyManagementError, match="Unable to read managed facade keys"):
        manager.get_registry()


def test_external_invalid_change_never_serves_stale_credentials(tmp_path):
    manager, routes = _manager(tmp_path, "facade_keys:\n  project-a: {}\n")
    manager.create_key("project-a", "srk-live")
    assert manager.get_registry().resolve_secret("srk-live") is not None
    routes.write_text("facade_keys: []\n", encoding="utf-8")
    with pytest.raises(ProjectKeyManagementError):
        manager.get_registry()
    with pytest.raises(ProjectKeyManagementError):
        manager.get_registry()


def test_external_replacement_during_read_never_pairs_stale_credentials_with_new_revision(tmp_path, monkeypatch):
    manager, routes = _manager(tmp_path, "facade_keys:\n  project-a: {}\n")
    manager.create_key("project-a", "srk-old")
    manager._snapshot = None
    real_read = manager._read_sources
    replaced = False

    def replace_after_stale_parse():
        nonlocal replaced
        result = real_read()
        if not replaced:
            replaced = True
            routes.write_text("facade_keys: {}\n", encoding="utf-8")
            manager.secrets_path.write_text("{}\n", encoding="utf-8")
        return result

    monkeypatch.setattr(manager, "_read_sources", replace_after_stale_parse)
    registry = manager.get_registry()
    assert registry.resolve_secret("srk-old") is None
    assert manager.get_registry().resolve_secret("srk-old") is None
    assert manager._snapshot_revision == manager.revision()


def test_threaded_key_creation_does_not_lose_updates(tmp_path):
    manager, _ = _manager(tmp_path, "facade_keys:\n  project-a: {}\n")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda number: manager.create_key("project-a", f"srk-{number}"), range(12)))
    assert len(manager.get_registry().get_secrets("project-a")) == 12
    assert len(manager.list_keys("project-a")) == 12


def test_process_key_creation_does_not_lose_updates(tmp_path):
    manager, routes = _manager(tmp_path, "facade_keys:\n  project-a: {}\n")
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(target=_process_create_keys, args=(routes, manager.secrets_path, start))
        for start in (0, 4, 8)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    assert len(manager.get_registry().get_secrets("project-a")) == 12


def test_external_edit_between_read_and_replace_returns_conflict(tmp_path, monkeypatch):
    manager, routes = _manager(tmp_path)
    original_read = manager._read_sources

    def racing_read():
        result = original_read()
        routes.write_text("facade_keys:\n  externally-added: {}\n", encoding="utf-8")
        return result

    monkeypatch.setattr(manager, "_read_sources", racing_read)
    with pytest.raises(ProjectKeyConflictError):
        manager.create_project("project-a")
    assert "externally-added" in routes.read_text(encoding="utf-8")


def test_revoke_requires_exact_full_fingerprint(tmp_path):
    manager, _ = _manager(tmp_path, "facade_keys:\n  project-a: {}\n")
    _, key_id = manager.create_key("project-a", "srk-live")
    assert facade_key_id("srk-live") == key_id
    with pytest.raises(Exception, match="Key not found"):
        manager.revoke_key("project-a", key_id[:-1])


def test_duplicate_secret_is_rejected_before_it_can_make_revocation_ambiguous(tmp_path):
    manager, _ = _manager(tmp_path, "facade_keys:\n  project-a: {}\n")
    manager.create_key("project-a", "srk-live")
    with pytest.raises(ProjectKeyConflictError, match="already exists"):
        manager.create_key("project-a", "srk-live")
    assert manager.get_registry().get_secrets("project-a") == ("srk-live",)


def test_project_delete_rolls_keys_back_when_routes_write_fails(tmp_path):
    manager, routes = _manager(tmp_path, "facade_keys:\n  project-a: {}\n")
    manager.create_key("project-a", "srk-live")
    real_replace = manager_module._atomic_replace

    def fail_routes(path, payload, **kwargs):
        if path == routes:
            raise OSError("routes write failed")
        return real_replace(path, payload, **kwargs)

    with patch("smolrouter.project_key_manager._atomic_replace", side_effect=fail_routes):
        with pytest.raises(OSError, match="routes write failed"):
            manager.delete_project("project-a")

    assert manager.get_registry().resolve_secret("srk-live") is not None
    assert yaml.safe_load(routes.read_text(encoding="utf-8"))["facade_keys"] == {"project-a": {}}


def test_project_delete_keeps_committed_files_consistent_when_directory_fsync_fails(tmp_path):
    manager, routes = _manager(tmp_path, "facade_keys:\n  project-a: {}\n")
    manager.create_key("project-a", "srk-live")
    real_fsync = manager_module._fsync_directory
    calls = 0

    def fail_after_routes_replace(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("routes directory fsync failed")
        return real_fsync(path)

    with patch("smolrouter.project_key_manager._fsync_directory", side_effect=fail_after_routes_replace):
        with pytest.raises(OSError, match="routes directory fsync failed"):
            manager.delete_project("project-a")

    assert yaml.safe_load(routes.read_text(encoding="utf-8"))["facade_keys"] == {}
    assert yaml.safe_load(manager.secrets_path.read_text(encoding="utf-8")) == {}
    assert manager.get_registry().get_config("project-a") is None


def test_project_delete_recovers_when_secret_directory_fsync_fails_after_replace(tmp_path):
    manager, routes = _manager(tmp_path, "facade_keys:\n  project-a: {}\n")
    manager.create_key("project-a", "srk-live")
    real_fsync = manager_module._fsync_directory
    calls = 0

    def fail_after_secret_replace(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("secret directory fsync failed")
        return real_fsync(path)

    with patch("smolrouter.project_key_manager._fsync_directory", side_effect=fail_after_secret_replace):
        with pytest.raises(OSError, match="secret directory fsync failed"):
            manager.delete_project("project-a")

    assert yaml.safe_load(routes.read_text(encoding="utf-8"))["facade_keys"] == {"project-a": {}}
    assert manager.get_registry().resolve_secret("srk-live") is not None


def test_project_delete_rollback_failure_installs_fail_closed_disk_snapshot(tmp_path):
    manager, routes = _manager(tmp_path, "facade_keys:\n  project-a: {}\n")
    manager.create_key("project-a", "srk-live")
    real_replace = manager_module._atomic_replace
    secret_writes = 0

    def fail_routes_and_rollback(path, payload, **kwargs):
        nonlocal secret_writes
        if path == routes:
            raise OSError("routes write failed")
        secret_writes += 1
        if secret_writes == 2:
            raise OSError("rollback failed")
        return real_replace(path, payload, **kwargs)

    with patch("smolrouter.project_key_manager._atomic_replace", side_effect=fail_routes_and_rollback):
        with pytest.raises(OSError, match="routes write failed"):
            manager.delete_project("project-a")

    assert manager.get_registry().get_config("project-a") is not None
    assert manager.get_registry().resolve_secret("srk-live") is None


def test_atomic_replace_falls_back_when_fchmod_is_unavailable(tmp_path, monkeypatch):
    manager, routes = _manager(tmp_path)
    monkeypatch.delattr(manager_module.os, "fchmod")
    manager.create_project("project-a")
    assert yaml.safe_load(routes.read_text(encoding="utf-8"))["facade_keys"]["project-a"]["enabled"] is True


@pytest.mark.asyncio
async def test_management_api_requires_same_origin_json_and_secret_is_no_store(async_client, tmp_path):
    manager, _ = _manager(tmp_path)
    container = Mock()
    container.get_project_key_manager.return_value = manager

    async def active(_legacy):
        return container

    security = Mock()
    security.check_webui_access.return_value = None
    headers = {"Origin": "http://test", "Content-Type": "application/json"}
    with patch("smolrouter.app._get_active_container", active), patch(
        "smolrouter.app.get_webui_security", return_value=security
    ):
        denied = await async_client.post(
            "/api/project-management/projects",
            json={"project_id": "project-a"},
            headers={"Origin": "https://evil.example"},
        )
        wrong_type = await async_client.post(
            "/api/project-management/projects",
            content='{"project_id":"project-a"}',
            headers={"Origin": "http://test", "Content-Type": "text/plain"},
        )
        created = await async_client.post(
            "/api/project-management/projects", json={"project_id": "project-a"}, headers=headers
        )
        keyed = await async_client.post(
            "/api/project-management/keys", json={"project_id": "project-a"}, headers=headers
        )

    assert denied.status_code == 403
    assert wrong_type.status_code == 415
    assert created.status_code == 201
    assert keyed.status_code == 201
    payload = keyed.json()
    assert payload["secret"].startswith("srk-")
    assert payload["key_id"].startswith("sha256:") and len(payload["key_id"]) == 71
    assert keyed.headers["cache-control"] == "no-store, no-cache"
    assert keyed.headers["pragma"] == "no-cache"
    assert manager.get_registry().resolve_secret(payload["secret"]) is not None


@pytest.mark.asyncio
async def test_management_api_returns_500_when_manager_dependency_is_unavailable(async_client):
    async def unavailable(_legacy):
        return None

    security = Mock()
    security.check_webui_access.return_value = None
    with patch("smolrouter.app._get_active_container", unavailable), patch(
        "smolrouter.app.get_webui_security", return_value=security
    ):
        response = await async_client.post(
            "/api/project-management/projects",
            json={"project_id": "project-a"},
            headers={"Origin": "http://test"},
        )
    assert response.status_code == 500
    assert response.json()["detail"] == "Project/key management is unavailable"


@pytest.mark.asyncio
async def test_management_api_preserves_webui_authentication_401(async_client):
    security = Mock()
    security.check_webui_access.side_effect = HTTPException(status_code=401, detail="authentication required")
    with patch("smolrouter.app.get_webui_security", return_value=security):
        response = await async_client.post(
            "/api/project-management/projects",
            json={"project_id": "project-a"},
            headers={"Origin": "http://test"},
        )
    assert response.status_code == 401


def test_management_openapi_declares_mutation_error_responses():
    app_module.app.openapi_schema = None
    schema = app_module.app.openapi()["paths"]
    expected_base = {"401", "403", "415", "422", "500"}
    expected_by_operation = {
        ("/api/project-management/projects", "post"): {"409"},
        ("/api/project-management/projects", "delete"): {"404", "409"},
        ("/api/project-management/keys", "post"): {"404", "409"},
        ("/api/project-management/keys", "delete"): {"404", "409"},
    }
    for (path, method), extra in expected_by_operation.items():
        assert expected_base | extra <= set(schema[path][method]["responses"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {"Content-Type": "application/json"},
        {"Origin": "null", "Content-Type": "application/json"},
        {"Origin": "http://test/path", "Content-Type": "application/json"},
        [("Origin", "http://test"), ("Origin", "http://test"), ("Content-Type", "application/json")],
        [("Host", "test"), ("Host", "evil.example"), ("Origin", "http://test"), ("Content-Type", "application/json")],
    ],
)
async def test_management_api_rejects_missing_null_malformed_or_duplicate_origin(async_client, tmp_path, headers):
    manager, _ = _manager(tmp_path)
    container = Mock()
    container.get_project_key_manager.return_value = manager

    async def active(_legacy):
        return container

    security = Mock()
    security.check_webui_access.return_value = None
    with patch("smolrouter.app._get_active_container", active), patch(
        "smolrouter.app.get_webui_security", return_value=security
    ):
        response = await async_client.post(
            "/api/project-management/projects",
            content='{"project_id":"project-a"}',
            headers=headers,
        )
    assert response.status_code == 403
    assert manager.get_registry().get_config("project-a") is None


@pytest.mark.asyncio
async def test_management_api_accepts_default_origin_port_and_rejects_extra_fields(async_client, tmp_path):
    manager, _ = _manager(tmp_path)
    container = Mock()
    container.get_project_key_manager.return_value = manager

    async def active(_legacy):
        return container

    security = Mock()
    security.check_webui_access.return_value = None
    with patch("smolrouter.app._get_active_container", active), patch(
        "smolrouter.app.get_webui_security", return_value=security
    ):
        extra = await async_client.post(
            "/api/project-management/projects",
            json={"project_id": "project-a", "unexpected": True},
            headers={"Origin": "http://test:80"},
        )
        created = await async_client.post(
            "/api/project-management/projects",
            json={"project_id": "project-a"},
            headers={"Origin": "http://test:80"},
        )
    assert extra.status_code == 422
    assert created.status_code == 201


@pytest.mark.asyncio
async def test_management_api_key_revoke_and_project_delete_update_runtime_immediately(async_client, tmp_path):
    manager, _ = _manager(tmp_path)
    container = Mock()
    container.get_project_key_manager.return_value = manager

    async def active(_legacy):
        return container

    security = Mock()
    security.check_webui_access.return_value = None
    headers = {"Origin": "http://test", "Content-Type": "application/json"}
    with patch("smolrouter.app._get_active_container", active), patch(
        "smolrouter.app.get_webui_security", return_value=security
    ):
        assert (
            await async_client.post(
                "/api/project-management/projects", json={"project_id": "project-a"}, headers=headers
            )
        ).status_code == 201
        created_key = await async_client.post(
            "/api/project-management/keys", json={"project_id": "project-a"}, headers=headers
        )
        secret = created_key.json()["secret"]
        key_id = created_key.json()["key_id"]
        assert manager.get_registry().resolve_secret(secret) is not None
        revoked = await async_client.request(
            "DELETE",
            "/api/project-management/keys",
            json={"project_id": "project-a", "key_id": key_id},
            headers=headers,
        )
        deleted = await async_client.request(
            "DELETE",
            "/api/project-management/projects",
            json={"project_id": "project-a"},
            headers=headers,
        )
    assert revoked.status_code == 200
    assert deleted.status_code == 200
    assert manager.get_registry().resolve_secret(secret) is None
    assert manager.get_registry().get_config("project-a") is None


@pytest.mark.asyncio
async def test_created_and_revoked_key_updates_actual_inference_auth_path(async_client, tmp_path, monkeypatch):
    manager, _ = _manager(tmp_path)
    container = Mock()
    container.get_project_key_manager.return_value = manager
    container.get_facade_key_registry.side_effect = manager.get_registry
    container.create_client_context.side_effect = lambda **kwargs: ClientContext(**kwargs)
    container.route_request = AsyncMock(return_value=({"choices": []}, 200, "provider:test", None))

    async def active(_legacy):
        return container

    security = Mock()
    security.check_webui_access.return_value = None
    monkeypatch.setattr(app_module, "_is_legacy_proxy_mode", lambda: False)
    headers = {"Origin": "http://test", "Content-Type": "application/json"}
    with patch("smolrouter.app._get_active_container", active), patch(
        "smolrouter.app.get_webui_security", return_value=security
    ):
        await async_client.post(
            "/api/project-management/projects", json={"project_id": "project-a"}, headers=headers
        )
        created = await async_client.post(
            "/api/project-management/keys", json={"project_id": "project-a"}, headers=headers
        )
        secret = created.json()["secret"]
        key_id = created.json()["key_id"]
        accepted = await async_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
            headers={app_module.FACADE_KEY_HEADER: secret},
        )
        await async_client.request(
            "DELETE",
            "/api/project-management/keys",
            json={"project_id": "project-a", "key_id": key_id},
            headers=headers,
        )
        rejected = await async_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
            headers={app_module.FACADE_KEY_HEADER: secret},
        )

    assert accepted.status_code == 200
    assert rejected.status_code == 401
    assert rejected.json()["detail"] == "Invalid SmolRouter facade key"


@pytest.mark.asyncio
async def test_management_api_rejects_deleting_historical_only_project(async_client, tmp_path):
    manager, _ = _manager(tmp_path)
    container = Mock()
    container.get_project_key_manager.return_value = manager

    async def active(_legacy):
        return container

    await RequestLog.create(
        source_ip="127.0.0.1",
        method="POST",
        path="/v1/chat/completions",
        identity_kind="facade_key",
        identity_subject_id="historical-only",
        request_id="historical-only-management-test",
    )
    security = Mock()
    security.check_webui_access.return_value = None
    with patch("smolrouter.app._get_active_container", active), patch(
        "smolrouter.app.get_webui_security", return_value=security
    ):
        response = await async_client.request(
            "DELETE",
            "/api/project-management/projects",
            json={"project_id": "historical-only"},
            headers={"Origin": "http://test", "Content-Type": "application/json"},
        )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_project_ui_has_management_controls_but_never_secret_values(async_client, tmp_path):
    manager, _ = _manager(tmp_path, "facade_keys:\n  project-a: {}\n")
    manager.create_key("project-a", "srk-never-render-this")
    container = Mock()
    container.get_facade_key_registry.side_effect = manager.get_registry
    limits = Mock(enabled=False, project_default=None)
    container.get_request_rate_limit_config.return_value = limits

    async def active(_legacy):
        return container

    security = Mock()
    security.check_webui_access.return_value = None
    with patch("smolrouter.app._get_active_container", active), patch(
        "smolrouter.app.get_webui_security", return_value=security
    ):
        listing = await async_client.get("/projects")
        detail = await async_client.get("/projects/project-a")
    assert listing.status_code == 200 and "Create project" in listing.text
    assert detail.status_code == 200 and "Create API key" in detail.text and "Delete project" in detail.text
    assert listing.text.count("window.smolrouterProjectMutation =") == 1
    assert detail.text.count("window.smolrouterProjectMutation =") == 1
    assert "fallbackMessage = 'Request failed'" in listing.text
    assert "smolrouterProjectMutation('/api/project-management/projects', 'POST', payload, 'Project creation failed')" in listing.text
    assert "async function mutate" not in detail.text
    assert "typeof result.detail === 'string' ? result.detail : fallbackMessage" in detail.text
    assert 'for="delete-confirmation"' in detail.text
    assert "Copy key" in detail.text
    assert "navigator.clipboard.writeText" in detail.text
    assert "Copy failed. Select and copy the key manually" in detail.text
    assert "pagehide" in detail.text
    assert "sha256:" in detail.text
    assert "srk-never-render-this" not in listing.text + detail.text
