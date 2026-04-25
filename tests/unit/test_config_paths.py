from smolrouter.config_paths import (
    resolve_blob_storage_path,
    normalize_provider_file_references,
    resolve_routes_config_path,
    routes_config_base_dir,
)


def test_resolve_routes_config_path_defaults_to_repo_config():
    resolved = resolve_routes_config_path()

    assert resolved.name == "routes.yaml"
    assert resolved.parent.name == "config"


def test_routes_config_base_dir_uses_repo_root_for_config_directory(tmp_path):
    routes_path = tmp_path / "repo" / "config" / "routes.yaml"
    routes_path.parent.mkdir(parents=True)
    routes_path.write_text("routes: []\n")

    assert routes_config_base_dir(routes_path) == tmp_path / "repo"


def test_normalize_provider_file_references_uses_routes_config_base_dir(tmp_path):
    repo_root = tmp_path / "repo"
    config_dir = repo_root / "config"
    routes_path = config_dir / "routes.yaml"
    config_dir.mkdir(parents=True)
    routes_path.write_text("routes: []\n")

    config_data = {
        "routes": [],
        "providers": [
            {"name": "google", "type": "google-genai", "api_keys_file": "config/google_api_keys.txt"},
            {"name": "zai", "type": "zai-coding", "api_key_file": "glm.env"},
        ],
    }

    normalized = normalize_provider_file_references(config_data, routes_path)

    assert normalized["providers"][0]["api_keys_file"] == str((repo_root / "config" / "google_api_keys.txt").resolve())
    assert normalized["providers"][1]["api_key_file"] == str((repo_root / "glm.env").resolve())


def test_resolve_routes_config_path_respects_explicit_relative_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    routes_path = config_dir / "routes.yaml"
    routes_path.write_text("routes: []\n")

    assert resolve_routes_config_path("config/routes.yaml") == routes_path.resolve()


def test_resolve_blob_storage_path_defaults_to_user_state_dir():
    resolved = resolve_blob_storage_path()

    assert resolved.name == "blob_storage"
    assert resolved.parent.name == ".smolrouter"


def test_resolve_blob_storage_path_respects_relative_dev_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    resolved = resolve_blob_storage_path("./blob_storage")

    assert resolved == (tmp_path / "blob_storage").resolve()