"""Unit tests for consolidated secrets store and provider injection."""

import logging

import pytest

from smolrouter.google_genai_provider import GoogleGenAIConfig, GoogleGenAIProvider
from smolrouter.providers import ProviderFactory
from smolrouter.secret_store import (
    get_facade_key_secrets,
    load_secrets,
    load_facade_key_secrets,
    reload_secrets,
    redact_secret,
    resolve_config_file,
    secrets_search_paths,
)


@pytest.fixture(autouse=True)
def _reset_secret_cache():
    """load_secrets() caches globally; reset around every test for isolation."""
    reload_secrets()
    yield
    reload_secrets()


@pytest.fixture
def configured_dirs(monkeypatch, tmp_path):
    """Provide deterministic platform directory overrides for secrets resolution."""
    cwd = tmp_path / "cwd"
    user = tmp_path / "user-config"
    site = tmp_path / "site-config"
    for path in (cwd, user, site):
        path.mkdir()

    monkeypatch.chdir(cwd)
    monkeypatch.setattr("smolrouter.secret_store.user_config_dir", lambda app: str(user))
    monkeypatch.setattr("smolrouter.secret_store.site_config_dir", lambda app: str(site))
    return cwd, user, site


def test_resolve_config_file_prefers_explicit_env(monkeypatch, configured_dirs):
    cwd, user, site = configured_dirs

    (cwd / "secrets.yaml").write_text("{}")
    (user / "secrets.yaml").write_text("{}")
    (site / "secrets.yaml").write_text("{}")

    explicit = cwd / "explicit-secrets.yaml"
    explicit.write_text("{}")

    monkeypatch.setenv("SMOLROUTER_SECRETS", str(explicit))
    assert resolve_config_file("secrets.yaml", "SMOLROUTER_SECRETS") == explicit.resolve()
    assert secrets_search_paths() == [
        str(explicit.resolve()),
        str((cwd / "secrets.yaml").resolve()),
        str((user / "secrets.yaml").resolve()),
        str((site / "secrets.yaml").resolve()),
    ]


def test_resolve_config_file_prefers_dev_then_user_then_site(monkeypatch, configured_dirs):
    cwd, user, site = configured_dirs

    cwd_file = cwd / "secrets.yaml"
    user_file = user / "secrets.yaml"
    site_file = site / "secrets.yaml"

    monkeypatch.delenv("SMOLROUTER_SECRETS", raising=False)
    cwd_file.write_text("{}")
    user_file.write_text("{}")
    site_file.write_text("{}")

    assert resolve_config_file("secrets.yaml", "SMOLROUTER_SECRETS") == cwd_file.resolve()

    cwd_file.unlink()
    assert resolve_config_file("secrets.yaml", "SMOLROUTER_SECRETS") == user_file.resolve()

    user_file.unlink()
    assert resolve_config_file("secrets.yaml", "SMOLROUTER_SECRETS") == site_file.resolve()


def test_load_secrets_explicit_env_override_must_exist(monkeypatch, configured_dirs):
    cwd, _user, _site = configured_dirs

    monkeypatch.chdir(cwd)
    monkeypatch.setenv("SMOLROUTER_SECRETS", str(cwd / "missing" / "secrets.yaml"))
    reload_secrets()

    with pytest.raises(FileNotFoundError, match="missing/secrets.yaml"):
        load_secrets()


def test_load_secrets_normalizes_scalar_and_list_values(monkeypatch, configured_dirs):
    cwd, _user, _site = configured_dirs

    (cwd / "secrets.yaml").write_text(
        "google: sk-google\nanthropic:\n  - sk-ant-1\n  -\n  -  null\n  - sk-ant-2\n"
    )
    monkeypatch.delenv("SMOLROUTER_SECRETS", raising=False)
    reload_secrets()

    assert load_secrets() == {"google": ["sk-google"], "anthropic": ["sk-ant-1", "sk-ant-2"]}


def test_load_secrets_supports_nested_facade_keys_without_affecting_provider_keys(monkeypatch, configured_dirs):
    cwd, _user, _site = configured_dirs

    (cwd / "secrets.yaml").write_text(
        "google: sk-google\n"
        "facade_keys:\n"
        "  project-a: srk-project-a\n"
        "  project-b:\n"
        "    - srk-project-b-1\n"
        "    - srk-project-b-2\n"
    )
    monkeypatch.delenv("SMOLROUTER_SECRETS", raising=False)
    reload_secrets()

    assert load_secrets() == {"google": ["sk-google"]}
    assert load_facade_key_secrets() == {
        "project-a": ["srk-project-a"],
        "project-b": ["srk-project-b-1", "srk-project-b-2"],
    }
    assert get_facade_key_secrets("project-b") == ["srk-project-b-1", "srk-project-b-2"]


def test_load_secrets_keeps_legacy_facade_keys_scalar_as_provider_entry(monkeypatch, configured_dirs):
    cwd, _user, _site = configured_dirs

    (cwd / "secrets.yaml").write_text("facade_keys: legacy-provider-secret\n")
    monkeypatch.delenv("SMOLROUTER_SECRETS", raising=False)
    reload_secrets()

    assert load_secrets() == {"facade_keys": ["legacy-provider-secret"]}
    assert load_facade_key_secrets() == {}


def test_load_facade_key_secrets_requires_nested_mapping(monkeypatch, configured_dirs):
    cwd, _user, _site = configured_dirs

    (cwd / "secrets.yaml").write_text("facade_keys:\n  project-a:\n    nested: bad\n")
    monkeypatch.delenv("SMOLROUTER_SECRETS", raising=False)
    reload_secrets()

    with pytest.raises(ValueError, match="Facade key secret entry 'project-a' must be a scalar string or list"):
        load_facade_key_secrets()


def test_load_secrets_empty_file_returns_empty(monkeypatch, configured_dirs):
    cwd, _user, _site = configured_dirs

    (cwd / "secrets.yaml").write_text("")
    monkeypatch.delenv("SMOLROUTER_SECRETS", raising=False)
    reload_secrets()

    assert load_secrets() == {}


def test_provider_factory_prefers_secret_store_keys_for_google_and_openai(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(
        "smolrouter.providers.get_keys",
        lambda name: {
            "google": ["g-secret-1", "g-secret-2", "g-secret-3"],
            "openai-test": ["o-secret-1", "o-secret-2"],
        }.get(name, []),
    )

    inline_google_key_file = tmp_path / "google-inline.txt"
    inline_google_key_file.write_text("ignored")

    provider_defs = [
        {
            "name": "google",
            "type": "google-genai",
            "enabled": True,
            "url": "https://generativelanguage.googleapis.com",
            "api_keys": ["inline-google"],
            "api_keys_file": str(inline_google_key_file),
        },
        {
            "name": "openai-test",
            "type": "openai",
            "enabled": True,
            "url": "https://example.com/openai/v1",
            "api_key": "inline-openai",
        },
    ]

    with caplog.at_level(logging.WARNING):
        providers = ProviderFactory.create_providers_from_config(provider_defs)

    google_provider = next(p for p in providers if p.get_provider_type() == "google-genai")
    openai_provider = next(p for p in providers if p.get_provider_type() == "openai")

    assert google_provider.config.api_keys == ["g-secret-1", "g-secret-2", "g-secret-3"]
    assert openai_provider.config.api_key == "o-secret-1"
    assert "Inline keys for provider 'google' are being overridden by the secrets store" in caplog.text
    assert "single-key provider; using the first" in caplog.text


def test_openai_bypass_remains_keyless_when_store_has_no_keys(monkeypatch):
    monkeypatch.setattr("smolrouter.providers.get_keys", lambda _name: [])

    providers = ProviderFactory.create_providers_from_config(
        [
            {
                "name": "openai-passthrough",
                "type": "openai",
                "enabled": True,
                "url": "https://example.com/openai/v1",
                "api_key": None,
            }
        ]
    )

    assert len(providers) == 1
    assert providers[0].config.api_key is None


def test_strict_secrets_mode_rejects_inline_keys(monkeypatch):
    monkeypatch.setattr("smolrouter.providers.get_keys", lambda _name: [])
    monkeypatch.setenv("SMOLROUTER_REQUIRE_SECRETS", "1")

    with pytest.raises(ValueError, match="inline key configuration"):
        ProviderFactory._apply_secrets_to_provider_config(
            {
                "name": "google",
                "type": "google-genai",
                "api_keys": ["inline-google"],
            }
        )


def test_strict_secrets_mode_allows_openai_bypass(monkeypatch):
    monkeypatch.setattr("smolrouter.providers.get_keys", lambda _name: [])
    monkeypatch.setenv("SMOLROUTER_REQUIRE_SECRETS", "1")

    processed = {"name": "openai", "type": "openai", "url": "http://localhost:8000", "api_key": None}
    ProviderFactory._apply_secrets_to_provider_config(processed)

    assert processed["api_key"] is None


def test_strict_secrets_mode_rejects_openai_without_explicit_null(monkeypatch):
    monkeypatch.setattr("smolrouter.providers.get_keys", lambda _name: [])
    monkeypatch.setenv("SMOLROUTER_REQUIRE_SECRETS", "1")

    with pytest.raises(ValueError, match="has no keys in the secrets store"):
        ProviderFactory._apply_secrets_to_provider_config(
            {
                "name": "openai",
                "type": "openai",
                "url": "http://localhost:8000",
            }
        )


def test_strict_secrets_mode_ignores_non_sensitive_provider(monkeypatch):
    monkeypatch.setattr("smolrouter.providers.get_keys", lambda _name: [])
    monkeypatch.setenv("SMOLROUTER_REQUIRE_SECRETS", "1")

    processed = {"name": "dummy", "type": "dummy", "url": "http://localhost:8000"}
    ProviderFactory._apply_secrets_to_provider_config(processed)

    assert processed["type"] == "dummy"


def test_openai_bypass_stays_keyless_even_when_secrets_exist(monkeypatch):
    monkeypatch.setattr("smolrouter.providers.get_keys", lambda _name: ["secret-key"])

    processed = {"name": "openai", "type": "openai", "url": "http://localhost:8000", "api_key": None}
    ProviderFactory._apply_secrets_to_provider_config(processed)

    assert processed["api_key"] is None


def test_google_config_error_includes_secret_search_paths(monkeypatch, configured_dirs):
    _cwd, _user, _site = configured_dirs

    monkeypatch.delenv("SMOLROUTER_SECRETS", raising=False)

    with pytest.raises(ValueError, match="Looked in:") as exc:
        GoogleGenAIConfig(name="google", type="google-genai", enabled=True)

    message = str(exc.value)
    for path in secrets_search_paths():
        assert path in message


def test_redact_secret_no_full_key_leaked_in_google_provider_init_logs(caplog):
    full_key = "sk-super-long-key-for-testing-no-leak"

    assert redact_secret(full_key) != full_key
    assert full_key not in redact_secret(full_key)

    with caplog.at_level(logging.INFO):
        config = GoogleGenAIConfig(name="google-log", type="google-genai", enabled=True, api_keys=[full_key])
        GoogleGenAIProvider(config)

    assert full_key not in caplog.text
