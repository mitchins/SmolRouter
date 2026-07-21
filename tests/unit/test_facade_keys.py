import pytest

from smolrouter.facade_keys import (
    FacadeKeyRegistry,
    RequestIdentity,
    load_facade_key_registry,
)


def test_registry_normalizes_config_and_rotation_lists(monkeypatch):
    monkeypatch.setattr(
        "smolrouter.facade_keys.load_facade_key_secrets",
        lambda: {"project-a": ["srk-a-v1", "srk-a-v2"]},
    )

    registry = load_facade_key_registry(
        {
            "project-a": {
                "display_name": "Project A",
                "tags": ["team:ml", "env:dev"],
                "default_class": "normal",
                "quota": {
                    "daily_requests_soft": 100,
                    "daily_tokens_soft": 2000,
                    "warn_threshold": 0.9,
                },
            }
        }
    )

    assert registry.key_ids() == ("project-a",)
    config = registry.get_config("project-a")
    assert config is not None
    assert config.display_name == "Project A"
    assert config.tags == ("team:ml", "env:dev")
    assert config.default_class == "normal"
    assert config.quota.daily_requests_soft == 100
    assert config.quota.daily_tokens_soft == 2000
    assert config.quota.action == "observe"
    assert config.quota.warn_threshold == 0.9
    assert registry.get_secrets("project-a") == ("srk-a-v1", "srk-a-v2")
    assert registry.to_dict()["project-a"]["secret_count"] == 2


def test_registry_rejects_unknown_secret_ids():
    with pytest.raises(ValueError, match="unknown logical ids"):
        FacadeKeyRegistry.from_sources(
            facade_key_configs={"project-a": {}},
            facade_key_secrets={"project-b": ["srk-b"]},
        )


def test_registry_rejects_disabled_keys_with_live_secrets():
    with pytest.raises(ValueError, match="Disabled facade keys cannot retain live secrets"):
        FacadeKeyRegistry.from_sources(
            facade_key_configs={"project-a": {"enabled": False}},
            facade_key_secrets={"project-a": ["srk-a"]},
        )


def test_registry_rejects_duplicate_secrets_across_logical_ids():
    with pytest.raises(ValueError, match="must be unique across logical ids"):
        FacadeKeyRegistry.from_sources(
            facade_key_configs={"project-a": {}, "project-b": {}},
            facade_key_secrets={"project-a": ["srk-shared"], "project-b": ["srk-shared"]},
        )


def test_registry_rejects_invalid_tags_and_warn_threshold():
    with pytest.raises(ValueError, match="field 'tags' must be a list"):
        FacadeKeyRegistry.from_sources(
            facade_key_configs={"project-a": {"tags": "not-a-list"}},
            facade_key_secrets={},
        )

    with pytest.raises(ValueError, match="warn_threshold"):
        FacadeKeyRegistry.from_sources(
            facade_key_configs={"project-a": {"quota": {"warn_threshold": 2}}},
            facade_key_secrets={},
        )


def test_registry_rejects_invalid_quota_action():
    with pytest.raises(ValueError, match="quota.action"):
        FacadeKeyRegistry.from_sources(
            facade_key_configs={"project-a": {"quota": {"action": "warnn"}}},
            facade_key_secrets={},
        )


def test_registry_parses_enabled_without_python_truthiness():
    registry = FacadeKeyRegistry.from_sources(
        facade_key_configs={
            "project-a": {"enabled": "false"},
            "project-b": {"enabled": "true"},
            "project-c": {"enabled": 0},
            "project-d": {"enabled": 1},
        },
        facade_key_secrets={},
    )

    assert registry.get_config("project-a").enabled is False
    assert registry.get_config("project-b").enabled is True
    assert registry.get_config("project-c").enabled is False
    assert registry.get_config("project-d").enabled is True

    with pytest.raises(ValueError, match="field 'enabled' must be a boolean"):
        FacadeKeyRegistry.from_sources(
            facade_key_configs={"project-e": {"enabled": "sometimes"}},
            facade_key_secrets={},
        )


def test_registry_rejects_duplicate_ids_after_normalization():
    with pytest.raises(ValueError, match="Duplicate facade key id after normalization"):
        FacadeKeyRegistry.from_sources(
            facade_key_configs={"project-a": {}, " project-a ": {}},
            facade_key_secrets={},
        )

    with pytest.raises(ValueError, match="Duplicate facade key secret id after normalization"):
        FacadeKeyRegistry.from_sources(
            facade_key_configs={"project-a": {}},
            facade_key_secrets={"project-a": ["srk-a"], " project-a ": ["srk-a-2"]},
        )


def test_request_identity_dataclass_has_stable_fields():
    identity = RequestIdentity(
        kind="facade_key",
        subject_id="project-a",
        display_name="Project A",
        tags=("team:ml",),
        default_class="normal",
        quota_policy={"daily_tokens_soft": 1000},
        token_accounting_state="estimated",
    )

    assert identity.subject_id == "project-a"
    assert identity.token_accounting_state == "estimated"


def test_registry_resolves_presented_secret_to_identity():
    registry = FacadeKeyRegistry.from_sources(
        facade_key_configs={
            "project-a": {
                "display_name": "Project A",
                "tags": ["team:ml"],
                "default_class": "normal",
                "quota": {"daily_tokens_soft": 1000},
            }
        },
        facade_key_secrets={"project-a": ["srk-a-v1"]},
    )

    resolved = registry.resolve_secret("  srk-a-v1  ")

    assert resolved is not None
    assert resolved.authentication_principal == "facade_key:project-a"
    assert resolved.identity is not None
    assert resolved.identity.subject_id == "project-a"
    assert resolved.identity.display_name == "Project A"
    assert resolved.identity.tags == ("team:ml",)
    assert resolved.identity.quota_policy["daily_tokens_soft"] == 1000


def test_registry_parses_complete_project_rate_limit_override():
    registry = FacadeKeyRegistry.from_sources(
        facade_key_configs={
            "project-a": {
                "rate_limit": {
                    "windows": [
                        {"requests": 5, "seconds": 1},
                        {"requests": 30, "seconds": 10},
                    ]
                }
            }
        },
        facade_key_secrets={"project-a": ["srk-a"]},
    )

    config = registry.get_config("project-a")
    assert config is not None
    assert config.rate_limit is not None
    assert config.rate_limit.to_dict() == {
        "windows": [
            {"requests": 5, "seconds": 1},
            {"requests": 30, "seconds": 10},
        ]
    }
    assert registry.to_dict()["project-a"]["rate_limit"] == config.rate_limit.to_dict()
    assert registry.resolve_secret("srk-a").identity.rate_limit_policy == config.rate_limit


def test_registry_keeps_missing_project_rate_limit_as_inherited():
    registry = FacadeKeyRegistry.from_sources(
        facade_key_configs={"project-a": {}},
        facade_key_secrets={},
    )

    assert registry.get_config("project-a").rate_limit is None
    assert registry.to_dict()["project-a"]["rate_limit"] is None


@pytest.mark.parametrize(
    "rate_limit",
    [
        {"windows": [{"requests": 5, "seconds": 1}]},
        {"windows": [{"requests": 5, "seconds": 1}, {"requests": 30}]},
        {"requests": 5, "seconds": 1},
    ],
)
def test_registry_rejects_malformed_or_partial_project_rate_limit(rate_limit):
    with pytest.raises(ValueError, match=r"facade_keys\.project-a\.rate_limit"):
        FacadeKeyRegistry.from_sources(
            facade_key_configs={"project-a": {"rate_limit": rate_limit}},
            facade_key_secrets={},
        )
