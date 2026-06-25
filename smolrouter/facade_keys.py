"""Facade-key configuration, identity, and registry primitives.

Phase A intentionally stops at data-model and validation groundwork.
No request-path behavior changes live in this module yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

from .secret_store import load_facade_key_secrets


DEFAULT_FACADE_KEY_ACTION = "observe"
DEFAULT_FACADE_KEY_WARN_THRESHOLD = 0.8


@dataclass(frozen=True)
class FacadeQuotaPolicy:
    daily_requests_soft: Optional[int] = None
    daily_tokens_soft: Optional[int] = None
    action: str = DEFAULT_FACADE_KEY_ACTION
    warn_threshold: float = DEFAULT_FACADE_KEY_WARN_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "daily_requests_soft": self.daily_requests_soft,
            "daily_tokens_soft": self.daily_tokens_soft,
            "action": self.action,
            "warn_threshold": self.warn_threshold,
        }


@dataclass(frozen=True)
class FacadeKeyConfig:
    key_id: str
    enabled: bool = True
    display_name: Optional[str] = None
    tags: tuple[str, ...] = ()
    default_class: Optional[str] = None
    quota: FacadeQuotaPolicy = field(default_factory=FacadeQuotaPolicy)


@dataclass(frozen=True)
class RequestIdentity:
    kind: str
    subject_id: str
    display_name: Optional[str] = None
    tags: tuple[str, ...] = ()
    default_class: Optional[str] = None
    quota_policy: Dict[str, Any] = field(default_factory=dict)
    token_accounting_state: Optional[str] = None


@dataclass(frozen=True)
class ResolvedCaller:
    authentication_principal: str
    identity: Optional[RequestIdentity] = None


def _coerce_positive_int(value: Any, *, field_name: str, key_id: str) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Facade key '{key_id}' field '{field_name}' must be an integer") from exc
    if normalized < 0:
        raise ValueError(f"Facade key '{key_id}' field '{field_name}' cannot be negative")
    return normalized


def _coerce_warn_threshold(value: Any, *, key_id: str) -> float:
    if value in (None, ""):
        return DEFAULT_FACADE_KEY_WARN_THRESHOLD
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Facade key '{key_id}' field 'warn_threshold' must be numeric") from exc
    if not 0 <= normalized <= 1:
        raise ValueError(f"Facade key '{key_id}' field 'warn_threshold' must be between 0 and 1")
    return normalized


def _normalize_tags(value: Any, *, key_id: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Facade key '{key_id}' field 'tags' must be a list")
    normalized = []
    for tag in value:
        if tag is None:
            continue
        candidate = str(tag).strip()
        if candidate:
            normalized.append(candidate)
    return tuple(normalized)


def _normalize_quota_policy(value: Any, *, key_id: str) -> FacadeQuotaPolicy:
    if value in (None, ""):
        return FacadeQuotaPolicy()
    if not isinstance(value, dict):
        raise ValueError(f"Facade key '{key_id}' field 'quota' must be a mapping")
    action = str(value.get("action", DEFAULT_FACADE_KEY_ACTION)).strip() or DEFAULT_FACADE_KEY_ACTION
    return FacadeQuotaPolicy(
        daily_requests_soft=_coerce_positive_int(value.get("daily_requests_soft"), field_name="daily_requests_soft", key_id=key_id),
        daily_tokens_soft=_coerce_positive_int(value.get("daily_tokens_soft"), field_name="daily_tokens_soft", key_id=key_id),
        action=action,
        warn_threshold=_coerce_warn_threshold(value.get("warn_threshold"), key_id=key_id),
    )


def _normalize_facade_key_config(key_id: str, raw_config: Any) -> FacadeKeyConfig:
    if raw_config in (None, ""):
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"Facade key '{key_id}' config must be a mapping")

    return FacadeKeyConfig(
        key_id=key_id,
        enabled=bool(raw_config.get("enabled", True)),
        display_name=(str(raw_config.get("display_name")).strip() if raw_config.get("display_name") not in (None, "") else None),
        tags=_normalize_tags(raw_config.get("tags"), key_id=key_id),
        default_class=(str(raw_config.get("default_class")).strip() if raw_config.get("default_class") not in (None, "") else None),
        quota=_normalize_quota_policy(raw_config.get("quota"), key_id=key_id),
    )


@dataclass(frozen=True)
class FacadeKeyRegistry:
    configs: Dict[str, FacadeKeyConfig]
    secrets: Dict[str, tuple[str, ...]]
    _secret_index: Dict[str, str] = field(init=False, repr=False, compare=False, default_factory=dict)

    def __post_init__(self):
        self._validate()
        secret_index: Dict[str, str] = {}
        for key_id, key_values in self.secrets.items():
            for secret in key_values:
                secret_index[secret] = key_id
        object.__setattr__(self, "_secret_index", secret_index)

    def _validate(self) -> None:
        unknown_secret_ids = sorted(set(self.secrets) - set(self.configs))
        if unknown_secret_ids:
            raise ValueError(
                f"Facade key secrets exist for unknown logical ids: {', '.join(unknown_secret_ids)}"
            )

        disabled_ids_with_secrets = sorted(
            key_id for key_id, config in self.configs.items() if not config.enabled and self.secrets.get(key_id)
        )
        if disabled_ids_with_secrets:
            raise ValueError(
                f"Disabled facade keys cannot retain live secrets: {', '.join(disabled_ids_with_secrets)}"
            )

        seen_secrets: Dict[str, str] = {}
        duplicate_claims = []
        for key_id, key_values in self.secrets.items():
            for secret in key_values:
                claimed_by = seen_secrets.setdefault(secret, key_id)
                if claimed_by != key_id:
                    duplicate_claims.append((secret, claimed_by, key_id))
        if duplicate_claims:
            duplicate_values = ", ".join(
                f"{first} and {second}" for _secret, first, second in duplicate_claims
            )
            raise ValueError(f"Facade key secrets must be unique across logical ids: {duplicate_values}")

    @classmethod
    def from_sources(
        cls,
        facade_key_configs: Optional[Mapping[str, Any]] = None,
        facade_key_secrets: Optional[Mapping[str, Iterable[str]]] = None,
    ) -> "FacadeKeyRegistry":
        normalized_configs: Dict[str, FacadeKeyConfig] = {}
        for key_id, raw_config in (facade_key_configs or {}).items():
            normalized_key_id = str(key_id).strip()
            if not normalized_key_id:
                raise ValueError("Facade key ids cannot be empty")
            normalized_configs[normalized_key_id] = _normalize_facade_key_config(normalized_key_id, raw_config)

        normalized_secrets: Dict[str, tuple[str, ...]] = {}
        for key_id, raw_secrets in (facade_key_secrets or {}).items():
            normalized_key_id = str(key_id).strip()
            if not normalized_key_id:
                raise ValueError("Facade key secret ids cannot be empty")

            cleaned = []
            for secret in raw_secrets:
                candidate = str(secret).strip()
                if candidate:
                    cleaned.append(candidate)
            if cleaned:
                normalized_secrets[normalized_key_id] = tuple(cleaned)

        return cls(configs=normalized_configs, secrets=normalized_secrets)

    def get_config(self, key_id: str) -> Optional[FacadeKeyConfig]:
        return self.configs.get(key_id)

    def get_secrets(self, key_id: str) -> tuple[str, ...]:
        return self.secrets.get(key_id, ())

    def resolve_secret(self, presented_secret: str) -> Optional[ResolvedCaller]:
        normalized_secret = str(presented_secret).strip()
        if not normalized_secret:
            return None

        key_id = self._secret_index.get(normalized_secret)
        if key_id is None:
            return None

        config = self.configs.get(key_id)
        if config is None or not config.enabled:
            return None

        return ResolvedCaller(
            authentication_principal=f"facade_key:{key_id}",
            identity=RequestIdentity(
                kind="facade_key",
                subject_id=key_id,
                display_name=config.display_name,
                tags=config.tags,
                default_class=config.default_class,
                quota_policy=config.quota.to_dict(),
                token_accounting_state="untracked",
            ),
        )

    def has_key(self, key_id: str) -> bool:
        return key_id in self.configs

    def key_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.configs))

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            key_id: {
                "enabled": config.enabled,
                "display_name": config.display_name,
                "tags": list(config.tags),
                "default_class": config.default_class,
                "quota": config.quota.to_dict(),
                "secret_count": len(self.secrets.get(key_id, ())),
            }
            for key_id, config in self.configs.items()
        }


def load_facade_key_registry(facade_key_configs: Optional[Mapping[str, Any]] = None) -> FacadeKeyRegistry:
    """Build a validated registry from config metadata and the shared secrets store."""
    return FacadeKeyRegistry.from_sources(
        facade_key_configs=facade_key_configs,
        facade_key_secrets=load_facade_key_secrets(),
    )
