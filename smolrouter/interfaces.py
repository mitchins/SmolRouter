"""
Core interfaces and abstractions for the SmolRouter architecture.

This module defines the contracts for model providers, strategies, and access control
following SOLID principles for clean, extensible architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import ipaddress
import socket
from typing import List, Dict, Any, Optional, Mapping, TYPE_CHECKING
from dataclasses import dataclass
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from .facade_keys import RequestIdentity


@dataclass
class ModelInfo:
    """Information about a model from a provider"""

    id: str  # Unique identifier (e.g., "llama3-70b@fast-kitten")
    name: str  # Display name (e.g., "llama3-70b")
    provider_id: str  # Provider identifier (e.g., "fast-kitten")
    provider_type: str  # Provider type (e.g., "ollama", "openai")
    endpoint: str  # Base URL of the provider
    aliases: Optional[List[str]] = None  # Alternative names for this model
    metadata: Dict[str, Any] = None  # Additional provider-specific metadata

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.metadata is None:
            self.metadata = {}

    @property
    def display_name(self) -> str:
        """Human-readable display name with provider context"""
        return f"{self.name} [{self.provider_id}]"

    def matches_request(self, requested_model: str) -> bool:
        """Check if this model matches a client request"""
        # Exact match on ID, name, or any alias
        if requested_model in [self.id, self.name] + self.aliases:
            return True

        # Match display name format
        if requested_model == self.display_name:
            return True

        return False


@dataclass
class ClientContext:
    """Context information about the requesting client"""

    ip: str
    auth_payload: Optional[Dict[str, Any]] = None
    user_agent: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    identity: Optional[RequestIdentity] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}

    @property
    def user_id(self) -> Optional[str]:
        """Extract user ID from auth payload if available"""
        if self.auth_payload:
            return self.auth_payload.get("sub") or self.auth_payload.get("user") or self.auth_payload.get("username")
        return None


@dataclass(frozen=True)
class SystemPromptConfig:
    """A provider/model-specific transformation for chat system messages."""

    mode: str
    content: str

    SUPPORTED_MODES = frozenset({"append", "prepend", "replace"})

    def __post_init__(self):
        if not isinstance(self.mode, str) or not self.mode.strip():
            raise ValueError("System prompt mode must be one of: append, prepend, replace")

        normalized_mode = self.mode.strip().lower()
        if normalized_mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported system prompt mode '{self.mode}'. "
                f"Expected one of: {', '.join(sorted(self.SUPPORTED_MODES))}"
            )
        object.__setattr__(self, "mode", normalized_mode)

        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("System prompt content must be a non-empty string")


def coerce_system_prompt_config(value: Any) -> Optional[SystemPromptConfig]:
    """Convert a YAML-shaped system prompt policy into its validated type."""
    if value is None or isinstance(value, SystemPromptConfig):
        return value
    if isinstance(value, Mapping):
        return SystemPromptConfig(**dict(value))
    raise TypeError("System prompt configuration must be a mapping or null")


def coerce_per_model_system_prompt(value: Any) -> Dict[str, Optional[SystemPromptConfig]]:
    """Convert and validate exact model-name system prompt overrides."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("per_model_system_prompt must be a mapping")

    converted = {}
    for model_name, prompt_config in value.items():
        if not isinstance(model_name, str) or not model_name.strip() or model_name != model_name.strip():
            raise ValueError("per_model_system_prompt keys must be non-empty model names without surrounding whitespace")
        converted[model_name] = coerce_system_prompt_config(prompt_config)
    return converted


class IModelProvider(ABC):
    """Abstraction for model discovery and health checking from providers"""

    @abstractmethod
    async def discover_models(self) -> List[ModelInfo]:
        """Discover available models from this provider"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy and reachable"""
        pass

    @abstractmethod
    def get_provider_id(self) -> str:
        """Return unique provider identifier (e.g., 'fast-kitten')"""
        pass

    @abstractmethod
    def get_provider_type(self) -> str:
        """Return provider type (e.g., 'ollama', 'openai')"""
        pass

    @abstractmethod
    def get_endpoint(self) -> str:
        """Return base endpoint URL"""
        pass


class IModelStrategy(ABC):
    """Handles model aliasing, transformation, and resolution rules"""

    @abstractmethod
    async def resolve_model_request(
        self, requested_model: str, available_models: List[ModelInfo]
    ) -> Optional[ModelInfo]:
        """
        Resolve a client's model request to an actual model.

        Args:
            requested_model: The model name requested by the client
            available_models: List of currently available models

        Returns:
            ModelInfo if resolution successful, None if not found
        """
        pass

    @abstractmethod
    async def apply_aliases(self, models: List[ModelInfo]) -> List[ModelInfo]:
        """Apply alias transformations to model list"""
        pass

    @abstractmethod
    async def get_model_priority_order(self, model_name: str) -> List[str]:
        """
        Get priority order of providers for a given model name.
        Used when multiple providers offer the same model.

        Returns:
            List of provider_ids in priority order
        """
        pass


class IAccessControl(ABC):
    """Controls what models clients can see and access"""

    @abstractmethod
    async def filter_models(self, models: List[ModelInfo], client: ClientContext) -> List[ModelInfo]:
        """Filter models based on client permissions"""
        pass

    @abstractmethod
    async def can_access_model(self, model: ModelInfo, client: ClientContext) -> bool:
        """Check if client can access specific model"""
        pass


class IModelCache(ABC):
    """Abstraction for model caching with TTL support"""

    @abstractmethod
    async def get_cached_models(self, provider_id: str) -> Optional[List[ModelInfo]]:
        """Get cached models for a provider"""
        pass

    @abstractmethod
    async def cache_models(self, provider_id: str, models: List[ModelInfo], ttl_seconds: int = 300):
        """Cache models for a provider with TTL"""
        pass

    @abstractmethod
    async def invalidate_cache(self, provider_id: str = None):
        """Invalidate cache for specific provider or all providers"""
        pass

    @abstractmethod
    async def is_cache_valid(self, provider_id: str) -> bool:
        """Check if cached data is still valid"""
        pass


_LAN_PROXY_HOSTNAMES = {"localhost", "localhost.localdomain"}
_LAN_PROXY_HOST_SUFFIXES = (".localhost", ".local", ".lan", ".internal", ".intranet", ".home.arpa")


def _is_lan_proxy_hostname(hostname: str) -> bool:
    return hostname in _LAN_PROXY_HOSTNAMES or hostname.endswith(_LAN_PROXY_HOST_SUFFIXES)


def _classify_proxy_ip_address(hostname: str) -> Optional[bool]:
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return None

    if address.is_private or address.is_loopback or address.is_link_local:
        return True
    if address.is_global:
        return False
    return None


def _classify_resolved_proxy_hosts(hostname: str) -> Optional[bool]:
    try:
        resolved_addresses = {
            record[4][0].split("%", 1)[0]
            for record in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
            if record and record[4]
        }
    except socket.gaierror:
        return None

    if not resolved_addresses:
        return None

    resolved_classifications = {_classify_proxy_ip_address(resolved_address) for resolved_address in resolved_addresses}
    if False in resolved_classifications:
        return False
    if resolved_classifications == {True}:
        return True
    return None


def _classify_proxy_host(hostname: str) -> Optional[bool]:
    normalized_hostname = hostname.strip().strip("[]").split("%", 1)[0].casefold()
    if not normalized_hostname:
        return None

    if _is_lan_proxy_hostname(normalized_hostname):
        return True

    ip_classification = _classify_proxy_ip_address(normalized_hostname)
    if ip_classification is not None:
        return ip_classification

    return _classify_resolved_proxy_hosts(normalized_hostname)


def _validate_http_proxy_url(proxy_url: Optional[str]) -> None:
    if not proxy_url:
        return

    parsed_url = urlsplit(proxy_url)
    if parsed_url.scheme != "http":
        return

    if not parsed_url.hostname:
        raise ValueError(f"HTTP proxy URL {proxy_url!r} must include a hostname")

    if _classify_proxy_host(parsed_url.hostname) is False:
        raise ValueError(
            f"HTTP proxy URL {proxy_url!r} points to a public address; use HTTPS or a LAN/private proxy instead"
        )


@dataclass
class ProxyConfig:
    """Configuration for HTTP proxy settings"""

    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    def __post_init__(self):
        _validate_http_proxy_url(self.http_proxy)
        _validate_http_proxy_url(self.https_proxy)

    def _apply_proxy_auth(self, proxy_url: str) -> str:
        if self.username and self.password and "://" in proxy_url:
            scheme, rest = proxy_url.split("://", 1)
            return f"{scheme}://{self.username}:{self.password}@{rest}"
        return proxy_url

    def to_httpx_proxy(self) -> Optional[str]:
        """Convert to httpx proxy format (single URL)"""
        # httpx expects a single proxy URL, prioritize HTTPS proxy
        if self.https_proxy:
            return self._apply_proxy_auth(self.https_proxy)
        if self.http_proxy:
            return self._apply_proxy_auth(self.http_proxy)
        return None

    def to_httpx_proxies(self) -> Dict[str, str]:
        """Convert to httpx proxy format (for backward compatibility)"""
        proxies = {}
        if self.http_proxy:
            # `http://` is the httpx scheme key only; clear-text proxy endpoints are rejected in __post_init__.
            proxies["http://"] = self._apply_proxy_auth(self.http_proxy)

        if self.https_proxy:
            proxies["https://"] = self._apply_proxy_auth(self.https_proxy)

        return proxies


def coerce_proxy_config(proxy_config: Any) -> Any:
    if isinstance(proxy_config, dict):
        return ProxyConfig(**proxy_config)
    return proxy_config


def coerce_provider_proxy_settings(provider_config: Mapping[str, Any]) -> Dict[str, Any]:
    config = dict(provider_config)

    if "proxy_config" in config:
        config["proxy_config"] = coerce_proxy_config(config["proxy_config"])

    per_model_proxy = config.get("per_model_proxy")
    if isinstance(per_model_proxy, dict):
        config["per_model_proxy"] = {
            model_name: coerce_proxy_config(proxy_value)
            for model_name, proxy_value in per_model_proxy.items()
        }

    proxy_pool = config.get("proxy_pool")
    if isinstance(proxy_pool, list):
        config["proxy_pool"] = [None if entry is None else coerce_proxy_config(entry) for entry in proxy_pool]

    if "system_prompt" in config:
        config["system_prompt"] = coerce_system_prompt_config(config["system_prompt"])

    per_model_system_prompt = config.get("per_model_system_prompt")
    if per_model_system_prompt is not None:
        config["per_model_system_prompt"] = coerce_per_model_system_prompt(per_model_system_prompt)

    return config


@dataclass
class ProviderConfig:
    """Configuration for a model provider"""

    name: str  # Human-readable name (becomes provider_id)
    type: str  # Provider type ('ollama', 'openai')
    url: str  # Base endpoint URL
    api_key: Optional[str] = None
    timeout: float = 10.0
    enabled: bool = True
    priority: int = 0  # Lower numbers have higher priority
    metadata: Dict[str, Any] = None
    static_models: Optional[List[str]] = None
    proxy_config: Optional[ProxyConfig] = None  # Default proxy for all models
    per_model_proxy: Optional[Dict[str, ProxyConfig]] = None  # Model-specific proxy overrides
    system_prompt: Optional[SystemPromptConfig] = None  # Default system prompt policy
    per_model_system_prompt: Optional[Dict[str, Optional[SystemPromptConfig]]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.static_models is None:
            self.static_models = []
        if self.per_model_proxy is None:
            self.per_model_proxy = {}
        self.system_prompt = coerce_system_prompt_config(self.system_prompt)
        self.per_model_system_prompt = coerce_per_model_system_prompt(self.per_model_system_prompt)

    def get_proxy_for_model(self, model_name: str) -> Optional[ProxyConfig]:
        """Get proxy configuration for a specific model"""
        # Check for model-specific proxy first
        per_model_proxy = self.per_model_proxy or {}
        if model_name in per_model_proxy:
            return per_model_proxy[model_name]
        # Fall back to default proxy
        return self.proxy_config

    def get_system_prompt_for_model(self, model_name: str) -> Optional[SystemPromptConfig]:
        """Return the exact model override, falling back to the provider default."""
        per_model_system_prompt = self.per_model_system_prompt or {}
        if model_name in per_model_system_prompt:
            return per_model_system_prompt[model_name]
        return self.system_prompt


@dataclass
class ModelResolution:
    """Result of model resolution process"""

    model: Optional[ModelInfo]
    resolved_from: str  # Original request
    fallback_used: bool = False
    resolution_path: Optional[List[str]] = None  # Steps taken during resolution

    def __post_init__(self):
        if self.resolution_path is None:
            self.resolution_path = []
