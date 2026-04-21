"""
Core interfaces and abstractions for the SmolRouter architecture.

This module defines the contracts for model providers, strategies, and access control
following SOLID principles for clean, extensible architecture.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a model from a provider"""

    id: str  # Unique identifier (e.g., "llama3-70b@fast-kitten")
    name: str  # Display name (e.g., "llama3-70b")
    provider_id: str  # Provider identifier (e.g., "fast-kitten")
    provider_type: str  # Provider type (e.g., "ollama", "openai")
    endpoint: str  # Base URL of the provider
    aliases: List[str] = None  # Alternative names for this model
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

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}

    @property
    def user_id(self) -> Optional[str]:
        """Extract user ID from auth payload if available"""
        if self.auth_payload:
            return self.auth_payload.get("sub") or self.auth_payload.get("user") or self.auth_payload.get("username")
        return None


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


@dataclass
class ProxyConfig:
    """Configuration for HTTP proxy settings"""

    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    def to_httpx_proxy(self) -> Optional[str]:
        """Convert to httpx proxy format (single URL)"""
        # httpx expects a single proxy URL, prioritize HTTPS proxy
        if self.https_proxy:
            proxy_url = self.https_proxy
            if self.username and self.password:
                # Insert auth into URL
                if "://" in proxy_url:
                    scheme, rest = proxy_url.split("://", 1)
                    proxy_url = f"{scheme}://{self.username}:{self.password}@{rest}"
            return proxy_url
        elif self.http_proxy:
            proxy_url = self.http_proxy
            if self.username and self.password:
                # Insert auth into URL
                if "://" in proxy_url:
                    scheme, rest = proxy_url.split("://", 1)
                    proxy_url = f"{scheme}://{self.username}:{self.password}@{rest}"
            return proxy_url
        return None

    def to_httpx_proxies(self) -> Dict[str, str]:
        """Convert to httpx proxy format (for backward compatibility)"""
        proxies = {}
        if self.http_proxy:
            proxy_url = self.http_proxy
            if self.username and self.password:
                # Insert auth into URL
                if "://" in proxy_url:
                    scheme, rest = proxy_url.split("://", 1)
                    proxy_url = f"{scheme}://{self.username}:{self.password}@{rest}"
            proxies["http://"] = proxy_url

        if self.https_proxy:
            proxy_url = self.https_proxy
            if self.username and self.password:
                # Insert auth into URL
                if "://" in proxy_url:
                    scheme, rest = proxy_url.split("://", 1)
                    proxy_url = f"{scheme}://{self.username}:{self.password}@{rest}"
            proxies["https://"] = proxy_url

        return proxies


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
    per_model_proxy: Dict[str, ProxyConfig] = None  # Model-specific proxy overrides

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.static_models is None:
            self.static_models = []
        if self.per_model_proxy is None:
            self.per_model_proxy = {}

    def get_proxy_for_model(self, model_name: str) -> Optional[ProxyConfig]:
        """Get proxy configuration for a specific model"""
        # Check for model-specific proxy first
        if model_name in self.per_model_proxy:
            return self.per_model_proxy[model_name]
        # Fall back to default proxy
        return self.proxy_config


@dataclass
class ModelResolution:
    """Result of model resolution process"""

    model: Optional[ModelInfo]
    resolved_from: str  # Original request
    fallback_used: bool = False
    resolution_path: List[str] = None  # Steps taken during resolution

    def __post_init__(self):
        if self.resolution_path is None:
            self.resolution_path = []
