"""
Google Generative AI provider implementation.

Provides integration with Google's Generative AI API, supporting multiple API keys
with intelligent rotation based on requests-per-day (RPD) quotas.
"""

import logging
import json
import asyncio
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
import httpx

from google import genai
from google.api_core.exceptions import ResourceExhausted, PermissionDenied, InvalidArgument

from .config_loading import load_config_entries
from .interfaces import IModelProvider, ModelInfo, ProviderConfig, ProxyConfig
from .database import ApiKeyQuota
from .redis_backend import QuotaRecord
from .rate_limiter import GoogleGenAIRequestFunnel
from .request_metadata import RequestMetadata

logger = logging.getLogger(__name__)

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


def _current_pacific_date() -> str:
    return datetime.now(PACIFIC_TZ).strftime("%Y-%m-%d")


def _to_pacific_datetime(value: datetime, assume_utc: bool = False) -> datetime:
    if value.tzinfo is None:
        if assume_utc:
            return value.replace(tzinfo=timezone.utc).astimezone(PACIFIC_TZ)
        return value.replace(tzinfo=PACIFIC_TZ)

    return value.astimezone(PACIFIC_TZ)


def _format_optional_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _quota_status(is_invalid: bool, is_exhausted: bool) -> str:
    if is_invalid:
        return "invalid"
    if is_exhausted:
        return "exhausted"
    return "available"


@dataclass
class ApiKeyModelStats:
    """Statistics for a single API key + model combination"""

    api_key: str
    model: str
    requests_today: int = 0
    tokens_today: int = 0
    last_request: Optional[datetime] = None
    last_error: Optional[str] = None
    error_count: int = 0
    quota_exhausted_at: Optional[datetime] = None
    invalid_key: bool = False  # Permanently mark invalid/expired keys

    @property
    def key_model_id(self) -> str:
        """Unique identifier for this key+model combination"""
        return f"{self.api_key[:8]}.../{self.model}"

    def is_day_reset_needed(self) -> bool:
        """Check if we need to reset daily counters (Pacific timezone reset)"""
        now_pacific = datetime.now(PACIFIC_TZ)
        now_date = now_pacific.date()

        # Check against last request time
        if self.last_request:
            last_request_pacific = _to_pacific_datetime(self.last_request)
            last_request_date = last_request_pacific.date()
            if now_date > last_request_date:
                return True

        # Also check against quota exhaustion time (important for 429 recovery)
        if self.quota_exhausted_at:
            exhausted_pacific = _to_pacific_datetime(self.quota_exhausted_at)
            exhausted_date = exhausted_pacific.date()
            if now_date > exhausted_date:
                return True

        return False

    def reset_daily_stats(self):
        """Reset daily counters"""
        self.requests_today = 0
        self.tokens_today = 0
        self.error_count = 0
        self.quota_exhausted_at = None  # Clear exhaustion marker
        logger.info(f"Reset daily stats for API key {self.api_key[:8]}...")


@dataclass
class ProxyHealthStatus:
    """Runtime health state for a configured proxy endpoint."""

    url: str
    status: str = "unknown"
    last_checked_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None
    last_error: Optional[str] = None
    failure_count: int = 0
    success_count: int = 0
    unhealthy_until: Optional[datetime] = None


class GoogleGenAIRequestError(RuntimeError):
    """Request-scoped Google GenAI error with provider metadata attached."""

    def __init__(
        self,
        message: str,
        *,
        provider_id: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key_suffix: Optional[str] = None,
        api_key_index: Optional[int] = None,
        api_key_total: Optional[int] = None,
        proxy_used: Optional[str] = None,
    ):
        super().__init__(message)
        self.provider_id = provider_id
        self.model_name = model_name
        self.api_key_suffix = api_key_suffix
        self.api_key_index = api_key_index
        self.api_key_total = api_key_total
        self.proxy_used = proxy_used
        self.retry_after_seconds: Optional[float] = None


@dataclass
class GoogleGenAIConfig(ProviderConfig):
    """Extended configuration for Google GenAI provider"""

    api_keys: List[str] = field(default_factory=list)
    api_keys_file: Optional[str] = None  # Path to file containing API keys
    max_requests_per_day: int = 1500  # Free tier limit
    max_tokens_per_minute: int = 32000  # Free tier limit

    # Rate limiting configuration (defaults to disabled)
    rate_limiting_enabled: bool = False
    max_concurrent_requests: int = 3
    max_requests_per_window: int = 12
    window_minutes: int = 4

    # Proxy pool for round-robin rotation across all requests
    # When enabled, models with proxy_config="auto" (or no config) will round-robin through this pool
    # Pool can include None for "direct" (no proxy) connections
    proxy_pool: Optional[List[Optional["ProxyConfig"]]] = None
    proxy_pool_enabled: bool = False  # Master switch for proxy pooling

    # Predictive 429 configuration (defaults to disabled)
    predictive_429_enabled: bool = False

    def __init__(self, **kwargs):
        # Extract Google-specific fields before calling parent
        self.api_keys = list(kwargs.pop("api_keys", []) or [])
        self.api_keys_file = kwargs.pop("api_keys_file", None)
        self.max_requests_per_day = kwargs.pop("max_requests_per_day", 1500)
        self.max_tokens_per_minute = kwargs.pop("max_tokens_per_minute", 32000)

        # Rate limiting configuration (defaults to disabled)
        self.rate_limiting_enabled = kwargs.pop("rate_limiting_enabled", False)
        self.max_concurrent_requests = kwargs.pop("max_concurrent_requests", 3)
        self.max_requests_per_window = kwargs.pop("max_requests_per_window", 12)
        self.window_minutes = kwargs.pop("window_minutes", 4)

        # Predictive 429 configuration (defaults to disabled)
        self.predictive_429_enabled = kwargs.pop("predictive_429_enabled", False)

        # Proxy pool configuration
        self.proxy_pool = kwargs.pop("proxy_pool", None)
        self.proxy_pool_enabled = kwargs.pop("proxy_pool_enabled", False)

        # Set required fields for base class
        if "url" not in kwargs:
            kwargs["url"] = "https://generativelanguage.googleapis.com"

        # Call parent constructor
        super().__init__(**kwargs)
        self._post_init_google()

    def _post_init_google(self):
        if not self.api_keys and not self.api_keys_file:
            raise ValueError("Either api_keys or api_keys_file must be provided")

        # Load API keys from file if specified
        if self.api_keys_file:
            try:
                file_keys = load_config_entries(self.api_keys_file)
                self.api_keys.extend(file_keys)
                logger.info(f"Loaded {len(file_keys)} API keys from {self.api_keys_file}")
            except Exception as e:
                logger.error(f"Failed to load API keys from {self.api_keys_file}: {e}")
                raise

        if not self.api_keys:
            raise ValueError("No valid API keys found")


@dataclass
class GoogleGenAICompletionContext:
    original_model: str
    observation_id: str
    model_name: str = ""
    genai_request: Dict[str, Any] = field(default_factory=dict)
    api_key: str = ""
    api_key_suffix: Optional[str] = None
    api_key_index: Optional[int] = None
    api_key_total: Optional[int] = None
    proxy_config: Optional[ProxyConfig] = None
    proxy_info: Optional[ProxyConfig] = None
    proxy_url: Optional[str] = None
    proxy_pool_index: Optional[int] = None
    pool_info: str = ""
    sync_transport: Any = None
    async_transport: Any = None
    client: Any = None


class GoogleGenAIProvider(IModelProvider):
    """Provider for Google Generative AI models with intelligent API key rotation"""

    # No mappings - let Google handle their own model aliasing
    MODEL_MAPPINGS = {}
    PROXY_HEALTH_CHECK_INTERVAL_SECONDS = 30
    PROXY_FAILURE_COOLDOWN_SECONDS = 45
    PROXY_PROBE_TIMEOUT_SECONDS = 2.0

    def __init__(self, config: GoogleGenAIConfig):
        if not isinstance(config, GoogleGenAIConfig):
            # Convert regular ProviderConfig to GoogleGenAIConfig
            if hasattr(config, "api_keys"):
                config = GoogleGenAIConfig(**config.__dict__)
            else:
                raise ValueError("GoogleGenAIProvider requires GoogleGenAIConfig")

        self.config = config
        self.config.type = "google-genai"

        # Database persistence - no more in-memory stats storage

        # Cache for discovered models
        self._cached_models: Optional[List[ModelInfo]] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=15)  # Cache for 15 minutes

        # Create provider-specific rate limiter instance
        self._rate_limiter = GoogleGenAIRequestFunnel(
            max_concurrent=self.config.max_concurrent_requests,
            max_requests_per_window=self.config.max_requests_per_window,
            window_minutes=self.config.window_minutes,
            enabled=self.config.rate_limiting_enabled,
        )

        # Proxy pool round-robin counter (thread-safe via atomicity of +=)
        self._proxy_pool_index = 0
        self._proxy_health: Dict[str, ProxyHealthStatus] = {}
        self._proxy_health_task: Optional[asyncio.Task] = None
        self._proxy_probe_tasks: Dict[str, asyncio.Task] = {}
        self._ensure_proxy_health_entries()

        logger.info(f"Initialized GoogleGenAI provider with {len(self.config.api_keys)} API keys")
        logger.info(f"Rate limiting: {'enabled' if self.config.rate_limiting_enabled else 'disabled'}")
        logger.info(f"Predictive 429: {'enabled' if self.config.predictive_429_enabled else 'disabled'}")
        if self.config.proxy_pool_enabled and self.config.proxy_pool:
            pool_size = len(self.config.proxy_pool)
            direct_count = sum(1 for p in self.config.proxy_pool if p is None)
            proxy_count = pool_size - direct_count
            logger.info(f"Proxy pool: enabled with {pool_size} entries ({proxy_count} proxies, {direct_count} direct)")
        else:
            logger.info("Proxy pool: disabled")

    def _utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _proxy_config_to_url(self, proxy_config: Optional[ProxyConfig]) -> Optional[str]:
        if proxy_config is None:
            return None
        if hasattr(proxy_config, "to_httpx_proxy"):
            return proxy_config.to_httpx_proxy()
        return str(proxy_config)

    def _configured_proxy_urls(self) -> List[str]:
        proxy_urls: List[str] = []
        seen_urls = set()

        def add_url(proxy_config: Optional[ProxyConfig]):
            proxy_url = self._proxy_config_to_url(proxy_config)
            if proxy_url and proxy_url not in seen_urls:
                seen_urls.add(proxy_url)
                proxy_urls.append(proxy_url)

        add_url(self.config.proxy_config)

        for proxy_config in (self.config.per_model_proxy or {}).values():
            add_url(proxy_config)

        for proxy_config in self.config.proxy_pool or []:
            add_url(proxy_config)

        return proxy_urls

    def _ensure_proxy_health_entries(self):
        for proxy_url in self._configured_proxy_urls():
            self._proxy_health.setdefault(proxy_url, ProxyHealthStatus(url=proxy_url))

    def _mask_proxy_url(self, proxy_url: Optional[str]) -> Optional[str]:
        if not proxy_url:
            return None

        parsed = urlparse(proxy_url)
        if not parsed.scheme or not parsed.hostname:
            return proxy_url

        netloc = parsed.hostname
        if parsed.port:
            netloc += f":{parsed.port}"
        return parsed._replace(netloc=netloc).geturl()

    def _proxy_probe_due(self, proxy_url: str) -> bool:
        status = self._proxy_health.get(proxy_url)
        if status is None or status.last_checked_at is None:
            return True

        age_seconds = (self._utc_now() - status.last_checked_at).total_seconds()
        return age_seconds >= self.PROXY_HEALTH_CHECK_INTERVAL_SECONDS

    def _proxy_available_for_use(self, proxy_url: str) -> bool:
        status = self._proxy_health.get(proxy_url)
        if status is None or status.status != "unhealthy":
            return True

        if status.unhealthy_until is None:
            return False

        return status.unhealthy_until <= self._utc_now()

    def _mark_proxy_health(self, proxy_url: str, success: bool, error: Optional[str] = None):
        self._ensure_proxy_health_entries()
        health = self._proxy_health.setdefault(proxy_url, ProxyHealthStatus(url=proxy_url))
        now = self._utc_now()
        health.last_checked_at = now

        if success:
            health.status = "healthy"
            health.last_success_at = now
            health.success_count += 1
            health.last_error = None
            health.unhealthy_until = None
            return

        health.status = "unhealthy"
        health.last_failure_at = now
        health.failure_count += 1
        health.last_error = error
        health.unhealthy_until = now + timedelta(seconds=self.PROXY_FAILURE_COOLDOWN_SECONDS)

    def _proxy_health_snapshot(self, proxy_url: Optional[str]) -> Dict[str, Any]:
        if not proxy_url:
            return {
                "status": "direct",
                "last_checked_at": None,
                "last_success_at": None,
                "last_failure_at": None,
                "last_error": None,
                "failure_count": 0,
                "success_count": 0,
                "cooldown_remaining_seconds": 0,
            }

        self._ensure_proxy_health_entries()
        health = self._proxy_health.get(proxy_url)
        if health is None:
            return {
                "status": "unknown",
                "last_checked_at": None,
                "last_success_at": None,
                "last_failure_at": None,
                "last_error": None,
                "failure_count": 0,
                "success_count": 0,
                "cooldown_remaining_seconds": 0,
            }

        cooldown_remaining_seconds = 0
        if health.unhealthy_until and health.unhealthy_until > self._utc_now():
            cooldown_remaining_seconds = max(int((health.unhealthy_until - self._utc_now()).total_seconds()), 0)

        return {
            "status": health.status,
            "last_checked_at": health.last_checked_at.isoformat() if health.last_checked_at else None,
            "last_success_at": health.last_success_at.isoformat() if health.last_success_at else None,
            "last_failure_at": health.last_failure_at.isoformat() if health.last_failure_at else None,
            "last_error": health.last_error,
            "failure_count": health.failure_count,
            "success_count": health.success_count,
            "cooldown_remaining_seconds": cooldown_remaining_seconds,
        }

    async def _probe_proxy_url(self, proxy_url: str):
        parsed = urlparse(proxy_url)
        host = parsed.hostname
        if not host:
            self._mark_proxy_health(proxy_url, success=False, error="Invalid proxy URL")
            return

        if parsed.port:
            port = parsed.port
        elif parsed.scheme in {"https", "socks5", "socks5h", "socks5s"}:
            port = 443
        else:
            port = 80

        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.PROXY_PROBE_TIMEOUT_SECONDS,
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            self._mark_proxy_health(proxy_url, success=True)
        except Exception as exc:
            self._mark_proxy_health(proxy_url, success=False, error=f"{type(exc).__name__}: {exc}")

    def _schedule_proxy_probe(self, proxy_url: Optional[str], force: bool = False):
        if not proxy_url:
            return

        self._ensure_proxy_health_entries()

        if not force and not self._proxy_probe_due(proxy_url):
            return

        existing_task = self._proxy_probe_tasks.get(proxy_url)
        if existing_task and not existing_task.done():
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        task = loop.create_task(self._probe_proxy_url(proxy_url))
        self._proxy_probe_tasks[proxy_url] = task

        def _cleanup_probe(done_task: asyncio.Task, url: str = proxy_url):
            if self._proxy_probe_tasks.get(url) is done_task:
                self._proxy_probe_tasks.pop(url, None)
            if done_task.cancelled():
                return
            try:
                done_task.result()
            except Exception as exc:
                logger.debug(f"Proxy health probe failed for {self._mask_proxy_url(url)}: {exc}")

        task.add_done_callback(_cleanup_probe)

    async def refresh_proxy_health(self, force: bool = False):
        self._ensure_proxy_health_entries()
        probes = []
        for proxy_url in self._configured_proxy_urls():
            if force or self._proxy_probe_due(proxy_url):
                probes.append(self._probe_proxy_url(proxy_url))

        if probes:
            await asyncio.gather(*probes, return_exceptions=True)

    async def _proxy_health_monitor_loop(self):
        try:
            while True:
                await self.refresh_proxy_health(force=False)
                await asyncio.sleep(self.PROXY_HEALTH_CHECK_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Proxy health monitor stopped unexpectedly for {self.config.name}: {exc}")

    def start_proxy_health_monitor(self):
        if not self._configured_proxy_urls():
            return

        if self._proxy_health_task and not self._proxy_health_task.done():
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        self._proxy_health_task = loop.create_task(self._proxy_health_monitor_loop())
        logger.info(f"Started proxy health monitor for provider {self.config.name}")

    async def stop_proxy_health_monitor(self):
        tasks = []
        if self._proxy_health_task is not None:
            tasks.append(self._proxy_health_task)
            self._proxy_health_task = None

        if self._proxy_probe_tasks:
            tasks.extend(self._proxy_probe_tasks.values())
            self._proxy_probe_tasks = {}

        for task in tasks:
            if task and not task.done():
                task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _build_proxy_entry(
        self,
        label: str,
        proxy_config: Optional[ProxyConfig],
        *,
        kind: str,
        model_name: Optional[str] = None,
        pool_index: Optional[int] = None,
        selected_next: bool = False,
    ) -> Dict[str, Any]:
        proxy_url = self._proxy_config_to_url(proxy_config)
        health = self._proxy_health_snapshot(proxy_url)
        return {
            "label": label,
            "kind": kind,
            "model_name": model_name,
            "pool_index": pool_index,
            "selected_next": selected_next,
            "url": self._mask_proxy_url(proxy_url) if proxy_url else "direct",
            **health,
        }

    def get_proxy_diagnostics(self) -> Dict[str, Any]:
        self._ensure_proxy_health_entries()

        default_proxy = None
        if self.config.proxy_config is not None:
            default_proxy = self._build_proxy_entry("Default", self.config.proxy_config, kind="default")

        model_overrides = []
        for model_name, proxy_config in sorted((self.config.per_model_proxy or {}).items()):
            if proxy_config is None:
                continue
            model_overrides.append(
                self._build_proxy_entry(model_name, proxy_config, kind="override", model_name=model_name)
            )

        pool_entries = []
        for idx, proxy_config in enumerate(self.config.proxy_pool or []):
            if proxy_config is None:
                pool_entries.append(
                    {
                        "label": f"Pool #{idx + 1}",
                        "kind": "direct",
                        "model_name": None,
                        "pool_index": idx + 1,
                        "selected_next": idx == self._proxy_pool_index,
                        "url": "direct",
                        "status": "direct",
                        "last_checked_at": None,
                        "last_success_at": None,
                        "last_failure_at": None,
                        "last_error": None,
                        "failure_count": 0,
                        "success_count": 0,
                        "cooldown_remaining_seconds": 0,
                    }
                )
                continue

            pool_entries.append(
                self._build_proxy_entry(
                    f"Pool #{idx + 1}",
                    proxy_config,
                    kind="pool",
                    pool_index=idx + 1,
                    selected_next=idx == self._proxy_pool_index,
                )
            )

        all_entries = [entry for entry in [default_proxy] if entry] + model_overrides + pool_entries
        status_counts = {
            "healthy": sum(1 for entry in all_entries if entry["status"] == "healthy"),
            "unhealthy": sum(1 for entry in all_entries if entry["status"] == "unhealthy"),
            "unknown": sum(1 for entry in all_entries if entry["status"] == "unknown"),
            "direct": sum(1 for entry in all_entries if entry["status"] == "direct"),
        }

        configured = bool(default_proxy or model_overrides or (self.config.proxy_pool_enabled and self.config.proxy_pool))
        return {
            "provider_id": self.get_provider_id(),
            "provider_type": self.get_provider_type(),
            "configured": configured,
            "pool_enabled": bool(self.config.proxy_pool_enabled and self.config.proxy_pool),
            "monitor_running": bool(self._proxy_health_task and not self._proxy_health_task.done()),
            "health_check_interval_seconds": self.PROXY_HEALTH_CHECK_INTERVAL_SECONDS,
            "failure_cooldown_seconds": self.PROXY_FAILURE_COOLDOWN_SECONDS,
            "next_pool_index": self._proxy_pool_index + 1 if self.config.proxy_pool else None,
            "default_proxy": default_proxy,
            "model_overrides": model_overrides,
            "pool_entries": pool_entries,
            "summary": {
                "proxy_count": len(self._configured_proxy_urls()),
                "entry_count": len(all_entries),
                "pool_entry_count": len(pool_entries),
                "direct_entry_count": status_counts["direct"],
                "healthy_count": status_counts["healthy"],
                "unhealthy_count": status_counts["unhealthy"],
                "unknown_count": status_counts["unknown"],
            },
        }

    def _is_proxy_connectivity_error(self, error: Exception) -> bool:
        if isinstance(error, (httpx.ProxyError, httpx.ConnectError, httpx.ConnectTimeout)):
            return True

        error_text = str(error).lower()
        proxy_error_markers = [
            "connection refused",
            "proxyerror",
            "proxy error",
            "all connection attempts failed",
            "nodename nor servname",
            "name or service not known",
            "temporary failure in name resolution",
            "connect timeout",
            "timed out",
            "network is unreachable",
        ]
        return any(marker in error_text for marker in proxy_error_markers)

    def _get_next_proxy_from_pool(self) -> tuple[Optional["ProxyConfig"], Optional[int]]:
        """Get next proxy from pool using round-robin selection.

        Returns None for "direct" entries (no proxy).
        Thread-safe via atomic increment.
        """
        if not self.config.proxy_pool_enabled or not self.config.proxy_pool:
            return None, None

        pool = self.config.proxy_pool
        pool_size = len(pool)
        start_idx = self._proxy_pool_index
        blocked_entries = []

        for offset in range(pool_size):
            idx = (start_idx + offset) % pool_size
            selected = pool[idx]
            self._proxy_pool_index = (idx + 1) % pool_size

            if selected is None:
                logger.debug(f"Proxy pool: selected DIRECT (index {idx + 1}/{pool_size})")
                return None, idx

            proxy_url = self._proxy_config_to_url(selected)
            if proxy_url and not self._proxy_available_for_use(proxy_url):
                blocked_entries.append(self._mask_proxy_url(proxy_url) or proxy_url)
                self._schedule_proxy_probe(proxy_url)
                continue

            self._schedule_proxy_probe(proxy_url)
            logger.debug(f"Proxy pool: selected {self._mask_proxy_url(proxy_url)} (index {idx + 1}/{pool_size})")
            return selected, idx

        blocked_label = ", ".join(blocked_entries) if blocked_entries else "all configured entries"
        raise RuntimeError(f"No healthy proxies available in pool for provider {self.config.name}: {blocked_label}")

    def _should_use_proxy_pool(self, model_name: str) -> bool:
        """Check if this model should use the proxy pool.

        Returns True if:
        - Proxy pool is enabled
        - Model has no explicit proxy override (or override is "auto")
        """
        if not self.config.proxy_pool_enabled or not self.config.proxy_pool:
            return False

        # Check if model has an explicit proxy override
        per_model_proxy = self.config.per_model_proxy or {}
        if model_name in per_model_proxy:
            model_proxy = per_model_proxy[model_name]
            # Special "auto" marker means use pool (proxy is None with a special flag)
            if model_proxy is None:
                return True  # Explicit "use pool"
            # Has explicit proxy config - don't use pool
            return False

        # No explicit override - use pool by default when enabled
        return True

    def get_model_daily_limit(self, model_name: str) -> int:
        """Get the daily request limit for a specific model based on Google's current quotas

        These are per-project (per API key) free tier limits as of December 2025.
        With multiple API keys, total capacity = limit * number_of_keys
        """
        model_lower = model_name.lower()

        limit_rules = [
            ("gemma" in model_lower and "3" in model_lower, 14400),
            ("gemini" in model_lower and ("3.0" in model_lower or "gemini-3" in model_lower), 20),
            ("2.5" in model_lower and "flash" in model_lower and "lite" in model_lower, 1000),
            ("2.5" in model_lower and "flash" in model_lower, 20),
            (
                "2.0" in model_lower and "flash" in model_lower and ("exp" in model_lower or "experimental" in model_lower),
                5,
            ),
            ("2.0" in model_lower and "flash" in model_lower, 20),
            ("pro" in model_lower and ("2.5" in model_lower or "2.0" in model_lower), 20),
            ("1.5" in model_lower and "pro" in model_lower, 50),
            ("1.5" in model_lower and "flash" in model_lower, 1000),
            (any(keyword in model_lower for keyword in ["preview", "experimental"]), 5),
        ]

        for matches, limit in limit_rules:
            if matches:
                return limit

        # Default to conservative limit for unknown models
        return self.config.max_requests_per_day

    def _create_proxy_transport(self, proxy_config: Optional[ProxyConfig] = None, observation_id: Optional[str] = None):
        """Create httpx Transport pair configured with a proxy and optional observation.

        We explicitly set transports because google-genai may fall back to aiohttp
        internally, which ignores simple 'proxies' mappings. Attaching transports
        forces httpx for both sync and async paths.

        If observation_id is provided, wraps transports with observers for ground truth tracking.
        """
        from .transport_observer import ObservingHTTPTransport, ObservingAsyncHTTPTransport

        config_to_use = proxy_config or self.config.proxy_config

        if config_to_use and config_to_use.to_httpx_proxy():
            proxy_url = config_to_use.to_httpx_proxy()
            logger.info(f"🔀 Using proxy URL for Google GenAI: {proxy_url}")

            # Create base transports with proxy
            base_sync_transport = httpx.HTTPTransport(proxy=proxy_url)
            base_async_transport = httpx.AsyncHTTPTransport(proxy=proxy_url)

            # Wrap with observers if observation requested
            if observation_id:
                sync_transport = ObservingHTTPTransport(observation_id, base_sync_transport, proxy=proxy_url)
                async_transport = ObservingAsyncHTTPTransport(observation_id, base_async_transport, proxy=proxy_url)
                logger.debug(f"🔬 Observation enabled for {observation_id}")
            else:
                sync_transport = base_sync_transport
                async_transport = base_async_transport

            return sync_transport, async_transport

        # No proxy case
        if observation_id:
            # Still observe even without proxy
            base_sync_transport = httpx.HTTPTransport()
            base_async_transport = httpx.AsyncHTTPTransport()
            sync_transport = ObservingHTTPTransport(observation_id, base_sync_transport)
            async_transport = ObservingAsyncHTTPTransport(observation_id, base_async_transport)
            logger.debug(f"🔬 Observation enabled for {observation_id} (no proxy)")
            return sync_transport, async_transport

        logger.debug(f"🚫 No proxy configured (proxy_config={proxy_config}, default={self.config.proxy_config})")
        return None, None

    def _proxy_info_to_string(self, proxy_info: Optional[ProxyConfig]) -> Optional[str]:
        """Convert proxy configuration to a storable/displayable string."""
        if proxy_info:
            return self._proxy_config_to_url(proxy_info)
        return "direct"

    def get_provider_id(self) -> str:
        return self.config.name

    def get_provider_type(self) -> str:
        return "google-genai"

    def get_endpoint(self) -> str:
        return "https://generativelanguage.googleapis.com"

    def _get_pacific_date(self) -> str:
        """Get current date in Pacific timezone as YYYY-MM-DD string"""
        return _current_pacific_date()

    async def _get_quota_record(self, api_key: str, model_name: str) -> QuotaRecord:
        """Get or create quota record for an API key + model combination"""
        quota, _ = await ApiKeyQuota.get_or_create_quota(
            api_key=api_key, provider_id=self.config.name, model_name=model_name
        )
        return quota

    async def _select_best_api_key(self, model_name: str) -> str:
        """
        Select the API key with lowest usage for the given model.

        Returns the first key from the set of keys with the lowest request count FOR THIS MODEL.
        Order among equals doesn't matter - just consistent selection.
        """
        # Group keys by their request count for THIS MODEL, excluding exhausted/error keys
        available_keys = []
        exhausted_keys = []
        error_prone_keys = []
        pacific_date = _current_pacific_date()

        for key in self.config.api_keys:
            quota = await self._get_quota_record(key, model_name)
            model_limit = self.get_model_daily_limit(model_name)
            actual_requests_today = self._classify_api_key_for_model(
                quota,
                key,
                model_name,
                pacific_date,
                model_limit,
                exhausted_keys,
                error_prone_keys,
            )

            if actual_requests_today is None:
                continue

            available_keys.append((key, actual_requests_today))

        if not available_keys:
            # All keys exhausted for this model - provide detailed status
            total_keys = len(self.config.api_keys)
            logger.error(f"🚫 ALL {total_keys} API KEYS EXHAUSTED FOR MODEL {model_name}:")
            logger.error(f"   - Quota exhausted: {len(exhausted_keys)} keys")
            logger.error(f"   - Error-prone: {len(error_prone_keys)} keys")

            # Calculate seconds until quota reset (midnight Pacific time)
            now_pacific = datetime.now(PACIFIC_TZ)
            tomorrow_pacific = (now_pacific + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            seconds_until_reset = int((tomorrow_pacific - now_pacific).total_seconds())

            logger.error(f"⏰ All API keys exhausted. Quota resets in {seconds_until_reset}s at midnight Pacific")

            # Only raise predictive 429 if enabled
            if self.config.predictive_429_enabled:
                # Raise a specific exception that the container can catch and convert to 429
                from google.api_core.exceptions import ResourceExhausted

                raise ResourceExhausted(
                    f"All {total_keys} API keys exhausted for model {model_name}. "
                    f"Quota resets in {seconds_until_reset} seconds at midnight Pacific time.",
                    errors=[{"reason": "QUOTA_EXHAUSTED", "retry_after_seconds": seconds_until_reset}],
                )
            else:
                # Predictive 429 disabled - fall back to first key and let API handle the error
                logger.warning(
                    "⚠️ Predictive 429 disabled - using first API key despite quota tracking showing exhaustion"
                )
                return self.config.api_keys[0]

        # Sort by request count, then by key name for consistent ordering
        available_keys.sort(key=lambda x: (x[1], x[0]))

        # Find all keys with the lowest usage count
        lowest_count = available_keys[0][1]
        lowest_usage_keys = [key for key, count in available_keys if count == lowest_count]

        # Random selection amongst equals (mitigation for when quota tracking is broken)
        best_key = secrets.choice(lowest_usage_keys)

        logger.debug(
            f"Selected API key {best_key[:8]}... for {model_name} with {lowest_count} requests today "
            f"({len(lowest_usage_keys)} keys at this usage level)"
        )

        return best_key

    def _classify_api_key_for_model(
        self,
        quota: QuotaRecord,
        key: str,
        model_name: str,
        pacific_date: str,
        model_limit: int,
        exhausted_keys: List[str],
        error_prone_keys: List[str],
    ) -> Optional[int]:
        """Return the effective daily request count for a quota record, or None if it should be skipped."""
        # Skip permanently invalid keys first.
        if quota.invalid_key:
            logger.debug(f"API key {key[:8]}... marked as invalid, skipping")
            return None

        actual_requests_today = quota.requests_today if quota.last_reset_date == pacific_date else 0

        if actual_requests_today >= model_limit:
            exhausted_keys.append(key)
            logger.debug(
                f"API key {key[:8]}... exhausted for {model_name} ({actual_requests_today}/{model_limit}) reset_date={quota.last_reset_date} today={pacific_date}"
            )
            return None

        if quota.error_count > 20:  # Increased threshold
            error_prone_keys.append(key)
            logger.debug(f"API key {key[:8]}... too many errors for {model_name} ({quota.error_count})")
            return None

        if self._is_recent_quota_exhaustion(quota, key, model_name, pacific_date):
            return None

        return actual_requests_today

    def _is_recent_quota_exhaustion(self, quota: QuotaRecord, key: str, model_name: str, pacific_date: str) -> bool:
        """Check whether a quota exhaustion timestamp is still current for the Pacific day."""
        if not quota.quota_exhausted_at:
            return False

        try:
            quota_exhausted_pacific = _to_pacific_datetime(quota.quota_exhausted_at, assume_utc=True)
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"API key {key[:8]}... has malformed quota_exhausted_at, allowing: {e}")
            return False

        exhausted_date = quota_exhausted_pacific.strftime("%Y-%m-%d")
        if exhausted_date == pacific_date:
            logger.debug(
                f"API key {key[:8]}... exhausted TODAY for {model_name} at {quota_exhausted_pacific.strftime('%H:%M')}, skipping"
            )
            return True

        logger.debug(
            f"API key {key[:8]}... exhaustion from {exhausted_date} is stale (today is {pacific_date}), allowing"
        )
        return False

    async def _update_api_key_stats(
        self,
        api_key: str,
        model_name: str,
        success: bool,
        tokens: int = 0,
        error: Optional[str] = None,
        status_code: Optional[int] = None,
    ):
        """Update statistics for an API key + model combination after a request

        Args:
            api_key: The API key used
            model_name: Model name
            success: Whether request succeeded
            tokens: Token count if successful
            error: Error message if failed
            status_code: HTTP status code (for detecting 403s without error messages)
        """
        from .redis_backend import RedisApiKeyQuota

        quota = await self._get_quota_record(api_key, model_name)

        # Database handles timezone automatically

        if success:
            await self._record_api_key_success(RedisApiKeyQuota, quota, api_key, model_name, tokens)
        else:
            error_message = error or ""
            # Check error type and handle appropriately
            # IMPORTANT: Check both error message AND status code (403 errors may have empty error string)
            if self._is_invalid_key_error(error_message, status_code):
                await self._record_invalid_api_key(ApiKeyQuota, api_key, error_message, status_code)

            # Check if this is a quota exhaustion error
            elif self._is_quota_exhausted_error(error_message):
                await self._record_quota_exhaustion(ApiKeyQuota, quota, api_key, model_name, error_message)
            else:
                await self._record_regular_error(ApiKeyQuota, quota, api_key, model_name, error_message, status_code)

        self._log_quota_status(api_key, model_name, quota)

    async def _record_api_key_success(
        self,
        quota_backend: Any,
        quota: QuotaRecord,
        api_key: str,
        model_name: str,
        tokens: int,
    ) -> None:
        try:
            await quota_backend.increment_usage(
                api_key=api_key,
                provider_id=self.config.name,
                model_name=model_name,
                request_count=1,
                token_count=tokens,
            )
            quota.mark_request_success(tokens=tokens)
            logger.info(
                "API key %s... successful request for %s: %s/%s RPD, %s tokens",
                api_key[:8],
                model_name,
                quota.requests_today,
                self.get_model_daily_limit(model_name),
                tokens,
            )
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to update quota for {api_key[:8]}... / {model_name}: {e}")
            logger.error("⚠️  Quota tracking broken - key rotation will not work correctly!")
            quota.mark_request_success(tokens=tokens)

    async def _record_invalid_api_key(
        self, quota_backend: Any, api_key: str, error_message: str, status_code: Optional[int]
    ) -> None:
        key_hash = quota_backend.hash_api_key(api_key)
        try:
            await quota_backend.mark_invalid_by_hash(key_hash, self.config.name)
            logger.error(f"🚫 API key {api_key[:8]}... MARKED INVALID (status={status_code}, error={error_message})")

            try:
                self.config.api_keys.remove(api_key)
                logger.warning(
                    f"🗑️  Removed API key {api_key[:8]}... from selection pool ({len(self.config.api_keys)} keys remaining)"
                )
            except ValueError:
                pass
        except Exception as e:
            logger.error(f"❌ Failed to mark API key as invalid: {e}")
            import traceback

            logger.error(traceback.format_exc())

    async def _record_quota_exhaustion(
        self,
        quota_backend: Any,
        quota: QuotaRecord,
        api_key: str,
        model_name: str,
        error_message: str,
    ) -> None:
        quota.mark_request_failure(error=error_message, quota_exhausted=True)

        try:
            await quota_backend.mark_quota_exhausted(api_key, self.config.name, model_name, error_message)
        except Exception as e:
            logger.error(f"Failed to persist quota exhaustion to Redis: {e}")

        logger.error(f"🚫 API key {api_key[:8]}... QUOTA EXHAUSTED (429) for {model_name}: Hard marked as depleted")

        retry_delay = self._extract_retry_delay(error_message)
        if retry_delay:
            logger.warning(f"🕒 Google suggests retry in {retry_delay}s for {api_key[:8]}... / {model_name}")
        else:
            logger.warning(f"🕒 Key {api_key[:8]}... / {model_name} exhausted, will reset at midnight Pacific")

    async def _record_regular_error(
        self,
        quota_backend: Any,
        quota: QuotaRecord,
        api_key: str,
        model_name: str,
        error_message: str,
        status_code: Optional[int],
    ) -> None:
        quota.mark_request_failure(error=error_message)

        try:
            await quota_backend.mark_error(api_key, self.config.name, model_name, error_message)
        except Exception as e:
            logger.error(f"❌ Failed to persist error to Redis: {e}")

        if status_code == 403:
            logger.warning(f"⚠️  403 error but not marked invalid - status={status_code}, error={error_message!r}")

        logger.warning(f"API key {api_key[:8]}... error #{quota.error_count} for {model_name}: {error_message}")

    def _log_quota_status(self, api_key: str, model_name: str, quota: QuotaRecord) -> None:
        model_limit = self.get_model_daily_limit(model_name)
        if model_limit <= 0:
            logger.warning("API key %s... / %s has non-positive daily limit: %s", api_key[:8], model_name, model_limit)
            return

        quota_percentage = (quota.requests_today / model_limit) * 100
        if quota.requests_today >= model_limit:
            logger.error(
                f"🚫 API key {api_key[:8]}... / {model_name} DAILY LIMIT REACHED: {quota.requests_today}/{model_limit}"
            )
        elif quota.requests_today >= (model_limit * 0.8):
            logger.warning(
                "API key %s... / %s approaching daily limit: %.1f%% used (%s/%s)",
                api_key[:8],
                model_name,
                quota_percentage,
                quota.requests_today,
                model_limit,
            )

    def _is_invalid_key_error(
        self, error_msg: Optional[str] = None, status_code: Optional[int] = None
    ) -> bool:
        """Check if error indicates invalid/expired API key

        Args:
            error_msg: Error message text (may be None or empty)
            status_code: HTTP status code

        Returns:
            True if this error indicates an invalid/expired API key
        """
        # CRITICAL FIX: 403 status code is an invalid key error even without error message
        # This handles cases where Google returns 403 but no error text
        if status_code == 403:
            return True

        # Check error message for known invalid key indicators
        if error_msg:
            error_lower = error_msg.lower()
            invalid_key_indicators = [
                "permission denied",
                "permission_denied",  # With underscore (PERMISSION_DENIED from Google)
                "permissiondenied",  # Sometimes without space or underscore
                "api key not valid",
                "invalid api key",
                "api_key_invalid",
                "authentication failed",
                "unauthorized",
                "forbidden",
                "invalid_argument",  # Often used for bad keys in Google APIs
                "credentials are missing or invalid",
                "api key expired",
                "403",  # Explicit 403 in error message
                "denied access",  # "Your project has been denied access"
            ]
            return any(indicator in error_lower for indicator in invalid_key_indicators)

        return False

    def _is_quota_exhausted_error(self, error_msg: Optional[str] = None) -> bool:
        """Check if error indicates quota exhaustion"""
        if not error_msg:
            return False

        error_lower = error_msg.lower()
        quota_indicators = [
            "429",
            "resource_exhausted",
            "quota exceeded",
            "current quota",
            "quota.*exceeded",
            "requests per day",
            "free_tier_requests",
        ]

        return any(indicator in error_lower for indicator in quota_indicators)

    def _extract_retry_delay(self, error_msg: Optional[str] = None) -> Optional[float]:
        """Extract retry delay from Google error message"""
        if not error_msg:
            return None

        # Look for patterns like "retry in 20.915074628s" or "retryDelay': '20s'"
        patterns = [
            r"retry in (\d+(?:\.\d+)?)s",
            r"retryDelay.*?(\d+(?:\.\d+)?)s",
            r"Please retry.*?(\d+(?:\.\d+)?).*?s",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_msg, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None

    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name to Google GenAI format"""
        return self.MODEL_MAPPINGS.get(model_name, model_name)

    async def health_check(self) -> bool:
        """Check if at least one API key is working"""
        for api_key in self.config.api_keys:
            try:
                # Create client with proxy transport
                proxy_config = self.config.get_proxy_for_model("health-check")
                sync_transport, async_transport = self._create_proxy_transport(proxy_config)

                from google.genai import types

                if sync_transport and async_transport:
                    http_options = types.HttpOptions(
                        client_args={"transport": sync_transport, "trust_env": False},
                        async_client_args={"transport": async_transport, "trust_env": False},
                    )
                    client = genai.Client(api_key=api_key, http_options=http_options)
                else:
                    client = genai.Client(api_key=api_key)

                # Try to list models as a health check
                models = await asyncio.to_thread(client.models.list)
                list(models)  # Force evaluation
                return True
            except Exception as e:
                logger.debug(f"Health check failed for key {api_key[:8]}...: {e}")
                continue
        return False

    async def discover_models(self) -> List[ModelInfo]:
        """Discover available models from Google GenAI API or return static list"""
        # Check cache first
        if self._cached_models and self._cache_time and datetime.now() - self._cache_time < self._cache_ttl:
            return self._cached_models

        # Try to discover models using available API keys (live discovery)
        for api_key in self.config.api_keys:
            try:
                # Create client with proxy transport
                proxy_config = self.config.get_proxy_for_model("model-discovery")
                sync_transport, async_transport = self._create_proxy_transport(proxy_config)

                from google.genai import types

                if sync_transport and async_transport:
                    http_options = types.HttpOptions(
                        client_args={"transport": sync_transport, "trust_env": False},
                        async_client_args={"transport": async_transport, "trust_env": False},
                    )
                    client = genai.Client(api_key=api_key, http_options=http_options)
                else:
                    client = genai.Client(api_key=api_key)

                models = []

                # List available models
                model_list = await asyncio.to_thread(client.models.list)
                for model in model_list:
                    # New API uses 'supported_actions' instead of 'supported_generation_methods'
                    supported_actions = getattr(model, "supported_actions", [])
                    if "generateContent" in supported_actions:
                        model_name = (getattr(model, "name", "") or "").split("/")[-1]  # Extract model name from full path

                        # Create model info
                        model_info = ModelInfo(
                            id=f"{model_name}@{self.get_provider_id()}",
                            name=model_name,
                            provider_id=self.get_provider_id(),
                            provider_type=self.get_provider_type(),
                            endpoint=self.get_endpoint(),
                            aliases=[model_name],
                            metadata={
                                "full_name": getattr(model, "name", ""),
                                "display_name": getattr(model, "display_name", model_name),
                                "description": getattr(model, "description", ""),
                                "supported_methods": supported_actions,
                                "input_token_limit": getattr(model, "input_token_limit", None),
                                "output_token_limit": getattr(model, "output_token_limit", None),
                            },
                        )
                        models.append(model_info)
                        logger.debug(f"Discovered Google GenAI model: {model_info.id}")

                # Cache the results
                self._cached_models = models
                self._cache_time = datetime.now()

                return models

            except Exception as e:
                logger.error(f"Error discovering models with API key {api_key[:8]}...: {e}")
                continue

        # All API keys failed - fall back to static model list
        logger.warning("Live model discovery failed for all API keys, falling back to static model list")
        return self._get_static_google_genai_models()

    def _get_static_google_genai_models(self) -> List[ModelInfo]:
        """Return static list of Google GenAI models from JSON file"""
        from pathlib import Path

        try:
            # Get the absolute path to the models JSON file
            current_dir = Path(__file__).parent
            models_file = current_dir / "models" / "google-genai-models-2025-september.json"

            if not models_file.exists():
                logger.error(f"Static Google GenAI models file not found: {models_file}")
                return []

            with open(models_file, "r") as f:
                data = json.load(f)

            models = []
            for model_data in data.get("models", []):
                # Only include models that support text generation
                supported_methods = model_data.get("supportedGenerationMethods", [])
                if "generateContent" not in supported_methods:
                    continue  # Skip embedding and other non-text generation models

                # Extract model name from full path (models/model-name -> model-name)
                full_name = model_data.get("name", "")
                model_name = full_name.split("/")[-1] if "/" in full_name else full_name

                # Create model info
                model_info = ModelInfo(
                    id=f"{model_name}@{self.get_provider_id()}",
                    name=model_name,
                    provider_id=self.get_provider_id(),
                    provider_type=self.get_provider_type(),
                    endpoint=self.get_endpoint(),
                    aliases=[model_name],
                    metadata={
                        "full_name": full_name,
                        "display_name": model_data.get("displayName", model_name),
                        "description": model_data.get("description", ""),
                        "supported_methods": supported_methods,
                        "input_token_limit": model_data.get("inputTokenLimit"),
                        "output_token_limit": model_data.get("outputTokenLimit"),
                        "version": model_data.get("version"),
                        "temperature": model_data.get("temperature"),
                        "top_p": model_data.get("topP"),
                        "top_k": model_data.get("topK"),
                        "max_temperature": model_data.get("maxTemperature"),
                        "thinking": model_data.get("thinking", False),
                    },
                )
                models.append(model_info)
                logger.debug(f"Static Google GenAI model: {model_info.id}")

            logger.info(f"Loaded {len(models)} static models for Google GenAI provider {self.get_provider_id()}")
            return models

        except Exception as e:
            logger.error(f"Error loading static Google GenAI models: {e}")
            return []

    def _convert_openai_to_genai_request(self, openai_request: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Convert OpenAI request format to Google GenAI format"""
        model_name = self._normalize_model_name(openai_request.get("model", ""))

        # Extract messages
        messages = openai_request.get("messages", [])
        if not messages:
            raise ValueError("No messages provided in request")

        contents = []
        for message in messages:
            genai_content = self._convert_openai_message_to_genai_content(message)
            if genai_content is not None:
                contents.append(genai_content)

        # Build generation config
        generation_config = {}

        # Map common parameters
        if "temperature" in openai_request:
            generation_config["temperature"] = openai_request["temperature"]
        if "max_tokens" in openai_request:
            generation_config["max_output_tokens"] = openai_request["max_tokens"]
        if "top_p" in openai_request:
            generation_config["top_p"] = openai_request["top_p"]

        # Google GenAI doesn't support streaming in the same way, so we'll handle that separately
        genai_request = {"contents": contents, "generation_config": generation_config}

        return model_name, genai_request

    def _convert_openai_message_to_genai_content(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        role = message.get("role", "user")
        parts = self._convert_openai_content_to_parts(message.get("content", ""))

        if not parts:
            return None

        if role == "system":
            parts[0]["text"] = f"System: {parts[0]['text']}"
            return {"role": "user", "parts": parts}

        if role == "assistant":
            return {"role": "model", "parts": parts}

        return {"role": "user", "parts": parts}

    def _convert_openai_content_to_parts(self, content: Any) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []

        if isinstance(content, str):
            parts.append({"text": content})
            return parts

        if not isinstance(content, list):
            return parts

        for item in content:
            if item.get("type") == "text":
                parts.append({"text": item.get("text", "")})
                continue

            if item.get("type") != "image_url":
                continue

            image_url = item.get("image_url", {}).get("url", "")
            if image_url.startswith("data:"):
                try:
                    header, base64_data = image_url.split(",", 1)
                    mime_type = header.split(":")[1].split(";")[0]
                    parts.append({"inline_data": {"mime_type": mime_type, "data": base64_data}})
                except Exception as e:
                    logger.warning(f"Failed to parse data URI: {e}")
            else:
                logger.warning(
                    f"Image URLs are not supported in this version, only base64 data URIs: {image_url[:30]}..."
                )

        return parts

    def _convert_genai_to_openai_response(self, genai_response: Any, original_model: str) -> Dict[str, Any]:
        """Convert Google GenAI response to OpenAI format"""
        try:
            text_content = self._extract_genai_text(genai_response)
            usage = self._extract_genai_usage(genai_response)

            # Build OpenAI-compatible response
            openai_response = {
                "id": f"chatcmpl-{datetime.now().timestamp()}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": original_model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": text_content}, "finish_reason": "stop"}
                ],
                "usage": usage,
            }

            return openai_response

        except Exception as e:
            logger.error(f"Error converting GenAI response to OpenAI format: {e}")
            raise

    def _extract_genai_text(self, genai_response: Any) -> str:
        text_content = ""
        if hasattr(genai_response, "text") and genai_response.text:
            return genai_response.text

        if not getattr(genai_response, "candidates", None):
            return text_content

        candidate = genai_response.candidates[0]
        if not hasattr(candidate, "content") or not candidate.content:
            return text_content

        parts = getattr(candidate.content, "parts", None)
        if not parts:
            return text_content

        for part in parts:
            if hasattr(part, "text"):
                text_content += part.text

        return text_content

    def _extract_genai_usage(self, genai_response: Any) -> Dict[str, int]:
        if not hasattr(genai_response, "usage_metadata") or not genai_response.usage_metadata:
            return {}

        return {
            "prompt_tokens": getattr(genai_response.usage_metadata, "prompt_token_count", 0),
            "completion_tokens": getattr(genai_response.usage_metadata, "candidates_token_count", 0),
            "total_tokens": getattr(genai_response.usage_metadata, "total_token_count", 0),
        }

    async def generate_completion(
        self, openai_request: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Optional["RequestMetadata"]]:
        """Generate completion using Google GenAI API

        Returns: (response_dict, metadata)
        """
        import uuid

        context = GoogleGenAICompletionContext(
            original_model=openai_request.get("model", ""),
            observation_id=f"obs_{uuid.uuid4().hex[:12]}",
        )

        try:
            context = await self._build_completion_context(openai_request, context)
            response = await self._run_completion_request(context)
            return await self._finalize_completion(context, response)

        except GoogleGenAIRequestError:
            raise
        except Exception as error:
            raise await self._handle_completion_exception(context, error)

    async def _build_completion_context(
        self, openai_request: Dict[str, Any], context: GoogleGenAICompletionContext
    ) -> GoogleGenAICompletionContext:
        context.model_name, context.genai_request = self._convert_openai_to_genai_request(openai_request)
        context.api_key = await self._select_best_api_key(context.model_name)
        context.api_key_suffix = context.api_key[-8:] if len(context.api_key) > 8 else context.api_key
        context.api_key_index, context.api_key_total = self._resolve_api_key_position(context.api_key)
        context.proxy_config, context.proxy_pool_index, context.pool_info = self._select_proxy_configuration(
            context.model_name
        )
        context.proxy_info = context.proxy_config if context.proxy_config else None
        context.proxy_url = self._proxy_config_to_url(context.proxy_info)

        if context.proxy_url and not self._proxy_available_for_use(context.proxy_url):
            health = self._proxy_health_snapshot(context.proxy_url)
            raise self._build_request_error_from_context(
                context,
                f"Configured proxy {self._mask_proxy_url(context.proxy_url)} is currently unhealthy"
                + (f": {health['last_error']}" if health.get("last_error") else ""),
            )

        self._schedule_proxy_probe(context.proxy_url)
        context.sync_transport, context.async_transport = self._create_proxy_transport(
            context.proxy_config, context.observation_id
        )
        context.client = self._create_completion_client(
            context.api_key, context.sync_transport, context.async_transport
        )

        key_position_str = f" key #{context.api_key_index}/{context.api_key_total}" if context.api_key_index else ""
        logger.info(
            f"🚀 Outbound request: model={context.model_name}, api_key=...{context.api_key_suffix}{key_position_str}, proxy={context.proxy_info or 'direct'}{context.pool_info} [obs={context.observation_id}]"
        )

        return context

    def _resolve_api_key_position(self, api_key: str) -> Tuple[Optional[int], Optional[int]]:
        try:
            return self.config.api_keys.index(api_key) + 1, len(self.config.api_keys)
        except (ValueError, AttributeError):
            return None, None

    def _select_proxy_configuration(
        self, model_name: str
    ) -> Tuple[Optional[ProxyConfig], Optional[int], str]:
        if self._should_use_proxy_pool(model_name):
            proxy_config, proxy_pool_index = self._get_next_proxy_from_pool()
            pool_info = (
                f" (pool #{(proxy_pool_index or 0) + 1}/{len(self.config.proxy_pool)})" if self.config.proxy_pool else ""
            )
            return proxy_config, proxy_pool_index, pool_info

        return self.config.get_proxy_for_model(model_name), None, ""

    def _create_completion_client(
        self,
        api_key: str,
        sync_transport: Any,
        async_transport: Any,
    ) -> Any:
        from google.genai import types

        if sync_transport and async_transport:
            http_options = types.HttpOptions(
                client_args={"transport": sync_transport, "trust_env": False},
                async_client_args={"transport": async_transport, "trust_env": False},
            )
            logger.debug("httpx transports active: True (proxy attached)")
            logger.debug("httpx trust_env disabled via HttpOptions: True")
            return genai.Client(api_key=api_key, http_options=http_options)

        logger.debug("httpx transports active: False (no proxy)")
        logger.debug("httpx trust_env disabled via HttpOptions: True")
        return genai.Client(api_key=api_key)

    async def _run_completion_request(self, context: GoogleGenAICompletionContext) -> Any:
        await self._rate_limiter.acquire_slot()
        try:
            return await asyncio.to_thread(
                context.client.models.generate_content,
                model=context.model_name,
                contents=context.genai_request["contents"],
                config=context.genai_request["generation_config"],
            )
        finally:
            await self._rate_limiter.release_slot()

    async def _finalize_completion(
        self, context: GoogleGenAICompletionContext, response: Any
    ) -> tuple[Dict[str, Any], Optional["RequestMetadata"]]:
        from .request_metadata import RequestMetadata
        from .transport_observer import get_observer

        openai_response = self._convert_genai_to_openai_response(response, context.original_model)
        tokens_used = openai_response.get("usage", {}).get("total_tokens", 0)
        await self._update_api_key_stats(context.api_key, context.model_name, success=True, tokens=tokens_used)

        if context.proxy_url:
            self._mark_proxy_health(context.proxy_url, success=True)

        observer = get_observer()
        observation = observer.get_observation(context.observation_id)
        actual_key_suffix, actual_proxy, key_verified, proxy_verified = self._resolve_observation_state(
            context, observation
        )

        metadata = RequestMetadata(
            api_key_suffix=actual_key_suffix,
            proxy_used=actual_proxy,
            provider_id=self.config.name,
            model_name=context.model_name,
            api_key_index=context.api_key_index if context.api_key_index else None,
            api_key_total=context.api_key_total if context.api_key_total else None,
            api_key_verified=key_verified,
            proxy_verified=proxy_verified,
            observation_id=context.observation_id,
        )

        return openai_response, metadata

    def _resolve_observation_state(
        self, context: GoogleGenAICompletionContext, observation: Any
    ) -> Tuple[Optional[str], Optional[str], bool, bool]:
        actual_key_suffix = context.api_key_suffix
        actual_proxy = self._proxy_info_to_string(context.proxy_info)
        key_verified = False
        proxy_verified = False

        if observation:
            if observation.api_key_used:
                actual_key_suffix = (
                    observation.api_key_used[-8:] if len(observation.api_key_used) > 8 else observation.api_key_used
                )
                key_verified = True

            if observation.proxy_url:
                actual_proxy = observation.proxy_url
                proxy_verified = True
            elif not context.proxy_info:
                proxy_verified = True

            logger.info(
                f"✅ GROUND TRUTH: key=...{actual_key_suffix} (verified={key_verified}), proxy={actual_proxy or 'NONE'} (verified={proxy_verified})"
            )
        else:
            logger.warning(
                f"⚠️ No observation for {context.observation_id} - using INTENT: key=...{context.api_key_suffix}, proxy={context.proxy_info or 'NONE'}"
            )

        return actual_key_suffix, actual_proxy, key_verified, proxy_verified

    def _build_request_error_from_context(
        self, context: GoogleGenAICompletionContext, message: str
    ) -> GoogleGenAIRequestError:
        return GoogleGenAIRequestError(
            message,
            provider_id=self.config.name,
            model_name=context.model_name or context.original_model or None,
            api_key_suffix=context.api_key_suffix,
            api_key_index=context.api_key_index,
            api_key_total=context.api_key_total,
            proxy_used=self._proxy_info_to_string(context.proxy_info),
        )

    async def _record_completion_failure(
        self,
        context: GoogleGenAICompletionContext,
        error_message: str,
        status_code: Optional[int],
    ) -> None:
        if context.api_key and context.model_name:
            await self._update_api_key_stats(
                context.api_key,
                context.model_name,
                success=False,
                error=error_message,
                status_code=status_code,
            )

    def _extract_retry_after_seconds(self, error: Exception) -> Optional[float]:
        retry_after_seconds = None
        error_details = getattr(error, "errors", None)
        if error_details:
            for error_detail in error_details:
                if isinstance(error_detail, dict) and "retry_after_seconds" in error_detail:
                    retry_after_seconds = error_detail["retry_after_seconds"]
                    break
        return retry_after_seconds

    def _extract_status_code_from_exception(self, error: Exception) -> Optional[int]:
        error_str = str(error).lower()
        if "403" in error_str or "permission" in error_str:
            return 403
        if "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
            return 429
        if "401" in error_str or "unauthorized" in error_str:
            return 401
        return None

    async def _handle_completion_exception(
        self, context: GoogleGenAICompletionContext, error: Exception
    ) -> GoogleGenAIRequestError:
        if isinstance(error, ResourceExhausted):
            error_msg = f"Google GenAI quota exhausted: {error}"
            await self._record_completion_failure(context, error_msg, None)
            quota_error = self._build_request_error_from_context(context, error_msg)
            quota_error.retry_after_seconds = self._extract_retry_after_seconds(error)
            return quota_error

        if isinstance(error, PermissionDenied):
            error_msg = f"Google GenAI permission denied: {error}"
            await self._record_completion_failure(context, error_msg, 403)
            return self._build_request_error_from_context(context, error_msg)

        if isinstance(error, InvalidArgument):
            error_msg = f"Google GenAI invalid argument: {error}"
            await self._record_completion_failure(context, error_msg, 400)
            return self._build_request_error_from_context(context, error_msg)

        error_msg = f"Google GenAI error: {error}"
        if context.proxy_url and self._is_proxy_connectivity_error(error):
            self._mark_proxy_health(context.proxy_url, success=False, error=str(error))

        await self._record_completion_failure(
            context,
            error_msg,
            self._extract_status_code_from_exception(error),
        )
        return self._build_request_error_from_context(context, error_msg)

    async def get_api_key_stats(self, include_unused_models: bool = False) -> Dict[str, Dict[str, Any]]:
        """Get current API key usage statistics (grouped by API key)

        Args:
            include_unused_models: If False (default), only include models with requests_today > 0
        """
        stats = {}

        # Get all quota records for this provider (Redis-based)
        quotas = await ApiKeyQuota.get_provider_usage(self.config.name)

        # Group stats by API key
        for quota in quotas:
            # Skip models with no usage unless explicitly requested
            if not include_unused_models and quota.requests_today == 0:
                continue

            # Use hash for privacy but show some chars for identification
            key_display = f"{quota.api_key_hash[:8]}..."
            model = quota.model_name

            if key_display not in stats:
                stats[key_display] = {
                    "models": {},
                    "total_requests_today": 0,
                    "total_tokens_today": 0,
                    "total_errors": 0,
                }

            model_limit = self.get_model_daily_limit(model)
            quota_percentage = (quota.requests_today / model_limit) * 100
            is_exhausted = quota.requests_today >= model_limit or quota.quota_exhausted_at is not None
            last_request = _format_optional_datetime(quota.updated_at)
            quota_exhausted_at = _format_optional_datetime(quota.quota_exhausted_at)

            stats[key_display]["models"][model] = {
                "requests_today": quota.requests_today,
                "tokens_today": quota.tokens_today,
                "error_count": quota.error_count,
                "last_request": last_request,
                "last_error": quota.last_error,
                "quota_percentage": quota_percentage,
                "quota_exhausted": is_exhausted,
                "quota_exhausted_at": quota_exhausted_at,
                "status": _quota_status(quota.invalid_key, is_exhausted),
            }

            # Aggregate totals for this API key
            stats[key_display]["total_requests_today"] += quota.requests_today
            stats[key_display]["total_tokens_today"] += quota.tokens_today
            stats[key_display]["total_errors"] += quota.error_count

        # Only add unused API keys if we're showing all models
        if include_unused_models:
            used_key_hashes = set()
            for quota in quotas:
                used_key_hashes.add(quota.api_key_hash)

            for api_key in self.config.api_keys:
                key_hash = ApiKeyQuota.hash_api_key(api_key)
                if key_hash not in used_key_hashes:
                    key_display = f"{key_hash[:8]}..."
                    stats[key_display] = {
                        "models": {},
                        "total_requests_today": 0,
                        "total_tokens_today": 0,
                        "total_errors": 0,
                    }

        # Add rate limiter statistics
        stats["_rate_limiter"] = self._rate_limiter.stats

        return stats
