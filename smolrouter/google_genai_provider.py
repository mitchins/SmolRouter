"""
Google Generative AI provider implementation.

Provides integration with Google's Generative AI API, supporting multiple API keys
with intelligent rotation based on requests-per-day (RPD) quotas.
"""

import logging
import json
import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse
import pytz
import httpx

from google import genai
from google.api_core.exceptions import ResourceExhausted, PermissionDenied, InvalidArgument

from .interfaces import IModelProvider, ModelInfo, ProviderConfig, ProxyConfig
from .database import ApiKeyQuota
from .rate_limiter import GoogleGenAIRequestFunnel
from .request_metadata import RequestMetadata

logger = logging.getLogger(__name__)


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
        pacific_tz = pytz.timezone("US/Pacific")
        now_pacific = datetime.now(pacific_tz)
        now_date = now_pacific.date()

        # Check against last request time
        if self.last_request:
            last_request_pacific = (
                self.last_request.replace(tzinfo=pacific_tz)
                if self.last_request.tzinfo is None
                else self.last_request.astimezone(pacific_tz)
            )
            last_request_date = last_request_pacific.date()
            if now_date > last_request_date:
                return True

        # Also check against quota exhaustion time (important for 429 recovery)
        if self.quota_exhausted_at:
            exhausted_pacific = (
                self.quota_exhausted_at.replace(tzinfo=pacific_tz)
                if self.quota_exhausted_at.tzinfo is None
                else self.quota_exhausted_at.astimezone(pacific_tz)
            )
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
        self.api_keys = kwargs.pop("api_keys", [])
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
                with open(self.api_keys_file, "r") as f:
                    file_keys = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                self.api_keys.extend(file_keys)
                logger.info(f"Loaded {len(file_keys)} API keys from {self.api_keys_file}")
            except Exception as e:
                logger.error(f"Failed to load API keys from {self.api_keys_file}: {e}")
                raise

        if not self.api_keys:
            raise ValueError("No valid API keys found")


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

        for proxy_config in self.config.per_model_proxy.values():
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
        for model_name, proxy_config in sorted(self.config.per_model_proxy.items()):
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
        if model_name in self.config.per_model_proxy:
            model_proxy = self.config.per_model_proxy[model_name]
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

        # Gemma-3 models have generous free tier limits
        if "gemma" in model_lower and "3" in model_lower:
            return 14400  # 14,400 RPD per project (free tier)

        # Gemini 3 Flash Preview (recommended, launched Dec 17, 2025)
        if "gemini" in model_lower and ("3.0" in model_lower or "gemini-3" in model_lower):
            return 20  # 20 RPD per project (free tier)

        # Gemini 2.5 Flash models (severely reduced limits after Dec 6, 2025)
        if "2.5" in model_lower and "flash" in model_lower:
            if "lite" in model_lower:
                return 1000  # Flash-lite still has generous limits (250-1500 RPD range, using conservative 1000)
            else:
                return 20  # 20 RPD per project (free tier, down from 250-500)

        # Gemini 2.0 Flash models
        if "2.0" in model_lower and "flash" in model_lower:
            if "exp" in model_lower or "experimental" in model_lower:
                return 5  # Experimental models have minimal free access
            else:
                return 20  # Similar to 2.5 Flash

        # Gemini 2.x Pro models (very limited free tier)
        if "pro" in model_lower and ("2.5" in model_lower or "2.0" in model_lower):
            return 20  # Conservative estimate for Pro models

        # Gemini 1.5 models (older generation)
        if "1.5" in model_lower:
            if "pro" in model_lower:
                return 50  # Gemini 1.5 Pro
            elif "flash" in model_lower:
                return 1000  # Gemini 1.5 Flash has better limits

        # Preview/experimental models (catch-all)
        if any(keyword in model_lower for keyword in ["preview", "experimental"]):
            return 5

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
        pacific_tz = pytz.timezone("US/Pacific")
        return datetime.now(pacific_tz).date().strftime("%Y-%m-%d")

    async def _get_quota_record(self, api_key: str, model_name: str) -> dict:
        """Get or create quota record for an API key + model combination"""
        quota, created = await ApiKeyQuota.get_or_create_quota(
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

        for key in self.config.api_keys:
            quota = await self._get_quota_record(key, model_name)

            # Skip if key is permanently invalid/expired
            if quota.invalid_key:
                logger.debug(f"API key {key[:8]}... marked as invalid, skipping")
                continue

            # Check if quota should be reset due to date change (defensive check)
            pacific_tz = pytz.timezone("US/Pacific")
            pacific_date = datetime.now(pacific_tz).strftime("%Y-%m-%d")

            # If the quota hasn't been reset today, consider it fresh
            actual_requests_today = quota.requests_today if quota.last_reset_date == pacific_date else 0

            # Check if key has hit daily limit for this model (either by count or 429 response)
            model_limit = self.get_model_daily_limit(model_name)
            if actual_requests_today >= model_limit:
                exhausted_keys.append(key)
                logger.debug(
                    f"API key {key[:8]}... exhausted for {model_name} ({actual_requests_today}/{model_limit}) reset_date={quota.last_reset_date} today={pacific_date}"
                )
                continue

            # Skip keys with too many recent errors for this model
            if quota.error_count > 20:  # Increased threshold
                error_prone_keys.append(key)
                logger.debug(f"API key {key[:8]}... too many errors for {model_name} ({quota.error_count})")
                continue

            # Check for recent quota errors for this model (even if not at limit)
            # Only skip if the quota error happened TODAY (same Pacific date as now)
            if quota.quota_exhausted_at:
                # quota_exhausted_at could be datetime object or string from database
                pacific_tz = pytz.timezone("US/Pacific")

                try:
                    if isinstance(quota.quota_exhausted_at, str):
                        # Parse datetime string from database
                        quota_exhausted_dt = datetime.fromisoformat(quota.quota_exhausted_at.replace("Z", "+00:00"))
                        if quota_exhausted_dt.tzinfo is None:
                            quota_exhausted_pacific = pytz.utc.localize(quota_exhausted_dt).astimezone(pacific_tz)
                        else:
                            quota_exhausted_pacific = quota_exhausted_dt.astimezone(pacific_tz)
                    else:
                        # Handle datetime object
                        if quota.quota_exhausted_at.tzinfo is None:
                            quota_exhausted_pacific = pytz.utc.localize(quota.quota_exhausted_at).astimezone(pacific_tz)
                        else:
                            quota_exhausted_pacific = quota.quota_exhausted_at.astimezone(pacific_tz)

                    # CRITICAL FIX: Only skip if exhaustion was TODAY (same Pacific date)
                    # Keys exhausted yesterday should be available again after midnight Pacific reset
                    exhausted_date = quota_exhausted_pacific.strftime("%Y-%m-%d")
                    if exhausted_date == pacific_date:
                        # Exhausted today - skip this key for now
                        logger.debug(
                            f"API key {key[:8]}... exhausted TODAY for {model_name} at {quota_exhausted_pacific.strftime('%H:%M')}, skipping"
                        )
                        continue
                    else:
                        # Exhausted on a previous day - key should be available again
                        logger.debug(
                            f"API key {key[:8]}... exhaustion from {exhausted_date} is stale (today is {pacific_date}), allowing"
                        )
                except (ValueError, TypeError, AttributeError) as e:
                    # Skip malformed timestamp - allow key through
                    logger.warning(f"API key {key[:8]}... has malformed quota_exhausted_at, allowing: {e}")
                    pass

            available_keys.append((key, actual_requests_today))

        if not available_keys:
            # All keys exhausted for this model - provide detailed status
            total_keys = len(self.config.api_keys)
            logger.error(f"🚫 ALL {total_keys} API KEYS EXHAUSTED FOR MODEL {model_name}:")
            logger.error(f"   - Quota exhausted: {len(exhausted_keys)} keys")
            logger.error(f"   - Error-prone: {len(error_prone_keys)} keys")

            # Calculate seconds until quota reset (midnight Pacific time)
            pacific_tz = pytz.timezone("US/Pacific")
            now_pacific = datetime.now(pacific_tz)
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
        import random

        best_key = random.choice(lowest_usage_keys)

        logger.debug(
            f"Selected API key {best_key[:8]}... for {model_name} with {lowest_count} requests today "
            f"({len(lowest_usage_keys)} keys at this usage level)"
        )

        return best_key

    async def _update_api_key_stats(
        self, api_key: str, model_name: str, success: bool, tokens: int = 0, error: str = None, status_code: int = None
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
            # Actually increment usage in Redis
            try:
                await RedisApiKeyQuota.increment_usage(
                    api_key=api_key,
                    provider_id=self.config.name,
                    model_name=model_name,
                    request_count=1,
                    token_count=tokens,
                )
                quota.mark_request_success(tokens=tokens)
                logger.info(
                    f"API key {api_key[:8]}... successful request for {model_name}: {quota.requests_today}/{self.get_model_daily_limit(model_name)} RPD, {tokens} tokens"
                )
            except Exception as e:
                logger.error(f"❌ CRITICAL: Failed to update quota for {api_key[:8]}... / {model_name}: {e}")
                logger.error("⚠️  Quota tracking broken - key rotation will not work correctly!")
                # Still update local object for logging purposes
                quota.mark_request_success(tokens=tokens)
        else:
            # Check error type and handle appropriately
            # IMPORTANT: Check both error message AND status code (403 errors may have empty error string)
            if self._is_invalid_key_error(error, status_code):
                # Mark invalid across all models for this key
                key_hash = ApiKeyQuota.hash_api_key(api_key)
                try:
                    await ApiKeyQuota.mark_invalid_by_hash(key_hash, self.config.name)
                    logger.error(f"🚫 API key {api_key[:8]}... MARKED INVALID (status={status_code}, error={error})")

                    # DYNAMIC REMOVAL: Immediately remove from in-memory pool
                    try:
                        self.config.api_keys.remove(api_key)
                        logger.warning(
                            f"🗑️  Removed API key {api_key[:8]}... from selection pool ({len(self.config.api_keys)} keys remaining)"
                        )
                    except ValueError:
                        pass  # Key already removed
                except Exception as e:
                    logger.error(f"❌ Failed to mark API key as invalid: {e}")
                    import traceback

                    logger.error(traceback.format_exc())

            # Check if this is a quota exhaustion error
            elif self._is_quota_exhausted_error(error):
                # Mark this key as exhausted for this model regardless of our internal counter
                quota.mark_request_failure(error=error, quota_exhausted=True)

                # Persist quota exhaustion to Redis so it's remembered across requests
                try:
                    await ApiKeyQuota.mark_quota_exhausted(api_key, self.config.name, model_name, error)
                except Exception as e:
                    logger.error(f"Failed to persist quota exhaustion to Redis: {e}")

                logger.error(
                    f"🚫 API key {api_key[:8]}... QUOTA EXHAUSTED (429) for {model_name}: Hard marked as depleted"
                )

                # Extract retry delay if available
                retry_delay = self._extract_retry_delay(error)
                if retry_delay:
                    logger.warning(f"🕒 Google suggests retry in {retry_delay}s for {api_key[:8]}... / {model_name}")
                else:
                    logger.warning(f"🕒 Key {api_key[:8]}... / {model_name} exhausted, will reset at midnight Pacific")
            else:
                # Regular error - still count the request
                quota.mark_request_failure(error=error)

                # CRITICAL FIX: Persist error to Redis (was only updating in-memory)
                try:
                    await ApiKeyQuota.mark_error(api_key, self.config.name, model_name, error)
                except Exception as e:
                    logger.error(f"❌ Failed to persist error to Redis: {e}")

                # Log if this looks like it should have been caught
                if status_code == 403:
                    logger.warning(f"⚠️  403 error but not marked invalid - status={status_code}, error={error!r}")

            logger.warning(f"API key {api_key[:8]}... error #{quota.error_count} for {model_name}: {error}")

        # Log quota status if approaching limits for this model
        model_limit = self.get_model_daily_limit(model_name)
        quota_percentage = (quota.requests_today / model_limit) * 100
        if quota_percentage >= 80:
            logger.warning(
                f"API key {api_key[:8]}... / {model_name} approaching daily limit: {quota_percentage:.1f}% used ({quota.requests_today}/{model_limit})"
            )
        elif quota_percentage >= 100:
            logger.error(
                f"🚫 API key {api_key[:8]}... / {model_name} DAILY LIMIT REACHED: {quota.requests_today}/{model_limit}"
            )

    def _is_invalid_key_error(self, error_msg: str = None, status_code: int = None) -> bool:
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

    def _is_quota_exhausted_error(self, error_msg: str) -> bool:
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

    def _extract_retry_delay(self, error_msg: str) -> Optional[float]:
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
                        model_name = model.name.split("/")[-1]  # Extract model name from full path

                        # Create model info
                        model_info = ModelInfo(
                            id=f"{model_name}@{self.get_provider_id()}",
                            name=model_name,
                            provider_id=self.get_provider_id(),
                            provider_type=self.get_provider_type(),
                            endpoint=self.get_endpoint(),
                            aliases=[model_name],
                            metadata={
                                "full_name": model.name,
                                "display_name": getattr(model, "display_name", model_name),
                                "description": getattr(model, "description", ""),
                                "supported_methods": supported_actions,  # Use new supported_actions
                                "input_token_limit": getattr(model, "input_token_limit", None),
                                "output_token_limit": getattr(model, "output_token_limit", None),
                            },
                        )
                        models.append(model_info)
                        logger.debug(f"Discovered Google GenAI model: {model_info.id}")

                # Cache the results
                self._cached_models = models
                self._cache_time = datetime.now()

                logger.info(f"Discovered {len(models)} models from Google GenAI provider {self.get_provider_id()}")
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

        # Convert messages to Google GenAI format
        contents = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            parts = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        parts.append({"text": item.get("text", "")})
                    elif item.get("type") == "image_url":
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

            if not parts:
                continue

            # Map OpenAI roles to Google GenAI roles
            if role == "system":
                # Prepend "System: " to the first text part if it exists
                if parts and "text" in parts[0]:
                    parts[0]["text"] = f"System: {parts[0]['text']}"
                else:
                    parts.insert(0, {"text": "System: "})

                contents.append({"role": "user", "parts": parts})
            elif role == "assistant":
                contents.append({"role": "model", "parts": parts})
            else:  # user
                contents.append({"role": "user", "parts": parts})

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

    def _convert_genai_to_openai_response(self, genai_response: Any, original_model: str) -> Dict[str, Any]:
        """Convert Google GenAI response to OpenAI format"""
        try:
            # Extract text content - new SDK provides direct .text attribute
            text_content = ""
            if hasattr(genai_response, "text") and genai_response.text:
                text_content = genai_response.text
            elif genai_response.candidates:
                # Fallback to candidates structure for compatibility
                candidate = genai_response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    parts = getattr(candidate.content, "parts", None)
                    if parts:
                        for part in parts:
                            if hasattr(part, "text"):
                                text_content += part.text

            # Extract usage information
            usage = {}
            if hasattr(genai_response, "usage_metadata") and genai_response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(genai_response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(genai_response.usage_metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(genai_response.usage_metadata, "total_token_count", 0),
                }

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

    async def generate_completion(
        self, openai_request: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Optional["RequestMetadata"]]:
        """Generate completion using Google GenAI API

        Returns: (response_dict, metadata)
        """
        from .request_metadata import RequestMetadata
        from .transport_observer import get_observer
        import uuid

        original_model = openai_request.get("model", "")

        # Generate observation ID for ground truth tracking
        observation_id = f"obs_{uuid.uuid4().hex[:12]}"
        model_name: Optional[str] = None
        api_key_suffix: Optional[str] = None
        api_key_index: Optional[int] = None
        api_key_total: Optional[int] = None
        proxy_info: Optional[ProxyConfig] = None
        proxy_url: Optional[str] = None
        proxy_pool_index: Optional[int] = None

        def _build_request_error(message: str) -> GoogleGenAIRequestError:
            return GoogleGenAIRequestError(
                message,
                provider_id=self.config.name,
                model_name=model_name or original_model or None,
                api_key_suffix=api_key_suffix,
                api_key_index=api_key_index,
                api_key_total=api_key_total,
                proxy_used=self._proxy_info_to_string(proxy_info),
            )

        try:
            # Convert request format
            model_name, genai_request = self._convert_openai_to_genai_request(openai_request)

            # Select best API key
            api_key = await self._select_best_api_key(model_name)
            api_key_suffix = api_key[-8:] if len(api_key) > 8 else api_key

            # Get key position for debugging/monitoring
            try:
                api_key_index = self.config.api_keys.index(api_key) + 1  # 1-based for display
                api_key_total = len(self.config.api_keys)
            except (ValueError, AttributeError):
                api_key_index = None
                api_key_total = None

            # Create client with proxy transport
            # Check if we should use proxy pool (round-robin) or per-model proxy
            if self._should_use_proxy_pool(model_name):
                proxy_config, proxy_pool_index = self._get_next_proxy_from_pool()
                pool_info = (
                    f" (pool #{(proxy_pool_index or 0) + 1}/{len(self.config.proxy_pool)})" if self.config.proxy_pool else ""
                )
            else:
                proxy_config = self.config.get_proxy_for_model(model_name)
                pool_info = ""
            proxy_info = proxy_config if proxy_config else None
            proxy_url = self._proxy_config_to_url(proxy_info)
            if proxy_url and not self._proxy_available_for_use(proxy_url):
                health = self._proxy_health_snapshot(proxy_url)
                raise _build_request_error(
                    f"Configured proxy {self._mask_proxy_url(proxy_url)} is currently unhealthy"
                    + (f": {health['last_error']}" if health.get("last_error") else "")
                )
            self._schedule_proxy_probe(proxy_url)
            key_position_str = f" key #{api_key_index}/{api_key_total}" if api_key_index else ""
            logger.info(
                f"🚀 Outbound request: model={model_name}, api_key=...{api_key_suffix}{key_position_str}, proxy={proxy_info or 'direct'}{pool_info} [obs={observation_id}]"
            )
            sync_transport, async_transport = self._create_proxy_transport(proxy_config, observation_id)

            from google.genai import types

            if sync_transport and async_transport:
                http_options = types.HttpOptions(
                    client_args={"transport": sync_transport, "trust_env": False},
                    async_client_args={"transport": async_transport, "trust_env": False},
                )
                client = genai.Client(api_key=api_key, http_options=http_options)
                logger.debug("httpx transports active: True (proxy attached)")
                logger.debug("httpx trust_env disabled via HttpOptions: True")
            else:
                client = genai.Client(api_key=api_key)
                logger.debug("httpx transports active: False (no proxy)")
                logger.debug("httpx trust_env disabled via HttpOptions: True")

            # Acquire rate limiting slot (blocks if needed)
            await self._rate_limiter.acquire_slot()

            try:
                # Generate content using new API
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model_name,
                    contents=genai_request["contents"],
                    config=genai_request["generation_config"],
                )
            finally:
                # Always release slot, even on error
                self._rate_limiter.release_slot()

            # Convert response back to OpenAI format
            openai_response = self._convert_genai_to_openai_response(response, original_model)

            # Update API key statistics
            tokens_used = openai_response.get("usage", {}).get("total_tokens", 0)
            await self._update_api_key_stats(api_key, model_name, success=True, tokens=tokens_used)
            if proxy_url:
                self._mark_proxy_health(proxy_url, success=True)

            # Verify ground truth via observer (prefer observation over intent)
            observer = get_observer()
            observation = observer.get_observation(observation_id)

            # Defaults (intent-based) - convert ProxyConfig to string for storage
            actual_key_suffix = api_key_suffix
            actual_proxy = self._proxy_info_to_string(proxy_info)
            key_verified = False
            proxy_verified = False

            if observation:
                # GROUND TRUTH available - use it as primary source
                if observation.api_key_used:
                    # Use observed API key (not intent)
                    actual_key_suffix = (
                        observation.api_key_used[-8:] if len(observation.api_key_used) > 8 else observation.api_key_used
                    )
                    key_verified = True

                    # Verify it matches our intent
                    if actual_key_suffix != api_key_suffix:
                        logger.error(
                            f"❌ API key MISMATCH: intended=...{api_key_suffix}, observed=...{actual_key_suffix}"
                        )
                    else:
                        logger.debug(f"✅ API key verified: ...{actual_key_suffix}")

                # Use observed proxy (not intent)
                if observation.proxy_url:
                    actual_proxy = observation.proxy_url
                    proxy_verified = True

                    # Verify it matches our intent by comparing the actual URL parts
                    # proxy_info is ProxyConfig, observation.proxy_url can be httpx URL object or string
                    if proxy_info:
                        # Extract the intended proxy URL from ProxyConfig
                        intended_proxy_url = proxy_info.https_proxy or proxy_info.http_proxy
                        if intended_proxy_url:
                            # Parse intended URL and compare host:port with observed
                            from urllib.parse import urlparse

                            intended_parsed = urlparse(intended_proxy_url)

                            # Handle both httpx.URL objects and strings
                            try:
                                if isinstance(actual_proxy, str):
                                    # If it's a string, parse it
                                    observed_parsed = urlparse(actual_proxy)
                                    obs_host = observed_parsed.hostname
                                    obs_port = observed_parsed.port
                                else:
                                    # httpx.URL object with host/port attributes
                                    obs_host = (
                                        actual_proxy.host.decode()
                                        if isinstance(actual_proxy.host, bytes)
                                        else actual_proxy.host
                                    )
                                    obs_port = actual_proxy.port

                                if intended_parsed.hostname == obs_host and intended_parsed.port == obs_port:
                                    logger.debug(f"✅ Proxy verified: {obs_host}:{obs_port}")
                                else:
                                    logger.error(
                                        f"❌ Proxy MISMATCH: intended={intended_parsed.hostname}:{intended_parsed.port}, observed={obs_host}:{obs_port}"
                                    )
                            except Exception as e:
                                logger.debug(f"✅ Proxy verification skipped (parse error): {e}")
                        else:
                            logger.debug(f"✅ Proxy verified (no specific URL in config): {actual_proxy}")
                    else:
                        logger.debug(f"✅ Proxy observed (none intended): {actual_proxy}")
                elif not proxy_info:
                    # No proxy in observation, none intended - verified
                    proxy_verified = True
                    logger.debug("✅ No proxy verified (direct connection)")

                logger.info(
                    f"✅ GROUND TRUTH: key=...{actual_key_suffix} (verified={key_verified}), proxy={actual_proxy or 'NONE'} (verified={proxy_verified})"
                )
            else:
                # Observation not available - fallback to intent
                logger.warning(
                    f"⚠️ No observation for {observation_id} - using INTENT: key=...{api_key_suffix}, proxy={proxy_info or 'NONE'}"
                )

            # Create metadata with ground truth (or intent as fallback)
            metadata = RequestMetadata(
                api_key_suffix=actual_key_suffix,
                proxy_used=actual_proxy,
                provider_id=self.config.name,
                model_name=model_name,
                api_key_index=api_key_index if api_key_index else None,
                api_key_total=api_key_total if api_key_total else None,
                api_key_verified=key_verified,
                proxy_verified=proxy_verified,
                observation_id=observation_id,
            )

            return openai_response, metadata

        except ResourceExhausted as e:
            error_msg = f"Google GenAI quota exhausted: {e}"
            # Don't update stats if this is our own "all keys exhausted" exception
            if "api_key" in locals() and "model_name" in locals():
                await self._update_api_key_stats(api_key, model_name, success=False, error=error_msg)

            # Check if this ResourceExhausted has retry timing info
            retry_after_seconds = None
            if hasattr(e, "errors") and e.errors:
                for error_detail in e.errors:
                    if isinstance(error_detail, dict) and "retry_after_seconds" in error_detail:
                        retry_after_seconds = error_detail["retry_after_seconds"]
                        break

            # Create a custom exception that the container can handle with proper HTTP response
            quota_error = _build_request_error(error_msg)
            quota_error.retry_after_seconds = retry_after_seconds
            raise quota_error
        except PermissionDenied as e:
            error_msg = f"Google GenAI permission denied: {e}"
            await self._update_api_key_stats(api_key, model_name, success=False, error=error_msg, status_code=403)
            raise _build_request_error(error_msg)
        except InvalidArgument as e:
            error_msg = f"Google GenAI invalid argument: {e}"
            # InvalidArgument can indicate API key issues
            await self._update_api_key_stats(api_key, model_name, success=False, error=error_msg, status_code=400)
            raise _build_request_error(error_msg)
        except Exception as e:
            error_msg = f"Google GenAI error: {e}"
            if proxy_url and self._is_proxy_connectivity_error(e):
                self._mark_proxy_health(proxy_url, success=False, error=str(e))
            if "api_key" in locals() and "model_name" in locals():
                # Extract status code from error message if present
                # Handles cases where exception isn't caught by specific handlers
                extracted_status = None
                error_str = str(e).lower()
                if "403" in error_str or "permission" in error_str:
                    extracted_status = 403
                elif "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                    extracted_status = 429
                elif "401" in error_str or "unauthorized" in error_str:
                    extracted_status = 401

                await self._update_api_key_stats(
                    api_key, model_name, success=False, error=error_msg, status_code=extracted_status
                )
            raise _build_request_error(error_msg)

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

            stats[key_display]["models"][model] = {
                "requests_today": quota.requests_today,
                "tokens_today": quota.tokens_today,
                "error_count": quota.error_count,
                "last_request": quota.updated_at.isoformat()
                if quota.updated_at and hasattr(quota.updated_at, "isoformat")
                else str(quota.updated_at)
                if quota.updated_at
                else None,
                "last_error": quota.last_error,
                "quota_percentage": quota_percentage,
                "quota_exhausted": is_exhausted,
                "quota_exhausted_at": quota.quota_exhausted_at.isoformat()
                if quota.quota_exhausted_at and hasattr(quota.quota_exhausted_at, "isoformat")
                else str(quota.quota_exhausted_at)
                if quota.quota_exhausted_at
                else None,
                "status": "invalid" if quota.invalid_key else ("exhausted" if is_exhausted else "available"),
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
