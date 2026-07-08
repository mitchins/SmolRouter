"""
Google Generative AI provider implementation.

Provides integration with Google's Generative AI API, supporting multiple API keys
with intelligent rotation based on requests-per-day (RPD) quotas.
"""

import logging
import json
import asyncio
import base64
import io
import re
import secrets
import wave
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
import httpx

from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted, PermissionDenied, InvalidArgument

from .config_loading import load_config_entries
from .interfaces import IModelProvider, ModelInfo, ProviderConfig, ProxyConfig
from .secret_store import redact_secret, resolve_config_file, secrets_search_paths
from .database import ApiKeyQuota
from .redis_backend import QuotaRecord
from .rate_limiter import GoogleGenAIRequestFunnel
from .request_metadata import RequestMetadata
from .task_utils import create_logged_task
from .transport_observer import get_observer

logger = logging.getLogger(__name__)

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")
OPENAI_CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"
OPENAI_COMPLETIONS_ENDPOINT = "/v1/completions"
OPENAI_RESPONSES_ENDPOINT = "/v1/responses"
OPENAI_EMBEDDINGS_ENDPOINT = "/v1/embeddings"
OPENAI_IMAGES_ENDPOINT = "/v1/images/generations"
GOOGLE_GENAI_IMAGE_ENDPOINT = OPENAI_IMAGES_ENDPOINT
GOOGLE_OPENAI_IMAGES_GENERATION_PATH = "/v1beta/openai/images/generations"
GOOGLE_IMAGEN_PREDICT_PATH = "/v1beta/models/{model}:predict"
GOOGLE_GENAI_STATIC_MODELS_SNAPSHOT = "google-genai-models-2026-july.json"
GOOGLE_GENAI_IMAGE_MODEL_ALLOWLIST = {
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",
    "gemini-3-pro-image",
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
}
GOOGLE_IMAGEN_IMAGE_MODEL_ALLOWLIST = {
    "imagen-4.0-generate-001",
    "imagen-4.0-ultra-generate-001",
    "imagen-4.0-fast-generate-001",
}
GOOGLE_IMAGEN_SIZE_TO_ASPECT_RATIO = {
    "1024x1024": "1:1",
    "1024x1536": "3:4",
    "1536x1024": "4:3",
    "1024x1792": "9:16",
    "1792x1024": "16:9",
}
GOOGLE_IMAGEN_SUPPORTED_ASPECT_RATIOS = {"1:1", "3:4", "4:3", "9:16", "16:9"}
GOOGLE_IMAGEN_SUPPORTED_IMAGE_SIZES = {"1K", "2K"}
GOOGLE_IMAGEN_SUPPORTED_OUTPUT_MIME_TYPES = {"image/png", "image/jpeg"}
GOOGLE_IMAGEN_ALLOWED_PARAMETER_NAMES = {
    "aspectRatio",
    "guidanceScale",
    "imageSize",
    "includeRaiReason",
    "includeSafetyAttributes",
    "language",
    "negativePrompt",
    "outputOptions",
    "personGeneration",
    "safetySetting",
}
GOOGLE_IMAGE_RESPONSE_FORMATS = {"url", "b64_json"}
OPENAI_IMAGE_REQUEST_FIELDS = {"model", "prompt", "n", "size", "response_format", "extra_body", "user"}
GOOGLE_NON_INVALID_403_INDICATORS = (
    "paid plan",
    "paid plans",
    "upgrade your account",
    "billing",
    "billed user",
    "not supported for predict",
    "forbidden for project",
    "model is not found",
)
GOOGLE_INVALID_KEY_INDICATORS = (
    "permission denied",
    "permission_denied",
    "permissiondenied",
    "api key not valid",
    "invalid api key",
    "api_key_invalid",
    "authentication failed",
    "unauthorized",
    "credentials are missing or invalid",
    "api key expired",
)
GOOGLE_FALLBACK_INVALID_KEY_INDICATORS = GOOGLE_INVALID_KEY_INDICATORS + ("invalid_argument",)
AUDIO_WAV_MIME_TYPE = "audio/wav"
AUDIO_PCM_MIME_TYPE = "audio/pcm"
AUDIO_MPEG_MIME_TYPE = "audio/mpeg"
TTS_DEFAULT_VOICE = "Kore"
TTS_SUPPORTED_FORMATS = {"wav", "pcm"}
TTS_SUPPORTED_ENDPOINTS = {OPENAI_CHAT_COMPLETIONS_ENDPOINT, OPENAI_RESPONSES_ENDPOINT}
TTS_MAX_SPEAKERS = 2
TTS_SAMPLE_RATE_HZ = 24000
TTS_CHANNELS = 1
TTS_SAMPLE_WIDTH_BYTES = 2
TTS_RECOMMENDED_CHUNKING = "split transcripts longer than a few minutes"


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
        return f"{redact_secret(self.api_key)}/{self.model}"

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
        logger.info(f"Reset daily stats for API key {redact_secret(self.api_key)}")


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

    url: str = "https://generativelanguage.googleapis.com"
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

    def __post_init__(self):
        super().__post_init__()
        self.api_keys = list(self.api_keys or [])

        if not self.api_keys and not self.api_keys_file:
            resolve_config_file("secrets.yaml", "SMOLROUTER_SECRETS")
            raise ValueError(
                f"Provider '{self.name}' has no API keys. "
                f"Looked in: {', '.join(secrets_search_paths())}"
            )

        if self.api_keys_file:
            try:
                file_keys = load_config_entries(
                    self.api_keys_file,
                    allow_assignments=True,
                    strip_inline_comments=True,
                )
                self.api_keys.extend(file_keys)
                logger.info(f"Loaded {len(file_keys)} API keys from {self.api_keys_file}")
            except Exception:
                logger.exception("Failed to load API keys from %s", self.api_keys_file)
                raise

        if not self.api_keys:
            raise ValueError("No valid API keys found")


@dataclass
class GoogleGenAICompletionContext:
    original_model: str
    observation_id: str
    endpoint: str = OPENAI_CHAT_COMPLETIONS_ENDPOINT
    request_kind: str = "generate_content"
    model_name: str = ""
    tts_request: bool = False
    tts_audio_format: str = "pcm"
    genai_request: Dict[str, Any] = field(default_factory=dict)
    input_token_count: Optional[int] = None
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

    # Per-minute (RPM) 429s are transient: bench the key for a short cooldown driven
    # by Google's RetryInfo, NOT for the whole Pacific day (which is what RPD/daily
    # exhaustion does). Used when the upstream error does not tell us how long to wait.
    DEFAULT_RPM_COOLDOWN_SECONDS = 60
    # Small buffer added on top of Google's suggested retryDelay to avoid racing the
    # edge of the rate-limit window and immediately re-tripping it.
    RPM_COOLDOWN_BUFFER_SECONDS = 2

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

        # Proxy pool round-robin counter
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

        def _cleanup_probe(done_task: asyncio.Task, url: str = proxy_url):
            if self._proxy_probe_tasks.get(url) is done_task:
                self._proxy_probe_tasks.pop(url, None)
            if done_task.cancelled():
                return
            try:
                done_task.result()
            except Exception as exc:
                logger.debug("Proxy health probe failed for %s: %s", self._mask_proxy_url(url), exc)

        task = create_logged_task(
            self._probe_proxy_url(proxy_url),
            task_name=f"google-proxy-probe:{self._mask_proxy_url(proxy_url)}",
            create_task_fn=loop.create_task,
            done_callback=_cleanup_probe,
        )
        self._proxy_probe_tasks[proxy_url] = task

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
        except Exception:
            logger.exception("Proxy health monitor stopped unexpectedly for %s", self.config.name)

    def start_proxy_health_monitor(self):
        if not self._configured_proxy_urls():
            return

        if self._proxy_health_task and not self._proxy_health_task.done():
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        self._proxy_health_task = create_logged_task(
            self._proxy_health_monitor_loop(),
            task_name=f"google-proxy-health-monitor:{self.config.name}",
            create_task_fn=loop.create_task,
            service=True,
        )
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
            ("imagen-4." in model_lower, 25),
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
            masked_proxy_url = self._mask_proxy_url(proxy_url) or proxy_url
            logger.debug(f"🔀 Using proxy URL for Google GenAI: {masked_proxy_url}")

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
        """Select the next eligible API key in serial rotary order for this model."""
        selection = await ApiKeyQuota.select_google_api_key(
            provider_id=self.config.name,
            model_name=model_name,
            api_keys=self.config.api_keys,
            model_limit=self.get_model_daily_limit(model_name),
        )

        if selection["status"] == "ok":
            selected_index = selection.get("selected_index")
            if selected_index is None or not (0 <= selected_index < len(self.config.api_keys)):
                raise RuntimeError(
                    f"Redis returned invalid Google key index for {self.config.name}/{model_name}: {selected_index}"
                )

            best_key = self.config.api_keys[selected_index]
            logger.debug(
                "Selected API key %s for %s via Redis rotary slot %s/%s",
                redact_secret(best_key),
                model_name,
                selected_index + 1,
                len(self.config.api_keys),
            )
            return best_key

        # Under rotary selection, request-path exclusion should stay narrow: explicit invalid keys,
        # active per-minute cooldowns, and observed same-day exhaustion. Favor availability over
        # aggressive predictive benching from dead-reckoned counters.
        if selection["status"] in {"no_keys", "all_invalid"}:
            raise RuntimeError(
                f"No usable Google API keys available for provider {self.config.name} / model {model_name} "
                f"(status={selection['status']}, invalid={int(selection.get('invalid_count') or 0)})"
            )

        total_keys = len(self.config.api_keys)
        invalid_count = int(selection.get("invalid_count") or 0)
        cooling_down_count = int(selection.get("cooling_down_count") or 0)
        exhausted_count = int(selection.get("exhausted_count") or 0)
        retry_after = int(selection.get("retry_after_seconds") or 0)

        logger.error(f"🚫 ALL {total_keys} API KEYS UNAVAILABLE FOR MODEL {model_name}:")
        logger.error(f"   - Invalid: {invalid_count} keys")
        logger.error(f"   - Quota exhausted: {exhausted_count} keys")
        logger.error(f"   - Cooling down (per-minute): {cooling_down_count} keys")

        if cooling_down_count > 0:
            logger.warning(
                "⏳ All keys busy for %s; soonest key recovers in ~%ss (per-minute cooldown)",
                model_name,
                retry_after,
            )
        else:
            logger.error(f"⏰ All API keys exhausted. Quota resets in {retry_after}s at midnight Pacific")

        raise ResourceExhausted(
            f"All {total_keys} API keys exhausted for model {model_name}. "
            f"Retry in {retry_after} seconds.",
            errors=[{"reason": "QUOTA_EXHAUSTED", "retry_after_seconds": retry_after}],
        )

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
                cooldown_seconds = self._quota_cooldown_seconds(error_message)
                if cooldown_seconds is not None:
                    # Transient per-minute (RPM) limit: short cooldown, not an all-day bench.
                    await self._record_rate_limit_cooldown(
                        ApiKeyQuota, quota, api_key, model_name, error_message, cooldown_seconds
                    )
                else:
                    # Per-day (RPD) / unrecognized quota exhaustion: bench until midnight Pacific.
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
            logger.debug(
                "API key %s successful request for %s: %s/%s RPD, %s tokens",
                redact_secret(api_key),
                model_name,
                quota.requests_today,
                self.get_model_daily_limit(model_name),
                tokens,
            )
        except Exception:
            logger.exception(
                "❌ CRITICAL: Failed to update quota for %s / %s", redact_secret(api_key), model_name
            )
            logger.error("⚠️  Quota tracking broken - key rotation will not work correctly!")
            quota.mark_request_success(tokens=tokens)

    async def _record_invalid_api_key(
        self, quota_backend: Any, api_key: str, error_message: str, status_code: Optional[int]
    ) -> None:
        key_hash = quota_backend.hash_api_key(api_key)
        try:
            await quota_backend.mark_invalid_by_hash(key_hash, self.config.name)
            logger.error(f"🚫 API key {redact_secret(api_key)} MARKED INVALID (status={status_code}, error={error_message})")

            try:
                self.config.api_keys.remove(api_key)
                logger.warning(
                    f"🗑️  Removed API key {redact_secret(api_key)} from selection pool ({len(self.config.api_keys)} keys remaining)"
                )
            except ValueError:
                pass
        except Exception:
            logger.exception("❌ Failed to mark API key as invalid")

    async def _record_rate_limit_cooldown(
        self,
        quota_backend: Any,
        quota: QuotaRecord,
        api_key: str,
        model_name: str,
        error_message: str,
        cooldown_seconds: float,
    ) -> None:
        """Handle a transient per-minute (RPM) 429.

        Sets a short cooldown window (driven by Google's RetryInfo) after which the key
        returns to the selection pool, instead of benching it until midnight Pacific the
        way genuine per-day (RPD) exhaustion does. This is the fix for RPM 429s cascading
        into "all keys exhausted" and pinning every request onto api_keys[0].
        """
        cooldown_until = self._utc_now() + timedelta(seconds=cooldown_seconds)
        quota.mark_rate_limited(cooldown_until, error=error_message)

        try:
            await quota_backend.mark_quota_cooldown(
                api_key, self.config.name, model_name, cooldown_until, error_message
            )
        except Exception:
            logger.exception("Failed to persist quota cooldown to Redis")

        logger.warning(
            "🕒 API key %s rate limited (per-minute) for %s: cooling down %.0fs (until %s)",
            redact_secret(api_key),
            model_name,
            cooldown_seconds,
            cooldown_until.isoformat(),
        )

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
        except Exception:
            logger.exception("Failed to persist quota exhaustion to Redis")

        logger.error(f"🚫 API key {redact_secret(api_key)} QUOTA EXHAUSTED (429) for {model_name}: Hard marked as depleted")

        retry_delay = self._extract_retry_delay(error_message)
        if retry_delay:
            logger.warning(f"🕒 Google suggests retry in {retry_delay}s for {redact_secret(api_key)} / {model_name}")
        else:
            logger.warning(f"🕒 Key {redact_secret(api_key)} / {model_name} exhausted, will reset at midnight Pacific")

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
        except Exception:
            logger.exception("Failed to persist error to Redis")

        if status_code == 403:
            logger.warning(f"⚠️  403 error but not marked invalid - status={status_code}, error={error_message!r}")

        logger.warning(f"API key {redact_secret(api_key)} error #{quota.error_count} for {model_name}: {error_message}")

    def _log_quota_status(self, api_key: str, model_name: str, quota: QuotaRecord) -> None:
        model_limit = self.get_model_daily_limit(model_name)
        if model_limit <= 0:
            logger.warning("API key %s / %s has non-positive daily limit: %s", redact_secret(api_key), model_name, model_limit)
            return

        quota_percentage = (quota.requests_today / model_limit) * 100
        if quota.requests_today >= model_limit:
            logger.error(
                f"🚫 API key {redact_secret(api_key)} / {model_name} DAILY LIMIT REACHED: {quota.requests_today}/{model_limit}"
            )
        elif quota.requests_today >= (model_limit * 0.8):
            logger.warning(
                "API key %s / %s approaching daily limit: %.1f%% used (%s/%s)",
                redact_secret(api_key),
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
        if status_code == 403:
            if error_msg:
                error_lower = error_msg.lower()
                if any(indicator in error_lower for indicator in GOOGLE_NON_INVALID_403_INDICATORS):
                    return False
                return any(indicator in error_lower for indicator in GOOGLE_INVALID_KEY_INDICATORS)
            return True

        # Check error message for known invalid key indicators
        if error_msg:
            error_lower = error_msg.lower()
            return any(indicator in error_lower for indicator in GOOGLE_FALLBACK_INVALID_KEY_INDICATORS)

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

    def _is_per_minute_quota_error(self, error_msg: Optional[str] = None) -> bool:
        """Check if a quota 429 is a per-minute (RPM) limit rather than per-day (RPD).

        Google returns the same quotaMetric (generate_content_free_tier_requests) for
        both RPM and RPD on the free tier; the only reliable discriminator is the
        quotaId, e.g. ``GenerateRequestsPerMinutePerProjectPerModel-FreeTier``.
        """
        if not error_msg:
            return False
        error_lower = error_msg.lower()
        return "perminute" in error_lower or "per minute" in error_lower

    def _is_per_day_quota_error(self, error_msg: Optional[str] = None) -> bool:
        """Check if a quota 429 is an explicit per-day (RPD) limit."""
        if not error_msg:
            return False
        error_lower = error_msg.lower()
        return "perday" in error_lower or "per day" in error_lower or "requests per day" in error_lower

    def _quota_cooldown_seconds(self, error_msg: Optional[str] = None) -> Optional[float]:
        """Return a transient cooldown (seconds) for a quota 429, or None for all-day exhaustion.

        - Explicit per-minute (RPM) → cooldown driven by Google's retryDelay (or default).
        - Everything else (explicit per-day, or no PerMinute quotaId) → None, i.e. the
          caller treats it as all-day exhaustion. Real Google free-tier 429s always carry
          a quotaId, so RPM is reliably detectable; we stay conservative for anything else.
        """
        if not self._is_per_minute_quota_error(error_msg):
            return None
        if self._is_per_day_quota_error(error_msg):
            return None

        retry_delay = self._extract_retry_delay(error_msg)
        base = retry_delay if retry_delay is not None else self.DEFAULT_RPM_COOLDOWN_SECONDS
        return base + self.RPM_COOLDOWN_BUFFER_SECONDS

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

    def _build_google_model_info(self, model_name: str, metadata: Dict[str, Any]) -> ModelInfo:
        return ModelInfo(
            id=f"{model_name}@{self.get_provider_id()}",
            name=model_name,
            provider_id=self.get_provider_id(),
            provider_type=self.get_provider_type(),
            endpoint=self.get_endpoint(),
            aliases=[model_name],
            metadata=metadata,
        )

    @staticmethod
    def _extract_google_model_name(full_name: str) -> str:
        return full_name.split("/")[-1] if "/" in full_name else full_name

    def _build_live_google_model_metadata(self, model: Any, supported_actions: List[str]) -> Dict[str, Any]:
        model_name = self._extract_google_model_name(getattr(model, "name", "") or "")
        metadata = {
            "full_name": getattr(model, "name", ""),
            "display_name": getattr(model, "display_name", model_name),
            "description": getattr(model, "description", ""),
            "supported_methods": supported_actions,
            "supports_embeddings": "embedContent" in supported_actions,
            "input_token_limit": getattr(model, "input_token_limit", None),
            "output_token_limit": getattr(model, "output_token_limit", None),
        }
        return self._augment_google_model_metadata(model_name, metadata)

    def _build_static_google_model_metadata(self, model_data: Dict[str, Any], supported_methods: List[str]) -> Dict[str, Any]:
        model_name = self._extract_google_model_name(model_data.get("name", ""))
        metadata = {
            "full_name": model_data.get("name", ""),
            "display_name": model_data.get("displayName", model_name),
            "description": model_data.get("description", ""),
            "supported_methods": supported_methods,
            "supports_embeddings": "embedContent" in supported_methods,
            "input_token_limit": model_data.get("inputTokenLimit"),
            "output_token_limit": model_data.get("outputTokenLimit"),
            "version": model_data.get("version"),
            "temperature": model_data.get("temperature"),
            "top_p": model_data.get("topP"),
            "top_k": model_data.get("topK"),
            "max_temperature": model_data.get("maxTemperature"),
            "thinking": model_data.get("thinking", False),
        }
        for key in (
            "supports_tts",
            "output_modalities",
            "audio_format",
            "sample_rate_hz",
            "channels",
            "sample_width_bytes",
            "max_context_tokens",
            "recommended_chunking",
        ):
            if key in model_data:
                metadata[key] = model_data[key]
        return self._augment_google_model_metadata(model_name, metadata)

    def _augment_google_model_metadata(self, model_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_tts_model_name(model_name):
            return metadata

        metadata.setdefault("supports_tts", True)
        metadata.setdefault("output_modalities", ["audio"])
        metadata.setdefault("audio_format", "pcm")
        metadata.setdefault("sample_rate_hz", TTS_SAMPLE_RATE_HZ)
        metadata.setdefault("channels", TTS_CHANNELS)
        metadata.setdefault("sample_width_bytes", TTS_SAMPLE_WIDTH_BYTES)
        input_token_limit = metadata.get("input_token_limit")
        if input_token_limit is not None:
            metadata.setdefault("max_context_tokens", input_token_limit)
        metadata.setdefault("recommended_chunking", TTS_RECOMMENDED_CHUNKING)
        return metadata

    @staticmethod
    def _is_tts_model_name(model_name: str) -> bool:
        normalized_name = GoogleGenAIProvider._extract_google_model_name(model_name).lower()
        return normalized_name in {
            "gemini-3.1-flash-tts-preview",
            "gemini-2.5-flash-preview-tts",
            "gemini-2.5-pro-preview-tts",
        }

    def _create_genai_client(self, api_key: str, sync_transport: Any, async_transport: Any) -> Any:
        http_options_kwargs = {
            "client_args": {"trust_env": False},
            "async_client_args": {"trust_env": False},
        }

        if sync_transport and async_transport:
            http_options_kwargs["client_args"]["transport"] = sync_transport
            http_options_kwargs["async_client_args"]["transport"] = async_transport
            logger.debug("httpx transports active: True (proxy attached)")
        else:
            logger.debug("httpx transports active: False (no proxy)")

        logger.debug("httpx trust_env disabled via HttpOptions: True")
        http_options = types.HttpOptions(**http_options_kwargs)
        return genai.Client(api_key=api_key, http_options=http_options)

    async def _health_check_api_key(self, api_key: str) -> bool:
        proxy_config = self.config.get_proxy_for_model("health-check")
        sync_transport, async_transport = self._create_proxy_transport(proxy_config)
        client = self._create_genai_client(api_key, sync_transport, async_transport)

        models = await asyncio.to_thread(client.models.list)
        list(models)
        return True

    async def _discover_models_with_api_key(self, api_key: str) -> List[ModelInfo]:
        proxy_config = self.config.get_proxy_for_model("model-discovery")
        sync_transport, async_transport = self._create_proxy_transport(proxy_config)
        client = self._create_genai_client(api_key, sync_transport, async_transport)

        models = []
        model_list = await asyncio.to_thread(client.models.list)
        for model in model_list:
            # New API uses 'supported_actions' instead of 'supported_generation_methods'
            supported_actions = getattr(model, "supported_actions", [])
            model_name = self._extract_google_model_name(getattr(model, "name", "") or "")
            if not self._is_google_supported_action_model(model_name, supported_actions):
                continue

            model_info = self._build_google_model_info(
                model_name,
                self._build_live_google_model_metadata(model, supported_actions),
            )
            models.append(model_info)
            logger.debug(f"Discovered Google GenAI model: {model_info.id}")

        return models

    async def health_check(self) -> bool:
        """Check if at least one API key is working"""
        for api_key in self.config.api_keys:
            try:
                if await self._health_check_api_key(api_key):
                    return True
            except Exception as exc:
                logger.debug("Health check failed for key %s: %s", redact_secret(api_key), exc)
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
                models = await self._discover_models_with_api_key(api_key)

                # Cache the results
                self._cached_models = models
                self._cache_time = datetime.now()

                return models

            except Exception as exc:
                logger.debug("Error discovering models with API key %s: %s", redact_secret(api_key), exc)
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
            models_file = current_dir / "models" / GOOGLE_GENAI_STATIC_MODELS_SNAPSHOT

            if not models_file.exists():
                logger.error(f"Static Google GenAI models file not found: {models_file}")
                return []

            with open(models_file, "r") as f:
                data = json.load(f)

            models = []
            for model_data in data.get("models", []):
                supported_methods = model_data.get("supportedGenerationMethods") or model_data.get("supportedActions", [])
                model_name = self._extract_google_model_name(model_data.get("name", ""))
                if not self._is_google_supported_action_model(model_name, supported_methods):
                    continue
                model_info = self._build_google_model_info(
                    model_name,
                    self._build_static_google_model_metadata(model_data, supported_methods),
                )
                models.append(model_info)
                logger.debug(f"Static Google GenAI model: {model_info.id}")

            logger.info(f"Loaded {len(models)} static models for Google GenAI provider {self.get_provider_id()}")
            return models

        except Exception:
            logger.exception("Error loading static Google GenAI models")
            return []

    def _convert_openai_to_genai_request(
        self,
        openai_request: Dict[str, Any],
        endpoint: str = OPENAI_CHAT_COMPLETIONS_ENDPOINT,
    ) -> Tuple[str, Dict[str, Any]]:
        """Convert OpenAI request format to Google GenAI format"""
        if endpoint == OPENAI_EMBEDDINGS_ENDPOINT:
            model_name = self._normalize_model_name(openai_request.get("model", ""))
            contents = openai_request.get("input")
            if contents is None:
                raise ValueError("No input provided in request")

            embed_config: Dict[str, Any] = {}
            if openai_request.get("dimensions") is not None:
                embed_config["output_dimensionality"] = openai_request["dimensions"]
            if openai_request.get("task_type") is not None:
                embed_config["task_type"] = openai_request["task_type"]
            if openai_request.get("title") is not None:
                embed_config["title"] = openai_request["title"]
            if openai_request.get("mime_type") is not None:
                embed_config["mime_type"] = openai_request["mime_type"]
            if openai_request.get("auto_truncate") is not None:
                embed_config["auto_truncate"] = openai_request["auto_truncate"]

            return model_name, {"contents": contents, "config": embed_config or None}

        request_payload = (
            self._convert_openai_responses_to_chat_request(openai_request)
            if endpoint == OPENAI_RESPONSES_ENDPOINT
            else openai_request
        )
        model_name = self._normalize_model_name(request_payload.get("model", ""))

        if self._is_tts_request(openai_request):
            self._validate_tts_request(openai_request, endpoint)
            tts_text = self._extract_tts_text(openai_request, endpoint)
            return model_name, {
                "contents": [{"role": "user", "parts": [{"text": tts_text}]}],
                "generation_config": self._build_tts_generation_config(openai_request),
            }

        # Extract messages
        messages = request_payload.get("messages", [])
        if not messages:
            raise ValueError("No messages provided in request")

        contents = self._convert_openai_messages_to_genai_contents(messages)
        generation_config = self._build_generation_config(request_payload)
        if "top_p" in request_payload:
            generation_config["top_p"] = request_payload["top_p"]

        # Google GenAI doesn't support streaming in the same way, so we'll handle that separately
        genai_request = {"contents": contents, "generation_config": generation_config}

        return model_name, genai_request

    def _convert_openai_messages_to_genai_contents(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        contents = []
        for message in messages:
            genai_content = self._convert_openai_message_to_genai_content(message)
            if genai_content is not None:
                contents.append(genai_content)
        return contents

    def _build_generation_config(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        generation_config = {}
        if "temperature" in request_payload:
            generation_config["temperature"] = request_payload["temperature"]

        max_output_tokens = self._extract_max_output_tokens(request_payload)
        if max_output_tokens is not None:
            generation_config["max_output_tokens"] = max_output_tokens

        return generation_config

    @staticmethod
    def _extract_max_output_tokens(request_payload: Dict[str, Any]) -> Optional[int]:
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            if key in request_payload:
                return request_payload[key]
        return None

    def _is_tts_request(self, request_payload: Dict[str, Any]) -> bool:
        modalities = request_payload.get("modalities")
        if isinstance(modalities, list) and any(str(modality).strip().lower() == "audio" for modality in modalities):
            return True

        response_format = request_payload.get("response_format")
        if isinstance(response_format, dict) and str(response_format.get("type", "")).strip().lower() == "audio":
            return True

        return isinstance(request_payload.get("audio"), dict)

    @staticmethod
    def _get_tts_audio_config(request_payload: Dict[str, Any]) -> Dict[str, Any]:
        audio_config = request_payload.get("audio")
        return audio_config if isinstance(audio_config, dict) else {}

    def _get_tts_audio_format(self, request_payload: Dict[str, Any]) -> str:
        requested_format = str(self._get_tts_audio_config(request_payload).get("format") or "pcm").strip().lower()
        return requested_format or "pcm"

    @staticmethod
    def _resolve_tts_voice(voice_name: Any, fallback: str = TTS_DEFAULT_VOICE) -> str:
        normalized = str(voice_name or "").strip()
        return normalized or fallback

    @staticmethod
    def _normalize_tts_modalities(request_payload: Dict[str, Any]) -> set[str]:
        modalities = request_payload.get("modalities")
        if not isinstance(modalities, list):
            return set()
        return {str(modality).strip().lower() for modality in modalities if str(modality).strip()}

    def _validate_tts_modalities(self, request_payload: Dict[str, Any]) -> None:
        normalized_modalities = self._normalize_tts_modalities(request_payload)
        if normalized_modalities and normalized_modalities != {"audio"}:
            raise ValueError("400 invalid argument: TTS requests currently support audio-only output")

    def _validate_tts_audio_settings(self, request_payload: Dict[str, Any], endpoint: str) -> None:
        if request_payload.get("stream"):
            raise ValueError(f"400 invalid argument: streaming is not supported for TTS requests on {endpoint}")

        requested_format = self._get_tts_audio_format(request_payload)
        if requested_format not in TTS_SUPPORTED_FORMATS:
            raise ValueError(
                f"400 invalid argument: unsupported TTS audio format '{requested_format}'. Supported formats: wav, pcm"
            )

    @staticmethod
    def _validate_tts_speakers(speakers: Any) -> None:
        if speakers is None:
            return

        if not isinstance(speakers, list) or not speakers:
            raise ValueError("400 invalid argument: audio.speakers must be a non-empty list")

        if len(speakers) > TTS_MAX_SPEAKERS:
            raise ValueError("400 invalid argument: Google GenAI TTS supports at most 2 speakers")

        for speaker_config in speakers:
            if not isinstance(speaker_config, dict):
                raise ValueError("400 invalid argument: each TTS speaker entry must be an object")

            if not str(speaker_config.get("speaker", "")).strip():
                raise ValueError("400 invalid argument: each TTS speaker entry must include a speaker name")

    def _validate_tts_request(self, request_payload: Dict[str, Any], endpoint: str) -> None:
        if endpoint not in TTS_SUPPORTED_ENDPOINTS:
            raise ValueError(
                "501 not implemented: audio-output TTS is only supported for "
                f"{OPENAI_CHAT_COMPLETIONS_ENDPOINT} and {OPENAI_RESPONSES_ENDPOINT}"
            )

        self._validate_tts_modalities(request_payload)
        self._validate_tts_audio_settings(request_payload, endpoint)
        self._validate_tts_speakers(self._get_tts_audio_config(request_payload).get("speakers"))
        self._extract_tts_text(request_payload, endpoint)

    def _extract_tts_text(self, request_payload: Dict[str, Any], endpoint: str) -> str:
        normalized_payload = (
            self._convert_openai_responses_to_chat_request(request_payload)
            if endpoint == OPENAI_RESPONSES_ENDPOINT
            else request_payload
        )
        messages = normalized_payload.get("messages", [])
        if not isinstance(messages, list) or not messages:
            raise ValueError("400 invalid argument: TTS requests require a text prompt or transcript")

        transcript_segments = []
        for message in messages:
            transcript_segment = self._extract_tts_text_from_message(message)
            if transcript_segment:
                transcript_segments.append(transcript_segment)

        transcript = "\n\n".join(transcript_segments).strip()
        if not transcript:
            raise ValueError("400 invalid argument: TTS requests require a text prompt or transcript")

        return transcript

    def _extract_tts_text_from_message(self, message: Any) -> str:
        if not isinstance(message, dict):
            return ""

        role = str(message.get("role") or "user").strip().lower()
        content = message.get("content", "")

        if isinstance(content, str):
            return self._label_tts_transcript_segment(role, content)

        if not isinstance(content, list):
            return ""

        text_parts = []
        for item in content:
            if not isinstance(item, dict):
                continue

            item_type = str(item.get("type", "")).strip().lower()
            if item_type in {"text", "input_text"}:
                text_parts.append(str(item.get("text") or ""))
                continue

            if item_type in {"image_url", "input_image", "input_audio"}:
                raise ValueError("400 invalid argument: TTS requests only support text input")

        return self._label_tts_transcript_segment(role, "\n".join(part for part in text_parts if part))

    @staticmethod
    def _label_tts_transcript_segment(role: str, text: str) -> str:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return ""

        if role == "system":
            return f"System: {normalized_text}"
        if role == "assistant":
            return f"Assistant: {normalized_text}"
        if role not in {"user", "system", "assistant"}:
            return f"{role.capitalize()}: {normalized_text}"
        return normalized_text

    def _build_tts_generation_config(self, request_payload: Dict[str, Any]) -> types.GenerateContentConfig:
        audio_config = self._get_tts_audio_config(request_payload)
        default_voice = self._resolve_tts_voice(audio_config.get("voice"))
        config_kwargs = self._build_generation_config(request_payload)
        if "top_p" in request_payload:
            config_kwargs["top_p"] = request_payload["top_p"]

        speakers = audio_config.get("speakers")
        if speakers:
            speaker_voice_configs = []
            for speaker_config in speakers:
                voice_name = self._resolve_tts_voice(speaker_config.get("voice"), default_voice)
                speaker_voice_configs.append(
                    types.SpeakerVoiceConfig(
                        speaker=speaker_config.get("speaker"),
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                        ),
                    )
                )
            speech_config = types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=speaker_voice_configs
                )
            )
        else:
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=default_voice)
                )
            )

        config_kwargs["response_modalities"] = ["AUDIO"]
        config_kwargs["speech_config"] = speech_config
        return types.GenerateContentConfig(**config_kwargs)

    def _convert_openai_responses_to_chat_request(self, openai_request: Dict[str, Any]) -> Dict[str, Any]:
        converted_request = dict(openai_request)
        messages = self._convert_responses_input_to_messages(openai_request.get("input"))
        messages = self._prepend_responses_instructions(messages, openai_request.get("instructions"))
        converted_request["messages"] = messages

        if "max_output_tokens" in openai_request and "max_completion_tokens" not in converted_request:
            converted_request["max_completion_tokens"] = openai_request["max_output_tokens"]

        return converted_request

    @staticmethod
    def _prepend_responses_instructions(messages: List[Dict[str, Any]], instructions: Any) -> List[Dict[str, Any]]:
        if isinstance(instructions, str) and instructions:
            return [{"role": "system", "content": instructions}, *messages]
        return messages

    def _convert_responses_input_to_messages(self, input_data: Any) -> List[Dict[str, Any]]:
        if isinstance(input_data, str):
            return [{"role": "user", "content": input_data}]

        if not isinstance(input_data, list):
            return []

        messages: List[Dict[str, Any]] = []
        for item in input_data:
            message = self._convert_responses_input_item_to_message(item)
            if message is not None:
                messages.append(message)

        return messages

    def _convert_responses_input_item_to_message(self, item: Any) -> Optional[Dict[str, Any]]:
        if isinstance(item, str):
            return {"role": "user", "content": item}

        if not isinstance(item, dict):
            return None

        role = item.get("role", "user")
        if "content" in item:
            return self._convert_responses_content_message(role, item.get("content", ""))

        converted_item = self._convert_responses_content_item_to_chat_item(item)
        if converted_item is not None:
            return {"role": role, "content": [converted_item]}

        return None

    def _convert_responses_content_message(self, role: str, content: Any) -> Optional[Dict[str, Any]]:
        if isinstance(content, str):
            return {"role": role, "content": content}

        converted_content = self._convert_responses_content_to_chat_content(content)
        if converted_content:
            return {"role": role, "content": converted_content}

        return None

    def _convert_responses_content_to_chat_content(self, content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        if not isinstance(content, list):
            return []

        converted_content: List[Dict[str, Any]] = []
        for item in content:
            converted_item = self._convert_responses_content_item_to_chat_item(item)
            if converted_item is not None:
                converted_content.append(converted_item)

        return converted_content

    def _convert_responses_content_item_to_chat_item(self, item: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None

        item_type = item.get("type")
        if item_type in {"input_text", "text"}:
            return {"type": "text", "text": item.get("text", "")}

        if item_type == "input_audio":
            return {"type": "input_audio", "input_audio": item.get("input_audio", {})}

        if item_type in {"input_image", "image_url"}:
            return self._convert_responses_image_item_to_chat_item(item)

        return None

    def _convert_responses_image_item_to_chat_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        image_url = item.get("image_url")
        if isinstance(image_url, str):
            return {"type": "image_url", "image_url": {"url": image_url}}
        if isinstance(image_url, dict):
            return {"type": "image_url", "image_url": image_url}

        input_image = item.get("input_image")
        if isinstance(input_image, str):
            return {"type": "image_url", "image_url": {"url": input_image}}
        if isinstance(input_image, dict):
            nested_url = input_image.get("image_url") or input_image.get("url")
            if isinstance(nested_url, str):
                return {"type": "image_url", "image_url": {"url": nested_url}}

        return None

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
            part = self._convert_openai_content_item_to_part(item)
            if part is not None:
                parts.append(part)

        return parts

    def _convert_openai_content_item_to_part(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        item_type = item.get("type")
        if item_type == "text":
            return {"text": item.get("text", "")}
        if item_type == "input_audio":
            return self._convert_openai_input_audio_to_part(item.get("input_audio", {}))
        if item_type == "image_url":
            return self._convert_openai_image_url_to_part(item.get("image_url", {}))
        return None

    def _convert_openai_input_audio_to_part(self, input_audio: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        audio_data = input_audio.get("data", "")
        audio_format = str(input_audio.get("format", "")).strip().lower()
        if not audio_data:
            return None

        return {
            "inline_data": {
                "mime_type": self._audio_format_to_mime_type(audio_format),
                "data": audio_data,
            }
        }

    def _convert_openai_image_url_to_part(self, image_url: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        image_url_value = image_url.get("url", "")
        if not image_url_value.startswith("data:"):
            logger.warning(
                "Image URLs are not supported in this version, only base64 data URIs: %s...",
                image_url_value[:30],
            )
            return None

        return self._parse_data_uri_inline_part(image_url_value)

    def _parse_data_uri_inline_part(self, data_uri: str) -> Optional[Dict[str, Any]]:
        try:
            header, base64_data = data_uri.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            return {"inline_data": {"mime_type": mime_type, "data": base64_data}}
        except Exception as exc:
            logger.warning("Failed to parse data URI: %s", exc)
            return None

    @staticmethod
    def _audio_format_to_mime_type(audio_format: str) -> str:
        format_key = audio_format.strip().lower()
        mime_types = {
            "pcm": AUDIO_PCM_MIME_TYPE,
            "wav": AUDIO_WAV_MIME_TYPE,
            "wave": AUDIO_WAV_MIME_TYPE,
            "mp3": AUDIO_MPEG_MIME_TYPE,
            "mpeg": AUDIO_MPEG_MIME_TYPE,
            "mpga": AUDIO_MPEG_MIME_TYPE,
            "m4a": "audio/mp4",
            "mp4": "audio/mp4",
            "aac": "audio/aac",
            "aiff": "audio/aiff",
            "aif": "audio/aiff",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
            "oga": "audio/ogg",
            "webm": "audio/webm",
        }
        if format_key in mime_types:
            return mime_types[format_key]
        if format_key:
            return f"audio/{format_key}"
        return AUDIO_WAV_MIME_TYPE

    def _convert_genai_to_openai_response(self, genai_response: Any, original_model: str) -> Dict[str, Any]:
        """Convert Google GenAI response to OpenAI format"""
        try:
            text_content = self._extract_genai_text(genai_response)
            usage = self._extract_genai_usage(genai_response)
            created = self._current_utc_unix_timestamp()
            openai_response = {
                "id": f"chatcmpl-{created}",
                "object": "chat.completion",
                "created": created,
                "model": original_model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": text_content}, "finish_reason": "stop"}
                ],
                "usage": usage,
            }

            return openai_response

        except Exception:
            logger.exception("Error converting GenAI response to OpenAI format")
            raise

    def _convert_genai_to_responses_response(self, genai_response: Any, original_model: str) -> Dict[str, Any]:
        """Convert Google GenAI response to OpenAI Responses format."""
        try:
            text_content = self._extract_genai_text(genai_response)
            usage = self._extract_genai_usage(genai_response)
            created = self._current_utc_unix_timestamp()
            response_id, message_id = self._create_responses_ids()

            return {
                "id": response_id,
                "object": "response",
                "created_at": created,
                "status": "completed",
                "model": original_model,
                "output": [
                    {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text_content, "annotations": []}],
                    }
                ],
                "output_text": text_content,
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }

        except Exception:
            logger.exception("Error converting GenAI response to OpenAI Responses format")
            raise

    def _extract_genai_audio_bytes(self, genai_response: Any) -> bytes:
        audio_chunks = []
        content = self._extract_genai_audio_content(genai_response)
        for part in getattr(content, "parts", None) or []:
            inline_data = self._extract_genai_inline_data(part)
            if inline_data is None or not self._is_genai_audio_inline_data(inline_data):
                continue

            audio_data = self._extract_genai_audio_data(inline_data)
            if audio_data:
                audio_chunks.append(audio_data)

        return b"".join(audio_chunks)

    @staticmethod
    def _extract_genai_audio_content(genai_response: Any) -> Any:
        candidates = getattr(genai_response, "candidates", None) or []
        if not candidates:
            return None
        return getattr(candidates[0], "content", None)

    @staticmethod
    def _extract_genai_inline_data(part: Any) -> Any:
        inline_data = getattr(part, "inline_data", None)
        if inline_data is None and isinstance(part, dict):
            return part.get("inline_data") or part.get("inlineData")
        return inline_data

    @staticmethod
    def _is_genai_audio_inline_data(inline_data: Any) -> bool:
        mime_type = getattr(inline_data, "mime_type", None)
        if mime_type is None and isinstance(inline_data, dict):
            mime_type = inline_data.get("mime_type") or inline_data.get("mimeType")
        return not mime_type or str(mime_type).lower().startswith("audio/")

    @staticmethod
    def _extract_genai_audio_data(inline_data: Any) -> bytes:
        audio_data = getattr(inline_data, "data", None)
        if audio_data is None and isinstance(inline_data, dict):
            audio_data = inline_data.get("data")
        if not audio_data:
            return b""

        if isinstance(audio_data, bytes):
            return audio_data
        if isinstance(audio_data, bytearray):
            return bytes(audio_data)
        return base64.b64decode(audio_data)

    @staticmethod
    def _pcm_to_wav_bytes(
        pcm: bytes,
        rate: int = TTS_SAMPLE_RATE_HZ,
        channels: int = TTS_CHANNELS,
        sample_width: int = TTS_SAMPLE_WIDTH_BYTES,
    ) -> bytes:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(rate)
            wav_file.writeframes(pcm)
        return wav_buffer.getvalue()

    def _build_openai_audio_payload(self, genai_response: Any, requested_format: str) -> Dict[str, str]:
        pcm_audio = self._extract_genai_audio_bytes(genai_response)
        if not pcm_audio:
            text_content = self._extract_genai_text(genai_response).strip()
            if text_content:
                raise RuntimeError("Google GenAI returned text instead of audio for TTS request; retry the request")
            raise RuntimeError("Google GenAI returned no audio data for TTS request")

        normalized_format = requested_format if requested_format in TTS_SUPPORTED_FORMATS else "pcm"
        if normalized_format == "wav":
            output_bytes = self._pcm_to_wav_bytes(pcm_audio)
            mime_type = AUDIO_WAV_MIME_TYPE
        else:
            output_bytes = pcm_audio
            mime_type = AUDIO_PCM_MIME_TYPE

        return {
            "data": base64.b64encode(output_bytes).decode("ascii"),
            "format": normalized_format,
            "mime_type": mime_type,
        }

    def _convert_genai_to_openai_audio_chat_response(
        self,
        genai_response: Any,
        original_model: str,
        requested_format: str,
    ) -> Dict[str, Any]:
        usage = self._extract_genai_usage(genai_response)
        created = self._current_utc_unix_timestamp()

        return {
            "id": f"chatcmpl-{created}",
            "object": "chat.completion",
            "created": created,
            "model": original_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "audio": self._build_openai_audio_payload(genai_response, requested_format),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    def _convert_genai_to_responses_audio_response(
        self,
        genai_response: Any,
        original_model: str,
        requested_format: str,
    ) -> Dict[str, Any]:
        usage = self._extract_genai_usage(genai_response)
        created = self._current_utc_unix_timestamp()
        response_id, message_id = self._create_responses_ids()

        return {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "completed",
            "model": original_model,
            "output": [
                {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_audio",
                            "audio": self._build_openai_audio_payload(genai_response, requested_format),
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

    @staticmethod
    def _create_responses_ids() -> Tuple[str, str]:
        return f"resp-{secrets.token_hex(8)}", f"msg-{secrets.token_hex(8)}"

    @staticmethod
    def _current_utc_unix_timestamp() -> int:
        return int(datetime.now(timezone.utc).timestamp())

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

    @staticmethod
    def _extract_openai_image_usage(response: Dict[str, Any]) -> int:
        usage = response.get("usage", {}) if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            return 0

        total_tokens = usage.get("total_tokens")
        if total_tokens is not None:
            return int(total_tokens)

        return int(usage.get("prompt_tokens", 0) or 0)

    def _build_google_image_generation_headers(self, api_key: str, is_native_imagen: bool = False) -> Dict[str, str]:
        if is_native_imagen:
            return {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }

        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_google_image_generation_url(self, model_name: Optional[str] = None) -> str:
        if model_name and self._is_imagen_generation_model(model_name):
            return f"{self.get_endpoint().rstrip('/')}{GOOGLE_IMAGEN_PREDICT_PATH.format(model=model_name)}"
        return f"{self.get_endpoint().rstrip('/')}{GOOGLE_OPENAI_IMAGES_GENERATION_PATH}"

    def _is_image_generation_model(self, model_name: str) -> bool:
        if not model_name:
            return False
        normalized_model_name = self._normalize_model_name(model_name).lower()
        return (
            normalized_model_name in GOOGLE_GENAI_IMAGE_MODEL_ALLOWLIST
            or normalized_model_name in GOOGLE_IMAGEN_IMAGE_MODEL_ALLOWLIST
        )

    def _is_imagen_generation_model(self, model_name: str) -> bool:
        if not model_name:
            return False
        return self._normalize_model_name(model_name).lower() in GOOGLE_IMAGEN_IMAGE_MODEL_ALLOWLIST

    @staticmethod
    def _normalize_imagen_response_format(response_format: Any) -> Optional[str]:
        if response_format is None:
            return None
        normalized = str(response_format).strip().lower()
        return normalized or None

    @staticmethod
    def _extract_imagen_native_config(openai_request: Dict[str, Any]) -> Dict[str, Any]:
        extra_body = openai_request.get("extra_body")
        if not isinstance(extra_body, dict):
            return {}

        google_body = extra_body.get("google")
        if not isinstance(google_body, dict):
            return {}

        imagen_body = google_body.get("imagen")
        if not isinstance(imagen_body, dict):
            return {}

        return dict(imagen_body)

    @staticmethod
    def _normalize_imagen_parameter_name(parameter_name: str) -> str:
        if parameter_name == "image_size":
            return "imageSize"
        if parameter_name == "aspect_ratio":
            return "aspectRatio"
        if parameter_name == "guidance_scale":
            return "guidanceScale"
        if parameter_name == "include_rai_reason":
            return "includeRaiReason"
        if parameter_name == "include_safety_attributes":
            return "includeSafetyAttributes"
        if parameter_name == "negative_prompt":
            return "negativePrompt"
        if parameter_name == "output_mime_type":
            return "output_mime_type"
        if parameter_name == "output_compression_quality":
            return "output_compression_quality"
        if parameter_name == "person_generation":
            return "personGeneration"
        if parameter_name == "safety_filter_level":
            return "safetySetting"
        return parameter_name

    @classmethod
    def _normalize_imagen_parameters(cls, parameters: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for raw_key, value in parameters.items():
            key = cls._normalize_imagen_parameter_name(str(raw_key).strip())
            if key == "output_mime_type":
                cls._set_imagen_output_option(normalized, "mimeType", value)
                continue
            if key == "output_compression_quality":
                cls._set_imagen_output_option(normalized, "compressionQuality", value)
                continue
            if key == "outputOptions":
                cls._merge_imagen_output_options(normalized, value)
                continue
            if key == "personGeneration":
                normalized[key] = str(value).strip().lower()
                continue
            normalized[key] = value
        return normalized

    @staticmethod
    def _merge_imagen_output_options(normalized: Dict[str, Any], value: Any) -> None:
        output_options = normalized.get("outputOptions")
        if isinstance(output_options, dict) and isinstance(value, dict):
            output_options.update(value)
            return
        normalized["outputOptions"] = value

    @staticmethod
    def _set_imagen_output_option(normalized: Dict[str, Any], option_name: str, value: Any) -> None:
        output_options = normalized.setdefault("outputOptions", {})
        if isinstance(output_options, dict):
            output_options[option_name] = value

    @staticmethod
    def _map_imagen_size_to_aspect_ratio(size: Any) -> Optional[str]:
        if not isinstance(size, str):
            return None
        return GOOGLE_IMAGEN_SIZE_TO_ASPECT_RATIO.get(size.strip().lower())

    def _extract_imagen_aspect_ratio(self, imagen_parameters: Dict[str, Any]) -> Optional[str]:
        aspect_ratio = imagen_parameters.get("aspectRatio")
        if aspect_ratio is None:
            return None
        return str(aspect_ratio).strip()

    @staticmethod
    def _extract_imagen_output_options(imagen_parameters: Dict[str, Any]) -> Dict[str, Any]:
        output_options = imagen_parameters.get("outputOptions")
        if isinstance(output_options, dict):
            return output_options
        return {}

    def _raise_invalid_image_request(self, context: GoogleGenAICompletionContext, detail: str) -> None:
        raise self._build_request_error_from_context(context, f"400 invalid argument: {detail}")

    def _validate_imagen_membership(
        self,
        value: Any,
        allowed_values: set[str],
        context: GoogleGenAICompletionContext,
        detail: str,
        *,
        formatter: Optional[Callable[[Any], str]] = None,
    ) -> None:
        if value is None:
            return
        normalized = formatter(value) if formatter is not None else str(value)
        if normalized not in allowed_values:
            self._raise_invalid_image_request(context, detail)

    def _validate_imagen_supported_parameters(
        self,
        imagen_parameters: Dict[str, Any],
        context: GoogleGenAICompletionContext,
    ) -> None:
        unsupported_parameters = set(imagen_parameters.keys()) - GOOGLE_IMAGEN_ALLOWED_PARAMETER_NAMES
        if unsupported_parameters:
            self._raise_invalid_image_request(
                context,
                "unsupported Imagen parameter(s): " + ",".join(sorted(unsupported_parameters)),
            )

    def _validate_imagen_output_options(
        self,
        imagen_parameters: Dict[str, Any],
        context: GoogleGenAICompletionContext,
    ) -> None:
        output_options = self._extract_imagen_output_options(imagen_parameters)
        mime_type = output_options.get("mimeType")
        self._validate_imagen_membership(
            mime_type,
            GOOGLE_IMAGEN_SUPPORTED_OUTPUT_MIME_TYPES,
            context,
            "extra_body.google.imagen.output_mime_type must be image/png or image/jpeg",
        )

        compression_quality = output_options.get("compressionQuality")
        if compression_quality is not None:
            if not isinstance(compression_quality, int) or isinstance(compression_quality, bool):
                self._raise_invalid_image_request(
                    context,
                    "extra_body.google.imagen.output_compression_quality must be an integer",
                )
            if compression_quality < 0 or compression_quality > 100:
                self._raise_invalid_image_request(
                    context,
                    "extra_body.google.imagen.output_compression_quality must be between 0 and 100",
                )
            if str(mime_type or "") != "image/jpeg":
                self._raise_invalid_image_request(
                    context,
                    "extra_body.google.imagen.output_compression_quality requires output_mime_type=image/jpeg",
                )

    def _validate_imagen_native_parameters(
        self,
        imagen_parameters: Dict[str, Any],
        context: GoogleGenAICompletionContext,
    ) -> None:
        self._validate_imagen_supported_parameters(imagen_parameters, context)
        aspect_ratio = self._extract_imagen_aspect_ratio(imagen_parameters)
        if aspect_ratio is not None and aspect_ratio not in GOOGLE_IMAGEN_SUPPORTED_ASPECT_RATIOS:
            self._raise_invalid_image_request(
                context,
                "extra_body.google.imagen.aspect_ratio must be one of "
                + ", ".join(sorted(GOOGLE_IMAGEN_SUPPORTED_ASPECT_RATIOS)),
            )

        membership_validations = (
            (
                imagen_parameters.get("imageSize"),
                GOOGLE_IMAGEN_SUPPORTED_IMAGE_SIZES,
                "extra_body.google.imagen.image_size must be one of 1K or 2K",
                None,
            ),
            (
                imagen_parameters.get("personGeneration"),
                {"dont_allow", "allow_adult", "allow_all"},
                "extra_body.google.imagen.person_generation is invalid",
                lambda value: str(value).strip().lower(),
            ),
            (
                imagen_parameters.get("safetySetting"),
                {"BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "BLOCK_NONE"},
                "extra_body.google.imagen.safety_filter_level is invalid",
                None,
            ),
        )
        for value, allowed_values, detail, formatter in membership_validations:
            self._validate_imagen_membership(value, allowed_values, context, detail, formatter=formatter)

        self._validate_imagen_output_options(imagen_parameters, context)

    def _convert_imagen_response_to_openai_images(self, response: Dict[str, Any]) -> Dict[str, Any]:
        predictions = response.get("predictions") if isinstance(response, dict) else None
        if not isinstance(predictions, list):
            raise ValueError("400 invalid argument: image generation response is missing predictions")
        if not predictions:
            raise ValueError("400 invalid argument: image generation returned no predictions")

        data = []
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            b64_data = prediction.get("bytesBase64Encoded")
            if b64_data is None:
                continue
            data.append({"b64_json": str(b64_data)})

        if not data:
            raise ValueError("400 invalid argument: image generation response did not include base64 image data")

        openai_response: Dict[str, Any] = {
            "created": self._current_utc_unix_timestamp(),
            "data": data,
        }

        tokens_used = self._extract_openai_image_usage(response)
        if tokens_used:
            openai_response["usage"] = {"total_tokens": tokens_used, "prompt_tokens": 0, "completion_tokens": 0}

        return openai_response

    @staticmethod
    def _extract_image_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                message = error_payload.get("message")
                if message:
                    return f"{response.status_code} {message}"

        body_text = response.text.strip()
        if body_text:
            return f"{response.status_code} {body_text}"
        return f"{response.status_code} {response.reason_phrase}"

    async def _initialize_request_context(
        self,
        openai_request: Dict[str, Any],
        context: GoogleGenAICompletionContext,
        endpoint: str,
    ) -> GoogleGenAICompletionContext:
        context.endpoint = endpoint
        context.model_name = self._normalize_model_name(openai_request.get("model", ""))
        context.api_key = await self._select_best_api_key(context.model_name)
        context.api_key_suffix = redact_secret(context.api_key)
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

        logger.debug(
            "🚀 Outbound request: model=%s, api_key=%s key=%s, proxy=%s%s [obs=%s]",
            context.model_name,
            context.api_key_suffix,
            f"#{context.api_key_index}/{context.api_key_total}" if context.api_key_index else "#unknown",
            self._mask_proxy_url(context.proxy_url) if context.proxy_url else "direct",
            context.pool_info,
            context.observation_id,
        )

        return context

    def _build_image_generation_request(
        self,
        openai_request: Dict[str, Any],
        context: GoogleGenAICompletionContext,
    ) -> Dict[str, Any]:
        if self._is_imagen_generation_model(context.model_name):
            return self._build_imagen_generation_request(openai_request, context)

        image_request = {
            "model": context.model_name,
            "prompt": str(openai_request.get("prompt", "")).strip(),
        }

        if "n" in openai_request:
            image_request["n"] = openai_request["n"]

        if "size" in openai_request:
            image_request["size"] = openai_request["size"]

        if "response_format" in openai_request:
            image_request["response_format"] = openai_request["response_format"]

        if "user" in openai_request:
            image_request["user"] = openai_request["user"]

        if "extra_body" in openai_request:
            image_request["extra_body"] = openai_request["extra_body"]

        return image_request

    def _resolve_imagen_top_level_aspect_ratio(
        self,
        size: Any,
        context: GoogleGenAICompletionContext,
    ) -> Optional[str]:
        if size is None:
            return None

        normalized_size = str(size).strip().lower() if isinstance(size, str) else ""
        if normalized_size == "auto":
            return None

        aspect_ratio = self._map_imagen_size_to_aspect_ratio(size)
        if aspect_ratio is None:
            self._raise_invalid_image_request(
                context,
                "Imagen models only support top-level sizes that map to supported provider aspect ratios; "
                "use extra_body.google.imagen.aspect_ratio and image_size for provider-native sizing",
            )

        return aspect_ratio

    def _build_imagen_generation_request(
        self,
        openai_request: Dict[str, Any],
        context: GoogleGenAICompletionContext,
    ) -> Dict[str, Any]:
        prompt = str(openai_request.get("prompt", "")).strip()
        imagen_parameters: Dict[str, Any] = self._normalize_imagen_parameters(
            self._extract_imagen_native_config(openai_request)
        )
        imagen_parameters["sampleCount"] = openai_request.get("n", 1)
        aspect_ratio = self._resolve_imagen_top_level_aspect_ratio(openai_request.get("size"), context)
        if aspect_ratio is not None:
            imagen_parameters["aspectRatio"] = aspect_ratio

        return {
            "instances": [{"prompt": prompt}],
            "parameters": imagen_parameters,
        }

    def _validate_image_generation_request(
        self,
        openai_request: Dict[str, Any],
        context: GoogleGenAICompletionContext,
    ) -> None:
        if not context.model_name:
            raise self._build_request_error_from_context(
                context,
                "400 invalid argument: image generation requests require a model",
            )

        if not self._is_image_generation_model(context.model_name):
            raise self._build_request_error_from_context(
                context,
                f"400 invalid argument: model '{context.model_name}' does not support image generation",
            )

        self._validate_supported_image_request_fields(openai_request, context)
        self._validate_image_prompt(openai_request, context)
        self._validate_image_request_count(openai_request, context)
        self._validate_image_response_format(openai_request, context)

        if self._is_imagen_generation_model(context.model_name):
            self._validate_imagen_request_contract(openai_request, context)

    def _validate_supported_image_request_fields(
        self,
        openai_request: Dict[str, Any],
        context: GoogleGenAICompletionContext,
    ) -> None:
        unsupported_fields = set(openai_request.keys()) - OPENAI_IMAGE_REQUEST_FIELDS
        if unsupported_fields:
            raise self._build_request_error_from_context(
                context,
                "400 invalid argument: unsupported image request field(s): "
                + ",".join(sorted(unsupported_fields)),
            )

    def _validate_image_prompt(
        self,
        openai_request: Dict[str, Any],
        context: GoogleGenAICompletionContext,
    ) -> None:
        prompt = openai_request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise self._build_request_error_from_context(
                context,
                "400 invalid argument: image generation requests require a non-empty prompt",
            )

    def _validate_image_request_count(
        self,
        openai_request: Dict[str, Any],
        context: GoogleGenAICompletionContext,
    ) -> None:
        if "n" not in openai_request:
            return

        n = openai_request["n"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise self._build_request_error_from_context(
                context,
                "400 invalid argument: n must be a positive integer",
            )
        if self._is_imagen_generation_model(context.model_name) and n > 4:
            raise self._build_request_error_from_context(
                context,
                "400 invalid argument: Imagen models only support 1 to 4 images per request",
            )

    def _validate_image_response_format(
        self,
        openai_request: Dict[str, Any],
        context: GoogleGenAICompletionContext,
    ) -> None:
        if "response_format" not in openai_request:
            return

        response_format = self._normalize_imagen_response_format(openai_request["response_format"])
        if response_format is None or response_format not in GOOGLE_IMAGE_RESPONSE_FORMATS:
            raise self._build_request_error_from_context(
                context,
                "400 invalid argument: response_format must be one of 'url' or 'b64_json'",
            )
        if self._is_imagen_generation_model(context.model_name) and response_format == "url":
            raise self._build_request_error_from_context(
                context,
                "400 invalid argument: response_format='url' is unsupported for Imagen models; request 'b64_json' instead",
            )

    def _validate_imagen_size_request(
        self,
        openai_request: Dict[str, Any],
        context: GoogleGenAICompletionContext,
        imagen_parameters: Dict[str, Any],
    ) -> None:
        aspect_ratio = self._resolve_imagen_top_level_aspect_ratio(openai_request.get("size"), context)
        if aspect_ratio is None:
            return

        existing_aspect_ratio = self._extract_imagen_aspect_ratio(imagen_parameters)
        if existing_aspect_ratio and existing_aspect_ratio != aspect_ratio:
            self._raise_invalid_image_request(
                context,
                "top-level size and extra_body.google.imagen.aspect_ratio conflict",
            )

    def _validate_imagen_request_contract(
        self,
        openai_request: Dict[str, Any],
        context: GoogleGenAICompletionContext,
    ) -> None:
        extra_body = openai_request.get("extra_body")
        if extra_body is not None and not isinstance(extra_body, dict):
            raise self._build_request_error_from_context(
                context,
                "400 invalid argument: extra_body must be an object",
            )

        imagen_parameters = self._normalize_imagen_parameters(self._extract_imagen_native_config(openai_request))
        self._validate_imagen_size_request(openai_request, context, imagen_parameters)
        self._validate_imagen_native_parameters(imagen_parameters, context)

    async def _run_image_generation_request(
        self,
        context: GoogleGenAICompletionContext,
        image_request: Dict[str, Any],
    ) -> Dict[str, Any]:
        client = httpx.AsyncClient(
            timeout=self.config.timeout,
            transport=context.async_transport,
            trust_env=False,
        )

        try:
            await self._rate_limiter.acquire_slot()
            response = await client.post(
                self._build_google_image_generation_url(context.model_name),
                json=image_request,
                headers=self._build_google_image_generation_headers(
                    context.api_key, is_native_imagen=self._is_imagen_generation_model(context.model_name)
                ),
            )
            if response.is_error:
                raise RuntimeError(self._extract_image_error_message(response))
            image_response = response.json()

            if self._is_imagen_generation_model(context.model_name):
                return self._convert_imagen_response_to_openai_images(image_response)

            return image_response
        finally:
            await self._rate_limiter.release_slot()
            await client.aclose()

    async def _finalize_image_generation(
        self,
        context: GoogleGenAICompletionContext,
        response: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Optional["RequestMetadata"]]:
        tokens_used = self._extract_openai_image_usage(response)
        return response, await self._build_completion_metadata(context, tokens_used)

    async def generate_image(
        self,
        openai_request: Dict[str, Any],
        endpoint: str = OPENAI_IMAGES_ENDPOINT,
    ) -> tuple[Dict[str, Any], Optional["RequestMetadata"]]:
        import uuid

        if endpoint != OPENAI_IMAGES_ENDPOINT:
            raise ValueError(f"Unsupported endpoint for image generation: {endpoint}")

        # Validate request contract first so malformed image requests get
        # deterministic input validation errors before transport/key/proxy
        # selection can fail for reasons unrelated to the image payload.
        context = GoogleGenAICompletionContext(
            original_model=openai_request.get("model", ""),
            observation_id=f"obs_{uuid.uuid4().hex[:12]}",
            endpoint=endpoint,
            request_kind="generate_images",
        )
        context.model_name = self._normalize_model_name(openai_request.get("model", ""))
        self._validate_image_generation_request(openai_request, context)

        try:
            context = await self._initialize_request_context(openai_request, context, endpoint)
            image_request = self._build_image_generation_request(openai_request, context)
            response = await self._run_image_generation_request(context, image_request)
            return await self._finalize_image_generation(context, response)

        except GoogleGenAIRequestError:
            raise
        except Exception as error:
            raise await self._handle_completion_exception(context, error)

    async def generate_completion(
        self,
        openai_request: Dict[str, Any],
        endpoint: str = OPENAI_CHAT_COMPLETIONS_ENDPOINT,
    ) -> tuple[Dict[str, Any], Optional["RequestMetadata"]]:
        """Generate completion using Google GenAI API

        Returns: (response_dict, metadata)
        """
        import uuid

        context = GoogleGenAICompletionContext(
            original_model=openai_request.get("model", ""),
            observation_id=f"obs_{uuid.uuid4().hex[:12]}",
            endpoint=endpoint,
            request_kind="embed_content" if endpoint == OPENAI_EMBEDDINGS_ENDPOINT else "generate_content",
        )

        try:
            self._validate_endpoint_request(openai_request, endpoint, context)
            context = await self._build_completion_context(openai_request, context, endpoint)
            response = await self._run_completion_request(context)
            return await self._finalize_completion(context, response)

        except GoogleGenAIRequestError:
            raise
        except Exception as error:
            raise await self._handle_completion_exception(context, error)

    async def _build_completion_context(
        self,
        openai_request: Dict[str, Any],
        context: GoogleGenAICompletionContext,
        endpoint: str = OPENAI_CHAT_COMPLETIONS_ENDPOINT,
    ) -> GoogleGenAICompletionContext:
        context.endpoint = endpoint
        context.tts_request = self._is_tts_request(openai_request)
        context.tts_audio_format = self._get_tts_audio_format(openai_request) if context.tts_request else "pcm"
        context.model_name, context.genai_request = self._convert_openai_to_genai_request(openai_request, endpoint)
        context.api_key = await self._select_best_api_key(context.model_name)
        context.api_key_suffix = redact_secret(context.api_key)
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

        proxy_label = self._mask_proxy_url(context.proxy_url) if context.proxy_url else "direct"
        key_position_str = f" key #{context.api_key_index}/{context.api_key_total}" if context.api_key_index else ""
        logger.debug(
            f"🚀 Outbound request: model={context.model_name}, api_key={context.api_key_suffix}{key_position_str}, "
            f"proxy={proxy_label}{context.pool_info} [obs={context.observation_id}]"
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
        return self._create_genai_client(api_key, sync_transport, async_transport)

    async def _run_completion_request(self, context: GoogleGenAICompletionContext) -> Any:
        await self._rate_limiter.acquire_slot()
        try:
            if context.request_kind == "embed_content":
                contents = context.genai_request["contents"]
                embed_inputs = contents if isinstance(contents, list) else [contents]
                embedding_responses = []
                token_total = 0

                for embed_input in embed_inputs:
                    try:
                        token_counts = await asyncio.to_thread(
                            context.client.models.count_tokens,
                            model=context.model_name,
                            contents=embed_input,
                        )
                        token_total += int(getattr(token_counts, "total_tokens", 0) or 0)
                    except Exception:
                        logger.debug(
                            "Embedding token preflight failed for %s",
                            context.model_name,
                            exc_info=True,
                        )

                    embedding_responses.append(
                        await asyncio.to_thread(
                            context.client.models.embed_content,
                            model=context.model_name,
                            contents=embed_input,
                            config=context.genai_request.get("config"),
                        )
                    )

                context.input_token_count = token_total or None
                if len(embedding_responses) == 1:
                    return embedding_responses[0]

                combined_embeddings = []
                combined_billable_chars = 0
                for embedding_response in embedding_responses:
                    response_embeddings = getattr(embedding_response, "embeddings", None)
                    if not response_embeddings:
                        single_embedding = getattr(embedding_response, "embedding", None)
                        response_embeddings = [single_embedding] if single_embedding else []
                    combined_embeddings.extend(response_embeddings)
                    metadata = getattr(embedding_response, "metadata", None)
                    combined_billable_chars += int(getattr(metadata, "billable_character_count", 0) or 0)

                return SimpleNamespace(
                    embeddings=combined_embeddings,
                    metadata=SimpleNamespace(billable_character_count=combined_billable_chars or token_total),
                )

            return await asyncio.to_thread(
                context.client.models.generate_content,
                model=context.model_name,
                contents=context.genai_request["contents"],
                config=context.genai_request["generation_config"],
            )
        finally:
            await self._rate_limiter.release_slot()

    def _validate_endpoint_request(
        self,
        openai_request: Dict[str, Any],
        endpoint: str,
        context: GoogleGenAICompletionContext,
    ) -> None:
        if endpoint == OPENAI_EMBEDDINGS_ENDPOINT and openai_request.get("input") is None:
            raise self._build_request_error_from_context(
                context,
                "400 invalid argument: embeddings requests require an input",
            )

        if self._is_tts_request(openai_request):
            try:
                self._validate_tts_request(openai_request, endpoint)
            except ValueError as exc:
                raise self._build_request_error_from_context(context, str(exc))

        if endpoint == OPENAI_RESPONSES_ENDPOINT and openai_request.get("stream"):
            raise self._build_request_error_from_context(
                context,
                f"400 invalid argument: streaming is not supported for {OPENAI_RESPONSES_ENDPOINT} "
                "in the Google GenAI compatibility shim",
            )

    async def _build_completion_metadata(
        self, context: GoogleGenAICompletionContext, tokens_used: int
    ) -> RequestMetadata:
        await self._update_api_key_stats(context.api_key, context.model_name, success=True, tokens=tokens_used)

        if context.proxy_url:
            self._mark_proxy_health(context.proxy_url, success=True)

        observer = get_observer()
        observation = observer.get_observation(context.observation_id)
        actual_key_suffix, actual_proxy, key_verified, proxy_verified = self._resolve_observation_state(
            context, observation
        )

        return RequestMetadata(
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

    async def _finalize_completion(
        self, context: GoogleGenAICompletionContext, response: Any
    ) -> tuple[Dict[str, Any], Optional["RequestMetadata"]]:
        if context.request_kind == "embed_content":
            openai_response = self._convert_genai_to_embeddings_response(
                response,
                context.original_model,
                context.input_token_count,
            )
            tokens_used = openai_response.get("usage", {}).get("total_tokens", 0)
            return openai_response, await self._build_completion_metadata(context, tokens_used)

        if context.tts_request and context.endpoint == OPENAI_RESPONSES_ENDPOINT:
            openai_response = self._convert_genai_to_responses_audio_response(
                response, context.original_model, context.tts_audio_format
            )
        elif context.tts_request and context.endpoint == OPENAI_CHAT_COMPLETIONS_ENDPOINT:
            openai_response = self._convert_genai_to_openai_audio_chat_response(
                response, context.original_model, context.tts_audio_format
            )
        elif context.endpoint == OPENAI_RESPONSES_ENDPOINT:
            openai_response = self._convert_genai_to_responses_response(response, context.original_model)
        else:
            openai_response = self._convert_genai_to_openai_response(response, context.original_model)

        tokens_used = openai_response.get("usage", {}).get("total_tokens", 0)
        return openai_response, await self._build_completion_metadata(context, tokens_used)

    @classmethod
    def _is_google_supported_action_model(cls, model_name: str, supported_actions: List[str]) -> bool:
        normalized_model_name = cls._extract_google_model_name(model_name).lower()
        return (
            "generateContent" in supported_actions
            or "embedContent" in supported_actions
            or ("predict" in supported_actions and normalized_model_name in GOOGLE_IMAGEN_IMAGE_MODEL_ALLOWLIST)
        )

    def _convert_genai_to_embeddings_response(
        self,
        genai_response: Any,
        original_model: str,
        token_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        embeddings = getattr(genai_response, "embeddings", None)
        if not embeddings:
            single_embedding = getattr(genai_response, "embedding", None)
            embeddings = [single_embedding] if single_embedding else []
        elif not isinstance(embeddings, list):
            embeddings = [embeddings]

        data = []
        for index, embedding in enumerate(embeddings):
            data.append(
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": list(getattr(embedding, "values", None) or []),
                }
            )

        usage_tokens = token_count
        if usage_tokens is None:
            metadata = getattr(genai_response, "metadata", None)
            usage_tokens = getattr(metadata, "billable_character_count", None)

        usage_tokens = int(usage_tokens or 0)
        return {
            "object": "list",
            "data": data,
            "model": original_model,
            "usage": {
                "prompt_tokens": usage_tokens,
                "total_tokens": usage_tokens,
            },
        }

    def _resolve_observation_state(
        self, context: GoogleGenAICompletionContext, observation: Any
    ) -> Tuple[Optional[str], Optional[str], bool, bool]:
        actual_key_suffix = context.api_key_suffix
        actual_proxy = self._proxy_info_to_string(context.proxy_info)
        key_verified = False
        proxy_verified = False

        if observation:
            if observation.api_key_used:
                # Identification suffix only - last 4 (dashboard convention).
                # Not redact_secret(): that keeps the *prefix*, and Google keys
                # all share the "AIza" prefix, so the suffix is what tells keys
                # apart. Last-4 minimizes the exposed fragment while staying useful.
                actual_key_suffix = observation.api_key_used[-4:]
                key_verified = True

            if observation.proxy_url:
                actual_proxy = self._mask_proxy_url(observation.proxy_url) or observation.proxy_url
                proxy_verified = True
            elif not context.proxy_info:
                proxy_verified = True

            logger.debug(
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
        if "400" in error_str or "invalid argument" in error_str:
            return 400
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
