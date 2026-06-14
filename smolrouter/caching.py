"""
Caching implementations for model information with TTL support.

This module provides caching strategies to reduce latency when aggregating
models from multiple providers, with configurable TTL and invalidation.
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable, Dict, List, Optional, Any, cast
from dataclasses import dataclass, field
from datetime import datetime

from .interfaces import IModelCache, ModelInfo
from .task_utils import create_logged_task

logger = logging.getLogger(__name__)


@dataclass
class ProviderHealthInfo:
    """Detailed health information for a provider"""

    healthy: Optional[bool] = None
    last_checked: Optional[datetime] = None
    last_healthy: Optional[datetime] = None

    @property
    def status(self) -> str:
        """Get human-readable status"""
        if self.healthy is None:
            return "unknown"
        return "healthy" if self.healthy else "unhealthy"


@dataclass
class CacheEntry:
    """Cached data with metadata"""

    data: Any
    cached_at: float
    ttl_seconds: int
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() - self.cached_at > self.ttl_seconds

    def touch(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = time.time()

    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return time.time() - self.cached_at


class InMemoryModelCache(IModelCache):
    """In-memory cache implementation with TTL and cleanup"""

    def __init__(self, default_ttl: int = 300, cleanup_interval: int = 60):
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = None
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = create_logged_task(
                self._cleanup_loop(),
                task_name=f"model-cache-cleanup:{id(self)}",
                create_task_fn=asyncio.create_task,
            )

    async def _cleanup_loop(self):
        """Background task to clean up expired entries"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")

    async def _cleanup_expired(self):
        """Remove expired cache entries"""
        async with self._lock:
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]

            for key in expired_keys:
                del self._cache[key]
                logger.debug(f"Cleaned up expired cache entry: {key}")

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def get_cached_models(self, provider_id: str) -> Optional[List[ModelInfo]]:
        """Get cached models for a provider"""
        async with self._lock:
            cache_key = f"models:{provider_id}"
            entry = self._cache.get(cache_key)

            if entry is None:
                logger.debug(f"Cache miss for provider {provider_id}")
                return None

            if entry.is_expired():
                logger.debug(f"Cache expired for provider {provider_id} (age: {entry.age_seconds:.1f}s)")
                del self._cache[cache_key]
                return None

            entry.touch()
            logger.debug(
                f"Cache hit for provider {provider_id} (age: {entry.age_seconds:.1f}s, accessed {entry.access_count} times)"
            )
            return entry.data.copy()  # Return copy to prevent external modification

    async def cache_models(self, provider_id: str, models: List[ModelInfo], ttl_seconds: Optional[int] = None):
        """Cache models for a provider with TTL"""
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl

        async with self._lock:
            cache_key = f"models:{provider_id}"
            entry = CacheEntry(
                data=models.copy(),  # Store copy to prevent external modification
                cached_at=time.time(),
                ttl_seconds=ttl_seconds,
            )

            self._cache[cache_key] = entry
            logger.debug(f"Cached {len(models)} models for provider {provider_id} (TTL: {ttl_seconds}s)")

    async def invalidate_cache(self, provider_id: Optional[str] = None):
        """Invalidate cache for specific provider or all providers"""
        async with self._lock:
            if provider_id is None:
                # Invalidate all
                count = len(self._cache)
                self._cache.clear()
                logger.info(f"Invalidated entire cache ({count} entries)")
            else:
                # Invalidate specific provider
                cache_key = f"models:{provider_id}"
                if cache_key in self._cache:
                    del self._cache[cache_key]
                    logger.info(f"Invalidated cache for provider {provider_id}")
                else:
                    logger.debug(f"No cache entry to invalidate for provider {provider_id}")

    async def is_cache_valid(self, provider_id: str) -> bool:
        """Check if cached data is still valid"""
        async with self._lock:
            cache_key = f"models:{provider_id}"
            entry = self._cache.get(cache_key)
            return entry is not None and not entry.is_expired()

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        async with self._lock:
            stats = {
                "total_entries": len(self._cache),
                "expired_entries": sum(1 for entry in self._cache.values() if entry.is_expired()),
                "entries_by_provider": {},
                "total_access_count": sum(entry.access_count for entry in self._cache.values()),
                "default_ttl": self.default_ttl,
                "cleanup_interval": self.cleanup_interval,
            }

            for key, entry in self._cache.items():
                if key.startswith("models:"):
                    provider_id = key.split(":", 1)[1]
                    stats["entries_by_provider"][provider_id] = {
                        "age_seconds": entry.age_seconds,
                        "ttl_seconds": entry.ttl_seconds,
                        "access_count": entry.access_count,
                        "model_count": len(entry.data) if isinstance(entry.data, list) else 0,
                        "expired": entry.is_expired(),
                    }

            return stats

    def close(self):
        """Clean shutdown of cache"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()


class NoOpModelCache(IModelCache):
    """No-operation cache that doesn't cache anything"""

    async def get_cached_models(self, provider_id: str) -> Optional[List[ModelInfo]]:
        return None

    async def cache_models(self, provider_id: str, models: List[ModelInfo], ttl_seconds: Optional[int] = None):
        return None

    async def invalidate_cache(self, provider_id: Optional[str] = None):
        return None

    async def is_cache_valid(self, provider_id: str) -> bool:
        return False


class ModelAggregator:
    """
    Aggregates models from multiple providers with intelligent caching.

    This is the core service that coordinates model discovery across providers,
    handles caching for performance, and provides health monitoring.
    """

    def __init__(
        self,
        providers: List,
        cache: Optional[IModelCache] = None,
        default_cache_ttl: int = 300,
        health_check_interval: int = 30,
        discovery_timeout: float = 10.0,
    ):
        self.providers = providers
        self.cache = cache or InMemoryModelCache(default_ttl=default_cache_ttl)
        self.default_cache_ttl = default_cache_ttl
        self.health_check_interval = health_check_interval
        self.discovery_timeout = discovery_timeout
        self._provider_health: Dict[str, ProviderHealthInfo] = {}
        self._last_known_models: Dict[str, List[ModelInfo]] = {}
        self._refresh_tasks: Dict[str, asyncio.Task[List[ModelInfo]]] = {}
        self._refresh_guard = asyncio.Lock()
        self._last_refresh_attempt: Dict[str, float] = {}
        self._refresh_min_interval: float = min(default_cache_ttl, 30.0)
        self._health_check_task = None

        # Initialize health info for all providers as unknown
        for provider in providers:
            provider_id = provider.get_provider_id()
            self._provider_health[provider_id] = ProviderHealthInfo()

        self._start_health_monitoring()

    def _start_health_monitoring(self):
        """Start background health monitoring"""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = create_logged_task(
                self._health_monitoring_loop(),
                task_name="provider-health-monitor",
                create_task_fn=asyncio.create_task,
            )

    async def _health_monitoring_loop(self):
        """Background task to monitor provider health"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._update_provider_health()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")

    async def _update_provider_health(self):
        """Update health status for all providers"""
        health_tasks = []
        for provider in self.providers:
            health_tasks.append(self._check_single_provider_health(provider))

        await asyncio.gather(*health_tasks, return_exceptions=True)

    async def _check_single_provider_health(self, provider):
        """Check health of a single provider"""
        provider_id = provider.get_provider_id()
        health_info = self._provider_health[provider_id]
        old_status = health_info.healthy

        try:
            is_healthy = await provider.health_check()
            now = datetime.now()

            # Update health info
            health_info.healthy = is_healthy
            health_info.last_checked = now
            if is_healthy:
                health_info.last_healthy = now

            # Log health status changes
            if old_status is not None and old_status != is_healthy:
                status_str = "healthy" if is_healthy else "unhealthy"
                logger.info(f"Provider {provider_id} is now {status_str}")

        except Exception as e:
            logger.warning(f"Health check failed for provider {provider_id}: {e}")
            health_info.healthy = False
            health_info.last_checked = datetime.now()

    async def get_all_models(self, force_refresh: bool = False, include_unhealthy: bool = False) -> List[ModelInfo]:
        """
        Get all models from all providers with intelligent caching.

        Args:
            force_refresh: Force refresh from all providers, ignoring cache
            include_unhealthy: Include models from unhealthy providers

        Returns:
            Aggregated list of all available models
        """
        # Determine which providers to query. Providers with last-known-good
        # models stay eligible even if their latest health status is unhealthy.
        # force_refresh (freshness) and include_unhealthy (visibility) are
        # intentionally orthogonal: a forced refresh still only surfaces healthy
        # providers unless include_unhealthy is set. Recovery of unhealthy
        # providers is the background health loop's job.
        providers_to_query = []
        if include_unhealthy:
            providers_to_query = self.providers
        else:
            providers_to_query = [
                p
                for p in self.providers
                if self._is_provider_queryable(p)
            ]

        discovery_tasks = [
            self._discover_models_from_provider(provider, force_refresh)
            for provider in providers_to_query
        ]

        provider_results = await asyncio.gather(*discovery_tasks, return_exceptions=True)

        all_models = []
        for provider, result in zip(providers_to_query, provider_results):
            if isinstance(result, Exception):
                logger.error(f"Error discovering models from {provider.get_provider_id()}: {result}")
                continue

            if isinstance(result, list):
                all_models.extend(result)

        logger.info(f"Aggregated {len(all_models)} models from {len(providers_to_query)} providers")
        return all_models

    def _is_provider_queryable(self, provider) -> bool:
        """Return whether a provider should be included in aggregated model reads."""
        provider_id = provider.get_provider_id()
        health_info = self._provider_health.get(provider_id, ProviderHealthInfo(healthy=True))
        return health_info.healthy is not False or provider_id in self._last_known_models

    async def _discover_models_from_provider(self, provider, force_refresh: bool) -> List[ModelInfo]:
        """Discover models from a single provider with caching"""
        provider_id = provider.get_provider_id()

        if force_refresh:
            task = await self._get_or_start_refresh(provider)
            return await self._await_bounded(task, provider_id)

        cached_models = await self.cache.get_cached_models(provider_id)
        if cached_models is not None:
            logger.debug(f"Using cached models for provider {provider_id}")
            self._last_known_models[provider_id] = cached_models.copy()
            return cached_models

        stale_models = self._last_known_models.get(provider_id)
        if stale_models is not None:
            await self._maybe_start_background_refresh(provider)
            return stale_models.copy()

        task = await self._get_or_start_refresh(provider)
        return await self._await_bounded(task, provider_id)

    async def _get_or_start_refresh(self, provider) -> asyncio.Task[List[ModelInfo]]:
        """Return the active provider refresh task, starting one if needed."""
        provider_id = provider.get_provider_id()
        async with self._refresh_guard:
            task = self._refresh_tasks.get(provider_id)
            if task is None or task.done():
                task = asyncio.create_task(self._run_refresh(provider))
                self._refresh_tasks[provider_id] = task
                self._last_refresh_attempt[provider_id] = time.monotonic()
                task.add_done_callback(lambda done_task, pid=provider_id: self._on_refresh_done(pid, done_task))
            return task

    async def _maybe_start_background_refresh(self, provider):
        """Start a throttled background provider refresh without awaiting it."""
        provider_id = provider.get_provider_id()
        elapsed = time.monotonic() - self._last_refresh_attempt.get(provider_id, 0.0)
        if elapsed < self._refresh_min_interval:
            return

        await self._get_or_start_refresh(provider)

    async def _run_refresh(self, provider) -> List[ModelInfo]:
        """Refresh a single provider with timeout and stale fallback."""
        provider_id = provider.get_provider_id()

        try:
            logger.debug(f"Discovering models from provider {provider_id}")
            async with asyncio.timeout(self.discovery_timeout):
                models = await provider.discover_models()

            await self.cache.cache_models(provider_id, models, self.default_cache_ttl)
            self._last_known_models[provider_id] = models.copy()
            self._mark_provider_healthy(provider_id)

            logger.debug(f"Discovered and cached {len(models)} models from {provider_id}")
            return models

        except TimeoutError:
            logger.warning(f"Timed out discovering models from provider {provider_id}")
            self._mark_provider_unhealthy(provider_id)
            return self._last_known_models.get(provider_id, []).copy()

        except Exception as e:
            logger.error(f"Failed to discover models from provider {provider_id}: {e}")
            self._mark_provider_unhealthy(provider_id)
            return self._last_known_models.get(provider_id, []).copy()

    async def _await_bounded(self, task: asyncio.Task[List[ModelInfo]], provider_id: str) -> List[ModelInfo]:
        """Wait for a shared refresh task without cancelling it on timeout."""
        done, _ = await asyncio.wait({task}, timeout=self.discovery_timeout)
        if task in done:
            # A refresh cancelled by close() raises CancelledError from result()
            # (a BaseException, so it would escape `except Exception`). Fall back
            # to last-known-good rather than propagating a shutdown cancellation
            # into a request.
            if task.cancelled():
                return self._last_known_models.get(provider_id, []).copy()
            try:
                return task.result()
            except Exception:
                return self._last_known_models.get(provider_id, []).copy()

        return self._last_known_models.get(provider_id, []).copy()

    def _on_refresh_done(self, provider_id: str, task: asyncio.Task[List[ModelInfo]]):
        """Remove finished refresh tasks and retrieve exceptions."""
        if self._refresh_tasks.get(provider_id) is task:
            self._refresh_tasks.pop(provider_id, None)

        if task.cancelled():
            return

        exc = task.exception()
        if exc:
            logger.warning(f"Background refresh for {provider_id} errored: {exc}")

    def _mark_provider_healthy(self, provider_id: str):
        """Mark a provider healthy after successful discovery."""
        now = datetime.now()
        health_info = self._provider_health.get(provider_id, ProviderHealthInfo())
        health_info.healthy = True
        health_info.last_checked = now
        health_info.last_healthy = now
        self._provider_health[provider_id] = health_info

    def _mark_provider_unhealthy(self, provider_id: str):
        """Mark a provider unhealthy after failed discovery."""
        health_info = self._provider_health.get(provider_id, ProviderHealthInfo())
        health_info.healthy = False
        health_info.last_checked = datetime.now()
        self._provider_health[provider_id] = health_info

    async def get_models_by_provider(self, provider_id: str, force_refresh: bool = False) -> List[ModelInfo]:
        """Get models from a specific provider"""
        provider = next((p for p in self.providers if p.get_provider_id() == provider_id), None)
        if not provider:
            logger.warning(f"Provider {provider_id} not found")
            return []

        return await self._discover_models_from_provider(provider, force_refresh)

    async def refresh_provider_cache(self, provider_id: Optional[str] = None):
        """Refresh cache for specific provider or all providers"""
        if provider_id:
            await self.cache.invalidate_cache(provider_id)
            # Also drop last-known-good + the refresh throttle so the next read
            # re-discovers instead of serving stale models (an explicit refresh
            # is meant to pick up added/removed upstream models).
            self._last_known_models.pop(provider_id, None)
            self._last_refresh_attempt.pop(provider_id, None)
            logger.info(f"Refreshed cache for provider {provider_id}")
        else:
            await self.cache.invalidate_cache()
            self._last_known_models.clear()
            self._last_refresh_attempt.clear()
            logger.info("Refreshed cache for all providers")

    def get_provider_health(self) -> Dict[str, bool]:
        """Get health status of all providers (backward compatibility).

        Synchronous: this reads the cached in-memory health map. The async
        refresh happens in the background health-monitoring loop.
        """
        return {
            provider_id: health_info.healthy if health_info.healthy is not None else False
            for provider_id, health_info in self._provider_health.items()
        }

    def get_provider_health_detailed(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed health status of all providers"""
        result = {}
        for provider_id, health_info in self._provider_health.items():
            result[provider_id] = {
                "healthy": health_info.healthy,
                "status": health_info.status,
                "last_checked": health_info.last_checked.isoformat() if health_info.last_checked else None,
                "last_healthy": health_info.last_healthy.isoformat() if health_info.last_healthy else None,
                "last_checked_ago": self._time_ago(health_info.last_checked) if health_info.last_checked else "never",
            }
        return result

    def _time_ago(self, timestamp: datetime) -> str:
        """Get human-readable time ago string"""
        if timestamp is None:
            return "never"

        now = datetime.now()
        diff = now - timestamp

        if diff.total_seconds() < 60:
            return f"{int(diff.total_seconds())}s ago"
        elif diff.total_seconds() < 3600:
            return f"{int(diff.total_seconds() / 60)}m ago"
        elif diff.total_seconds() < 86400:
            return f"{int(diff.total_seconds() / 3600)}h ago"
        else:
            return f"{int(diff.total_seconds() / 86400)}d ago"

    async def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics for monitoring"""
        cache_stats = {}
        get_cache_stats = cast(
            Optional[Callable[[], Awaitable[Dict[str, Any]]]],
            getattr(self.cache, "get_cache_stats", None),
        )
        if get_cache_stats is not None:
            cache_stats = await get_cache_stats()

        provider_health = self._provider_health.copy()
        healthy_providers = sum(1 for health_info in provider_health.values() if health_info.healthy is True)

        return {
            "provider_count": len(self.providers),
            "provider_health": provider_health,
            "healthy_providers": healthy_providers,
            "cache_stats": cache_stats,
            "default_cache_ttl": self.default_cache_ttl,
            "health_check_interval": self.health_check_interval,
            "discovery_timeout": self.discovery_timeout,
        }

    def close(self):
        """Clean shutdown of aggregator"""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()

        for task in self._refresh_tasks.values():
            if not task.done():
                task.cancel()

        close_cache = cast(Optional[Callable[[], None]], getattr(self.cache, "close", None))
        if close_cache is not None:
            close_cache()
