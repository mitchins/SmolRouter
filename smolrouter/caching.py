"""
Caching implementations for model information with TTL support.

This module provides caching strategies to reduce latency when aggregating
models from multiple providers, with configurable TTL and invalidation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .interfaces import IModelCache, ModelInfo

logger = logging.getLogger(__name__)


@dataclass
class ProviderHealthInfo:
    """Detailed health information for a provider"""
    healthy: bool = None  # None = unknown, True = healthy, False = unhealthy
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
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background task to clean up expired entries"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
    
    async def _cleanup_expired(self):
        """Remove expired cache entries"""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() 
                if entry.is_expired()
            ]
            
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
            logger.debug(f"Cache hit for provider {provider_id} (age: {entry.age_seconds:.1f}s, accessed {entry.access_count} times)")
            return entry.data.copy()  # Return copy to prevent external modification
    
    async def cache_models(self, provider_id: str, models: List[ModelInfo], ttl_seconds: int = None):
        """Cache models for a provider with TTL"""
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl
        
        async with self._lock:
            cache_key = f"models:{provider_id}"
            entry = CacheEntry(
                data=models.copy(),  # Store copy to prevent external modification
                cached_at=time.time(),
                ttl_seconds=ttl_seconds
            )
            
            self._cache[cache_key] = entry
            logger.debug(f"Cached {len(models)} models for provider {provider_id} (TTL: {ttl_seconds}s)")
    
    async def invalidate_cache(self, provider_id: str = None):
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
                'total_entries': len(self._cache),
                'expired_entries': sum(1 for entry in self._cache.values() if entry.is_expired()),
                'entries_by_provider': {},
                'total_access_count': sum(entry.access_count for entry in self._cache.values()),
                'default_ttl': self.default_ttl,
                'cleanup_interval': self.cleanup_interval
            }
            
            for key, entry in self._cache.items():
                if key.startswith('models:'):
                    provider_id = key.split(':', 1)[1]
                    stats['entries_by_provider'][provider_id] = {
                        'age_seconds': entry.age_seconds,
                        'ttl_seconds': entry.ttl_seconds,
                        'access_count': entry.access_count,
                        'model_count': len(entry.data) if isinstance(entry.data, list) else 0,
                        'expired': entry.is_expired()
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
    
    async def cache_models(self, provider_id: str, models: List[ModelInfo], ttl_seconds: int = None):
        pass
    
    async def invalidate_cache(self, provider_id: str = None):
        pass
    
    async def is_cache_valid(self, provider_id: str) -> bool:
        return False


class ModelAggregator:
    """
    Aggregates models from multiple providers with intelligent caching.
    
    This is the core service that coordinates model discovery across providers,
    handles caching for performance, and provides health monitoring.
    """
    
    def __init__(self, providers: List, cache: IModelCache = None, 
                 default_cache_ttl: int = 300, health_check_interval: int = 30):
        self.providers = providers
        self.cache = cache or InMemoryModelCache(default_ttl=default_cache_ttl)
        self.default_cache_ttl = default_cache_ttl
        self.health_check_interval = health_check_interval
        self._provider_health: Dict[str, ProviderHealthInfo] = {}
        self._discovery_lock = asyncio.Lock()
        self._health_check_task = None
        
        # Initialize health info for all providers as unknown
        for provider in providers:
            provider_id = provider.get_provider_id()
            self._provider_health[provider_id] = ProviderHealthInfo()
        
        self._start_health_monitoring()
    
    def _start_health_monitoring(self):
        """Start background health monitoring"""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_monitoring_loop())
    
    async def _health_monitoring_loop(self):
        """Background task to monitor provider health"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._update_provider_health()
            except asyncio.CancelledError:
                break
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
    
    async def get_all_models(self, force_refresh: bool = False, 
                           include_unhealthy: bool = False) -> List[ModelInfo]:
        """
        Get all models from all providers with intelligent caching.
        
        Args:
            force_refresh: Force refresh from all providers, ignoring cache
            include_unhealthy: Include models from unhealthy providers
            
        Returns:
            Aggregated list of all available models
        """
        async with self._discovery_lock:
            # Determine which providers to query
            providers_to_query = []
            if include_unhealthy:
                providers_to_query = self.providers
            else:
                providers_to_query = [
                    p for p in self.providers 
                    if self._provider_health.get(p.get_provider_id(), ProviderHealthInfo(healthy=True)).healthy is not False
                ]
            
            # Collect models from all providers
            all_models = []
            discovery_tasks = []
            
            for provider in providers_to_query:
                discovery_tasks.append(
                    self._discover_models_from_provider(provider, force_refresh)
                )
            
            # Execute all discovery tasks concurrently
            provider_results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
            
            # Collect results
            for provider, result in zip(providers_to_query, provider_results):
                if isinstance(result, Exception):
                    logger.error(f"Error discovering models from {provider.get_provider_id()}: {result}")
                    continue
                
                if isinstance(result, list):
                    all_models.extend(result)
            
            logger.info(f"Aggregated {len(all_models)} models from {len(providers_to_query)} providers")
            return all_models
    
    async def _discover_models_from_provider(self, provider, force_refresh: bool) -> List[ModelInfo]:
        """Discover models from a single provider with caching"""
        provider_id = provider.get_provider_id()
        
        # Check cache first (unless forced refresh)
        if not force_refresh:
            cached_models = await self.cache.get_cached_models(provider_id)
            if cached_models is not None:
                logger.debug(f"Using cached models for provider {provider_id}")
                return cached_models
        
        # Cache miss or forced refresh - discover from provider
        try:
            logger.debug(f"Discovering models from provider {provider_id}")
            models = await provider.discover_models()
            
            # Cache the results
            await self.cache.cache_models(provider_id, models, self.default_cache_ttl)
            
            logger.debug(f"Discovered and cached {len(models)} models from {provider_id}")
            return models
            
        except Exception as e:
            logger.error(f"Failed to discover models from provider {provider_id}: {e}")
            # Mark provider as unhealthy
            health_info = self._provider_health.get(provider_id, ProviderHealthInfo())
            health_info.healthy = False
            health_info.last_checked = datetime.now()
            self._provider_health[provider_id] = health_info
            return []
    
    async def get_models_by_provider(self, provider_id: str, 
                                   force_refresh: bool = False) -> List[ModelInfo]:
        """Get models from a specific provider"""
        provider = next((p for p in self.providers if p.get_provider_id() == provider_id), None)
        if not provider:
            logger.warning(f"Provider {provider_id} not found")
            return []
        
        return await self._discover_models_from_provider(provider, force_refresh)
    
    async def refresh_provider_cache(self, provider_id: str = None):
        """Refresh cache for specific provider or all providers"""
        if provider_id:
            await self.cache.invalidate_cache(provider_id)
            logger.info(f"Refreshed cache for provider {provider_id}")
        else:
            await self.cache.invalidate_cache()
            logger.info("Refreshed cache for all providers")
    
    async def get_provider_health(self) -> Dict[str, bool]:
        """Get health status of all providers (backward compatibility)"""
        return {
            provider_id: health_info.healthy if health_info.healthy is not None else False
            for provider_id, health_info in self._provider_health.items()
        }
    
    async def get_provider_health_detailed(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed health status of all providers"""
        result = {}
        for provider_id, health_info in self._provider_health.items():
            result[provider_id] = {
                'healthy': health_info.healthy,
                'status': health_info.status,
                'last_checked': health_info.last_checked.isoformat() if health_info.last_checked else None,
                'last_healthy': health_info.last_healthy.isoformat() if health_info.last_healthy else None,
                'last_checked_ago': self._time_ago(health_info.last_checked) if health_info.last_checked else "never"
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
        if hasattr(self.cache, 'get_cache_stats'):
            cache_stats = await self.cache.get_cache_stats()
        
        return {
            'provider_count': len(self.providers),
            'provider_health': self._provider_health.copy(),
            'healthy_providers': sum(1 for healthy in self._provider_health.values() if healthy),
            'cache_stats': cache_stats,
            'default_cache_ttl': self.default_cache_ttl,
            'health_check_interval': self.health_check_interval
        }
    
    def close(self):
        """Clean shutdown of aggregator"""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
        
        if hasattr(self.cache, 'close'):
            self.cache.close()