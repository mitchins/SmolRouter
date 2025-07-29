"""
Dependency injection container for SmolRouter architecture.

This module provides a central container for managing dependencies and
configuration, enabling clean separation of concerns and easy testing.
"""

import os
import json
import yaml
import logging
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .interfaces import ProviderConfig, ClientContext
from .providers import ProviderFactory, IModelProvider
from .mediator import ModelMediatorFactory, ModelMediator
from .caching import InMemoryModelCache, NoOpModelCache, IModelCache

logger = logging.getLogger(__name__)


@dataclass
class SmolRouterConfig:
    """Complete configuration for SmolRouter"""
    # Provider configuration
    providers: List[Dict[str, Any]]
    
    # Legacy configuration (for backward compatibility)
    default_upstream: str = "http://localhost:8000"
    model_map: Dict[str, str] = None
    
    # Strategy configuration
    strategy: Dict[str, Any] = None
    
    # Access control configuration
    access_control: Dict[str, Any] = None
    
    # Caching configuration
    cache_enabled: bool = True
    cache_ttl: int = 300
    cache_cleanup_interval: int = 60
    
    # Health monitoring configuration
    enable_background_health_checks: bool = True
    health_check_interval: int = 60  # seconds
    
    # Routing configuration (legacy support)
    routes: List[Dict[str, Any]] = None
    servers: Dict[str, str] = None
    aliases: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.model_map is None:
            self.model_map = {}
        if self.strategy is None:
            self.strategy = {}
        if self.access_control is None:
            self.access_control = {}
        if self.routes is None:
            self.routes = []
        if self.servers is None:
            self.servers = {}
        if self.aliases is None:
            self.aliases = {}


class SmolRouterContainer:
    """
    Dependency injection container for SmolRouter.
    
    This container manages the lifecycle of all components and provides
    a single point of configuration and initialization.
    """
    
    def __init__(self, config: SmolRouterConfig = None):
        self.config = config or self._create_default_config()
        self._providers = None
        self._mediator = None
        self._cache = None
        self._initialized = False
    
    def _create_default_config(self) -> SmolRouterConfig:
        """Create default configuration from environment variables"""
        # Load legacy environment variables
        default_upstream = os.getenv("DEFAULT_UPSTREAM", "http://localhost:8000")
        raw_model_map = os.getenv("MODEL_MAP", "{}")
        routes_config_path = os.getenv("ROUTES_CONFIG", "routes.yaml")
        
        try:
            model_map = json.loads(raw_model_map)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse MODEL_MAP: {raw_model_map}")
            model_map = {}
        
        # Load routes configuration if it exists
        routes_data = self._load_routes_config(routes_config_path)
        
        # Create providers list from configuration
        providers = self._create_providers_from_legacy_config(default_upstream, routes_data)
        
        # Create strategy configuration
        strategy_config = {
            'model_map': model_map,
            'servers': routes_data.get('servers', {}),
            'aliases': routes_data.get('aliases', {}),
            'provider_priorities': self._extract_provider_priorities(routes_data)
        }
        
        # Create access control configuration from routes
        access_control_config = self._extract_access_control_from_routes(routes_data)
        
        # Load health check configuration from environment
        enable_background_health_checks = os.getenv("ENABLE_BACKGROUND_HEALTH_CHECKS", "true").lower() == "true" 
        health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))
        
        return SmolRouterConfig(
            providers=providers,
            default_upstream=default_upstream,
            model_map=model_map,
            strategy=strategy_config,
            access_control=access_control_config,
            routes=routes_data.get('routes', []),
            servers=routes_data.get('servers', {}),
            aliases=routes_data.get('aliases', {}),
            enable_background_health_checks=enable_background_health_checks,
            health_check_interval=health_check_interval
        )
    
    def _load_routes_config(self, config_path: str) -> Dict[str, Any]:
        """Load routes configuration from file"""
        if not os.path.exists(config_path):
            logger.info(f"No routes config file found at {config_path}")
            return {"routes": [], "servers": {}, "aliases": {}}
        
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    return json.load(f)
                else:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load routes config from {config_path}: {e}")
            return {"routes": [], "servers": {}, "aliases": {}}
    
    def _create_providers_from_legacy_config(self, default_upstream: str, 
                                           routes_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create providers list from legacy configuration"""
        providers = []
        
        # Add default upstream as primary provider
        providers.append({
            'name': 'default',
            'type': 'openai',  # Assume OpenAI-compatible by default
            'url': default_upstream,
            'priority': 0,
            'enabled': True
        })
        
        # Add servers from routes config
        servers = routes_data.get('servers', {})
        for i, (name, url) in enumerate(servers.items()):
            # Try to detect provider type from URL patterns
            provider_type = 'openai'  # Default assumption
            if '/api/tags' in url or 'ollama' in url.lower():
                provider_type = 'ollama'
            
            providers.append({
                'name': name,
                'type': provider_type,
                'url': url,
                'priority': i + 1,  # Default provider has priority 0
                'enabled': True
            })
        
        return providers
    
    def _extract_provider_priorities(self, routes_data: Dict[str, Any]) -> Dict[str, int]:
        """Extract provider priorities from routes configuration"""
        priorities = {}
        
        servers = routes_data.get('servers', {})
        for i, name in enumerate(servers.keys()):
            priorities[name] = i
        
        return priorities
    
    def _extract_access_control_from_routes(self, routes_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract access control rules from routes configuration"""
        # For now, return no access control
        # Future enhancement: analyze routes for IP-based rules
        return {'type': 'none'}
    
    async def initialize(self):
        """Initialize all components"""
        if self._initialized:
            return
        
        logger.info("Initializing SmolRouter container...")
        
        # Create cache
        self._cache = self._create_cache()
        
        # Create providers
        self._providers = self._create_providers()
        
        # Create mediator
        self._mediator = self._create_mediator()
        
        # Perform startup health checks
        await self._perform_startup_health_checks()
        
        # Start background health monitoring
        if self.config.enable_background_health_checks:
            self._start_background_health_monitoring()
        
        self._initialized = True
        logger.info(f"SmolRouter container initialized with {len(self._providers)} providers")
    
    async def _perform_startup_health_checks(self):
        """Perform health checks on all providers during startup"""
        logger.info("Performing startup health checks for all providers...")
        
        # Use the mediator's aggregator to perform initial health checks
        # This will populate the detailed health tracking
        await self._mediator.aggregator._update_provider_health()
        
        # Get detailed health status
        detailed_health = await self._mediator.get_provider_health_detailed()
        
        healthy_count = 0
        total_count = len(self._providers)
        
        for provider in self._providers:
            provider_id = provider.get_provider_id()
            health_info = detailed_health.get(provider_id, {})
            is_healthy = health_info.get('healthy', False)
            
            try:
                if is_healthy:
                    healthy_count += 1
                    logger.info(f"✓ Provider {provider_id} ({provider.get_provider_type()}) is healthy")
                    
                    # Proactively discover models to populate cache
                    try:
                        models = await provider.discover_models()
                        if models:
                            # Cache the models with default TTL
                            await self._cache.cache_models(
                                provider_id, 
                                models, 
                                self.config.cache_ttl
                            )
                            logger.info(f"✓ Cached {len(models)} models for provider {provider_id}")
                        else:
                            logger.warning(f"⚠ Provider {provider_id} is healthy but has no models")
                    except Exception as e:
                        logger.warning(f"⚠ Could not discover models for healthy provider {provider_id}: {e}")
                else:
                    status = health_info.get('status', 'unknown')
                    logger.warning(f"✗ Provider {provider_id} ({provider.get_provider_type()}) is {status}")
            except Exception as e:
                logger.error(f"✗ Startup health check failed for provider {provider_id}: {e}")
        
        logger.info(f"Startup health checks complete: {healthy_count}/{total_count} providers healthy")
        
        if healthy_count == 0:
            logger.warning("⚠ No providers are healthy! SmolRouter may not function correctly.")
        elif healthy_count < total_count:
            logger.warning(f"⚠ {total_count - healthy_count} provider(s) are unhealthy. Check configuration and connectivity.")
    
    def _start_background_health_monitoring(self):
        """Start background task for periodic health monitoring"""
        logger.info(f"Starting background health monitoring (interval: {self.config.health_check_interval}s)")
        
        async def monitor_health():
            while True:
                try:
                    await asyncio.sleep(self.config.health_check_interval)
                    await self._perform_background_health_check()
                except Exception as e:
                    logger.error(f"Error in background health monitoring: {e}")
        
        # Start the background task
        asyncio.create_task(monitor_health())
    
    async def _perform_background_health_check(self):
        """Perform periodic health checks and refresh cache for healthy providers"""
        logger.debug("Performing background health checks...")
        
        healthy_providers = []
        unhealthy_providers = []
        
        for provider in self._providers:
            try:
                is_healthy = await provider.health_check()
                if is_healthy:
                    healthy_providers.append(provider)
                    
                    # Check if cache is stale and refresh if needed
                    if not await self._cache.is_cache_valid(provider.get_provider_id()):
                        try:
                            models = await provider.discover_models()
                            await self._cache.cache_models(
                                provider.get_provider_id(), 
                                models, 
                                self.config.cache_ttl
                            )
                            logger.debug(f"Refreshed cache for provider {provider.get_provider_id()}: {len(models)} models")
                        except Exception as e:
                            logger.warning(f"Failed to refresh models for {provider.get_provider_id()}: {e}")
                else:
                    unhealthy_providers.append(provider)
            except Exception as e:
                logger.warning(f"Background health check failed for {provider.get_provider_id()}: {e}")
                unhealthy_providers.append(provider)
        
        # Log any changes in provider health status
        if unhealthy_providers:
            unhealthy_names = [p.get_provider_id() for p in unhealthy_providers]
            logger.debug(f"Background health check: {len(unhealthy_names)} unhealthy providers: {unhealthy_names}")
    
    def _create_cache(self) -> IModelCache:
        """Create cache implementation"""
        if not self.config.cache_enabled:
            return NoOpModelCache()
        
        return InMemoryModelCache(
            default_ttl=self.config.cache_ttl,
            cleanup_interval=self.config.cache_cleanup_interval
        )
    
    def _create_providers(self) -> List[IModelProvider]:
        """Create provider instances"""
        providers = []
        
        for provider_config in self.config.providers:
            try:
                config = ProviderConfig(**provider_config)
                if config.enabled:
                    provider = ProviderFactory.create_provider(config)
                    providers.append(provider)
                    logger.info(f"Created provider: {config.name} ({config.type}) -> {config.url}")
            except Exception as e:
                logger.error(f"Failed to create provider from config {provider_config}: {e}")
        
        return providers
    
    def _create_mediator(self) -> ModelMediator:
        """Create model mediator"""
        return ModelMediatorFactory.create_mediator(
            providers=self._providers,
            strategy_config=self.config.strategy,
            access_control_config=self.config.access_control,
            cache=self._cache,
            cache_ttl=self.config.cache_ttl
        )
    
    async def get_mediator(self) -> ModelMediator:
        """Get the model mediator (singleton)"""
        if not self._initialized:
            await self.initialize()
        return self._mediator
    
    def get_providers(self) -> List[IModelProvider]:
        """Get list of providers"""
        if not self._initialized:
            raise RuntimeError("Container not initialized")
        return self._providers
    
    def get_cache(self) -> IModelCache:
        """Get cache implementation"""
        if not self._initialized:
            raise RuntimeError("Container not initialized")
        return self._cache
    
    def create_client_context(self, ip: str, auth_payload: Dict[str, Any] = None,
                            user_agent: str = None, headers: Dict[str, str] = None) -> ClientContext:
        """Create client context for requests"""
        return ClientContext(
            ip=ip,
            auth_payload=auth_payload,
            user_agent=user_agent,
            headers=headers or {}
        )
    
    async def get_legacy_smart_router(self):
        """
        Get a legacy-compatible smart router for backward compatibility.
        
        This allows existing app.py code to work with minimal changes.
        """
        if not self._initialized:
            await self.initialize()
        
        # Create a wrapper that mimics the old SmartRouter interface
        return LegacySmartRouterAdapter(self._mediator, self.config)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        if not self._initialized:
            await self.initialize()
        
        provider_health = await self._mediator.get_provider_health()
        stats = await self._mediator.get_mediator_stats()
        
        return {
            'status': 'healthy',
            'initialized': self._initialized,
            'provider_count': len(self._providers),
            'provider_health': provider_health,
            'healthy_providers': sum(1 for healthy in provider_health.values() if healthy),
            'cache_enabled': self.config.cache_enabled,
            'cache_ttl': self.config.cache_ttl,
            'stats': stats
        }
    
    async def close(self):
        """Clean shutdown of container"""
        if self._mediator:
            self._mediator.close()
        
        if self._cache and hasattr(self._cache, 'close'):
            self._cache.close()
        
        self._initialized = False
        logger.info("SmolRouter container shut down")


class LegacySmartRouterAdapter:
    """
    Adapter to make the new architecture compatible with existing app.py code.
    
    This provides the same interface as the old SmartRouter class.
    """
    
    def __init__(self, mediator: ModelMediator, config: SmolRouterConfig):
        self.mediator = mediator
        self.config = config
    
    async def route_request(self, source_ip: str, model: str, request_payload: Dict[str, Any],
                           path: str, headers: Dict[str, str], timeout: float):
        """
        Route a request using the new architecture.
        
        Returns: (response_data, status_code, upstream_used)
        """
        client = ClientContext(ip=source_ip, headers=headers)
        
        # Resolve the model
        resolved_model = await self.mediator.resolve_model_for_request(model, client)
        
        if resolved_model is None:
            return {
                "error": "model_not_found",
                "message": f"Model '{model}' not found or not accessible"
            }, 404, "none"
        
        # Use the original routing logic for now (fallback to legacy system)
        # This would need to be enhanced to actually make the HTTP requests
        # For now, return a placeholder
        return {
            "error": "not_implemented",
            "message": "Legacy adapter routing not fully implemented"
        }, 501, resolved_model.endpoint


# Global container instance
_container: Optional[SmolRouterContainer] = None


def get_container() -> SmolRouterContainer:
    """Get or create the global container instance"""
    global _container
    if _container is None:
        _container = SmolRouterContainer()
    return _container


def set_container(container: SmolRouterContainer):
    """Set a custom container instance (useful for testing)"""
    global _container
    _container = container


async def initialize_container():
    """Initialize the global container"""
    container = get_container()
    await container.initialize()
    return container