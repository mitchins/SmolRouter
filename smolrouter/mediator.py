"""
Model mediator for orchestrating model discovery, resolution, and access control.

This module provides the central orchestration layer that coordinates between
model aggregation, strategy resolution, and access control to provide a unified
interface for model operations.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .interfaces import (
    IModelStrategy, IAccessControl, ModelInfo, ClientContext
)
from .caching import ModelAggregator, IModelCache
from .providers import IModelProvider

logger = logging.getLogger(__name__)


class ModelMediator:
    """
    Central orchestrator for model operations.
    
    This class coordinates between:
    - ModelAggregator: Discovers and caches models from providers
    - IModelStrategy: Resolves model requests and applies aliases
    - IAccessControl: Filters models based on client permissions
    """
    
    def __init__(self, 
                 aggregator: ModelAggregator,
                 strategy: IModelStrategy,
                 access_control: IAccessControl):
        self.aggregator = aggregator
        self.strategy = strategy
        self.access_control = access_control
        self._last_refresh = {}
    
    async def get_available_models(self, client: ClientContext, 
                                 force_refresh: bool = False,
                                 include_unhealthy: bool = False) -> List[ModelInfo]:
        """
        Get models available to a specific client.
        
        This is the main entry point for /v1/models endpoints and similar operations.
        
        Args:
            client: Client context for access control
            force_refresh: Force refresh from all providers
            include_unhealthy: Include models from unhealthy providers
            
        Returns:
            List of models accessible to the client
        """
        logger.debug(f"Getting available models for client {client.ip}")
        
        # Step 1: Get all models from aggregator
        all_models = await self.aggregator.get_all_models(
            force_refresh=force_refresh,
            include_unhealthy=include_unhealthy
        )
        
        logger.debug(f"Aggregator returned {len(all_models)} models")
        
        # Step 2: Apply strategy transformations (aliases, etc.)
        transformed_models = await self.strategy.apply_aliases(all_models)
        
        # Step 3: Apply access control filtering
        filtered_models = await self.access_control.filter_models(transformed_models, client)
        
        logger.info(f"Returning {len(filtered_models)} models to client {client.ip} "
                   f"(filtered from {len(all_models)} total)")
        
        return filtered_models
    
    async def resolve_model_for_request(self, requested_model: str, 
                                      client: ClientContext,
                                      force_refresh: bool = False) -> Optional[ModelInfo]:
        """
        Resolve a client's model request to an actual model.
        
        This is used for chat completions and other model-specific requests.
        
        Args:
            requested_model: The model name requested by the client
            client: Client context for access control and logging
            force_refresh: Force refresh from providers before resolution
            
        Returns:
            ModelInfo if resolution successful, None if not found or not allowed
        """
        logger.debug(f"Resolving model request '{requested_model}' for client {client.ip}")
        
        # Step 1: Get all available models for this client
        available_models = await self.get_available_models(
            client, 
            force_refresh=force_refresh,
            include_unhealthy=False  # Don't route to unhealthy providers
        )
        
        if not available_models:
            logger.warning(f"No models available for client {client.ip}")
            return None
        
        # Step 2: Use strategy to resolve the request
        resolved_model = await self.strategy.resolve_model_request(
            requested_model, 
            available_models
        )
        
        if resolved_model is None:
            logger.warning(f"Could not resolve model '{requested_model}' for client {client.ip}")
            return None
        
        # Step 3: Final access control check (should pass since model came from filtered list)
        if not await self.access_control.can_access_model(resolved_model, client):
            logger.warning(f"Access denied to resolved model '{resolved_model.id}' for client {client.ip}")
            return None
        
        logger.info(f"Resolved '{requested_model}' -> '{resolved_model.id}' for client {client.ip}")
        return resolved_model
    
    async def get_model_by_id(self, model_id: str, client: ClientContext) -> Optional[ModelInfo]:
        """
        Get a specific model by its full ID.
        
        Args:
            model_id: Full model ID (e.g., "llama3-70b@fast-kitten")
            client: Client context for access control
            
        Returns:
            ModelInfo if found and accessible, None otherwise
        """
        available_models = await self.get_available_models(client)
        
        for model in available_models:
            if model.id == model_id:
                return model
        
        return None
    
    async def get_models_by_provider(self, provider_id: str, client: ClientContext,
                                   force_refresh: bool = False) -> List[ModelInfo]:
        """
        Get models from a specific provider.
        
        Args:
            provider_id: Provider identifier
            client: Client context for access control
            force_refresh: Force refresh from provider
            
        Returns:
            List of accessible models from the provider
        """
        # Get models from specific provider
        provider_models = await self.aggregator.get_models_by_provider(
            provider_id, force_refresh
        )
        
        # Apply transformations and access control
        transformed_models = await self.strategy.apply_aliases(provider_models)
        filtered_models = await self.access_control.filter_models(transformed_models, client)
        
        return filtered_models
    
    async def refresh_models(self, provider_id: str = None):
        """
        Refresh model cache.
        
        Args:
            provider_id: Specific provider to refresh, or None for all providers
        """
        await self.aggregator.refresh_provider_cache(provider_id)
        self._last_refresh[provider_id or 'all'] = datetime.now()
        
        logger.info(f"Refreshed models for {provider_id or 'all providers'}")
    
    async def get_provider_health(self) -> Dict[str, bool]:
        """Get health status of all providers"""
        return await self.aggregator.get_provider_health()
    
    async def get_provider_health_detailed(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed health status of all providers"""
        return await self.aggregator.get_provider_health_detailed()
    
    async def get_mediator_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for monitoring"""
        aggregation_stats = await self.aggregator.get_aggregation_stats()
        
        stats = {
            'aggregation': aggregation_stats,
            'last_refresh': {
                k: v.isoformat() if isinstance(v, datetime) else v
                for k, v in self._last_refresh.items()
            },
            'strategy_type': type(self.strategy).__name__,
            'access_control_type': type(self.access_control).__name__
        }
        
        return stats
    
    async def validate_model_request(self, requested_model: str, client: ClientContext) -> Dict[str, Any]:
        """
        Validate a model request and return detailed information.
        
        Useful for debugging and API validation.
        
        Returns:
            Dict with validation results and details
        """
        result = {
            'requested_model': requested_model,
            'client_ip': client.ip,
            'client_user': client.user_id,
            'timestamp': datetime.now().isoformat(),
            'valid': False,
            'resolved_model': None,
            'available_models_count': 0,
            'resolution_path': [],
            'access_granted': False,
            'errors': []
        }
        
        try:
            # Get available models
            available_models = await self.get_available_models(client)
            result['available_models_count'] = len(available_models)
            
            if not available_models:
                result['errors'].append("No models available for client")
                return result
            
            # Try to resolve
            resolved_model = await self.strategy.resolve_model_request(
                requested_model, available_models
            )
            
            if resolved_model is None:
                result['errors'].append("Could not resolve model request")
                return result
            
            result['resolved_model'] = {
                'id': resolved_model.id,
                'name': resolved_model.name,
                'provider_id': resolved_model.provider_id,
                'provider_type': resolved_model.provider_type,
                'display_name': resolved_model.display_name
            }
            
            # Check access
            access_granted = await self.access_control.can_access_model(resolved_model, client)
            result['access_granted'] = access_granted
            
            if not access_granted:
                result['errors'].append("Access denied to resolved model")
                return result
            
            result['valid'] = True
            
        except Exception as e:
            result['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"Error validating model request: {e}")
        
        return result
    
    def close(self):
        """Clean shutdown of mediator"""
        self.aggregator.close()


class ModelMediatorFactory:
    """Factory for creating model mediator instances"""
    
    @classmethod
    def create_mediator(cls, 
                       providers: List[IModelProvider],
                       strategy_config: Dict[str, Any] = None,
                       access_control_config: Dict[str, Any] = None,
                       cache: IModelCache = None,
                       cache_ttl: int = 300) -> ModelMediator:
        """
        Create a complete model mediator from configuration.
        
        Args:
            providers: List of model providers
            strategy_config: Configuration for model strategy
            access_control_config: Configuration for access control
            cache: Cache implementation (optional)
            cache_ttl: Default cache TTL in seconds
            
        Returns:
            Configured ModelMediator instance
        """
        # Import here to avoid circular imports
        from .strategies import StrategyFactory
        from .access_control import AccessControlFactory
        from .caching import InMemoryModelCache
        
        # Create aggregator
        if cache is None:
            cache = InMemoryModelCache(default_ttl=cache_ttl)
        
        aggregator = ModelAggregator(providers, cache, cache_ttl)
        
        # Create strategy
        strategy = StrategyFactory.create_strategy(
            strategy_type="smart",
            config=strategy_config
        )
        
        # Create access control
        access_control = AccessControlFactory.create_access_control(access_control_config)
        
        return ModelMediator(aggregator, strategy, access_control)