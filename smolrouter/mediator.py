"""
Model mediator for orchestrating model discovery, resolution, and access control.

This module provides the central orchestration layer that coordinates between
model aggregation, strategy resolution, and access control to provide a unified
interface for model operations.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .interfaces import IModelStrategy, IAccessControl, ModelInfo, ClientContext
from .caching import ModelAggregator, IModelCache
from .providers import IModelProvider
from .google_genai_provider import GoogleGenAIProvider
from .dummy_provider import DummyProvider
from .load_balancer import model_load_balancer
from .request_metadata import RequestMetadata

logger = logging.getLogger(__name__)


def _should_use_openai_completion_tokens(model_name: Any) -> bool:
    return isinstance(model_name, str) and model_name.lower().startswith("gpt-5")


def _normalize_resolved_openai_request_payload(resolved_model: ModelInfo, request_payload: Dict[str, Any]) -> None:
    """Apply OpenAI-only request compatibility after provider/model resolution."""
    if resolved_model.provider_type != "openai":
        return

    if not _should_use_openai_completion_tokens(request_payload.get("model")):
        return

    if "max_completion_tokens" not in request_payload and "max_tokens" in request_payload:
        request_payload["max_completion_tokens"] = request_payload.pop("max_tokens")
        logger.debug("Remapped max_tokens -> max_completion_tokens for model %s", request_payload.get("model"))
    else:
        request_payload.pop("max_tokens", None)


class ModelMediator:
    """
    Central orchestrator for model operations.

    This class coordinates between:
    - ModelAggregator: Discovers and caches models from providers
    - IModelStrategy: Resolves model requests and applies aliases
    - IAccessControl: Filters models based on client permissions
    """

    def __init__(self, aggregator: ModelAggregator, strategy: IModelStrategy, access_control: IAccessControl):
        self.aggregator = aggregator
        self.strategy = strategy
        self.access_control = access_control
        self._last_refresh = {}
        self._models_registered_in_lb = False
        self._active_lb_instance = None  # Track active load balanced instance

    async def get_available_models(
        self, client: ClientContext, force_refresh: bool = False, include_unhealthy: bool = False
    ) -> List[ModelInfo]:
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
            force_refresh=force_refresh, include_unhealthy=include_unhealthy
        )

        logger.debug(f"Aggregator returned {len(all_models)} models")

        # Step 2: Apply strategy transformations (aliases, etc.)
        transformed_models = await self.strategy.apply_aliases(all_models)

        # Step 3: Apply access control filtering
        filtered_models = await self.access_control.filter_models(transformed_models, client)

        logger.debug(
            f"Returning {len(filtered_models)} models to client {client.ip} (filtered from {len(all_models)} total)"
        )

        return filtered_models

    def _register_models_in_load_balancer(self, models: List[ModelInfo]) -> None:
        """Register discovered models with the load balancer for instance distribution."""
        if self._models_registered_in_lb:
            return

        for model in models:
            # Register each model instance with its provider information
            # Use model.name (raw) instead of model.id (tagged) for proper load balancing
            model_load_balancer.register_model_instance(
                model_id=model.name, provider_id=model.provider_id, provider_url=model.endpoint
            )

        self._models_registered_in_lb = True
        logger.info(f"Registered {len(models)} models in load balancer")

    async def _resolve_model_via_load_balancer(
        self, requested_model: str, available_models: List[ModelInfo], pinned_provider: Optional[str] = None
    ) -> Optional[ModelInfo]:
        """Resolve a model name through the load balancer and attach LB metadata.

        pinned_provider constrains selection to one provider so an explicit
        "model [provider]" request is honored even when the bare name is offered
        by several providers.
        """
        instance = await model_load_balancer.select_instance(requested_model, provider_id=pinned_provider)
        if not instance:
            return None

        logger.debug(f"Load balancer selected instance: {instance.model_id} for request: {requested_model}")

        for model in available_models:
            if model.name == instance.model_id and model.endpoint == instance.provider_url:
                model._lb_selected_instance = instance.model_id
                model._lb_instance = instance
                return model

        # select_instance() already incremented active_requests for this instance.
        # If we can't map it back to an available model nothing downstream will
        # ever decrement it, so release it here to avoid leaking the counter.
        logger.warning(
            f"Load balancer selected instance {instance.model_id} but no matching model found; releasing it"
        )
        await model_load_balancer.end_request(instance, 0.0, success=False)
        return None

    async def resolve_model_for_request(
        self, requested_model: str, client: ClientContext, force_refresh: bool = False
    ) -> Optional[ModelInfo]:
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
            include_unhealthy=False,  # Don't route to unhealthy providers
        )

        if not available_models:
            logger.warning(f"No models available for client {client.ip}")
            return None

        # Register models in load balancer if not already done
        self._register_models_in_load_balancer(available_models)

        # Step 2: Resolve aliases and other strategy rules before load balancing.
        # This preserves legacy remaps like gpt-4 -> llama3-70b even when the
        # original request name also exists as a concrete model.
        resolved_model = await self.strategy.resolve_model_request(requested_model, available_models)

        # If the client explicitly pinned a provider ("model [provider]" / full id),
        # constrain load balancing to that provider so an overlapping bare name on
        # another provider can't be selected instead.
        pinned_provider = (
            resolved_model.provider_id
            if resolved_model and requested_model in (resolved_model.display_name, resolved_model.id)
            else None
        )

        candidate_model_name = resolved_model.name if resolved_model else requested_model
        lb_resolved_model = await self._resolve_model_via_load_balancer(
            candidate_model_name, available_models, pinned_provider=pinned_provider
        )
        if lb_resolved_model:
            return lb_resolved_model

        if resolved_model is None:
            logger.warning(f"Could not resolve model '{requested_model}' for client {client.ip}")
            return None

        # Step 4: Final access control check (should pass since model came from filtered list)
        if not await self.access_control.can_access_model(resolved_model, client):
            logger.warning(f"Access denied to resolved model '{resolved_model.id}' for client {client.ip}")
            return None

        logger.debug(f"Resolved '{requested_model}' -> '{resolved_model.id}' for client {client.ip}")
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

    async def get_models_by_provider(
        self, provider_id: str, client: ClientContext, force_refresh: bool = False
    ) -> List[ModelInfo]:
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
        provider_models = await self.aggregator.get_models_by_provider(provider_id, force_refresh)

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
        self._last_refresh[provider_id or "all"] = datetime.now()

        logger.info(f"Refreshed models for {provider_id or 'all providers'}")

    def get_provider_health(self) -> Dict[str, bool]:
        """Get health status of all providers (synchronous cached read)"""
        return self.aggregator.get_provider_health()

    def get_provider_health_detailed(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed health status of all providers (synchronous cached read)"""
        return self.aggregator.get_provider_health_detailed()

    async def get_mediator_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for monitoring"""
        aggregation_stats = await self.aggregator.get_aggregation_stats()

        stats = {
            "aggregation": aggregation_stats,
            "last_refresh": {k: v.isoformat() if isinstance(v, datetime) else v for k, v in self._last_refresh.items()},
            "strategy_type": type(self.strategy).__name__,
            "access_control_type": type(self.access_control).__name__,
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
            "requested_model": requested_model,
            "client_ip": client.ip,
            "client_user": client.user_id,
            "timestamp": datetime.now().isoformat(),
            "valid": False,
            "resolved_model": None,
            "available_models_count": 0,
            "resolution_path": [],
            "access_granted": False,
            "errors": [],
        }

        try:
            # Get available models
            available_models = await self.get_available_models(client)
            result["available_models_count"] = len(available_models)

            if not available_models:
                result["errors"].append("No models available for client")
                return result

            # Try to resolve
            resolved_model = await self.strategy.resolve_model_request(requested_model, available_models)

            if resolved_model is None:
                result["errors"].append("Could not resolve model request")
                return result

            result["resolved_model"] = {
                "id": resolved_model.id,
                "name": resolved_model.name,
                "provider_id": resolved_model.provider_id,
                "provider_type": resolved_model.provider_type,
                "display_name": resolved_model.display_name,
            }

            # Check access
            access_granted = await self.access_control.can_access_model(resolved_model, client)
            result["access_granted"] = access_granted

            if not access_granted:
                result["errors"].append("Access denied to resolved model")
                return result

            result["valid"] = True

        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")
            logger.exception("Error validating model request")

        return result

    def _attach_lb_instance(self, lb_instance: Any, metadata: Optional[RequestMetadata]) -> Optional[RequestMetadata]:
        if not lb_instance:
            return metadata

        if metadata is None:
            metadata = RequestMetadata()
        metadata.lb_instance = lb_instance
        return metadata

    def _apply_resolved_model_to_payload(self, resolved_model: ModelInfo, request_payload: Dict[str, Any]) -> None:
        if hasattr(resolved_model, "_lb_selected_instance") and "model" in request_payload:
            original_model = request_payload["model"]
            request_payload["model"] = resolved_model._lb_selected_instance
            logger.debug(
                f"Load balancer: mutated request model '{original_model}' -> '{resolved_model._lb_selected_instance}'"
            )
            return

        if request_payload.get("model") != resolved_model.name:
            request_payload["model"] = resolved_model.name

    @staticmethod
    def _google_error_status_code(error_message: str) -> int:
        error_lower = error_message.lower()
        if "quota exhausted" in error_lower or "429" in error_lower:
            return 429
        if "permission denied" in error_lower or "403" in error_lower:
            return 403
        if "invalid argument" in error_lower or "400" in error_lower:
            return 400
        return 500

    async def _handle_google_provider_request(
        self,
        provider: GoogleGenAIProvider,
        resolved_model: ModelInfo,
        request_payload: Dict[str, Any],
        path: str,
        lb_instance: Any,
    ) -> Tuple[Dict[str, Any], int, str, Any]:
        try:
            response_data, metadata = await provider.generate_completion(request_payload, path)
            return (
                response_data,
                200,
                f"google-genai:{resolved_model.provider_id}",
                self._attach_lb_instance(lb_instance, metadata),
            )
        except Exception as e:
            error_msg = str(e)
            error_metadata = RequestMetadata(
                api_key_suffix=getattr(e, "api_key_suffix", None),
                proxy_used=getattr(e, "proxy_used", None),
                provider_id=getattr(e, "provider_id", resolved_model.provider_id),
                model_name=getattr(e, "model_name", resolved_model.name),
                api_key_index=getattr(e, "api_key_index", None),
                api_key_total=getattr(e, "api_key_total", None),
            )

            return (
                {"error": {"type": "api_error", "message": error_msg, "provider": "google-genai"}},
                self._google_error_status_code(error_msg),
                f"google-genai:{resolved_model.provider_id}",
                self._attach_lb_instance(lb_instance, error_metadata),
            )

    async def _handle_openai_compatible_provider_request(
        self,
        provider: Any,
        resolved_model: ModelInfo,
        request_payload: Dict[str, Any],
        headers: Dict[str, str],
        path: str,
        lb_instance: Any,
    ) -> Tuple[Dict[str, Any], int, str, Any]:
        base_metadata = RequestMetadata(provider_id=resolved_model.provider_id, model_name=resolved_model.name)
        try:
            response_data, status_code = await provider.generate_completion(request_payload, headers, path)
            return (
                response_data,
                status_code,
                f"{resolved_model.provider_type}:{resolved_model.provider_id}",
                self._attach_lb_instance(lb_instance, base_metadata),
            )
        except Exception as e:
            logger.exception("Error handling %s provider request", resolved_model.provider_type)
            return (
                {"error": {"type": "api_error", "message": str(e), "provider": resolved_model.provider_type}},
                500,
                f"{resolved_model.provider_type}:{resolved_model.provider_id}",
                self._attach_lb_instance(lb_instance, base_metadata),
            )

    async def _handle_dummy_provider_request(
        self,
        provider: DummyProvider,
        resolved_model: ModelInfo,
        request_payload: Dict[str, Any],
        headers: Dict[str, str],
        lb_instance: Any,
    ) -> Tuple[Dict[str, Any], int, str, Any]:
        try:
            response_data = await provider.make_request(request_payload, headers)
            return (
                response_data,
                200,
                f"dummy:{resolved_model.provider_id}",
                self._attach_lb_instance(lb_instance, None),
            )
        except Exception as e:
            logger.exception("Dummy provider error for %s", resolved_model.provider_id)
            return (
                {"error": {"type": "api_error", "message": str(e), "provider": "dummy"}},
                500,
                f"dummy:{resolved_model.provider_id}",
                self._attach_lb_instance(lb_instance, None),
            )

    async def _route_request_internal(
        self,
        resolved_model: ModelInfo,
        request_payload: Dict[str, Any],
        path: str,
        headers: Dict[str, str],
        lb_instance: Any,
    ) -> Tuple[Dict[str, Any], int, str, Any]:
        provider = self._get_provider_by_id(resolved_model.provider_id)
        if not provider:
            return (
                {"error": {"type": "internal_server_error", "message": "Provider not available"}},
                500,
                resolved_model.endpoint,
                self._attach_lb_instance(lb_instance, None),
            )

        if isinstance(provider, GoogleGenAIProvider):
            return await self._handle_google_provider_request(provider, resolved_model, request_payload, path, lb_instance)

        if hasattr(provider, "generate_completion") and resolved_model.provider_type in {"openai", "zai-coding"}:
            return await self._handle_openai_compatible_provider_request(
                provider,
                resolved_model,
                request_payload,
                headers,
                path,
                lb_instance,
            )

        if isinstance(provider, DummyProvider):
            return await self._handle_dummy_provider_request(provider, resolved_model, request_payload, headers, lb_instance)

        return (
            {
                "error": {
                    "type": "not_implemented",
                    "message": f"HTTP routing for {resolved_model.provider_type} not implemented in new architecture",
                }
            },
            501,
            resolved_model.endpoint,
            self._attach_lb_instance(lb_instance, None),
        )

    async def route_request(
        self,
        source_ip: str,
        model: str,
        request_payload: Dict[str, Any],
        path: str,
        headers: Dict[str, str],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], int, str, Any]:
        """
        Route a request to the appropriate provider and execute it.

        This method handles the complete request lifecycle:
        1. Resolve the requested model to a specific provider
        2. Route the request to that provider
        3. Handle provider-specific request/response transformations

        Args:
            source_ip: Client IP address
            model: Requested model name
            request_payload: OpenAI-format request payload
            path: Request path (e.g., "/v1/chat/completions")
            headers: Request headers
            args/kwargs: Optional timeout argument for backward compatibility

        Returns:
            Tuple of (response_data, status_code, upstream_used, metadata)
        """
        client = kwargs.get("client_context") or ClientContext(ip=source_ip, headers=headers)
        timeout = kwargs.get("timeout")
        if timeout is None and args:
            timeout = args[0]
        if timeout is None:
            timeout = 30.0

        # select_instance() (inside resolve_model_for_request) atomically
        # increments the chosen instance's active_requests counter. Once that
        # happens we MUST return the instance in the metadata on every path -
        # success, timeout, or error - otherwise the counter is never
        # decremented and the load balancer permanently believes the instance is
        # busy (the active_requests "leak"). We hoist lb_instance into the outer
        # scope so the except handlers below can attach it. Resolution stays
        # inside the timeout so a slow resolution still yields a 504.
        lb_instance: Any = None
        try:
            async with asyncio.timeout(timeout):
                resolved_model = await self.resolve_model_for_request(model, client)
                if resolved_model is None:
                    return (
                        {
                            "error": {
                                "type": "invalid_request_error",
                                "message": f"Model '{model}' not found or not accessible",
                            }
                        },
                        404,
                        "none",
                        None,
                    )

                lb_instance = getattr(resolved_model, "_lb_instance", None)
                self._apply_resolved_model_to_payload(resolved_model, request_payload)
                _normalize_resolved_openai_request_payload(resolved_model, request_payload)

                return await self._route_request_internal(
                    resolved_model, request_payload, path, headers, lb_instance
                )

        except asyncio.CancelledError:
            # Client disconnect/shutdown. CancelledError is a BaseException, so it
            # bypasses the handlers below and we can't return metadata (we must
            # propagate the cancellation). select_instance() already incremented
            # active_requests, so release it here - best-effort, so a cleanup
            # failure can never mask the original cancellation.
            if lb_instance is not None:
                try:
                    await model_load_balancer.end_request(lb_instance, 0.0, success=False)
                except Exception:
                    logger.exception("Failed to release load balancer instance on cancellation")
            raise

        except TimeoutError:
            logger.exception("Request timeout while routing model '%s'", model)
            return (
                {"error": {"type": "timeout_error", "message": "Request timed out"}},
                504,
                "timeout",
                self._attach_lb_instance(lb_instance, None),
            )

        except Exception:
            logger.exception("Error routing request for model '%s'", model)
            return (
                {"error": {"type": "internal_server_error", "message": "Request routing failed"}},
                500,
                "unknown",
                self._attach_lb_instance(lb_instance, None),
            )

    def _get_provider_by_id(self, provider_id: str) -> Optional[IModelProvider]:
        """Get a provider instance by ID"""
        for provider in self.aggregator.providers:
            if provider.get_provider_id() == provider_id:
                return provider
        return None

    def close(self):
        """Clean shutdown of mediator"""
        self.aggregator.close()


class ModelMediatorFactory:
    """Factory for creating model mediator instances"""

    @classmethod
    def create_mediator(
        cls,
        providers: List[IModelProvider],
        strategy_config: Dict[str, Any] = None,
        access_control_config: Dict[str, Any] = None,
        cache: IModelCache = None,
        cache_ttl: int = 300,
    ) -> ModelMediator:
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
        strategy = StrategyFactory.create_strategy(strategy_type="smart", config=strategy_config)

        # Create access control
        access_control = AccessControlFactory.create_access_control(access_control_config)

        return ModelMediator(aggregator, strategy, access_control)
