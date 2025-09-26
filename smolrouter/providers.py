"""
Model provider implementations for different upstream services.

This module contains concrete implementations of IModelProvider for various
AI model serving platforms like Ollama and OpenAI-compatible APIs.
"""

import logging
import httpx
from typing import List, Dict, Any, Tuple
from urllib.parse import urljoin

from .interfaces import IModelProvider, ModelInfo, ProviderConfig
from .google_genai_provider import GoogleGenAIProvider, GoogleGenAIConfig
from .anthropic_provider import AnthropicProvider, AnthropicConfig

logger = logging.getLogger(__name__)


class BaseModelProvider(IModelProvider):
    """Base implementation with common provider functionality"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._validate_config()
    
    def _validate_config(self):
        """Validate provider configuration"""
        if not self.config.name:
            raise ValueError("Provider name is required")
        if not self.config.url:
            raise ValueError("Provider URL is required")
        if not self.config.url.startswith(('http://', 'https://')):
            raise ValueError(f"Provider URL must include protocol: {self.config.url}")
    
    def get_provider_id(self) -> str:
        return self.config.name
    
    def get_provider_type(self) -> str:
        return self.config.type
    
    def get_endpoint(self) -> str:
        return self.config.url
    
    async def health_check(self) -> bool:
        """Default health check implementation"""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                health_url = self._get_health_check_url()
                headers = self._get_headers()
                response = await client.get(health_url, headers=headers)
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed for {self.get_provider_id()}: {e}")
            return False
    
    def _get_health_check_url(self) -> str:
        """Override in subclasses to provide specific health check endpoints"""
        return self.config.url
    
    def _create_model_info(self, model_id: str, model_name: str = None, 
                          aliases: List[str] = None, metadata: Dict[str, Any] = None) -> ModelInfo:
        """Helper to create ModelInfo with provider context"""
        return ModelInfo(
            id=f"{model_id}@{self.get_provider_id()}",
            name=model_name or model_id,
            provider_id=self.get_provider_id(),
            provider_type=self.get_provider_type(),
            endpoint=self.get_endpoint(),
            aliases=aliases or [],
            metadata=metadata or {}
        )


class OllamaProvider(BaseModelProvider):
    """Provider for Ollama-compatible model servers"""
    
    def __init__(self, config: ProviderConfig):
        config.type = "ollama"  # Ensure type is set
        super().__init__(config)
    
    def _get_health_check_url(self) -> str:
        return urljoin(self.config.url, "/api/tags")
    
    async def discover_models(self) -> List[ModelInfo]:
        """Discover models from Ollama /api/tags endpoint"""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                url = urljoin(self.config.url, "/api/tags")
                headers = self._get_headers()
                
                logger.debug(f"Discovering Ollama models from {url}")
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                models = []
                
                for model_data in data.get('models', []):
                    model_name = model_data.get('name', 'unknown')
                    
                    # Extract metadata
                    metadata = {
                        'size': model_data.get('size', 0),
                        'modified_at': model_data.get('modified_at'),
                        'digest': model_data.get('digest'),
                        'details': model_data.get('details', {})
                    }
                    
                    # Create aliases (original name and any variations)
                    aliases = [model_name]
                    
                    # Handle model name variations (e.g., llama3:8b -> llama3-8b)
                    if ':' in model_name:
                        normalized = model_name.replace(':', '-')
                        aliases.append(normalized)
                    
                    model_info = self._create_model_info(
                        model_id=model_name,
                        model_name=model_name,
                        aliases=aliases,
                        metadata=metadata
                    )
                    
                    models.append(model_info)
                    logger.debug(f"Discovered Ollama model: {model_info.id}")
                
                logger.info(f"Discovered {len(models)} models from Ollama provider {self.get_provider_id()}")
                return models
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error discovering Ollama models from {self.get_provider_id()}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error discovering Ollama models from {self.get_provider_id()}: {e}")
            return []
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Ollama requests"""
        headers = {'Content-Type': 'application/json'}
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        return headers


class OpenAIProvider(BaseModelProvider):
    """Provider for OpenAI-compatible model servers"""

    def __init__(self, config: ProviderConfig):
        config.type = "openai"  # Ensure type is set
        super().__init__(config)

    def _get_health_check_url(self) -> str:
        return urljoin(self.config.url, "/v1/models")

    async def health_check(self) -> bool:
        """OpenAI provider health check - always healthy when configured for client passthrough"""
        if not self.config.api_key:
            # No API key means client passthrough mode - always healthy
            logger.debug(f"OpenAI provider {self.get_provider_id()} in client passthrough mode - marking as healthy")
            return True

        # If we have an API key, do normal health check
        return await super().health_check()
    
    async def discover_models(self) -> List[ModelInfo]:
        """Discover models from OpenAI /v1/models endpoint or return static list"""
        # If no API key, return static list for client passthrough
        if not self.config.api_key:
            return self._get_static_openai_models()

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                url = urljoin(self.config.url, "/v1/models")
                headers = self._get_headers()

                logger.debug(f"Discovering OpenAI models from {url}")
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                data = response.json()
                models = []

                for model_data in data.get('data', []):
                    model_id = model_data.get('id', 'unknown')

                    # Extract metadata
                    metadata = {
                        'object': model_data.get('object'),
                        'created': model_data.get('created'),
                        'owned_by': model_data.get('owned_by'),
                        'permission': model_data.get('permission', []),
                        'root': model_data.get('root'),
                        'parent': model_data.get('parent')
                    }

                    # Create aliases (original ID and common variations)
                    aliases = [model_id]

                    model_info = self._create_model_info(
                        model_id=model_id,
                        model_name=model_id,
                        aliases=aliases,
                        metadata=metadata
                    )

                    models.append(model_info)
                    logger.debug(f"Discovered OpenAI model: {model_info.id}")

                logger.info(f"Discovered {len(models)} models from OpenAI provider {self.get_provider_id()}")
                return models

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.info(f"OpenAI API authentication failed, falling back to static model list")
                return self._get_static_openai_models()
            logger.error(f"HTTP error discovering OpenAI models from {self.get_provider_id()}: {e}")
            return self._get_static_openai_models()
        except Exception as e:
            logger.error(f"Error discovering OpenAI models from {self.get_provider_id()}: {e}")
            return self._get_static_openai_models()

    def _get_static_openai_models(self) -> List[ModelInfo]:
        """Return static list of OpenAI models from JSON file"""
        import json
        import os
        from pathlib import Path

        try:
            # Get path relative to this file
            current_dir = Path(__file__).parent
            models_file = current_dir / "models" / "openai-models-2025-september.json"

            if not models_file.exists():
                logger.warning(f"OpenAI models file not found at {models_file}, falling back to minimal set")
                return self._get_fallback_models()

            with open(models_file, 'r') as f:
                data = json.load(f)

            models = []
            for model_data in data.get('data', []):
                model_id = model_data.get('id', 'unknown')

                # Extract metadata
                metadata = {
                    'object': model_data.get('object'),
                    'created': model_data.get('created'),
                    'owned_by': model_data.get('owned_by'),
                    'static': True  # Mark as static definition
                }

                model_info = self._create_model_info(
                    model_id=model_id,
                    model_name=model_id,
                    aliases=[model_id],
                    metadata=metadata
                )
                models.append(model_info)

            logger.info(f"Loaded {len(models)} OpenAI models from static file for client passthrough from {self.get_provider_id()}")
            return models

        except Exception as e:
            logger.error(f"Error loading static OpenAI models file: {e}, falling back to minimal set")
            return self._get_fallback_models()

    def _get_fallback_models(self) -> List[ModelInfo]:
        """Minimal fallback model set when JSON file fails"""
        fallback_models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]

        models = []
        for model_name in fallback_models:
            model_info = self._create_model_info(
                model_id=model_name,
                model_name=model_name,
                aliases=[model_name],
                metadata={
                    'object': 'model',
                    'owned_by': 'openai',
                    'static': True,
                    'fallback': True
                }
            )
            models.append(model_info)

        logger.info(f"Providing {len(models)} fallback OpenAI models for {self.get_provider_id()}")
        return models
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenAI requests"""
        headers = {'Content-Type': 'application/json'}
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        return headers

    async def generate_completion(self, openai_request: Dict[str, Any], client_headers: Dict[str, str] = None, endpoint: str = "/v1/chat/completions") -> Tuple[Dict[str, Any], int]:
        """Generate completion by passing through to OpenAI API"""
        try:
            # Use client's Authorization header if provided, otherwise fall back to config
            headers = self._get_headers()
            if client_headers:
                for key, value in client_headers.items():
                    # Convert bytes to string if needed
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')

                    # Handle authorization header (case insensitive)
                    if key.lower() == 'authorization':
                        headers['Authorization'] = value
                    # Pass through other relevant headers
                    elif key.lower() in ['openai-organization', 'openai-project', 'user-agent']:
                        headers[key] = value

            url = urljoin(self.config.url, endpoint)

            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    url,
                    json=openai_request,  # Pass through request as-is
                    headers=headers
                )
                response.raise_for_status()
                return response.json(), 200

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
            # Return error in OpenAI format with proper status code
            try:
                error_response = e.response.json()
                return error_response, e.response.status_code
            except:
                return {
                    "error": {
                        "message": f"OpenAI API error: {e.response.status_code}",
                        "type": "api_error",
                        "code": str(e.response.status_code)
                    }
                }, e.response.status_code
        except httpx.TimeoutException as e:
            logger.error(f"OpenAI API timeout: {e}")
            return {
                "error": {
                    "message": f"Request to OpenAI API timed out after {self.config.timeout}s",
                    "type": "timeout_error",
                    "code": "timeout"
                }
            }, 408
        except httpx.ConnectError as e:
            logger.error(f"OpenAI API connection error: {e}")
            return {
                "error": {
                    "message": f"Failed to connect to OpenAI API: {str(e)}",
                    "type": "connection_error",
                    "code": "connection_failed"
                }
            }, 503
        except Exception as e:
            error_msg = str(e) if str(e).strip() else f"Unknown error of type {type(e).__name__}"
            logger.error(f"Error calling OpenAI API: {error_msg}")
            return {
                "error": {
                    "message": f"Failed to call OpenAI API: {error_msg}",
                    "type": "api_error"
                }
            }, 500


class ProviderFactory:
    """Factory for creating model providers from configuration"""
    
    _provider_classes = {
        'ollama': OllamaProvider,
        'openai': OpenAIProvider,
        'google-genai': GoogleGenAIProvider,
        'anthropic': AnthropicProvider,
    }
    
    @classmethod
    def create_provider(cls, config: ProviderConfig) -> IModelProvider:
        """Create a provider instance from configuration"""
        provider_class = cls._provider_classes.get(config.type.lower())
        if not provider_class:
            raise ValueError(f"Unknown provider type: {config.type}")
        
        return provider_class(config)
    
    @classmethod
    def create_providers_from_config(cls, providers_config: List[Dict[str, Any]]) -> List[IModelProvider]:
        """Create multiple providers from configuration list"""
        providers = []
        
        for provider_config in providers_config:
            try:
                # Handle special provider configs
                if provider_config.get('type') == 'google-genai':
                    config = GoogleGenAIConfig(**provider_config)
                elif provider_config.get('type') == 'anthropic':
                    config = AnthropicConfig(**provider_config)
                else:
                    config = ProviderConfig(**provider_config)

                if config.enabled:
                    provider = cls.create_provider(config)
                    providers.append(provider)
                    logger.info(f"Created provider: {config.name} ({config.type}) -> {getattr(config, 'url', 'N/A')}")
                else:
                    logger.info(f"Skipping disabled provider: {config.name}")
            except Exception as e:
                logger.error(f"Failed to create provider from config {provider_config}: {e}")
        
        # Sort providers by priority (lower numbers first)
        providers.sort(key=lambda p: p.config.priority)
        
        return providers
    
    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported provider types"""
        return list(cls._provider_classes.keys())