"""
Model provider implementations for different upstream services.

This module contains concrete implementations of IModelProvider for various
AI model serving platforms like Ollama and OpenAI-compatible APIs.
"""

import logging
from dataclasses import dataclass
import httpx
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

from .config_loading import load_first_config_entry
from .interfaces import IModelProvider, ModelInfo, ProviderConfig, coerce_provider_proxy_settings
from .google_genai_provider import GoogleGenAIProvider, GoogleGenAIConfig
from .anthropic_provider import AnthropicProvider, AnthropicConfig
from .dummy_provider import DummyProvider, DummyConfig

logger = logging.getLogger(__name__)
OPENAI_PASSTHROUGH_HEADERS = frozenset({"openai-organization", "openai-project", "user-agent"})


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
        if not self.config.url.startswith(("http://", "https://")):
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

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _create_model_info(
        self,
        model_id: str,
        model_name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelInfo:
        """Helper to create ModelInfo with provider context"""
        return ModelInfo(
            id=f"{model_id}@{self.get_provider_id()}",
            name=model_name or model_id,
            provider_id=self.get_provider_id(),
            provider_type=self.get_provider_type(),
            endpoint=self.get_endpoint(),
            aliases=aliases or [],
            metadata=metadata or {},
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

                for model_data in data.get("models", []):
                    model_name = model_data.get("name", "unknown")

                    # Extract metadata
                    metadata = {
                        "size": model_data.get("size", 0),
                        "modified_at": model_data.get("modified_at"),
                        "digest": model_data.get("digest"),
                        "details": model_data.get("details", {}),
                    }

                    # Create aliases (original name and any variations)
                    aliases = [model_name]

                    # Handle model name variations (e.g., llama3:8b -> llama3-8b)
                    if ":" in model_name:
                        normalized = model_name.replace(":", "-")
                        aliases.append(normalized)

                    model_info = self._create_model_info(
                        model_id=model_name, model_name=model_name, aliases=aliases, metadata=metadata
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

class OpenAIProvider(BaseModelProvider):
    """Provider for OpenAI-compatible model servers"""

    def __init__(self, config: ProviderConfig):
        config.type = "openai"  # Ensure type is set
        super().__init__(config)

    def _get_health_check_url(self) -> str:
        return self._build_request_url("/v1/models")

    def _build_request_url(self, endpoint: str) -> str:
        normalized_endpoint = (endpoint or "").strip()
        if not normalized_endpoint:
            return self.config.url.rstrip("/")
        if normalized_endpoint.startswith(("http://", "https://")):
            return normalized_endpoint

        parsed_base = urlsplit(self.config.url.rstrip("/"))
        base_segments = [segment for segment in parsed_base.path.split("/") if segment]
        endpoint_segments = [segment for segment in normalized_endpoint.lstrip("/").split("/") if segment]

        if base_segments and endpoint_segments and endpoint_segments[0] == base_segments[-1]:
            endpoint_segments = endpoint_segments[1:]

        path_segments = [*base_segments, *endpoint_segments]
        combined_path = f"/{'/'.join(path_segments)}" if path_segments else "/"

        return urlunsplit(
            (
                parsed_base.scheme,
                parsed_base.netloc,
                combined_path,
                parsed_base.query,
                parsed_base.fragment,
            )
        )

    async def health_check(self) -> bool:
        """OpenAI provider health check - always healthy when configured for client passthrough"""
        if not self.config.api_key:
            # No API key means client passthrough mode - always healthy
            logger.debug(f"OpenAI provider {self.get_provider_id()} in client passthrough mode - marking as healthy")
            return True

        if self.config.static_models:
            try:
                async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                    response = await client.get(self._get_health_check_url(), headers=self._get_headers())
                if response.status_code == 200:
                    return True
                if response.status_code in {404, 405}:
                    logger.info(
                        "OpenAI provider %s uses configured static models and does not expose %s; marking healthy",
                        self.get_provider_id(),
                        self._get_health_check_url(),
                    )
                    return True
                logger.debug(
                    "OpenAI provider %s static-model health check returned %s",
                    self.get_provider_id(),
                    response.status_code,
                )
                return False
            except Exception:
                logger.debug(
                    "OpenAI provider %s static-model health check failed unexpectedly",
                    self.get_provider_id(),
                    exc_info=True,
                )
                return False

        # If we have an API key, do normal health check
        return await super().health_check()

    async def discover_models(self) -> List[ModelInfo]:
        """Discover models from OpenAI /v1/models endpoint or return static list"""
        if self.config.static_models:
            return self._get_configured_static_models()

        # If no API key, return static list for client passthrough
        if not self.config.api_key:
            return self._get_static_openai_models()

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                url = self._build_request_url("/v1/models")
                headers = self._get_headers()

                logger.debug(f"Discovering OpenAI models from {url}")
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                data = response.json()
                models = []

                for model_data in data.get("data", []):
                    model_id = model_data.get("id", "unknown")

                    # Extract metadata
                    metadata = {
                        "object": model_data.get("object"),
                        "created": model_data.get("created"),
                        "owned_by": model_data.get("owned_by"),
                        "permission": model_data.get("permission", []),
                        "root": model_data.get("root"),
                        "parent": model_data.get("parent"),
                    }

                    # Create aliases (original ID and common variations)
                    aliases = [model_id]

                    model_info = self._create_model_info(
                        model_id=model_id, model_name=model_id, aliases=aliases, metadata=metadata
                    )

                    models.append(model_info)
                    logger.debug(f"Discovered OpenAI model: {model_info.id}")

                logger.info(f"Discovered {len(models)} models from OpenAI provider {self.get_provider_id()}")
                return models

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.info("OpenAI API authentication failed, falling back to static model list")
                return self._get_static_openai_models()
            logger.error(f"HTTP error discovering OpenAI models from {self.get_provider_id()}: {e}")
            return self._get_static_openai_models()
        except Exception as e:
            logger.error(f"Error discovering OpenAI models from {self.get_provider_id()}: {e}")
            return self._get_static_openai_models()

    def _get_configured_static_models(self) -> List[ModelInfo]:
        models = []

        for model_name in self.config.static_models or []:
            model_info = self._create_model_info(
                model_id=model_name,
                model_name=model_name,
                aliases=[model_name],
                metadata={"object": "model", "owned_by": self.get_provider_id(), "static": True, "configured": True},
            )
            models.append(model_info)

        logger.info(
            f"Loaded {len(models)} configured static models for OpenAI provider {self.get_provider_id()}"
        )
        return models

    def _get_static_openai_models(self) -> List[ModelInfo]:
        """Return static list of OpenAI models from JSON file"""
        import json
        from pathlib import Path

        try:
            # Get path relative to this file
            current_dir = Path(__file__).parent
            models_file = current_dir / "models" / "openai-models-2025-september.json"

            if not models_file.exists():
                logger.warning(f"OpenAI models file not found at {models_file}, falling back to minimal set")
                return self._get_fallback_models()

            with open(models_file, "r") as f:
                data = json.load(f)

            models = []
            for model_data in data.get("data", []):
                model_id = model_data.get("id", "unknown")

                # Extract metadata
                metadata = {
                    "object": model_data.get("object"),
                    "created": model_data.get("created"),
                    "owned_by": model_data.get("owned_by"),
                    "static": True,  # Mark as static definition
                }

                model_info = self._create_model_info(
                    model_id=model_id, model_name=model_id, aliases=[model_id], metadata=metadata
                )
                models.append(model_info)

            logger.info(
                f"Loaded {len(models)} OpenAI models from static file for client passthrough from {self.get_provider_id()}"
            )
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
                metadata={"object": "model", "owned_by": "openai", "static": True, "fallback": True},
            )
            models.append(model_info)

        logger.info(f"Providing {len(models)} fallback OpenAI models for {self.get_provider_id()}")
        return models

    @staticmethod
    def _normalize_client_header_value(value: Any) -> Any:
        return value.decode("utf-8") if isinstance(value, bytes) else value

    def _merge_client_headers(self, headers: Dict[str, str], client_headers: Optional[Dict[str, str]]) -> Dict[str, str]:
        if not client_headers:
            return headers

        for key, value in client_headers.items():
            normalized_value = self._normalize_client_header_value(value)
            normalized_key = key.lower()

            if normalized_key == "authorization":
                if not self.config.api_key:
                    headers["Authorization"] = normalized_value
                continue

            if normalized_key in OPENAI_PASSTHROUGH_HEADERS:
                headers[key] = normalized_value

        return headers

    async def _post_completion_request(
        self,
        url: str,
        openai_request: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Tuple[Dict[str, Any], int]:
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                url,
                json=openai_request,
                headers=headers,
            )
            response.raise_for_status()
            return response.json(), 200

    @staticmethod
    def _http_status_error_response(error: httpx.HTTPStatusError) -> Tuple[Dict[str, Any], int]:
        logger.error(f"OpenAI API error: {error.response.status_code} - {error.response.text}")
        try:
            return error.response.json(), error.response.status_code
        except Exception:
            return {
                "error": {
                    "message": f"OpenAI API error: {error.response.status_code}",
                    "type": "api_error",
                    "code": str(error.response.status_code),
                }
            }, error.response.status_code

    def _timeout_error_response(self, error: httpx.TimeoutException) -> Tuple[Dict[str, Any], int]:
        logger.error(f"OpenAI API timeout: {error}")
        return {
            "error": {
                "message": f"Request to OpenAI API timed out after {self.config.timeout}s",
                "type": "timeout_error",
                "code": "timeout",
            }
        }, 408

    @staticmethod
    def _connection_error_response(error: httpx.ConnectError) -> Tuple[Dict[str, Any], int]:
        logger.error(f"OpenAI API connection error: {error}")
        return {
            "error": {
                "message": f"Failed to connect to OpenAI API: {str(error)}",
                "type": "connection_error",
                "code": "connection_failed",
            }
        }, 503

    @staticmethod
    def _unexpected_api_error_response(error: Exception) -> Tuple[Dict[str, Any], int]:
        error_msg = str(error) if str(error).strip() else f"Unknown error of type {type(error).__name__}"
        logger.error(f"Error calling OpenAI API: {error_msg}")
        return {"error": {"message": f"Failed to call OpenAI API: {error_msg}", "type": "api_error"}}, 500

    async def generate_completion(
        self,
        openai_request: Dict[str, Any],
        client_headers: Optional[Dict[str, str]] = None,
        endpoint: str = "/v1/chat/completions",
    ) -> Tuple[Dict[str, Any], int]:
        """Generate completion by passing through to OpenAI API"""
        try:
            headers = self._merge_client_headers(self._get_headers(), client_headers)
            return await self._post_completion_request(self._build_request_url(endpoint), openai_request, headers)

        except httpx.HTTPStatusError as e:
            return self._http_status_error_response(e)
        except httpx.TimeoutException as e:
            return self._timeout_error_response(e)
        except httpx.ConnectError as e:
            return self._connection_error_response(e)
        except Exception as e:
            return self._unexpected_api_error_response(e)


@dataclass
class ZaiCodingConfig(ProviderConfig):
    """Configuration for Z.AI GLM Coding Plan provider"""

    url: str = "https://api.z.ai/api/coding/paas/v4"
    api_key_file: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.api_key and self.api_key_file:
            self.api_key = self._load_api_key_from_file(self.api_key_file)
        if not self.api_key:
            raise ValueError("Z.AI Coding provider requires api_key or api_key_file")

    @staticmethod
    def _load_api_key_from_file(api_key_file: str) -> str:
        return load_first_config_entry(
            api_key_file,
            allow_assignments=True,
            strip_inline_comments=True,
            value_label="API key",
        )


class ZaiCodingProvider(OpenAIProvider):
    """Provider for the Z.AI GLM Coding Plan endpoint"""

    CODING_BASE_PATH = "api/coding/paas/v4"
    SUPPORTED_MODELS = ["glm-5.1", "glm-5-turbo", "glm-4.7", "glm-4.5-air"]

    def __init__(self, config: ProviderConfig):
        if not isinstance(config, ZaiCodingConfig):
            config = ZaiCodingConfig(**config.__dict__)

        super().__init__(config)
        self.config.type = "zai-coding"

    def _build_request_url(self, endpoint: str) -> str:
        base_url = self._get_coding_base_url()
        normalized_endpoint = endpoint.lstrip("/")

        if normalized_endpoint.startswith("v1/"):
            normalized_endpoint = normalized_endpoint[3:]

        if not normalized_endpoint:
            return base_url

        return f"{base_url}/{normalized_endpoint}"

    def _get_coding_base_url(self) -> str:
        base_url = self.config.url.rstrip("/")
        suffix = f"/{self.CODING_BASE_PATH}"

        if not base_url.endswith(suffix):
            base_url = f"{base_url}{suffix}"

        return base_url

    def get_provider_type(self) -> str:
        return "zai-coding"

    def get_endpoint(self) -> str:
        return self._get_coding_base_url()

    async def generate_completion(
        self,
        openai_request: Dict[str, Any],
        client_headers: Optional[Dict[str, str]] = None,
        endpoint: str = "/v1/chat/completions",
    ) -> Tuple[Dict[str, Any], int]:
        """Generate a completion using the configured Z.AI coding key."""
        passthrough_headers: Dict[str, str] = {}

        if client_headers:
            for key, value in client_headers.items():
                if isinstance(value, bytes):
                    value = value.decode("utf-8")

                if key.lower() in OPENAI_PASSTHROUGH_HEADERS:
                    passthrough_headers[key] = value

        return await super().generate_completion(openai_request, passthrough_headers, endpoint)

    async def health_check(self) -> bool:
        if not self.config.api_key:
            logger.error(f"Z.AI provider {self.get_provider_id()} has no API key configured")
            return False

        return await super().health_check()

    async def discover_models(self) -> List[ModelInfo]:
        """Return the supported GLM Coding Plan models."""
        return [
            self._create_model_info(
                model_id=model_id,
                model_name=model_id,
                aliases=[model_id],
                metadata={"static": True, "coding_plan": True},
            )
            for model_id in self.SUPPORTED_MODELS
        ]


class ProviderFactory:
    """Factory for creating model providers from configuration"""

    _provider_classes = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "zai-coding": ZaiCodingProvider,
        "google-genai": GoogleGenAIProvider,
        "anthropic": AnthropicProvider,
        "dummy": DummyProvider,
    }
    _config_classes = {
        "google-genai": GoogleGenAIConfig,
        "anthropic": AnthropicConfig,
        "dummy": DummyConfig,
        "zai-coding": ZaiCodingConfig,
    }

    @classmethod
    def create_provider(cls, config: ProviderConfig) -> IModelProvider:
        """Create a provider instance from configuration"""
        provider_class = cls._provider_classes.get(config.type.lower())
        if not provider_class:
            raise ValueError(f"Unknown provider type: {config.type}")

        return provider_class(config)

    @classmethod
    def _convert_proxy_configs(cls, provider_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dictionary proxy configurations to ProxyConfig objects."""
        return coerce_provider_proxy_settings(provider_config)

    @classmethod
    def create_providers_from_config(cls, providers_config: List[Dict[str, Any]]) -> List[IModelProvider]:
        """Create multiple providers from configuration list"""
        providers = []

        for provider_config in providers_config:
            try:
                # Convert proxy configurations from dicts to ProxyConfig objects
                processed_config = cls._convert_proxy_configs(provider_config)

                config_type = str(processed_config.get("type", "")).lower()
                config_class = cls._config_classes.get(config_type, ProviderConfig)
                config = config_class(**processed_config)

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
