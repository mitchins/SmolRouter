"""
HTTP client factory for managing proxy configurations and HTTP clients.

This module provides centralized HTTP client creation with support for per-model
proxy configurations, ensuring consistent behavior across all providers.
"""

import logging
import httpx
from typing import Optional, Dict
from .interfaces import ProxyConfig

logger = logging.getLogger(__name__)


class HttpClientFactory:
    """Factory for creating HTTP clients with proxy support"""

    def __init__(self):
        self._clients: Dict[str, httpx.AsyncClient] = {}

    def create_client(
        self, timeout: float = 30.0, proxy_config: Optional[ProxyConfig] = None, **kwargs
    ) -> httpx.AsyncClient:
        """
        Create an HTTP client with optional proxy configuration.

        Args:
            timeout: Request timeout in seconds
            proxy_config: Proxy configuration for this client
            **kwargs: Additional httpx.AsyncClient arguments

        Returns:
            Configured httpx.AsyncClient instance
        """
        client_kwargs = {"timeout": httpx.Timeout(timeout), "follow_redirects": True, **kwargs}

        # Add proxy configuration if provided
        if proxy_config:
            proxy_url = proxy_config.to_httpx_proxy()
            if proxy_url:
                # httpx uses 'proxy' parameter and takes a single proxy URL
                client_kwargs["proxy"] = proxy_url
                logger.info(f"Creating HTTP client with proxy: {proxy_url}")

        return httpx.AsyncClient(**client_kwargs)

    def get_client_for_model(
        self,
        provider_name: str,
        model_name: str,
        timeout: float = 30.0,
        proxy_config: Optional[ProxyConfig] = None,
        **kwargs,
    ) -> httpx.AsyncClient:
        """
        Get or create an HTTP client for a specific provider and model.

        Args:
            provider_name: Name of the provider
            model_name: Name of the model
            timeout: Request timeout in seconds
            proxy_config: Proxy configuration for this model
            **kwargs: Additional httpx.AsyncClient arguments

        Returns:
            Cached or new httpx.AsyncClient instance
        """
        # Create a cache key based on provider, model, and proxy config
        proxy_key = ""
        if proxy_config:
            proxy_url = proxy_config.to_httpx_proxy()
            proxy_key = proxy_url or ""

        cache_key = f"{provider_name}:{model_name}:{proxy_key}:{timeout}"

        # Return cached client if available
        if cache_key in self._clients:
            client = self._clients[cache_key]
            if not client.is_closed:
                return client
            else:
                # Remove closed client from cache
                del self._clients[cache_key]

        # Create new client
        client = self.create_client(timeout=timeout, proxy_config=proxy_config, **kwargs)

        # Cache the client
        self._clients[cache_key] = client

        logger.debug(f"Created new HTTP client for {provider_name}:{model_name}")
        return client

    async def close_all(self):
        """Close all cached HTTP clients"""
        for client in self._clients.values():
            if not client.is_closed:
                await client.aclose()
        self._clients.clear()
        logger.debug("Closed all HTTP clients")

    def clear_cache(self):
        """Clear the client cache (for testing)"""
        self._clients.clear()


# Global instance for use across providers
http_client_factory = HttpClientFactory()
