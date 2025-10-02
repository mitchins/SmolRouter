"""
Transport observer for 100% verification of API keys and proxies.

This module provides custom httpx transports that wrap the actual transports
and observe every request/response with 100% certainty about API keys and proxies used.
"""

import logging
from typing import Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)


@dataclass
class RequestObservation:
    """Ground truth observation of a request"""

    request_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # API Key verification (extracted from actual headers)
    api_key_used: Optional[str] = None  # Full key from Authorization header  # pragma: allowlist secret
    api_key_header_name: Optional[str] = None  # Which header was used

    # Proxy verification (extracted from actual connection)
    proxy_host: Optional[str] = None
    proxy_port: Optional[int] = None
    proxy_url: Optional[str] = None

    # Request details
    method: Optional[str] = None
    url: Optional[str] = None
    host: Optional[str] = None

    # Response details
    status_code: Optional[int] = None
    response_received: bool = False
    error: Optional[str] = None


class TransportObserver:
    """
    Observer for httpx transports to capture ground truth about requests.

    Uses httpx event hooks (delegation pattern) to observe actual network traffic.
    """

    def __init__(self):
        self.observations: Dict[str, RequestObservation] = {}
        self._request_counter = 0

    def create_request_hook(self, request_id: str):
        """Create a request hook that captures API key and proxy info"""

        def on_request(request):
            """Called when request is about to be sent - captures ground truth"""
            obs = RequestObservation(request_id=request_id)

            # Extract API key from headers (ground truth)
            for header_name in ["x-goog-api-key", "authorization", "api-key"]:
                if header_name in request.headers:
                    obs.api_key_header_name = header_name
                    header_value = request.headers[header_name]

                    # Extract key from "Bearer <key>" or raw key
                    if header_value.startswith("Bearer "):
                        obs.api_key_used = header_value[7:]
                    else:
                        obs.api_key_used = header_value
                    break

            # Extract request details
            obs.method = request.method
            obs.url = str(request.url)
            obs.host = request.url.host

            # Proxy info is in the request extensions (set by httpx transport)
            if hasattr(request, "extensions"):
                proxy_info = request.extensions.get("proxy", None)
                if proxy_info:
                    obs.proxy_url = str(proxy_info)

            self.observations[request_id] = obs
            logger.debug(
                f"📡 Ground truth captured for {request_id}: key=...{obs.api_key_used[-8:] if obs.api_key_used else 'NONE'}, url={obs.url}"
            )

        return on_request

    def create_response_hook(self, request_id: str):
        """Create a response hook that captures response details"""

        def on_response(response):
            """Called when response is received"""
            if request_id in self.observations:
                obs = self.observations[request_id]
                obs.status_code = response.status_code
                obs.response_received = True

                logger.debug(f"📥 Response received for {request_id}: status={obs.status_code}")

        return on_response

    def get_observation(self, request_id: str) -> Optional[RequestObservation]:
        """Get the ground truth observation for a request"""
        return self.observations.get(request_id)

    def verify_api_key(self, request_id: str, expected_suffix: str) -> bool:
        """
        Verify that the API key used matches expectation.

        Returns True if verified, False if mismatch, None if no observation.
        """
        obs = self.observations.get(request_id)
        if not obs or not obs.api_key_used:
            return None

        actual_suffix = obs.api_key_used[-8:] if len(obs.api_key_used) > 8 else obs.api_key_used
        matches = actual_suffix == expected_suffix

        if not matches:
            logger.error(
                f"❌ API key mismatch for {request_id}: expected=...{expected_suffix}, actual=...{actual_suffix}"
            )
        else:
            logger.debug(f"✅ API key verified for {request_id}: ...{expected_suffix}")

        return matches

    def verify_proxy(self, request_id: str, expected_proxy: Optional[str]) -> bool:
        """
        Verify that the proxy used matches expectation.

        Returns True if verified, False if mismatch, None if no observation.
        """
        obs = self.observations.get(request_id)
        if not obs:
            return None

        actual_proxy = obs.proxy_url

        # Both None = match (no proxy)
        if expected_proxy is None and actual_proxy is None:
            logger.debug(f"✅ Proxy verified for {request_id}: no proxy used (as expected)")
            return True

        # One is None = mismatch
        if (expected_proxy is None) != (actual_proxy is None):
            logger.error(f"❌ Proxy mismatch for {request_id}: expected={expected_proxy}, actual={actual_proxy}")
            return False

        # Both not None - compare
        matches = str(expected_proxy) in str(actual_proxy) or str(actual_proxy) in str(expected_proxy)

        if not matches:
            logger.error(f"❌ Proxy mismatch for {request_id}: expected={expected_proxy}, actual={actual_proxy}")
        else:
            logger.debug(f"✅ Proxy verified for {request_id}: {actual_proxy}")

        return matches

    def cleanup_old_observations(self, max_age_seconds: int = 3600):
        """Clean up observations older than max_age_seconds"""
        now = datetime.now()
        to_remove = [
            req_id
            for req_id, obs in self.observations.items()
            if (now - obs.timestamp).total_seconds() > max_age_seconds
        ]

        for req_id in to_remove:
            del self.observations[req_id]

        if to_remove:
            logger.debug(f"🧹 Cleaned up {len(to_remove)} old observations")


# Global observer instance
_global_observer = TransportObserver()


def get_observer() -> TransportObserver:
    """Get the global transport observer"""
    return _global_observer


class ObservingHTTPTransport(httpx.HTTPTransport):
    """
    Sync HTTP transport wrapper that observes all requests for ground truth tracking.

    Wraps an existing httpx.HTTPTransport and intercepts handle_request to capture:
    - Actual API key used (from headers)
    - Actual proxy used (from connection)
    - Request/response details
    """

    def __init__(self, observation_id: str, wrapped_transport: httpx.HTTPTransport, *args, **kwargs):
        # Don't pass wrapped_transport to super - we'll delegate manually
        super().__init__(*args, **kwargs)
        self.observation_id = observation_id
        self.wrapped_transport = wrapped_transport
        self.observer = get_observer()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Intercept and observe the request before delegation"""
        # Capture ground truth BEFORE sending
        obs = RequestObservation(request_id=self.observation_id)

        # Extract API key from headers (ground truth)
        for header_name in ["x-goog-api-key", "authorization", "api-key"]:
            if header_name.lower() in [h.lower() for h in request.headers.keys()]:
                obs.api_key_header_name = header_name
                header_value = request.headers[header_name]

                # Extract key from "Bearer <key>" or raw key
                if isinstance(header_value, str) and header_value.startswith("Bearer "):
                    obs.api_key_used = header_value[7:]
                else:
                    obs.api_key_used = str(header_value)
                break

        # Extract request details
        obs.method = request.method
        obs.url = str(request.url)
        obs.host = request.url.host

        # Proxy is configured on the transport itself
        if hasattr(self.wrapped_transport, "_pool") and hasattr(self.wrapped_transport._pool, "_proxy_url"):
            obs.proxy_url = str(self.wrapped_transport._pool._proxy_url)
        elif hasattr(self, "_pool") and hasattr(self._pool, "_proxy_url"):
            obs.proxy_url = str(self._pool._proxy_url)

        self.observer.observations[self.observation_id] = obs
        logger.info(
            f"📡 GROUND TRUTH [sync]: key=...{obs.api_key_used[-8:] if obs.api_key_used else 'NONE'}, proxy={obs.proxy_url or 'NONE'}, url={obs.url}"
        )

        # Delegate to wrapped transport
        response = self.wrapped_transport.handle_request(request)

        # Capture response
        obs.status_code = response.status_code
        obs.response_received = True

        logger.info(f"📥 RESPONSE [sync]: status={obs.status_code}")

        return response


class ObservingAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    """
    Async HTTP transport wrapper that observes all requests for ground truth tracking.

    Wraps an existing httpx.AsyncHTTPTransport and intercepts handle_async_request to capture:
    - Actual API key used (from headers)
    - Actual proxy used (from connection)
    - Request/response details
    """

    def __init__(self, observation_id: str, wrapped_transport: httpx.AsyncHTTPTransport, *args, **kwargs):
        # Don't pass wrapped_transport to super - we'll delegate manually
        super().__init__(*args, **kwargs)
        self.observation_id = observation_id
        self.wrapped_transport = wrapped_transport
        self.observer = get_observer()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Intercept and observe the request before delegation"""
        # Capture ground truth BEFORE sending
        obs = RequestObservation(request_id=self.observation_id)

        # Extract API key from headers (ground truth)
        for header_name in ["x-goog-api-key", "authorization", "api-key"]:
            if header_name.lower() in [h.lower() for h in request.headers.keys()]:
                obs.api_key_header_name = header_name
                header_value = request.headers[header_name]

                # Extract key from "Bearer <key>" or raw key
                if isinstance(header_value, str) and header_value.startswith("Bearer "):
                    obs.api_key_used = header_value[7:]
                else:
                    obs.api_key_used = str(header_value)
                break

        # Extract request details
        obs.method = request.method
        obs.url = str(request.url)
        obs.host = request.url.host

        # Proxy is configured on the transport itself
        if hasattr(self.wrapped_transport, "_pool") and hasattr(self.wrapped_transport._pool, "_proxy_url"):
            obs.proxy_url = str(self.wrapped_transport._pool._proxy_url)
        elif hasattr(self, "_pool") and hasattr(self._pool, "_proxy_url"):
            obs.proxy_url = str(self._pool._proxy_url)

        self.observer.observations[self.observation_id] = obs
        logger.info(
            f"📡 GROUND TRUTH [async]: key=...{obs.api_key_used[-8:] if obs.api_key_used else 'NONE'}, proxy={obs.proxy_url or 'NONE'}, url={obs.url}"
        )

        # Delegate to wrapped transport
        response = await self.wrapped_transport.handle_async_request(request)

        # Capture response
        obs.status_code = response.status_code
        obs.response_received = True

        logger.info(f"📥 RESPONSE [async]: status={obs.status_code}")

        return response
