"""
Request metadata for tracking provider-specific information.

This module provides a dataclass for passing metadata about requests
(like API key suffixes, proxy info) from providers through to logging.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RequestMetadata:
    """Metadata about a request for logging purposes

    Prefers ground truth (observed) over intent when available.
    """

    api_key_suffix: Optional[str] = None  # pragma: allowlist secret
    proxy_used: Optional[str] = None
    provider_id: Optional[str] = None
    model_name: Optional[str] = None

    # API key position tracking (for multi-key providers like Google GenAI)
    api_key_index: Optional[int] = None  # 1-based position of key in pool
    api_key_total: Optional[int] = None  # Total number of keys in pool

    # Ground truth verification flags
    api_key_verified: bool = False  # True if from httpx observation
    proxy_verified: bool = False  # True if from httpx observation
    observation_id: Optional[str] = None  # For debugging/tracing

    # Load balancer instance tracking (for decrementing active_requests on completion)
    lb_instance: Optional[object] = None  # ModelInstance from load_balancer

    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            "api_key_suffix": self.api_key_suffix,
            "proxy_used": self.proxy_used,
            "provider_id": self.provider_id,
            "model_name": self.model_name,
            "api_key_index": self.api_key_index,
            "api_key_total": self.api_key_total,
            "api_key_verified": self.api_key_verified,
            "proxy_verified": self.proxy_verified,
            "observation_id": self.observation_id,
        }
