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

    # Ground truth verification flags
    api_key_verified: bool = False  # True if from httpx observation
    proxy_verified: bool = False  # True if from httpx observation
    observation_id: Optional[str] = None  # For debugging/tracing

    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            "api_key_suffix": self.api_key_suffix,
            "proxy_used": self.proxy_used,
            "provider_id": self.provider_id,
            "model_name": self.model_name,
            "api_key_verified": self.api_key_verified,
            "proxy_verified": self.proxy_verified,
            "observation_id": self.observation_id,
        }
