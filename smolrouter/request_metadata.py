"""
Request metadata for tracking provider-specific information.

This module provides a dataclass for passing metadata about requests
(like API key suffixes, proxy info) from providers through to logging.
"""

from dataclasses import dataclass
from typing import Any, Optional


REQUEST_METADATA_FIELDS = (
    "api_key_suffix",
    "proxy_used",
    "provider_id",
    "model_name",
    "api_key_index",
    "api_key_total",
    "api_key_verified",
    "proxy_verified",
    "observation_id",
)

REQUEST_LOG_METADATA_FIELDS = (
    "api_key_suffix",
    "proxy_used",
    "provider_id",
    "api_key_index",
    "api_key_total",
)

REQUEST_METADATA_INT_FIELDS = (
    "api_key_index",
    "api_key_total",
)


def normalize_request_metadata_value(field_name: str, value: Any) -> Any:
    if field_name == "provider_id" and value == "":
        return None

    if field_name in REQUEST_METADATA_INT_FIELDS:
        if value in (None, ""):
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    return value


def serialize_request_metadata(source: Any, *, fields: tuple[str, ...] = REQUEST_METADATA_FIELDS) -> dict[str, Any]:
    return {
        field_name: normalize_request_metadata_value(field_name, getattr(source, field_name, None))
        for field_name in fields
    }


def apply_request_metadata(target: Any, metadata: Any, *, fields: tuple[str, ...] = REQUEST_LOG_METADATA_FIELDS) -> None:
    if target is None or metadata is None:
        return

    for field_name, value in serialize_request_metadata(metadata, fields=fields).items():
        setattr(target, field_name, value)


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging"""
        return serialize_request_metadata(self)
