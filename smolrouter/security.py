import os
import logging
from enum import Enum
from typing import Optional
from fastapi import Request, HTTPException, status

logger = logging.getLogger("model-rerouter")

class SecurityPolicy(Enum):
    """Security policy options for WebUI access"""
    NONE = "NONE"                    # No security, always allow WebUI
    AUTH_WHEN_PROXIED = "AUTH_WHEN_PROXIED"  # Disable WebUI when reverse proxy detected
    ALWAYS_AUTH = "ALWAYS_AUTH"      # Always require JWT for WebUI

class WebUISecurityManager:
    """Simple policy-based WebUI security manager"""
    
    def __init__(self):
        # Parse security policy from environment
        policy_str = os.getenv("WEBUI_SECURITY", "AUTH_WHEN_PROXIED").upper()
        
        try:
            self.policy = SecurityPolicy(policy_str)
        except ValueError:
            logger.error(f"Invalid WEBUI_SECURITY value: {policy_str}. Must be one of: NONE, AUTH_WHEN_PROXIED, ALWAYS_AUTH")
            logger.error("Falling back to AUTH_WHEN_PROXIED for security")
            self.policy = SecurityPolicy.AUTH_WHEN_PROXIED
        
        # Common reverse proxy headers (as set for O(1) lookup)
        self.proxy_headers_set = {
            "x-forwarded-for",
            "x-real-ip", 
            "cf-connecting-ip",
            "x-forwarded-proto",
            "x-forwarded-host",
            "x-original-forwarded-for"
        }
        
        # Check if JWT is configured and valid when required (only for ALWAYS_AUTH now)
        jwt_secret = os.getenv("JWT_SECRET")
        jwt_configured = False
        
        if self.policy == SecurityPolicy.ALWAYS_AUTH:
            if not jwt_secret:
                logger.error("WEBUI_SECURITY is set to ALWAYS_AUTH but JWT_SECRET is not configured!")
                logger.error("WebUI will be inaccessible. Either:")
                logger.error("  1. Set JWT_SECRET environment variable, or")
                logger.error("  2. Set WEBUI_SECURITY=NONE (NOT recommended for production)")
            else:
                # Validate JWT secret strength
                from smolrouter.auth import _validate_jwt_secret
                if not _validate_jwt_secret(jwt_secret):
                    logger.error("WEBUI_SECURITY is set to ALWAYS_AUTH but JWT_SECRET is not configured!")
                    logger.error("WebUI will be inaccessible due to invalid JWT_SECRET.")
                else:
                    jwt_configured = True
        
        # Pre-import auth verification function to avoid circular imports during request processing
        self._verify_request_auth = None
        if self.policy == SecurityPolicy.ALWAYS_AUTH and jwt_configured:
            try:
                from smolrouter.auth import verify_request_auth
                self._verify_request_auth = verify_request_auth
                logger.debug("Pre-loaded JWT verification function")
            except ImportError as e:
                logger.error(f"Failed to import JWT verification: {e}")
        
        logger.info(f"WebUI Security Policy: {self.policy.value}")
        if self.policy == SecurityPolicy.ALWAYS_AUTH:
            if jwt_configured:
                logger.info("JWT authentication is configured and valid")
            else:
                logger.warning("JWT authentication is NOT properly configured")
        elif jwt_secret:
            logger.info("JWT authentication is available")
        else:
            logger.warning("JWT authentication is NOT configured")
    
    def _is_proxied_request(self, request: Request) -> bool:
        """Case-insensitive check if request is coming through a reverse proxy.
        
        Uses O(1) set intersection for performance and handles case sensitivity
        to prevent header case bypass attacks.
        """
        # Convert all request header names to lowercase for case-insensitive comparison
        request_headers_lower = {k.lower() for k in request.headers.keys()}
        
        # Use set intersection for O(1) performance
        return bool(self.proxy_headers_set & request_headers_lower)
    
    def is_webui_accessible(self, request: Request) -> tuple[bool, str]:
        """Determine if WebUI should be accessible for this request.
        
        Returns:
            (is_accessible, reason)
        """
        if self.policy == SecurityPolicy.NONE:
            return True, "security_policy_none"
        
        elif self.policy == SecurityPolicy.AUTH_WHEN_PROXIED:
            is_proxied = self._is_proxied_request(request)
            if is_proxied:
                return False, "webui_disabled_when_proxied"
            else:
                return True, "direct_request_allowed"
        
        elif self.policy == SecurityPolicy.ALWAYS_AUTH:
            # Check if valid JWT provided using pre-loaded function
            if self._verify_request_auth is None:
                return False, "jwt_verification_not_available"
            
            try:
                self._verify_request_auth(request)
                return True, "valid_jwt_provided"
            except Exception:
                return False, "jwt_required"
        
        # This shouldn't happen but fallback to denying access
        return False, "unknown_policy_fallback"
    
    def check_webui_access(self, request: Request):
        """Check web UI access and raise HTTPException if denied"""
        is_accessible, reason = self.is_webui_accessible(request)
        
        if is_accessible:
            logger.debug(f"WebUI access granted: {reason}")
            return
        
        # Access denied
        logger.warning(f"WebUI access denied: {reason}")
        
        if reason == "webui_disabled_when_proxied":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "webui_disabled_when_proxied",
                    "message": "WebUI is disabled when accessed through reverse proxy. Use API endpoints with Bearer token instead.",
                    "hint": "Advanced users: Remove proxy headers to access WebUI directly, or set WEBUI_SECURITY=NONE",
                    "policy": self.policy.value
                }
            )
        elif reason == "jwt_required":
            jwt_secret = os.getenv("JWT_SECRET")
            if not jwt_secret:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "configuration_error",
                        "message": "WEBUI_SECURITY=ALWAYS_AUTH but JWT_SECRET is not configured.",
                        "reason": reason
                    }
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={
                        "error": "authentication_required",
                        "message": "WebUI access requires valid JWT Bearer token in Authorization header.",
                        "policy": self.policy.value
                    },
                    headers={"WWW-Authenticate": "Bearer"}
                )
        else:
            # Generic denial
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "webui_access_denied",
                    "message": "WebUI access denied.",
                    "reason": reason,
                    "policy": self.policy.value
                }
            )

# Global security manager instance
_webui_security: Optional[WebUISecurityManager] = None

def get_webui_security() -> WebUISecurityManager:
    """Get the global WebUI security manager instance"""
    global _webui_security
    if _webui_security is None:
        _webui_security = WebUISecurityManager()
    return _webui_security

def init_webui_security():
    """Initialize WebUI security (called at startup)"""
    security = get_webui_security()
    logger.info("WebUI security manager initialized")
    return security