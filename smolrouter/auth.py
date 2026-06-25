import os
import logging
import jwt
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import HTTPException, Request, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger("model-rerouter")


class JWTAuth:
    """Simple JWT authentication for SmolRouter"""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        logger.info(f"JWT authentication initialized with {algorithm}")

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload if valid"""
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith("Bearer "):
                token = token[7:]

            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check expiration if 'exp' claim is present
            if "exp" in payload:
                exp_timestamp = payload["exp"]
                if datetime.now(timezone.utc).timestamp() > exp_timestamp:
                    logger.debug("Token expired")
                    return None

            return payload

        except jwt.ExpiredSignatureError:
            logger.debug("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.debug(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None

    def create_token(self, payload: Dict[str, Any], expires_in_hours: int = 24) -> str:
        """Create a JWT token with given payload"""
        # Add expiration
        exp_timestamp = datetime.now(timezone.utc).timestamp() + (expires_in_hours * 3600)
        payload["exp"] = exp_timestamp

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)


# Global auth instance
_jwt_auth: Optional[JWTAuth] = None
_jwt_auth_initialized = False
_jwt_auth_cached_secret: Optional[str] = None
_jwt_auth_cached_state: str = "uninitialized"


def _validate_jwt_secret(secret: str) -> bool:
    """Validate JWT secret meets minimum security requirements"""
    if not secret:
        logger.error("JWT_SECRET is empty or whitespace-only")
        return False

    # Remove whitespace
    secret = secret.strip()

    if not secret:
        logger.error("JWT_SECRET is empty or whitespace-only")
        return False

    if len(secret) < 32:
        logger.error(f"JWT_SECRET is too short ({len(secret)} chars). Minimum 32 characters required for security.")
        return False

    # Check for common weak secrets
    weak_secrets = {
        "your-secret-key",
        "test-secret",
        "password",
        "123456789012345678901234567890123",
        "secret",
        "jwt-secret",
        "my-secret-key",
        "development-secret-key",
    }

    if secret.lower() in weak_secrets:
        logger.error("JWT_SECRET appears to be a weak/default secret. Use a cryptographically secure random key.")
        return False

    # Check for repeated characters (like "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    if len(set(secret)) < 8:
        logger.error("JWT_SECRET has too few unique characters. Use a cryptographically secure random key.")
        return False

    return True


def get_jwt_auth() -> Optional[JWTAuth]:
    """Get JWT auth instance if enabled"""
    global _jwt_auth, _jwt_auth_initialized, _jwt_auth_cached_secret, _jwt_auth_cached_state

    jwt_secret = os.getenv("JWT_SECRET")
    normalized_secret = jwt_secret.strip() if jwt_secret else None

    if _jwt_auth_initialized and normalized_secret == _jwt_auth_cached_secret:
        if _jwt_auth_cached_state == "enabled":
            if _jwt_auth is not None:
                return _jwt_auth
            # If _jwt_auth was manually reset during tests, rebuild the auth object.
        else:
            return None

    _jwt_auth_initialized = True
    _jwt_auth_cached_secret = normalized_secret

    if normalized_secret and _validate_jwt_secret(normalized_secret):
        _jwt_auth = JWTAuth(normalized_secret)
        _jwt_auth_cached_state = "enabled"
        logger.info("JWT authentication enabled with validated secret")
    elif normalized_secret:
        _jwt_auth_cached_state = "disabled_invalid_secret"
        _jwt_auth = None
        logger.error("JWT authentication disabled due to invalid JWT_SECRET")
    else:
        _jwt_auth_cached_state = "disabled_no_secret"
        _jwt_auth = None
        logger.info("JWT authentication disabled (no JWT_SECRET provided)")

    return _jwt_auth


def verify_request_auth(request: Request) -> Optional[Dict[str, Any]]:
    """Verify authentication for incoming request"""
    auth = get_jwt_auth()
    if not auth:
        # No auth configured, allow all requests
        return None

    # Get Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify token
    payload = auth.verify_token(auth_header)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


def create_auth_middleware():
    """Create FastAPI middleware for JWT authentication"""
    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class JWTAuthMiddleware(BaseHTTPMiddleware):
        def __init__(self, app):
            super().__init__(app)
            self.exempt_paths = {
                "/",  # Dashboard
                "/performance",  # Performance dashboard
                "/api/logs",  # API endpoints might need auth, but keeping exempt for now
                "/api/stats",
                "/api/inflight",
                "/api/performance",
            }
            self.inference_auth_paths = {
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/responses",
            }

        async def dispatch(self, request: Request, call_next):
            # Skip auth for exempt paths and static files
            if (
                request.url.path in self.exempt_paths
                or request.url.path.startswith("/static/")
                or request.url.path.startswith("/request/")
            ):
                return await call_next(request)

            if request.url.path in self.inference_auth_paths:
                return await call_next(request)

            allow_error_dashboard_without_auth = os.getenv(
                "ALLOW_UNAUTHENTICATED_ERROR_DASHBOARD", "false"
            ).lower() in ("1", "true", "yes", "on")

            # Only apply auth to API endpoints
            if (
                request.method in {"GET", "HEAD", "OPTIONS"}
                and request.url.path.startswith("/api/errors")
                and allow_error_dashboard_without_auth
            ):
                return await call_next(request)

            if not (request.url.path.startswith("/v1/") or request.url.path.startswith("/api/")):
                return await call_next(request)

            try:
                # Verify authentication
                verify_request_auth(request)
                return await call_next(request)
            except HTTPException as e:
                return JSONResponse(
                    status_code=e.status_code, content={"error": "authentication_failed", "detail": e.detail}
                )

    return JWTAuthMiddleware


# Rate limiting for authentication failures
limiter = Limiter(key_func=get_remote_address)


def setup_rate_limiting(app):
    """Setup rate limiting for the FastAPI app"""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("Rate limiting enabled for authentication failures")
