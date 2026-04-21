"""
Redis configuration with environment-aware fallback policy.

Policy:
- DEV/CI: Automatically use FakeRedis if REDIS_URL missing (with warning)
- PROD: Require real Redis URL, fail fast if missing
- Override: ALLOW_FAKE_REDIS_IN_PROD=1 for emergencies (with loud warning)
"""

import os
import sys
import logging
from typing import Tuple, Any, Protocol, Optional

import redis.asyncio as redis
from redis.exceptions import RedisError

# Optional import for fakeredis - only used if REDIS_URL is not provided
try:  # pragma: no cover - import success path exercised indirectly in tests
    import fakeredis.aioredis as _fakeredis_module
except ImportError:  # pragma: no cover - guarded by explicit test
    _fakeredis_module = None

logger = logging.getLogger("smolrouter.redis")

def _env() -> str:
    return os.getenv("APP_ENV", "dev").lower()


def _redis_url() -> Optional[str]:
    return os.getenv("REDIS_URL")


def _allow_fake_in_prod() -> bool:
    return os.getenv("ALLOW_FAKE_REDIS_IN_PROD", "0").lower() in ("1", "true", "yes")


def _max_conns() -> int:
    return int(os.getenv("REDIS_MAX_CONNS", "256"))


def _socket_timeout() -> float:
    return float(os.getenv("REDIS_SOCKET_TIMEOUT", "2.0"))


def _connect_timeout() -> float:
    return float(os.getenv("REDIS_CONNECT_TIMEOUT", "1.0"))


def _health_check_interval() -> int:
    return int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))


class RedisLike(Protocol):
    """Protocol defining the interface that both redis.Redis and fakeredis.FakeRedis implement"""

    async def ping(self) -> str: ...
    async def eval(self, _script: str, _numkeys: int, *_keys_and_args) -> Any: ...
    async def hset(self, name: str, key: str = None, value: str = None, _mapping: dict = None) -> int: ...
    async def hget(self, name: str, key: str) -> str: ...
    async def expire(self, name: str, time: int) -> bool: ...
    async def delete(self, *_names: str) -> int: ...


def create_redis_client(
    *,
    redis_url: Optional[str] = None,
    env: Optional[str] = None,
    allow_fake_in_prod: Optional[bool] = None,
    fakeredis_module: Any = _fakeredis_module,
) -> Tuple[Any, bool]:
    """
    Create Redis client with environment-aware fallback policy.

    Returns:
        Tuple of (redis_client, is_fake_redis)
    """

    env = (env or _env()).lower()
    redis_url = redis_url if redis_url is not None else _redis_url()
    allow_fake_in_prod = (
        allow_fake_in_prod
        if allow_fake_in_prod is not None
        else _allow_fake_in_prod()
    )

    if redis_url and redis_url.lower() != "fake":
        # Real Redis configuration
        try:
            client = redis.from_url(
                redis_url,
                decode_responses=True,
                max_connections=_max_conns(),
                socket_timeout=_socket_timeout(),
                socket_connect_timeout=_connect_timeout(),
                health_check_interval=_health_check_interval(),
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
            )
            logger.info(f"Redis: using REAL redis at {_redact_url(redis_url)}")
            return client, False

        except Exception as e:
            logger.error(f"Failed to create real Redis client: {e}")
            if env == "prod":
                logger.error("Redis: Cannot fallback to FakeRedis in production")
                sys.exit(1)
            # Fall through to fake Redis in dev/ci

    # No REDIS_URL or failed to create real client
    if env == "prod" and not allow_fake_in_prod:
        logger.error(
            "🚨 Redis: REDIS_URL is required in production environment. "
            "Set REDIS_URL or ALLOW_FAKE_REDIS_IN_PROD=1 to override (unsafe)."
        )
        sys.exit(1)

    # Use FakeRedis for dev/ci (or prod override)
    if fakeredis_module is None:
        logger.error(
            "🚨 Redis: fakeredis not available and no REDIS_URL provided. "
            "Install fakeredis package for development/testing: pip install fakeredis"
        )
        sys.exit(1)

    client = fakeredis_module.FakeRedis(decode_responses=True)

    if env == "prod":
        logger.warning(
            "🚨 Redis: using FAKE redis in PRODUCTION! "
            "This is unsafe and not supported for performance/reliability. "
            "Set REDIS_URL to use real Redis."
        )
    elif env in ("staging", "uat"):
        logger.warning(
            "⚠️  Redis: using FAKE redis in staging environment. "
            "Consider using real Redis for staging to match production."
        )
    else:
        logger.warning(
            "💛 Redis: using FAKE redis (fakeredis) for development. "
            "Set REDIS_URL to use real Redis for performance testing."
        )

    return client, True


async def redis_startup_check(client: Any, is_fake: bool):
    """
    Perform Redis startup health check to catch misconfigurations early.

    Args:
        client: Redis client instance
        is_fake: Whether using FakeRedis
    """
    try:
        # Basic ping
        await client.ping()
        logger.debug("Redis: ping successful")

        # Verify we can perform hot path operations (EVAL + EXPIRE)
        try:
            lua_script = """
            redis.call('HINCRBY', KEYS[1], 'startup_check', 1)
            redis.call('EXPIRE', KEYS[1], 10)
            return redis.call('HGET', KEYS[1], 'startup_check')
            """

            result = await client.eval(lua_script, 1, "smolrouter:startup_check")
            logger.debug(f"Redis: Lua script execution successful, result: {result}")

            # Clean up test key
            await client.delete("smolrouter:startup_check")

        except Exception as e:
            if is_fake and "unknown command" in str(e).lower():
                # FakeRedis doesn't support all Lua features - use alternative test
                logger.debug("FakeRedis: Lua scripts not supported, testing basic operations")
                await client.hset("smolrouter:startup_check", "test", 1)
                await client.expire("smolrouter:startup_check", 10)
                result = await client.hget("smolrouter:startup_check", "test")
                await client.delete("smolrouter:startup_check")
                logger.debug(f"FakeRedis: Basic operations successful, result: {result}")
            else:
                raise

        # Warn about fake Redis in non-dev environments
        current_env = _env()
        if is_fake and current_env not in ("dev", "test"):
            logger.warning(
                f"Redis: FakeRedis active while APP_ENV={current_env.upper()}. "
                "This may not accurately model network failures or performance characteristics."
            )

        logger.info("✅ Redis startup check passed")

    except RedisError as e:
        logger.error(f"❌ Redis startup check failed: {e}")
        if _env() == "prod":
            logger.error("Exiting due to Redis failure in production")
            sys.exit(1)
        raise

    except Exception as e:
        logger.error(f"❌ Unexpected error during Redis startup check: {e}")
        if _env() == "prod":
            sys.exit(1)
        raise


def get_redis_status() -> dict:
    """
    Get Redis configuration status for health endpoints.

    Returns:
        Dict with Redis status information
    """
    return {
        "redis_backend": "fake" if _is_fake_redis else "real",
        "redis_url": _redact_url(_redis_url()) if _redis_url() else "not_configured",
        "environment": _env(),
        "allow_fake_in_prod": _allow_fake_in_prod(),
        "max_connections": _max_conns(),
        "socket_timeout": _socket_timeout(),
        "connect_timeout": _connect_timeout(),
        "health_check_interval": _health_check_interval(),
    }


def _redact_url(url: str) -> str:
    """Redact sensitive information from Redis URL for logging."""
    if not url:
        return "not_configured"

    # Basic redaction - hide password but show host/port
    if "://" in url:
        scheme, rest = url.split("://", 1)
        if "@" in rest:
            # redis://user:password@host:port -> redis://user:***@host:port  # pragma: allowlist secret
            auth, host_port = rest.split("@", 1)
            if ":" in auth:
                user, _ = auth.split(":", 1)
                return f"{scheme}://{user}:***@{host_port}"
            else:
                return f"{scheme}://***@{host_port}"
        else:
            return url  # No auth, safe to show

    return "***"


# Create global Redis client
redis_client, _is_fake_redis = create_redis_client()


# Export for backwards compatibility with existing code
def get_redis():
    """Get the configured Redis client."""
    return redis_client


def is_fake_redis() -> bool:
    """Check if using FakeRedis."""
    return _is_fake_redis
