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
from typing import Tuple, Any, Protocol

import redis.asyncio as redis
from redis.exceptions import RedisError

# Optional import for fakeredis - only used if REDIS_URL is not provided
try:
    import fakeredis.aioredis

    FAKEREDIS_AVAILABLE = True
except ImportError:
    FAKEREDIS_AVAILABLE = False

logger = logging.getLogger("smolrouter.redis")

# Environment detection
ENV = os.getenv("APP_ENV", "dev").lower()  # dev|ci|staging|prod
REDIS_URL = os.getenv("REDIS_URL")
ALLOW_FAKE_IN_PROD = os.getenv("ALLOW_FAKE_REDIS_IN_PROD", "0").lower() in ("1", "true", "yes")

# Configuration
REDIS_MAX_CONNS = int(os.getenv("REDIS_MAX_CONNS", "256"))
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "2.0"))
REDIS_CONNECT_TIMEOUT = float(os.getenv("REDIS_CONNECT_TIMEOUT", "1.0"))
REDIS_HEALTH_CHECK_INTERVAL = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))


class RedisLike(Protocol):
    """Protocol defining the interface that both redis.Redis and fakeredis.FakeRedis implement"""

    async def ping(self) -> str: ...
    async def eval(self, script: str, numkeys: int, *keys_and_args) -> Any: ...
    async def hset(self, name: str, key: str = None, value: str = None, mapping: dict = None) -> int: ...
    async def hget(self, name: str, key: str) -> str: ...
    async def expire(self, name: str, time: int) -> bool: ...
    async def delete(self, *names: str) -> int: ...


def create_redis_client() -> Tuple[Any, bool]:
    """
    Create Redis client with environment-aware fallback policy.

    Returns:
        Tuple of (redis_client, is_fake_redis)
    """

    if REDIS_URL and REDIS_URL.lower() != "fake":
        # Real Redis configuration
        try:
            client = redis.from_url(
                REDIS_URL,
                decode_responses=True,
                max_connections=REDIS_MAX_CONNS,
                socket_timeout=REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
                health_check_interval=REDIS_HEALTH_CHECK_INTERVAL,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
            )
            logger.info(f"Redis: using REAL redis at {_redact_url(REDIS_URL)}")
            return client, False

        except Exception as e:
            logger.error(f"Failed to create real Redis client: {e}")
            if ENV == "prod":
                logger.error("Redis: Cannot fallback to FakeRedis in production")
                sys.exit(1)
            # Fall through to fake Redis in dev/ci

    # No REDIS_URL or failed to create real client
    if ENV == "prod" and not ALLOW_FAKE_IN_PROD:
        logger.error(
            "🚨 Redis: REDIS_URL is required in production environment. "
            "Set REDIS_URL or ALLOW_FAKE_REDIS_IN_PROD=1 to override (unsafe)."
        )
        sys.exit(1)

    # Use FakeRedis for dev/ci (or prod override)
    if not FAKEREDIS_AVAILABLE:
        logger.error(
            "🚨 Redis: fakeredis not available and no REDIS_URL provided. "
            "Install fakeredis package for development/testing: pip install fakeredis"
        )
        sys.exit(1)

    client = fakeredis.aioredis.FakeRedis(decode_responses=True)

    if ENV == "prod":
        logger.warning(
            "🚨 Redis: using FAKE redis in PRODUCTION! "
            "This is unsafe and not supported for performance/reliability. "
            "Set REDIS_URL to use real Redis."
        )
    elif ENV in ("staging", "uat"):
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
        if is_fake and ENV not in ("dev", "test"):
            logger.warning(
                f"Redis: FakeRedis active while APP_ENV={ENV.upper()}. "
                "This may not accurately model network failures or performance characteristics."
            )

        logger.info("✅ Redis startup check passed")

    except RedisError as e:
        logger.error(f"❌ Redis startup check failed: {e}")
        if ENV == "prod":
            logger.error("Exiting due to Redis failure in production")
            sys.exit(1)
        raise

    except Exception as e:
        logger.error(f"❌ Unexpected error during Redis startup check: {e}")
        if ENV == "prod":
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
        "redis_url": _redact_url(REDIS_URL) if REDIS_URL else "not_configured",
        "environment": ENV,
        "allow_fake_in_prod": ALLOW_FAKE_IN_PROD,
        "max_connections": REDIS_MAX_CONNS,
        "socket_timeout": REDIS_SOCKET_TIMEOUT,
        "connect_timeout": REDIS_CONNECT_TIMEOUT,
        "health_check_interval": REDIS_HEALTH_CHECK_INTERVAL,
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
