"""
Pytest configuration for OpenAI Model Rerouter tests
"""

import pytest
import os
from unittest.mock import patch

# Use Redis backend for tests - FakeRedis will handle the testing automatically


@pytest.fixture(autouse=True)
def fresh_redis_for_tests():
    """
    Ensure fresh FakeRedis instance for each test.
    Using the new redis_config system - just flush all data to start fresh.
    """
    import asyncio

    # Force development environment for tests
    os.environ["APP_ENV"] = "test"

    # Get the redis client from the new config system
    from smolrouter.redis_config import redis_client, is_fake_redis

    # Ensure we're using FakeRedis for tests
    if not is_fake_redis():
        import warnings

        warnings.warn("Tests should use FakeRedis, but real Redis is configured")

    # Flush all data to start fresh
    try:
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create task
                loop.create_task(redis_client.flushall())
            else:
                # Sync context, run directly
                asyncio.run(redis_client.flushall())
        except RuntimeError:
            # No event loop, that's fine for sync tests
            pass
    except Exception:
        # Reset failed, that's okay for tests
        pass

    yield


@pytest.fixture(autouse=True)
def suppress_jinja2_deprecation_warnings():
    """Globally suppress specific Jinja2 DeprecationWarning about utcnow during tests.

    This avoids noisy test output and prevents CI from treating warnings as failures.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="datetime.datetime.utcnow() is deprecated",
            category=DeprecationWarning,
        )
        yield


@pytest.fixture(scope="function")
def isolated_db():
    """
    Create isolated FakeRedis for each test.
    Since we're using Redis as the hot path, tests use FakeRedis automatically.
    """
    import asyncio
    from smolrouter.redis_config import redis_client

    # Initialize fresh Redis connection (will use FakeRedis)
    async def init_redis():
        await redis_client.flushall()  # Clear any existing data
        return redis_client

    # Run the async initialization
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context
            client = loop.run_until_complete(init_redis())
        else:
            client = asyncio.run(init_redis())
    except RuntimeError:
        # No event loop, create one
        client = asyncio.run(init_redis())

    yield client

    # Cleanup - flush all data
    try:
        asyncio.run(client.flushall())
    except Exception:
        pass  # Cleanup failed, that's okay


@pytest.fixture(scope="function")
def disable_logging():
    """
    Disable logging during regular API tests to avoid database side effects.
    """
    with patch("smolrouter.app.ENABLE_LOGGING", False):
        yield


# Automatically use isolated database for logging tests
pytest_plugins = []
