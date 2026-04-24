"""
Pytest configuration for OpenAI Model Rerouter tests
"""

import pytest
import os
import asyncio
from unittest.mock import patch

# Use Redis backend for tests - FakeRedis will handle the testing automatically


@pytest.fixture(autouse=True)
def fresh_redis_for_tests():
    """
    Ensure fresh FakeRedis instance for each test.
    Using the new redis_config system - just flush all data to start fresh.
    """
    # Force development environment for tests
    os.environ["APP_ENV"] = "test"
    os.environ.setdefault("REDIS_URL", "fake")

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
    # Disable logging side-effects and enable think stripping for deterministic tests
    # Force legacy proxy mode so unit tests hit mocked upstreams directly
    with (
        patch("smolrouter.app.ENABLE_LOGGING", False),
        patch("smolrouter.app.STRIP_THINKING", True),
        patch.dict(os.environ, {"USE_LEGACY_PROXY": "true"}, clear=False),
    ):
        yield


@pytest.fixture(scope="session", autouse=True)
def init_runtime_components(tmp_path_factory):
    """Initialize Redis (fakeredis), Lua scripts, and blob storage once per test session."""
    os.environ["APP_ENV"] = "test"
    os.environ.setdefault("REDIS_URL", "fake")
    os.environ.setdefault("BLOB_STORAGE_PATH", str(tmp_path_factory.mktemp("smolrouter_blob_storage")))

    async def _init():
        from smolrouter.redis_backend import init_redis_db, RedisApiKeyQuota
        from smolrouter.storage import init_blob_storage

        await init_redis_db()
        # Initialize Lua script used for quota tracking
        try:
            await RedisApiKeyQuota.initialize_lua_script()
        except Exception:
            # Some fakeredis builds may not support scripts; tests that depend on it will handle skips
            pass
        # Initialize blob storage (starts janitor but harmless in tests)
        init_blob_storage()

    try:
        asyncio.run(_init())
    except RuntimeError:
        # If there's already an event loop (e.g., in CI plugins), create a new task
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_init())


# Automatically use isolated database for logging tests
pytest_plugins = []
