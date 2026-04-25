"""
Pytest configuration for OpenAI Model Rerouter tests
"""

import pytest
import os
import asyncio
import threading
from typing import Any, cast
from unittest.mock import Mock, patch

import httpx
import pytest_asyncio
from fastapi.testclient import TestClient

# Use Redis backend for tests - FakeRedis will handle the testing automatically


def _run_async_fixture_step(coro):
    """Run async fixture setup/teardown safely from sync fixtures."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {"value": None, "error": None}

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result["value"] = loop.run_until_complete(coro)
        except Exception as exc:
            result["error"] = exc
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if result["error"]:
        raise result["error"]

    return result["value"]


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

    try:
        _run_async_fixture_step(redis_client.flushall())
    except Exception:
        pass

    yield

    try:
        _run_async_fixture_step(redis_client.flushall())
    except Exception:
        pass


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

    client = cast(Any, _run_async_fixture_step(init_redis()))

    yield client

    # Cleanup - flush all data
    try:
        _run_async_fixture_step(client.flushall())
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


@pytest_asyncio.fixture
async def async_client():
    from smolrouter.app import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as test_client:
        yield test_client


@pytest.fixture
def client():
    from smolrouter.app import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def webui_env(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("WEBUI_"):
            monkeypatch.delenv(key, raising=False)

    monkeypatch.delenv("JWT_SECRET", raising=False)
    return monkeypatch


@pytest.fixture
def mock_request_factory():
    def _factory(headers=None, client_ip="127.0.0.1"):
        request = Mock()
        request.client = Mock()
        request.client.host = client_ip  # NOSONAR S1313
        request.headers = headers or {}
        return request

    return _factory


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

    _run_async_fixture_step(_init())


# Automatically use isolated database for logging tests
pytest_plugins = []
