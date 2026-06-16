import pytest

from smolrouter import redis_config


@pytest.mark.asyncio
async def test_create_client_uses_fakeredis_when_no_url(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    client, is_fake = redis_config.create_redis_client(env="test", redis_url=None)

    assert is_fake is True
    await client.flushall()
    await client.set("redis_config:test", "value")
    assert await client.get("redis_config:test") == "value"
    await client.flushall()


@pytest.mark.asyncio
async def test_create_client_falls_back_to_fake_on_real_failure(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

    def _boom(*args, **kwargs):  # pragma: no cover - executed in test
        raise ConnectionError("simulated connection failure")

    monkeypatch.setattr(redis_config.redis.BlockingConnectionPool, "from_url", _boom)

    client, is_fake = redis_config.create_redis_client(env="dev", redis_url="redis://localhost:6379")

    assert is_fake is True
    assert client.__class__.__name__ == "FakeRedis"
    await client.set("fallback:test", "ok")
    assert await client.get("fallback:test") == "ok"
    await client.flushall()


def test_create_client_exits_when_fakeredis_missing(monkeypatch):
    with pytest.raises(SystemExit):
        redis_config.create_redis_client(env="dev", fakeredis_module=None)


def test_get_redis_status_tracks_environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "ci")
    monkeypatch.setenv("REDIS_MAX_CONNS", "99")
    monkeypatch.setenv("REDIS_SOCKET_TIMEOUT", "3.5")
    monkeypatch.setenv("REDIS_CONNECT_TIMEOUT", "1.5")
    monkeypatch.setenv("REDIS_HEALTH_CHECK_INTERVAL", "12")

    status = redis_config.get_redis_status()

    assert status["environment"] == "ci"
    assert status["max_connections"] == 99
    assert status["socket_timeout"] == 3.5
    assert status["connect_timeout"] == 1.5
    assert status["health_check_interval"] == 12


def test_redact_url_hides_credentials():
    secret_url = "redis://user:password@localhost:6379/0"
    redacted = redis_config._redact_url(secret_url)
    assert redacted == "redis://user:***@localhost:6379/0"

    no_auth_url = "redis://localhost:6379/0"
    assert redis_config._redact_url(no_auth_url) == no_auth_url

    assert redis_config._redact_url("") == "not_configured"
