import asyncio
from unittest.mock import AsyncMock

import pytest

import smolrouter.container as container_module
from smolrouter.container import SmolRouterConfig, SmolRouterContainer
from smolrouter.facade_keys import RequestIdentity


@pytest.fixture(autouse=True)
def _stub_facade_key_loader(monkeypatch):
    monkeypatch.setattr(
        "smolrouter.container.load_facade_key_registry",
        lambda facade_key_configs: container_module.FacadeKeyRegistry.from_sources(
            facade_key_configs=facade_key_configs,
            facade_key_secrets={},
        ),
    )


def test_start_background_health_monitoring_uses_logged_task(monkeypatch):
    container = SmolRouterContainer(SmolRouterConfig(providers=[]))
    captured = {}

    def fake_create_logged_task(coro, *, task_name, create_task_fn=None, done_callback=None, service=False):
        captured["task_name"] = task_name
        captured["create_task_fn"] = create_task_fn
        captured["service"] = service
        coro.close()
        return object()

    monkeypatch.setattr(container_module, "create_logged_task", fake_create_logged_task)

    container._start_background_health_monitoring()

    assert captured["task_name"] == "provider-background-health-monitor"
    assert captured["create_task_fn"] is asyncio.create_task
    assert captured["service"] is True  # long-lived loop -> cancelled (not awaited) on shutdown


def test_container_create_client_context_preserves_identity():
    container = SmolRouterContainer(SmolRouterConfig(providers=[]))
    identity = RequestIdentity(kind="facade_key", subject_id="project-a")

    context = container.create_client_context(
        ip="127.0.0.1",
        auth_payload={"sub": "user-1"},
        headers={"x-test": "1"},
        identity=identity,
    )

    assert context.user_id == "user-1"
    assert context.identity == identity
    assert context.headers == {"x-test": "1"}


def test_container_builds_facade_key_registry_from_config(monkeypatch):
    captured = []

    monkeypatch.setattr(
        "smolrouter.container.load_facade_key_registry",
        lambda facade_key_configs: captured.append(dict(facade_key_configs or {})) or container_module.FacadeKeyRegistry.from_sources(
            facade_key_configs=facade_key_configs,
            facade_key_secrets={"project-a": ["srk-a"]} if facade_key_configs else {},
        ),
    )

    container = SmolRouterContainer(
        SmolRouterConfig(
            providers=[],
            facade_keys={"project-a": {"display_name": "Project A"}},
        )
    )

    registry = container.get_facade_key_registry()
    assert registry.get_config("project-a").display_name == "Project A"
    assert registry.get_secrets("project-a") == ("srk-a",)
    assert captured == [{"project-a": {"display_name": "Project A"}}]


def test_container_uses_shared_facade_key_loader_for_empty_config(monkeypatch):
    captured = []

    monkeypatch.setattr(
        "smolrouter.container.load_facade_key_registry",
        lambda facade_key_configs: captured.append(dict(facade_key_configs or {})) or container_module.FacadeKeyRegistry.from_sources(),
    )

    container = SmolRouterContainer(SmolRouterConfig(providers=[], facade_keys={}))

    assert isinstance(container.get_facade_key_registry(), container_module.FacadeKeyRegistry)
    assert captured == [{}]


@pytest.mark.asyncio
async def test_container_route_request_forwards_client_context_to_mediator():
    container = SmolRouterContainer(SmolRouterConfig(providers=[]))
    identity = RequestIdentity(kind="facade_key", subject_id="project-a")
    client_context = container.create_client_context(ip="127.0.0.1", identity=identity)

    container._initialized = True
    container._mediator = AsyncMock()
    container._mediator.route_request.return_value = ({"ok": True}, 200, "provider:test", None)

    result = await container.route_request(
        "127.0.0.1",
        "gpt-4",
        {"model": "gpt-4"},
        "/v1/chat/completions",
        {"authorization": "Bearer token"},
        30.0,
        client_context=client_context,
    )

    assert result == ({"ok": True}, 200, "provider:test", None)
    assert container._mediator.route_request.await_args.kwargs["client_context"] == client_context
