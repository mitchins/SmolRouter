import asyncio

import smolrouter.container as container_module
from smolrouter.container import SmolRouterConfig, SmolRouterContainer


def test_start_background_health_monitoring_uses_logged_task(monkeypatch):
    container = SmolRouterContainer(SmolRouterConfig(providers=[]))
    captured = {}

    def fake_create_logged_task(coro, *, task_name, create_task_fn=None, done_callback=None):
        captured["task_name"] = task_name
        captured["create_task_fn"] = create_task_fn
        coro.close()
        return object()

    monkeypatch.setattr(container_module, "create_logged_task", fake_create_logged_task)

    container._start_background_health_monitoring()

    assert captured["task_name"] == "provider-background-health-monitor"
    assert captured["create_task_fn"] is asyncio.create_task
