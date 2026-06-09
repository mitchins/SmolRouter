"""Unit tests for smolrouter.routing.SmartRouter.

Covers alias/route resolution, instance construction, regex matching, and the
async failover behaviour of route_request/try_upstream.
"""

import httpx
import pytest
import respx

from smolrouter import routing
from smolrouter.routing import (
    ModelAlias,
    SmartRouter,
    UpstreamInstance,
    get_smart_router,
    reload_router_config,
)


SERVERS = {
    "alpha": "http://alpha:8000",
    "beta": "http://beta:8000",
}


def make_router(**config_overrides):
    config = {"servers": SERVERS}
    config.update(config_overrides)
    return SmartRouter(config, default_upstream="http://default:9000")


# --------------------------------------------------------------------------
# dataclass __str__ helpers
# --------------------------------------------------------------------------


def test_upstream_instance_str():
    inst = UpstreamInstance("alpha", "http://alpha:8000", "llama")
    assert str(inst) == "alpha(http://alpha:8000)"


def test_model_alias_str():
    alias = ModelAlias(
        "big",
        [UpstreamInstance("alpha", "http://alpha:8000"), UpstreamInstance("beta", "http://beta:8000")],
    )
    assert str(alias) == "big -> [alpha(http://alpha:8000), beta(http://beta:8000)]"


# --------------------------------------------------------------------------
# _build_upstream_instance
# --------------------------------------------------------------------------


def test_build_instance_from_string_with_model_override():
    router = make_router()
    inst = router._build_upstream_instance("alpha/llama-3", "myalias", SERVERS)
    assert inst == UpstreamInstance("alpha", "http://alpha:8000", "llama-3")


def test_build_instance_from_string_without_model():
    router = make_router()
    inst = router._build_upstream_instance("beta", "myalias", SERVERS)
    assert inst == UpstreamInstance("beta", "http://beta:8000", None)


def test_build_instance_from_string_unknown_server_returns_none():
    router = make_router()
    assert router._build_upstream_instance("ghost", "myalias", SERVERS) is None


def test_build_instance_from_dict_with_server_name():
    router = make_router()
    inst = router._build_upstream_instance(
        {"server": "alpha", "model": "qwen"}, "myalias", SERVERS
    )
    assert inst == UpstreamInstance("alpha", "http://alpha:8000", "qwen")


def test_build_instance_from_dict_with_explicit_url():
    router = make_router()
    inst = router._build_upstream_instance(
        {"url": "http://custom:1111", "model": "m"}, "myalias", SERVERS
    )
    assert inst == UpstreamInstance("http://custom:1111", "http://custom:1111", "m")


def test_build_instance_from_dict_without_url_or_server_returns_none():
    router = make_router()
    assert router._build_upstream_instance({"model": "m"}, "myalias", SERVERS) is None


def test_build_instance_invalid_type_returns_none():
    router = make_router()
    assert router._build_upstream_instance(42, "myalias", SERVERS) is None


# --------------------------------------------------------------------------
# _load_config / aliases
# --------------------------------------------------------------------------


def test_load_config_builds_aliases_and_routes():
    router = make_router(
        aliases={
            "big": {"instances": ["alpha/llama", "beta"]},
        },
        routes=[{"match": {"model": "x"}, "route": {"upstream": "http://x"}}],
    )
    assert "big" in router.aliases
    assert len(router.aliases["big"].instances) == 2
    assert len(router.routes) == 1


def test_load_config_skips_alias_with_no_valid_instances():
    router = make_router(aliases={"broken": {"instances": ["ghost", "phantom"]}})
    assert "broken" not in router.aliases


# --------------------------------------------------------------------------
# _route_matches
# --------------------------------------------------------------------------


def test_route_matches_empty_criteria_matches_anything():
    router = make_router()
    assert router._route_matches({}, "host", "any-model") is True


def test_route_matches_source_host_mismatch():
    router = make_router()
    assert router._route_matches({"source_host": "a"}, "b", "m") is False


def test_route_matches_exact_model():
    router = make_router()
    assert router._route_matches({"model": "gpt-4"}, "h", "gpt-4") is True
    assert router._route_matches({"model": "gpt-4"}, "h", "gpt-3") is False


def test_route_matches_regex_pattern():
    router = make_router()
    assert router._route_matches({"model": "/gpt-.*/"}, "h", "gpt-4o") is True
    assert router._route_matches({"model": "/^claude/"}, "h", "gpt-4o") is False


# --------------------------------------------------------------------------
# find_route
# --------------------------------------------------------------------------


def test_find_route_alias_wins():
    router = make_router(aliases={"big": {"instances": ["alpha", "beta"]}})
    instances, override = router.find_route("host", "big")
    assert [i.name for i in instances] == ["alpha", "beta"]
    assert override is None


def test_find_route_matches_explicit_route_with_model_override():
    router = make_router(
        routes=[
            {
                "match": {"model": "gpt-4"},
                "route": {"upstream": "http://router-target", "model": "local-model"},
            }
        ]
    )
    instances, override = router.find_route("host", "gpt-4")
    assert len(instances) == 1
    assert instances[0].url == "http://router-target"
    assert instances[0].model == "local-model"
    assert override == "local-model"


def test_find_route_falls_back_to_default():
    router = make_router()
    instances, override = router.find_route("host", "unknown")
    assert instances[0].url == "http://default:9000"
    assert instances[0].name == "default"
    assert override is None


def test_find_route_skips_route_without_upstream():
    router = make_router(routes=[{"match": {"model": "gpt-4"}, "route": {}}])
    instances, _ = router.find_route("host", "gpt-4")
    # No upstream in route -> falls through to default
    assert instances[0].name == "default"


# --------------------------------------------------------------------------
# try_upstream (async, httpx mocked with respx)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_try_upstream_success():
    router = make_router()
    respx.post("http://alpha:8000/v1/chat").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    inst = UpstreamInstance("alpha", "http://alpha:8000")
    success, data, status = await router.try_upstream(inst, {"model": "m"}, "/v1/chat", {}, 5.0)
    assert success is True
    assert data == {"ok": True}
    assert status == 200


@pytest.mark.asyncio
@respx.mock
async def test_try_upstream_applies_model_override():
    router = make_router()
    route = respx.post("http://alpha:8000/v1/chat").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    inst = UpstreamInstance("alpha", "http://alpha:8000", "override-model")
    payload = {"model": "original"}
    await router.try_upstream(inst, payload, "/v1/chat", {}, 5.0)
    sent = route.calls.last.request
    import json

    assert json.loads(sent.content)["model"] == "override-model"
    # Original payload must not be mutated
    assert payload["model"] == "original"


@pytest.mark.asyncio
@respx.mock
async def test_try_upstream_http_error_with_json_body():
    router = make_router()
    respx.post("http://alpha:8000/v1/chat").mock(
        return_value=httpx.Response(429, json={"error": "rate limited"})
    )
    inst = UpstreamInstance("alpha", "http://alpha:8000")
    success, data, status = await router.try_upstream(inst, {}, "/v1/chat", {}, 5.0)
    assert success is False
    assert data == {"error": "rate limited"}
    assert status == 429


@pytest.mark.asyncio
@respx.mock
async def test_try_upstream_http_error_non_json_body():
    router = make_router()
    respx.post("http://alpha:8000/v1/chat").mock(
        return_value=httpx.Response(500, text="boom")
    )
    inst = UpstreamInstance("alpha", "http://alpha:8000")
    success, data, status = await router.try_upstream(inst, {}, "/v1/chat", {}, 5.0)
    assert success is False
    assert data == "HTTP 500"
    assert status == 500


@pytest.mark.asyncio
@respx.mock
async def test_try_upstream_success_with_invalid_json():
    router = make_router()
    respx.post("http://alpha:8000/v1/chat").mock(
        return_value=httpx.Response(200, text="not json")
    )
    inst = UpstreamInstance("alpha", "http://alpha:8000")
    success, data, status = await router.try_upstream(inst, {}, "/v1/chat", {}, 5.0)
    assert success is False
    assert "JSON parse error" in data
    assert status == 200


@pytest.mark.asyncio
@respx.mock
async def test_try_upstream_connect_error():
    router = make_router()
    respx.post("http://alpha:8000/v1/chat").mock(
        side_effect=httpx.ConnectError("refused")
    )
    inst = UpstreamInstance("alpha", "http://alpha:8000")
    success, data, status = await router.try_upstream(inst, {}, "/v1/chat", {}, 5.0)
    assert success is False
    assert "Connection error" in data
    assert status == 502


@pytest.mark.asyncio
@respx.mock
async def test_try_upstream_unexpected_error():
    router = make_router()
    respx.post("http://alpha:8000/v1/chat").mock(
        side_effect=ValueError("weird")
    )
    inst = UpstreamInstance("alpha", "http://alpha:8000")
    success, data, status = await router.try_upstream(inst, {}, "/v1/chat", {}, 5.0)
    assert success is False
    assert "Unexpected error" in data
    assert status == 500


# --------------------------------------------------------------------------
# route_request (failover orchestration)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_route_request_first_upstream_succeeds():
    router = make_router(aliases={"big": {"instances": ["alpha", "beta"]}})
    respx.post("http://alpha:8000/v1/chat").mock(
        return_value=httpx.Response(200, json={"from": "alpha"})
    )
    data, status, used = await router.route_request("h", "big", {}, "/v1/chat", {}, 5.0)
    assert data == {"from": "alpha"}
    assert status == 200
    assert "alpha" in used


@pytest.mark.asyncio
@respx.mock
async def test_route_request_fails_over_to_second():
    router = make_router(aliases={"big": {"instances": ["alpha", "beta"]}})
    respx.post("http://alpha:8000/v1/chat").mock(
        return_value=httpx.Response(503, json={"error": "down"})
    )
    respx.post("http://beta:8000/v1/chat").mock(
        return_value=httpx.Response(200, json={"from": "beta"})
    )
    data, status, used = await router.route_request("h", "big", {}, "/v1/chat", {}, 5.0)
    assert data == {"from": "beta"}
    assert status == 200
    assert "beta" in used


@pytest.mark.asyncio
@respx.mock
async def test_route_request_all_fail_returns_last_status():
    router = make_router(aliases={"big": {"instances": ["alpha", "beta"]}})
    respx.post("http://alpha:8000/v1/chat").mock(
        return_value=httpx.Response(500, json={"error": "a"})
    )
    respx.post("http://beta:8000/v1/chat").mock(
        return_value=httpx.Response(503, json={"error": "b"})
    )
    data, status, used = await router.route_request("h", "big", {}, "/v1/chat", {}, 5.0)
    assert data["error"] == "all_upstreams_failed"
    assert status == 503
    assert used == "none"
    assert len(data["details"]) == 2


@pytest.mark.asyncio
async def test_route_request_timeout(monkeypatch):
    router = make_router(aliases={"big": {"instances": ["alpha"]}})

    async def slow_try_upstream(*args, **kwargs):
        import asyncio

        await asyncio.sleep(1.0)
        return True, {}, 200

    monkeypatch.setattr(router, "try_upstream", slow_try_upstream)
    data, status, used = await router.route_request("h", "big", {}, "/v1/chat", {}, 0.05)
    assert status == 504
    assert data["error"] == "request_timeout"
    assert used == "none"


# --------------------------------------------------------------------------
# global singleton helpers
# --------------------------------------------------------------------------


def test_get_smart_router_is_singleton(monkeypatch):
    monkeypatch.setattr(routing, "_smart_router", None)
    r1 = get_smart_router({"servers": SERVERS}, "http://default")
    r2 = get_smart_router({"servers": {}}, "http://other")
    assert r1 is r2  # second call returns cached instance


def test_reload_router_config_replaces_instance(monkeypatch):
    monkeypatch.setattr(routing, "_smart_router", None)
    r1 = get_smart_router({"servers": SERVERS}, "http://default")
    reload_router_config({"servers": SERVERS, "aliases": {"big": {"instances": ["alpha"]}}}, "http://default")
    r2 = routing._smart_router
    assert r2 is not r1
    assert "big" in r2.aliases
