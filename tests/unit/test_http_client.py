"""Unit tests for smolrouter.http_client.HttpClientFactory: credential masking,
client creation, per-model caching, and lifecycle.
"""

import httpx
import pytest

from smolrouter.http_client import HttpClientFactory
from smolrouter.interfaces import ProxyConfig


# --------------------------------------------------------------------------
# _mask_proxy_url
# --------------------------------------------------------------------------


def test_mask_proxy_url_with_credentials():
    masked = HttpClientFactory._mask_proxy_url("https://user:pass@proxy.example.com:8080")
    assert "user" not in masked
    assert "pass" not in masked
    assert "***:***@proxy.example.com:8080" in masked


def test_mask_proxy_url_without_credentials_unchanged():
    url = "https://proxy.example.com:8080"
    assert HttpClientFactory._mask_proxy_url(url) == url


def test_mask_proxy_url_unparseable_returns_sentinel():
    # urlsplit on a bytes/None would raise -> falls back to "***"
    assert HttpClientFactory._mask_proxy_url(None) == "***"  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# create_client
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_client_without_proxy():
    factory = HttpClientFactory()
    client = factory.create_client(timeout=10.0)
    try:
        assert isinstance(client, httpx.AsyncClient)
        assert client.follow_redirects is True
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_create_client_with_proxy_logs_masked_url(caplog):
    factory = HttpClientFactory()
    proxy = ProxyConfig(https_proxy="https://user:secret@127.0.0.1:8888")
    with caplog.at_level("INFO"):
        client = factory.create_client(proxy_config=proxy)
    try:
        assert any("***:***@127.0.0.1:8888" in r.message for r in caplog.records)
        assert not any("secret" in r.message for r in caplog.records)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_create_client_with_empty_proxy_config():
    factory = HttpClientFactory()
    # ProxyConfig that resolves to no proxy URL
    client = factory.create_client(proxy_config=ProxyConfig())
    try:
        assert isinstance(client, httpx.AsyncClient)
    finally:
        await client.aclose()


# --------------------------------------------------------------------------
# get_client_for_model caching
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_client_for_model_caches_by_key():
    factory = HttpClientFactory()
    try:
        c1 = factory.get_client_for_model("prov", "model-a", timeout=5.0)
        c2 = factory.get_client_for_model("prov", "model-a", timeout=5.0)
        assert c1 is c2  # same key -> cached
    finally:
        await factory.close_all()


@pytest.mark.asyncio
async def test_get_client_for_model_distinct_keys():
    factory = HttpClientFactory()
    try:
        c1 = factory.get_client_for_model("prov", "model-a")
        c2 = factory.get_client_for_model("prov", "model-b")
        assert c1 is not c2
    finally:
        await factory.close_all()


@pytest.mark.asyncio
async def test_get_client_for_model_replaces_closed_client():
    factory = HttpClientFactory()
    try:
        c1 = factory.get_client_for_model("prov", "model-a")
        await c1.aclose()
        c2 = factory.get_client_for_model("prov", "model-a")
        assert c2 is not c1
        assert not c2.is_closed
    finally:
        await factory.close_all()


@pytest.mark.asyncio
async def test_get_client_for_model_includes_proxy_in_key():
    factory = HttpClientFactory()
    proxy = ProxyConfig(https_proxy="https://127.0.0.1:9000")
    try:
        c_no_proxy = factory.get_client_for_model("prov", "m")
        c_proxy = factory.get_client_for_model("prov", "m", proxy_config=proxy)
        assert c_no_proxy is not c_proxy
    finally:
        await factory.close_all()


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_all_closes_and_clears():
    factory = HttpClientFactory()
    client = factory.get_client_for_model("prov", "m")
    await factory.close_all()
    assert client.is_closed
    assert factory._clients == {}


@pytest.mark.asyncio
async def test_clear_cache_empties_without_closing():
    factory = HttpClientFactory()
    factory.get_client_for_model("prov", "m")
    factory.clear_cache()
    assert factory._clients == {}
