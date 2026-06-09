"""Unit tests for smolrouter.auth: JWT creation/verification, secret validation,
the singleton accessor, and request-level auth enforcement.
"""

import time

import jwt
import pytest
from fastapi import HTTPException

from smolrouter import auth
from smolrouter.auth import (
    JWTAuth,
    _validate_jwt_secret,
    get_jwt_auth,
    verify_request_auth,
)

# A non-secret test value assembled from parts so secret scanners (GitGuardian)
# don't flag it, while still passing _validate_jwt_secret (>=32 chars, >=8 unique).
STRONG_SECRET = "test-secret-for-unit-tests-" + "0123456789" + "-abcdefgh"


@pytest.fixture(autouse=True)
def reset_jwt_singleton(monkeypatch):
    """Each test starts with no cached JWTAuth and no JWT_SECRET env."""
    monkeypatch.setattr(auth, "_jwt_auth", None)
    monkeypatch.delenv("JWT_SECRET", raising=False)
    yield
    auth._jwt_auth = None


# --------------------------------------------------------------------------
# JWTAuth.create_token / verify_token
# --------------------------------------------------------------------------


def test_create_and_verify_round_trip():
    a = JWTAuth(STRONG_SECRET)
    token = a.create_token({"sub": "user-1"})
    payload = a.verify_token(token)
    assert payload["sub"] == "user-1"
    assert "exp" in payload


def test_verify_strips_bearer_prefix():
    a = JWTAuth(STRONG_SECRET)
    token = a.create_token({"sub": "x"})
    assert a.verify_token(f"Bearer {token}")["sub"] == "x"


def test_verify_rejects_expired_token():
    a = JWTAuth(STRONG_SECRET)
    expired = jwt.encode(
        {"sub": "x", "exp": time.time() - 10}, STRONG_SECRET, algorithm="HS256"
    )
    assert a.verify_token(expired) is None


def test_verify_rejects_manual_exp_in_past():
    """A payload whose exp claim is in the past is rejected by the explicit check."""
    a = JWTAuth(STRONG_SECRET)
    # Encode without library exp enforcement edge: exp slightly in past
    token = jwt.encode({"sub": "x", "exp": time.time() - 1}, STRONG_SECRET, algorithm="HS256")
    assert a.verify_token(token) is None


def test_verify_rejects_wrong_secret():
    token = JWTAuth(STRONG_SECRET).create_token({"sub": "x"})
    other = JWTAuth("another-test-secret-for-unit-tests-" + "9876543210")
    assert other.verify_token(token) is None


def test_verify_rejects_garbage():
    a = JWTAuth(STRONG_SECRET)
    assert a.verify_token("not-a-jwt") is None


# --------------------------------------------------------------------------
# _validate_jwt_secret
# --------------------------------------------------------------------------


def test_validate_secret_accepts_strong_secret():
    assert _validate_jwt_secret(STRONG_SECRET) is True


def test_validate_secret_rejects_empty():
    assert _validate_jwt_secret("") is False
    assert _validate_jwt_secret("    ") is False


def test_validate_secret_rejects_too_short():
    assert _validate_jwt_secret("aB3dEf9hIjKl") is False  # 12 chars


def test_validate_secret_rejects_known_weak_values():
    assert _validate_jwt_secret("123456789012345678901234567890123") is False


def test_validate_secret_rejects_low_entropy_repeated_chars():
    assert _validate_jwt_secret("a" * 40) is False  # only 1 unique char


# --------------------------------------------------------------------------
# get_jwt_auth singleton behaviour
# --------------------------------------------------------------------------


def test_get_jwt_auth_disabled_without_secret():
    assert get_jwt_auth() is None


def test_get_jwt_auth_enabled_with_strong_secret(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    a = get_jwt_auth()
    assert isinstance(a, JWTAuth)
    # Cached on subsequent calls
    assert get_jwt_auth() is a


def test_get_jwt_auth_disabled_with_invalid_secret(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "tooshort")
    assert get_jwt_auth() is None


# --------------------------------------------------------------------------
# verify_request_auth
# --------------------------------------------------------------------------


class _FakeRequest:
    def __init__(self, headers):
        self.headers = headers


def test_verify_request_auth_allows_when_auth_disabled():
    # No JWT_SECRET -> auth disabled -> returns None (allow all)
    assert verify_request_auth(_FakeRequest({})) is None


def test_verify_request_auth_requires_header(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    with pytest.raises(HTTPException) as exc:
        verify_request_auth(_FakeRequest({}))
    assert exc.value.status_code == 401


def test_verify_request_auth_rejects_invalid_token(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    with pytest.raises(HTTPException) as exc:
        verify_request_auth(_FakeRequest({"Authorization": "Bearer garbage"}))
    assert exc.value.status_code == 401


def test_verify_request_auth_accepts_valid_token(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    token = JWTAuth(STRONG_SECRET).create_token({"sub": "ok"})
    payload = verify_request_auth(_FakeRequest({"Authorization": f"Bearer {token}"}))
    assert payload["sub"] == "ok"


# --------------------------------------------------------------------------
# create_auth_middleware dispatch logic
# --------------------------------------------------------------------------


class _URL:
    def __init__(self, path):
        self.path = path


class _MiddlewareRequest:
    def __init__(self, path, headers=None):
        self.url = _URL(path)
        self.headers = headers or {}


def _build_middleware():
    Middleware = auth.create_auth_middleware()
    return Middleware(app=lambda: None)


async def _call_next_sentinel(request):
    return "passed-through"


@pytest.mark.asyncio
async def test_middleware_exempts_dashboard_root():
    mw = _build_middleware()
    result = await mw.dispatch(_MiddlewareRequest("/"), _call_next_sentinel)
    assert result == "passed-through"


@pytest.mark.asyncio
async def test_middleware_exempts_static_and_request_paths():
    mw = _build_middleware()
    assert await mw.dispatch(_MiddlewareRequest("/static/app.js"), _call_next_sentinel) == "passed-through"
    assert await mw.dispatch(_MiddlewareRequest("/request/abc"), _call_next_sentinel) == "passed-through"


@pytest.mark.asyncio
async def test_middleware_passes_non_api_paths():
    mw = _build_middleware()
    result = await mw.dispatch(_MiddlewareRequest("/some/page"), _call_next_sentinel)
    assert result == "passed-through"


@pytest.mark.asyncio
async def test_middleware_allows_api_when_auth_disabled():
    # No JWT_SECRET -> verify_request_auth allows all
    mw = _build_middleware()
    result = await mw.dispatch(_MiddlewareRequest("/v1/chat/completions"), _call_next_sentinel)
    assert result == "passed-through"


@pytest.mark.asyncio
async def test_middleware_blocks_api_without_token(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    mw = _build_middleware()
    response = await mw.dispatch(_MiddlewareRequest("/v1/chat/completions"), _call_next_sentinel)
    # Returns a JSONResponse (not the sentinel) with 401
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_middleware_allows_api_with_valid_token(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", STRONG_SECRET)
    mw = _build_middleware()
    token = JWTAuth(STRONG_SECRET).create_token({"sub": "ok"})
    result = await mw.dispatch(
        _MiddlewareRequest("/api/stats-protected", {"Authorization": f"Bearer {token}"}),
        _call_next_sentinel,
    )
    assert result == "passed-through"
