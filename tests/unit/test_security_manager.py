"""Unit tests for smolrouter.security.WebUISecurityManager covering all three
policies, the ALWAYS_AUTH JWT paths, and check_webui_access HTTPException
behaviour. Complements the existing test_webui_security tests.
"""

import pytest
from fastapi import HTTPException

from smolrouter import security
from smolrouter.security import (
    SecurityPolicy,
    WebUISecurityManager,
    get_webui_security,
)

# Non-secret test value (assembled from parts so secret scanners don't flag it)
# that still satisfies _validate_jwt_secret.
STRONG_SECRET = "test-secret-for-unit-tests-" + "0123456789" + "-abcdefgh"


# --------------------------------------------------------------------------
# Policy parsing
# --------------------------------------------------------------------------


def test_invalid_policy_falls_back_to_auth_when_proxied(webui_env):
    webui_env.setenv("WEBUI_SECURITY", "NONSENSE")
    manager = WebUISecurityManager()
    assert manager.policy == SecurityPolicy.AUTH_WHEN_PROXIED


# --------------------------------------------------------------------------
# NONE policy
# --------------------------------------------------------------------------


def test_none_policy_always_accessible(webui_env, mock_request_factory):
    webui_env.setenv("WEBUI_SECURITY", "NONE")
    manager = WebUISecurityManager()
    accessible, reason = manager.is_webui_accessible(mock_request_factory({"x-forwarded-for": "1.2.3.4"}))
    assert accessible is True
    assert reason == "security_policy_none"


# --------------------------------------------------------------------------
# AUTH_WHEN_PROXIED policy
# --------------------------------------------------------------------------


def test_proxied_direct_request_allowed(webui_env, mock_request_factory):
    webui_env.setenv("WEBUI_SECURITY", "AUTH_WHEN_PROXIED")
    manager = WebUISecurityManager()
    accessible, reason = manager.is_webui_accessible(mock_request_factory({}))
    assert accessible is True
    assert reason == "direct_request_allowed"


def test_proxied_request_blocked(webui_env, mock_request_factory):
    webui_env.setenv("WEBUI_SECURITY", "AUTH_WHEN_PROXIED")
    manager = WebUISecurityManager()
    accessible, reason = manager.is_webui_accessible(mock_request_factory({"X-Real-IP": "1.2.3.4"}))
    assert accessible is False
    assert reason == "webui_disabled_when_proxied"


# --------------------------------------------------------------------------
# ALWAYS_AUTH policy
# --------------------------------------------------------------------------


def test_always_auth_without_secret_blocks(webui_env, mock_request_factory):
    webui_env.setenv("WEBUI_SECURITY", "ALWAYS_AUTH")
    # No JWT_SECRET -> verification function not available
    manager = WebUISecurityManager()
    accessible, reason = manager.is_webui_accessible(mock_request_factory({}))
    assert accessible is False
    assert reason == "jwt_verification_not_available"


def test_always_auth_with_invalid_secret_blocks(webui_env, mock_request_factory):
    webui_env.setenv("WEBUI_SECURITY", "ALWAYS_AUTH")
    webui_env.setenv("JWT_SECRET", "tooshort")
    manager = WebUISecurityManager()
    accessible, reason = manager.is_webui_accessible(mock_request_factory({}))
    assert accessible is False


def test_always_auth_valid_jwt_accepted(webui_env, mock_request_factory, monkeypatch):
    webui_env.setenv("WEBUI_SECURITY", "ALWAYS_AUTH")
    webui_env.setenv("JWT_SECRET", STRONG_SECRET)
    # Reset auth singleton so the strong secret is picked up
    from smolrouter import auth

    monkeypatch.setattr(auth, "_jwt_auth", None)

    manager = WebUISecurityManager()
    assert manager._verify_request_auth is not None

    token = auth.JWTAuth(STRONG_SECRET).create_token({"sub": "ok"})
    request = mock_request_factory({"Authorization": f"Bearer {token}"})
    accessible, reason = manager.is_webui_accessible(request)
    assert accessible is True
    assert reason == "valid_jwt_provided"
    # _jwt_auth restored automatically by monkeypatch teardown.


def test_always_auth_missing_token_denied(webui_env, mock_request_factory, monkeypatch):
    webui_env.setenv("WEBUI_SECURITY", "ALWAYS_AUTH")
    webui_env.setenv("JWT_SECRET", STRONG_SECRET)
    from smolrouter import auth

    monkeypatch.setattr(auth, "_jwt_auth", None)

    manager = WebUISecurityManager()
    accessible, reason = manager.is_webui_accessible(mock_request_factory({}))
    assert accessible is False
    assert reason == "jwt_required"
    # _jwt_auth restored automatically by monkeypatch teardown.


# --------------------------------------------------------------------------
# check_webui_access raises
# --------------------------------------------------------------------------


def test_check_access_passes_silently_when_allowed(webui_env, mock_request_factory):
    webui_env.setenv("WEBUI_SECURITY", "NONE")
    manager = WebUISecurityManager()
    # Should not raise
    assert manager.check_webui_access(mock_request_factory({})) is None


def test_check_access_raises_403_when_proxied(webui_env, mock_request_factory):
    webui_env.setenv("WEBUI_SECURITY", "AUTH_WHEN_PROXIED")
    manager = WebUISecurityManager()
    with pytest.raises(HTTPException) as exc:
        manager.check_webui_access(mock_request_factory({"x-forwarded-for": "1.2.3.4"}))
    assert exc.value.status_code == 403
    assert exc.value.detail["error"] == "webui_disabled_when_proxied"


def test_check_access_raises_500_when_jwt_required_but_unconfigured(webui_env, mock_request_factory):
    """jwt_required reason with no JWT_SECRET set -> configuration error 500."""
    webui_env.setenv("WEBUI_SECURITY", "ALWAYS_AUTH")
    manager = WebUISecurityManager()
    # Force the jwt_required reason path even though verification isn't wired
    manager.is_webui_accessible = lambda request: (False, "jwt_required")
    with pytest.raises(HTTPException) as exc:
        manager.check_webui_access(mock_request_factory({}))
    assert exc.value.status_code == 500
    assert exc.value.detail["error"] == "configuration_error"


def test_check_access_raises_401_when_jwt_required_with_secret(webui_env, mock_request_factory):
    webui_env.setenv("WEBUI_SECURITY", "ALWAYS_AUTH")
    webui_env.setenv("JWT_SECRET", STRONG_SECRET)
    manager = WebUISecurityManager()
    manager.is_webui_accessible = lambda request: (False, "jwt_required")
    with pytest.raises(HTTPException) as exc:
        manager.check_webui_access(mock_request_factory({}))
    assert exc.value.status_code == 401
    assert exc.value.headers["WWW-Authenticate"] == "Bearer"


def test_check_access_raises_403_for_generic_denial(webui_env, mock_request_factory):
    webui_env.setenv("WEBUI_SECURITY", "NONE")
    manager = WebUISecurityManager()
    manager.is_webui_accessible = lambda request: (False, "unknown_policy_fallback")
    with pytest.raises(HTTPException) as exc:
        manager.check_webui_access(mock_request_factory({}))
    assert exc.value.status_code == 403
    assert exc.value.detail["error"] == "webui_access_denied"


# --------------------------------------------------------------------------
# singleton accessor
# --------------------------------------------------------------------------


def test_get_webui_security_is_singleton(webui_env, monkeypatch):
    monkeypatch.setattr(security, "_webui_security", None)
    s1 = get_webui_security()
    s2 = get_webui_security()
    assert s1 is s2
    # Cleanup is handled by monkeypatch's automatic restore of _webui_security.
