#!/usr/bin/env python3
"""
Simple test script to verify WebUI security policy logic
Relocated into tests/.
"""

from smolrouter.security import WebUISecurityManager


def test_policy_scenarios(webui_env, mock_request_factory):
    webui_env.setenv("WEBUI_SECURITY", "NONE")
    security = WebUISecurityManager()
    request = mock_request_factory()
    accessible, _ = security.is_webui_accessible(request)
    assert accessible
