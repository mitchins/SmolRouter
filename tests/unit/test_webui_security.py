#!/usr/bin/env python3
"""
Simple test script to verify WebUI security policy logic
Relocated into tests/.
"""

import os
from unittest.mock import Mock
from smolrouter.security import WebUISecurityManager


def create_mock_request(headers=None):
    request = Mock()
    request.client = Mock()
    request.client.host = "127.0.0.1"  # NOSONAR S1313
    request.headers = headers or {}
    return request


def test_policy_scenarios():
    os.environ["WEBUI_SECURITY"] = "NONE"
    os.environ.pop("JWT_SECRET", None)
    security = WebUISecurityManager()
    request = create_mock_request()
    accessible, reason = security.is_webui_accessible(request)
    assert accessible
