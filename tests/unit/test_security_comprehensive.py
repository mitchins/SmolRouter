#!/usr/bin/env python3
"""
Comprehensive security tests including edge cases and attack scenarios
Relocated into tests/ and annotated for Sonar suppression.
"""

import os
import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from smolrouter.security import WebUISecurityManager, SecurityPolicy


def create_mock_request(headers=None, client_ip="127.0.0.1"):
    request = Mock()
    request.client = Mock()
    request.client.host = client_ip  # NOSONAR S1313
    request.headers = headers or {}
    return request


class TestWebUISecurityComprehensive:
    def setup_method(self):
        for key in list(os.environ.keys()):
            if key.startswith("WEBUI_"):
                del os.environ[key]
        os.environ.pop("JWT_SECRET", None)
    
    def test_header_case_sensitivity_attack(self):
        os.environ["WEBUI_SECURITY"] = "AUTH_WHEN_PROXIED"
        security = WebUISecurityManager()
        attack_headers = [
            {"X-Forwarded-For": "1.2.3.4"},
            {"X-FORWARDED-FOR": "1.2.3.4"},
            {"x-Forwarded-For": "1.2.3.4"},
            {"X-Real-IP": "1.2.3.4"},
            {"CF-Connecting-IP": "1.2.3.4"},
        ]
        
        for headers in attack_headers:
            request = create_mock_request(headers)
            accessible, reason = security.is_webui_accessible(request)
            assert not accessible
