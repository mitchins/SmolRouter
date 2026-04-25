#!/usr/bin/env python3
"""
Comprehensive security tests including edge cases and attack scenarios
Relocated into tests/ and annotated for Sonar suppression.
"""

from smolrouter.security import WebUISecurityManager


class TestWebUISecurityComprehensive:
    def test_header_case_sensitivity_attack(self, webui_env, mock_request_factory):
        webui_env.setenv("WEBUI_SECURITY", "AUTH_WHEN_PROXIED")
        security = WebUISecurityManager()
        attack_headers = [
            {"X-Forwarded-For": "1.2.3.4"},
            {"X-FORWARDED-FOR": "1.2.3.4"},
            {"x-Forwarded-For": "1.2.3.4"},
            {"X-Real-IP": "1.2.3.4"},
            {"CF-Connecting-IP": "1.2.3.4"},
        ]

        for headers in attack_headers:
            request = mock_request_factory(headers)
            accessible, _ = security.is_webui_accessible(request)
            assert not accessible
