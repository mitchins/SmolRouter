#!/usr/bin/env python3
"""
Simple test script to verify WebUI security policy logic
"""

import os
from unittest.mock import Mock
from smolrouter.security import WebUISecurityManager, SecurityPolicy

def create_mock_request(headers=None):
    """Create a mock FastAPI Request object"""
    request = Mock()
    request.client = Mock()
    request.client.host = "127.0.0.1"
    request.headers = headers or {}
    return request

def test_policy_scenarios():
    """Test different security policy scenarios"""
    
    print("=== WebUI Security Policy Test Scenarios ===\n")
    
    # Test 1: NONE policy
    print("Test 1: WEBUI_SECURITY=NONE")
    os.environ["WEBUI_SECURITY"] = "NONE"
    os.environ.pop("JWT_SECRET", None)  # Remove JWT secret
    security = WebUISecurityManager()
    
    # Direct request - should be accessible
    request = create_mock_request()
    accessible, reason = security.is_webui_accessible(request)
    print(f"  Direct request: accessible={accessible}, reason={reason}")
    
    # Proxied request - should be accessible (NONE policy)
    request = create_mock_request({"x-forwarded-for": "1.2.3.4"})
    accessible, reason = security.is_webui_accessible(request)
    print(f"  Proxied request: accessible={accessible}, reason={reason}")
    
    # Test 2: AUTH_WHEN_PROXIED policy (default)
    print("\nTest 2: WEBUI_SECURITY=AUTH_WHEN_PROXIED (default)")
    os.environ["WEBUI_SECURITY"] = "AUTH_WHEN_PROXIED"
    os.environ.pop("JWT_SECRET", None)  # No JWT needed for this policy
    security = WebUISecurityManager()
    
    # Direct request - should be accessible
    request = create_mock_request()
    accessible, reason = security.is_webui_accessible(request)
    print(f"  Direct request: accessible={accessible}, reason={reason}")
    
    # Proxied request - should be disabled
    request = create_mock_request({"x-forwarded-for": "1.2.3.4"})
    accessible, reason = security.is_webui_accessible(request)
    print(f"  Proxied request: accessible={accessible}, reason={reason}")
    
    # Various proxy headers
    for header in ["x-real-ip", "cf-connecting-ip", "x-forwarded-proto"]:
        request = create_mock_request({header: "some-value"})
        accessible, reason = security.is_webui_accessible(request)
        print(f"  With {header} header: accessible={accessible}")
    
    # Test 3: ALWAYS_AUTH policy
    print("\nTest 3: WEBUI_SECURITY=ALWAYS_AUTH")
    os.environ["WEBUI_SECURITY"] = "ALWAYS_AUTH"
    os.environ["JWT_SECRET"] = "test-secret"
    security = WebUISecurityManager()
    
    # Direct request without JWT - should be denied
    request = create_mock_request()
    accessible, reason = security.is_webui_accessible(request)
    print(f"  Direct request (no JWT): accessible={accessible}, reason={reason}")
    
    # Proxied request without JWT - should be denied
    request = create_mock_request({"x-forwarded-for": "1.2.3.4"})
    accessible, reason = security.is_webui_accessible(request)
    print(f"  Proxied request (no JWT): accessible={accessible}, reason={reason}")
    
    # Test 4: Configuration error scenarios
    print("\nTest 4: Configuration validation")
    
    # ALWAYS_AUTH without JWT_SECRET
    os.environ["WEBUI_SECURITY"] = "ALWAYS_AUTH"
    os.environ.pop("JWT_SECRET", None)
    print("  ALWAYS_AUTH without JWT_SECRET:")
    security = WebUISecurityManager()  # Should log errors
    
    # Invalid policy
    os.environ["WEBUI_SECURITY"] = "INVALID_POLICY"
    print("  Invalid policy:")
    security = WebUISecurityManager()  # Should fall back to AUTH_WHEN_PROXIED
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_policy_scenarios()