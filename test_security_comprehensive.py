#!/usr/bin/env python3
"""
Comprehensive security tests including edge cases and attack scenarios
"""

import os
import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from smolrouter.security import WebUISecurityManager, SecurityPolicy

def create_mock_request(headers=None, client_ip="127.0.0.1"):
    """Create a mock FastAPI Request object"""
    request = Mock()
    request.client = Mock()
    request.client.host = client_ip
    request.headers = headers or {}
    return request

class TestWebUISecurityComprehensive:
    
    def setup_method(self):
        """Reset environment before each test"""
        for key in list(os.environ.keys()):
            if key.startswith("WEBUI_"):
                del os.environ[key]
        os.environ.pop("JWT_SECRET", None)
    
    def test_header_case_sensitivity_attack(self):
        """Test that proxy detection works regardless of header case"""
        os.environ["WEBUI_SECURITY"] = "AUTH_WHEN_PROXIED"
        security = WebUISecurityManager()
        
        # Test various case combinations that attackers might try
        attack_headers = [
            {"X-Forwarded-For": "1.2.3.4"},           # Capital letters
            {"X-FORWARDED-FOR": "1.2.3.4"},           # All caps
            {"x-Forwarded-For": "1.2.3.4"},           # Mixed case
            {"X-Real-IP": "1.2.3.4"},                 # Different header, caps
            {"CF-Connecting-IP": "1.2.3.4"},          # Cloudflare header, caps
        ]
        
        for headers in attack_headers:
            request = create_mock_request(headers)
            accessible, reason = security.is_webui_accessible(request)
            assert not accessible, f"Should block WebUI with headers: {headers}"
            assert reason == "webui_disabled_when_proxied"
    
    def test_multiple_proxy_headers(self):
        """Test behavior with multiple proxy headers"""
        os.environ["WEBUI_SECURITY"] = "AUTH_WHEN_PROXIED"
        security = WebUISecurityManager()
        
        # Multiple headers (common in complex proxy setups)
        request = create_mock_request({
            "x-forwarded-for": "1.2.3.4, 5.6.7.8",
            "x-real-ip": "9.10.11.12",
            "cf-connecting-ip": "13.14.15.16"
        })
        
        accessible, reason = security.is_webui_accessible(request)
        assert not accessible
        assert reason == "webui_disabled_when_proxied"
    
    def test_empty_and_malformed_headers(self):
        """Test handling of empty or malformed proxy headers"""
        os.environ["WEBUI_SECURITY"] = "AUTH_WHEN_PROXIED"
        security = WebUISecurityManager()
        
        malformed_cases = [
            {"x-forwarded-for": ""},                   # Empty value
            {"x-forwarded-for": " "},                  # Whitespace only
            {"x-forwarded-for": "invalid-ip"},         # Invalid IP
            {"x-forwarded-for": "300.300.300.300"},    # Invalid IP range
        ]
        
        for headers in malformed_cases:
            request = create_mock_request(headers)
            accessible, reason = security.is_webui_accessible(request)
            # Should still detect as proxied regardless of header value validity
            assert not accessible, f"Should still block with malformed headers: {headers}"
    
    def test_exception_raising_scenarios(self):
        """Test that check_webui_access properly raises HTTPExceptions"""
        os.environ["WEBUI_SECURITY"] = "AUTH_WHEN_PROXIED"
        security = WebUISecurityManager()
        
        # Test proxied request raises 403
        proxied_request = create_mock_request({"x-forwarded-for": "1.2.3.4"})
        
        with pytest.raises(HTTPException) as exc_info:
            security.check_webui_access(proxied_request)
        
        assert exc_info.value.status_code == 403
        assert "webui_disabled_when_proxied" in str(exc_info.value.detail)
        
        # Test direct request succeeds (no exception)
        direct_request = create_mock_request()
        try:
            security.check_webui_access(direct_request)  # Should not raise
        except HTTPException:
            pytest.fail("Direct request should not raise HTTPException")
    
    def test_always_auth_without_jwt_secret(self):
        """Test ALWAYS_AUTH policy without JWT_SECRET configured"""
        os.environ["WEBUI_SECURITY"] = "ALWAYS_AUTH"
        # No JWT_SECRET set
        
        security = WebUISecurityManager()  # Should log error but not crash
        
        request = create_mock_request()
        
        with pytest.raises(HTTPException) as exc_info:
            security.check_webui_access(request)
        
        assert exc_info.value.status_code == 403  # Access denied (better than 500)  
        assert "jwt_verification_not_available" in str(exc_info.value.detail)
    
    def test_policy_fallback_on_invalid_config(self):
        """Test that invalid policy falls back to secure default"""
        os.environ["WEBUI_SECURITY"] = "INVALID_POLICY_NAME"
        
        security = WebUISecurityManager()
        
        # Should fall back to AUTH_WHEN_PROXIED
        assert security.policy == SecurityPolicy.AUTH_WHEN_PROXIED
        
        # Test that fallback policy works correctly
        proxied_request = create_mock_request({"x-forwarded-for": "1.2.3.4"})
        accessible, reason = security.is_webui_accessible(proxied_request)
        assert not accessible
    
    def test_dos_via_many_headers(self):
        """Test performance with many headers (potential DoS)"""
        os.environ["WEBUI_SECURITY"] = "AUTH_WHEN_PROXIED"
        security = WebUISecurityManager()
        
        # Create request with many headers (simulating DoS attempt)
        many_headers = {}
        for i in range(1000):
            many_headers[f"custom-header-{i}"] = f"value-{i}"
        
        # Add one proxy header among the many
        many_headers["x-forwarded-for"] = "1.2.3.4"
        
        request = create_mock_request(many_headers)
        
        # Should still work but we should measure performance
        import time
        start = time.time()
        accessible, reason = security.is_webui_accessible(request)
        duration = time.time() - start
        
        assert not accessible
        assert duration < 0.1, f"Header processing took too long: {duration}s"
    
    def test_concurrent_access_attempts(self):
        """Test thread safety of security manager"""
        import threading
        
        os.environ["WEBUI_SECURITY"] = "AUTH_WHEN_PROXIED"
        security = WebUISecurityManager()
        
        results = []
        
        def test_access():
            request = create_mock_request({"x-forwarded-for": "1.2.3.4"})
            accessible, reason = security.is_webui_accessible(request)
            results.append((accessible, reason))
        
        # Run 10 concurrent access checks
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=test_access)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All should have same result
        assert len(results) == 10
        assert all(not accessible for accessible, reason in results)
        assert all(reason == "webui_disabled_when_proxied" for accessible, reason in results)
    
    @patch('smolrouter.auth.verify_request_auth')
    def test_jwt_verification_in_always_auth(self, mock_verify):
        """Test JWT verification actually gets called in ALWAYS_AUTH mode"""
        os.environ["WEBUI_SECURITY"] = "ALWAYS_AUTH"
        os.environ["JWT_SECRET"] = "this-is-a-valid-32-character-key"
        
        security = WebUISecurityManager()
        
        # Mock successful JWT verification
        mock_verify.return_value = {"sub": "test-user"}
        
        request = create_mock_request()
        accessible, reason = security.is_webui_accessible(request)
        
        assert accessible
        assert reason == "valid_jwt_provided"
        mock_verify.assert_called_once_with(request)
        
        # Test JWT failure
        mock_verify.reset_mock()
        mock_verify.side_effect = HTTPException(status_code=401, detail="Invalid token")
        
        accessible, reason = security.is_webui_accessible(request)
        assert not accessible
        assert reason == "jwt_required"

if __name__ == "__main__":
    # Run tests manually
    test_instance = TestWebUISecurityComprehensive()
    
    print("Running comprehensive security tests...")
    
    # Run each test method
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    for method_name in test_methods:
        print(f"\n--- {method_name} ---")
        try:
            test_instance.setup_method()
            method = getattr(test_instance, method_name)
            method()
            print("✅ PASSED")
        except Exception as e:
            print(f"❌ FAILED: {e}")
    
    print("\n=== All tests completed ===")