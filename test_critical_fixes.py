#!/usr/bin/env python3
"""
Test the critical security fixes we implemented
"""

import os
import sys
from unittest.mock import Mock, patch

# Add project to path
sys.path.insert(0, '.')

def test_header_case_sensitivity_fix():
    """Test that header case sensitivity vulnerability is fixed"""
    print("🔍 Testing header case sensitivity fix...")
    
    try:
        from smolrouter.security import WebUISecurityManager
        
        # Set up environment
        os.environ["WEBUI_SECURITY"] = "AUTH_WHEN_PROXIED"
        security = WebUISecurityManager()
        
        # Create mock request with various header cases
        def create_mock_request(headers):
            request = Mock()
            request.client = Mock()
            request.client.host = "127.0.0.1" 
            request.headers = headers
            return request
        
        # Test different case variations that would bypass old implementation
        attack_cases = [
            {"X-Forwarded-For": "1.2.3.4"},           # Capital X
            {"X-FORWARDED-FOR": "1.2.3.4"},           # All caps
            {"x-Forwarded-For": "1.2.3.4"},           # Mixed case
            {"X-Real-IP": "1.2.3.4"},                 # Different header
            {"CF-Connecting-IP": "1.2.3.4"},          # Cloudflare
        ]
        
        all_blocked = True
        for headers in attack_cases:
            request = create_mock_request(headers)
            accessible, reason = security.is_webui_accessible(request)
            
            if accessible:
                print(f"❌ FAILED: Headers {headers} were not blocked!")
                all_blocked = False
            else:
                print(f"✅ Correctly blocked: {headers}")
        
        # Test that direct requests still work
        direct_request = create_mock_request({})
        accessible, reason = security.is_webui_accessible(direct_request)
        if not accessible:
            print("❌ FAILED: Direct request was incorrectly blocked!")
            all_blocked = False 
        else:
            print("✅ Direct request correctly allowed")
        
        assert all_blocked, "Some requests were not properly blocked"
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        assert False, f"Test failed with exception: {e}"

def test_jwt_secret_validation():
    """Test that weak JWT secrets are rejected"""
    print("\n🔐 Testing JWT secret validation...")
    
    try:
        from smolrouter.auth import _validate_jwt_secret
        
        # Test cases that should FAIL
        weak_secrets = [
            "",                           # Empty
            "   ",                       # Whitespace only
            "password",                  # Common weak
            "test-secret",              # Common weak
            "a" * 31,                   # Too short
            "a" * 40,                   # Repeated chars
        ]
        
        all_rejected = True
        for secret in weak_secrets:
            if _validate_jwt_secret(secret):
                print(f"❌ FAILED: Weak secret was accepted: '{secret}'")
                all_rejected = False
            else:
                print(f"✅ Correctly rejected weak secret: '{secret}'")
        
        # Test cases that should PASS
        strong_secrets = [
            "this-is-a-valid-32-character-key",
            "Kx9#mP2$vN8@qR5&wT7!eY4^uI0*oL3%", 
            "secure-jwt-secret-with-good-entropy-12345",
        ]
        
        for secret in strong_secrets:
            if not _validate_jwt_secret(secret):
                print(f"❌ FAILED: Strong secret was rejected: '{secret}'")
                all_rejected = False
            else:
                print(f"✅ Correctly accepted strong secret: '{secret[:8]}...'")
        
        assert all_rejected, "Some weak secrets were incorrectly accepted or strong secrets were rejected"
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        assert False, f"Test failed with exception: {e}"

def test_blob_size_limits():
    """Test that blob size limits prevent DoS"""
    print("\n💾 Testing blob size limits...")
    
    try:
        from smolrouter.storage import FilesystemBlobStorage, MAX_BLOB_SIZE
        
        # Create temporary storage
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FilesystemBlobStorage(temp_dir)
            
            # Test oversized blob gets truncated
            large_data = b"x" * (MAX_BLOB_SIZE + 1000)  # Larger than limit
            
            with patch('smolrouter.storage.logger') as mock_logger:
                key = storage.store(large_data)
                
                # Should have logged a warning about truncation
                mock_logger.warning.assert_called()
                warning_call = mock_logger.warning.call_args[0][0]
                assert "exceeds limit" in warning_call, "No truncation warning logged"
                
                # Retrieved data should be truncated
                retrieved = storage.retrieve(key)
                assert len(retrieved) == MAX_BLOB_SIZE, f"Retrieved data was {len(retrieved)} bytes, expected {MAX_BLOB_SIZE}"
                
                print(f"✅ Large blob correctly truncated to {MAX_BLOB_SIZE} bytes")
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        assert False, f"Test failed with exception: {e}"

def test_performance_improvements():
    """Test that performance improvements work correctly"""
    print("\n⚡ Testing performance improvements...")
    
    try:
        from smolrouter.security import WebUISecurityManager
        import time
        
        os.environ["WEBUI_SECURITY"] = "AUTH_WHEN_PROXIED"
        security = WebUISecurityManager()
        
        # Create request with many headers (potential DoS)
        many_headers = {}
        for i in range(500):  # 500 headers
            many_headers[f"custom-header-{i}"] = f"value-{i}"
        
        # Add one proxy header
        many_headers["x-forwarded-for"] = "1.2.3.4"
        
        def create_mock_request(headers):
            request = Mock()
            request.client = Mock()
            request.client.host = "127.0.0.1"
            request.headers = headers
            return request
        
        request = create_mock_request(many_headers)
        
        # Time the operation
        start = time.time()
        accessible, reason = security.is_webui_accessible(request)
        duration = time.time() - start
        
        # Should be fast despite many headers
        assert duration <= 0.1, f"Operation took too long: {duration:.3f}s"
        
        # Should correctly identify as proxied
        assert not accessible, "Request with proxy header was not blocked"
        
        print(f"✅ Fast header processing: {duration:.3f}s for {len(many_headers)} headers")
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        assert False, f"Test failed with exception: {e}"

def main():
    """Run all critical security tests"""
    print("🛡️  Running Critical Security Fix Tests\n")
    
    tests = [
        ("Header Case Sensitivity", test_header_case_sensitivity_fix),
        ("JWT Secret Validation", test_jwt_secret_validation), 
        ("Blob Size Limits", test_blob_size_limits),
        ("Performance Improvements", test_performance_improvements),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"📋 Running {test_name} tests...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"💥 {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All critical security fixes are working correctly!")
        print("🔒 The application is now secure against the identified vulnerabilities.")
    else:
        print(f"\n⚠️  {len(results) - passed} critical security issues remain!")
        print("🚨 Do not deploy until all tests pass!")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)