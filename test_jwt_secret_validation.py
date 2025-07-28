#!/usr/bin/env python3
"""
Test JWT secret validation to prevent weak/empty secrets
"""

import os
import pytest
from unittest.mock import patch, Mock
from smolrouter.auth import _validate_jwt_secret, get_jwt_auth, JWTAuth

class TestJWTSecretValidation:
    
    def setup_method(self):
        """Reset global auth state before each test"""
        # Reset global auth instance
        import smolrouter.auth
        smolrouter.auth._jwt_auth = None
        
        # Clean environment
        os.environ.pop("JWT_SECRET", None)
    
    def test_empty_secrets_rejected(self):
        """Test that empty/None secrets are rejected"""
        # None secret
        assert not _validate_jwt_secret(None)
        
        # Empty string
        assert not _validate_jwt_secret("")
        
        # Whitespace only
        assert not _validate_jwt_secret("   ")
        assert not _validate_jwt_secret("\t\n  ")
    
    def test_short_secrets_rejected(self):
        """Test that secrets shorter than 32 chars are rejected"""
        # Too short
        assert not _validate_jwt_secret("short")
        assert not _validate_jwt_secret("a" * 31)  # 31 chars
        
        # Exactly 32 chars should pass (if not weak)
        valid_32_char = "this-is-a-valid-32-character-key"
        assert len(valid_32_char) == 32
        assert _validate_jwt_secret(valid_32_char)
    
    def test_weak_default_secrets_rejected(self):
        """Test that common weak/default secrets are rejected"""
        weak_secrets = [
            "your-secret-key",
            "test-secret",
            "password",
            "secret",
            "jwt-secret",
            "my-secret-key",
            "development-secret-key",
            "123456789012345678901234567890123",  # 33 digit string
        ]
        
        for weak_secret in weak_secrets:
            assert not _validate_jwt_secret(weak_secret), f"Weak secret should be rejected: {weak_secret}"
            
            # Test case variations
            assert not _validate_jwt_secret(weak_secret.upper())
            assert not _validate_jwt_secret(weak_secret.capitalize())
    
    def test_repeated_character_secrets_rejected(self):
        """Test that secrets with too few unique characters are rejected"""
        # All same character
        assert not _validate_jwt_secret("a" * 40)
        
        # Only 2 unique characters  
        assert not _validate_jwt_secret("abababababababababababababababab")
        
        # Only 7 unique characters (less than 8 required)
        assert not _validate_jwt_secret("abcdefg" * 10)
        
        # 8 unique characters should pass
        valid_secret = "abcdefgh" * 4  # 32 chars, 8 unique
        assert _validate_jwt_secret(valid_secret)
    
    def test_valid_secrets_accepted(self):
        """Test that properly generated secrets are accepted"""
        valid_secrets = [
            # 32+ chars, good entropy
            "this-is-a-valid-32-character-key",
            "Kx9#mP2$vN8@qR5&wT7!eY4^uI0*oL3%",
            "secure-jwt-secret-with-good-entropy-12345",
            "abcdefghijklmnopqrstuvwxyz123456",  # Simple but meets requirements
        ]
        
        for secret in valid_secrets:
            assert _validate_jwt_secret(secret), f"Valid secret should be accepted: {secret}"
    
    def test_whitespace_trimming(self):
        """Test that leading/trailing whitespace is handled correctly"""
        # Valid secret with whitespace should be trimmed and accepted
        secret_with_spaces = "  this-is-a-valid-32-character-key  "
        assert _validate_jwt_secret(secret_with_spaces)
        
        # But if trimming makes it too short, should be rejected
        short_with_spaces = "  short  "
        assert not _validate_jwt_secret(short_with_spaces)
    
    @patch('smolrouter.auth.logger')
    def test_validation_error_logging(self, mock_logger):
        """Test that validation failures are properly logged"""
        # Test empty secret logging
        _validate_jwt_secret("")
        mock_logger.error.assert_called_with("JWT_SECRET is empty or whitespace-only")
        
        mock_logger.reset_mock()
        
        # Test short secret logging
        _validate_jwt_secret("short")
        mock_logger.error.assert_called_with("JWT_SECRET is too short (5 chars). Minimum 32 characters required for security.")
        
        mock_logger.reset_mock()
        
        # Test weak secret logging
        _validate_jwt_secret("password")
        mock_logger.error.assert_called_with("JWT_SECRET appears to be a weak/default secret. Use a cryptographically secure random key.")
        
        mock_logger.reset_mock()
        
        # Test repeated character logging
        _validate_jwt_secret("a" * 40)
        mock_logger.error.assert_called_with("JWT_SECRET has too few unique characters. Use a cryptographically secure random key.")
    
    def test_get_jwt_auth_with_invalid_secrets(self):
        """Test that get_jwt_auth handles invalid secrets correctly"""
        # Test with empty secret
        os.environ["JWT_SECRET"] = ""
        auth = get_jwt_auth()
        assert auth is None
        
        # Reset global state
        import smolrouter.auth
        smolrouter.auth._jwt_auth = None
        
        # Test with weak secret
        os.environ["JWT_SECRET"] = "password"
        auth = get_jwt_auth()
        assert auth is None
        
        # Reset global state
        smolrouter.auth._jwt_auth = None
        
        # Test with valid secret
        os.environ["JWT_SECRET"] = "this-is-a-valid-32-character-key"
        auth = get_jwt_auth()
        assert auth is not None
        assert isinstance(auth, JWTAuth)
    
    def test_security_policy_with_invalid_jwt_secret(self):
        """Test that security policy handles invalid JWT secrets correctly"""
        from smolrouter.security import WebUISecurityManager, SecurityPolicy
        
        # Test ALWAYS_AUTH with invalid secret
        os.environ["WEBUI_SECURITY"] = "ALWAYS_AUTH"
        os.environ["JWT_SECRET"] = "weak"  # Too short
        
        # Should create manager but log error about inaccessible WebUI
        with patch('smolrouter.security.logger') as mock_logger:
            security = WebUISecurityManager()
            
            # Should log error about invalid JWT_SECRET
            mock_logger.error.assert_any_call("WEBUI_SECURITY is set to ALWAYS_AUTH but JWT_SECRET is not configured!")
    
    def test_auth_with_various_token_formats(self):
        """Test auth with different token formats to ensure empty tokens are rejected"""
        os.environ["JWT_SECRET"] = "this-is-a-valid-32-character-key"
        auth = get_jwt_auth()
        
        # Test empty token
        assert auth.verify_token("") is None
        
        # Test whitespace token
        assert auth.verify_token("   ") is None
        
        # Test just "Bearer " with no token
        assert auth.verify_token("Bearer ") is None
        
        # Test Bearer with empty token
        assert auth.verify_token("Bearer    ") is None
    
    def test_environment_variable_edge_cases(self):
        """Test various edge cases with JWT_SECRET environment variable"""
        import smolrouter.auth
        
        # Test with only whitespace in env var
        os.environ["JWT_SECRET"] = "   \t\n   "
        smolrouter.auth._jwt_auth = None
        auth = get_jwt_auth()
        assert auth is None
        
        # Test with newlines (common copy-paste error)
        os.environ["JWT_SECRET"] = "this-is-a-valid-32-character-key\n"
        smolrouter.auth._jwt_auth = None
        auth = get_jwt_auth()
        assert auth is not None  # Should trim and accept
        
        # Test with tabs and spaces
        os.environ["JWT_SECRET"] = "\t  this-is-a-valid-32-character-key  \t"
        smolrouter.auth._jwt_auth = None 
        auth = get_jwt_auth()
        assert auth is not None  # Should trim and accept

if __name__ == "__main__":
    # Run tests manually
    test_instance = TestJWTSecretValidation()
    
    print("🔐 Running JWT Secret Validation Tests...")
    
    # Get all test methods
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        print(f"\n--- {method_name} ---")
        try:
            test_instance.setup_method()
            method = getattr(test_instance, method_name)
            method()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
    
    print(f"\n=== Test Results: {passed} passed, {failed} failed ===")
    
    if failed == 0:
        print("🎉 All JWT secret validation tests passed!")
    else:
        print("⚠️  Some tests failed - security vulnerabilities may exist")