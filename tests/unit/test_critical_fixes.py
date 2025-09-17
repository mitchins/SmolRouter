#!/usr/bin/env python3
"""
Test the critical security fixes we implemented
Relocated into tests/ and annotated for Sonar suppression where literal IPs appear.
"""

import os
import sys
from unittest.mock import Mock, patch

# Add project to path
sys.path.insert(0, '.')

def test_header_case_sensitivity_fix():
    from smolrouter.security import WebUISecurityManager
    
    os.environ["WEBUI_SECURITY"] = "AUTH_WHEN_PROXIED"
    security = WebUISecurityManager()
    
    def create_mock_request(headers):
        request = Mock()
        request.client = Mock()
        request.client.host = "127.0.0.1"  # NOSONAR S1313
        request.headers = headers
        return request
    
    attack_cases = [
        {"X-Forwarded-For": "1.2.3.4"},
        {"X-FORWARDED-FOR": "1.2.3.4"},
        {"x-Forwarded-For": "1.2.3.4"},
        {"X-Real-IP": "1.2.3.4"},
        {"CF-Connecting-IP": "1.2.3.4"},
    ]
    
    for headers in attack_cases:
        request = create_mock_request(headers)
        accessible, reason = security.is_webui_accessible(request)
        assert not accessible


def test_jwt_secret_validation():
    from smolrouter.auth import _validate_jwt_secret
    
    weak_secrets = ["", "   ", "password", "test-secret", "a" * 31, "a" * 40]
    for secret in weak_secrets:
        assert not _validate_jwt_secret(secret)


def test_blob_size_limits():
    from smolrouter.storage import FilesystemBlobStorage, MAX_BLOB_SIZE
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = FilesystemBlobStorage(temp_dir)
        large_data = b"x" * (MAX_BLOB_SIZE + 1000)
        with patch('smolrouter.storage.logger') as mock_logger:
            key = storage.store(large_data)
            mock_logger.warning.assert_called()
            retrieved = storage.retrieve(key)
            assert len(retrieved) == MAX_BLOB_SIZE
