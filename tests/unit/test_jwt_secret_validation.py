#!/usr/bin/env python3
"""
Test JWT secret validation to prevent weak/empty secrets
Relocated into tests/.
"""

import os
from smolrouter.auth import _validate_jwt_secret


class TestJWTSecretValidation:
    def setup_method(self):
        import smolrouter.auth

        smolrouter.auth._jwt_auth = None
        os.environ.pop("JWT_SECRET", None)

    def test_empty_secrets_rejected(self):
        assert not _validate_jwt_secret(None)
        assert not _validate_jwt_secret("")
        assert not _validate_jwt_secret("   ")
