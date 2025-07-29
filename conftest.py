"""
Pytest configuration for OpenAI Model Rerouter tests
"""

import pytest
import tempfile
import os
from unittest.mock import patch
from peewee import SqliteDatabase

from smolrouter.database import RequestLog


@pytest.fixture(scope="function")
def isolated_db():
    """
    Create an isolated in-memory database for each test.
    This ensures tests don't interfere with each other or the production database.
    """
    # Create a temporary file for the test database
    temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_file.close()
    
    test_db = SqliteDatabase(temp_file.name)
    
    # Store the original database reference
    original_db = RequestLog._meta.database
    
    # Patch the database in the model
    RequestLog._meta.database = test_db
    
    # Connect and create tables
    test_db.connect()
    test_db.create_tables([RequestLog], safe=True)
    
    yield test_db
    
    # Cleanup: restore original database and remove temp file
    test_db.close()
    RequestLog._meta.database = original_db
    try:
        os.unlink(temp_file.name)
    except OSError:
        pass  # File might already be deleted


@pytest.fixture(scope="function") 
def disable_logging():
    """
    Disable logging during regular API tests to avoid database side effects.
    """
    with patch('smolrouter.app.ENABLE_LOGGING', False):
        yield


# Automatically use isolated database for logging tests
pytest_plugins = []