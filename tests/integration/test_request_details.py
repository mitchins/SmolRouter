"""
Integration tests for request details page.

Tests that the request details endpoint properly loads and displays request information
after the Redis migration with proper type conversions.
"""

from fastapi.testclient import TestClient
from smolrouter.app import app


class TestRequestDetailsPage:
    """Test the request details page renders correctly"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    def test_request_details_page_with_completed_request(self):
        """Test that request details page loads for a completed request"""
        # Make a test request that will fail (no upstream configured properly)
        response = self.client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "test"}]},
        )

        # Request should complete (even if it fails)
        assert response.status_code in [200, 400, 401, 404, 500, 501, 502]

        # Get recent logs to find our request ID
        logs_response = self.client.get("/api/logs")
        assert logs_response.status_code == 200
        logs = logs_response.json()

        # Should have at least one log entry
        if len(logs) > 0:
            request_id = logs[0]["id"]

            # Try to load the request details page
            details_response = self.client.get(f"/request/{request_id}")

            # Should load successfully (200 OK) - not 500 error
            assert details_response.status_code == 200

            # Should contain HTML (not JSON error)
            assert "<!DOCTYPE html>" in details_response.text or "<html" in details_response.text

            # Should not contain error messages about type comparisons
            assert "'<' not supported between instances of 'str' and 'int'" not in details_response.text
            assert "can't subtract offset-naive and offset-aware datetimes" not in details_response.text

            # Should contain some expected content
            assert request_id in details_response.text

    def test_request_details_page_with_invalid_uuid(self):
        """Test that request details page handles invalid UUIDs gracefully"""
        # Try to access a non-existent request
        response = self.client.get("/request/nonexistent-id-12345")

        # Should either return 404 or display "not found" message
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            # Should show not found message in HTML
            assert "not found" in response.text.lower() or "does not exist" in response.text.lower()

    def test_request_details_api_endpoint(self):
        """Test the JSON API endpoint for request details"""
        # Make a test request
        response = self.client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "test"}]},
        )

        # Get logs to find request ID
        logs_response = self.client.get("/api/logs")
        assert logs_response.status_code == 200
        logs = logs_response.json()

        if len(logs) > 0:
            request_id = logs[0]["id"]

            # Try the API endpoint
            api_response = self.client.get(f"/api/requests/{request_id}")

            # Should return JSON successfully
            assert api_response.status_code == 200
            data = api_response.json()

            # Should have expected fields with proper types
            assert "id" in data
            assert data["id"] == request_id

            # Status code should be an integer (or "pending" string), not a string like "500"
            if "status_code" in data and data["status_code"] is not None:
                assert isinstance(data["status_code"], (int, str))
                if isinstance(data["status_code"], str):
                    assert data["status_code"] == "pending"

            # Duration should be an integer, not a string
            if "duration_ms" in data and data["duration_ms"] is not None:
                assert isinstance(data["duration_ms"], int)
