# Security and Performance Fixes

## 1. Header Case Sensitivity Fix (Critical)

```python
# In WebUISecurityManager.__init__()
self.proxy_headers = {
    "x-forwarded-for", "x-real-ip", "cf-connecting-ip",
    "x-forwarded-proto", "x-forwarded-host", "x-original-forwarded-for"
}

def _is_proxied_request(self, request: Request) -> bool:
    # Case-insensitive header check
    request_headers = {k.lower() for k in request.headers.keys()}
    return bool(self.proxy_headers & request_headers)
```

## 2. Enhanced Test Suite

```python
def test_security_edge_cases():
    """Test edge cases and security scenarios"""
    
    # Test case sensitivity
    request = create_mock_request({"X-Forwarded-For": "1.2.3.4"})  # Capital letters
    accessible, reason = security.is_webui_accessible(request)
    assert accessible == False, "Should detect proxy headers regardless of case"
    
    # Test multiple headers
    request = create_mock_request({
        "x-forwarded-for": "1.2.3.4",
        "x-real-ip": "5.6.7.8"
    })
    
    # Test malformed headers
    request = create_mock_request({"x-forwarded-for": ""})
    
    # Test actual exception raising
    try:
        security.check_webui_access(proxied_request)
        assert False, "Should have raised HTTPException"
    except HTTPException as e:
        assert e.status_code == 403
```

## 3. Performance Optimizations

```python
class WebUISecurityManager:
    def __init__(self):
        # Pre-compile header set for O(1) lookup
        self.proxy_headers_set = {
            "x-forwarded-for", "x-real-ip", "cf-connecting-ip",
            "x-forwarded-proto", "x-forwarded-host", "x-original-forwarded-for"
        }
        
        # Move import to class level to avoid circular imports
        if self.policy == SecurityPolicy.ALWAYS_AUTH:
            from smolrouter.auth import verify_request_auth
            self._verify_request_auth = verify_request_auth
    
    def _is_proxied_request(self, request: Request) -> bool:
        """Fast O(1) proxy detection"""
        return any(k.lower() in self.proxy_headers_set for k in request.headers.keys())
```

## 4. Blob Storage Limits

```python
# In storage.py
MAX_BLOB_SIZE = int(os.getenv("MAX_BLOB_SIZE", "10485760"))  # 10MB default

def store(self, data: bytes, content_type: str = "application/json") -> str:
    if len(data) > MAX_BLOB_SIZE:
        logger.warning(f"Blob size {len(data)} exceeds limit {MAX_BLOB_SIZE}, truncating")
        data = data[:MAX_BLOB_SIZE]
    # ... rest of method
```

## 5. Rate Limiting on WebUI Access Attempts

```python
# Add to security.py
from collections import defaultdict
from time import time

class WebUISecurityManager:
    def __init__(self):
        self.failed_attempts = defaultdict(list)  # IP -> [timestamp, ...]
        self.max_attempts = 10  # per minute
        
    def _check_rate_limit(self, client_ip: str) -> bool:
        now = time()
        # Clean old attempts (older than 1 minute)
        self.failed_attempts[client_ip] = [
            t for t in self.failed_attempts[client_ip] 
            if now - t < 60
        ]
        return len(self.failed_attempts[client_ip]) < self.max_attempts
```

## 6. Configuration Reload Without Restart

```python
# Add endpoint for config reload
@app.post("/admin/reload-config")
async def reload_config(request: Request):
    # Verify admin auth
    verify_request_auth(request)  # Require JWT for admin actions
    
    # Reload router config
    global smart_router, _webui_security
    ROUTES_CONFIG_DATA = load_routes_config()
    smart_router = get_smart_router(ROUTES_CONFIG_DATA, DEFAULT_UPSTREAM)
    _webui_security = None  # Force recreation
    
    return {"status": "Config reloaded successfully"}
```