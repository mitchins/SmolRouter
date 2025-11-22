# Development guide

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

This installs runtime dependencies (FastAPI, redis, google-genai, aiohttp) plus the development toolchain (pytest, flake8, ruff, pre-commit).

## Common tasks

```bash
make dev-install   # optional helper that runs pip install -e .[dev]
make test          # run the full pytest suite (160 tests)
make lint          # flake8 + ruff checks
make check         # lint + tests in one go
make pre-commit    # execute the configured pre-commit hooks
```

## Test suites

- **Unit tests** cover request routing, model remapping, quota logic, security guards, and provider integrations.
- **Integration tests** exercise the FastAPI app via ASGI clients, ensuring compatibility across OpenAI, Ollama, and Google GenAI paths.
- **Load tests** (`test_load.py`) provide a harness for stress-testing and require `aiohttp` when executed.

The official release build runs the entire suite; keep your branch green by mirroring the same `pytest` command locally or in CI.

## Coding standards

- Keep imports free of side-effect catching (no try/except around imports).
- Prefer small, focused modules; leverage the provider factory instead of hard-coding upstream behaviour.
- Update documentation in `README.md` or `DEVELOPMENT.md` when you add new configuration flags or observable metrics.

## Architecture

### Provider System

SmolRouter uses a provider factory pattern with dependency injection to support multiple AI backends:

- **Provider Factory** – Creates provider instances based on configuration type (openai, ollama, google-genai)
- **Provider Abstraction** – Common interface for all providers with backend-specific implementations
- **Quota Management** – Per-key, per-model tracking with timezone-aware quota resets

### Ground Truth Verification System

Ensures 100% certainty about which API key and proxy were actually used for each request through transport layer observation.

**Architecture:**
```
┌─────────────────────┐
│  Google GenAI SDK   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────┐
│ ObservingHTTPTransport      │  ◄── Captures ground truth
│ - Intercepts handle_request │  ◄── Extracts API key from headers
│ - Observes proxy config     │  ◄── Observes actual connection
│ - Delegates to wrapped      │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────┐
│ httpx.HTTPTransport │  ◄── Actual HTTP transport
│ (with proxy config) │
└─────────────────────┘
```

**What we observe:**
- API Key: Extracted from actual `x-goog-api-key` header sent on wire
- Proxy: Extracted from transport's `_pool._proxy_url` (actual connection)
- URL, Method, Host: Actual request details
- Status Code: Actual response

**Design Principles:**
1. Observe, don't interfere – Zero impact on actual requests
2. Delegate, don't duplicate – Wrap, don't reimplement
3. Verify, don't trust – Compare intent vs reality
4. Evidence, not belief – Store actual headers/connections
5. Explicit, not implicit – Clear logging of what was observed

### Security Architecture

**Header Case Sensitivity Fix:**
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

**Performance Optimizations:**
- Pre-compile header set for O(1) lookup
- Move imports to class level to avoid circular dependencies
- Fast O(1) proxy detection

**Blob Storage Limits:**
```python
MAX_BLOB_SIZE = int(os.getenv("MAX_BLOB_SIZE", "10485760"))  # 10MB default

def store(self, data: bytes, content_type: str = "application/json") -> str:
    if len(data) > MAX_BLOB_SIZE:
        logger.warning(f"Blob size {len(data)} exceeds limit {MAX_BLOB_SIZE}, truncating")
        data = data[:MAX_BLOB_SIZE]
    # ... rest of method
```

## Production Deployment

### Redis Hot Path Configuration

**Performance:** 120+ TPS achieved (4.8x improvement over SQLite ~25 TPS)

**Environment Variables:**
```bash
# Redis Configuration
REDIS_URL=redis://your-redis-host:6379
UVICORN_WORKERS=4

# Production Settings
PERSIST_DB=false  # Use Redis for hot path
ENABLE_LOGGING=true
REQUEST_TIMEOUT=30.0
```

**Redis Server Settings:**
```bash
# /etc/redis/redis.conf production settings
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000

# Optional: Enable AOF for durability
appendonly yes
appendfsync everysec
```

**Uvicorn Production Command:**
```bash
uvicorn smolrouter.app:app \
  --workers 4 \
  --loop uvloop \
  --no-access-log \
  --host 0.0.0.0 \
  --port 8000
```

### Production Hardening

**Connection Pooling:**
- `max_connections`: Scales with workers (4 workers × 64 = 256 connections)
- `socket_timeout`: 2s per operation
- `socket_connect_timeout`: 1s connection timeout
- `health_check_interval`: 30s keep-alive

**Circuit Breaker:**
- Opens after 5 consecutive failures
- 30s reset timeout
- Non-blocking fallback to prevent request blocking

**Atomic Operations:**
- Lua scripts with EVALSHA optimization
- FakeRedis compatibility for development
- Pipeline fallbacks for unsupported operations

**Monitoring Requirements:**
```python
# Metrics to emit (example using Prometheus)
redis_op_latency_ms.observe(duration)
redis_errors_total.labels(kind="timeout").inc()
requests_per_second.set(current_tps)
```

**Alert Thresholds:**
- Redis error rate > 1%
- Redis operation p95 > 50ms
- Connection pool utilization > 80%
- Circuit breaker state = OPEN

**Security Checklist:**
- [ ] Redis AUTH enabled
- [ ] TLS encryption for off-box Redis
- [ ] Network policies restrict access to app subnets
- [ ] `ulimit -n` set to ≥65k for high concurrency
- [ ] Redis not exposed to public internet

### Performance Comparison

| Backend | Sequential TPS | Concurrency | Notes |
|---------|---------------|-------------|-------|
| SQLite | ~25 | Limited | Blocking bottleneck |
| Redis | 120+ | High | Async parallelism |
| **Improvement** | **4.8x faster** | **Unlimited** | **Hot path optimized** |

## Authentication System

### JWT Authentication

SmolRouter supports JWT authentication for securing the admin dashboard when accessed externally.

**Quick Setup:**
```bash
# Generate a strong JWT secret (32+ characters)
export JWT_SECRET="your-super-secure-32-character-secret"

# Enable always-on authentication for external access
export WEBUI_SECURITY="ALWAYS_AUTH"

# Start SmolRouter
python -m smolrouter.cli
```

**Security Policies:**

| Policy | Description | When to Use |
|--------|-------------|-------------|
| `NONE` | No authentication required | Local development only |
| `AUTH_WHEN_PROXIED` | Auth required when reverse proxy detected | **Default** - good for most setups |
| `ALWAYS_AUTH` | Always require JWT | External access/production |

**JWT Secret Requirements:**
- ✅ Minimum 32 characters long
- ✅ At least 8 unique characters
- ✅ Not a common/weak password
- ✅ Cryptographically secure random

**Generate a Secure Secret:**
```bash
# Option 1: OpenSSL
openssl rand -base64 48

# Option 2: Python
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

**Creating JWT Tokens:**
```python
from smolrouter.auth import get_jwt_auth

# Get auth instance (requires JWT_SECRET to be set)
auth = get_jwt_auth()

# Create a token for a user  
token = auth.create_token({
    "sub": "admin_user",        # Subject (username)
    "name": "Admin User",       # Display name
    "role": "admin"             # Role/permissions
}, expires_in_hours=24)         # 24 hour expiry

print(f"Authorization: Bearer {token}")
```

**Security Features:**
- ✅ Strong secret validation – Rejects weak/common passwords
- ✅ Automatic token expiry – 24 hour default, configurable
- ✅ Case-insensitive proxy detection – Prevents header bypass attacks
- ✅ Detailed security logging – Audit access attempts
- ✅ Graceful fallbacks – Secure defaults on configuration errors

**Authentication Scope:**
When `JWT_SECRET` is configured, JWT authentication is required for:
- All `/v1/*` endpoints (chat completions, completions, etc.)
- All `/api/*` endpoints (logs, stats, performance, etc.)
- WebUI/dashboard pages (when `WEBUI_SECURITY` policy requires it)

The following paths are exempt from JWT authentication:
- `/` (Dashboard - controlled by `WEBUI_SECURITY` policy)
- `/performance` (Performance dashboard - controlled by `WEBUI_SECURITY` policy)
- `/static/*` (Static assets)
- `/request/*` (Request details)

To allow unauthenticated API traffic, do not set `JWT_SECRET`. The `WEBUI_SECURITY` policy controls dashboard access independently.

## Web UI Navigation

**Quick access to the Upstreams view:**
1. Open the Dashboard (`/`)
2. Use either navigation option:
   - Top navigation bar → Upstreams
   - "Recent Requests" action buttons → Upstreams (purple button)
3. The Upstreams page is also linked from the Performance view header

**Page Flow:**
```
Dashboard (/)
 ├─ Performance (/performance)
 │   └─ Upstreams (/upstreams)
 └─ Upstreams (/upstreams)
     ├─ Dashboard (/)
     └─ Performance (/performance)
```

**What the Upstreams page shows:**
- Summary cards — provider count, health status, total models, cache entries
- Controls — refresh provider data or clear the discovery cache
- Provider cards — health indicator, provider type, endpoint URL, priority, available models, alias coverage
- Cache metrics — TTL settings and hit counters for each provider

## Troubleshooting

- Missing dependencies (e.g., `redis`, `google.api_core`) indicate `pip install -e .[dev]` was skipped.
- Redis features default to `fakeredis`, so tests run without a live server. For manual testing, point `REDIS_URL` at your instance.
- If the Web UI cannot serve templates, confirm the package data section in `pyproject.toml` includes `smolrouter/templates`.

### Common Issues

**"JWT_SECRET is too short"**
```bash
# Your secret needs to be at least 32 characters
export JWT_SECRET="make-this-secret-at-least-32-characters-long"
```

**"JWT_SECRET appears to be a weak/default secret"**
```bash
# Don't use common passwords - generate a random secret
export JWT_SECRET=$(openssl rand -base64 48)
```

**Dashboard shows 403 errors**
```bash
# Include JWT token in Authorization header
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:1234/
```
