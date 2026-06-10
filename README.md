# SmolRouter

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=mitchins_SmolRouter&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=mitchins_SmolRouter)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=mitchins_SmolRouter&metric=coverage)](https://sonarcloud.io/summary/overall?id=mitchins_SmolRouter)

SmolRouter is a lightweight AI gateway that keeps OpenAI-compatible clients running while you move traffic across any mix of providers.

## What

- OpenAI-compatible proxy with drop-in model remapping (`gpt-4` can become `llama3-70b` without touching client code).
- Smart routing engine that can balance traffic, apply per-team policies, and fail over automatically.
- Built-in observability, quotas, and security controls for production-facing deployments.

## Why

Use SmolRouter when you need to:

- Consolidate multiple model vendors or on-prem hosts behind a single OpenAI-style endpoint.
- Preserve legacy client integrations while experimenting with new models or providers.
- Inspect, audit, and meter requests without giving every application direct access to provider API keys.
- Enforce consistent policies (JWT auth, request throttling, content filters) at the edge of your AI estate.

## How

### Quick Start

1. **Run SmolRouter.** Pick Docker or Python—both expose the same HTTP interface.

   ```bash
   # Docker
   docker build -t smolrouter .
   docker run -d -p 1234:1234 \
     -e DEFAULT_UPSTREAM="http://host.docker.internal:8000" \
     -e MODEL_MAP='{"gpt-4":"llama3-70b"}' \
     smolrouter
   ```

   ```bash
   # Python
   pip install smolrouter
   export DEFAULT_UPSTREAM="http://localhost:8000"
   export MODEL_MAP='{"gpt-4":"llama3-70b"}'
   smolrouter
   ```

2. **Point clients at the router.** Swap the base URL and keep your existing model IDs.

   ```python
   import openai

   client = openai.OpenAI(
       base_url="http://localhost:1234/v1",
       api_key="local-proxy-key",  # forwarded to your upstreams
   )

   response = client.chat.completions.create(
       model="gpt-4",  # transparently remapped if MODEL_MAP defines it
       messages=[{"role": "user", "content": "Hello"}],
   )
   ```

3. **Configure deeper routing rules when ready.** Drop a `routes.yaml` next to the binary or set `ROUTES_CONFIG` to point at a shared config repo.

### Configuration

**Core Environment Variables:**

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEFAULT_UPSTREAM` | `http://localhost:8000` | Upstream endpoint used when no routing rule matches |
| `LISTEN_HOST` | `127.0.0.1` | Bind address for the FastAPI app. Switch to `0.0.0.0` only behind a reverse proxy |
| `LISTEN_PORT` | `1234` | Port that accepts OpenAI-compatible traffic |
| `MODEL_MAP` | `{}` | JSON mapping of incoming model names to replacements (exact keys or regex) |
| `ROUTES_CONFIG` | `config/routes.yaml` | Path to YAML or JSON smart-routing configuration. Relative paths are resolved from the current working directory when explicitly set; the default is the repo-local `config/routes.yaml` |
| `REQUEST_TIMEOUT` | `3000.0` | Upstream timeout in seconds |
| `ENABLE_LOGGING` | `true` | Toggle request logging, database writes, and the Web UI dashboard |

**Feature Flags:**

| Variable | Default | Purpose |
| --- | --- | --- |
| `STRIP_THINKING` | `false` | Remove `<think>...</think>` blocks before returning responses |
| `STRIP_JSON_MARKDOWN` | `false` | Convert fenced JSON markdown into raw JSON payloads |
| `DISABLE_THINKING` | `false` | Append `/no_think` hints to prompts for providers that respect it |
| `DB_PATH` | `requests.db` | SQLite database path for request metadata |
| `MAX_LOG_AGE_DAYS` | `7` | Retention window for automatic log cleanup |
| `BLOB_STORAGE_TYPE` | `filesystem` | Storage backend for request/response bodies (`filesystem` or `memory`) |
| `BLOB_STORAGE_PATH` | `~/.smolrouter/blob_storage` | Directory used when `BLOB_STORAGE_TYPE=filesystem`. Set this to `./blob_storage` for checkout-local dev storage if desired |
| `MAX_BLOB_SIZE` | `10485760` | Per-request blob size cap in bytes (10 MiB) |
| `MAX_TOTAL_STORAGE_SIZE` | `1073741824` | Aggregate blob storage cap in bytes (1 GiB) |

**Security & Auth:**

| Variable | Default | Purpose |
| --- | --- | --- |
| `JWT_SECRET` | _(unset)_ | Enables JWT authentication for `/v1/*` endpoints and most `/api/*` endpoints (some API endpoints like `/api/logs`, `/api/stats` are exempt). Must be ≥32 characters with good entropy. Leave unset for unauthenticated access |
| `WEBUI_SECURITY` | `AUTH_WHEN_PROXIED` | Controls Web UI/dashboard access policy independently of API authentication: `NONE`, `AUTH_WHEN_PROXIED`, or `ALWAYS_AUTH` |
| `WEBUI_ALLOWED_ORIGINS` | _(unset)_ | Comma-separated list of origins allowed to access the dashboard |
| `RATE_LIMIT_REQUESTS` | _(unset)_ | Requests per minute per API key. Leave empty to disable rate limiting |
| `RATE_LIMIT_TOKENS` | _(unset)_ | Token budget per minute per API key (estimated from request payloads) |

**Routing Configuration (`routes.yaml`):**

```yaml
servers:
  fast-box: "http://192.168.1.100:8000"
  backup-box: "http://192.168.1.101:8000"

aliases:
  coding-assistant:
    instances:
      - server: "fast-box"
        model: "llama3-70b"
      - server: "backup-box"
        model: "llama3-8b"

routes:
  - match:
      model: "/gpt-4.*/"
    route:
      upstream: "http://fast-box:8000"
      model: "coding-assistant"
  - match:
      source_host: "10.0.1.100"
    route:
      upstream: "http://backup-box:8000"
```

### Provider-Specific Setup

**Google GenAI (Gemini Models):**

Configure in `routes.yaml`:
```yaml
providers:
  - name: "google-prod"
    type: "google-genai"
    enabled: true
    api_keys:
      - "YOUR_GOOGLE_API_KEY_1"
      - "YOUR_GOOGLE_API_KEY_2"  # Multiple keys for rotation
    max_requests_per_day: 1500  # Tune this to your quota plan
```

Google GenAI models are discovered live from the provider. Use the provider and system dashboards to inspect configured backends, quotas, and proxy status.

OpenAI-compatible providers can also point at vendor-prefixed API bases such as `/openai/v1` or `/zen/go/v1`. When a vendor does not expose a usable `/v1/models` listing, set `static_models` explicitly in YAML:

```yaml
providers:
  - name: "groq-scout"
    type: "openai"
    url: "https://api.groq.com/openai/v1"
    api_key: "YOUR_GROQ_KEY"
    static_models:
      - "meta-llama/llama-4-scout-17b-16e-instruct"
```

## Features

**Intelligent routing:**
- Match traffic on model IDs, regex patterns, or request metadata such as source IPs
- Rewrite either the upstream target or the model name on a per-rule basis
- Compose reusable aliases that handle provider failover or split traffic by weight
- Layer rule sets from `routes.yaml` with environment-driven defaults

**Protocol compatibility:**
- OpenAI, Ollama, and Google GenAI transports with shared request/response semantics
- Streaming for chat, completions, and Ollama generate endpoints
- Legacy model remapping via `MODEL_MAP` continues to work exactly as it did in 1.x
- Optional content transformations: `<think>` scrubbing, fenced JSON stripping, and `/no_think` hints

**Observability:**
- Persistent request log with token estimation, latency histograms, and recent traffic views
- Scatter plots to visualize token counts versus latency for capacity planning
- Blob storage abstraction for request/response payloads with size limits and retention policies
- Quota tracking and rate limiting at the API key or token level

## Operations & Security

**Deployment:**
- Run behind a trusted reverse proxy (nginx, Caddy, Cloudflare) and publish only HTTPS endpoints
- Bind to `127.0.0.1` by default; switch to `0.0.0.0` only when the proxy handles TLS and authentication
- Keep provider API keys and upstream URLs in environment variables or a secret manager

**Authentication:**
1. Set `JWT_SECRET` to a 32+ character random value to enable JWT authentication for `/v1/*` endpoints and most `/api/*` endpoints. Weak secrets are rejected during startup
2. Choose a Web UI access policy with `WEBUI_SECURITY` (controls dashboard/UI access independently):
   - `NONE` — no authentication required for Web UI (local development only)
   - `AUTH_WHEN_PROXIED` — require auth when `X-Forwarded-For` headers are present (default)
   - `ALWAYS_AUTH` — always require authentication for Web UI access
3. Important: When `JWT_SECRET` is set, most API requests require a valid JWT token in the `Authorization` header. The following endpoints are exempt: `/api/logs`, `/api/stats`, `/api/inflight`, `/api/performance`. Leave `JWT_SECRET` unset for completely unauthenticated access
4. Pair JWT auth with rate limiting by defining `RATE_LIMIT_REQUESTS` or `RATE_LIMIT_TOKENS` for each API key

**Logging & Retention:**
- `ENABLE_LOGGING=false` disables the request log and UI for ultra-lightweight proxies
- Request metadata uses the Redis-backed logging path, with blob storage for large request/response payloads
- Adjust `MAX_LOG_AGE_DAYS`, `MAX_BLOB_SIZE`, and `MAX_TOTAL_STORAGE_SIZE` to control cost
- Background cleanup jobs run automatically during the FastAPI lifespan events

**Path guidance:**
- Use `-C/--config` or `ROUTES_CONFIG` for deterministic routing config selection.
- Use `BLOB_STORAGE_PATH` explicitly in production if you want a location other than the safe default under `~/.smolrouter/blob_storage`.
- For dev checkouts, `BLOB_STORAGE_PATH=./blob_storage` keeps blobs next to the repo as before.

## Testing

SmolRouter ships with an automated pytest suite covering model remapping, routing, logging, and quota enforcement:

```bash
pip install -e .[dev]
pytest
```

## Next Steps

- [Development guide](DEVELOPMENT.md) — setup, architecture, infrastructure, and contribution guide
- [Project status](STATUS.md) — roadmap, tasks, and current progress

## Changelog

**2.0.0** - Production-ready AI gateway with Google GenAI support, enhanced observability, and hardened security. See full details in [release notes](docs/RELEASE_NOTES_v2.0.0.md).
