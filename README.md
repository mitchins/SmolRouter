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
       api_key="local-proxy-key",  # forwarded upstream only for keyless OpenAI BYOK/passthrough providers
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
| `LOG_LEVEL` | `INFO` | Global logging verbosity (`DEBUG`, `INFO`, `WARNING`, etc.) |

**Feature Flags:**

| Variable | Default | Purpose |
| --- | --- | --- |
| `STRIP_THINKING` | `false` | Remove `<think>...</think>` blocks before returning responses |
| `STRIP_JSON_MARKDOWN` | `false` | Convert fenced JSON markdown into raw JSON payloads |
| `DISABLE_THINKING` | `false` | Append `/no_think` hints to prompts for providers that respect it |
| `MAX_LOG_AGE_DAYS` | `7` | Retention window for automatic log cleanup |
| `BLOB_STORAGE_TYPE` | `filesystem` | Storage backend for request/response bodies (`filesystem` or `memory`) |
| `BLOB_STORAGE_PATH` | `~/.smolrouter/blob_storage` | Directory used when `BLOB_STORAGE_TYPE=filesystem`. Set this to `./blob_storage` for checkout-local dev storage if desired |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection used for request/audit logs and exception telemetry |
| `MAX_BLOB_SIZE` | `10485760` | Per-request blob size cap in bytes (10 MiB) |
| `MAX_TOTAL_STORAGE_SIZE` | `1073741824` | Aggregate blob storage cap in bytes (1 GiB) |
| `LOG_DIR` | `/app/logs` | Directory for persisted ERROR log files (rotated via `ERROR_LOG_*`). In non-Docker runs, override to a writable host path such as `./logs` or `/tmp/smolrouter/logs`. |
| `ERROR_LOG_FILE` | `/app/logs/error.log` | Primary ERROR log file |
| `ERROR_LOG_MAX_BYTES` | `10485760` | Max size per ERROR log file before rotation |
| `ERROR_LOG_BACKUP_COUNT` | `5` | Number of rotated ERROR log backups to retain |

**Security & Auth:**

| Variable | Default | Purpose |
| --- | --- | --- |
| `JWT_SECRET` | _(unset)_ | Enables JWT authentication for `/v1/*` endpoints and most `/api/*` endpoints (some API endpoints like `/api/logs`, `/api/stats` are exempt). Must be ≥32 characters with good entropy. Leave unset for unauthenticated access |
| `SMOLROUTER_SECRETS` | _(auto-discovered)_ | Explicit path to `secrets.yaml` for provider API keys. If unset, SmolRouter searches `./secrets.yaml`, the user config dir, then the site config dir |
| `SMOLROUTER_FACADE_KEYS` | _(auto-discovered)_ | Explicit path to `facade_keys.yaml` for router-owned facade keys. If unset, SmolRouter searches `./facade_keys.yaml`, the user config dir, then the site config dir |
| `SMOLROUTER_REQUIRE_SECRETS` | `false` | When `true`, key-bearing providers must load keys from `secrets.yaml`; inline `api_key` / `api_keys` fields are rejected, except OpenAI `api_key: null` BYOK passthrough |
| `WEBUI_SECURITY` | `AUTH_WHEN_PROXIED` | Controls Web UI/dashboard access policy independently of API authentication: `NONE`, `AUTH_WHEN_PROXIED`, or `ALWAYS_AUTH` |
| `ALLOW_UNAUTHENTICATED_ERROR_DASHBOARD` | `false` | Set `true` only on trusted LANs to allow unauthenticated access to `/api/errors/*`. Enables stack traces and exception metadata without auth checks. |
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
    max_requests_per_day: 1500  # Tune this to your quota plan
```

Store provider keys in `secrets.yaml`:
```yaml
google-prod:
  - "YOUR_GOOGLE_API_KEY_1"
  - "YOUR_GOOGLE_API_KEY_2"
```

If you already have inline or file-based keys in an existing routes config, `python -m smolrouter.migrate_secrets` will consolidate them into `secrets.yaml`.

Google GenAI models are discovered live from the provider. Use the provider and system dashboards to inspect configured backends, quotas, and proxy status.

Google GenAI TTS/audio-output is available through the existing OpenAI-style rails on `/v1/responses` and `/v1/chat/completions`; no separate TTS endpoint is required. Supported Google TTS models include `gemini-3.1-flash-tts-preview`, `gemini-2.5-flash-preview-tts`, and `gemini-2.5-pro-preview-tts`. TTS requests are text-input/audio-output only, support `wav` and `pcm`, and return base64 audio data in the OpenAI-compatible JSON response. Native audio is 24 kHz mono 16-bit PCM; `wav` responses wrap that PCM into a WAV container. The current July 2026 fallback snapshot exposes an `8192` token input limit for these preview TTS models, and transcripts longer than a few minutes should still be chunked for more stable output quality.

`/v1/responses` TTS example:
```json
{
  "model": "gemini-3.1-flash-tts-preview",
  "input": "Say cheerfully: Have a wonderful day!",
  "modalities": ["audio"],
  "audio": {"voice": "Kore", "format": "wav"}
}
```

`/v1/chat/completions` TTS example:
```json
{
  "model": "gemini-3.1-flash-tts-preview",
  "messages": [
    {"role": "user", "content": "Say cheerfully: Have a wonderful day!"}
  ],
  "modalities": ["audio"],
  "audio": {"voice": "Kore", "format": "wav"}
}
```

OpenAI-compatible providers can also point at vendor-prefixed API bases such as `/openai/v1` or `/zen/go/v1`. When a vendor does not expose a usable `/v1/models` listing, set `static_models` explicitly in YAML and keep the provider key in `secrets.yaml`:

```yaml
providers:
  - name: "groq-scout"
    type: "openai"
    url: "https://api.groq.com/openai/v1"
    static_models:
      - "meta-llama/llama-4-scout-17b-16e-instruct"
```

```yaml
groq-scout: "YOUR_GROQ_KEY"
```

For OpenAI-compatible providers, a configured provider `api_key` takes precedence over the client's `Authorization` header. Client `Authorization` is only forwarded upstream when that provider is intentionally keyless (`api_key: null` / BYOK passthrough).

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
- Keep provider API keys in `secrets.yaml` (or an external secret manager that renders it), and keep routing/provider topology in `routes.yaml`

**Authentication:**
1. Set `JWT_SECRET` to a 32+ character random value to enable JWT authentication for `/v1/*` endpoints and most `/api/*` endpoints. Weak secrets are rejected during startup
2. Choose a Web UI access policy with `WEBUI_SECURITY` (controls dashboard/UI access independently):
   - `NONE` — no authentication required for Web UI (local development only)
   - `AUTH_WHEN_PROXIED` — require auth when `X-Forwarded-For` headers are present (default)
   - `ALWAYS_AUTH` — always require authentication for Web UI access
3. Important: When `JWT_SECRET` is set, most API requests require a valid JWT token in the `Authorization` header. The following endpoints are exempt: `/api/logs`, `/api/stats`, `/api/inflight`, `/api/performance`. `/api/errors/*` is *not* exempt by default; enable `ALLOW_UNAUTHENTICATED_ERROR_DASHBOARD=true` only on trusted LANs where unauthenticated diagnostics are acceptable. Leave `JWT_SECRET` unset for completely unauthenticated access.
4. Pair JWT auth with rate limiting by defining `RATE_LIMIT_REQUESTS` or `RATE_LIMIT_TOKENS` for each API key

**Logging & Retention:**
- `ENABLE_LOGGING=false` disables request dashboard persistence and Redis request/audit writes; it does **not** disable ERROR file logging, which remains enabled independently.
- Request metadata uses the Redis-backed request/audit path (`REDIS_URL`) for searchable diagnostics and route-level summary.
- Request and response payloads use blob storage (`BLOB_STORAGE_PATH`) for larger bodies outside Redis.
- Stdout/stderr mirror application logs. Defaults are compact at `INFO`; set `LOG_LEVEL=DEBUG` to include detailed routing and provider selection diagnostics such as provider dispatch, proxy selection, and ground-truth verification.
- Persisted ERROR logs are written to `ERROR_LOG_FILE` with rotation (`ERROR_LOG_MAX_BYTES`, `ERROR_LOG_BACKUP_COUNT`), which is recommended for post-restart forensics.
- Set `LOG_DIR` to a writable folder whenever running outside Docker; `/app/logs` is container-oriented and may be unwritable on host shells unless overridden.
- Adjust `MAX_LOG_AGE_DAYS`, `MAX_BLOB_SIZE`, and `MAX_TOTAL_STORAGE_SIZE` to control cost and retention
- Background cleanup jobs run automatically during the FastAPI lifespan events

**Path guidance:**
- Use `-C/--config` or `ROUTES_CONFIG` for deterministic routing config selection.
- Use `SMOLROUTER_SECRETS` for deterministic secrets file selection; otherwise SmolRouter searches `./secrets.yaml`, the user config dir, then the site config dir.
- Use `SMOLROUTER_FACADE_KEYS` for dedicated facade-key storage (`./facade_keys.yaml`, user config dir, then site config dir).
- Provision facade keys with `python -m smolrouter.manage_facade_keys create --project <id> --routes-config <routes.yaml>` (generates `srk-...` secrets; default append behavior avoids clobbering old secrets).
- Use `BLOB_STORAGE_PATH` explicitly in production if you want a location other than the safe default under `~/.smolrouter/blob_storage`.
- For dev checkouts, `BLOB_STORAGE_PATH=./blob_storage` keeps blobs next to the repo as before.
- `docker-compose.yml` mounts `./logs:/app/logs`, so rotated ERROR logs persist across container restarts.

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
