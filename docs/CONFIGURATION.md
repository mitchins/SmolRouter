# Configuration reference

SmolRouter reads configuration from environment variables and optional routing files. The tables below list the most frequently tuned values. All booleans accept `1`, `true`, or `yes` (case-insensitive).

## Core environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEFAULT_UPSTREAM` | `http://localhost:8000` | Upstream endpoint used when no routing rule matches. |
| `LISTEN_HOST` | `127.0.0.1` | Bind address for the FastAPI app. Switch to `0.0.0.0` only behind a reverse proxy. |
| `LISTEN_PORT` | `1234` | Port that accepts OpenAI-compatible traffic. |
| `MODEL_MAP` | `{}` | JSON mapping of incoming model names to replacements (exact keys or regex). |
| `ROUTES_CONFIG` | `config/routes.yaml` | Path to YAML or JSON smart-routing configuration. |
| `REQUEST_TIMEOUT` | `3000.0` | Upstream timeout in seconds. |
| `ENABLE_LOGGING` | `true` | Toggle request logging, database writes, and the Web UI dashboard. |

## Feature flags and storage

| Variable | Default | Purpose |
| --- | --- | --- |
| `STRIP_THINKING` | `false` | Remove `<think>...</think>` blocks before returning responses. |
| `STRIP_JSON_MARKDOWN` | `false` | Convert fenced JSON markdown into raw JSON payloads. |
| `DISABLE_THINKING` | `false` | Append `/no_think` hints to prompts for providers that respect it. |
| `DB_PATH` | `requests.db` | SQLite database path for request metadata. |
| `MAX_LOG_AGE_DAYS` | `7` | Retention window for automatic log cleanup. |
| `BLOB_STORAGE_TYPE` | `filesystem` | Storage backend for request/response bodies (`filesystem` or `memory`). |
| `BLOB_STORAGE_PATH` | `blob_storage` | Directory used when `BLOB_STORAGE_TYPE=filesystem`. |
| `MAX_BLOB_SIZE` | `10485760` | Per-request blob size cap in bytes (10 MiB). |
| `MAX_TOTAL_STORAGE_SIZE` | `1073741824` | Aggregate blob storage cap in bytes (1 GiB). |

## Security, auth, and rate limiting

| Variable | Default | Purpose |
| --- | --- | --- |
| `JWT_SECRET` | _(unset)_ | Enables JWT authentication for the API and Web UI. Must be ≥32 characters with good entropy. |
| `WEBUI_SECURITY` | `AUTH_WHEN_PROXIED` | Controls UI access policy: `NONE`, `AUTH_WHEN_PROXIED`, or `ALWAYS_AUTH`. |
| `WEBUI_ALLOWED_ORIGINS` | _(unset)_ | Comma-separated list of origins allowed to access the dashboard. |
| `RATE_LIMIT_REQUESTS` | _(unset)_ | Requests per minute per API key. Leave empty to disable rate limiting. |
| `RATE_LIMIT_TOKENS` | _(unset)_ | Token budget per minute per API key (estimated from request payloads). |

## Routing configuration (`routes.yaml`)

SmolRouter loads routes at startup from `ROUTES_CONFIG`. YAML supports reusable servers, aliases, and ordered rules. Example:

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

Hot reload is not currently supported; restart the service after editing the routing file.
