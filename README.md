# SmolRouter

SmolRouter is a lightweight, observable proxy for routing AI model traffic. It keeps your existing OpenAI-compatible clients working while you experiment with different local or hosted model providers.

[![PyPI version](https://badge.fury.io/py/SmolRouter.svg)](https://badge.fury.io/py/SmolRouter)
[![codecov](https://codecov.io/gh/mitchins/smolrouter/branch/main/graph/badge.svg)](https://codecov.io/gh/mitchins/smolrouter)

## TL;DR

- Acts as a drop-in replacement for OpenAI and Ollama endpoints (`http://localhost:1234` by default).
- Routes requests by model name, source host, or custom aliases with automatic failover.
- Logs every request to SQLite with a built-in dashboard for live metrics and request inspection.
- Ships with security controls (JWT auth, reverse-proxy detection) but is intended to run behind your own proxy.

## Quick Start

### Docker deployment (fastest path)

```bash
docker build -t smolrouter .
docker run -d \
  --name smolrouter \
  --restart unless-stopped \
  -p 1234:1234 \
  -e DEFAULT_UPSTREAM="http://localhost:8000" \
  -e MODEL_MAP='{"gpt-3.5-turbo":"llama3-8b"}' \
  -v ./routes.yaml:/app/routes.yaml \
  smolrouter
```

### Python deployment (pip install)

```bash
pip install smolrouter

export DEFAULT_UPSTREAM="http://localhost:8000"
export MODEL_MAP='{"gpt-3.5-turbo":"llama3-8b"}'
smolrouter

# Alternate entrypoint if you need to change the listen port
LISTEN_PORT=8080 python -m smolrouter.app
```

### Point clients at SmolRouter

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="your-api-key"  # forwarded to the upstream server
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # transparently remapped to "llama3-8b"
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Using Google GenAI models

To use Google Gemini models, simply use the model name directly:

```python
response = client.chat.completions.create(
    model="gemini-2.0-flash-exp",  # Automatically routes to Google GenAI
    messages=[{"role": "user", "content": "Hello!"}]
)
```

Any model starting with `gemini-` automatically routes to Google GenAI when configured. See [Google GenAI Setup](docs/GOOGLE_GENAI.md) for details.

<p align="center">
  <img src="images/main-ui.png" alt="SmolRouter Main UI" width="80%">
</p>

## Core capabilities

### Intelligent routing
- Match on model names, regex patterns, or source IP ranges.
- Override model names or upstream targets on a per-rule basis.
- Define reusable aliases that automatically fail over between providers.

### Observability built in
- Live dashboard summarising recent traffic and latency statistics.
- Performance scatter plot to compare token counts against response time.
- Full request and response payload capture (with optional blob storage for large bodies).

### Protocol compatibility and content hygiene
- OpenAI and Ollama API parity, including streaming support.
- Optional model remapping via `MODEL_MAP` JSON.
- Automatic stripping of `<think>` traces or markdown-wrapped JSON before returning responses.

## Security essentials

> **Important:** Run SmolRouter behind a reverse proxy such as nginx, Caddy, or Cloudflare. It binds to `127.0.0.1` by default and is not hardened for direct internet exposure.

- Configure JWT authentication with `WEBUI_SECURITY=ALWAYS_AUTH` for the Web UI or API access when you expose it beyond localhost.
- Keep API keys and upstream hosts in environment variables or secret managers; they are forwarded without modification.
- For shared deployments, set `JWT_SECRET` to a 32+ character random value. Weak secrets are rejected at startup.

## Configuration reference

### High-impact environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEFAULT_UPSTREAM` | `http://localhost:8000` | Upstream endpoint used when no routing rule matches. |
| `LISTEN_HOST` | `127.0.0.1` | Bind address for the FastAPI app. Change to `0.0.0.0` only behind a reverse proxy. |
| `LISTEN_PORT` | `1234` | Port that accepts OpenAI-compatible traffic. |
| `MODEL_MAP` | `{}` | JSON mapping of incoming model names to replacements. |
| `ROUTES_CONFIG` | `routes.yaml` | Path to YAML or JSON smart-routing configuration. |
| `ENABLE_LOGGING` | `true` | Enables request logging and the Web UI. Disable for fully stateless proxying. |
| `REQUEST_TIMEOUT` | `3000.0` | Upstream timeout in seconds (float). |

### Additional controls

| Variable | Default | Purpose |
| --- | --- | --- |
| `DB_PATH` | `requests.db` | SQLite database file for request metadata. |
| `MAX_LOG_AGE_DAYS` | `7` | Automatic cleanup window for historical logs. |
| `STRIP_THINKING` | `true` | Remove `<think>...</think>` blocks from responses. |
| `STRIP_JSON_MARKDOWN` | `false` | Convert fenced JSON markdown to raw JSON payloads. |
| `DISABLE_THINKING` | `false` | Append `/no_think` hint to prompts (for upstream models that respect it). |
| `JWT_SECRET` | _unset_ | Required for JWT-protected deployments; must be ≥32 characters with good entropy. |
| `WEBUI_SECURITY` | `AUTH_WHEN_PROXIED` | Web UI policy: `NONE`, `AUTH_WHEN_PROXIED`, or `ALWAYS_AUTH`. |
| `BLOB_STORAGE_TYPE` | `filesystem` | Storage backend for request/response bodies (`filesystem` or `memory`). |
| `BLOB_STORAGE_PATH` | `blob_storage` | Directory used when `BLOB_STORAGE_TYPE=filesystem`. |
| `MAX_BLOB_SIZE` | `10485760` | Per-request blob size cap in bytes (10 MiB). |
| `MAX_TOTAL_STORAGE_SIZE` | `1073741824` | Aggregate blob storage cap in bytes (1 GiB). |

### Routing and failover (`routes.yaml`)

SmolRouter loads routes at startup from `routes.yaml` (or the path set by `ROUTES_CONFIG`). The file supports server aliases, model aliases, and ordered rule evaluation.

```yaml
# Define servers once and reuse them elsewhere
servers:
  fast-box: "http://192.168.1.100:8000"
  slow-box: "http://192.168.1.101:8000"
  gpu-server: "http://192.168.1.102:8000"

# Aliases expose friendly names to clients with automatic failover
aliases:
  git-commit-model:
    instances:
      - "fast-box/llama3-8b"
      - "slow-box/llama3-3b"

  coding-assistant:
    instances:
      - server: "gpu-server"
        model: "codellama-34b"
      - server: "fast-box"
        model: "llama3-8b"

# Rules run top-to-bottom after alias resolution
routes:
  - match:
      model: "/.*-1.5b/"
    route:
      upstream: "http://gpu-server:8000"

  - match:
      source_host: "10.0.1.100"
    route:
      upstream: "http://dev-server:8000"

  - match:
      model: "gpt-4"
    route:
      upstream: "http://claude-server:8000"
      model: "claude-3-opus"
```

Alias resolution tries each instance in order until one responds successfully; errors are raised only after all candidates fail.

Sample configurations are available in [`routes.yaml.example`](routes.yaml.example) and [`routes.yaml.enhanced`](routes.yaml.enhanced).

### Authentication

```bash
# Generate a random 256-bit secret (recommended)
export JWT_SECRET=$(openssl rand -base64 32)

# Enforce JWT for Web UI and proxied access
export WEBUI_SECURITY="ALWAYS_AUTH"
smolrouter
```

JWT validation rejects secrets that are shorter than 32 characters, blank, or obvious defaults. When `WEBUI_SECURITY=AUTH_WHEN_PROXIED` (the default), the Web UI is automatically disabled if SmolRouter detects proxy headers.

### Blob storage for large payloads

Request and response bodies can be offloaded to disk to keep the SQLite database lean:

```bash
export BLOB_STORAGE_TYPE="filesystem"  # or "memory"
export BLOB_STORAGE_PATH="./blob_storage"
```

Storage limits are enforced per blob (`MAX_BLOB_SIZE`) and across all blobs (`MAX_TOTAL_STORAGE_SIZE`).

## Web UI and monitoring

- **Dashboard (`/`)** — recent traffic, latency summaries, and quick actions.
- **Performance (`/performance`)** — scatter plot of response time versus token count.
- **Request detail (`/request/{id}`)** — inspect full payloads, headers, and routing decisions.

## Development

### Running tests

```bash
pytest
```

### Contributing

Issues and pull requests are welcome. Please discuss major changes before submitting a PR.

## License

SmolRouter is released under the MIT License. See [`LICENSE`](LICENSE) for the full text.
