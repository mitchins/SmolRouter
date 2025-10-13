# Operations and security

This guide covers deployment practices for running SmolRouter in production.

## Deployment posture

- Run behind a trusted reverse proxy (nginx, Caddy, Cloudflare) and publish only HTTPS endpoints.
- Bind to `127.0.0.1` by default; switch to `0.0.0.0` only when the proxy handles TLS and authentication.
- Keep provider API keys and upstream URLs in environment variables or a secret manager; SmolRouter forwards them unchanged.

## Authentication and access control

1. Set `JWT_SECRET` to a 32+ character random value. Weak secrets are rejected during startup.
2. Choose a Web UI policy with `WEBUI_SECURITY`:
   - `NONE` — local experiments only.
   - `AUTH_WHEN_PROXIED` — require auth when `X-Forwarded-For` headers are present.
   - `ALWAYS_AUTH` — safest option when exposed beyond localhost.
3. Pair JWT auth with rate limiting by defining `RATE_LIMIT_REQUESTS` or `RATE_LIMIT_TOKENS` for each API key.

## Logging and retention

- `ENABLE_LOGGING=false` disables the request log and UI for ultra-lightweight proxies.
- Logs and payloads are persisted via SQLite and blob storage. Adjust `MAX_LOG_AGE_DAYS`, `MAX_BLOB_SIZE`, and `MAX_TOTAL_STORAGE_SIZE` to control cost.
- Background cleanup jobs run automatically during the FastAPI lifespan events.

## Testing and validation

- Install development dependencies with `pip install -e .[dev]`.
- Run `pytest` to execute the 160-unit and integration test suite. Coverage spans model remapping, routing policies, logging, quotas, and security guards.
- Use `make check` for linting plus tests if you prefer the Makefile workflow.

## Upgrades

- Review [Release notes for 2.0.0](RELEASE_NOTES_v2.0.0.md) before rolling forward.
- Verify that custom `MODEL_MAP` or `routes.yaml` files still produce the expected remapping behaviour. The v1.x semantics are unchanged, but new provider identifiers (such as Google Gemini) may require updated upstream credentials.
- When running in Docker, rebuild the image to pick up dependency updates including `google-api-core` and the unified Google GenAI SDK.
