# SmolRouter Project Status

## High Priority

- 🟡 Ensure Consistent Information Architecture
    Google GenAI provider has extra features like key status tracking and exhaustion detection. Need to ensure consistent IA across all providers (OpenAI, Anthropic, Ollama). Consider: Should other providers have similar detailed status tracking? Location: Provider interfaces and dashboard consistency.

- 🔵 Add Client Class-based Priority Injection (TODO - unstarted)
    SmolRouter should own request class policy (`interactive` / `background` / `best-effort`), inject upstream `priority`, strip raw client priority unless trusted, and apply class-based token/prompt caps. Default unknown clients to background/normal (not interactive). Route/model alias should resolve to explicit model + injected `priority` only.

- 🟡 Token/Request Counting for All API Keys
    Google GenAI has comprehensive token/request counting with quota tracking. OpenAI and Anthropic providers need similar request/token counting against their API keys. All keys should have consistent metrics regardless of provider type. Google can keep extra features (exhaustion status, least-used key selection) as provider-specific enhancements. Files: `providers.py` (OpenAI path), `anthropic_provider.py`, `database.py`

## Low Priority

- 🔵 Move from Static Pages to Dynamic API
    Current dashboard uses server-side rendered HTML templates. Move to SPA (Single Page App) with separate API endpoints. Benefits: Better UX, real-time updates, easier testing. Templates to migrate: `templates/index.html`, `templates/system.html`, etc. New endpoints needed: `/api/dashboard`, `/api/providers`, `/api/stats`
    Current dashboard uses server-side rendered templates with API-backed data fetching. Most main pages are API-driven but render via Jinja templates (`templates/index.html`, `providers.html`, `performance.html`). Core dashboard data endpoints exist (`/api/dashboard`, `/api/stats`), and provider list data is currently exposed via `/api/upstreams` (not `/api/providers`).

- 🔵 Improve JSON formatting + raw copy for request/body views (TODO - unstarted)
    Request and response payload blocks in web detail views should be consistently pretty-printed with a dedicated raw-copy action, including request and body payloads.

- 🔵 Treat facade API keys as routing/analytics/QoS identity (TODO - unstarted)
    Document and enforce that facade API keys are first-class credentials but are primarily used for request routing, usage attribution, and QoS policy, not for direct provider security semantics.

- 🔵 Clean up BYOK and upstream auth precedence (TODO - unstarted)
    Simplify and standardize auth precedence so BYOK/passthrough behavior is obvious and consistent instead of provider-specific and surprising.

- 🔵 Add segment/block blob storage backend (TODO - unstarted)
    Add a future blob backend that appends payloads into bounded segment files with offset-based lookup, reducing inode count and making retention pruning whole-segment and O(1).

## Completed

- 🟢 Fixed Google GenAI quota tracking bug
    Implemented timezone-aware reset logic for accurate quota management

- 🟢 Organized root directory structure
    Created config/, data/, images/ directories for better organization

- 🟢 Set up proper gitignore rules
    Runtime data properly excluded from version control

- 🟢 Verified git history security
    Confirmed git history contains no real secrets (only test data)

- 🟢 Google GenAI Provider Implementation
    Native Gemini routing with official `google-genai` SDK and HTTPX transport

- 🟢 Provider Abstraction Refresh
    Dependency-injection container and provider factory for adding new backends

- 🟢 Observability Upgrades
    Richer performance plots, quota tracking, and blob storage sharding

- 🟢 Security Hardening
    JWT validation, stricter rate-limiting primitives, and explicit environment toggles

- 🟢 API Key Storage Security Policy
    Enforced that key-bearing providers pull API keys from `secrets.yaml` when `SMOLROUTER_REQUIRE_SECRETS=true`; inline key fields are rejected in strict mode, while `api_key: null` still supports OpenAI BYOK passthrough.

## Architecture Notes

- Google GenAI provider has the most sophisticated quota management
- Other providers could benefit from similar request tracking
- Database schema already supports per-key, per-model tracking
- Current git history is clean of real secrets

## Test Status

- The automated unit and integration suites are active and release-gated through pytest
- Keep this file focused on roadmap/status rather than brittle exact pass-count snapshots
