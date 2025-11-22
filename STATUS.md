# SmolRouter Project Status

## High Priority

- 🟡 Ensure Consistent Information Architecture
    Google GenAI provider has extra features like key status tracking and exhaustion detection. Need to ensure consistent IA across all providers (OpenAI, Anthropic, Ollama). Consider: Should other providers have similar detailed status tracking? Location: Provider interfaces and dashboard consistency.

- 🟡 Token/Request Counting for All API Keys
    Google GenAI has comprehensive token/request counting with quota tracking. OpenAI and Anthropic providers need similar request/token counting against their API keys. All keys should have consistent metrics regardless of provider type. Google can keep extra features (exhaustion status, least-used key selection) as provider-specific enhancements. Files: `openai_provider.py`, `anthropic_provider.py`, `database.py`

## Medium Priority

- 🔵 API Key Storage Security Analysis
    Current: API keys stored in app config, hashed in database for tracking. Question: Should we store full keys in DB or just hashes for identification? Analyze best practices: keys needed for actual API calls but DB tracking could use hashes. Current approach uses `ApiKeyQuota.hash_api_key()` for DB records. Consider: Security vs functionality tradeoffs. Files: `database.py`, provider configs

## Low Priority

- 🔵 Move from Static Pages to Dynamic API
    Current dashboard uses server-side rendered HTML templates. Move to SPA (Single Page App) with separate API endpoints. Benefits: Better UX, real-time updates, easier testing. Templates to migrate: `templates/index.html`, `templates/system.html`, etc. New endpoints needed: `/api/dashboard`, `/api/providers`, `/api/stats`

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

## Architecture Notes

- Google GenAI provider has the most sophisticated quota management
- Other providers could benefit from similar request tracking
- Database schema already supports per-key, per-model tracking
- Current git history is clean of real secrets

## Test Status

- 79 tests passing, 1 expected failure
- All major functionality verified
- Directory reorganization did not break any tests
