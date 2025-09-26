# Claude Code Status

This file tracks ongoing tasks and status for the SmolRouter project.

## Current Branch: google-genai

## Completed Tasks
- ✅ Fixed Google GenAI quota tracking bug with timezone-aware reset logic
- ✅ Organized root directory structure (config/, data/, images/)
- ✅ Set up proper gitignore rules for runtime data
- ✅ Verified git history contains no real secrets (only test data)

## Pending Tasks

### 1. Ensure Consistent Information Architecture
**Priority: High**
- Google GenAI provider has extra features like key status tracking and exhaustion detection
- Need to ensure consistent IA across all providers (OpenAI, Anthropic, Ollama)
- Consider: Should other providers have similar detailed status tracking?
- Location: Provider interfaces and dashboard consistency

### 2. Token/Request Counting for All API Keys
**Priority: High**
- Google GenAI has comprehensive token/request counting with quota tracking
- OpenAI and Anthropic providers need similar request/token counting against their API keys
- All keys should have consistent metrics regardless of provider type
- Google can keep extra features (exhaustion status, least-used key selection) as provider-specific enhancements
- Files: `openai_provider.py`, `anthropic_provider.py`, `database.py`

### 3. API Key Storage Security Analysis
**Priority: Medium**
- Current: API keys stored in app config, hashed in database for tracking
- Question: Should we store full keys in DB or just hashes for identification?
- Analyze best practices: keys needed for actual API calls but DB tracking could use hashes
- Current approach uses `ApiKeyQuota.hash_api_key()` for DB records
- Consider: Security vs functionality tradeoffs
- Files: `database.py`, provider configs

### 4. Move from Static Pages to Dynamic API
**Priority: Low**
- Current dashboard uses server-side rendered HTML templates
- Move to SPA (Single Page App) with separate API endpoints
- Benefits: Better UX, real-time updates, easier testing
- Templates to migrate: `templates/index.html`, `templates/system.html`, etc.
- New endpoints needed: `/api/dashboard`, `/api/providers`, `/api/stats`

## Architecture Notes
- Google GenAI provider has the most sophisticated quota management
- Other providers could benefit from similar request tracking
- Database schema already supports per-key, per-model tracking
- Current git history is clean of real secrets

## Test Status
- 79 tests passing, 1 expected failure
- All major functionality verified
- Directory reorganization did not break any tests