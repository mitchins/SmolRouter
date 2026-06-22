# SmolRouter Project Status

## High Priority

- 🟡 Ensure Consistent Information Architecture
    Google GenAI provider has extra features like key status tracking and exhaustion detection. Need to ensure consistent IA across all providers (OpenAI, Anthropic, Ollama). Consider: Should other providers have similar detailed status tracking? Location: Provider interfaces and dashboard consistency.

- 🔵 Add Client Class-based Priority Injection (TODO - unstarted)
    SmolRouter should own request-class policy and inject vLLM's `priority` so background services don't have to. Target deploy is a single dedicated GPU running vLLM with `--scheduling-policy priority` (lower numeric = runs earlier); with no external backpressure, per-request `priority` is the only lever protecting rare human long-context requests from a pile of background/system jobs. Clients declare *intent* (`interactive`/`cli`/`normal`/`background`/`best-effort`) via model alias, `X-SmolRouter-Class` header, or facade-key identity; router maps class → explicit `priority` (0/10/50/100/200) and injects it into the OpenAI-compatible body. Key rules: unknown/undeclared class defaults to **background, never interactive** (anti-starvation); raw client `priority` is stripped/clamped for untrusted clients; elevated classes gated behind trusted identity; pair with class-based `max_tokens`/prompt caps (priority protects scheduling, caps protect KV budget). Plugs into the existing alias-resolution / request-mutation path (`smolrouter/mediator.py`) and reuses facade-key identity + per-key accounting. Full design: `docs/PROPOSAL_request_class_priority.md`.

- 🔴 Fix O(N²) blob-storage write scan (body-archival lag) (TODO - unstarted)
    `FilesystemBlobStorage.store()` calls `_total_size_bytes()` (full `rglob`+`stat` over the whole tree) on every write, twice per request, synchronously on the asyncio event loop — O(N) per write, O(N²) over time. On the production box this caused a multi-hour request/response body archival backlog (bodies show blank in the request detail view until they eventually land). Fix: incremental size counter (or cap-enforcement only in the janitor), offload file I/O via `asyncio.to_thread`, cache created hour-dirs. Paradigm-agnostic; prerequisite for the segment store below. Files: `smolrouter/storage.py` (`store`, `_total_size_bytes`, `_cleanup_for_space`, `_janitor_loop`), `smolrouter/database.py` (`_archive_bodies_after_completion`). Context: `docs/PROPOSAL_segment_blob_storage.md` (Prerequisite section).

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

- 🔵 Separate dashboard auth from request auth headers (TODO - unstarted)
    Keep request API and model-routing auth using existing `Authorization` behavior while decoupling dashboard/session security to avoid header collisions (for example via `X-Auth-Bearer` or cookie/session-based auth). This supports the future state of API keys for LAN quota/monitoring and stronger dashboard auth without competing transport semantics.

- 🔵 Fix duplicate-panel recency windowing (TODO - unstarted)
    The request detail "duplicates" list silently empties once siblings age past the global 1000-most-recent window. `get_recent_duplicate_request_ids` (`smolrouter/redis_backend.py`) intersects the body-hash set with `ZREVRANGE requests:by_time 0..max_scan-1` (max_scan=1000), so under load (thousands of requests/hour) real duplicates drop out of view even though the records and their status are intact. Not data loss — a display artifact. Fix: order/filter duplicates without capping on the global recency window (e.g. score the by-body set members directly, or raise/remove max_scan with a bounded fallback).

- 🔵 Bound the `requests:by_body:*` duplicate index (TODO - unstarted)
    The per-body-hash duplicate sets (`requests:by_body:{hash}`, written via `SADD` in `_check_and_queue_duplicate_request_body`, `smolrouter/redis_backend.py`) have no TTL and no `SREM` on retention, so they grow unbounded and accumulate IDs for requests long pruned from the log. Add a TTL and/or trim members when their request records age out. Related: `duplicate_count`/`is_duplicate` are frozen at create-time snapshots and drift from reality over a record's life — consider computing them live or refreshing.

- 🔵 Add segment/block blob storage backend (TODO - unstarted)
    Add a future blob backend that appends payloads into bounded segment files with offset-based lookup, reducing inode count and making retention pruning whole-segment and O(1). Log-structured (Bitcask-style), not B-tree+free-list: Redis already owns the index (`*_body_key`) and retention is temporal, so we drop whole oldest segments. Single appender + batched fsync + lock-free `pread` reads (deliberately avoids SQLite's single-writer locking). Behind the existing `BlobStorage` ABC; depends on the hot-path scan fix below. Full design: `docs/PROPOSAL_segment_blob_storage.md`.

- 🔵 Offline Batch Job Corralling (TODO - unstarted, future near/mid-term)
    Add a single local-network shopfront for offline batch jobs that fronts Anthropic Message Batches, OpenAI Batch (file/JSONL), Gemini Batch, xAI/Grok Batch, and any provider with a batch tier. Clients submit batches with facade keys only and never see downstream provider keys (keys stay in `secrets.yaml` via `secret_store`, same property the sync path already enforces). Every batch is cost-projected before submission — exact input tokens where available (Anthropic `count_tokens`), local estimate otherwise, plus `$` from a per-model pricing table (tokens-only when price is unknown). Batches over a configurable spend/token threshold are held for manual approval (dashboard + approve/reject API) and nothing is sent downstream until approved — the chokepoint against runaway spend from agentic loops holding loose keys. Reuses routing/alias resolution, facade-key identity, per-key/per-model accounting, and blob storage; adds an `IBatchProvider` adapter per backend (parallel to `IModelProvider`). Phased: Phase 1 Anthropic inline, Phase 2 OpenAI file-based, Phase 3 Gemini + multi-provider split/re-aggregate, Phase 4 hardening. Full design: `docs/PROPOSAL_offline_batch_corralling.md`.

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

- 🟢 Console Logging Cleanup
    Added `LOG_LEVEL` with sane defaults and reduced default INFO-level console noise by moving provider/routing debug details behind `DEBUG`, improving request-level signal quality.

## Architecture Notes

- Google GenAI provider has the most sophisticated quota management
- Other providers could benefit from similar request tracking
- Database schema already supports per-key, per-model tracking
- Current git history is clean of real secrets

## Test Status

- The automated unit and integration suites are active and release-gated through pytest
- Keep this file focused on roadmap/status rather than brittle exact pass-count snapshots
