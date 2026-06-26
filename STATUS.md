# SmolRouter Project Status

## High Priority

- 🟡 Ensure Consistent Information Architecture
    Google GenAI provider has extra features like key status tracking and exhaustion detection. Need to ensure consistent IA across all providers (OpenAI, Anthropic, Ollama). Consider: Should other providers have similar detailed status tracking? Location: Provider interfaces and dashboard consistency.

- 🔴 Facade-key identity and attribution (Phase A complete, Phase B first milestone complete)
    Treat facade API keys as first-class SmolRouter identity for request attribution, soft quota accounting, routing/QoS defaults, and future batch approval policy. Phase A groundwork is in place: facade-key config/secret registry, validation, and container plumbing. Phase B now accepts local facade keys on the OpenAI write path (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`) via `Authorization: Bearer srk-...`, keeps `X-SmolRouter-Key` as a transitional alias, exempts those write routes from JWT-in-`Authorization` collision, threads identity through `ClientContext`, persists a generic request subject (`identity_kind`, `identity_subject_id`, `identity_display_name`) on request creation, surfaces project identity in dashboard/client/request-detail views, and adds read-only `/projects` inventory for configured facade-key identities plus `/projects/{id}` drilldown backed by a Redis identity-recency index. Facade-key provisioning now has an operator CLI (`python -m smolrouter.manage_facade_keys create`) and dedicated secret boundary (`facade_keys.yaml` / `SMOLROUTER_FACADE_KEYS`) as the single runtime source of truth for facade-key secrets. Future deliberate gaps: no quota ledger yet, no hard enforcement, no Ollama compatibility-path parity, no `/v1/models` or `/api/tags` identity resolution yet, and no broader dashboard/browser auth redesign or provider-BYOK transport cleanup yet. Historical traffic-only identities are reachable via drilldown when known, but are not listed as inventory. Full design: `docs/PROPOSAL_facade_key_identity_accounting.md`.

- 🟡 Facade-key request/token accounting and soft quotas
    Build on facade-key identity first, then add approximate request/token accounting and soft quota state for local project/use-case keys. Keep semantics honest: soft/best-effort, streaming-aware, and fail-open on accounting gaps. This should precede broad provider-key accounting convergence because the main product question is "which local project/use case consumed the budget?" not just "which downstream secret was used?". Full design: `docs/PROPOSAL_facade_key_identity_accounting.md`.

- 🔵 Add Client Class-based Priority Injection (TODO - unstarted)
    SmolRouter should own request-class policy and inject vLLM's `priority` so background services don't have to. Target deploy is a single dedicated GPU running vLLM with `--scheduling-policy priority` (lower numeric = runs earlier); with no external backpressure, per-request `priority` is the only lever protecting rare human long-context requests from a pile of background/system jobs. Clients declare *intent* (`interactive`/`cli`/`normal`/`background`/`best-effort`) via model alias, `X-SmolRouter-Class` header, or facade-key identity; router maps class → explicit `priority` (0/10/50/100/200) and injects it into the OpenAI-compatible body. Key rules: unknown/undeclared class defaults to **background, never interactive** (anti-starvation); raw client `priority` is stripped/clamped for untrusted clients; elevated classes gated behind trusted identity; pair with class-based `max_tokens`/prompt caps (priority protects scheduling, caps protect KV budget). Plugs into the existing alias-resolution / request-mutation path (`smolrouter/mediator.py`) and reuses facade-key identity + per-key accounting. Full design: `docs/PROPOSAL_request_class_priority.md`.

- 🟡 Reassess body-archival lag after blob-storage hot-path fix
    The original O(N²) write-path issue has been addressed: `FilesystemBlobStorage` now uses a persisted usage counter instead of scanning the whole tree on every write, and request/response body archival stores are already offloaded via `asyncio.to_thread`. If operators still observe blank request/response bodies that appear later, the remaining work is to measure the current bottleneck accurately before changing storage again (for example: async eventual-consistency on detail views, write amplification from per-request files, or janitor contention). This is still relevant context for any future segment-store work, but the backlog item should no longer assume the old synchronous full-tree scan is the active bug. Files: `smolrouter/storage.py`, `smolrouter/database.py`. Context: `docs/PROPOSAL_segment_blob_storage.md`.

- 🟡 Provider-key accounting convergence
    Google GenAI has comprehensive per-key token/request counting with quota tracking, and the Redis quota primitives already exist for reuse. The remaining gap is provider integration consistency: Anthropic currently reports model-level stats rather than true per-key accounting, and OpenAI-compatible flows still need explicit key-level usage tracking across configured-key and BYOK/passthrough modes. This should follow facade-key identity/accounting so downstream-secret observability complements, rather than substitutes for, project/use-case accounting. Files: `providers.py` (OpenAI path), `anthropic_provider.py`, `database.py`, `redis_backend.py`

- 🔴 Per-project protective rate limit (TODO - unstarted, planning only)
    Add a defensive per-identity (facade-key/project) request-rate cap whose job is *blast-radius containment*, not accounting: stop a buggy client — an agent loop missing a `sleep`, a retry storm, a fork bomb of near-duplicate calls — from spamming upstreams and burning quota/spend before a human notices. Distinct from the soft-quota item (which measures consumption honestly and fails open): this is a hard, fail-safe ceiling that sheds load (429 with `Retry-After`) when a single project exceeds a sane req/sec or req/min budget. Open design questions for planning: where it sits relative to identity resolution and the existing `rate_limiter` (per-identity token bucket vs. fixed window); sensible default ceiling + per-project override in facade-key config; whether it counts pre- or post-routing; interaction with the request-class priority work (background classes should be throttled first); and how breaches surface on the dashboard. Depends on facade-key identity (done); complements per-key accounting/soft quotas and offline batch corralling as the third spend-safety chokepoint.

- 🔴 Provider readiness by key availability (TODO - unstarted)
    Do not include provider instances in routing until required provider keys are present/valid; emit `UNAVAILABLE_NO_KEY` status so operators can distinguish config/dependency outages from infra/network issues.

- 🟠 Health check failure taxonomy (TODO - unstarted)
    Split provider health outcomes by cause (`UNAVAILABLE_AUTH`, `UNAVAILABLE_NETWORK`, `UNAVAILABLE_SERVICE`) and keep the dashboard/API reasoned state machine on root-cause, not only boolean up/down.

## Low Priority

- 🔵 Move from Static Pages to Dynamic API
    Current dashboard uses server-side rendered HTML templates. Move to SPA (Single Page App) with separate API endpoints. Benefits: Better UX, real-time updates, easier testing. Templates to migrate: `templates/index.html`, `templates/system.html`, etc. New endpoints needed: `/api/dashboard`, `/api/providers`, `/api/stats`
    Current dashboard uses server-side rendered templates with API-backed data fetching. Most main pages are API-driven but render via Jinja templates (`templates/index.html`, `providers.html`, `performance.html`). Core dashboard data endpoints exist (`/api/dashboard`, `/api/stats`), and provider list data is currently exposed via `/api/upstreams` (not `/api/providers`).

- 🔵 Improve JSON formatting + raw copy for request/body views (TODO - unstarted)
    Request and response payload blocks in web detail views should be consistently pretty-printed with a dedicated raw-copy action, including request and body payloads.

- 🔵 Clean up BYOK and upstream auth precedence (TODO - unstarted)
    Simplify and standardize auth precedence so BYOK/passthrough behavior is obvious and consistent instead of provider-specific and surprising.

- 🔵 Parse google_genai non-text response parts
    Explicitly parse/record non-text `candidates.content.parts` (e.g., `thought_signature`) so we stop implicit text-only concatenation and warning spam while preserving backward-compatible text output.

- 🔵 Enforce request span start/completion parity
    Guarantee every request path logs matching `Request started` and `Request completed` events (including early validation/adapter/timeout fail paths) with completion cause.

- 🔵 Add non-invasive latency attribution telemetry
    Track per-provider/per-model/request-source p50/p90/p95/p99 and max latency without altering routing, timeout, retry, or fallback behavior.

- 🔵 Tune dashboard refresh/reconnect noise
    The main dashboard is already primarily WebSocket-driven, with a small debounced `/api/dashboard` refresh after request events. The remaining work is narrower than "polling noise": reduce unnecessary refreshes during reconnect/error conditions, coalesce bursts more aggressively if needed, and verify that dashboard control-plane load stays low under sustained request churn without changing any request-serving pathways.

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
