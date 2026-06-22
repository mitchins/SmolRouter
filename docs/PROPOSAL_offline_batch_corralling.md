# Proposal: Offline Batch Job Corralling

**Status:** Proposed (future / not-now). No code yet — this is the design of record.
**Author:** Mitchell Currie
**Date:** 2026-06-20
**Depends on / extends:** facade-key identity, secret store (`secrets.yaml`), provider abstraction (`IModelProvider`), `ModelMediator`, per-key/per-model accounting, blob storage.

---

## 1. Problem

SmolRouter today is a **synchronous** OpenAI-compatible gateway: it exposes
`/v1/chat/completions`, `/v1/completions`, and `/v1/responses`, holds the
downstream provider keys in `secrets.yaml`, and brokers each call so clients
never touch the real keys.

There is no equivalent for **offline batch jobs**. Today, anyone who wants to run
a large asynchronous job against Anthropic's or OpenAI's batch APIs needs the raw
provider key on the client. That is exactly the surface we don't want exposed:

- **Runaway spend.** A batch can enqueue 100,000 requests in one call. A loose
  key plus an over-eager agentic coding loop can burn a budget with no chokepoint.
- **Key sprawl.** Every machine/agent that runs batches ends up holding a
  long-lived provider key. We already eliminated this for the sync path; batch
  re-introduces it.
- **No projection, no approval.** There is no point at which a human (or a
  policy) sees "this job is projected to cost $X / N tokens" before it is
  committed downstream.

Batch is a logical near-mid-term expansion of the existing facade. This proposal
defines a **single local-network shopfront for offline batch jobs**: clients
submit batches to SmolRouter using facade keys only; SmolRouter holds the network
keys, projects cost, gates anything over threshold for manual approval, then fans
the work out to whichever downstream provider actually serves the model.

---

## 2. What batch APIs look like today (grounding)

All major providers now offer an async batch tier at roughly **50% of standard
price**, with a **completion window** (commonly 24h) and **client-supplied IDs**
to correlate results. The shapes differ in two axes: *how requests go in*
(inline array vs. uploaded file) and *how results come out* (streamed JSONL vs.
downloaded file), plus per-provider status vocabularies.

| Provider | Endpoint(s) | Input shape | Result retrieval | Discount | Notes |
|---|---|---|---|---|---|
| **Anthropic** | `POST /v1/messages/batches` | **Inline** array of `{custom_id, params}` where `params` is a full Messages request (≤100k reqs / 256MB) | Stream JSONL via `batches.results(id)`; results kept **29 days** | 50% | `processing_status`: `in_progress`→`ended`/`canceling`; `request_counts.{processing,succeeded,errored,canceled,expired}`. Most finish < 1h, max 24h. Supports all Messages features incl. prompt caching, tools, vision. |
| **OpenAI** | `POST /v1/batches` (+ Files API) | **File**: upload JSONL (`purpose="batch"`) of `{custom_id, method, url, body}`, reference `input_file_id` | Download `output_file_id` / `error_file_id` (JSONL) | 50% | `endpoint` ∈ `/v1/chat/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/completions`; `completion_window="24h"`; status `validating`→`in_progress`→`finalizing`→`completed`/`failed`/`expired`/`cancelled`. |
| **Google Gemini** | Batch Mode (GenAI SDK `batches`) | **File or inline** GenerateContent requests | Download / inline results keyed by request key | ~50% | Also Vertex AI batch prediction (GCS/BigQuery in+out) for the cloud-native path. |
| **Azure OpenAI** | Global Batch | File (JSONL) | File | 50% | OpenAI shape with Azure deployment names. |
| **AWS Bedrock** | Batch inference | S3 JSONL in | S3 JSONL out | ~50% | Model-arn addressed; IAM/S3 plumbing. |
| **xAI (Grok)** | `POST /v1/batches` (+ Files API) **and** incremental `POST /v1/batches/{id}/requests` | **File or inline**: JSONL of `{custom_id, method, url, body}`, or SDK `batch.add()` (≤200MB / 50k reqs / 25MB per request) | Paginated `GET /v1/batches/{id}/results` (`limit`+`pagination_token`), **available per-item as each request completes** | Reduced (batch tier) | `custom_id`→`batch_request_id`; batch counters `num_{pending,success,error,cancelled}`, item states `pending/succeeded/failed/cancelled`; cancel via `:cancel`. Supports **multimodal** batch endpoints — `/v1/chat/completions`, `/v1/responses`, images, videos. Signed image/video result URLs expire after **1h** — download promptly. |
| **Mistral / Together / Groq** | Batch API | File (JSONL, OpenAI-ish) | File | ~50% | Converging on the OpenAI file shape. |

**The normalizable core** every one of these shares:

1. A collection of **independent** requests, each carrying a caller-chosen
   `custom_id`.
2. Submit → receive an opaque **batch handle**.
3. **Poll** status until terminal.
4. **Fetch results** keyed back to `custom_id` (success / error / expired per item).
5. A **discount** and a **completion window**.

That common core is what the shopfront exposes. The per-provider differences
(inline vs. file, status enums, retention) are what the per-provider **adapters**
absorb.

Two variations the adapter contract must allow for, both surfaced by xAI:

- **Incremental submission.** xAI lets you append requests to a live batch
  (`POST /v1/batches/{id}/requests`), unlike OpenAI's submit-once model. The
  shopfront treats a client batch as immutable for projection/approval purposes
  (you must know the whole job to cost it), but an adapter may *fulfil* it via
  incremental adds downstream.
- **Incremental results.** xAI (paginated, per-item as each completes) and
  Anthropic (streamed JSONL) both yield results before the whole batch is done —
  so `IBatchProvider.results()` is modelled as an async iterator rather than a
  single download, and OpenAI's "download the finished file" is just the
  degenerate case. xAI also batches **multimodal** endpoints (images/videos)
  whose result URLs are signed and expire in ~1h — the adapter must fetch/persist
  those into our blob storage promptly rather than hand back a soon-dead URL.

---

## 3. Design

### 3.1 Principles

- **One front door, many backends.** Clients speak one batch dialect to
  SmolRouter; SmolRouter speaks each provider's native batch dialect downstream.
- **Keys never leave the host.** Same property the sync path already enforces via
  `secret_store` — extended to batch. Clients authenticate with **facade keys**
  only; provider keys stay in `secrets.yaml` and are used only inside the adapter.
- **Project before you commit.** Every batch is costed (tokens, and `$` where a
  price is known) *before* anything is sent downstream.
- **Gate by policy, fail safe.** Over-threshold batches are **held for manual
  approval**; the default for an unpriceable/over-budget batch is *hold*, not
  *send*.
- **Reuse, don't reinvent.** Routing/alias resolution, identity, accounting, and
  blob storage all already exist — batch is a new entry path into them, not a
  parallel stack.

### 3.2 Client-facing surface

A normalized batch envelope. Each item is a request SmolRouter already
understands (a `/v1/chat/completions`, `/v1/responses`, or `/v1/completions`
body), wrapped with a `custom_id`:

```
POST /v1/batches            (facade key in Authorization)
{
  "completion_window": "24h",
  "requests": [
    { "custom_id": "row-1", "endpoint": "/v1/chat/completions",
      "body": { "model": "gpt-4", "messages": [...] } },
    { "custom_id": "row-2", "endpoint": "/v1/chat/completions",
      "body": { "model": "claude-sonnet", "messages": [...] } }
  ]
}
```

For large jobs, also accept a JSONL upload (OpenAI-compatible) so existing OpenAI
batch tooling can point its base URL at SmolRouter unchanged.

Lifecycle endpoints (provider-agnostic, mirror the common core):

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/batches` | Submit; returns a SmolRouter batch id + status (`pending_approval` or `submitted`) and the **projection** |
| `GET` | `/v1/batches/{id}` | Normalized status + request counts + projected vs. actual cost |
| `GET` | `/v1/batches/{id}/results` | Normalized results keyed by `custom_id` (success/error/expired) |
| `POST` | `/v1/batches/{id}/cancel` | Cancel (maps to downstream cancel where supported) |
| `GET` | `/v1/batches` | List (filter by facade key / status) |
| `POST` | `/v1/batches/{id}/approve` · `/reject` | Approval-gate actions (also surfaced in the dashboard) |

Because a single batch can mix models that resolve to **different** downstream
providers, the shopfront may internally split one client batch into several
downstream batches (one per provider) and re-aggregate results under the original
`custom_id`s. The client sees one batch.

### 3.3 Internal architecture

```
client ──facade key──▶ /v1/batches
                          │
                          ▼
                 ┌────────────────────┐
                 │  Batch Coordinator │
                 └────────────────────┘
   1. resolve each request's model → provider     (reuse routing.py / MODEL_MAP / ModelMediator)
   2. project tokens + $                          (token counter + pricing table)
   3. apply spend policy                           → pending_approval | submit
   4. on submit/approve: group by provider, hand to adapter
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
  IBatchProvider     IBatchProvider     IBatchProvider
  (Anthropic)        (OpenAI)           (Gemini)
   inline create      file upload+create  file/inline
   poll → results     poll → download     poll → download
        └─────────────────┴──────────────────┘
                          │ keys from secret_store (never to client)
                          ▼
              persist: batch meta, projected vs actual,
              per-key/per-model accounting, payloads in blob storage
```

New abstraction, parallel to the existing `IModelProvider`:

```python
class IBatchProvider(ABC):
    def supports_model(self, model: ModelInfo) -> bool: ...
    async def project(self, requests: list[BatchItem]) -> Projection: ...   # tokens + $ (best effort)
    async def submit(self, requests: list[BatchItem]) -> ProviderBatchRef: ...
    async def poll(self, ref: ProviderBatchRef) -> BatchStatus: ...         # normalized status
    async def results(self, ref: ProviderBatchRef) -> AsyncIterator[BatchResult]: ...
    async def cancel(self, ref: ProviderBatchRef) -> None: ...
```

The coordinator owns policy, identity, projection rollup, and re-aggregation; the
adapter owns the wire dialect. This mirrors the existing container/mediator/
provider split, so it slots into `SmolRouterContainer` the same way.

### 3.4 Cost / token projection ("$ if available")

Projection runs **before** submission and is stored alongside the batch:

- **Input tokens** — exact where the provider offers a server-side counter
  (Anthropic `count_tokens`, free); otherwise a local estimate. The repo already
  has a rough `estimate_token_count` (`database.py:1339`) as the floor; a
  per-provider tokenizer (e.g. tiktoken for OpenAI models) is the upgrade path.
- **Output tokens** — unknown at submit time. Project an **upper bound** from
  each request's `max_tokens` (and a configurable expected-output ratio for the
  display estimate). Surface both "cap" and "expected".
- **Dollars** — `tokens × per-model rate × 0.5` (batch discount) from a pricing
  table in config. **If a model has no known price, show tokens only, mark `$`
  as `unknown`, and gate on the token threshold instead of the dollar threshold.**
  This is the "$ if available" requirement made explicit.
- The projection object: `{ requests, input_tokens, projected_output_tokens_max,
  projected_cost_usd | null, per_provider_breakdown, priced: bool }`.

After completion, record **actual** usage from each provider's results and store
projected-vs-actual so the pricing table and ratios can be tuned over time
(feeds the existing per-key/per-model accounting).

### 3.5 Approval gating

Thresholds configured globally and per-facade-key (and optionally per-provider),
e.g. in `routes.yaml` / env:

```yaml
batch:
  auto_approve_under_usd: 5.00       # below → submit immediately
  hard_block_over_usd: 500.00        # above → reject outright (optional)
  auto_approve_under_tokens: 2_000_000   # used when price is unknown
  default_when_unpriceable: hold     # hold | reject  (fail safe)
  per_key_overrides:
    ci-runner: { auto_approve_under_usd: 50.00 }
```

Flow:

1. Project. If under all thresholds → `submitted` immediately.
2. If over threshold (or unpriceable and `default_when_unpriceable: hold`) →
   `pending_approval`. **Nothing is sent downstream. No provider key is touched.**
3. A human approves/rejects via the dashboard or the approve/reject endpoints.
   The dashboard lists pending batches with **projected tokens and `$`**,
   per-provider breakdown, submitting facade key, and request count.
4. On approve → submit downstream. On reject → terminal, never sent.

This is the chokepoint that addresses the core motivation: an agentic coding loop
holding a facade key can *queue* work, but it cannot *commit spend* above the
line without a human in the loop, and it never possesses a provider key.

### 3.6 Persistence & accounting

Reuse the existing stores:

- **Batch metadata** (status, projection, provider refs, approver, timestamps) →
  database, with per-key/per-model rows consistent with the current accounting
  (ties into STATUS item *"Token/Request Counting for All API Keys"*).
- **Request/result payloads** → blob storage (the existing sharded backend; the
  proposed segment backend would suit large JSONL well).
- **Mirror provider retention.** Track each provider's result-expiry (Anthropic
  29 days; OpenAI file lifetimes differ) and expose a normalized `expires_at`.

---

## 4. Phasing

Deliberately incremental; each phase is independently useful.

- **Phase 0 — now:** This document. No code. Freeze the client envelope + the
  `IBatchProvider` contract so downstream work and clients can build against a
  stable spec.
- **Phase 1 — Anthropic, inline:** Single-provider proof. Anthropic's inline
  create maps most closely to our existing request shape; `count_tokens` gives
  exact input projection. Implement coordinator, projection, approval gate, and
  the dashboard pending-batches view. End-to-end: submit → project → gate →
  approve → submit → poll → normalized results.
- **Phase 2 — OpenAI, file-based:** Add the Files-upload + create + poll +
  download adapter and the JSONL ingress so OpenAI batch tooling works unchanged.
  Normalize OpenAI's status enum and per-item errors into the common shape.
- **Phase 3 — Gemini + xAI + generalize:** Add Gemini batch and xAI/Grok batch
  (xAI reuses most of the Phase 2 OpenAI file path; new work is its incremental
  add/results pagination and multimodal result-URL capture). Harden the
  `IBatchProvider` registry and the multi-provider split/re-aggregate path
  (one client batch → N downstream batches).
- **Phase 4 — hardening:** cancellation across providers, partial-failure &
  expired-item semantics, idempotency keys on submit, per-key rolling budgets,
  scheduled submission windows, projected-vs-actual tuning, retention mirroring.

---

## 5. Non-goals (for now)

- Not a job scheduler / cron (out of scope; this is submit-and-poll).
- Not real-time/streaming — that's the existing sync path.
- Not embeddings-specific tooling beyond passing them through where a provider's
  batch endpoint supports the endpoint type.
- No new auth model — batch uses the **existing facade-key identity** for
  attribution and policy; provider security semantics stay server-side.

---

## 6. Open questions

1. **Envelope dialect:** lead with the OpenAI `/v1/batches` + JSONL shape (max
   tooling compatibility) or the Anthropic inline shape (closest to our internals)?
   Current lean: expose **OpenAI-compatible** ingress, normalize internally.
2. **Mixed-provider batches:** silently split, or require single-provider batches
   in Phase 1–2 and lift the restriction in Phase 3?
3. **Output-token projection:** is `max_tokens` upper bound + ratio good enough,
   or do we want a learned per-model ratio from historical actuals from day one?
4. **Approval transport:** dashboard-only first, or ship the approve/reject API
   and a webhook/notification at the same time?
5. **Pricing source of truth:** hand-maintained table vs. pulling from a provider
   pricing feed; how to handle unpriced/new models beyond "tokens only".
```
