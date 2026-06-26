# Proposal: Facade-Key Identity and Soft Quota Accounting

**Status:** In progress (Phase A complete, Phase B first milestone complete; accounting still pending)
**Author:** Mitchell Currie
**Date:** 2026-06-25
**Depends on / extends:** request logging, Redis-backed counters, `ClientContext`, provider abstraction, `docs/PROPOSAL_request_class_priority.md`, `docs/PROPOSAL_offline_batch_corralling.md`

---

## 1. Goal

### Current implementation status

As of 2026-06-25, the repo now has:

- Phase A groundwork complete:
  - facade-key registry and validation,
  - facade-key secrets loading,
  - request-identity primitives,
  - container wiring.
- Phase B first milestone complete:
  - local facade-key resolution on the OpenAI write path via `Authorization: Bearer srk-...`,
  - `X-SmolRouter-Key` retained as a transitional alias on that write path,
  - JWT request-auth collision removed for those OpenAI write routes,
  - `ClientContext.identity` propagation through `app.py -> container -> mediator`,
  - durable per-request subject fields (`identity_kind`, `identity_subject_id`, `identity_display_name`) in request logs,
  - identity-aware dashboard/client/request-detail surfaces,
  - read-only `/projects` inventory for configured facade-key identities,
  - `/projects/{id}` drilldown and API backed by a per-identity recency index,
  - retention cleanup for that identity index.

Deliberate non-goals of the current implementation:

- no quota ledger writes yet,
- no hard reject / admission control,
- no provisioning or mutation UI for facade-key projects yet,
- no Ollama compatibility-path parity yet,
- no `/v1/models` / `/api/tags` identity resolution yet,
- no broader dashboard/browser auth redesign yet,
- no provider-BYOK transport cleanup yet,
- no hot-reload story for rotated facade-key secrets yet,
- configured project inventory is not a complete historical identity catalog; traffic-only historical identities are reachable by drilldown when known but are not listed as inventory.

Make **facade API keys** a first-class SmolRouter identity for:

- request attribution,
- soft request/token quotas,
- routing/QoS defaults,
- dashboard and API reporting,
- future batch approval and budget policy.

Important boundary: facade-key enforcement is aimed at protecting
SmolRouter-managed wrapped credentials and policy-controlled access, where the
router is standing in front of embedded downstream secrets. It is not merely a
claim that any caller who can already bring their own upstream/BYOK credential
must be blocked from hitting an inference endpoint unless they also present a
local facade key.

The immediate next step is **not** "billing-grade quotas" and **not** "full auth redesign".
The immediate next step is:

1. define a stable facade-key identity model,
2. attribute every request to that identity,
3. persist request/token usage against that identity,
4. expose soft quota state based on that ledger.

This is the smallest slice that solves the stated use case:

> "Use local API keys to identify projects and use cases, and get approximate usage / quota tracking."

It also creates the policy boundary needed by the broader epic without painting the router into a corner.

---

## 2. Problem

Today SmolRouter has several useful pieces, but they are not joined into a coherent local-identity story:

- The router already computes per-request token counts (actual when the upstream returns `usage`, estimated otherwise).
- Request logs already persist per-request metrics and provider metadata.
- Redis already has durable counter primitives for Google GenAI's provider-key quota tracking.
- Existing proposals already assume "facade-key identity" exists, but the implementation is still implicit.

What is missing is a **canonical local identity layer** for requests that is independent of downstream provider keys.

That gap matters because the desired questions are local questions:

- "How many tokens did project A use today?"
- "Which use case is driving most traffic?"
- "Should this project default to background QoS?"
- "Should this key be allowed to submit big batch jobs?"

Those are **not** provider-key questions. They are router-identity questions.

If the design jumps straight to provider-key quota tracking everywhere, the result is a mismatch:

- it tells the operator which upstream key was used,
- but it does not cleanly answer which local project or use case consumed the budget,
- and it gets muddy quickly in BYOK/passthrough flows.

So the next step should center the **facade key as the subject of policy and accounting**.

---

## 3. Current Constraints and Footguns

The design has to respect the code as it exists today.

### 3.1 `Authorization` is already overloaded

Right now:

- `Authorization` may carry a JWT for SmolRouter auth (`smolrouter/auth.py`),
- `Authorization` may be forwarded upstream for keyless OpenAI-compatible BYOK providers (`smolrouter/providers.py`),
- Anthropic also falls back to the client `Authorization` header for passthrough (`smolrouter/anthropic_provider.py`).

That means:

> Reusing `Authorization: Bearer ...` for facade keys as the *next step* is a footgun.

It would immediately collide with:

- JWT auth,
- upstream BYOK passthrough,
- and future dashboard/session separation work.

### 3.2 The existing quota backend is specialized to provider keys

`RedisApiKeyQuota` is keyed around:

- provider id,
- hashed API key,
- model name,
- Pacific-day reset semantics inherited from Google GenAI.

That is useful, but it is not a clean fit for facade-key identity:

- facade keys should be keyed by **logical project identity**, not secret hash,
- not all facade quotas want Pacific resets,
- and future usage subjects are broader than provider keys.

> The next step should not hard-wire facade-key accounting into the current provider-key record shape.

### 3.3 Token counts are approximate in some paths

SmolRouter already does the pragmatic thing:

- use upstream `usage` when available,
- estimate otherwise from request/response bodies.

That is good enough for:

- soft quotas,
- project accounting,
- alerting,
- QoS heuristics.

It is **not** billing-grade metering.

The design must say this explicitly so later enforcement semantics stay honest.

### 3.4 Existing proposals already depend on facade-key identity

Two existing proposals assume this layer exists:

- request-class priority injection,
- offline batch corralling.

If the next step invents a narrow, auth-only facade-key mechanism that is not reusable by routing, QoS, and batch policy, the epic gets harder, not easier.

---

## 4. Design Summary

Introduce a **router-local request identity layer** centered on facade keys.

Long-term, the canonical request-plane credential for `/v1/*` should be:

```text
Authorization: Bearer <facade-key>
```

That is the only transport that cleanly preserves compatibility with:

- stock OpenAI clients,
- `/v1/batches`,
- existing "facade key as the local API credential" mental model.

However, phase 1 cannot pretend the current codebase is already there, because today
`Authorization` is still shared with JWT auth and some BYOK/passthrough paths.

So the design is:

- **End state:** facade key in `Authorization`
- **Phase-1 transitional alias:** `X-SmolRouter-Key`
- **Future BYOK transport:** dedicated upstream-auth header, not raw `Authorization`
- **Future dashboard/session auth:** out-of-band from request-plane `Authorization`

That makes `X-SmolRouter-Key` explicitly transitional rather than accidentally permanent.

The next-step implementation slice is:

1. **Facade key registry**
   - facade key metadata in `routes.yaml`
   - facade key secrets in `secrets.yaml`
2. **Facade-key request transport**
   - accept local facade keys on OpenAI write routes via `Authorization: Bearer <facade-key>`
   - keep `X-SmolRouter-Key` only as a transitional alias
   - do not broaden this slice to browser/dashboard auth redesign
3. **Request identity resolution**
   - resolve the facade key early in request handling
   - attach identity to `ClientContext`, logs, and response headers
4. **Facade usage ledger**
   - new Redis namespace for facade-key counters
   - keyed by logical facade key id, not raw key or secret hash
5. **Soft quota state**
   - observe and report over-budget state
   - optionally warn
   - defer hard reject / admission control to a later phase
6. **Identity indexes**
   - cheap drilldown by facade key
   - no full-log scans for per-project views

The full epic then builds on the same identity:

- class/QoS defaults,
- batch approval thresholds,
- per-project dashboards,
- optional preflight quota rejection,
- eventual unification with provider-key accounting under a broader subject model.

---

## 5. Next-Step Scope

### 5.1 Facade key registry

Add a top-level `facade_keys` section to `routes.yaml` for metadata and policy:

```yaml
facade_keys:
  project-foo:
    enabled: true
    display_name: "Project Foo"
    tags: ["team:ml", "env:dev", "usecase:agent"]
    default_class: "normal"
    quota:
      daily_requests_soft: 1000
      daily_tokens_soft: 2000000
      action: "observe"     # observe | warn
      warn_threshold: 0.8
```

Store the actual secret values in `secrets.yaml`:

```yaml
facade_keys:
  project-foo: "srk_live_project_foo"
  project-bar: "srk_live_project_bar"
```

This requires an **additive extension** to the current secret-store schema.
Today `secret_store.py` only understands the flat provider-key shape
(`provider_name: key-or-list`). Phase 1 should extend it backward-compatibly so:

- existing top-level provider entries continue to work unchanged,
- an optional top-level `facade_keys:` mapping is recognized separately,
- facade key lookup uses a dedicated helper instead of overloading provider-key lookup.

Rationale:

- metadata belongs with routing/policy config,
- secrets belong with secrets,
- logical key ids (`project-foo`) stay stable even if the secret rotates.

That stability matters because usage should accumulate by **project identity**, not be fragmented across rotated secrets.

Facade keys must also have **one authoritative runtime registry**.

Phase 1 should not let `app.py` and `SmolRouterContainer` parse facade-key config independently.
Instead:

- build a single `FacadeKeyRegistry` from:
  - resolved routes config,
  - resolved secrets config,
- validate it at startup,
- expose it to request handlers through the active container/runtime context.

Required startup validation:

- duplicate logical ids,
- duplicate presented secrets,
- unknown secret entries,
- disabled ids that still have live secrets,
- multiple ids claiming the same secret.

Required lifecycle semantics:

- one loader path for registry construction,
- explicit cache invalidation on config reload,
- no hidden module-global facade-key map separate from the container/runtime registry.

### 5.2 Dedicated transport header

For the next step, the compatibility facade-key transport should be:

```text
X-SmolRouter-Key: srk_live_project_foo
```

This is the **phase-1 compatibility transport**, not the long-term canonical one.

Phase-1 resolver behavior should be:

1. accept `Authorization: Bearer <facade-key>` only in request-auth modes where `/v1/*` is **not** using JWT-in-`Authorization`,
2. accept `X-SmolRouter-Key` as a transitional alias,
3. continue to accept JWT in `Authorization` where JWT request auth still owns that header,
4. treat upstream BYOK as a separate transport concern, not as caller identity.

Phase-1 request-auth modes must therefore be explicit:

- `jwt_request_auth`
  - `/v1/*` `Authorization` is reserved for JWT
  - facade key may be supplied only via `X-SmolRouter-Key`
- `facade_request_auth`
  - `/v1/*` `Authorization` is the facade key transport
  - JWT request auth is not using `Authorization` on those routes
- `transition_alias`
  - JWT request auth is off for `/v1/*`
  - facade key may arrive via `Authorization` or `X-SmolRouter-Key`

Facade-key requirement mode is a separate axis:

- `off`
- `optional`
- `required`

Not every combination is legal. The legal matrix for phase 1 is:

| Request-auth mode | Facade-key requirement | Valid? | Missing key | Invalid key | Both JWT + facade key present |
|---|---|---:|---|---|---|
| `jwt_request_auth` | `off` | yes | no facade subject; JWT principal only | n/a | facade key ignored or rejected by config policy; default `ignore` |
| `jwt_request_auth` | `optional` | yes | JWT principal, `anonymous` accounting subject | reject facade key, keep JWT auth result | JWT principal + facade accounting subject |
| `jwt_request_auth` | `required` | yes | reject request | reject request | JWT principal + required facade accounting subject |
| `facade_request_auth` | `off` | no | startup reject | startup reject | n/a |
| `facade_request_auth` | `optional` | no | startup reject | startup reject | n/a |
| `facade_request_auth` | `required` | yes | reject request | reject request | JWT not used on `/v1/*` in this mode |
| `transition_alias` | `off` | no | startup reject | startup reject | n/a |
| `transition_alias` | `optional` | yes | `anonymous` accounting subject | reject facade key if present | if `Authorization` and `X-SmolRouter-Key` both present they must resolve to the same facade key or reject |
| `transition_alias` | `required` | yes | reject request | reject request | if `Authorization` and `X-SmolRouter-Key` both present they must resolve to the same facade key or reject |

Conventions:

- `reject request` here means a caller-visible 4xx, not a startup error.
- `startup reject` means the mode combination is invalid configuration and the service should fail validation before serving.
- "invalid key" means a presented facade key that does not resolve to any enabled logical facade-key id.

That requires an explicit precedence matrix:

| Concern | Long-term transport | Phase-1 allowance |
|---|---|---|
| Request-plane facade identity | `Authorization` | `Authorization` or `X-SmolRouter-Key` |
| Dashboard/session auth | cookie or `X-Auth-Bearer` | existing Web UI policy until separated |
| Upstream BYOK / passthrough | dedicated upstream-auth header | legacy raw `Authorization` passthrough only in deployments not using `Authorization` for facade keys |

Transition rules:

- if facade-key-in-`Authorization` mode is enabled for `/v1/*`, legacy raw `Authorization` passthrough is disabled for those requests;
- BYOK must then move to the dedicated upstream-auth transport;
- `X-SmolRouter-Key` exists only to bridge deployments that have not completed that split yet.
- if JWT request auth is enabled for `/v1/*`, phase 1 does **not** attempt to multiplex facade keys into the same `Authorization` header; use `X-SmolRouter-Key` instead.

This is the hard rule:

> SmolRouter must distinguish **authentication principal**, **accounting subject**, and **upstream credential** explicitly. They are not the same thing.

The proposal therefore updates the earlier assumption that facade keys only need a
custom header. They do not. The custom header is only the transition bridge.

### 5.3 Request identity resolution

Introduce a resolver near the top of request handling:

```text
HTTP request
  ├─ determine request-auth mode
  ├─ authenticate principal using the transport owned by that mode
  ├─ resolve facade key from Authorization or X-SmolRouter-Key (per mode rules)
  ├─ build RequestIdentity
  └─ pass identity into ClientContext, logging, and policy
```

New conceptual model:

```python
@dataclass
class RequestIdentity:
    kind: str                 # "facade_key" | "anonymous" | later: "jwt_user"
    subject_id: str           # stable logical id, e.g. "project-foo"
    display_name: str | None
    tags: list[str]
    default_class: str | None
    quota_policy: dict[str, Any]
    token_accounting_state: str | None   # "actual" | "estimated" | "missing"
```

`ClientContext` should gain an identity field instead of forcing every future feature to rediscover the caller from raw headers.

This becomes the shared substrate for:

- access control,
- routing,
- QoS,
- accounting,
- batch policy.

Critical implementation note:

> The app boundary must resolve identity once and then pass it explicitly through the request path.

Today the routed request path rebuilds `ClientContext` inside the mediator from
`source_ip` plus filtered headers. That would silently drop facade identity and
create split-brain behavior between logging and routing policy.

Phase 1 should therefore:

- resolve the caller before `_build_openai_forward_headers()`,
- store it on `request.state`,
- pass either a full `ClientContext` or a `ResolvedCaller` object through:
  - `container.route_request()`
  - `container.route_streaming_request()`
  - `mediator.route_request()`
- reuse the same resolved identity for:
  - model listing,
  - request routing,
  - logging,
  - future batch submission.

Do **not** make later layers rediscover the facade key from raw headers.

### 5.4 Principal semantics

The design must define both principal and subject semantics up front.

- **Authentication principal**
  - JWT user/session when JWT request auth is enabled
  - otherwise facade key when facade-key auth is required
  - otherwise anonymous
- **Accounting subject**
  - facade key if present
  - otherwise `anonymous` / `unattributed` bucket when facade keys are optional
- **Upstream credential**
  - downstream provider key chosen by SmolRouter or explicitly supplied BYOK credential

Rules:

1. If both JWT and facade key are present:
   - JWT authenticates the request
   - facade key is the accounting/policy subject
2. JWT claims may later constrain which facade keys are allowed, but that is an additive policy layer.
3. BYOK does **not** suppress facade accounting.
   - The local caller should still be billed to its project/use-case identity even when it supplies the upstream credential.
4. Facade-key requirement is mode-dependent:
   - `off`: no facade-key resolution
   - `optional`: missing key falls into `anonymous`
   - `required`: request is rejected if no valid facade key is supplied

This separation is what keeps the design usable for both local project accounting and later access-control policy.

### 5.5 Request logging and attribution

Extend request logs with facade identity fields, separate from downstream provider metadata.

Suggested fields:

- `identity_kind`
- `identity_subject_id`
- optional `identity_display_name`

Do **not** log the raw facade key.

Do **not** reuse `api_key_suffix` for facade identity.

`api_key_suffix` already means downstream/provider key observation. Mixing the two would destroy operator clarity.

The request detail view should eventually answer both questions separately:

- "Which local project sent this request?"
- "Which downstream provider/key handled it?"

### 5.6 Facade usage ledger

Add a new Redis-backed ledger for facade-key usage rather than forcing facade records into `RedisApiKeyQuota`.

Suggested namespace shape:

```text
usage:facade_key:{facade_key_id}:{provider_id}:{resolved_model}
usage:facade_key:{facade_key_id}:__all__:__all__
usage:facade_key:{facade_key_id}:{provider_id}:__all__
```

Each record stores:

- `requests_today`
- `actual_tokens_today`
- `estimated_tokens_today`
- `updated_at`
- `last_reset`
- optional `over_soft_limit`

Why a new namespace:

- facade-key counters should key on logical identity, not hashed secret,
- provider id and resolved model must stay distinct for fallback, aliasing, and same-name models across providers,
- requested/original model alias should remain request metadata, not the primary counter key,
- reset semantics may evolve independently from provider-key quotas,
- it avoids risky migration pressure on the Google-specific quota backend,
- it leaves room for a later generalized subject ledger.

The usage API should also maintain cheap request indexes for drilldown, for example:

```text
requests:by_facade_key:{facade_key_id}
```

Use a recency-ordered structure (zset or equivalent), not a scan.

### 5.7 Soft quota behavior

For the next step, quota behavior should be **observational** by default:

- track usage,
- compute over-soft-limit status,
- optionally emit warning headers,
- surface the state in dashboard/API views.

Suggested response headers:

```text
X-SmolRouter-Facade-Key-Id: project-foo
X-SmolRouter-Quota-State: ok | warning | over_soft_limit | degraded
X-SmolRouter-Accounting-State: complete | partial | degraded
```

Hard rejection is deliberately deferred.

Reason:

- token counts may be partly estimated,
- quotas are project/accounting guidance first,
- admission control needs a preflight projection model and explicit policy semantics.

That is a different step.

Quota policy needs explicit window and failure semantics even in phase 1:

- default window type: `calendar_day`
- default timezone: configurable IANA timezone, default `UTC`
- per-key quota thresholds may vary later without data migration because policy is not baked into the ledger key
- facade-ledger writes are **fail-open**
  - request serving continues
  - gaps are logged and surfaced as accounting-health warnings
  - this is not allowed to inherit the provider-quota backend's fail-closed posture

Quota state must also be honest about accounting completeness:

- `ok`, `warning`, and `over_soft_limit` are only valid when accounting completeness is above the configured confidence threshold for the relevant rollup
- `degraded` means:
  - recent ledger write failures exist, or
  - token-accounting completeness for the subject/window is below threshold, or
  - the system cannot determine quota state confidently from available data

Phase 1 should prefer `degraded` over a misleading green `ok`.

The completeness rules must be deterministic:

- `request` accounting is considered complete unless there are recorded ledger-write gaps for request counts in the relevant window
- `token` accounting completeness is measured as:

```text
token_coverage_ratio =
  token_covered_requests / total_counted_requests
```

where:

- `token_covered_requests` = requests whose `token_accounting_state` is `actual` or `estimated`
- `total_counted_requests` = requests counted for the subject/window

Missing-token requests (`token_accounting_state = missing`) lower token coverage.

Default policy:

- `minimum_token_coverage_ratio = 0.95`
- configurable globally, overridable later per facade key if needed

Deterministic output rules:

1. If request-count ledger gaps exist, `X-SmolRouter-Accounting-State = degraded` and `X-SmolRouter-Quota-State = degraded`.
2. If request counts are complete but `token_coverage_ratio < minimum_token_coverage_ratio`, then:
   - `X-SmolRouter-Accounting-State = partial`
   - token-based quota evaluation is considered not trustworthy
3. If token-based quota evaluation is not trustworthy and a token soft quota is configured for the surfaced rollup, `X-SmolRouter-Quota-State = degraded`.
4. If only request soft quotas are configured for the surfaced rollup and request counts are complete, request quota state may still be `ok | warning | over_soft_limit` even when token accounting is partial.
5. If both request and token soft quotas are configured, final quota state precedence is:

```text
degraded > over_soft_limit > warning > ok
```

That gives one scalar quota header deterministic semantics while keeping accounting completeness explicit in a second header.

### 5.8 Honest token-accounting semantics

Phase 1 must not claim exact token attribution for every request.

Each request record and ledger update should carry:

```text
token_accounting_state = actual | estimated | missing
```

Rules:

- non-streaming requests:
  - use upstream `usage` when present -> `actual`
  - otherwise use existing request/response estimation -> `estimated`
- streaming requests in phase 1:
  - request count is always recorded
  - token usage is recorded as `estimated` only if SmolRouter has an explicit estimate path for that stream type
  - otherwise token usage is `missing`

Soft quota reporting should therefore distinguish:

- `actual_tokens_today`
- `estimated_tokens_today`
- requests with missing token attribution

This keeps the project-accounting story honest while leaving room for a later stream aggregation pass.

---

## 6. Implementation Plan

The design above describes the target shape. This section defines the **delivery order**
for implementation so the feature can land safely, be tested incrementally, and keep
accuracy claims honest at every step.

The sequencing principle is:

> ship identity first, then attribution, then ledgering, then operator-facing quota state, and only then optional request-plane auth transport changes.

That minimizes blast radius and avoids coupling early value to the riskiest auth/BYOK changes.

### 6.1 Delivery Phase A: Registry and data-model groundwork

**Goal**

Land the data structures and config loading path with no request-routing behavior change.

**Scope**

- add `FacadeKeyRegistry`
- extend `secret_store.py` schema backward-compatibly for `facade_keys:`
- add facade-key metadata model from `routes.yaml`
- add startup validation
- add `RequestIdentity` / `ResolvedCaller` types
- extend `ClientContext` to carry resolved identity

**Non-goals**

- no request rejection
- no quota writes
- no request-path auth changes

**Why first**

- smallest blast radius
- easiest unit-test surface
- de-risks all later phases by establishing one authoritative runtime registry

**Suggested tests**

- secret-store parsing:
  - flat provider-only secrets still work unchanged
  - nested `facade_keys:` entries parse correctly
  - mixed provider + facade secrets parse correctly
- registry validation:
  - duplicate presented secrets rejected
  - disabled keys with live secrets rejected or warned as designed
  - secret rotation lists map to one logical id
- `ClientContext` construction:
  - identity field is optional and defaults cleanly

**Exit criteria**

- runtime can build a validated `FacadeKeyRegistry`
- no behavior change on normal `/v1/*` traffic when facade-key resolution is disabled

### 6.2 Delivery Phase B: Request identity resolution and propagation

**Goal**

Resolve facade identity at the app boundary and thread it through the runtime without
changing quota or provider behavior yet.

**Scope**

- resolve facade key from `Authorization: Bearer <facade-key>` on the OpenAI write path, with `X-SmolRouter-Key` as the transitional alias
- store resolved caller on `request.state`
- thread the identity through:
  - `proxy_request()`
  - `container.route_request()`
  - `container.route_streaming_request()`
  - `mediator.route_request()`
  - model-listing endpoints
- do not rely on raw-header rediscovery downstream

**Initial transport recommendation**

The original `X-SmolRouter-Key`-first sequencing is now superseded.
Current Phase B behavior accepts local facade keys on the OpenAI write path via:

- `Authorization: Bearer srk-...`
- `X-SmolRouter-Key` as a transitional alias

The remaining migration concern is not whether `Authorization` is allowed, but that
request-path auth stays split cleanly for the unfinished phases:

- SmolRouter-local facade keys resolve locally and can be tracked
- caller-supplied upstream/BYOK bearer tokens continue to pass through unchanged
- dashboard/admin authentication can keep its separate transport and enforcement model

**Suggested tests**

- request with valid `X-SmolRouter-Key` resolves to expected logical id
- invalid key yields anonymous or rejection per configured mode
- resolved identity survives all the way into mediator/container policy context
- non-completion endpoints (`/v1/models`) see the same identity semantics
- requests without facade keys behave exactly as before when mode is `off`

**Exit criteria**

- one resolved identity object is available end-to-end on request paths
- no split-brain behavior between logging and routing layers

### 6.3 Delivery Phase C: Request logging and identity indexes

**Goal**

Persist facade identity on request records and add the minimum indexes required for cheap drilldown.

**Scope**

- add request-log fields:
  - `identity_kind`
  - `identity_subject_id`
  - optional `identity_display_name`
  - `token_accounting_state`
- serialize those fields in API responses
- add recency index for request lookup by facade key

**Why before quota writes**

- proves attribution correctness independently of counters
- makes debugging the accounting path much easier
- gives an operator-visible sanity check before any quota reporting appears

**Suggested tests**

- request detail serialization includes facade identity when present
- requests are indexed by facade key in recency order
- raw facade secrets never appear in logs or serialized payloads
- downstream provider metadata and facade identity remain distinct

**Exit criteria**

- operators can answer “which project sent this request?” from request logs alone

### 6.4 Delivery Phase D: Usage ledger in observe-only mode

**Goal**

Write facade-key usage counters after request completion, with explicit accounting state.

**Scope**

- add Redis usage ledger for facade keys
- write request-count counters for all requests
- write token counters with:
  - `actual`
  - `estimated`
  - `missing`
- keep fail-open behavior on ledger write failure
- emit accounting-health logs/telemetry on gaps

**Why observe-only**

- quota state is only as trustworthy as attribution
- this phase proves data quality before any operator-facing budget semantics are attached

**Suggested tests**

- non-streaming request with upstream `usage` increments `actual_tokens_today`
- non-streaming request without `usage` increments `estimated_tokens_today`
- streaming request with no reliable token path increments request count and marks token accounting `missing`
- Redis ledger write failures do not fail the request
- accounting gap is surfaced to logs and/or health state

**Exit criteria**

- per-project request counts are trustworthy
- token-accounting quality is visible, not hidden

### 6.5 Delivery Phase E: Soft quota reporting and APIs

**Goal**

Expose observed quota state to operators and callers without rejecting traffic.

**Scope**

- compute `ok | warning | over_soft_limit`
- return quota-state response headers
- add `/api/facade-keys`
- add `/api/facade-keys/{id}`
- optionally add dashboard views

**Why after observe-only ledger**

- the UI should reflect already-validated data, not define what “correct” means

**Suggested tests**

- threshold crossing updates quota state deterministically
- API rollups match ledger entries
- per-model and per-provider rollups reconcile to aggregate totals
- warning headers appear only when expected

**Exit criteria**

- operators can see approximate per-project usage and over-soft-limit state
- no request blocking exists yet

### 6.6 Delivery Phase F: Request-plane auth transport split

**Goal**

Add safe support for `Authorization: Bearer <facade-key>` on `/v1/*` in the modes that want OpenAI-client compatibility.

**Scope**

- implement explicit request-auth mode selection
- add dedicated upstream-auth transport for BYOK/passthrough
- disable legacy raw `Authorization` passthrough in `facade_request_auth`
- document deployment migration from `X-SmolRouter-Key`

**Why later**

- this is the riskiest phase
- it touches request auth and upstream credential transport
- it is not required to prove the accounting model itself

**Suggested tests**

- `facade_request_auth` mode:
  - `Authorization` authenticates/resolves facade identity
  - BYOK only works through dedicated upstream-auth transport
- `jwt_request_auth` mode:
  - JWT still owns `Authorization`
  - facade identity via `X-SmolRouter-Key`
- `transition_alias` mode:
  - both transports resolve the same facade identity

**Exit criteria**

- stock OpenAI-compatible clients can use facade keys via `Authorization`
- no ambiguity remains between auth principal and upstream credential

### 6.7 Delivery Phase G: Later policy phases

After the above is stable:

- preflight estimation and optional reject semantics
- facade-key-driven request class/QoS
- batch budgets and approval policy
- provider-key accounting convergence

These later phases should reuse the same registry, identity object, request indexes, and ledger rather than inventing parallel mechanisms.

## 7. Full Epic Shape

The next step must not preclude the full closeout of the epic.

The intended full shape is:

### Phase 1: Identity + attribution + usage ledger

- facade key registry
- canonical `Authorization` end-state defined
- `X-SmolRouter-Key` transitional alias
- request identity resolution
- per-request attribution
- post-completion request/token counters with explicit accounting state
- soft quota reporting
- identity indexes for drilldown

### Phase 2: Preflight estimation and optional admission policy

- estimate request impact before dispatch
- compare against current ledger state
- support per-key `action: observe | warn | reject`
- reconcile projected vs actual after completion

### Phase 3: Facade-key-driven QoS and routing policy

- default request class per facade key
- class-based `priority` injection
- class-based token/output caps
- model allow/deny or provider preferences per facade key

### Phase 4: Batch and offline budget policy

- reuse facade identity for `/v1/batches`
- thresholds by facade key
- pending approval queues attributed to submitting project/use case

### Phase 5: Ledger convergence

- introduce a generalized usage-subject abstraction if desired:
  - `facade_key`
  - `provider_key`
  - `jwt_user`
  - `batch_job`
- either wrap or migrate the older provider-key quota backend into the same conceptual model

The crucial point is:

> Phase 1 is not a throwaway.

It creates the canonical identity boundary that every later phase wants anyway.

---

## 8. Limitations and Risks

This section is deliberately blunt. These are not edge cases to hide in implementation notes.

### 8.1 Phase-1 token accuracy is incomplete by design

Phase 1 is good enough for soft accounting, not exact metering.

- streaming requests may contribute only request counts and `missing` token state
- estimated tokens are approximate, not provider-billed truth
- mixed workloads with heavy streaming will undercut quota usefulness until stream aggregation improves

That limitation must be surfaced in:

- API fields,
- dashboard labels,
- operator docs.

Do not market phase 1 as “accurate token billing”.

### 8.2 Dual transport modes increase operational complexity

During transition, operators may need to reason about:

- JWT request auth mode
- facade request auth mode
- transition alias mode
- BYOK transport mode

That is manageable only if deployment docs are explicit.

Risk control:

- keep modes explicit, not heuristic
- log the active request-auth mode at startup
- reject invalid mode combinations at startup

### 8.3 Fail-open accounting means usage gaps are possible

The design intentionally keeps facade-ledger writes fail-open.

That protects serving availability, but it means:

- counters may have gaps during Redis or serialization failures
- soft quota state can be stale or incomplete

Risk control:

- log and count accounting-write failures
- surface “accounting degraded” state to operators
- never silently pretend the ledger is complete when it is not

### 8.4 One authoritative registry is easy to violate accidentally

The current repo has multiple config-loading habits.

Risk:

- convenience reads from `routes.yaml` or `secret_store.py` later bypass the registry
- app and container drift again

Risk control:

- keep facade-key lookup behind one module/service
- add tests that request handlers and container runtime use the same registry instance or source of truth
- document “no direct facade-key config reads outside the registry” as an implementation rule

### 8.5 Request-log fanout adds write-path cost

Adding:

- extra log fields,
- a facade-key recency index,
- ledger writes,

means more Redis work on every request completion.

Risk control:

- phase the work so attribution lands before quota APIs
- benchmark completion-path overhead
- prefer batched Redis updates where practical

### 8.6 Soft quotas are policy guidance, not enforcement

Until preflight estimation and reject semantics exist:

- requests can exceed soft limits,
- bursty traffic can overshoot daily budgets,
- over-limit status is descriptive, not protective.

That is acceptable only if documented clearly.

## 9. What This Proposal Deliberately Does Not Do Yet

### 9.1 It does not repurpose `Authorization`

Phase 1 does **not universally repurpose** `Authorization` across all deployments.

Specifically:

- in `jwt_request_auth` mode, `Authorization` stays owned by JWT on `/v1/*`,
- in `facade_request_auth` mode, `Authorization` is the facade-key transport,
- the document's long-term end state is still facade-key-in-`Authorization`, but only after request auth and BYOK transports are properly split.

### 9.2 It does not replace JWT auth

JWT remains transport/authentication for protected routes when configured.
Facade keys are router-local identity and policy subjects.

Those are related, but not the same thing.

### 9.3 It does not collapse provider-key and facade-key accounting on day one

That convergence is desirable eventually, but forcing it into the next step would overfit the new feature to an old backend shape.

### 9.4 It does not promise exact billing-grade quotas

Soft quotas are approximate:

- best effort,
- suitable for local budgeting and trend analysis,
- not an invoicing primitive.

---

## 10. Data Model and API Details

### 10.1 New config concepts

`routes.yaml`

```yaml
facade_keys:
  project-foo:
    enabled: true
    display_name: "Project Foo"
    tags: ["team:ml", "usecase:agent"]
    default_class: "normal"
    quota:
      daily_requests_soft: 1000
      daily_tokens_soft: 2000000
      action: "observe"
      warn_threshold: 0.8
```

`secrets.yaml`

```yaml
facade_keys:
  project-foo: "srk_live_project_foo"
```

Future-compatible extension:

```yaml
facade_keys:
  project-foo:
    - "srk_live_project_foo_v1"
    - "srk_live_project_foo_v2"
```

That allows key rotation without changing the logical identity.

### 10.2 Suggested Python surfaces

New module:

- `smolrouter/facade_keys.py`

Responsibilities:

- load facade key metadata from routes config,
- load key secrets from the extended secrets store,
- validate presented facade key,
- return a `RequestIdentity` / `ResolvedCaller`.

Likely touchpoints:

- `smolrouter/interfaces.py`
- `smolrouter/container.py`
- `smolrouter/app.py`
- `smolrouter/mediator.py`
- `smolrouter/request_metadata.py`
- `smolrouter/database.py`
- `smolrouter/redis_backend.py`

### 10.3 Logging fields

Suggested additions to request log create/completion schema:

- `identity_kind`
- `identity_subject_id`
- optional `identity_display_name`
- `token_accounting_state`

These should be serialized in API/dashboard payloads the same way provider metadata is today.

### 10.4 Dashboard/API shape

Future-facing but phase-1-compatible endpoints:

- `/api/facade-keys`
  - configured keys, usage, soft limit state
- `/api/facade-keys/{id}`
  - per-model usage breakdown
- request detail / dashboard rows
  - include `identity_subject_id`

The next step only needs the data model and serialization path to support this; the UI can follow immediately after or in a short second patch.

---

## 11. Why This Enables the Other Proposals

### 11.1 Request-class priority injection

That proposal needs a stable answer to:

> "Which caller should default to `interactive`, `normal`, or `background`?"

Facade keys are a better policy handle than source IPs or ad hoc headers.

### 11.2 Offline batch corralling

That proposal needs a stable answer to:

> "Which local project submitted this batch, and which budget/approval threshold applies?"

Again, facade keys are the obvious subject.

### 11.3 Provider-key observability

The router can report both:

- local project identity via facade key,
- downstream key/provider identity via provider metadata.

Those are complementary dimensions, not competing ones.

---

## 12. Verification

The next-step implementation should verify:

1. A request with `X-SmolRouter-Key` resolves to the configured logical identity and persists `identity_kind=facade_key` plus `identity_subject_id=<logical-id>` on the request log.
2. The raw facade key is never stored in logs, Redis records, or forwarded upstream.
3. A request can carry both `Authorization: Bearer <jwt>` and `X-SmolRouter-Key: <facade>` without ambiguity.
4. BYOK/passthrough behavior remains mode-correct:
   - unchanged in `jwt_request_auth` and `transition_alias` until migrated,
   - explicitly moved to dedicated upstream-auth transport in `facade_request_auth`.
5. Rotating a facade key secret while keeping the same logical id continues usage accumulation under the same project identity.
6. Soft token/request counters increment from request completion using:
   - actual usage when present,
   - estimation fallback where explicitly supported,
   - `missing` accounting state where phase-1 token attribution is not reliable.
7. `X-SmolRouter-Accounting-State` and `X-SmolRouter-Quota-State` follow the documented completeness/state rules, including `partial` and `degraded` cases.
8. Over-soft-limit state is observable without blocking traffic.
9. The design leaves room for later `reject` semantics based on preflight estimation rather than forcing hard enforcement into the first slice.

---

## 13. Test Strategy

Implementation should be validated at four layers, not just with a few happy-path unit tests.

### 13.1 Unit tests

- registry parsing and validation
- request-auth mode resolution
- facade-key resolution precedence
- token-accounting-state classification
- quota-state threshold logic
- accounting-completeness calculation

### 13.2 Integration tests

- app -> container -> mediator identity propagation
- request log persistence with facade identity
- ledger writes and recency indexes
- BYOK behavior under each request-auth mode
- streaming vs non-streaming accounting-state outcomes
- degraded/partial accounting states map to headers deterministically

### 13.3 Migration and compatibility tests

- existing deployments without facade keys behave unchanged
- existing provider secrets continue to load unchanged
- JWT request auth still works in `jwt_request_auth`
- legacy BYOK passthrough remains functional until explicitly migrated

### 13.4 Negative and degradation tests

- invalid facade keys
- duplicate secrets at startup
- Redis ledger failures
- missing `usage` blocks in upstream responses
- stream-heavy workloads with `missing` token accounting

The bar for moving between delivery phases should be:

- no auth regressions,
- no provider-routing regressions,
- accounting accuracy claims matched by test outcomes,
- degradation states surfaced, not hidden.

## 14. Recommendation

Reframe the backlog priority around this sequence:

1. **High:** Facade-key identity and attribution
2. **High:** Facade-key request/token accounting
3. **Medium-high:** Soft quota reporting / warnings
4. **Medium-high:** Provider-key accounting convergence
5. **Later:** Hard admission/reject semantics

This sequence matches the user-facing value:

- identify projects and use cases,
- see rough budget usage,
- later add policy and enforcement,
- without breaking auth or passthrough semantics in the process.
