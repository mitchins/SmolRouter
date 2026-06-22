# Proposal: Request-class priority injection (vLLM QoS)

## Goal

Make SmolRouter the **policy boundary** for request class. Clients declare
*intent* (interactive / background / best-effort); SmolRouter normalizes that into
an explicit upstream **`priority`** value and injects it into the OpenAI-compatible
request body before it reaches vLLM. Background services should not have to know or
set priority — the router owns the contract.

```
client declares intent  →  SmolRouter classifies + clamps  →  vLLM gets explicit priority
```

This is a routing/QoS feature, not a correctness fix. It exists because a single
dedicated GPU has no external backpressure to lean on — fairness has to come from
the scheduler, and the scheduler only does the right thing if priorities are set
consistently and on purpose.

## Problem

The target deployment is **one RTX 3090 dedicated to one model** (`gpt-oss-20b`)
served by vLLM with the priority scheduler:

```
--scheduling-policy priority      # lower numeric priority runs earlier; arrival breaks ties
--max-model-len 20000
--max-num-seqs 6
--max-num-batched-tokens 20000
# EAGLE3 speculative decoding for the 4k-context hammer traffic
```

The mixed workload is the whole point: rare human long-context requests
(~16k–20k, latency-sensitive) sharing the GPU with frequent background/system jobs
(~4k, throughput-oriented, latency-tolerant). With no global background-queue
coordinator, the *only* lever that keeps an interactive request from sitting behind
a pile of background work is vLLM's per-request `priority`.

vLLM priority is **relative**: a request only "wins" because its number is lower
than what it's competing with. That means:

- If priority is left unset, requests fall to vLLM's default and the QoS guarantee
  is implicit and fragile.
- If *background* callers can default to `priority: 0` (or omit it and land
  somewhere favourable), they starve the interactive class — the exact failure we
  want to prevent.
- Requiring every background worker to remember to set `priority: 100` is a
  contract that *will* be violated, because it lives in N scattered codebases.

vLLM exposes `priority` as a vLLM-specific request param (via `extra_body` on the
OpenAI client, or top-level in raw JSON), so the value is trivial to set — the hard
part is setting it *consistently and trustworthily*, which is a router concern.

## Key insight

**A router is exactly where request-class policy belongs.** SmolRouter already sits
in front of vLLM, already resolves model aliases, already has facade-key identity
and per-key/per-model accounting. Adding "what service class is this, and what
priority does that map to" reuses all of that and gives a *single* enforcement
point for:

- priority injection,
- class-based token/prompt caps,
- model-alias routing,
- (future) backend selection / fallback.

The crucial policy rule falls out of the relativity of vLLM priority:

> **Unknown/undeclared class defaults to background, never interactive.**

"No declared priority" must mean *background*, so a random worker that forgets the
contract degrades gracefully instead of silently claiming the interactive class.

## Design

### Class contract (intent, not raw numbers)

Clients declare a **class**, not a raw priority. SmolRouter maps class → priority:

| Class                  | Injected `priority` | Expected shape                    |
|------------------------|---------------------|-----------------------------------|
| `interactive` (human)  | `0`                 | ctx up to server cap; `max_tokens` ≤ ~2048 |
| `cli` (trusted semi-interactive) | `10`      | ctx up to cap; modest `max_tokens` |
| `normal` / default     | `50`                | general API traffic               |
| `background` / system  | `100`               | prompt ≤ ~4k; `max_tokens` ≤ ~512  |
| `best-effort` batch     | `200`               | prompt ≤ ~4k; `max_tokens` ≤ ~256  |

Numbers are relative and tunable; what matters is the ordering and that the gaps
leave room to insert classes later.

### How class is declared

In rough precedence order (router resolves the first that applies):

1. **Model alias** — the cleanest, since SmolRouter already does alias→model
   resolution. e.g.
   ```
   gpt-oss-20b-human  → model openai/gpt-oss-20b, priority 0
   gpt-oss-20b        → model openai/gpt-oss-20b, priority 50
   gpt-oss-20b-batch  → model openai/gpt-oss-20b, priority 100
   ```
2. **Header** — `X-SmolRouter-Class: interactive|background|best-effort`.
3. **Facade key / source identity** — a key or client known to be human-facing
   (OpenWebUI, aichat) maps to `interactive`; a worker key maps to `background`.
4. **Default** — `background`/`normal` (see the key rule above), *not* interactive.

### Trust & clamping (the security-ish part)

Raw `priority` from the wire is **not** honoured for untrusted clients:

- Untrusted clients **cannot** submit raw `priority`; it is stripped or clamped to
  their class's value before forwarding.
- Only **trusted** clients/keys may request an elevated class (e.g. `interactive`).
- A request asking for a class it isn't allowed is clamped down, never up.

Sketch (illustrative, not final API):

```python
def classify(req, client) -> int:
    if client in TRUSTED_HUMAN_CLIENTS:
        return 0
    cls = req.headers.get("X-SmolRouter-Class")
    if cls == "interactive" and client in ALLOWED_INTERACTIVE:
        return 0
    if cls == "best-effort":
        return 200
    return 100  # default: background, never interactive

# then, after alias resolution:
body["priority"] = classify(req, client)   # overwrites any client-supplied value
```

### Token / prompt clamping

Priority protects *scheduling*; it does nothing for the KV budget. Pair it with
class-based caps so background work cannot define the service class by sheer size:

- clamp `max_tokens` down to the class ceiling,
- (optionally) reject / downgrade background requests whose prompt blows past the
  class's expected context.

This keeps the conservative serving profile intact: a long interactive request
should always have room to be preferred, and background jobs cannot permanently
consume the whole KV budget.

### Where it plugs in

Injection happens in the request-mutation path that already rewrites the model on
the way upstream (`smolrouter/mediator.py` — the load-balancer "mutated request
model" step is the natural sibling). Class resolution leans on existing alias and
facade-key plumbing. No new client SDK; clients keep using stock OpenAI clients and
either pick an alias, set one header, or simply rely on their key's default class.

## Phasing

1. **Inject by alias** — add priority-bearing model aliases that resolve to the
   real model + an injected `priority`. Zero client changes beyond picking the
   alias. Smallest useful slice; proves the injection path end-to-end.
2. **Header + key/identity classification** — `X-SmolRouter-Class` and
   per-key/per-client default class, with the default-to-background rule.
3. **Trust & clamping** — strip/clamp raw client `priority`; gate elevated classes
   behind trusted identity.
4. **Token/prompt caps per class** — clamp `max_tokens`, optionally guard context.
5. **Observability** — surface resolved class/priority on the request record and in
   the dashboard so QoS decisions are auditable (reuses existing per-request logging).

## Non-goals (for now)

- A global cross-process background queue / scheduler — explicitly out; we rely on
  vLLM admission + priority fairness on the single GPU.
- Guaranteed preemption of every running background job — priority biases
  *scheduling*, it does not promise instant preemption; the conservative serving
  profile is what bounds worst case.
- Multi-GPU / multi-model fan-out priority — single dedicated 3090 is the target.
- Per-token fair-share accounting / billing-grade QoS.

## Open questions

- Canonical class declaration: lean on **model aliases** (clean, already-resolved)
  vs. a **header** (works without minting aliases) — likely support both, alias
  taking precedence.
- The trust boundary: what designates a client "trusted" for elevated class —
  facade key, source IP, client name, or an explicit allowlist? (Reuses facade-key
  identity work.)
- Exact numeric ladder and whether to expose it as config vs. fixed constants.
- Interaction with token clamping: hard-reject oversized background requests vs.
  silently clamp `max_tokens` vs. downgrade class.
- Should the resolved priority/class be persisted on the request record for
  analytics from day one (phase 1) rather than phase 5?

## Verification

1. **Alias injection**: request via `gpt-oss-20b-batch` arrives upstream with
   `model=openai/gpt-oss-20b` and `priority=100`; `gpt-oss-20b-human` with
   `priority=0`.
2. **Default is background**: a request with no class declared and an untrusted key
   is forwarded with the background priority, not interactive.
3. **Clamping**: an untrusted client that puts `priority: 0` in the body is
   forwarded with its class's priority, not `0` (raw value stripped/clamped).
4. **Trust gate**: only an allowlisted client/key can resolve to `interactive`/`0`.
5. **Token cap**: a background request with `max_tokens` above the class ceiling is
   clamped to the ceiling before forwarding.
6. **Passthrough**: with the feature disabled/unconfigured, requests forward
   byte-for-byte as today (no `priority` injected).
