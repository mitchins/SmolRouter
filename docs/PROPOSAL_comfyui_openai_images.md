# Proposal: ComfyUI behind the OpenAI Images ingress

**Status:** Proposed (future / not-now). No code yet — this is the design of record.
**Author:** Mitchell Currie
**Date:** 2026-07-04
**Depends on / extends:** existing `/v1/images/generations` ingress, provider abstraction, mediator image routing, request logging, blob storage, health monitoring.

---

## 1. Problem

SmolRouter now has a single OpenAI-compatible image-generation ingress:

- `POST /v1/images/generations`

That surface is useful because existing OpenAI-oriented clients can already talk to
it without learning a second image API dialect. Today, the downstreams are hosted
providers. There is no equivalent local-first backend for the common case:

> "Generate me an image, locally, without making the client understand ComfyUI."

ComfyUI is a strong fit as a local image engine, but its native API is a workflow
graph API, not a simple image-generation API:

- callers submit a workflow graph to `/prompt`
- execution is queued asynchronously
- completion is normally observed via WebSocket and/or `/history`
- output files are then fetched via `/view`

That is exactly the complexity most callers should **not** have to adopt.

The opportunity is to keep the router’s **OpenAI image surface** as the client
contract, while letting a fixed ComfyUI workflow do the real downstream work.

---

## 2. Goals

- Preserve one client ingress: `POST /v1/images/generations`.
- Keep clients OpenAI-compatible. Do **not** make tooling speak workflow graphs,
  queue ids, or ComfyUI WebSocket message types.
- Treat the workflow JSON as the place where complexity lives.
- Expose only a very small ergonomic request surface on top of the workflow.
- Support a local, free-first image backend for existing OpenAI-oriented tooling.
- Keep the first implementation narrow enough to be production-grade.

### Non-goals

- No general plugin runtime or dynamic plugin loader as a prerequisite.
- No attempt to expose arbitrary ComfyUI graph editing through the request API.
- No `/v1/images/edits` or `/v1/images/variations` in the first slice.
- No streaming image-generation API.
- No batch image generation in the first slice.
- No promise that every OpenAI image field maps cleanly to every workflow.

---

## 3. Grounding in the upstream API

This proposal is grounded in the current ComfyUI server model:

- workflows must be submitted in **API format** JSON, not the normal UI save
  format
- workflow execution is queued with `POST /prompt`
- queue/history can be inspected with `/queue`, `/history`, and `/history/{prompt_id}`
- generated files can be retrieved via `/view`
- execution progress is emitted via `/ws`

The official ComfyUI docs also describe three interaction patterns and recommend
the **WebSocket + history** pattern for most use cases, with `SaveImageWebsocket`
as a more specialized real-time alternative.

SmolRouter should build on that reality, not pretend ComfyUI is a stateless
single-call image endpoint.

---

## 4. Architecture fit in SmolRouter

### 4.1 Recommended shape

The clean fit is a **first-party bundled provider**:

- provider type: `comfyui`
- disabled by default
- shipped with the router as a bundled integration
- optionally described as a "first-party plugin/bundle" in docs and config, but
  implemented initially as a normal provider type

This avoids inventing a runtime plugin system before the product need is proven.

### 4.2 Why not a general plugin loader first

The repo already has the extension seam that matters:

- provider config
- provider factory
- mediator dispatch
- provider health/model discovery

Adding ComfyUI through those seams is a bounded change. Building a generic plugin
runtime first would expand scope into:

- plugin loading and isolation
- plugin configuration schema
- plugin lifecycle and dependency management
- plugin health and observability contracts

That is architectural overreach for a feature whose immediate value is simply
"local image generation behind the existing OpenAI surface."

### 4.3 Follow-on cleanup this proposal intentionally calls for

Current image routing is still somewhat provider-specific. Before or alongside a
future ComfyUI implementation, the mediator should move toward a **capability-led**
image dispatch path rather than accumulating provider-type branches.

Desired direction:

```text
image request
  -> mediator resolves model/provider
  -> provider advertises image-generation capability
  -> provider.generate_image(...)
```

Not:

```text
if google image provider:
    ...
elif comfyui:
    ...
elif next image backend:
    ...
```

This is not just a style cleanup. The current repo shape means a future ComfyUI
implementation needs explicit new seams:

- an image-generation capability on the provider side rather than a Google-only
  special path
- a typed `ComfyUIConfig` registered in `ProviderFactory`
- provider config parsing that accepts workflow-backed model definitions rather
  than pretending plain `ProviderConfig` is enough

Phase 0 should state that work directly instead of hiding it inside "future
provider cleanup."

---

## 5. Client-facing contract

### 5.1 Supported ingress

Keep the existing router ingress:

```json
POST /v1/images/generations
{
  "model": "local-sdxl [comfyui-local]",
  "prompt": "A brutalist concrete house in rain at dusk",
  "size": "1024x1024"
}
```

### 5.2 Narrow supported request surface

The first implementation should stay intentionally small:

- required:
  - `model`
  - `prompt`
- supported generic fields:
  - `size`
  - `user`
- provider extension fields, namespaced:
  - `extra_body.comfy.steps`
  - `extra_body.comfy.seed`
  - `extra_body.comfy.negative_prompt`

### 5.3 Deliberate restrictions

For a sane first slice:

- `n` must be `1`
- `stream` must be false/absent
- `/v1/images/edits` and `/v1/images/variations` remain unsupported
- unsupported OpenAI image fields should be rejected explicitly, not silently
  ignored

### 5.4 Response shape

First slice should return standard OpenAI-style image response objects with one
image item.

Recommended first response mode:

- treat absent `response_format` as `b64_json`
- support explicit `response_format: "b64_json"`
- reject `response_format: "url"` initially unless SmolRouter is also ready to
  store the returned bytes and issue a router-managed URL

This keeps the first implementation honest and avoids leaking raw ComfyUI file
paths or host-local URLs into client responses.

This is an explicit compatibility choice for the ComfyUI-backed provider surface:
the router should make the default deterministic instead of inheriting ambiguous
downstream behavior.

If `url` parity is later added, the URL should be **SmolRouter-managed**, backed
by blob storage or another controlled file-serving path, not ComfyUI’s native
output path.

---

## 6. Workflow model

### 6.1 Operator-owned workflow templates

The key design choice is:

> Clients do not send workflows. Operators configure workflows.

Each exposed SmolRouter model maps to a fixed exported ComfyUI workflow in API
format plus a small binding spec.

Example configuration sketch:

```yaml
providers:
  - name: "comfyui-local"
    type: "comfyui"
    enabled: false
    url: "http://127.0.0.1:8188"
    models:
      - name: "local-sdxl"
        workflow_file: "/etc/smolrouter/workflows/sdxl_basic_api.json"
        output_mode: "history"
        bindings:
          prompt: ["6", "inputs", "text"]
          negative_prompt: ["7", "inputs", "text"]
          steps: ["3", "inputs", "steps"]
          seed: ["3", "inputs", "seed"]
          width: ["5", "inputs", "width"]
          height: ["5", "inputs", "height"]
          save_node: "9"
        defaults:
          steps: 20
          negative_prompt: ""
        limits:
          min_steps: 1
          max_steps: 40
          supported_sizes: ["1024x1024", "1536x1024"]
```

This is intentionally a **new typed config shape**, not a claim that current
`ProviderConfig` already accepts or understands these fields.

### 6.2 Why bindings matter

The binding layer is what keeps the API ergonomic while letting the workflow hold
all the real complexity:

- node ids stay local to the workflow template
- the router knows which fields are safely patchable
- unsupported request fields are rejected before they mutate an arbitrary graph

This prevents the feature from turning into "remote workflow editing over the
OpenAI image endpoint," which would be the wrong product.

### 6.3 Workflow expectations

The first supported workflow mode should be conservative:

- one output image
- one known output node
- fixed graph topology
- only bounded scalar/text substitutions at configured binding points

Out of scope for the first slice:

- arbitrary graph rewrites
- operator-less workflow imports from clients
- surfacing every node parameter as a request knob
- multi-image fan-out inside one request

---

## 7. Execution model

### 7.1 Recommended downstream pattern

Use the official **WebSocket + history** model internally:

1. build a patched workflow payload from the configured template
2. submit with `POST /prompt`, including a provider-owned `client_id`
3. use a router-generated request correlation id as the desired `prompt_id` when
   supported, but do **not** assume duplicate suppression or idempotency from
   that alone
4. wait for completion via WebSocket events correlated by both `client_id` and
   `prompt_id`
5. fetch final output metadata from `/history/{prompt_id}`
6. fetch image bytes via `/view`
7. return OpenAI-style image JSON to the client

This is the best balance of simplicity and correctness for the first slice.

### 7.2 Why not `SaveImageWebsocket` first

`SaveImageWebsocket` is useful, and the official docs show it as a real option.
But it is a more specialized transport:

- binary frame parsing is required
- output collection is more tightly coupled to one special node
- debugging and post-mortem inspection are weaker than with history-backed output

It is a valid future optimization, especially if output-file accumulation becomes a
pain point, but not the recommended first path.

### 7.3 WebSocket ownership

The provider should own WebSocket lifecycle, not the app route.

Recommended future shape:

- one provider-level WebSocket session manager per ComfyUI provider
- one provider-owned stable `client_id` per provider instance / WebSocket session
- prompt waiters keyed by `prompt_id`
- reconnect handling in one place
- explicit handling for `execution_success`, `execution_error`,
  `execution_interrupted`, `executing`, and connection loss

This avoids one raw WebSocket implementation per request and keeps queue-state
handling inside the provider where it belongs.

For the first production slice, if the WebSocket path is degraded after startup,
the provider should fail fast rather than silently switching behavior. A polling
fallback is a valid Phase 2 hardening feature, but should not be half-promised in
Phase 1.

### 7.4 Synchronous router, asynchronous downstream

The router remains synchronous from the client point of view:

- client calls `/v1/images/generations`
- router waits
- router returns one final response

ComfyUI remains asynchronous behind that facade. The provider absorbs the queueing
and wait mechanics.

### 7.5 Admission control and backpressure

A synchronous OpenAI facade over ComfyUI still needs a local protection layer.
Otherwise parallel callers can enqueue unbounded expensive jobs and only learn
about trouble after work is already accepted upstream.

Phase 1 should include provider-local admission control:

- a per-provider in-flight request cap inside SmolRouter
- a per-provider pending-waiter cap inside SmolRouter
- optional `/queue` preflight checks when the operator enables them

Recommended behavior:

- reject before `POST /prompt` when SmolRouter’s local cap is exceeded
- return `429` when the router’s own protective cap rejects the request
- return `503` when ComfyUI is reachable but already too deep/unhealthy to accept
  more synchronous facade traffic safely

The goal is to stop blast-radius expansion before a queue timeout merely reports
damage after the fact.

Deployment assumption for the first production slice:

- support one SmolRouter process per ComfyUI-backed provider/backend pair

If operators want multiple SmolRouter workers or instances targeting the same
ComfyUI queue, Phase 1 needs either:

- shared/distributed admission control, or
- mandatory `/queue` preflight policy with clearly documented race limits

Do not present multi-worker fan-in to one ComfyUI queue as a solved problem in
the first slice.

---

## 8. Failure, timeout, and retry semantics

### 8.1 No automatic retries after uncertain submission

`POST /prompt` is not something the router should blindly retry once the request
may already have been accepted upstream. Otherwise a transient network ambiguity
can enqueue duplicate image jobs.

Future rule:

- retry only before a submission is known to have been sent
- once submission may have happened, treat the request as in-doubt and surface a
  controlled error rather than guessing

### 8.2 Global interrupt is dangerous

ComfyUI exposes `/interrupt`, but that is a **global** operation, not a per-job
cancel primitive in the local server model.

So the first implementation should **not** auto-interrupt on request timeout.

If the router times out waiting for a prompt:

- return a timeout to the caller
- detach the waiter in a cancellation-safe `finally` path
- leave the upstream job alone

That is safer than killing unrelated queued or running work.

### 8.3 Timeout model

The provider should have separate time budgets for:

- connect timeout
- queue wait timeout
- execution timeout
- output fetch timeout

Do not collapse all of these into one generic HTTP timeout.

Also account for SmolRouter’s outer request timeout/cancellation behavior:

- provider cleanup must still run when the outer route wrapper cancels the task
- per-prompt waiters must not leak
- no selected upstream/request-state counters should be left pinned after
  cancellation
- timeout cleanup must never degrade into a hidden `/interrupt`

### 8.4 Error mapping

The provider should map ComfyUI failures into explicit OpenAI-style error payloads:

- invalid workflow/configuration -> `500` internal error on operator side
- unsupported request fields -> `400 invalid_request_error`
- queue/execution timeout -> `504`
- upstream unavailable -> `503`

The client should never receive raw workflow node dumps unless explicitly enabled
for debug logging.

---

## 9. Observability and storage

### 9.1 Request metadata to preserve

For each routed image request, record at least:

- provider id
- workflow model name
- ComfyUI `prompt_id`
- queue wait duration
- execution duration
- output fetch duration
- timeout/failure phase

This is the minimum needed to debug queue stalls and broken workflows.
These fields belong in the **first production slice**, not as a later nice-to-have.
That in turn means Phase 0/1 must explicitly extend SmolRouter’s request
metadata/log plumbing to persist and expose these fields rather than assuming they
fit into the current provider-generic schema for free.

### 9.2 Output bytes vs. ComfyUI files

If the first slice uses `SaveImage` plus `/view`, ComfyUI will persist output files
on its own side. That creates an operational risk:

- local disk growth
- lingering outputs even when the router only needed transient bytes

Mitigations for the future implementation:

- dedicate workflow output prefixes for router-managed requests
- document retention expectations clearly
- optionally add a janitor only when the output path is known and intentionally
  managed by the operator

This is one reason `response_format: "url"` should not be promised too early.

---

## 10. Security model

SmolRouter should remain the client-facing auth boundary.

Inference from the current ComfyUI server documentation:

- the local server API is presented as plain HTTP/WebSocket routes
- this proposal should assume ComfyUI is bound to localhost or a trusted private
  network segment
- if ComfyUI is exposed beyond that, the operator should put it behind a reverse
  proxy or equivalent network boundary

Do not design the first provider around internet-exposed raw ComfyUI instances.

---

## 11. First implementation scope

The future build should stay narrow.

### In scope

- one new provider type: `comfyui`
- one ingress: `/v1/images/generations`
- one image per request
- one configured workflow per exposed model
- bounded request overrides: prompt, optional negative prompt, optional steps,
  optional seed, supported size mapping
- synchronous response with `b64_json`
- provider health check and basic model discovery/config-backed model list
- provider-local admission control with explicit `429` / `503` backpressure
  behavior
- single-process deployment assumption per ComfyUI backend unless stronger shared
  coordination is added

### Explicitly out of scope

- generic plugin runtime
- arbitrary workflow upload by clients
- edits/variations
- streaming
- `n > 1`
- batching
- dashboard redesign
- exposing raw queue/history endpoints to clients

---

## 12. Testing plan

### 12.1 Unit tests

- provider config validation
- workflow template loading and binding-path patching
- request field validation and rejection
- `response_format` handling for absent / `b64_json` / `url`
- size mapping and override bounds
- mediator dispatch to the ComfyUI image provider
- error mapping for timeout / unavailable / invalid workflow cases
- provider-local admission control decisions
- request metadata/log schema persistence for `prompt_id`, phase timings, and
  failure phase

### 12.2 Integration tests

Use a fake ComfyUI test server that simulates:

- `/prompt`
- `/ws`
- `/history/{prompt_id}`
- `/view`

Scenarios:

- successful image generation
- queue timeout
- execution failure
- `execution_error`
- `execution_interrupted`
- duplicate/ambiguous submission handling
- malformed `/history`
- missing configured output node
- zero output images
- unexpected multiple output images
- `/view` failure on output fetch
- `/view` query parameter/path escaping correctness
- outer-route cancellation / timeout cleanup with no leaked waiters or hidden
  `/interrupt`
- local admission-control rejection before `POST /prompt`
- upstream queue/backpressure rejection path

### 12.3 Local smoke

Real local smoke against a running ComfyUI instance should prove:

- exported workflow loads correctly
- prompt binding works
- one synchronous OpenAI-compatible image request returns a valid image
- concurrent requests do not cross-wire `prompt_id` waiters
- provider-owned `client_id` correlation behaves correctly across reconnects or
  fresh sessions

### 12.4 What should not be required for the first proof

- no large custom-node matrix
- no batch generation coverage
- no edits/variations coverage
- no `url` response parity proof

---

## 13. Risks and callouts

### Biggest real risks

- **Workflow brittleness.** Node ids and graph structure are fragile unless the
  workflow template is treated as an operator-owned contract.
- **Queue semantics.** ComfyUI is asynchronous and queue-backed; the router must
  not pretend this is a single downstream HTTP call.
- **Retry ambiguity.** Duplicate prompt submission is easy to cause if retry logic
  is naive.
- **Output accumulation.** `SaveImage`-based workflows will create files that need
  retention thought.
- **Global interrupt.** Cancellation is dangerous if implemented casually.
- **Surface creep.** Exposing too many workflow knobs would destroy the ergonomic
  goal and create an untestable compatibility surface.

### The main architectural rule

Do not make the router "ComfyUI over OpenAI by reflection."

Make it:

- fixed workflow
- tiny request surface
- explicit validation
- one stable image-generation contract

---

## 14. Phasing

### Phase 0: architecture cleanup

- move image dispatch toward provider capability rather than provider-type checks
- add an explicit image-provider capability/interface
- define and register a typed `ComfyUIConfig`
- define provider config shape for workflow-backed image providers
- extend request metadata/log plumbing for Comfy-specific execution fields

### Phase 1: narrow provider bundle

- add bundled `comfyui` provider type
- config-backed model list
- workflow patching
- WebSocket + history execution path
- provider-owned `client_id` management and explicit `prompt_id` correlation
- `b64_json` response only
- provider-local in-flight / pending caps with explicit `429` / `503`
  backpressure behavior
- single-process deployment assumption per ComfyUI backend
- phase-1 observability: `prompt_id`, queue/execution/fetch timings, failure phase
- cancellation-safe waiter/state cleanup under outer route timeout
- no automatic `/interrupt`

### Phase 2: hardening

- polling fallback
- better health/failure taxonomy
- local smoke harness against a real ComfyUI server
- optional richer dashboard surfacing of the phase-1 metadata

### Phase 3: optional ergonomics upgrades

- router-managed `response_format: "url"`
- optional `SaveImageWebsocket` mode
- richer but still bounded override model where justified

---

## 15. Recommendation

Proceed later as a **first-party bundled `comfyui` provider** behind the existing
OpenAI image ingress, not as a general plugin-system project.

The important product idea is correct:

- clients keep speaking OpenAI images
- the workflow JSON absorbs the complexity
- operators choose the workflow
- users get "generate me an image, free and local" without adopting ComfyUI’s API

The important implementation guardrails are equally important:

- one narrow endpoint
- one bounded override surface
- no fake retry semantics
- no premature URL parity
- no general plugin runtime before the provider proves its value

---

## References

- ComfyUI Server Routes: https://docs.comfy.org/development/comfyui-server/comms_routes
- ComfyUI API Examples: https://docs.comfy.org/development/comfyui-server/api-examples
- ComfyUI Server Messages: https://docs.comfy.org/development/comfyui-server/comms_messages
- ComfyUI Workflow API Format: https://docs.comfy.org/development/api-development/workflow-api-format
- ComfyUI Cloud API Overview: https://docs.comfy.org/development/cloud/overview
