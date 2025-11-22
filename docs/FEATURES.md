# Feature tour

SmolRouter combines compatibility, routing logic, and observability so you can keep legacy applications online while evolving your model estate.

## Intelligent routing

- Match traffic on model IDs, regex patterns, or request metadata such as source IPs.
- Rewrite either the upstream target or the model name on a per-rule basis.
- Compose reusable aliases that handle provider failover or split traffic by weight.
- Layer rule sets from `routes.yaml` with environment-driven defaults so that ops teams can manage policy without shipping code.

## Protocol compatibility and subsumption

- OpenAI, Ollama, and Google GenAI transports with shared request/response semantics.
- Streaming for chat, completions, and Ollama generate endpoints.
- Legacy model remapping via the `MODEL_MAP` environment variable continues to work exactly as it did in 1.x.
- Optional content transformations: `<think>` scrubbing, fenced JSON stripping, and `/no_think` hints.

## Observability and operations

- Persistent request log with token estimation, latency histograms, and recent traffic views.
- Scatter plots to visualise token counts versus latency for capacity planning.
- Blob storage abstraction for request/response payloads with size limits and retention policies.
- Quota tracking and rate limiting at the API key or token level.

## Extensibility

- Provider factory pattern with dependency-injection container so you can bring your own backends.
- Web UI built with FastAPI and HTMX templates for rapid customisation.
- Ready for sidecar deployments or centralised gateways thanks to configuration-over-code design.
