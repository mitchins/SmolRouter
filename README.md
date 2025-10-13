# SmolRouter

SmolRouter is a lightweight AI gateway that keeps OpenAI-compatible clients running while you move traffic across any mix of providers.

## What

- OpenAI-compatible proxy with drop-in model remapping (`gpt-4` can become `llama3-70b` without touching client code).
- Smart routing engine that can balance traffic, apply per-team policies, and fail over automatically.
- Built-in observability, quotas, and security controls for production-facing deployments.

## Why

Use SmolRouter when you need to:

- Consolidate multiple model vendors or on-prem hosts behind a single OpenAI-style endpoint.
- Preserve legacy client integrations while experimenting with new models or providers.
- Inspect, audit, and meter requests without giving every application direct access to provider API keys.
- Enforce consistent policies (JWT auth, request throttling, content filters) at the edge of your AI estate.

## How

1. **Run SmolRouter.** Pick Docker or Python—both expose the same HTTP interface.

   ```bash
   # Docker
   docker build -t smolrouter .
   docker run -d -p 1234:1234 \
     -e DEFAULT_UPSTREAM="http://host.docker.internal:8000" \
     -e MODEL_MAP='{"gpt-4":"llama3-70b"}' \
     smolrouter
   ```

   ```bash
   # Python
   pip install smolrouter
   export DEFAULT_UPSTREAM="http://localhost:8000"
   export MODEL_MAP='{"gpt-4":"llama3-70b"}'
   smolrouter
   ```

2. **Point clients at the router.** Swap the base URL and keep your existing model IDs.

   ```python
   import openai

   client = openai.OpenAI(
       base_url="http://localhost:1234/v1",
       api_key="local-proxy-key",  # forwarded to your upstreams
   )

   response = client.chat.completions.create(
       model="gpt-4",  # transparently remapped if MODEL_MAP defines it
       messages=[{"role": "user", "content": "Hello"}],
   )
   ```

3. **Configure deeper routing rules when ready.** Drop a `routes.yaml` next to the binary or set `ROUTES_CONFIG` to point at a shared config repo.

## Next steps

- [Feature tour](docs/FEATURES.md) — deep dive into routing, observability, and subsumption strategies.
- [Configuration reference](docs/CONFIGURATION.md) — every environment variable, default, and YAML option.
- [Operations and security](docs/OPERATIONS.md) — deployment hardening, auth modes, and quota tooling.
- [Google GenAI setup](docs/GOOGLE_GENAI.md) — provider-specific credential guidance.
- [Web UI navigation](NAVIGATION.md) — how to use the dashboard.
- [Release notes for 2.0.0](docs/RELEASE_NOTES_v2.0.0.md) — summary of what changed since 1.x.

SmolRouter ships with 160 automated tests covering model remapping, routing, logging, and quota enforcement. Run `pytest` after `pip install -e .[dev]` to execute the full suite.
