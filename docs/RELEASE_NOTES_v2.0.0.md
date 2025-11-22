# SmolRouter 2.0.0 release notes

SmolRouter 2.0.0 focuses on turning the project into a production-ready AI gateway while preserving the core proposition from 1.0: transparent model remapping and protocol subsumption.

## Highlights

- **Google GenAI transport** – native Gemini routing with the official `google-genai` SDK and HTTPX transport.
- **Provider abstraction refresh** – dependency-injection container and provider factory for adding new backends without touching request handlers.
- **Observability upgrades** – richer performance plots, quota tracking, and blob storage sharding for large deployments.
- **Security posture** – hardened JWT validation, stricter rate-limiting primitives, and explicit environment toggles for logging and thinking-stripping.

## Backwards compatibility

- `MODEL_MAP` behaves exactly as it did in 1.x (exact keys or regex patterns). Existing remapping files continue to work.
- OpenAI-compatible paths (`/v1/chat/completions`, `/v1/completions`) and Ollama endpoints remain unchanged.
- Default environment values are identical to 1.x. The only additions are new optional knobs for quotas and blob storage sizing.

## Dependency updates

- Added `google-api-core` to guarantee availability of Google transports.
- Bumped runtime dependencies to include `aiohttp`, `fakeredis`, and `pytz` so the packaged wheel matches the features used in tests.

## Upgrade checklist

1. Update to SmolRouter 2.0.0 via `pip install -U smolrouter` or rebuild your Docker image.
2. Ensure the environment contains any new credentials required for Google GenAI, if used.
3. Run `pytest` (160 tests) against your configuration or CI pipeline to confirm routing rules behave as expected.
4. Review new docs in `docs/` for configuration, operations, and feature overviews.

Need help? Open an issue at [github.com/mitchins/smolrouter](https://github.com/mitchins/smolrouter).
