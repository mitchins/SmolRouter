# Development guide

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

This installs runtime dependencies (FastAPI, redis, google-genai, aiohttp) plus the development toolchain (pytest, flake8, ruff, pre-commit).

## Common tasks

```bash
make dev-install   # optional helper that runs pip install -e .[dev]
make test          # run the full pytest suite (160 tests)
make lint          # flake8 + ruff checks
make check         # lint + tests in one go
make pre-commit    # execute the configured pre-commit hooks
```

## Test suites

- **Unit tests** cover request routing, model remapping, quota logic, security guards, and provider integrations.
- **Integration tests** exercise the FastAPI app via ASGI clients, ensuring compatibility across OpenAI, Ollama, and Google GenAI paths.
- **Load tests** (`test_load.py`) provide a harness for stress-testing and require `aiohttp` when executed.

The official release build runs the entire suite; keep your branch green by mirroring the same `pytest` command locally or in CI.

## Coding standards

- Keep imports free of side-effect catching (no try/except around imports).
- Prefer small, focused modules; leverage the provider factory instead of hard-coding upstream behaviour.
- Update documentation in `docs/` when you add new configuration flags or observable metrics.

## Troubleshooting

- Missing dependencies (e.g., `redis`, `google.api_core`) indicate `pip install -e .[dev]` was skipped.
- Redis features default to `fakeredis`, so tests run without a live server. For manual testing, point `REDIS_URL` at your instance.
- If the Web UI cannot serve templates, confirm the package data section in `pyproject.toml` includes `smolrouter/templates`.
