.PHONY: help install test test-fast test-unit-cov test-provider-contract-cov test-cov smoke-local lint format clean dev-install pre-commit

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

dev-install: ## Install development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	pytest -v

test-fast: ## Run tests with minimal output
	pytest -q

test-unit-cov: ## Run unit tests with coverage output
	pytest tests/unit/ --cov=smolrouter --cov-report=xml:coverage.xml --cov-report=term-missing -v --tb=short

test-provider-contract-cov: ## Run focused provider-contract coverage checks
	pytest tests/unit/test_architecture.py tests/unit/test_new_integrations.py tests/unit/test_app.py tests/unit/test_local_smoke.py tests/integration/test_logging_features.py --cov=smolrouter.providers --cov=smolrouter.google_genai_provider --cov=smolrouter.anthropic_provider --cov=smolrouter.interfaces --cov=smolrouter.request_metadata --cov=smolrouter.config_loading --cov=smolrouter.local_smoke --cov=smolrouter.app --cov-report=xml:coverage.xml --cov-report=term-missing -v --tb=short

test-cov: ## Run the full test suite with coverage output
	pytest tests/ --cov=smolrouter --cov-report=xml:coverage.xml --cov-report=term-missing -v --tb=short

smoke-local: ## Run a quick localhost end-to-end smoke flow against an OpenAI-compatible upstream
	python -m smolrouter.local_smoke

lint: ## Run linting tools
	ruff check smolrouter/
	# Run vulture only on the package to avoid scanning virtualenvs
	vulture smolrouter/ --min-confidence=80 --exclude .venv --exclude venv --exclude .conda --exclude .cache

format: ## Check code formatting (placeholder for black/isort if added later)
	@echo "No formatter configured yet. Consider adding black + isort"

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

check: lint test ## Run linting and tests

all: dev-install pre-commit-install lint test ## Setup everything and run checks