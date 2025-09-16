.PHONY: help install test lint format clean dev-install pre-commit

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