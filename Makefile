.PHONY: help install install-dev install-gpu sync test test-cov format lint clean run-example

help:
	@echo "Available commands:"
	@echo "  make sync          - Sync all dependencies with uv (recommended)"
	@echo "  make install       - Install core dependencies with uv"
	@echo "  make install-dev   - Install development dependencies with uv"
	@echo "  make install-gpu   - Install GPU support with uv"
	@echo "  make test          - Run test suite"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make clean         - Clean temporary files"
	@echo "  make run-example   - Run example calculation"

sync:
	uv sync

install:
	uv sync --no-dev

install-dev:
	uv sync --all-extras

install-gpu:
	uv sync --extra gpu

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=src/q2m3 --cov-report=html --cov-report=term

format:
	uv run black src/ tests/ --line-length 100

lint:
	uv run ruff check src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

run-example:
	@echo "Running H3O+ example calculation..."
	@uv run python examples/h3o_basic.py
