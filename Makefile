.PHONY: help install install-dev install-gpu test test-cov format lint clean run-example

help:
	@echo "Available commands:"
	@echo "  make install       - Install core dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make install-gpu   - Install GPU support"
	@echo "  make test          - Run test suite"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make clean         - Clean temporary files"
	@echo "  make run-example   - Run example calculation"

install:
	pip install --upgrade pip
	pip install -e .

install-dev:
	pip install --upgrade pip
	pip install -e ".[dev]"

install-gpu:
	pip install --upgrade pip
	pip install -e ".[gpu]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src/q2m3 --cov-report=html --cov-report=term

format:
	black src/ tests/ --line-length 100

lint:
	ruff check src/ tests/

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
	@python examples/h3o_basic.py