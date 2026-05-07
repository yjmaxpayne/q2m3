.PHONY: help install install-dev install-gpu sync test test-fast test-cov test-collect format format-check lint pre-commit docs docs-clean docs-doctest changelog-dry changelog build version release-check clean run-example

DEV_EXTRAS := --extra dev --extra catalyst --extra solvation --extra viz
DOCS_EXTRAS := --extra docs --extra catalyst --extra solvation --extra viz

help:
	@echo "Available commands:"
	@echo "  make sync          - Sync development dependencies with uv (recommended)"
	@echo "  make install       - Install core dependencies with uv"
	@echo "  make install-dev   - Install non-GPU development dependencies with uv"
	@echo "  make install-gpu   - Install development dependencies plus GPU support with uv"
	@echo "  make test          - Run test suite"
	@echo "  make test-fast     - Run fast tests (skip slow H3O+ tests)"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make test-collect  - Verify pytest collection"
	@echo "  make format        - Format code with black"
	@echo "  make format-check  - Check formatting with black"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make pre-commit    - Run pre-commit checks"
	@echo "  make docs          - Build Sphinx HTML documentation"
	@echo "  make docs-clean    - Remove Sphinx build output"
	@echo "  make docs-doctest  - Run Sphinx doctest builder"
	@echo "  make changelog-dry - Preview Commitizen changelog output"
	@echo "  make changelog     - Update CHANGELOG.md with Commitizen"
	@echo "  make build         - Build source and wheel distributions"
	@echo "  make version       - Print the runtime q2m3 version"
	@echo "  make release-check - Run release validation checks"
	@echo "  make clean         - Clean temporary files"
	@echo "  make run-example   - Run example calculation"

sync:
	uv sync $(DEV_EXTRAS)

install:
	uv sync --no-dev

install-dev:
	uv sync $(DEV_EXTRAS)

install-gpu:
	uv sync $(DEV_EXTRAS) --extra gpu

test:
	uv run $(DEV_EXTRAS) pytest tests/ -v

test-fast:
	uv run $(DEV_EXTRAS) pytest tests/ -v -m "not slow"

test-cov:
	uv run $(DEV_EXTRAS) pytest tests/ --cov=src/q2m3 --cov-report=html --cov-report=term

test-collect:
	uv run $(DEV_EXTRAS) pytest tests/ --collect-only -q

format:
	uv run --extra dev black src/ tests/ --line-length 100

format-check:
	uv run --extra dev black --check src/ tests/ --line-length 100

lint:
	uv run --extra dev ruff check src/ tests/

pre-commit:
	uv run --extra dev pre-commit run --all-files

docs:
	uv run $(DOCS_EXTRAS) sphinx-build -W --keep-going -b html doc/source doc/build/html

docs-clean:
	$(MAKE) -C doc clean

docs-doctest:
	uv run $(DOCS_EXTRAS) sphinx-build -W --keep-going -b doctest doc/source doc/build/doctest

changelog-dry:
	uv run --extra dev cz changelog --dry-run

changelog:
	uv run --extra dev cz changelog

build:
	uv build

version:
	uv run --extra dev python -c "import q2m3; print(q2m3.__version__)"

release-check:
	uv run --extra dev cz check --rev-range origin/main..HEAD
	uv run --extra dev cz changelog --dry-run
	uv build
	uv run --extra dev python -c "import importlib.metadata as md, importlib.util, q2m3; spec = importlib.util.spec_from_file_location('q2m3_docs_conf', 'doc/source/conf.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print(f\"metadata={md.version('q2m3')} runtime={q2m3.__version__} docs={mod.release}\")"
	uv run --extra dev pre-commit run check-yaml --all-files
	uv run --extra dev ruff check src/ tests/
	uv run --extra dev black --check src/ tests/ --line-length 100
	uv run $(DEV_EXTRAS) pytest tests/ --collect-only -q

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
	@echo "Running H2 QPE validation..."
	@uv run python examples/h2_qpe_validation.py
