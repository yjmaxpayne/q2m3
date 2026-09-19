# Development

q2m3 uses a src-layout package, `uv` for dependency management, Ruff for linting,
Black for formatting, and pytest for tests.

## Environment Setup

```bash
uv sync --frozen --extra dev --extra sqd --extra catalyst --extra solvation --extra viz
```

Use narrower extras when working on isolated parts of the project:

```bash
uv sync --extra dev
uv sync --frozen --extra dev --extra sqd
uv sync --extra catalyst --extra solvation
uv sync --frozen --extra docs --extra sqd --extra catalyst --extra solvation --extra viz
```

## Testing

Use the narrowest useful command first.

```bash
uv run pytest tests/test_basic.py -v
uv run pytest tests/solvation -v -m "not slow and not gpu"
uv run pytest tests/ --collect-only -q --no-cov
uv run pytest tests/ --cov=src/q2m3 --cov-report=term-missing
```

Registered markers include `slow`, `solvation`, `catalyst`, `gpu`, `rdm`, and `sqd`.
The standard CI path skips slow and GPU tests.

The SQD-only installed-profile check must run without Catalyst. Provision a
separate ignored environment so that a previously installed full environment
cannot contaminate the dependency check:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
UV_PROJECT_ENVIRONMENT=.cache/venvs/sqd uv sync --frozen --extra dev --extra sqd
UV_PROJECT_ENVIRONMENT=.cache/venvs/sqd uv run --no-sync python tools/sqd/ci_profile.py --profile sqd --collect-only
UV_PROJECT_ENVIRONMENT=.cache/venvs/sqd uv run --no-sync pytest -o addopts='' tests/sqd tests/examples -q
UV_PROJECT_ENVIRONMENT=.cache/venvs/sqd uv run --no-sync python -m examples.sqd.h2_ground_state
```

Larger tutorial campaigns are serial by design and belong in the manual
slow-science workflow. CI provisions the same dependency isolation through its
profile matrix.

## Linting And Formatting

```bash
uv run ruff check src/ tests/
uv run black --check src/ tests/ --line-length 100
uv run black src/ tests/ --line-length 100
```

For GitHub-facing changes, also run:

```bash
uv run pre-commit run check-yaml --all-files
```

## Documentation Workflow

Build the Sphinx site with warnings treated as errors:

```bash
make docs-clean docs
make docs-doctest
```

The equivalent explicit build command is:

```bash
uv run --extra docs --extra sqd --extra catalyst --extra solvation --extra viz sphinx-build -W --keep-going -b html doc/source doc/build/html
```

Documentation examples should be lightweight by default. H3O+, 8-bit QPE,
dynamic Trotter scans, and memory profiles belong in optional diagnostic
sections, not in first-run tutorials.

## Contribution Notes

Public documentation should be checked against source code, tests, maintained
examples, and the current README. Prefer current source and tests over older
planning notes when they conflict.

Scientific claims should name the active space, qubit count, QPE precision,
Trotter settings, and unit conversion assumptions when those values affect the
interpretation.

Generated coverage reports, Sphinx build output, Catalyst IR cache files,
benchmark outputs, and temporary profiling artifacts should not be committed.
