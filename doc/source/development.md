# Development

q2m3 uses a src-layout package, `uv` for dependency management, Ruff for linting,
Black for formatting, and pytest for tests.

## Environment Setup

```bash
uv sync --extra dev --extra catalyst --extra solvation --extra viz
```

Use narrower extras when working on isolated parts of the project:

```bash
uv sync --extra dev
uv sync --extra catalyst --extra solvation
uv sync --extra docs --extra catalyst --extra solvation --extra viz
```

## Testing

Use the narrowest useful command first.

```bash
uv run pytest tests/test_basic.py -v
uv run pytest tests/solvation -v -m "not slow and not gpu"
uv run pytest tests/ --collect-only -q --no-cov
uv run pytest tests/ --cov=src/q2m3 --cov-report=term-missing
```

Registered markers include `slow`, `solvation`, `catalyst`, `gpu`, and `rdm`.
The standard CI path skips slow and GPU tests.

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
uv run --extra docs --extra catalyst --extra solvation --extra viz sphinx-build -W --keep-going -b html doc/source doc/build/html
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
