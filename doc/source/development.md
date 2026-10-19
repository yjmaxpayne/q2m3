# Development

q2m3 uses a src-layout package, `uv` for dependency management, Ruff for linting,
Black for formatting, and pytest for tests.

## Environment Setup

```bash
uv sync --frozen --extra dev --extra sqd --extra catalyst --extra solvation --extra viz
source .venv/bin/activate
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

### Installed profiles

Run these commands from the checkout root. Use a separate environment for each
profile. The `core` environment must contain neither SQD nor Catalyst extras.
The `sqd` environment must contain SQD dependencies and must not contain Catalyst.
A full-extra environment cannot validate either isolation boundary.

Set these variables before Python starts:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cpu
UV_PROJECT_ENVIRONMENT=.cache/venvs/core uv sync --frozen --extra dev
UV_PROJECT_ENVIRONMENT=.cache/venvs/core uv run --no-sync pytest -n 0 --q2m3-profile core --collect-only --no-cov
UV_PROJECT_ENVIRONMENT=.cache/venvs/core uv run --no-sync pytest -n 0 --q2m3-profile core

UV_PROJECT_ENVIRONMENT=.cache/venvs/sqd uv sync --frozen --extra dev --extra sqd
UV_PROJECT_ENVIRONMENT=.cache/venvs/sqd uv run --no-sync pytest -n 0 --q2m3-profile sqd --collect-only --no-cov
UV_PROJECT_ENVIRONMENT=.cache/venvs/sqd uv run --no-sync pytest -n 0 --q2m3-profile sqd

UV_PROJECT_ENVIRONMENT=.cache/venvs/sqd-catalyst uv sync --frozen --extra dev --extra sqd --extra catalyst --extra solvation
UV_PROJECT_ENVIRONMENT=.cache/venvs/sqd-catalyst uv run --no-sync pytest -n 0 --q2m3-profile sqd-catalyst --collect-only --no-cov
UV_PROJECT_ENVIRONMENT=.cache/venvs/sqd-catalyst uv run --no-sync pytest -n 0 --q2m3-profile sqd-catalyst
```

Collection checks dependency isolation and test selection. It does not satisfy
the execution gate. Run each complete profile without a test-path restriction
to validate its required nodes and coverage. A passing subset is insufficient.
Profiles exclude slow and GPU tests by default. Non-Catalyst profiles also
exclude Catalyst and solvation tests.

For the optional serial slow run, use the provisioned `sqd-catalyst` environment
and the exported variables above:

```bash
UV_PROJECT_ENVIRONMENT=.cache/venvs/sqd-catalyst uv run --no-sync pytest -n 0 --q2m3-profile sqd-catalyst --q2m3-include-slow
```

This command still excludes GPU tests. Larger tutorial campaigns belong in the
manual slow-science workflow. Shared CI does not promise to regenerate local
calibration, historical G2B or replay studies, or placeholder experiment reports.

Shared artifact checks live in `tests/examples/test_sqd_showcase.py`.
They check saved fields, signed errors, controls, failures, and output directories.
`test_real_h2_tutorial_matches_independent_casci` checks the real H₂ tutorial.
`test_saved_showcase_directories` additionally requires `Q2M3_SHOWCASE_ROOT` with
the `ci-scan`, `ci-embedding`, and `ci-integrals` directories. Without that input,
the test skips. Collection alone does not validate those artifacts.

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

Contributors should check public documentation against source code, tests,
maintained examples, and the current README. Prefer current source and tests over older
planning notes when they conflict.

Scientific claims should name the active space, qubit count, QPE precision,
Trotter settings, and unit conversion assumptions when those values affect the
interpretation.

Contributors should not commit generated coverage reports, Sphinx build output,
Catalyst IR cache files, benchmark outputs, or temporary profiling artifacts.
