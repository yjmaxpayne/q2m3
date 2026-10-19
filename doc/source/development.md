# Development

q2m3 uses a src-layout package, `uv` for dependency management, Ruff for linting,
Black for formatting, and pytest for tests.

## Environment Setup

```bash
uv sync --frozen --extra dev --extra sqd --extra catalyst --extra solvation --extra viz
source .venv/bin/activate
```

Install only the extras required by your task:

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

English sources remain in `doc/source/`. Chinese translations, including API
text extracted from source docstrings, remain in
`doc/locales/zh_CN/LC_MESSAGES/`. Keep source docstrings in English.

When you change documentation or an API docstring:

1. Edit and simplify the English source.
2. Extract messages and update the translation catalogs.
3. Review the Chinese translations. Preserve conditions, units, formulas,
   reference targets, and negative or inconclusive results.
4. Build and check both languages.

```bash
make docs-gettext
uv run python doc/translations.py update
# Review and translate the updated PO files before the check.
uv run python doc/translations.py check
make docs
make docs-doctest
```

`make docs` extracts messages, checks translations, and builds both languages
with warnings treated as errors. It also checks the generated site.
English remains the default at `/q2m3/`. Chinese is at `/q2m3/zh_CN/`.
Use the sidebar language links to open the corresponding page in either language.
The switch does not preserve query parameters or section anchors.

The project UI message list is `doc/ui-messages.json`. Update it when theme text changes.
`doc/site-baseline.json` records the starting commit and published page paths.
Change its page list only when you intend to add or remove published pages.

For an English-only preview, use:

```bash
make -C doc html
```

Commit PO catalogs. POT, MO, HTML, and doctree files are build artifacts.
English HTML is written to `doc/build/html/`. Chinese HTML is written to its
`zh_CN/` subdirectory. The builds use separate doctrees.

Keep default documentation examples small. Put H3O+, 8-bit QPE, dynamic
Trotter scans, and memory profiles in optional diagnostic sections.
First-run tutorials should use the smaller examples.

## Contribution Notes

Check public documentation against source code, tests, maintained examples,
and the current README. If these sources conflict with older planning notes,
use current source code and tests.

Scientific claims should name the active space, qubit count, QPE precision,
Trotter settings, and unit conversion assumptions when those values affect the
interpretation.

Contributors should not commit generated coverage reports, Sphinx build output,
Catalyst IR cache files, benchmark outputs, or temporary profiling artifacts.
