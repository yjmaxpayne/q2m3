# SQD development and verification tools

These versioned tools complement the [SQD tutorials](../../examples/sqd/README.md).
Run from the checkout root after `uv sync --frozen --extra sqd --extra dev`.
Set OMP/MKL/OPENBLAS/NUMEXPR threads to 1 before starting Python. Outputs are
explicit run directories (typically ignored `tmp/` or `data/output/`). No old
temporary environment or ignored dev prototype is required.

| Tool | Purpose | Command |
|---|---|---|
| `molecular_benchmark.py` | Molecular/resource audit, same original parameters | `python -m tools.sqd.molecular_benchmark --system h2 --output tmp/audit-h2` |
| `calibrate_resources.py` | Resource model fit/holdout campaign | `python -m tools.sqd.calibrate_resources --help` |
| `compare_fixed_space.py` | Fixed determinant-space controls | `python -m tools.sqd.compare_fixed_space --help` |
| `orbital_basis_scan.py` | Orbital basis research | `python -m tools.sqd.orbital_basis_scan --help` |
| `connectivity_comparison.py` | Connectivity reachability research | `python -m tools.sqd.connectivity_comparison --help` |
| `ci_profile.py` | Actual installed-extra profile and mandatory solver nodes | `python -m tools.sqd.ci_profile --profile sqd` |
| `check_showcase.py` | Saved scientific/CSV/figure acceptance | `python -m tools.sqd.check_showcase --help` |
| `verify_showcase.py` | Complete-process independent RSS/time monitor | `python -m tools.sqd.verify_showcase --output tmp/monitor -- python -m examples.sqd.h2_ground_state` |
| [replay/](replay/README.md) | Historical dependency probes with checksummed files | `python -m tools.sqd.replay.replay --probe h2 --output tmp/replay-h2.json` |

`molecular_benchmark --system` accepts h2, h3o, glycine and n2; it replaces the three
retired forwarding wrappers and the old n2-named driver. Audit scope and registered
resource certification are unchanged. Full campaigns may take hours; inspect
`--help` before launching. The [replay manifest](replay/manifest.json) records
original source hashes and checksums for the bundled files.

Validation: `python -m pytest -o addopts='' tests/examples tests/sqd`.
Quality checks include `src/ tests/ examples/ tools/`. Do not commit generated output.
