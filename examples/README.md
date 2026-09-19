# Examples by capability

Run commands from the checkout root. Start small, then follow each chapter's
learning order. The new SQD tutorials use `python -m examples.…`; no installation
of this examples directory is required. Each chapter documents inputs, outputs,
dependencies and cost. Numerical outputs are generated locally, not checked in.

| Capability | Start here | Progression |
|---|---|---|
| [QPE](qpe/README.md) | H₂ validation | H₂ and H₃O⁺ resolution benchmarks |
| [SQD](sqd/README.md) | H₂ ground state | Glycine CAS(6,6) → (8,8) → (10,10), five seeds, integral adapter |
| [QM/MM](qmmm/README.md) | Fixed-MO embedding | Water and glycine SQD; existing QPE–MC workflows |
| [Resources](resources/README.md) | H₂ resource estimate | Multimolecule survey |
| [Performance](performance/README.md) | Catalyst benchmark | QPE memory, Trotter scans, IR compilation and correlation |

Use `uv sync --frozen --extra dev --extra sqd` for SQD, and add
`--extra catalyst --extra solvation` for QPE–MC/JIT work. Scientific SQD runs
require serial native libraries, set **before starting Python**:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
uv run --no-sync python -m examples.sqd.h2_ground_state
```

SQD outputs default to a new exclusive directory beneath `data/output/examples/`.
`--output` names a new directory; existing directories are rejected to preserve evidence.
Legacy QPE/MC/resource/performance scripts retain their established output locations.

Calibration, audits, orbital/connectivity studies and dependency replays live in
[tools/sqd](../tools/sqd/README.md). They answer validation questions; the tutorials
above explain the user workflow. The [SQD chapter](sqd/README.md#example-results)
also includes measured example results.
