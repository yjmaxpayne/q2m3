# Examples by capability

Run all commands from the checkout root. Start with a small example, then follow
the learning order of each chapter. The new SQD tutorials use
`python -m examples.…`. You do not need to install this examples directory.
Each chapter documents its inputs, outputs, dependencies, and cost. The examples
generate numerical outputs locally. The repository does not track these outputs.

| Capability | Start here | Progression |
|---|---|---|
| [QPE](qpe/README.md) | H₂ validation | H₂ and H₃O⁺ resolution benchmarks |
| [SQD](sqd/README.md) | H₂ ground state | Glycine CAS(6,6) → (8,8) → (10,10), five seeds, integral adapter |
| [QM/MM](qmmm/README.md) | Fixed-MO embedding | Water and glycine SQD, existing QPE–MC workflows |
| [Resources](resources/README.md) | H₂ resource estimate | Multimolecule survey |
| [Performance](performance/README.md) | Catalyst benchmark | QPE memory, Trotter scans, IR compilation and correlation |

For SQD on Linux x86_64, run `uv sync --frozen --extra dev --extra sqd`. For
QPE–MC/JIT work, add `--extra catalyst --extra solvation`. Scientific SQD runs
require serial native libraries. Set the thread variables **before you start
Python**:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
uv run --no-sync python -m examples.sqd.h2_ground_state
```

By default, SQD writes outputs to a new exclusive directory under
`data/output/examples/`. `--output` names a new directory. To preserve evidence,
the SQD scripts reject existing directories. The legacy QPE, MC, resource, and
performance scripts keep their established output locations.

[tools/sqd](../tools/sqd/README.md) contains calibration, audits,
orbital/connectivity studies, and dependency replays. These tools answer
validation questions. The tutorials above explain the user workflow. The
[SQD chapter](sqd/README.md#example-results) also includes measured example
results.
