# Examples by capability

[English](README.md) | [简体中文](README_CN.md)

Run all commands from the checkout root. Start with a small example, then follow
the learning order of each chapter. The SQD and fragment tutorials use
`python -m examples.…`. You do not need to install this examples directory.
Each chapter documents its inputs, outputs, dependencies, and cost. The examples
generate numerical outputs locally. The repository does not track these outputs.

| Capability | Start here | Progression |
|---|---|---|
| [QPE](qpe/README.md) | H₂ validation | H₂ and H₃O⁺ resolution benchmarks |
| [SQD](sqd/README.md) | H₂ ground state | Glycine CAS(6,6) → (8,8) → (10,10), five seeds, integral adapter |
| [QM/MM](qmmm/README.md) | Fixed-MO embedding | Water and glycine SQD, existing QPE–MC workflows |
| [Fragments](fragments/README.md) | Water-dimer HF/MP2 | Full water calculation → crystal preparation → crystal energies → aggregate report |
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

## Fragment interaction workflow

The [fragment chapter](fragments/README.md) provides five modules with bundled inputs:

```bash
uv sync --frozen --extra dev --extra viz --extra sqd
export PYTHONPATH="$PWD/src"
uv run --no-sync python -m examples.fragments.water_dimer_classical
uv run --no-sync python -m examples.fragments.water_dimer_full
uv run --no-sync python -m examples.fragments.crystal_dimer_prepare
uv run --no-sync python -m examples.fragments.crystal_dimer_interaction
uv run --no-sync python -m examples.fragments.fragment_report
```

Use the serial thread settings above. `viz` supplies ASE for CIF input.
Crystal interaction and aggregate report entries default to classical mode.
Add `--mode full` to either entry for SQD, references, and accuracy acceptance.
Full mode checks subsystem errors as well as the interaction-energy error.
An accuracy failure preserves the results and returns a nonzero exit code.
The [dated measurements](fragments/README.md#measured-example-results) specify the geometry, basis, versions, and sampling settings.
The QPE resource tables do not represent executed QPE calculations.

## Output locations

By default, SQD and fragment examples write outputs to a new exclusive directory under
`data/output/examples/`. `--output` names a new directory. To preserve evidence,
these scripts reject existing directories. The legacy QPE, MC, resource, and
performance scripts keep their established output locations.

The tutorials above are the public workflow and do not require local
development tools. The [SQD chapter](sqd/README.md#example-results) also
includes measured example results.
