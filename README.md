# q2m3: Quantum-Classical QM/MM with QPE

<p align="center">
  <img src="doc/source/_static/logo.svg" alt="q2m3 logo" width="360">
</p>

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20114945.svg)](https://doi.org/10.5281/zenodo.20114945)
[![PennyLane](https://img.shields.io/badge/PennyLane-%3E%3D0.44.0-01A982?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0wIDE4Yy00LjQxIDAtOC0zLjU5LTgtOHMzLjU5LTggOC04IDggMy41OSA4IDgtMy41OSA4LTggOHoiLz48L3N2Zz4=)](https://pennylane.ai/)
[![PySCF](https://img.shields.io/badge/PySCF-%3E%3D2.0.0-blue)](https://pyscf.org/)

q2m3 is a hybrid quantum-classical QM/MM proof-of-concept for coupling PySCF,
PennyLane Quantum Phase Estimation (QPE), and explicit MM point-charge
environments in a single Python package.

It is designed as a research and workflow framework, not a polished end-user
chemistry application. The maintained entry points focus on small first-run
validations, compile-once/run-many solvation experiments, and EFTQC resource
estimation for active-space Hamiltonians.

## Overview

q2m3 connects four things that are usually scattered across separate scripts:

- PySCF molecular integrals and Hartree-Fock references
- PennyLane Hamiltonian conversion and real QPE circuits
- QM/MM point-charge embedding with explicit water environments
- EFTQC-oriented resource estimation, profiling, and solvation workflows

The repository is intentionally tiered. Start with H2 validation on a laptop,
then move to Catalyst-backed Monte Carlo solvation or heavier H3O+ diagnostics
only when you need them.

Useful next stops:

- [Getting started](doc/source/getting-started.md)
- [Core concepts](doc/source/core-concepts.md)
- [Examples](examples/README.md)
- [API reference index](doc/source/api-reference/index.rst)
- [Development guide](doc/source/development.md)

## Why q2m3

- It keeps the scientific boundary explicit. Energies stay in Hartree
  internally, active spaces are part of the workflow configuration, and QPE
  results stay separated from EFTQC resource-planning outputs.
- It exposes a real PennyLane QPE path instead of reducing the repository to
  classical placeholders. The package also keeps a fallback path for data-pipeline
  validation when a circuit-backed route is not the right tool for a given check.
- It treats solvation as a workflow problem as well as a circuit problem. The
  package includes compile-once/run-many Catalyst-oriented paths for Monte Carlo
  sampling and profiling.
- It keeps first-run and high-memory tasks separate. The root README stays small,
  while detailed script coverage lives in [examples/README.md](examples/README.md)
  and the Sphinx docs.

## Install

q2m3 targets Python 3.11+ and uses `uv` for environment management.

```bash
git clone https://github.com/yjmaxpayne/q2m3.git
cd q2m3
```

Choose the narrowest install that matches what you want to do:

```bash
# Core package only
uv sync

# Development workflow
uv sync --extra dev

# Solvation workflows with Catalyst/JAX
uv sync --extra catalyst --extra solvation

# Full local development set used by the docs and maintained examples
uv sync --extra dev --extra catalyst --extra solvation --extra viz
```

Documentation-only dependencies are separate:

```bash
uv sync --extra docs --extra catalyst --extra solvation --extra viz
```

GPU support is optional. Add the `gpu` extra only on machines with compatible
NVIDIA drivers, CUDA runtime support, and a reason to run GPU-backed circuits.

## First Run

Start with the maintained H2 smoke path.

```bash
uv run python examples/h2_qpe_validation.py
uv run python examples/h2_resource_estimation.py
uv run python examples/full_oneelectron_embedding.py
```

Then read the tiered guides before moving to heavier workflows:

- [examples/README.md](examples/README.md) for the maintained script matrix
- [doc/source/getting-started.md](doc/source/getting-started.md) for install and
  runtime tiers
- [doc/source/core-concepts.md](doc/source/core-concepts.md) for the QM/MM, QPE,
  and solvation model

If you want a Monte Carlo solvation smoke test after the H2 path:

```bash
uv run python examples/h2_mc_solvation.py
```

Treat `examples/h3o_mc_solvation.py`, the 8-bit benchmarks, and dynamic Trotter
scans as optional diagnostics, not as the default entry point.

## Python API

The main package entry point is `QuantumQMMM`. For direct EFTQC planning, use
`estimate_resources` from `q2m3.core`.

```python
import numpy as np

from q2m3.core import QuantumQMMM, estimate_resources
from q2m3.core.qmmm_system import Atom

qm_atoms = [
    Atom("H", np.array([0.0, 0.0, 0.0])),
    Atom("H", np.array([0.0, 0.0, 0.74])),
]

qmmm = QuantumQMMM(
    qm_atoms=qm_atoms,
    mm_waters=2,
    qpe_config={
        "use_real_qpe": True,
        "active_electrons": 2,
        "active_orbitals": 2,
        "n_estimation_wires": 4,
        "n_trotter_steps": 20,
        "device_type": "auto",
    },
)

result = qmmm.compute_ground_state(include_resource_estimation=True)
print(f"QPE energy: {result['energy']:.6f} Ha")
print(f"HF energy:  {result['energy_hf']:.6f} Ha")
print(result["eftqc_resources"].logical_qubits)

resources = estimate_resources(
    symbols=["H", "H"],
    coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
    basis="sto-3g",
)
print(resources.toffoli_gates)
```

For maintained, runnable examples of the public API rather than inline snippets,
use:

- [examples/h2_qpe_validation.py](examples/h2_qpe_validation.py)
- [examples/h2_resource_estimation.py](examples/h2_resource_estimation.py)
- [examples/full_oneelectron_embedding.py](examples/full_oneelectron_embedding.py)
- [examples/h2_mc_solvation.py](examples/h2_mc_solvation.py)

## How It Fits Together

At a high level, q2m3 moves through this stack:

1. Build a QM/MM system and MM point-charge environment.
2. Convert the PySCF molecular problem into a PennyLane-compatible Hamiltonian.
3. Run real QPE when the circuit-backed path is available, or use the fallback
   route for validation-oriented checks.
4. Measure derived quantities such as 1-RDM observables and Mulliken charges.
5. Optionally estimate EFTQC resources or reuse compiled circuits in solvation
   workflows.

The package layout mirrors that flow:

| Package area | Role |
|--------------|------|
| `q2m3.core` | QPE engine, QM/MM orchestration, RDM hooks, resource estimation |
| `q2m3.interfaces` | PySCF to PennyLane bridge and density-matrix conversions |
| `q2m3.solvation` | Monte Carlo solvation orchestration, circuit building, analysis |
| `q2m3.profiling` | Timing, memory, and Catalyst IR diagnostics |
| `q2m3.utils` | I/O and plotting helpers |

Read more in:

- [doc/source/core-concepts.md](doc/source/core-concepts.md)
- [doc/source/architecture.md](doc/source/architecture.md)
- [doc/source/api-reference/index.rst](doc/source/api-reference/index.rst)

## Runtime Notes

- `device_type="auto"` selects the best available PennyLane simulator for
  standard execution.
- Catalyst execution depends on the JAX backend and related optional
  dependencies, not just the base PennyLane install.
- Use Catalyst for compile-once/run-many solvation or profiling workflows, not
  for single QPE smoke tests where compile overhead dominates.
- `uv run` uses the managed `.venv`, so manual activation is optional.
- H2 first-run examples fit the intended smoke-test path. H3O+ examples are
  intentionally separated because Catalyst IR and memory pressure scale much
  more aggressively there.

## Resource Estimation

q2m3 exposes EFTQC planning through `q2m3.core.estimate_resources` and keeps a
checked-in survey at `data/output/qre_survey.csv` for the current small-molecule
matrix.

Current canonical survey rows:

| System | Active space | Logical qubits | Toffoli gates |
|--------|--------------|----------------|---------------|
| H2 `(2e,2o)` STO-3G | 4 system qubits | `115` | `1,224,608` |
| H3O+ `(4e,4o)` STO-3G | 8 system qubits | `131` | `6,511,085` |

These are EFTQC resource estimates, not runtime benchmarks. They help answer
questions such as "what does this active-space Hamiltonian imply for logical
qubits and non-Clifford cost?" rather than "how long will this local Catalyst
compile take?"

The maintained entry points are:

- [examples/h2_resource_estimation.py](examples/h2_resource_estimation.py)
- [examples/full_oneelectron_embedding.py](examples/full_oneelectron_embedding.py)
- [examples/resource_estimation_survey.py](examples/resource_estimation_survey.py)

### Fixed-MO Full-One-Electron Embedding

The resource-estimation API exposes explicit-MM embedding modes for fixed-MO
one-electron perturbations:

| Mode | One-electron MM terms | Boundary |
|------|------------------------|----------|
| `diagonal` | Adds only active-space `Delta h_pp` terms | Matches the current dynamic coefficient-update path |
| `full_oneelectron` | Adds the full active-space `Delta h_pq` matrix | Fixed operator support only |

Both modes keep the vacuum MO frame and vacuum two-electron tensor fixed. They
are resource-estimation and fixed-Hamiltonian operator tools, not relaxed
orbital, correlated-solvent, or polarizable-MM solvation energies.

```bash
uv run python examples/full_oneelectron_embedding.py
```

To reproduce a tagged release, cite the tag and run:

```bash
git checkout <release-tag>
uv sync --extra dev --extra catalyst --extra solvation --extra viz
uv run python examples/full_oneelectron_embedding.py
uv run pytest tests/examples/test_full_oneelectron_embedding.py -x -q -n 0
```

## Project Status

- **Stage**: alpha / proof-of-concept research framework
- **Validated first-run path**: H2 QPE validation and H2 resource estimation
- **Maintained heavier diagnostics**: H2 and H3O+ solvation workflows, IR-QRE
  studies, Catalyst profiling, and memory-guarded scans
- **QPE state**: real PennyLane QPE paths exist in the package; a classical
  fallback remains useful for POC and data-pipeline validation
- **Documentation strategy**: this README is the landing page; detailed runtime,
  example, and concept material lives in the linked docs rather than the root
  file

q2m3 is useful today if you are exploring workflow integration, active-space
resource planning, and QM/MM + QPE research scaffolding. It is not positioned
as a production-ready general-purpose QM/MM engine.

## Development

The project uses a `src/` layout, `uv` for dependency management, pytest for
tests, Ruff for linting, and Black for formatting.

```bash
uv sync --extra dev --extra catalyst --extra solvation --extra viz

# Optional: install GPU test/runtime extras for local CUDA machines.
uv sync --extra dev --extra catalyst --extra solvation --extra viz --extra gpu

uv run pytest tests/ -v
uv run pytest tests/ -v -m "not slow and not gpu"
uv run pytest tests/path/to/test_file.py -v -n 0

uv run ruff check src/ tests/
uv run black --check src/ tests/ --line-length 100

make docs
```

Useful references:

- [doc/source/development.md](doc/source/development.md)
- [AGENTS.md](AGENTS.md)
- [Makefile](Makefile)

Use the narrowest useful test command first, and avoid committing generated
coverage, build, benchmark, or temporary profiling output.
Pytest uses `pytest-xdist` by default for full-suite parallelism. Add `-n 0`
for single-file debugging or other narrow runs where worker startup dominates.

## Citation

If you use q2m3 in research, cite the repository or adapt the BibTeX entry
below to your publication format.

```bibtex
@software{q2m3_2026,
  title = {q2m3: A Hybrid Quantum-Classical Framework for QM/MM Simulations},
  author = {Ye Jun},
  year = {2026},
  url = {https://github.com/yjmaxpayne/q2m3}
}
```

## License

q2m3 is released under the [MIT License](LICENSE).
