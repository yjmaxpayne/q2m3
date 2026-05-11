# q2m3: A Framework for Quantum QM/MM Simulation Workflows

<p align="center">
  <img src="doc/source/_static/logo.svg" alt="q2m3 logo" width="360">
</p>

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://github.com/yjmaxpayne/q2m3/actions/workflows/docs.yml/badge.svg)](https://github.com/yjmaxpayne/q2m3/actions/workflows/docs.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://yjmaxpayne.github.io/q2m3/)
[![codecov](https://codecov.io/gh/yjmaxpayne/q2m3/graph/badge.svg)](https://codecov.io/gh/yjmaxpayne/q2m3)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20114945.svg)](https://doi.org/10.5281/zenodo.20114945)
[![PennyLane](https://img.shields.io/badge/PennyLane-%3E%3D0.44.0-01A982)](https://pennylane.ai/)
[![Catalyst](https://img.shields.io/badge/Catalyst-%3E%3D0.14.0-01A982)](https://github.com/PennyLaneAI/catalyst)
[![PySCF](https://img.shields.io/badge/PySCF-%3E%3D2.0.0-blue)](https://pyscf.org/)

q2m3 is a hybrid quantum-classical QM/MM proof-of-concept that couples PySCF,
PennyLane Quantum Phase Estimation (QPE), and explicit MM point-charge
environments inside one Python package.

This is research scaffolding, not production chemistry. The maintained entry
points are narrow on purpose: H2 first-run validation, Catalyst-backed
solvation Monte Carlo, and EFTQC resource estimation for active-space
Hamiltonians. Everything else in the repo is diagnostic.

## Overview

q2m3 connects four things that are usually scattered across separate scripts:

- PySCF molecular integrals and Hartree-Fock references
- PennyLane Hamiltonian conversion and real QPE circuits
- QM/MM point-charge embedding with explicit water environments
- EFTQC-oriented resource estimation, profiling, and solvation workflows

The repository is tiered. Start with H2 validation on a laptop, then move to
Catalyst-backed Monte Carlo solvation or heavier H3O+ diagnostics when you
need them.

See also:

- [Documentation site](https://yjmaxpayne.github.io/q2m3/) — full rendered docs with search and navigation
- [Getting started](https://yjmaxpayne.github.io/q2m3/getting-started.html)
- [Core concepts](https://yjmaxpayne.github.io/q2m3/core-concepts.html)
- [Examples](examples/README.md)
- [API reference index](https://yjmaxpayne.github.io/q2m3/api-reference/index.html)
- [Development guide](doc/source/development.md) (contributor-facing, repo-local)

## Why q2m3

- The scientific boundary stays explicit. Energies live in Hartree internally,
  active spaces are part of the workflow configuration, and QPE outputs do
  not bleed into EFTQC resource-planning outputs.
- The QPE path runs real PennyLane circuits, not classical placeholders. A
  fallback path exists for data-pipeline validation when a circuit-backed
  route is the wrong tool for a given check.
- Solvation gets treated as a workflow problem as much as a circuit problem.
  Catalyst compile-once/run-many paths for Monte Carlo sampling and profiling
  ship with the package.
- First-run material and high-memory diagnostics live on separate tracks.
  This README stays short on purpose; detailed script coverage sits in
  [examples/README.md](examples/README.md) and the Sphinx docs.

## Install

q2m3 targets Python 3.11+ and is published on PyPI as
[`q2m3`](https://pypi.org/project/q2m3/). Pick the path that matches what you
want to do.

### From PyPI (recommended for library use)

Use this path when you want to import `QuantumQMMM` or `estimate_resources`
from your own code without cloning the repository.

```bash
# Core package only
pip install q2m3

# Solvation workflows with Catalyst/JAX
pip install "q2m3[catalyst,solvation]"

# Full optional set used by the maintained examples and docs
pip install "q2m3[catalyst,solvation,viz]"
```

The `[...]` extras syntax must be quoted in most shells (zsh/bash treat
unquoted brackets as glob patterns).

Equivalent commands with `uv`:

```bash
uv pip install q2m3
uv pip install "q2m3[catalyst,solvation,viz]"
```

Or, to track q2m3 inside an existing `uv`-managed project, add it as a
dependency from PyPI:

```bash
uv add q2m3
uv add "q2m3[catalyst,solvation]"
```

GPU support is optional. Add the `gpu` extra only on machines with compatible
NVIDIA drivers and a CUDA runtime: `pip install "q2m3[gpu]"`.

Verify the install:

```bash
python -c "import q2m3; print(q2m3.__version__)"
```

### From source (recommended for examples and development)

The maintained example scripts (`examples/`), the Sphinx docs, and the
development tooling live in the repository, so clone the source if you plan
to run those flows or contribute changes. The source workflow uses `uv` for
environment management.

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

To reproduce a tagged release, check out the tag before syncing:

```bash
git checkout v0.1.1
uv sync --extra dev --extra catalyst --extra solvation --extra viz
```

## First Run

The maintained smoke-test scripts live under `examples/` in the repository, so
this section assumes the [from-source install](#from-source-recommended-for-examples-and-development).
If you installed q2m3 from PyPI, jump to [Python API](#python-api) for an
equivalent inline snippet.

Start with the maintained H2 smoke path.

```bash
uv run python examples/h2_qpe_validation.py
uv run python examples/h2_resource_estimation.py
uv run python examples/full_oneelectron_embedding.py
```

Then read the tiered guides before moving to heavier workflows:

- [examples/README.md](examples/README.md) for the maintained script matrix
- [Getting started](https://yjmaxpayne.github.io/q2m3/getting-started.html) for
  install and runtime tiers
- [Core concepts](https://yjmaxpayne.github.io/q2m3/core-concepts.html) for the
  QM/MM, QPE, and solvation model

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

The pipeline goes:

1. Build a QM/MM system and MM point-charge environment.
2. Convert the PySCF molecular problem into a PennyLane-compatible Hamiltonian.
3. Run real QPE when the circuit-backed path is available, or fall back to
   the classical route for data-pipeline checks.
4. Measure derived quantities like 1-RDM observables and Mulliken charges.
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

- [Core concepts](https://yjmaxpayne.github.io/q2m3/core-concepts.html)
- [Architecture](https://yjmaxpayne.github.io/q2m3/architecture.html)
- [API reference](https://yjmaxpayne.github.io/q2m3/api-reference/index.html)

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

These are EFTQC resource estimates, not runtime benchmarks. They answer
"what does this active-space Hamiltonian imply for logical qubits and
non-Clifford cost?" — not "how long will this local Catalyst compile take?"

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
resource planning, or QM/MM + QPE research scaffolding. It is not a
production-ready general-purpose QM/MM engine.

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

If you use q2m3 in research, please cite the Zenodo archive. The concept
DOI [10.5281/zenodo.20114945](https://doi.org/10.5281/zenodo.20114945)
always resolves to the latest published release; switch to a
version-specific DOI (for example
[10.5281/zenodo.20114946](https://doi.org/10.5281/zenodo.20114946) for
`v0.1.1`) when you need to reference the exact code that produced a
result. GitHub's "Cite this repository" button reads `CITATION.cff`,
which lists both DOIs.

```bibtex
@software{q2m3_2026,
  title     = {q2m3: A Hybrid Quantum-Classical Framework for QM/MM Simulations},
  author    = {Ye Jun},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20114945},
  url       = {https://doi.org/10.5281/zenodo.20114945}
}
```

## License

q2m3 is released under the [MIT License](LICENSE).
