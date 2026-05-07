# Getting Started

This guide gets a new q2m3 checkout to a working H2 validation run without
starting the high-memory Catalyst diagnostics by accident.

## Requirements

q2m3 targets Python 3.11 or newer and uses `uv` for environment management.
The core package uses PySCF, PennyLane, NumPy, SciPy, Matplotlib, and Rich.
Optional workflows add Catalyst/JAX, GPU backends, and molecular visualization
tools.

| Need | Install command |
| --- | --- |
| Core package only | `uv sync` |
| Development tools | `uv sync --extra dev` |
| Catalyst solvation workflow | `uv sync --extra catalyst --extra solvation` |
| Documentation build | `uv sync --extra docs --extra catalyst --extra solvation --extra viz` |
| Full local development set | `uv sync --extra dev --extra catalyst --extra solvation --extra viz` |

```{note}
GPU support is optional. The `gpu` extra installs CUDA-oriented packages and
should only be used on machines with compatible NVIDIA drivers and CUDA
runtime support.
```

## Installation

```bash
git clone https://github.com/yjmaxpayne/q2m3.git
cd q2m3

uv sync --extra dev --extra catalyst --extra solvation --extra viz
```

`uv run` automatically uses the managed `.venv`, so activating the environment
is optional.

## First Validation

Start with the H2 examples. They are intentionally small and exercise the
same public APIs used by the larger workflows.

```bash
uv run python examples/h2_qpe_validation.py
uv run python examples/h2_resource_estimation.py
```

The maintained H2 validation script checks vacuum and MM-embedded Hamiltonians,
compares PySCF Hartree-Fock references with QPE estimates, and reports the
solvation stabilization in kcal/mol after explicit conversion from Hartree.

## Quick API Check

The package-level constants are ordinary Python floats, which makes them safe
to use in pure Python, NumPy, and Catalyst-facing code paths.

```{doctest}
>>> round(HARTREE_TO_KCAL_MOL * KCAL_TO_HARTREE, 12)
1.0
```

## Runtime And Memory Tiers

Do not run every script in `examples/` as a smoke test. Catalyst compile memory
scales with estimation wires, Trotter depth, and Hamiltonian term count.

| Tier | Scripts | Expected environment |
| --- | --- | --- |
| First run | `h2_qpe_validation.py`, `h2_resource_estimation.py` | CPU laptop or workstation |
| Standard MC | `h2_mc_solvation.py` | Catalyst/JAX installed; 8 GB+ RAM recommended |
| H3O+ MC | `h3o_mc_solvation.py` | Catalyst/JAX installed; 16 GB+ RAM recommended |
| High-memory diagnostics | `h3o_8bit_qpe_benchmark.py`, `h3o_dynamic_trotter_oom_scan.py`, `qpe_memory_profile.py` | 30 GB+ RAM recommended; use provided guards/options |

## Basic Solvation Run

```bash
uv run python examples/h2_mc_solvation.py
```

This runs a fixed-mode H2 MC solvation workflow with IR caching enabled. The
first run may compile Catalyst IR; later runs can reuse the cache when the
circuit structure is unchanged.

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Import error for `catalyst` or `jax` | Solvation extras are missing | Run `uv sync --extra catalyst --extra solvation` |
| Very slow first MC run | Catalyst is compiling QPE IR | Let the first compile finish, then reuse the cache |
| H3O+ example is killed or times out | H3O+ IR is much larger than H2 IR | Use the H2 examples first; lower Trotter depth or run on a larger machine |
| GPU device is not selected | CUDA, Lightning GPU, and JAX CUDA availability are separate checks | Inspect `q2m3.core.device_utils` and fall back to `lightning.qubit` or `default.qubit` |
| QPE energy differs from HF by a large amount | Low precision, phase wrapping, or Trotter error | Check estimation wires, `base_time`, energy shift, and Trotter depth before interpreting the value as chemistry |

## Next Steps

Read [](core-concepts.md) for the scientific model, then run the tutorials in
order from H2 QPE validation through the three-mode solvation comparison.
