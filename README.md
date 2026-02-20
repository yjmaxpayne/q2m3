# q2m3: Quantum-QM/MM Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PennyLane](https://img.shields.io/badge/PennyLane-%3E%3D0.44.0-01A982?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0wIDE4Yy00LjQxIDAtOC0zLjU5LTgtOHMzLjU5LTggOC04IDggMy41OSA4IDgtMy41OSA4LTggOHoiLz48L3N2Zz4=)](https://pennylane.ai/)
[![PySCF](https://img.shields.io/badge/PySCF-%3E%3D2.0.0-blue)](https://pyscf.org/)

> **MVP Status** - Hybrid Quantum-Classical QM/MM for Early Fault-Tolerant Quantum Computers (EFTQC)

A proof-of-concept framework for hybrid quantum-classical QM/MM (Quantum Mechanics/Molecular Mechanics) calculations. The framework bridges PySCF classical computations with PennyLane quantum circuits using Quantum Phase Estimation (QPE) algorithms.

## Overview

q2m3 demonstrates the integration of QPE algorithms with molecular mechanics environments for quantum chemistry simulations. The framework implements **real quantum circuits** with Trotter time evolution, MM point charge embedding, and quantum 1-RDM measurement.

**Key Capabilities:**

- Standard QPE circuit with Trotter time evolution (`qml.TrotterProduct`)
- PySCF to PennyLane Hamiltonian conversion with MM embedding
- QM/MM system setup with TIP3P water solvation
- GPU acceleration via `lightning.gpu` device (4.3x speedup)
- Catalyst `@qjit` JIT compilation support
- Quantum 1-RDM measurement for Mulliken population analysis
- EFTQC resource estimation (Toffoli gates, logical qubits, 1-norm)
- Circuit visualization via `qml.draw(decimals=None, level=0)`
- Solvation effect analysis (vacuum vs explicit MM embedding)

## Quick Start

### Requirements

- Python >= 3.11
- CUDA >= 12.0 (optional, for GPU acceleration)
- uv package manager (recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/yjmaxpayne/q2m3.git
cd q2m3

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync all dependencies (recommended)
uv sync --all-extras

# Or install specific extras:
uv sync                    # Core only
uv sync --extra dev        # Development tools
uv sync --extra gpu        # GPU support (requires NVIDIA GPU)
uv sync --extra catalyst   # Catalyst JIT support
```

### Run Examples

```bash
# Activate virtual environment
source .venv/bin/activate

# Run minimal validation (H2 + TIP3P waters, ~30s)
python examples/h2_qpe_validation.py

# Run full demo (H3O+ + 8 TIP3P waters, ~3min)
python examples/h3o_qpe_full_demo.py
```

### Basic Usage

```python
import numpy as np
from q2m3.core import QuantumQMMM
from q2m3.core.qmmm_system import Atom

# Define H3O+ geometry (Angstrom)
h3o_atoms = [
    Atom("O", np.array([0.0, 0.0, 0.0]), charge=-2.0),
    Atom("H", np.array([0.96, 0.0, 0.0]), charge=1.0),
    Atom("H", np.array([-0.48, 0.831, 0.0]), charge=1.0),
    Atom("H", np.array([-0.48, -0.831, 0.0]), charge=1.0),
]

# Configure QPE parameters
qpe_config = {
    "use_real_qpe": True,
    "n_estimation_wires": 4,      # Precision bits
    "base_time": "auto",          # Auto-computed to avoid phase overflow
    "n_trotter_steps": 10,
    "n_shots": 100,
    "active_electrons": 4,        # Active space
    "active_orbitals": 4,
    "device_type": "auto",        # Auto-select best device
}

# Run QM/MM calculation with solvation
qmmm = QuantumQMMM(
    qm_atoms=h3o_atoms,
    mm_waters=8,
    qpe_config=qpe_config,
)

results = qmmm.compute_ground_state()
print(f"Ground State Energy: {results['energy']:.6f} Hartree")
print(f"HF Reference Energy: {results['energy_hf']:.6f} Hartree")

# Visualize circuits
circuits = qmmm.draw_circuits()
print(circuits["qpe"])  # QPE circuit diagram
print(circuits["rdm"])  # RDM measurement circuit
```

## Architecture

```
QuantumQMMM (main interface)
    |
    +-- QMMMSystem (system builder)
    |       +-- QM region: molecular atoms
    |       +-- MM region: TIP3P water molecules
    |       +-- Point charge embedding
    |
    +-- QPEEngine (quantum algorithm)
    |       +-- HF state preparation (X gates, Catalyst-compatible)
    |       +-- Controlled Trotter evolution (qml.TrotterProduct)
    |       +-- Inverse QFT (qml.adjoint(qml.QFT))
    |       +-- Phase-to-energy extraction
    |       +-- Device selection (GPU/CPU auto)
    |
    +-- RDMEstimator (quantum measurement)
    |       +-- Batch Pauli expectation values
    |       +-- 1-RDM reconstruction (diagonal + off-diagonal)
    |       +-- Active space MO-to-AO transformation
    |
    +-- PySCFPennyLaneConverter (interface layer)
            +-- Vacuum Hamiltonian (qml.qchem)
            +-- MM correction terms (PySCF -> Pauli operators)
            +-- HF state generation
```

## QPE Circuit Implementation

The QPE circuit follows the standard structure:

```
Estimation Register (n_est qubits):
|0> --H--[ctrl-U^8]--[ctrl-U^4]--[ctrl-U^2]--[ctrl-U^1]--QFT†--Sample
          |          |          |          |
System Register (n_sys qubits):
|HF> --------------------------------------------------------
```

**Circuit Components:**

1. **Initial State Preparation**: HF reference state via `qml.BasisState` (now Catalyst-compatible) or explicit X gates (legacy workaround)
2. **Hadamard Gates**: Superposition on estimation qubits
3. **Controlled Time Evolution**: `qml.ctrl(qml.adjoint(qml.TrotterProduct))` for U^(2^k)
4. **Inverse QFT**: `qml.adjoint(qml.QFT)` for phase readout
5. **Measurement**: Sample estimation register, extract most frequent phase

**Quantum Resources (H3O+ with active space):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Active Space | 4e, 4o | 4 electrons in 4 spatial orbitals |
| System Qubits | 8 | Spin orbitals (4 orbitals x 2 spins) |
| Estimation Qubits | 4 | Precision bits for phase readout |
| Total Qubits | **12** | System + estimation registers |
| Qubit Mapping | Jordan-Wigner | Fermion-to-qubit encoding |
| Trotter Steps | 10 | Time evolution accuracy |

## Device Selection

The `device_type` parameter controls quantum device selection:

| Value | Backend | Performance | Use Case |
|-------|---------|-------------|----------|
| `"auto"` | Best available | Optimal | **Recommended** |
| `"lightning.gpu"` | NVIDIA GPU | Fastest (4.3x) | Large circuits, GPU available |
| `"lightning.qubit"` | CPU (optimized) | Fast (2.7x) | CPU-only, Catalyst JIT |
| `"default.qubit"` | CPU (standard) | Baseline | Development, debugging |

**Performance Benchmark** (H3O+ + 8 waters, 12 qubits, 10 Trotter steps):

| Configuration | Device | Time | Energy (Ha) |
|---------------|--------|------|-------------|
| Standard QPE | lightning.gpu | ~28s | -76.509220 |
| Catalyst QPE | lightning.qubit | ~78s | -76.509220 |
| Standard QPE | default.qubit | ~120s | -76.509220 |

**Note:** Catalyst `@qjit` now supports `lightning.gpu` as of PennyLane Lightning 0.44.0. GPU acceleration is available for Catalyst JIT compilation. See [Known Issues](#known-issues) for details.

**Important:** Catalyst incurs significant compilation overhead for single-shot QPE. See [Catalyst Performance Guidelines](#catalyst-qjit-performance-guidelines) for when to use Catalyst.

## Catalyst @qjit Performance Guidelines

Catalyst's `@qjit` JIT compilation provides **~36x faster execution** but incurs **~7463x slower compilation** overhead. Understanding when to use Catalyst is critical for optimal performance.

### Performance Characteristics (H3O+, 12 qubits)

| Stage | Standard PennyLane | Catalyst @qjit | Ratio |
|-------|-------------------|----------------|-------|
| Circuit Build | 0.009s | 67.163s | ~7463x slower |
| Circuit Execution | 23.289s | 0.652s | 36x faster |
| **Total (single-shot)** | 23.298s | 67.815s | **2.9x slower** |

### When to Use Catalyst

| Use Case | Recommendation | Expected Performance |
|----------|----------------|---------------------|
| Single QPE execution | **Use standard PennyLane** | Baseline |
| Vacuum vs Solvated comparison | **Use standard PennyLane** | Avoid 3x compilation overhead |
| Iterative MC/MD sampling | **Use Catalyst with pre-compilation** | Reduced per-evaluation overhead |
| VQE/QAOA optimization | **Use Catalyst** | 10-50x speedup |

### Pre-compilation Strategy for Iterative Workflows

For workflows with multiple QPE evaluations on the same molecular system, use the modular `mc_solvation` framework:

```python
from examples.mc_solvation import MoleculeConfig, QPEConfig, SolvationConfig, run_solvation

config = SolvationConfig(
    molecule=MoleculeConfig(name="H2", symbols=["H", "H"], ...),
    qpe_config=QPEConfig(use_catalyst=True, qpe_interval=10),
    qpe_mode="vacuum_correction",  # Pre-compiled QPE, reused across MC steps
    n_waters=10,
    n_mc_steps=100,
)

result = run_solvation(config)
```

See [`examples/h2_mc_solvation.py`](examples/h2_mc_solvation.py) for complete implementation.

## EFTQC Resource Estimation

q2m3 provides quantum resource estimation using PennyLane's `DoubleFactorization` API to assess feasibility for Early Fault-Tolerant Quantum Computers (EFTQC).

**Key Results (Chemical Accuracy = 0.0016 Ha):**

| System | Basis | Toffoli Gates | Logical Qubits |
|--------|-------|---------------|----------------|
| H2 | STO-3G | ~1.2M | ~115 |
| H3O+ | STO-3G | ~143M | ~314 |
| H3O+ | 6-31G | ~494M | ~778 |

**Quick Start:**

```python
import pennylane as qml
import numpy as np
from pennylane.estimator import DoubleFactorization

# Define molecule
mol = qml.qchem.Molecule(['O', 'H', 'H', 'H'], coords, charge=1, basis_name='sto-3g')
_, one, two = qml.qchem.electron_integrals(mol)()

# Estimate resources
algo = DoubleFactorization(one, two, error=0.0016)
print(f"Toffoli gates: {algo.gates:,}")
print(f"Logical qubits: {algo.qubits}")
```

**Documentation:**
- Complete API guide: [dev/reports/catalyst_qpe_compilation_overhead_analysis.md](dev/reports/catalyst_qpe_compilation_overhead_analysis.md)
- Pre-compilation optimization: [dev/reports/catalyst_qpe_precompilation_optimization_report.md](dev/reports/catalyst_qpe_precompilation_optimization_report.md)

## Examples

### Example 1: H2 QPE Validation

`examples/h2_qpe_validation.py` - Fast validation of QPE + MM embedding + Catalyst benchmark (~30s)

```bash
python examples/h2_qpe_validation.py
```

**Test System**: H2 molecule + TIP3P waters (8 qubits: 4 system + 4 estimation)

| Method | Vacuum (Ha) | Solvated (Ha) | Stabilization |
|--------|-------------|---------------|---------------|
| PySCF HF | -1.116759 | -1.116674 | -0.054 kcal/mol |
| QPE | -1.134209 | -1.134122 | -0.054 kcal/mol |

### Example 2: H3O+ QPE Full Demo

`examples/h3o_qpe_full_demo.py` - Complete workflow demonstration (~3min)

```bash
python examples/h3o_qpe_full_demo.py
```

**Test System**: H3O+ ion + 8 TIP3P waters (12 qubits)

**Demo Steps:**
1. System Configuration (QM/MM setup, device selection)
1.5. Circuit Visualization
2. EFTQC Resource Estimation
3. Classical HF Solvation Analysis
4. Standard QPE Solvation Analysis
5. Catalyst QPE Solvation Analysis
6. Results Comparison
7. Save Results (JSON output)

### Example 3: MC Solvation with Modular Framework

`examples/h2_mc_solvation.py` - Modular MC solvation with Catalyst @qjit optimization

```bash
python examples/h2_mc_solvation.py
```

**Test System**: H2 molecule + 10 TIP3P waters with Monte Carlo sampling

**Key Features:**
- Modular `mc_solvation` framework with configurable QPE modes
- 100 MC steps with QPE validation every 10 steps
- Full Catalyst integration: `for_loop`, `cond`, `pure_callback`, `debug.print`
- Rich console output with timing statistics

See [examples/README.md](examples/README.md) for detailed documentation.

## Project Structure

```
q2m3/
+-- src/q2m3/
|   +-- core/
|   |   +-- quantum_qmmm.py    # Main entry point
|   |   +-- qpe.py             # QPE engine (real quantum circuits)
|   |   +-- qmmm_system.py     # QM/MM system builder
|   |   +-- rdm.py             # 1-RDM quantum measurement
|   |   +-- device_utils.py    # Device selection utilities
|   +-- interfaces/
|   |   +-- pyscf_pennylane.py # PySCF-PennyLane converter + MM embedding
|   +-- sampling/
|   |   +-- mc_moves.py       # Monte Carlo move proposals
|   |   +-- metropolis.py     # Metropolis acceptance criterion
|   |   +-- mm_forcefield.py  # MM force field (TIP3P)
|   |   +-- water_molecule.py # Water molecule representation
|   +-- utils/
|       +-- io.py              # File I/O utilities
+-- dev/reports/               # Technical analysis reports
+-- tests/                     # Test suite
+-- examples/                  # Example scripts
|   +-- h2_qpe_validation.py          # H2 QPE validation + Catalyst benchmark
|   +-- h3o_qpe_full_demo.py          # H3O+ full demo
|   +-- h2_mc_solvation.py            # H2 MC solvation with QJIT
|   +-- h3o_mc_solvation.py           # H3O+ MC solvation
|   +-- h3op_demo/                     # Sub-modules for h3op demo
|   +-- mc_solvation/                  # Modular MC solvation framework
|   +-- data/output/                   # Example output files (JSON results)
+-- data/                      # Input data files
```

## Development

```bash
# Run tests
make test

# Run fast tests (skip slow H3O+ tests)
make test-fast

# Run tests with coverage
make test-cov

# Format code
make format

# Lint code
make lint

# Run example
make run-example
```

### Test Markers

```bash
pytest -m "not slow"      # Skip slow tests
pytest -m "not catalyst"  # Skip Catalyst tests
pytest -m "not gpu"       # Skip GPU tests
pytest -m "catalyst"      # Run only Catalyst tests
pytest -m "gpu"           # Run only GPU tests
```

## Dependencies

**Core:**
- `pyscf>=2.0.0` - Classical quantum chemistry
- `pennylane>=0.33.0` - Quantum circuits (tested with 0.44.0)
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing

**Optional:**
- `pennylane-lightning[gpu]` - GPU acceleration (requires cuQuantum)
- `pennylane-catalyst>=0.14.0` - JIT compilation
- `cupy-cuda12x` - CUDA support

## Known Issues

### Issue 1: BasisState + Controlled Operations under @qjit

**Status**: Fixed (as of PennyLane 0.44.0, Catalyst 0.14.0) | **Tracking**: [Catalyst #2235](https://github.com/PennyLaneAI/catalyst/issues/2235) - Closed

`qml.BasisState` now works correctly with `qml.ctrl()` under `@qjit`. Previous workaround (explicit X gates) can be kept or replaced with `qml.BasisState` for cleaner code.

### Issue 2: Catalyst @qjit + lightning.gpu Incompatibility

**Status**: Fixed (as of PennyLane Lightning 0.44.0) | **Tracking**: [pennylane-lightning #1298](https://github.com/PennyLaneAI/pennylane-lightning/pull/1298) - Merged

Catalyst `@qjit` now supports `lightning.gpu` for `qml.ctrl(qml.TrotterProduct)`. GPU acceleration is now available for Catalyst JIT compilation.

### Issue 3: Lightning Device + MM Hamiltonian Type

**Status**: Fixed

`qml.Hamiltonian` returns `LinearCombination` type which lightning devices don't support for controlled evolution. **Fix**: Use `qml.s_prod` + `qml.sum` to maintain `Sum` type (`pyscf_pennylane.py:316-340`).

See [examples/README.md](examples/README.md) for detailed diagnosis and workarounds.

## Limitations

1. **QPE Precision**: Limited estimation qubits (4) result in approximate energies. Production use requires more precision bits.

2. **HF-Level Only**: No electron correlation (CCSD, etc.). Only Hartree-Fock level accuracy.

3. **Active Space**: Full H3O+ requires 16 qubits with ~2000 Pauli terms. Active space approximation (4e, 4o = 8 qubits) used for simulation feasibility.

4. **Small Systems**: Validated only for H2 (4 qubits) and H3O+ (12 qubits).

## License

MIT License

## Citation

If you use q2m3 in your research, please cite:

```bibtex
@software{q2m3_2025,
  title = {q2m3: A Hybrid Quantum-Classical Framework for QM/MM Simulations},
  author = {Ye Jun <yjmaxpayne@hotmail.com>},
  year = {2025},
  url = {https://github.com/yjmaxpayne/q2m3}
}
```

## Contact

- Repository: https://github.com/yjmaxpayne/q2m3
- Issues: https://github.com/yjmaxpayne/q2m3/issues
