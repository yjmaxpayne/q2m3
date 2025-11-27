# q2m3: Quantum-QM/MM Framework

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

# Run minimal validation (H2 + 2 TIP3P waters, ~30s)
python examples/h2_qpe_h2o_mm_minimal.py

# Run full demo (H3O+ + 8 TIP3P waters, ~3min)
python examples/h3op_qpe_h2o_mm_full.py
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

1. **Initial State Preparation**: HF reference state via explicit X gates (Catalyst-compatible workaround for BasisState + ctrl issue)
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

**Note:** Catalyst `@qjit` currently works best with `lightning.qubit`. There are known compatibility issues when combining Catalyst with `lightning.gpu` for `qml.ctrl(qml.TrotterProduct)`. See [Known Issues](#known-issues).

## EFTQC Resource Estimation

q2m3 provides quantum resource estimation using PennyLane's `DoubleFactorization` API to assess feasibility for Early Fault-Tolerant Quantum Computers (EFTQC).

```bash
# Run resource estimation demo
python examples/resource_estimation_demo.py
```

**Key Results (Chemical Accuracy = 0.0016 Ha):**

| System | Basis | Toffoli Gates | Logical Qubits |
|--------|-------|---------------|----------------|
| H2 | STO-3G | ~1.2M | ~115 |
| H3O+ | STO-3G | ~148M | ~314 |
| H3O+ | 6-31G | ~494M | ~778 |

**Quick Start:**

```python
import pennylane as qml
import numpy as np
from pennylane.resource import DoubleFactorization

# Define molecule
mol = qml.qchem.Molecule(['O', 'H', 'H', 'H'], coords, charge=1, basis_name='sto-3g')
_, one, two = qml.qchem.electron_integrals(mol)()

# Estimate resources
algo = DoubleFactorization(one, two, error=0.0016)
print(f"Toffoli gates: {algo.gates:,}")
print(f"Logical qubits: {algo.qubits}")
```

**Documentation:**
- Complete API guide: [docs/resource_estimation_api_research.md](docs/resource_estimation_api_research.md)
- Quick reference: [docs/resource_estimation_quickstart.md](docs/resource_estimation_quickstart.md)

## Examples

### Example 1: H2 + MM Water (Minimal Validation)

`examples/h2_qpe_h2o_mm_minimal.py` - Fast validation of QPE + MM embedding (~30s)

```bash
python examples/h2_qpe_h2o_mm_minimal.py
```

**Test System**: H2 molecule + 2 TIP3P waters (8 qubits: 4 system + 4 estimation)

| Method | Vacuum (Ha) | Solvated (Ha) | Stabilization |
|--------|-------------|---------------|---------------|
| PySCF HF | -1.116759 | -1.116674 | -0.054 kcal/mol |
| QPE | -1.134209 | -1.134122 | -0.054 kcal/mol |

### Example 2: H3O+ QPE Full Demo

`examples/h3op_qpe_h2o_mm_full.py` - Complete workflow demonstration (~3min)

```bash
python examples/h3op_qpe_h2o_mm_full.py
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

### Example 3: EFTQC Resource Estimation

`examples/resource_estimation_demo.py` - Quantum resource requirements analysis (~10s)

```bash
python examples/resource_estimation_demo.py
```

**Analysis:**
- H2 and H3O+ resource estimates (STO-3G, 6-31G)
- Error tolerance scaling (1×, 10×, 100× chemical accuracy)
- EFTQC feasibility assessment
- Resource-error trade-off analysis

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
|   +-- utils/
|       +-- io.py              # File I/O utilities
+-- docs/
|   +-- resource_estimation_api_research.md  # Complete API documentation
|   +-- resource_estimation_quickstart.md    # Quick reference guide
+-- tests/                     # Test suite
+-- examples/                  # Example scripts
|   +-- h2_qpe_h2o_mm_minimal.py     # H2 minimal validation
|   +-- h3op_qpe_h2o_mm_full.py      # H3O+ full demo
|   +-- resource_estimation_demo.py  # EFTQC resource analysis
+-- data/output/               # Output files (JSON results)
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
- `pennylane>=0.43.0` - Quantum circuits
- `numpy>=1.21.0,<2.0.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing

**Optional:**
- `pennylane-lightning[gpu]` - GPU acceleration (requires cuQuantum)
- `pennylane-catalyst>=0.13.0` - JIT compilation
- `cupy-cuda12x` - CUDA support

## Known Issues

### Issue 1: BasisState + Controlled Operations under @qjit

**Status**: Workaround applied | **Tracking**: [Catalyst #2235](https://github.com/PennyLaneAI/catalyst/issues/2235)

`qml.BasisState` combined with `qml.ctrl()` under `@qjit` produces incorrect quantum states. **Workaround**: Use explicit X gates for HF state preparation (`qpe.py:173-177`).

### Issue 2: Catalyst @qjit + lightning.gpu Incompatibility

**Status**: Workaround applied, fix in progress | **Tracking**: [pennylane-lightning #1298](https://github.com/PennyLaneAI/pennylane-lightning/pull/1298)

`qml.ctrl(qml.TrotterProduct)` with `lightning.gpu` under `@qjit` triggers custatevec error. **Workaround**: Auto-fallback to `lightning.qubit` for Catalyst execution.

### Issue 3: Lightning Device + MM Hamiltonian Type

**Status**: Fixed

`qml.Hamiltonian` returns `LinearCombination` type which lightning devices don't support for controlled evolution. **Fix**: Use `qml.s_prod` + `qml.sum` to maintain `Sum` type (`pyscf_pennylane.py:296-320`).

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
