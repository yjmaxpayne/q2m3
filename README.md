# q2m3: Quantum-QM/MM Framework

A proof-of-concept framework for hybrid quantum-classical QM/MM (Quantum Mechanics/Molecular Mechanics) calculations targeting early fault-tolerant quantum computers (EFTQC).

## Overview

q2m3 demonstrates the integration of Quantum Phase Estimation (QPE) algorithms with molecular mechanics environments for quantum chemistry simulations. The framework bridges PySCF classical computations with PennyLane quantum circuits.

**Key Features:**

- Standard QPE circuit implementation with Trotter time evolution
- PySCF to PennyLane Hamiltonian conversion
- QM/MM system setup with TIP3P water solvation
- GPU acceleration via `lightning.gpu` device
- Optional Catalyst `@qjit` JIT compilation
- Quantum 1-RDM measurement for Mulliken population analysis
- Circuit visualization via `qml.draw()`

## Quick Start

### Requirements

- Python >= 3.11
- CUDA >= 12.0 (optional, for GPU acceleration)

### Installation

```bash
# Clone repository
git clone https://github.com/yjmaxpayne/q2m3.git
cd q2m3

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install core dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install GPU support (optional, requires NVIDIA GPU)
pip install -e ".[gpu]"

# Install Catalyst JIT support (optional)
pip install -e ".[catalyst]"
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
    "base_time": 0.1,
    "n_trotter_steps": 10,
    "n_shots": 100,
    "active_electrons": 4,        # Active space
    "active_orbitals": 4,
    "device_type": "auto",        # Auto-select best device
}

# Run QM/MM calculation
qmmm = QuantumQMMM(
    qm_atoms=h3o_atoms,
    mm_waters=8,
    qpe_config=qpe_config,
)

results = qmmm.compute_ground_state()
print(f"Ground State Energy: {results['energy']:.6f} Hartree")
print(f"HF Reference Energy: {results['energy_hf']:.6f} Hartree")

# Visualize circuits (optional)
circuits = qmmm.draw_circuits()
print(circuits["qpe"])  # QPE circuit diagram
print(circuits["rdm"])  # RDM measurement circuit
```

## Architecture

```
QuantumQMMM (main interface)
    |
    +-- QMMMSystem (system builder)
    |       +-- QM region: H3O+ atoms
    |       +-- MM region: TIP3P water molecules
    |
    +-- QPEEngine (quantum algorithm)
    |       +-- Standard QPE circuit
    |       +-- Trotter time evolution
    |       +-- Device selection (GPU/CPU)
    |       +-- Circuit visualization (qml.draw)
    |
    +-- PySCFPennyLaneConverter (interface layer)
            +-- Hamiltonian conversion
            +-- HF state generation
```

## QPE Circuit Implementation

The QPE circuit follows the standard structure:

1. **Initial State Preparation**: HF reference state via `qml.BasisState`
2. **Hadamard Gates**: Superposition on estimation qubits
3. **Controlled Time Evolution**: `qml.ctrl(qml.TrotterProduct)` for U^(2^k)
4. **Inverse QFT**: `qml.adjoint(qml.QFT)` for phase readout
5. **Measurement**: Sample estimation register

**Quantum Resources (H3O+ with active space):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Active Space | 4e, 4o | 4 electrons in 4 spatial orbitals |
| System Qubits | 8 | Spin orbitals (4 orbitals x 2 spins) |
| Estimation Qubits | 4 | Precision bits for phase readout |
| Total Qubits | 12 | System + estimation registers |

## Device Selection

The `device_type` parameter controls quantum device selection:

| Value | Backend | Description |
|-------|---------|-------------|
| `"auto"` | Best available | GPU > lightning.qubit > default.qubit |
| `"lightning.gpu"` | NVIDIA GPU | Fastest (requires cuQuantum) |
| `"lightning.qubit"` | CPU (optimized) | High-performance CPU simulator |
| `"default.qubit"` | CPU (standard) | Standard PennyLane simulator |

**Note:** Catalyst `@qjit` works best with `lightning.qubit`. There are known compatibility issues when combining Catalyst with `lightning.gpu` for controlled Trotter operations.

## Project Structure

```
q2m3/
+-- src/q2m3/
|   +-- core/
|   |   +-- quantum_qmmm.py    # Main entry point
|   |   +-- qpe.py             # QPE engine
|   |   +-- qmmm_system.py     # QM/MM system builder
|   +-- interfaces/
|   |   +-- pyscf_pennylane.py # PySCF-PennyLane converter
|   +-- utils/
|       +-- io.py              # File I/O utilities
+-- tests/                     # Test suite
+-- examples/                  # Example scripts
    +-- h3o_quantum_qpe.py     # H3O+ QPE demo
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
- `pyscf>=2.0.0`
- `pennylane>=0.33.0`
- `numpy>=1.21.0,<2.0.0`
- `scipy>=1.7.0`

**Optional:**
- `pennylane-lightning[gpu]` - GPU acceleration
- `pennylane-catalyst>=0.5.0` - JIT compilation
- `cupy-cuda12x` - CUDA support

## Known Limitations

1. **QPE Precision**: The current implementation uses a limited number of estimation qubits (4), resulting in approximate energy estimates. For production use, more precision bits would be needed.

2. **Catalyst + GPU**: Combining `@qjit` with `lightning.gpu` has compatibility issues for `qml.ctrl(qml.TrotterProduct)`. Use `lightning.qubit` with Catalyst for best results.

3. **Active Space**: Full H3O+ requires 16 qubits with ~2000 Pauli terms. The implementation uses active space approximation (4e, 4o = 8 qubits) to make simulation feasible.

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
