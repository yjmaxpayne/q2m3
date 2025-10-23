# Quantum-QM/MM POC Framework

A proof-of-concept framework for hybrid quantum-classical QM/MM calculations targeting early fault-tolerant quantum computers (EFTQC) in the Context of Hybrid Quantum-Classical Computing.

## Overview

This POC demonstrates the feasibility of integrating Quantum Phase Estimation (QPE) algorithms with molecular mechanics environments for quantum chemistry simulations. The framework bridges PySCF classical computations with PennyLane quantum circuits to explore the potential of quantum computing in molecular systems.

## Key Features

- 🚀 Iterative QPE implementation with 5-10 iteration convergence
- 🔄 Seamless PySCF-PennyLane integration interface
- 💧 H3O+ in TIP3P water environment QM/MM calculations
- 📊 Mulliken charge analysis and ground state energy computation
- 🖥️ GPU-accelerated quantum circuit simulation (12-20 qubits)

## Quick Start

### Requirements

- Python >= 3.11
- CUDA >= 11.0 (optional for GPU acceleration)
- 16GB+ GPU memory (recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/quantum-qmmm/poc.git
cd qqm_mm_poc

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install core dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install GPU support (optional)
pip install -e ".[gpu]"
```

### Basic Usage

```python
from q2m3.core import QuantumQMMM
from q2m3.utils import load_xyz

# Load H3O+ structure
h3o_geom = load_xyz("data/h3o_plus.xyz")

# Configure QPE parameters
qpe_config = {
    "algorithm": "iterative",
    "iterations": 8,
    "mapping": "jordan_wigner",
    "system_qubits": 12,
    "error_tolerance": 0.005
}

# Run QM/MM calculation
qmmm = QuantumQMMM(
    qm_atoms=h3o_geom,
    mm_waters=8,
    qpe_config=qpe_config
)

results = qmmm.compute_ground_state()
print(f"Ground State Energy: {results['energy']} Hartree")
print(f"Mulliken Charges: {results['atomic_charges']}")
```

## Project Structure

```
qqm_mm_poc/
├── src/
│   └── q2m3/
│       ├── core/           # Core algorithm implementations
│       ├── interfaces/     # PySCF-PennyLane interfaces
│       └── utils/          # Utility functions
├── tests/                  # Test suite
├── data/                   # Data files
├── docs/                   # Documentation
└── examples/               # Example scripts
```

## Development

```bash
# Run tests
make test

# Format code
make format

# Lint code
make lint

# Clean temporary files
make clean
```

## Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Energy Accuracy | < 10-15 kcal/mol | POC validation phase |
| QPE Convergence | 5-10 iterations | Based on 2024 research |
| Qubit Requirements | 12-20 qubits | H3O+ active space |
| Computation Time | < 60 minutes | GPU simulator |

## Documentation

For detailed documentation, see:
- [API Reference](docs/api/)

## Contributing

Issues and pull requests are welcome. Please ensure all tests pass and code follows the project style guide.

## License

MIT License

## Citation

If you use q2m3 in your research, please cite:

```bibtex
@software{q2m3_2025,
  title = {q2m3: A Hybrid-Quantum Classical Framework for QM/MM Simulations},
  author = {Ye Jun <yjmaxpayne@hotmail.com>},
  year = {2025},
  url = {https://github.com/yjmaxpayne/q2m3}
}
```

```

## Contact

- Repository: https://github.com/yjmaxpayne/q2m3
- Issues: https://github.com/yjmaxpayne/q2m3/issues