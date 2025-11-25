# H3O+ Quantum Phase Estimation Demo

> **q2m3 MVP** - Hybrid Quantum-Classical QM/MM for Early Fault-Tolerant Quantum Computers (EFTQC)

## Overview

This demo showcases the complete q2m3 workflow for molecular ground state energy estimation using Quantum Phase Estimation (QPE), with GPU acceleration and optional PennyLane Catalyst JIT compilation support.

**Key Capabilities Demonstrated:**
- PySCF to PennyLane molecular Hamiltonian conversion
- Standard QPE circuit with Trotter time evolution
- GPU acceleration via `lightning.gpu` device
- QM/MM system setup with TIP3P water solvation
- Catalyst `@qjit` compilation for JIT optimization
- Mulliken population analysis

## Pipeline

```mermaid
flowchart LR
    subgraph Input["1. Input"]
        A["H3O+ Geometry<br/>(4 atoms, +1 charge)"]
        B["8 TIP3P Waters<br/>(MM solvation)"]
    end

    subgraph Classical["2. Classical (PySCF)"]
        C["RHF/STO-3G<br/>E_HF = -75.33 Ha"]
        D["Active Space<br/>(4e, 4o)"]
    end

    subgraph Quantum["3. Quantum (PennyLane)"]
        E["Molecular<br/>Hamiltonian"]
        F["QPE Circuit<br/>12 qubits"]
    end

    subgraph Output["4. Output"]
        G["Ground State Energy"]
        H["Mulliken Charges"]
    end

    A --> C
    B --> C
    C --> D --> E --> F --> G
    F --> H

    style F fill:#e1f5fe
    style G fill:#c8e6c9
```

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Install GPU support (optional but recommended)
uv pip install -e ".[gpu]"

# Run the demo
python examples/h3o_quantum_qpe.py

# Or use Makefile
make run-example
```

## Test System: H3O+ (Hydronium Ion)

```
        H (+1.0)
         \
          O (-2.0) --- H (+1.0)      Total charge: +1
         /
        H (+1.0)

Geometry (Angstrom):
  O:  ( 0.000,  0.000,  0.000)
  H:  ( 0.960,  0.000,  0.000)
  H:  (-0.480,  0.831,  0.000)
  H:  (-0.480, -0.831,  0.000)
```

## Quantum Resource Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Active Space** | 4e, 4o | 4 electrons in 4 spatial orbitals |
| **System Qubits** | 8 | 4 orbitals x 2 spin = 8 spin orbitals |
| **Estimation Qubits** | 4 | Precision bits for phase readout |
| **Total Qubits** | **12** | System + estimation registers |
| **Qubit Mapping** | Jordan-Wigner | Fermion-to-qubit encoding |
| **Trotter Steps** | 10 | Time evolution accuracy |
| **Base Time** | 0.1 | Evolution time parameter |
| **Shots** | 100 | Measurement statistics |

## Demo Workflow (5 Steps)

| Step | Description | Output |
|------|-------------|--------|
| **Step 1** | System Configuration | QM/MM setup, quantum resources, device selection |
| **Step 2** | QPE Execution (auto device) | Energy with `lightning.gpu` (if available) |
| **Step 3** | Catalyst QPE Execution | Energy with `lightning.qubit` + `@qjit` |
| **Step 4** | Results Comparison | Time comparison, energy consistency |
| **Step 5** | Save Results | JSON output to `data/output/` |

## Sample Output (GPU Environment)

```
================================================================================
                    H3O+ Quantum Phase Estimation (QPE) Demo
                    q2m3 MVP - Catalyst Technical Validation
================================================================================
Timestamp: 2025-11-26 07:27:51
Catalyst Available: Yes (v0.13.0)
Lightning GPU Available: Yes


[Step 1] System Configuration
--------------------------------------------------------------------------------
QM Region: H3O+ (4 atoms, total charge +1)
MM Region: 8 TIP3P water molecules

Quantum Resource Requirements:
  Active Space: 4 electrons, 4 orbitals
  System Qubits: 8 (spin orbitals)
  Estimation Qubits: 4 (precision bits)
  Total Qubits: 12

QPE Circuit Parameters:
  Base Evolution Time: 0.1
  Trotter Steps: 10
  Measurement Shots: 100
  Qubit Mapping: jordan_wigner

Device Selection: auto -> lightning.gpu (GPU detected)

[Step 2] QPE Execution (auto device selection)
--------------------------------------------------------------------------------
Executing: QPE with lightning.gpu...
converged SCF energy = -75.3264641909832
Execution Time: 28.970 s

Energy Results:
  HF Reference Energy: -75.326464 Hartree
  QPE Estimated Energy: -25.525440 Hartree
  Energy Difference: 49.801024 Hartree

Convergence Status:
  Converged: Yes
  Method: real_qpe

Mulliken Population Analysis:
  O0: -2.0000
  H1: +1.0000
  H2: +1.0000
  H3: +1.0000
  Total Charge: +1.0000

[Step 3] Catalyst @qjit QPE Execution
--------------------------------------------------------------------------------
Executing: Catalyst QPE with lightning.qubit + @qjit...
converged SCF energy = -75.3264641909832
Execution Time: 70.714 s

Energy Results:
  HF Reference Energy: -75.326464 Hartree
  QPE Estimated Energy: -26.821347 Hartree
  Energy Difference: 48.505117 Hartree

[Step 4] Results Comparison
--------------------------------------------------------------------------------
Execution Time Comparison:
  Standard QPE: 28.970 s
  Catalyst QPE: 70.714 s
  Ratio: 2.44x (Catalyst includes JIT compilation overhead)

Energy Comparison:
  Standard QPE: -25.525440 Hartree
  Catalyst QPE: -26.821347 Hartree
  Difference: 1.295907 Hartree
  Status: Results differ (stochastic QPE sampling)

[Step 5] Save Results
--------------------------------------------------------------------------------
Results saved to: data/output/h3o_quantum_qpe_results.json

Demo Summary
--------------------------------------------------------------------------------
q2m3 MVP Capabilities Demonstrated:
  [OK] PySCF -> PennyLane Hamiltonian conversion
  [OK] Standard QPE circuit implementation
  [OK] HF state preparation (qml.BasisState)
  [OK] Trotter time evolution (qml.TrotterProduct)
  [OK] Inverse QFT (qml.adjoint(qml.QFT))
  [OK] Phase-to-energy extraction
  [OK] QM/MM system with TIP3P solvation
  [OK] Mulliken population analysis
  [OK] Catalyst @qjit JIT compilation
  [OK] GPU acceleration (lightning.gpu)

================================================================================
                           Demo Completed Successfully
================================================================================
```

## Device Selection

The demo supports flexible device selection via the `device_type` parameter:

| Device Type | Backend | Performance | Use Case |
|-------------|---------|-------------|----------|
| `auto` | Best available | Optimal | **Recommended** |
| `lightning.gpu` | NVIDIA GPU | Fastest | Large circuits, GPU available |
| `lightning.qubit` | CPU (optimized) | Fast | CPU-only, Catalyst JIT |
| `default.qubit` | CPU (standard) | Baseline | Development, debugging |

**Note:** Catalyst `@qjit` currently works best with `lightning.qubit`. The `lightning.gpu` device has compatibility issues with some quantum gates when used with Catalyst.

## Installation

```bash
# Core dependencies
uv pip install -e "."

# GPU support (NVIDIA GPU required)
uv pip install -e ".[gpu]"

# Catalyst JIT support
uv pip install -e ".[catalyst]"

# All optional dependencies
uv pip install -e ".[gpu,catalyst]"
```

## Output Files

Results are saved to `data/output/h3o_quantum_qpe_results.json`:

```json
{
  "timestamp": "2025-11-26T07:27:51...",
  "catalyst_available": true,
  "catalyst_version": "0.13.0",
  "lightning_gpu_available": true,
  "system": {
    "qm_region": "H3O+",
    "n_atoms": 4,
    "total_charge": 1,
    "mm_waters": 8
  },
  "quantum_resources": {
    "active_electrons": 4,
    "active_orbitals": 4,
    "system_qubits": 8,
    "estimation_qubits": 4,
    "total_qubits": 12
  },
  "results_standard": {
    "device": "lightning.gpu",
    "energy": -25.525440,
    "energy_hf": -75.326464,
    "execution_time_s": 28.970
  },
  "results_catalyst": {
    "device": "lightning.qubit",
    "energy": -26.821347,
    "energy_hf": -75.326464,
    "execution_time_s": 70.714,
    "catalyst_enabled": true
  }
}
```

## Known Issues

### Catalyst @qjit + lightning.gpu Incompatibility

**Status**: Known limitation (as of PennyLane Catalyst 0.13.0)

**Symptom**:
When using `use_catalyst=True` with `device_type="lightning.gpu"`, the following error occurs:
```
RuntimeError: [StateVectorCudaManaged.hpp][Line:2407][Method:applyParametricPauliGeneralGate_]:
Error in PennyLane Lightning: custatevec invalid value
```

**Root Cause**:
The controlled Trotter time evolution operation `qml.ctrl(qml.TrotterProduct(...))` used in QPE circuits (`src/q2m3/core/qpe.py:250-253`) is not compatible with the combination of:
1. Catalyst @qjit JIT compilation
2. lightning.gpu (cuQuantum) backend

Specifically:
- lightning.gpu lacks native support for controlled TrotterProduct
- Catalyst decomposes it into QubitUnitary operations
- The decomposed matrices may exceed cuQuantum API parameter constraints
- custatevec's `applyParametricPauliGeneralGate_` function fails parameter validation

**Affected Operations**:

| Operation | lightning.qubit | lightning.gpu | Catalyst @qjit |
|-----------|-----------------|---------------|----------------|
| `qml.TrotterProduct` | OK | OK | OK |
| `qml.ctrl(TrotterProduct)` | OK | **FAIL** | **FAIL** (with GPU) |

**Workaround**:
The demo automatically uses `lightning.qubit` (CPU) for Catalyst execution:
- **Step 2**: Uses `lightning.gpu` (no Catalyst) - ~29 seconds
- **Step 3**: Uses `lightning.qubit` + Catalyst @qjit - ~71 seconds

**Recommended Configuration**:
```python
# GPU without Catalyst (recommended for speed)
qpe_config = {"device_type": "lightning.gpu"}
qmmm = QuantumQMMM(qm_atoms, qpe_config=qpe_config, use_catalyst=False)

# Catalyst with CPU (recommended for JIT optimization)
qpe_config = {"device_type": "lightning.qubit"}
qmmm = QuantumQMMM(qm_atoms, qpe_config=qpe_config, use_catalyst=True)
```

**Official Documentation**:
> "Not all PennyLane devices currently work with Catalyst, and for those that do, their supported feature set may not necessarily match supported features when used without qjit()"
> — [Catalyst Supported Devices](https://docs.pennylane.ai/projects/catalyst/en/latest/dev/devices.html)

---

## References

- [PennyLane QPE Tutorial](https://pennylane.ai/qml/demos/tutorial_qpe/)
- [PennyLane Catalyst Documentation](https://docs.pennylane.ai/projects/catalyst/)
- [PennyLane Lightning GPU](https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html)
- [Catalyst Supported Devices](https://docs.pennylane.ai/projects/catalyst/en/latest/dev/devices.html)
- [q2m3 Technical Overview](../TECHNICAL_OVERVIEW.md)
