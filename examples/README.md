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
- Quantum 1-RDM measurement for Mulliken population analysis
- Circuit visualization via `qml.draw(decimals=None, level=0)`

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

## Demo Workflow (6 Steps)

| Step | Description | Output |
|------|-------------|--------|
| **Step 1** | System Configuration | QM/MM setup, quantum resources, device selection |
| **Step 2** | Circuit Visualization | QPE + RDM circuit diagrams via `qml.draw()` |
| **Step 3** | QPE Execution (auto device) | Energy with `lightning.gpu` (if available) |
| **Step 4** | Catalyst QPE Execution | Energy with `lightning.qubit` + `@qjit` |
| **Step 5** | Results Comparison | Time comparison, energy consistency |
| **Step 6** | Save Results | JSON output to `data/output/` |

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

[Step 2] Circuit Visualization (PennyLane)
--------------------------------------------------------------------------------
Generating QPE + RDM circuit diagrams...

QPE Circuit (Standard Phase Estimation):
------------------------------------------------------------
PennyLane Circuit (decimals=None, level=0):
 0: ─╭|Ψ⟩─╭TrotterProduct─╭TrotterProduct───────┤
 1: ─├|Ψ⟩─├TrotterProduct─├TrotterProduct───────┤
...
 8: ──H───╰●──────────────│───────────────╭QFT†─┤ ╭Sample
 9: ──H───────────────────╰●──────────────├QFT†─┤ ├Sample

RDM Measurement Circuit (Pauli Expectation Values):
------------------------------------------------------------
PennyLane Circuit (decimals=None, level=0):
0: ─╭|Ψ⟩─╭TrotterProduct─┤  <Z>
1: ─├|Ψ⟩─├TrotterProduct─┤  <Z>
...

[Step 3] QPE Execution (auto device selection)
--------------------------------------------------------------------------------
Executing: QPE with lightning.gpu...
converged SCF energy = -75.3264641909832
Execution Time: 42.290 s

Energy Results:
  HF Reference Energy: -75.326464 Hartree
  QPE Estimated Energy: -76.503440 Hartree
  Energy Difference: 1.176976 Hartree

Convergence Status:
  Converged: Yes
  Method: real_qpe

Mulliken Population Analysis (RDM source: quantum_measurement):
  O0: +0.8477
  H1: -0.0538
  H2: +0.1030
  H3: +0.1030
  Total Charge: +1.0000

[Step 4] Catalyst @qjit QPE Execution
--------------------------------------------------------------------------------
Executing: Catalyst QPE with lightning.qubit + @qjit...
converged SCF energy = -75.3264641909832
Execution Time: 89.479 s

Energy Results:
  HF Reference Energy: -75.326464 Hartree
  QPE Estimated Energy: -76.503440 Hartree
  Energy Difference: 1.176976 Hartree

[Step 5] Results Comparison
--------------------------------------------------------------------------------
Execution Time Comparison:
  Standard QPE (lightning.gpu): 42.290 s
  Catalyst QPE (lightning.qubit): 89.479 s
  Ratio: 2.12x slower with Catalyst

Energy Comparison:
  Standard QPE: -76.503440 Hartree
  Catalyst QPE: -76.503440 Hartree
  Difference: 0.000000 Hartree
  Status: Results consistent (diff < 0.01 Ha)

[Step 6] Save Results
--------------------------------------------------------------------------------
Results saved to: data/output/h3o_quantum_qpe_results.json

Demo Summary
--------------------------------------------------------------------------------
q2m3 MVP Capabilities Demonstrated:
  [OK] PySCF -> PennyLane Hamiltonian conversion
  [OK] Standard QPE circuit implementation
  [OK] HF state preparation (explicit X gates, Catalyst-compatible)
  [OK] Trotter time evolution (qml.TrotterProduct)
  [OK] Inverse QFT (qml.adjoint(qml.QFT))
  [OK] Phase-to-energy extraction
  [OK] QM/MM system with TIP3P solvation
  [OK] Quantum RDM measurement (Pauli expectation values)
  [OK] Mulliken population analysis (from quantum RDM)
  [OK] Circuit visualization (qml.draw)
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
    "energy": -76.503440,
    "energy_hf": -75.326464,
    "execution_time_s": 42.290
  },
  "results_catalyst": {
    "device": "lightning.qubit",
    "energy": -76.503440,
    "energy_hf": -75.326464,
    "execution_time_s": 89.479,
    "catalyst_enabled": true
  }
}
```

## Known Issues

### Catalyst @qjit + BasisState Incompatibility (Resolved)

**Status**: Fixed in Phase 11

**Symptom**:
When using `qml.BasisState` with `qml.ctrl(qml.adjoint(TrotterProduct))` under `@qjit`, QPE produces incorrect results (uniform distribution instead of peaked phase estimation).

**Root Cause**:
Catalyst's `set_basis_state_p` primitive lacks control wire support. When BasisState coexists with controlled operations in the same circuit, Catalyst's compilation produces incorrect behavior.

**Evidence** ([Catalyst jax_primitives.py](https://github.com/PennyLaneAI/catalyst/blob/main/frontend/catalyst/jax_primitives.py)):
```python
@set_basis_state_p.def_abstract_eval
def _set_basis_state_abstract_eval(state, basis_state):
    # No handling for control wires - returns unchanged shape
    return state
```

**Fix**: Use explicit X gates instead of `qml.BasisState` for HF state preparation:
```python
# Before (incompatible):
qml.BasisState(hf_state, wires=wires)

# After (Catalyst-compatible):
for wire, state in zip(wires, hf_state):
    if state == 1:
        qml.PauliX(wires=wire)
```

**Related Issues**:
- [Catalyst #1631](https://github.com/PennyLaneAI/catalyst/issues/1631) - BasisState support
- [Catalyst #1301](https://github.com/PennyLaneAI/catalyst/issues/1301) - Nested adjoint/ctrl handling

---

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
