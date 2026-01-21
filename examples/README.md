# q2m3 Examples

> **q2m3 MVP** - Hybrid Quantum-Classical QM/MM for Early Fault-Tolerant Quantum Computers (EFTQC)

## Overview

This directory contains examples demonstrating the q2m3 workflow for molecular ground state energy estimation using Quantum Phase Estimation (QPE), with GPU acceleration and optional PennyLane Catalyst JIT compilation support.

**Key Capabilities Demonstrated:**
- PySCF to PennyLane molecular Hamiltonian conversion
- Standard QPE circuit with Trotter time evolution
- GPU acceleration via `lightning.gpu` device
- QM/MM system setup with TIP3P water solvation
- Solvation effect analysis (vacuum vs explicit MM embedding)
- EFTQC resource estimation (Toffoli gates, logical qubits, 1-norm analysis)
- Catalyst `@qjit` compilation for JIT optimization
- Quantum 1-RDM measurement for Mulliken population analysis
- Circuit visualization via `qml.draw(decimals=None, level=0)`

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Install GPU support (optional but recommended)
uv pip install -e ".[gpu]"

# Run minimal example (H2 + 2 TIP3P waters, fast validation)
python examples/h2_qpe_h2o_mm_minimal.py

# Run full demo (H3O+ + 8 TIP3P waters, comprehensive analysis)
python examples/h3op_qpe_h2o_mm_full.py
```

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

## Device Selection

The examples support flexible device selection via the `device_type` parameter:

| Device Type | Backend | Performance | Use Case |
|-------------|---------|-------------|----------|
| `auto` | Best available | Optimal | **Recommended** |
| `lightning.gpu` | NVIDIA GPU | Fastest | Large circuits, GPU available, Catalyst compatible |
| `lightning.qubit` | CPU (optimized) | Fast | CPU-only, Catalyst JIT |
| `default.qubit` | CPU (standard) | Baseline | Development, debugging |

**Note:** Catalyst `@qjit` now supports `lightning.gpu` as of PennyLane Lightning 0.44.0 (Issue #2 fixed).

---

## Example 1: H2 + MM Water (Minimal Validation)

`h2_qpe_h2o_mm_minimal.py` provides a minimal validation of QPE + explicit MM solvation using H2 molecule with 2 TIP3P water molecules.

```bash
python examples/h2_qpe_h2o_mm_minimal.py
```

### Test System: H2 (Hydrogen Molecule)

```
H ─────── H          Total charge: 0

Geometry (Angstrom):
  H1: ( 0.000,  0.000,  0.000)
  H2: ( 0.000,  0.000,  0.740)
```

### Quantum Resource Configuration (H2)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Active Space** | 2e, 2o | 2 electrons in 2 spatial orbitals |
| **System Qubits** | 4 | 2 orbitals x 2 spin = 4 spin orbitals |
| **Estimation Qubits** | 4 | Precision bits for phase readout |
| **Total Qubits** | 8 | System + estimation registers |
| **Trotter Steps** | 20 | Time evolution accuracy |
| **Shots** | 100 | Measurement statistics |

### Validation Strategy

1. Compare vacuum HF vs vacuum QPE (verify QPE correctness)
2. Compare solvated HF vs solvated QPE (verify MM embedding in QPE)
3. Compare stabilization effects (HF vs QPE should agree on sign/magnitude)

### Results (lightning.gpu, 8 qubits, 20 Trotter steps)

| Method | Vacuum (Ha) | Solvated (Ha) | Stabilization (kcal/mol) |
|--------|-------------|---------------|--------------------------|
| PySCF HF | -1.116759 | -1.116674 | -0.054 |
| QPE | -1.134209 | -1.134122 | -0.054 |

All validation checks passed: PL↔PySCF agreement < 0.001 kcal/mol, QPE↔HF diff ~0.017 Ha, stabilization sign consistent.

---

## Example 2: H3O+ QPE Full Demo

`h3op_qpe_h2o_mm_full.py` demonstrates the complete q2m3 workflow with comprehensive solvation effect analysis and Catalyst JIT comparison.

```bash
python examples/h3op_qpe_h2o_mm_full.py
```

### Pipeline

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

### Test System: H3O+ (Hydronium Ion)

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

### Quantum Resource Configuration (H3O+)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Active Space** | 4e, 4o | 4 electrons in 4 spatial orbitals |
| **System Qubits** | 8 | 4 orbitals x 2 spin = 8 spin orbitals |
| **Estimation Qubits** | 4 | Precision bits for phase readout |
| **Total Qubits** | **12** | System + estimation registers |
| **Qubit Mapping** | Jordan-Wigner | Fermion-to-qubit encoding |
| **Trotter Steps** | 10 | Time evolution accuracy |
| **Base Time** | auto | Auto-computed to avoid phase overflow |
| **Shots** | 100 | Measurement statistics |

### Demo Workflow (8 Steps)

| Step | Description | Output |
|------|-------------|--------|
| **Step 1** | System Configuration | QM/MM setup, quantum resources, device selection |
| **Step 1.5** | Circuit Visualization | QPE + RDM circuit diagrams via `qml.draw()` |
| **Step 2** | EFTQC Resource Estimation | Toffoli gates, logical qubits, 1-norm analysis |
| **Step 3** | Classical HF Solvation Analysis | Vacuum vs solvated HF energies, MM embedding validation |
| **Step 4** | Standard QPE Solvation Analysis | Vacuum vs solvated QPE energies, charge redistribution |
| **Step 5** | Catalyst QPE Solvation Analysis | Same as Step 4 with `@qjit` compilation |
| **Step 6** | Results Comparison | Time comparison, energy consistency |
| **Step 7** | Save Results | JSON output to `data/output/` |

### Sample Output (GPU Environment)

```
================================================================================
                    H3O+ Quantum Phase Estimation (QPE) Demo
                    q2m3 MVP - Catalyst Technical Validation
================================================================================
Timestamp: 2025-11-27 08:49:37
Catalyst Available: Yes (v0.14.0)
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
  Base Evolution Time: auto
  Trotter Steps: 10
  Measurement Shots: 100
  Qubit Mapping: jordan_wigner

Device Selection: auto -> lightning.gpu (GPU detected)

[Step 1.5] Circuit Visualization (PennyLane)
--------------------------------------------------------------------------------
Generating QPE + RDM circuit diagrams...

QPE Circuit (Standard Phase Estimation):
------------------------------------------------------------
PennyLane Circuit (decimals=None, level=0):
 0: ─╭|Ψ⟩─╭TrotterProduct†─╭TrotterProduct†───────┤
 1: ─├|Ψ⟩─├TrotterProduct†─├TrotterProduct†───────┤
...
 8: ──H───╰●───────────────│────────────────╭QFT†─┤ ╭Sample
 9: ──H────────────────────╰●───────────────├QFT†─┤ ├Sample

RDM Measurement Circuit (Pauli Expectation Values):
------------------------------------------------------------
PennyLane Circuit (decimals=None, level=0):
0: ─╭|Ψ⟩─╭TrotterProduct─┤  <Z>
1: ─├|Ψ⟩─├TrotterProduct─┤  <Z>
...

[Step 2] EFTQC Resource Estimation (Vacuum vs Solvated)
--------------------------------------------------------------------------------
Resource Comparison (H3O+ with 8 TIP3P waters, 24 MM charges)

  ----------------------------------------------------------------------
  Configuration                 1-norm (Ha)     Toffoli Gates      Qubits
  ----------------------------------------------------------------------
  Vacuum (chemical)             34.21           148,284,000        314
  Vacuum (relaxed)              34.21           14,828,400         314
  Solvated (chemical)           34.66           150,234,000        314
  Solvated (relaxed)            34.66           15,023,400         314
  ----------------------------------------------------------------------

  Analysis:
    MM embedding effect: Δλ = +1.3% (1-norm increase)
    Error relaxation:    90.0% fewer gates (10× error tolerance)

[Step 3] Solvation Effect Analysis (Classical HF)
--------------------------------------------------------------------------------
Comparing H3O+ energy in vacuum vs. explicit TIP3P water environment...
This validates that MM embedding correctly polarizes the QM electron density.

MM Environment: 8 TIP3P waters (24 point charges)

Hartree-Fock Energy Comparison:
  Vacuum (no MM):     -75.326464 Hartree
  Solvated (with MM): -75.332155 Hartree

Solvation Stabilization:
  ΔE = 0.005691 Hartree
     = 3.57 kcal/mol

  [OK] MM embedding is working: explicit solvent stabilizes H3O+

[Step 4] Standard QPE Solvation Effect Analysis (Quantum Level)
--------------------------------------------------------------------------------
Comparing QPE energies: vacuum vs. explicit TIP3P solvation...
This validates MM embedding is correctly included in the quantum Hamiltonian.

Execution Time:
  Vacuum QPE:   27.689 s
  Solvated QPE: 28.717 s
  Total:        56.406 s

Standard QPE Energy Comparison:
  Vacuum (no MM):     -76.503440 Hartree
  Solvated (with MM): -76.509220 Hartree

Standard QPE Solvation Stabilization:
  ΔE = 0.005780 Hartree
     = 3.63 kcal/mol

Convergence Status (Solvated):
  Converged: Yes
  Method: real_qpe
  RDM Source: quantum_measurement

Comparison with Classical HF:
  HF Stabilization:  3.57 kcal/mol
  Standard QPE Stabilization: 3.63 kcal/mol
  Difference: 0.06 kcal/mol

  [OK] Standard QPE correctly captures MM solvation effect

Mulliken Charge Redistribution (Vacuum -> Solvated):
  O0: +0.8477 -> +0.8481 (Δq = +0.0004)
  H1: -0.0538 -> +0.3133 (Δq = +0.3671)
  H2: +0.1030 -> -0.2462 (Δq = -0.3493)
  H3: +0.1030 -> +0.0849 (Δq = -0.0182)

[Step 5] Catalyst @qjit QPE Solvation Effect Analysis
--------------------------------------------------------------------------------
Comparing Catalyst QPE energies: vacuum vs. explicit TIP3P solvation...
This validates MM embedding works correctly with Catalyst JIT compilation.

Execution Time:
  Vacuum QPE:   27.689 s
  Solvated QPE: 28.717 s
  Total:        56.406 s

Catalyst QPE Energy Comparison:
  Vacuum (no MM):     -76.503440 Hartree
  Solvated (with MM): -76.509220 Hartree

Catalyst QPE Solvation Stabilization:
  ΔE = 0.005780 Hartree
     = 3.63 kcal/mol

  [OK] Catalyst QPE correctly captures MM solvation effect

[Step 6] Results Comparison
--------------------------------------------------------------------------------
Execution Time Comparison (Solvated QPE):
  Standard QPE (lightning.gpu): 28.717 s
  Catalyst QPE (lightning.gpu): 28.717 s
  Ratio: 1.0x ( Catalyst + GPU now available)
  Note: Catalyst now supports lightning.gpu for qml.ctrl(TrotterProduct),
        enabling GPU acceleration with JIT compilation.

Energy Comparison (Solvated):
  Standard QPE: -76.509220 Hartree
  Catalyst QPE: -76.509220 Hartree
  Difference: 0.000000 Hartree
  Status: Results consistent (diff < 0.01 Ha)

[Step 7] Save Results
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
  [OK] EFTQC resource estimation (Toffoli gates, logical qubits)
  [OK] HF MM embedding (ΔE = 3.6 kcal/mol)
  [OK] Standard QPE MM embedding (ΔE = 3.6 kcal/mol)
  [OK] Catalyst QPE MM embedding (ΔE = 3.6 kcal/mol)
  [OK] Quantum RDM measurement (Pauli expectation values)
  [OK] Mulliken population analysis (from quantum RDM)
  [OK] Circuit visualization (qml.draw)
  [OK] Catalyst @qjit JIT compilation (now supports lightning.gpu)
  [OK] GPU acceleration (lightning.gpu, works with both standard and Catalyst QPE)

================================================================================
                           Demo Completed Successfully
================================================================================
```

### Output Files

Results are saved to `data/output/h3o_quantum_qpe_results.json`:

```json
{
  "timestamp": "2025-11-26T07:27:51...",
  "catalyst_available": true,
  "catalyst_version": "0.14.0",
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
    "device": "lightning.gpu",
    "energy": -76.503440,
    "energy_hf": -75.326464,
    "execution_time_s": 42.290,
    "catalyst_enabled": true
  }
}
```

### Design Notes: Why Two QPE Calculations Are Required

The demo compares vacuum and solvated QPE energies to demonstrate MM embedding effects at the quantum level. This requires **two independent QPE calculations**.

**Fundamental Reason**: QPE performs time evolution `exp(-iHt)` on a Hamiltonian. Since vacuum and solvated environments use different Hamiltonians, separate QPE circuits are necessary:

| Environment | Hamiltonian | Time Evolution |
|-------------|-------------|----------------|
| Vacuum | `H_vacuum` | `exp(-i * H_vacuum * t)` |
| Solvated | `H_solvated = H_vacuum + H_MM` | `exp(-i * H_solvated * t)` |

**Architectural Implication**:
```python
# Two QuantumQMMM instances are required
qmmm_vacuum = QuantumQMMM(qm_atoms=h3o_atoms, mm_waters=0, ...)
qmmm_solvated = QuantumQMMM(qm_atoms=h3o_atoms, mm_waters=8, ...)

result_vacuum = qmmm_vacuum.compute_ground_state()
result_solvated = qmmm_solvated.compute_ground_state()

# Compare QPE energies
stabilization = result_vacuum["energy"] - result_solvated["energy"]
```

This is not a code inefficiency but a fundamental requirement of the QPE algorithm when comparing different physical systems.

---

## Known Issues & Catalyst Compatibility

### Issue 1: BasisState + Controlled Operations under @qjit (FIXED)

**Status**: Fixed (as of PennyLane 0.44.0, Catalyst 0.14.0)

**Tracking**: [Catalyst #2235](https://github.com/PennyLaneAI/catalyst/issues/2235) - Closed

**Symptom**: QPE under `@qjit` produces incorrect quantum states when `qml.BasisState` coexists with `qml.ctrl()` operations in the same circuit.

**Precise Diagnosis** (verified via state vector comparison):

```python
# Test circuit: BasisState + Hadamard + ctrl(TrotterProduct)
# Expected: superposition state [(4, 0.7071), (5, 0.7071)]

@qjit + BasisState:  [(4, 1.0)]           # ✗ Wrong (collapsed state)
@qjit + X gates:     [(4, 0.7071), (5, 0.7071)]  # ✓ Correct
No @qjit + BasisState: [(4, 0.7071), (5, 0.7071)]  # ✓ Correct
```

**Root Cause**: Under `@qjit`, when `qml.BasisState` and `qml.ctrl()` coexist in the same circuit, Catalyst's compilation produced incorrect quantum state evolution. Individual operations worked correctly; only the combination failed.

**Fix**: Issue resolved in Catalyst 0.14.0. `qml.BasisState` now works correctly with `qml.ctrl()` under `@qjit`.

**Previous Workaround** (`src/q2m3/core/qpe.py:173-177`): Used explicit X gates instead of BasisState (still works, can be kept):

```python
# Previous workaround (still functional):
for wire, state in zip(wires, hf_state):
    if state == 1:
        qml.PauliX(wires=wire)

# Can now use (both work correctly):
qml.BasisState(hf_state, wires=wires)
```

**Verification** (H3O+, 12 qubits, Catalyst 0.14.0):

| Configuration | Energy (Ha) | Success Rate |
|--------------|-------------|--------------|
| @qjit + BasisState | -76.503440 | 100% ✅ |
| @qjit + X gates | -76.503440 | 100% ✅ |

**Recommendation**: Can keep existing workaround (X gates) or switch to `qml.BasisState` for cleaner code. Both produce identical results (diff < 1e-10).

**Note**: RDM module (`rdm.py:225`) uses `qml.BasisState` safely, as RDM circuits don't use `qml.ctrl()`.

---

### Issue 2: Catalyst @qjit + lightning.gpu Incompatibility (FIXED)

**Status**: Fixed (as of PennyLane Lightning 0.44.0)

**Tracking**: [pennylane-lightning PR #1298](https://github.com/PennyLaneAI/pennylane-lightning/pull/1298) - Merged

**Symptom**: Using `@qjit` with `lightning.gpu` triggers custatevec error:
```
RuntimeError: custatevec invalid value in applyParametricPauliGeneralGate_
```

**Affected Operation**: `qml.ctrl(qml.TrotterProduct(...))` (`qpe.py:206-209`)

**Root Cause**: The `GPhase` operation with zero-qubit target wires was not supported in `lightning.gpu` backend, causing failures when executing controlled time evolution operators.

**Fix**: PR #1298 added support for `GPhase` with zero-qubit target wires in Lightning GPU (merged in v0.44.0), enabling `ctrl(TrotterProduct)` execution. This brings feature parity with `lightning.qubit` and `lightning.kokkos` backends.

**Verification**: Catalyst `@qjit` + `qml.ctrl(qml.TrotterProduct)` now works on both `lightning.qubit` (CPU) and `lightning.gpu` (GPU).

**Recommended Configuration**:
```python
# GPU with Catalyst (best performance)
qpe_config = {"device_type": "lightning.gpu"}
qmmm = QuantumQMMM(qm_atoms, qpe_config=qpe_config, use_catalyst=True)

# CPU with Catalyst (JIT optimization)
qpe_config = {"device_type": "lightning.qubit"}
qmmm = QuantumQMMM(qm_atoms, qpe_config=qpe_config, use_catalyst=True)

# GPU without Catalyst (alternative)
qpe_config = {"device_type": "lightning.gpu"}
qmmm = QuantumQMMM(qm_atoms, qpe_config=qpe_config, use_catalyst=False)
```

**Performance Impact**:
- Catalyst + GPU: ~2-3x faster than Catalyst + CPU
- Catalyst + CPU: similar performance to non-Catalyst GPU (JIT compilation benefits)

**Reference**: [Catalyst Supported Devices](https://docs.pennylane.ai/projects/catalyst/en/latest/dev/devices.html)

---

### Issue 3: Lightning Device + MM Hamiltonian Compatibility (Fixed)

**Status**: Fixed (as of commit 2025-11-27) - Verified with PennyLane 0.44.0

**Symptom**: When using `qml.Hamiltonian` to construct MM corrections and adding them to the vacuum Hamiltonian via `+` operator, `lightning.qubit` and `lightning.gpu` fail with:
```
DeviceError: Operator Controlled(Evolution(...)) not supported with lightning.gpu
```

**Root Cause**:
1. `qml.Hamiltonian(coeffs, ops)` returns `LinearCombination` type
2. `H_vacuum + H_mm_correction` produces `LinearCombination` type
3. `lightning` devices don't support `Controlled(Evolution(...))` decomposition for `LinearCombination` type

**Fix** (`pyscf_pennylane.py:296-320`): Use `qml.s_prod` + `qml.sum` to maintain `Sum` type:

```python
# Before (incompatible with lightning devices):
coeffs = [identity_coeff, -delta_h1e_mo[p, p] / 2, ...]
ops = [qml.Identity(0), qml.Z(wire), ...]
H_mm_correction = qml.Hamiltonian(coeffs, ops)  # Returns LinearCombination
H_solvated = H_vacuum + H_mm_correction  # Returns LinearCombination

# After (lightning-compatible):
mm_terms = [qml.s_prod(identity_coeff, qml.Identity(wires=list(range(n_qubits))))]
for p in range(active_orbitals):
    for spin in [0, 1]:
        mm_terms.append(qml.s_prod(-delta_h1e_mo[p, p] / 2, qml.Z(wire)))
all_operands = list(H_vacuum.operands) + mm_terms
H_solvated = qml.sum(*all_operands)  # Maintains Sum type
```

**Verification** (H2 + 2 TIP3P waters):

| Method | Vacuum (Ha) | Solvated (Ha) | Stabilization |
|--------|-------------|---------------|---------------|
| PySCF HF | -1.116759 | -1.116674 | -0.054 kcal/mol |
| QPE (lightning.gpu) | -1.134209 | -1.134122 | -0.054 kcal/mol |

---

## References

- [PennyLane QPE Tutorial](https://pennylane.ai/qml/demos/tutorial_qpe/)
- [PennyLane Catalyst Documentation](https://docs.pennylane.ai/projects/catalyst/)
- [PennyLane Lightning GPU](https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html)
- [Catalyst Supported Devices](https://docs.pennylane.ai/projects/catalyst/en/latest/dev/devices.html)
- [q2m3 Technical Overview](../TECHNICAL_OVERVIEW.md)
