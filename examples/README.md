# q2m3 Examples

> Hybrid Quantum-Classical QM/MM for Early Fault-Tolerant Quantum Computers (EFTQC)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation](#2-motivation)
3. [Development Journey](#3-development-journey)
4. [Phase 1-2: H2 QPE Validation](#phase-1-2-h2-qpe-validation)
5. [Phase 3: H3O+ Full Demo](#phase-3-h3o-full-demo)
6. [Phase 4-5: MC Solvation Sampling](#phase-4-5-mc-solvation-sampling)
7. [Phase 6: HF-Level MM Embedding](#phase-6-hf-level-mm-embedding)
8. [The Missing Physics](#8-the-missing-physics)
9. [Path Forward](#9-path-forward)

---

## 1. Project Overview

This directory contains examples demonstrating the q2m3 development journey from simple vacuum QPE to the current MC solvation workflow with Catalyst integration.

**Core Capabilities:**
- PySCF to PennyLane molecular Hamiltonian conversion
- Standard QPE circuit with Trotter time evolution
- GPU acceleration via `lightning.gpu` device
- QM/MM system setup with TIP3P water solvation
- Solvation effect analysis (vacuum vs explicit MM embedding)
- EFTQC resource estimation (Toffoli gates, logical qubits, 1-norm)
- Catalyst `@qjit` compilation for iterative workflows
- Quantum 1-RDM measurement for Mulliken population analysis
- Circuit visualization via `qml.draw()`

**Quick Start:**

```bash
# Activate virtual environment
source .venv/bin/activate

# Run examples in development order
python examples/h2_qpe_validation.py       # Phase 1-2
python examples/h3o_qpe_full_demo.py      # Phase 3
python examples/h2_mc_solvation.py        # Phase 4-5
python examples/h3o_mc_solvation.py       # Phase 6
```

---

## 2. Motivation

### Scientific Motivation: Why Quantum QM/MM?

In the electrostatic embedding QM/MM formalism (Warshel & Levitt, 1976), the effective Hamiltonian is:

$$H_{\mathrm{eff}} = H_{\mathrm{QM}} + H_{\mathrm{QM-MM}}(R_{\mathrm{MM}}) + H_{\mathrm{MM}}$$

The QM-MM coupling term $H_{\mathrm{QM-MM}}(R_{\mathrm{MM}})$ depends explicitly on the MM coordinates. For molecular solvation, this means:

- Each solvent configuration (different water positions) creates a **different Hamiltonian**
- The correlation energy $E_{\mathrm{corr}}$ varies with solvent geometry
- Classical methods (HF, DFT) cannot capture configuration-specific correlation effects

**The Scientific Question**: Can quantum computers efficiently estimate correlation energies for each solvent configuration, enabling accurate solvation free energy calculations?

### Technical Motivation: Why This Stack?

| Component | Rationale |
|-----------|-----------|
| **PySCF** | Mature classical QM; provides Hartree-Fock reference and molecular integrals |
| **PennyLane** | Flexible quantum circuit library; supports multiple backends |
| **Catalyst (@qjit)** | JIT compilation for iterative workflows (MC sampling, optimization) |
| **Lightning GPU** | High-performance simulation; essential for QPE feasibility |

### Project Motivation: What Are We Validating?

q2m3 is a proof-of-concept (POC) targeting **Early Fault-Tolerant Quantum Computers (EFTQC)**. Our primary validation goals:

1. **Algorithm correctness**: Can QPE reproduce HF energies and capture correlation?
2. **MM embedding**: Does explicit solvent affect QPE energies consistently with HF?
3. **Scalability**: What are the resource requirements for realistic molecular systems?
4. **Compiler performance**: Can Catalyst enable practical iterative quantum-classical workflows?

---

## 3. Development Journey

```
Phase 1: Vacuum H2 QPE
    ↓
Phase 2: H2 + TIP3P MM Embedding
    ↓
Phase 3: H3O+ (More Complex System)
    ↓
Phase 4: MC Sampling (Dynamic Environment)
    ↓
Phase 5: Catalyst Integration → Discovered Bottleneck
    ↓
Phase 6: Pre-compilation Workaround (Current State)
```

### Phase 1: Vacuum H2 QPE (Algorithm Validation)

**Goal**: Validate that QPE can reproduce Hartree-Fock energies and capture correlation.

**Key Results**:
- QPE-HF gap: ~0.017 Ha (10.9 kcal/mol of correlation energy)
- Standard QPE achieves chemical accuracy for small molecules

**File**: `h2_qpe_validation.py`

### Phase 2: H2 + TIP3P MM Embedding (MM Validation)

**Goal**: Verify that MM embedding is correctly incorporated into the quantum Hamiltonian.

**Key Results**:
- PySCF HF stabilization: -0.054 kcal/mol (2 waters)
- QPE stabilization: -0.054 kcal/mol (consistent!)
- PL↔PySCF agreement: < 0.001 kcal/mol

**File**: `h2_qpe_validation.py`

### Phase 3: H3O+ (More Complex System)

**Goal**: Scale to a more complex cation (hydronium ion) with explicit solvation shell.

**Key Results**:
- 12 qubits (8 system + 4 estimation)
- Solvation stabilization: 3.63 kcal/mol (8 TIP3P waters)
- Mulliken charge redistribution captured from quantum RDM

**File**: `h3o_qpe_full_demo.py`

### Phase 4: MC Sampling (Dynamic Environment)

**Goal**: Move from static solvent to dynamic MC sampling where each step changes the Hamiltonian.

**Key Challenge Identified**: Every MC step (translation ±0.3Å, rotation ±15°) changes all one-electron integral coefficients simultaneously. For H3O+ (4e,4o), this affects **1086 Hamiltonian terms**.

**File**: `h2_mc_solvation.py`

### Phase 5: Catalyst Integration (Bottleneck Discovery)

**Goal**: Use Catalyst @qjit to accelerate the MC loop.

**Critical Finding**: Catalyst/XLA treats Hamiltonian coefficients as **compile-time constants**. Even with identical operator structure, coefficient changes trigger full recompilation:

| Stage | Standard | Catalyst | Ratio |
|-------|----------|----------|-------|
| Circuit Build | 0.009s | 67.2s | ~7467x slower |
| Circuit Exec | 23.3s | 0.65s | 36x faster |
| **Total** | 23.3s | 67.8s | **2.9x slower** |

**Root Cause**: XLA embeds Hamiltonian coefficients in the MLIR graph at compile time. New coefficient values = structurally new input = full recompilation (~67s per MC step).

**File**: `h3o_qpe_full_demo.py`

### Phase 6: Pre-compilation Workaround (Current Solution)

**Goal**: Avoid per-step recompilation by using pre-compiled vacuum QPE + classical MM correction.

**Energy Decomposition**:
$$E_{\mathrm{total}} = E_{\mathrm{QPE}}(H_{\mathrm{vacuum}}) + \Delta E_{\mathrm{MM}}(\mathrm{HF})$$

**Performance**:
- QPE compilation: 10.55s (one-time)
- Subsequent QPE calls: ~124 ms (vs 65s with recompilation)
- Total (100 MC steps, 10 QPE evals): 38s

**Trade-off**: This approximation ignores correlation-polarization coupling ($\delta_{\mathrm{corr-pol}}$), which is ~0.6-3 kcal/mol for H3O+ (comparable to the solvation energy itself).

**File**: `h3o_mc_solvation.py`

---

## Phase 1-2: H2 QPE Validation

`h2_qpe_validation.py` provides minimal validation of QPE + explicit MM solvation.

```bash
python examples/h2_qpe_validation.py
```

### Test System: H2 (Hydrogen Molecule)

```
H ─────── H          Total charge: 0

Geometry (Angstrom):
  H1: ( 0.000,  0.000,  0.000)
  H2: ( 0.000,  0.000,  0.740)
```

### Quantum Resources

| Parameter | Value |
|-----------|-------|
| Active Space | 2e, 2o |
| System Qubits | 4 |
| Estimation Qubits | 4 |
| Total Qubits | 8 |
| Trotter Steps | 20 |

### Results

| Method | Vacuum (Ha) | Solvated (Ha) | Stabilization |
|--------|-------------|---------------|---------------|
| PySCF HF | -1.116759 | -1.116674 | -0.054 kcal/mol |
| QPE | -1.134209 | -1.134122 | -0.054 kcal/mol |

**Catalyst Performance**: For small systems (H2, 8 qubits), Catalyst shows nearly identical performance (1.01x slower).

---

## Phase 3: H3O+ Full Demo

`h3o_qpe_full_demo.py` demonstrates the complete workflow with comprehensive solvation analysis.

```bash
python examples/h3o_qpe_full_demo.py
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

### Quantum Resources

| Parameter | Value |
|-----------|-------|
| Active Space | 4e, 4o |
| System Qubits | 8 |
| Estimation Qubits | 4 |
| Total Qubits | 12 |
| Qubit Mapping | Jordan-Wigner |
| Trotter Steps | 10 |

### Solvation Effect

| Method | Stabilization (kcal/mol) | Δ vs HF |
|--------|-------------------------|---------|
| PySCF HF | 3.57 | --- |
| Standard QPE | 3.63 | +0.06 |
| Catalyst QPE | 3.63 | +0.06 |

### EFTQC Resources

| Configuration | Toffoli Gates | Qubits |
|--------------|---------------|--------|
| Vacuum (chemical acc.) | 142,564,912 | 314 |
| Vacuum (relaxed 10×) | 14,257,873 | 308 |
| Solvated (chemical acc.) | 142,447,459 | 314 |

---

## Phase 4-5: MC Solvation Sampling

`h2_mc_solvation.py` demonstrates MC sampling with Catalyst @qjit and pre-compiled QPE.

```bash
python examples/h2_mc_solvation.py
```

### System

- **QM Region**: H2 molecule (4 qubits)
- **MM Region**: 10 TIP3P water molecules
- **Workflow**: 100 MC steps with QPE validation every 10 steps

### Architecture

```
1. Pre-compilation Phase (one-time)
   └─ Build vacuum Hamiltonian → @qjit compile QPE circuit → Ready (6.15s)

2. MC Sampling Loop (100 steps)
   ├─ Propose move (translation + rotation)
   ├─ pure_callback (PySCF HF energy)
   ├─ Metropolis acceptance
   └─ catalyst.cond (QPE every 10 steps)

3. QPE Validation
   ├─ Pre-compiled circuit (via closure)
   ├─ Extract energy (phase analysis)
   └─ MM correction (pure_callback)
```

### Performance Results

| Metric | Value |
|--------|-------|
| QPE Compilation | 6.15s (one-time) |
| MC Loop Compilation | ~8.61s |
| Total Wall Time | 17.99s |
| QPE Evaluations | 10 (every 10 steps) |
| Acceptance Rate | 38% |
| QPE Energy per Call | ~30 ms |

### Energy Formula

$$E_{\mathrm{total}} = E_{\mathrm{QPE}}(\mathrm{vacuum}) + \Delta E_{\mathrm{MM}}(\mathrm{HF})$$

---

## Phase 6: HF-Level MM Embedding

`h3o_mc_solvation.py` extends MC sampling to H3O+ with HF-level MM correction.

```bash
python examples/h3o_mc_solvation.py
```

### System

- **QM Region**: H3O+ (hydronium ion, 4 atoms, +1 charge)
- **MM Region**: 10 TIP3P water molecules
- **Active Space**: 2 electrons, 2 orbitals (reduced for efficiency)

### Performance Results

| Metric | Value |
|--------|-------|
| QPE Compilation | 10.55s (one-time) |
| Total Wall Time | 38.07s |
| Acceptance Rate | 32% |
| QPE Energy per Call (after first) | ~124 ms |

### Comparison: vacuum_correction vs mm_embedded

| Aspect | vacuum_correction | mm_embedded |
|--------|------------------|-------------|
| QPE Circuit | Pre-compiled once | Re-built per config |
| Compilation | ~10.5s (one-time) | ~10s × N configs |
| Energy Accuracy | Approximate | Rigorous |
| Correlation-Polarization | Ignored | Included |
| Best For | MC Sampling | Final Validation |

---

## 8. The Missing Physics

### The Approximation: $\delta_{\mathrm{corr-pol}}$

The current workaround uses:

$$E_{\mathrm{QPE}}(H_{\mathrm{eff}}) \approx E_{\mathrm{QPE}}(H_{\mathrm{vac}}) + \Delta E_{\mathrm{HF}}$$

This assumes the coupling term vanishes:
$$\delta_{\mathrm{corr-pol}} \equiv E_{\mathrm{corr}}(H_{\mathrm{eff}}) - E_{\mathrm{corr}}(H_{\mathrm{vac}}) = 0$$

### Literature Evidence: Why $\delta_{\mathrm{corr-pol}} \neq 0$

| Paper | Finding |
|-------|---------|
| Yoshida et al. (JCTC 2024) | Different solvent snapshots yield different QPE/VQE energies; HF alone cannot recover configuration-specific correlation |
| Weisburn et al. (arXiv:2409.06813) | Correlation energy variation ~100 kcal/mol across configurations; not predictable from HF |
| Reinholdt et al. (JPCA 2025) | Static HF environment underestimates excitation energies; PE-UCCSD recovers accuracy |

### Quantifying the Error

- $\delta_{\mathrm{corr-pol}} \sim 10$-$30\%$ of $E_{\mathrm{corr}}$
- For H3O+: **~0.6-3 kcal/mol**
- Comparable to total solvation energy (3.57 kcal/mol)

### Current Status

| Quantity | Computed? | Value |
|----------|-----------|-------|
| HF solvation energy (H3O+) | ✓ | 3.57 kcal/mol |
| QPE solvation energy (H3O+) | ✓ | 3.63 kcal/mol |
| QPE-HF gap (vacuum, H2) | ✓ | 10.9 kcal/mol |
| QPE-HF gap (vacuum, H3O+) | ✓ | ~1.18 Ha† |
| $\delta_{\mathrm{corr-pol}}$ per config | ✗ | ~0.6-3 kcal/mol (est.) |
| Config ranking error | ✗ | Unknown |

†H3O+ QPE-HF gap is dominated by 4-bit phase estimation systematic error; not representative of true correlation energy. The δ_corr-pol estimate uses H2's correlation energy as a proxy.

### Catalyst Compilation Bottleneck

The root cause of the approximation is **not** a scientific limitation but a **compiler engineering challenge**:

**Root Cause**: Catalyst/XLA treats Hamiltonian coefficients as compile-time constants. Changing coefficients triggers full recompilation.

```python
# Scientifically complete (blocked):
for step in mc_steps:
    H_new = build_hamiltonian(qm_atoms, mm_charges_new)
    # Triggers ~67s recompilation!
    E_qpe = QPE(H_new)
```

**Fine-grained timing** (H3O+, 12 qubits, 1086 Hamiltonian terms):

| Stage | Standard | Catalyst | Ratio |
|-------|----------|----------|-------|
| Hamiltonian build | 1.24s | 1.17s | 1.06x |
| Circuit compilation | 0.009s | 67.2s | ~7467x |
| Circuit execution | 23.3s | 0.65s | 36x |
| **Total** | 23.5s | 68.0s | **2.9x** |

### static_argnums Test Results

| Test Case | Result |
|-----------|--------|
| Simplified H (4 terms) | Cache hit: 606× speedup |
| Real H3O+ (1086 terms, same structure) | Full recompilation: no speedup |

`static_argnums` addresses recompilation from type/shape changes. Our problem is **coefficient value** changes — a different class, not covered by the existing mechanism.

---

## 9. Path Forward

### Solution Paths

| Path | Status | Completeness |
|------|--------|--------------|
| Coefficient parameterization in `TrotterProduct` | Open; MLIR layer change needed | Full |
| `ParametrizedEvolution` + lightning | Compile error in Catalyst 0.14.0 | Full |
| QA-VMC (Li 2025) | Available now | Partial |
| HSB-QSCI (Sugisaki 2025) | Per-step recompile in Catalyst | Partial |

### Target Architecture

```python
@qjit
def mc_loop(solvents, op_structure):
    @for_loop(0, n_steps, 1)
    def step(i, state):
        h_coeffs = pure_callback(
            compute_mm_integrals, solvents)
        H_dyn = build_H(op_structure, h_coeffs)
        E = qpe_circuit(H_dyn)  # no recompile
        return metropolis_update(state, E)
    return for_loop(0, n_steps, 1)(step)(init)
```

### Open Questions for Xanadu/PennyLane Team

1. **Coefficient parameterization**: Is passing Hamiltonian coefficients as runtime arguments on the Catalyst roadmap?

2. **ParametrizedEvolution + Catalyst**: Is `qml.evolve(H, params)` with lightning devices planned for a future Catalyst release?

3. **Alternative patterns**: Are there patterns beyond `static_argnums` that handle per-step coefficient value changes without full recompilation?

4. **Recommended strategy**: For workflows where MM charges update at every MC step, what is the currently recommended path?

### Roadmap

| Milestone | Phase | Status |
|-----------|-------|--------|
| Real QPE circuit (H2) | 1 | ✓ Complete |
| H2 + TIP3P solvation benchmark | 2 | ✓ Complete |
| QJIT performance analysis | 1-2 | ✓ Complete |
| H3O+ QPE demo | 3 | ✓ Complete |
| H3O+ MC solvation demo | 6 | ✓ Complete |
| NH3 / formamide examples | Future | ☐ Not started |
| PennyLane blog post | Future | ☐ Not started |
| JCTC paper | Future | ☐ Not started |

### System Expansion

| System | Active Space | System Qubits | Hamiltonian Terms |
|--------|--------------|---------------|------------------|
| H2 | (2e,2o) | 4 | ~15 |
| H3O+ | (4e,4o) | 8 | 1086 |
| NH3 | (8e,8o) | 16 | ~5k |
| Formamide | (12e,10o) | 20 | ~15k |

### Publication Timeline

- arXiv pre-print: ~May 2026
- JCTC submission: ~June 2026
- PennyLane demo: concurrent with arXiv

---

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

| Device Type | Backend | Performance | Use Case |
|-------------|---------|-------------|----------|
| `auto` | Best available | Optimal | **Recommended** |
| `lightning.gpu` | NVIDIA GPU | Fastest | Large circuits, GPU available |
| `lightning.qubit` | CPU (optimized) | Fast | CPU-only, Catalyst JIT |
| `default.qubit` | CPU (standard) | Baseline | Development, debugging |

---

## References

- [PennyLane QPE Tutorial](https://pennylane.ai/qml/demos/tutorial_qpe/)
- [PennyLane Catalyst Documentation](https://docs.pennylane.ai/projects/catalyst/)
- [PennyLane Lightning GPU](https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html)
- [Catalyst Supported Devices](https://docs.pennylane.ai/projects/catalyst/en/latest/dev/devices.html)
- [q2m3 Technical Overview](../TECHNICAL_OVERVIEW.md)
