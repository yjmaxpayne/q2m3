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
8. [Phase 7: Runtime Coefficient Architecture](#phase-7-runtime-coefficient-architecture)
9. [The Missing Physics](#9-the-missing-physics)
10. [Path Forward](#10-path-forward)

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
python examples/h2_mm_embedded_mc.py      # Phase 7
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
Phase 6: Pre-compilation Workaround
    ↓
Phase 7: Runtime Coefficient Parameterization (Breakthrough)
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

**Key Challenge Identified**: Every MC step (translation ±0.3Å, rotation ±15°) changes all one-electron integral coefficients simultaneously. For H3O+ (4e,4o), this affects **105 Hamiltonian terms**.

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

### Phase 6: Pre-compilation Workaround

**Goal**: Avoid per-step recompilation by using pre-compiled vacuum QPE + classical MM correction.

**Energy Decomposition**:
$$E_{\mathrm{total}} = E_{\mathrm{corr}}(\mathrm{vac}) + E_{\mathrm{HF}}(R)$$

QPE uses energy-shifted Hamiltonian $H' = H_{\mathrm{vac}} - E_{\mathrm{HF}} \cdot I$.

**Performance**:
- QPE compilation: 10.55s (one-time)
- Subsequent QPE calls: ~124 ms (vs 65s with recompilation)
- Total (100 MC steps, 10 QPE evals): 38s

**Trade-off**: This approximation ignores correlation-polarization coupling ($\delta_{\mathrm{corr-pol}}$), which is ~0.6-3 kcal/mol for H3O+ (comparable to the solvation energy itself).

**File**: `h3o_mc_solvation.py`

### Phase 7: Runtime Coefficient Parameterization (Breakthrough)

**Goal**: Solve the Catalyst recompilation bottleneck by making Hamiltonian coefficients JAX-traceable runtime parameters, enabling compile-once QPE with MM embedding.

**Key Discovery**: `TrotterProduct` accepts JAX-traced coefficients when `check_hermitian=False`. Operators remain compile-time constants; only coefficients vary at runtime.

**Performance** (H2, 8 qubits, 500 MC steps, 50 QPE evaluations):
- mm_embedded compilation: 219.4s (one-time)
- Subsequent QPE execution: ~45ms (no recompilation)
- Compile/Execute ratio: 4836×
- Projected speedup at 1000 evaluations: 829×

**File**: `h2_mm_embedded_mc.py`

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

$$E_{\mathrm{total}} = E_{\mathrm{corr}}(\mathrm{vac}) + E_{\mathrm{HF}}(R)$$

QPE operates on the energy-shifted Hamiltonian $H' = H_{\mathrm{vac}} - E_{\mathrm{HF}} \cdot I$, measuring correlation energy directly.

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

## Phase 7: Runtime Coefficient Architecture

`h2_mm_embedded_mc.py` solves the Catalyst recompilation bottleneck (Phase 5) by making Hamiltonian coefficients JAX-traceable runtime parameters. This enables the scientifically rigorous mm_embedded QPE mode within a compile-once architecture.

```bash
python examples/h2_mm_embedded_mc.py
```

### Breakthrough: Compile Once, Run Many

The key insight: `TrotterProduct` accepts JAX-traced coefficients when `check_hermitian=False`. Operators are captured in the closure as compile-time constants; only coefficients vary at runtime.

```python
@qjit
def qpe_with_coeffs(coeffs_arr):
    H_runtime = qml.dot(coeffs_arr, ops)  # ops: compile-time; coeffs: runtime
    ...
    qml.TrotterProduct(H_runtime, time=t, n=n, check_hermitian=False)
```

This eliminates the ~67s per-step recompilation discovered in Phase 5, replacing it with a one-time compilation cost amortized over all MC evaluations.

### Two QPE Modes Compared

The experiment runs the same H2 + 10 TIP3P system with two energy strategies:

| Aspect | vacuum_correction | mm_embedded |
|--------|------------------|-------------|
| Energy Formula | $E_{\mathrm{corr}}(\mathrm{vac}) + E_{\mathrm{HF}}(R)$ | $E_{\mathrm{corr\_eff}}(R) + E_{\mathrm{HF}}(\mathrm{vac})$ |
| Compilation | 8.6s (one-time) | 219.4s (one-time) |
| Per-step QPE | ~27ms | ~45ms |
| Total Wall Time | 23.0s | 235.9s |
| Physics | Approximate (ignores $\delta_{\mathrm{corr-pol}}$) | Rigorous (MM in QPE Hamiltonian) |

### Circuit Configuration

| Parameter | vacuum_correction | mm_embedded |
|-----------|------------------|-------------|
| Hamiltonian Terms | 15 | 15 |
| System Qubits | 4 | 4 |
| Estimation Qubits | 4 | 4 |
| Total Qubits | 8 | 8 |
| Trotter Steps | 10 | 10 |
| Energy Formula | $E_{\mathrm{corr}}(\mathrm{vac}) + E_{\mathrm{HF}}(R)$ | $E_{\mathrm{corr\_eff}}(R) + E_{\mathrm{HF}}(\mathrm{vac})$ |

**Note**: Both modes use energy-shifted QPE ($H' = H - E_{\mathrm{HF}} \cdot I$) to measure correlation energy directly. Phase extraction uses probability-weighted expected value ($\sum_k p_k \cdot k / 2^n$) over `qml.probs()` output, preserving continuous sensitivity to sub-bin MM corrections (~0.1 mHa) that integer-bin argmax would discard. mm_embedded Trotter steps are guarded by `_MAX_TROTTER_STEPS_RUNTIME = 20`; for H2 (15 terms) the requested 10 steps are within budget. Symbolic IR scales as $n_{\mathrm{est}} \times n_{\mathrm{trotter}} \times n_{\mathrm{terms}}$.

### Performance Results

| Metric | Value |
|--------|-------|
| mm_embedded Compilation | 219.4s (one-time) |
| First QPE Execution | 62.1ms |
| Subsequent QPE Average | 45.4ms ± 14.7ms (n=49) |
| Compile/Execute Ratio | 4836× |
| Speedup (50 evaluations) | 49.5× vs recompile-each-step |
| Projected Speedup (1000 evaluations) | 829× |
| MLIR IR Scale | 4 × 10 × 15 = 600 ops |

### Energy Results

| Metric | vacuum_correction | mm_embedded | Difference |
|--------|------------------|-------------|------------|
| Best QPE Energy (Ha) | -1.165914 | -1.201618† | -0.035704 |
| Solvation Stabilization (kcal/mol) | 30.84 | 53.25 | +22.40 |
| Best HF Energy (Ha) | -1.167186 | -1.167186 | 0.000000 |
| Acceptance Rate | 47.6% | 47.6% | — |

†mm_embedded "best" is a 3.1σ statistical outlier from the expected-value phase extraction noise (4-bit QPE). See δ_corr-pol per-step analysis for reliable comparison.

### δ_corr-pol Summary (Per-step Analysis)

| Metric | Value |
|--------|-------|
| ⟨δ_corr-pol⟩ | -0.10 kcal/mol (mm_embedded captures more correlation) |
| σ(δ_corr-pol) | 7.14 kcal/mol (dominated by 4-bit QPE bin width) |
| SEM (n=50) | 1.01 kcal/mol |
| Trotter bias cancellation | 177× (systematic bias mostly drops out of difference) |
| Vacuum QPE σ | 0.012 mHa (fixed vacuum circuit, stable) |
| MM-embedded QPE σ | 11.365 mHa (configuration-dependent, sensitive) |

### Architecture

```
1. Compilation Phase (one-time, QPE inlined into MC loop @qjit)
   ├─ vacuum_correction: Build H' = H_vac - E_HF·I → @qjit compile (~8.6s)
   └─ mm_embedded: Build H + decompose → @qjit compile parametrized QPE (~219s)

2. MC Sampling Loop (500 steps)
   ├─ Propose move (translation + rotation)
   ├─ pure_callback (PySCF HF energy for Metropolis acceptance)
   └─ Every 10 steps: QPE evaluation
       ├─ vacuum_correction: pre-compiled circuit + MM correction callback
       └─ mm_embedded: pure_callback(compute_new_coeffs) → compiled circuit(coeffs)

3. Phase Extraction (both modes)
   └─ qml.probs() → expected value (Σ probs[k]·k / 2^n) → continuous phase → energy
```

---

## 9. The Missing Physics

### The Approximation: $\delta_{\mathrm{corr-pol}}$

The vacuum_correction mode approximation assumes:

$$E_{\mathrm{corr}}(H_{\mathrm{eff}}) \approx E_{\mathrm{corr}}(H_{\mathrm{vac}})$$

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
| $\delta_{\mathrm{corr-pol}}$ per config (H2) | ✓ (Phase 7) | ⟨δ⟩ = -0.10 kcal/mol, σ = 7.14 kcal/mol (n=50)‡ |
| Config ranking error | ✗ | Unknown |

†H3O+ QPE-HF gap is dominated by 4-bit phase estimation systematic error; not representative of true correlation energy. The δ_corr-pol estimate uses H2's correlation energy as a proxy.

‡Phase 7 measures δ_corr-pol via expected-value phase extraction from `qml.probs()`. The systematic Trotter bias (~28.6 mHa) cancels at 177× ratio in the per-step difference. The large σ is dominated by 4-bit QPE bin width (~12 mHa ≈ 7.5 kcal/mol); true δ for H2/STO-3G is expected ~0.01-0.1 kcal/mol. Higher QPE resolution (more estimation qubits) will improve SNR.

### Catalyst Compilation Bottleneck (Solved in Phase 7)

The root cause of the Phase 6 approximation was a **compiler engineering challenge**:

**Root Cause**: Catalyst/XLA treats Hamiltonian coefficients as compile-time constants. Changing coefficients triggers full recompilation.

```python
# Phase 5-6: blocked by recompilation
for step in mc_steps:
    H_new = build_hamiltonian(qm_atoms, mm_charges_new)
    # Triggers ~67s recompilation!
    E_qpe = QPE(H_new)

# Phase 7: solved via runtime coefficient parameterization
@qjit
def qpe_with_coeffs(coeffs):  # coeffs: JAX-traceable runtime parameter
    H = qml.dot(coeffs, ops)   # ops: compile-time constants
    ...                          # Compile once → ~68ms per evaluation
```

**Fine-grained timing** (H3O+, 12 qubits, 105 Hamiltonian terms):

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
| Real H3O+ (105 terms, same structure) | Full recompilation: no speedup |

`static_argnums` addresses recompilation from type/shape changes. Our problem is **coefficient value** changes — a different class, not covered by the existing mechanism.

---

## 10. Path Forward

### Solution Paths

| Path | Status | Completeness |
|------|--------|--------------|
| Coefficient parameterization in `TrotterProduct` | ✓ Implemented (Phase 7) | Full |
| `ParametrizedEvolution` + lightning | Compile error in Catalyst 0.14.0 | Full |
| QA-VMC (Li 2025) | Available now | Partial |
| HSB-QSCI (Sugisaki 2025) | Per-step recompile in Catalyst | Partial |

### Implemented Architecture (Phase 7)

The target architecture from Phase 5 is now implemented and validated:

```python
@qjit
def qpe_with_coeffs(coeffs_arr):
    H_runtime = qml.dot(coeffs_arr, ops)  # ops: compile-time; coeffs: runtime
    @qml.qnode(dev)
    def qnode():
        # ... HF state prep, Hadamard on estimation qubits ...
        for k, ew in enumerate(est_wires):
            t = (2 ** (n_est - 1 - k)) * base_time
            qml.ctrl(
                qml.adjoint(
                    qml.TrotterProduct(H_runtime, time=t, n=n_trotter,
                                       check_hermitian=False)  # key enabler
                ),
                control=ew,
            )
        qml.adjoint(qml.QFT)(wires=est_wires)
        return qml.probs(wires=est_wires)  # Born probabilities over all 2^n bins
    # Phase extraction: expected value preserves sub-bin sensitivity
    probs = qnode()
    expected_bin = sum(probs[k] * k for k in range(2 ** n_est))
    phase = expected_bin / (2 ** n_est)
    return -2 * jnp.pi * phase / base_time + energy_shift
```

**Key technical requirements**:
1. `check_hermitian=False` bypasses `math.iscomplex()` check that fails on JAX tracers. This is a PennyLane API feature (not a hack) that enables Catalyst to trace through the Hamiltonian construction.
2. `qml.probs()` + expected value (instead of per-bit `expval(PauliZ)` or `argmax`) provides continuous phase sensitivity to sub-bin MM corrections (~0.1 mHa), critical for detecting δ_corr-pol.

### Remaining Open Questions

1. **ParametrizedEvolution + Catalyst**: Is `qml.evolve(H, params)` with lightning devices planned for a future Catalyst release?

2. **Trotter step scaling**: Can the MLIR compilation memory ceiling be raised for larger systems (current guard: `_MAX_TROTTER_STEPS_RUNTIME = 20`; H2 uses 10 within budget, but H3O+ with 105 terms may require lower cap)?

### Roadmap

| Milestone | Phase | Status |
|-----------|-------|--------|
| Real QPE circuit (H2) | 1 | ✓ Complete |
| H2 + TIP3P solvation benchmark | 2 | ✓ Complete |
| QJIT performance analysis | 1-2 | ✓ Complete |
| H3O+ QPE demo | 3 | ✓ Complete |
| H3O+ MC solvation demo | 6 | ✓ Complete |
| Runtime coefficient parameterization | 7 | ✓ Complete |
| MM-embedded QPE comparison | 7 | ✓ Complete |
| H3O+ mm_embedded validation | Future | ☐ Not started |
| NH3 / formamide examples | Future | ☐ Not started |
| PennyLane blog post | Future | ☐ Not started |
| JCTC paper | Future | ☐ Not started |

### System Expansion

| System | Active Space | System Qubits | Hamiltonian Terms |
|--------|--------------|---------------|------------------|
| H2 | (2e,2o) | 4 | ~15 |
| H3O+ | (4e,4o) | 8 | 105 |
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
