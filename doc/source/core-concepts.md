# Core Concepts

q2m3 connects classical quantum chemistry with quantum phase-estimation
circuits and explicit MM point-charge environments. The framework is small
enough for H2 validation while exposing the same architectural issues that
appear in larger EFTQC-oriented studies.

## QM/MM Partitioning

QM/MM separates a system into a quantum mechanical region and a molecular
mechanics environment. q2m3 currently uses PySCF for the classical quantum
chemistry side and TIP3P/SPC/E-style point charges for explicit water
environments. Energies are stored in Hartree internally; kcal/mol appears only
after an explicit conversion.

The current embedding model is electrostatic point-charge embedding. It does
not include a polarizable MM force field or advanced solvent response model.

## QPE Workflow

QPE estimates an eigenphase of the time-evolution operator and converts that
phase back into an energy. In q2m3, real PennyLane QPE paths build HF state
preparation, controlled Trotter time evolution, inverse QFT, and phase
decoding. A classical fallback path is still useful for testing the QM/MM data
pipeline when a PennyLane Hamiltonian is not available.

```{mermaid}
flowchart LR
    accTitle: QPE Energy Flow
    accDescr: The q2m3 QPE path prepares a reference state, applies controlled time evolution, reads out a phase, and converts the phase to an energy.

    hamiltonian["PySCF and PennyLane Hamiltonian"]
    hf_state["HF reference state"]
    qpe["QPE circuit"]
    phase["Estimated phase"]
    energy["Energy in Hartree"]

    hamiltonian --> qpe
    hf_state --> qpe
    qpe --> phase
    phase --> energy
```

## Active Spaces

Active-space truncation keeps simulation sizes manageable. An active space is
reported as electrons/orbitals, followed by the resulting qubit count. For
Jordan-Wigner mappings, each spatial orbital contributes two spin-orbital
qubits.

| System | Typical active space | System qubits | Notes |
| --- | --- | --- | --- |
| H2 | 2 electrons, 2 orbitals | 4 | First-run validation path |
| H3O+ | 4 electrons, 4 orbitals | 8 | Optional ionic solvation diagnostics |

The full H3O+ STO-3G space is larger than the default examples. The public H3O+
scripts therefore use conservative active-space and Trotter settings.

## Phase Decoding And Energy Shifts

QPE measures a phase modulo one. Large negative molecular energies can wrap
around the phase register, so q2m3 uses shifted QPE parameters to estimate a
smaller energy difference relative to a Hartree-Fock reference when needed.

Two phase-extraction conventions exist in the current code:

| Path | Implementation | Interpretation |
| --- | --- | --- |
| Sampled QPE | `QPEEngine._extract_energy_from_samples()` | Uses the most frequent measured bitstring |
| Analytical solvation | `q2m3.solvation.phase_extraction` | Uses a probability-weighted expected bin |

Do not compare those paths without noting the decoding convention and shot
model.

## RDM Measurement

The one-particle reduced density matrix (1-RDM) maps quantum-state information
back into classical chemical analysis. q2m3 uses RDM estimation for Mulliken
population analysis, charge conservation checks, and future QM/MM feedback
loops.

RDM values must obey physical constraints such as Hermiticity, electron count,
and non-negative occupations. Measurement noise can violate these constraints,
so the estimator includes projection utilities for physically valid matrices.

## Resource Estimation

The `q2m3.core.resource_estimation` API estimates EFTQC resources such as
Hamiltonian 1-norm, logical qubits, Toffoli gates, system qubits, and target
error. These are hardware-planning estimates; they are not the same as
Catalyst compile memory or LLVM IR size.

The H2 resource example demonstrates that MM point charges mainly modify
one-electron terms. For small H2/STO-3G runs, the two-electron integrals
dominate the resource estimate, so vacuum and solvated estimates are close.

## Catalyst Guidance

Catalyst is most useful when the workflow can compile once and execute many
times. Single QPE executions often pay too much compile overhead. MC solvation
is a better fit because each accepted solvent configuration can reuse the same
compiled circuit structure when only coefficients change at runtime.

| Use case | Recommended path |
| --- | --- |
| H2 first validation | Standard PennyLane or existing example script |
| MC with fixed vacuum Hamiltonian | Catalyst fixed mode with IR cache |
| MC with runtime MM embedding | Catalyst dynamic mode, treated as a heavier diagnostic |
| H3O+ high precision or Trotter scans | Optional diagnostics on high-memory machines |

## Three Solvation Modes

| Mode | Energy model | Purpose |
| --- | --- | --- |
| `fixed` | `E_QPE(H_vac) + E_MM` | Fast compile-once baseline |
| `hf_corrected` | `E_HF(R) + E_MM` with interval QPE diagnostics | Intermediate mode for throughput |
| `dynamic` | `E_QPE(H_eff) + E_MM` | Most complete current model for MM-embedded QPE |

The difference between fixed and dynamic correlation behavior is used to study
`delta_corr-pol`, the correlation-polarization coupling term.
