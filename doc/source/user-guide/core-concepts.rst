Core Concepts
=============

q2m3 combines quantum chemistry, quantum algorithms, and molecular mechanics.
The :doc:`/glossary` defines the terms used below.

QM/MM Partitioning
------------------

QM/MM separates a system into a quantum mechanical region and a molecular
mechanics environment. q2m3 currently uses PySCF for the classical quantum
chemistry side and TIP3P/SPC/E-style point charges for explicit water
environments. q2m3 stores energies in Hartree internally. The unit kcal/mol
appears only after an explicit conversion.

The current embedding model is electrostatic point-charge embedding. It does
not include a polarizable MM force field or advanced solvent response model.

QPE Workflow
------------

QPE estimates an eigenphase of the time-evolution operator and converts that
phase back into an energy. The PennyLane QPE paths prepare an HF state, apply controlled Trotter
time evolution and inverse QFT, and decode the phase. A classical fallback tests
the QM/MM data pipeline when no PennyLane Hamiltonian is available.

.. mermaid::

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

Active Spaces
-------------

Active-space truncation reduces the simulation size. Report the active electrons, spatial orbitals, and resulting qubit count. For
Jordan-Wigner mappings, each spatial orbital contributes two spin-orbital
qubits.

.. list-table::
   :header-rows: 1

   * - System
     - Typical active space
     - System qubits
     - Notes
   * - H2
     - 2 electrons, 2 orbitals
     - 4
     - First-run validation path
   * - H3O+
     - 4 electrons, 4 orbitals
     - 8
     - Optional ionic solvation diagnostics
   * - Glycine
     - 6, 8, or 10 electrons in the same number of orbitals
     - 12, 16, or 20
     - SQD scaling tutorial. Each space has its own Hamiltonian and reference

The full H3O+ STO-3G space is larger than the default examples. The public H3O+
scripts therefore use conservative active-space and Trotter settings.

Sample-Based Quantum Diagonalization
------------------------------------

SQD samples a CCSD-seeded LUCJ state with ffsim.
It diagonalizes the Hamiltonian in the recovered determinant subspace.
The result includes HF, active-space CCSD, selected-CI, random-subspace, and reference energies.
All comparisons use the same Hamiltonian. See :doc:`sqd` for the reference policy.

The signed primary error is
``1000 * (E_SQD - E_reference)`` in mHa, so a positive value places SQD above the
reference. Only a T0 exact-active-space result supports an exact-error claim.
Seed spread is descriptive. It is not a confidence interval. The sampled
determinant fraction also does not equal simulator-memory savings because ffsim
still represents the full fixed-particle-number state.

SQD runs on a classical simulator. It does not execute on quantum hardware
or demonstrate quantum advantage.

Fragment interaction energies
-----------------------------

The fragment workflow computes ``E(AB) - E(A) - E(B)`` for a dimer.
Counterpoise uses the dimer basis for all three calculations.
AVAS selects active orbitals from explicit atomic-shell labels and a threshold.
The workflow checks correspondence between the dimer and monomer active spaces.

``classical_only`` computes HF and frozen-core MP2.
``full`` adds budgeted CCSD(T), active-space SQD, and resource estimates.
Only a T0 exact-CASCI reference supplies the CASCI energy.
Missing reference values remain absent with reasons.
See :doc:`/tutorials/fragment-interaction` for a complete calculation.

These interaction energies describe fixed geometries.
They do not represent a solvent ensemble or a solvation free energy.

Phase Decoding And Energy Shifts
--------------------------------

QPE measures a phase modulo one. Large negative molecular energies can
therefore produce phase wrapping. When needed, q2m3 uses shifted QPE
parameters to estimate a smaller energy difference relative to a Hartree-Fock
reference.

Two phase-extraction conventions exist in the current code:

.. list-table::
   :header-rows: 1

   * - Path
     - Implementation
     - Interpretation
   * - Sampled QPE
     - ``QPEEngine._extract_energy_from_samples()``
     - Uses the most frequent measured bitstring
   * - Analytical solvation
     - ``q2m3.solvation.phase_extraction``
     - Uses a probability-weighted expected bin

Do not compare those paths without noting the decoding convention and shot
model.

RDM Measurement
---------------

The one-particle reduced density matrix (1-RDM) connects quantum states to
classical observables. q2m3 estimates 1-RDMs for Mulliken populations and charge checks.
Correlated RDM feedback into SQD/MM is not implemented.

RDM values must obey physical constraints such as Hermiticity, electron count,
and non-negative occupations. Measurement noise can violate these constraints,
so the estimator includes projection utilities for physically valid matrices.

Resource Estimation
-------------------

The ``q2m3.core.resource_estimation`` API estimates EFTQC resources such as
Hamiltonian 1-norm, logical qubits, Toffoli gates, system qubits, and target
error. Use these estimates to plan quantum hardware resources. They are not the same as
Catalyst compile memory or LLVM IR size.

The H2 resource example demonstrates that MM point charges mainly change
one-electron terms. For small H2/STO-3G runs, the two-electron integrals
dominate the resource estimate, so vacuum and solvated estimates are close.

Fixed-MO Full-One-Electron Embedding
------------------------------------

Explicit MM point charges perturb the one-electron Hamiltonian. q2m3 exposes
two public resource-estimation modes for this perturbation:

.. list-table::
   :header-rows: 1

   * - Mode
     - Included active-space perturbation
     - Intended use
   * - ``diagonal``
     - Diagonal ``Delta h_pp`` terms only
     - Compatibility row for dynamic coefficient-update workflows
   * - ``full_oneelectron``
     - Full fixed-MO ``Delta h_pq`` matrix
     - Resource rows and fixed-Hamiltonian operator support

Both modes are fixed-MO models. They reuse the vacuum molecular-orbital frame
and hold the two-electron tensor at its vacuum value. They compute no orbital
relaxation, polarizable MM response, or relaxed solvation energy. The
diagnostic ``delta_h_offdiag_fro`` reports the Frobenius norm of the off-diagonal
active-space perturbation that ``diagonal`` mode omits and ``full_oneelectron``
mode includes.

Catalyst Guidance
-----------------

Use Catalyst when repeated executions can reuse a compiled circuit.
MC solvation can reuse the circuit when only Hamiltonian coefficients change.
A single execution may not recover the compilation cost.

.. list-table::
   :header-rows: 1

   * - Use case
     - Recommended path
   * - H2 first validation
     - Standard PennyLane or existing example script
   * - MC with fixed vacuum Hamiltonian
     - Catalyst fixed mode with IR cache
   * - MC with runtime MM embedding
     - Catalyst dynamic mode, treated as a heavier diagnostic
   * - Fixed full-one-electron embedding
     - Fixed Hamiltonian/operator path only
   * - H3O+ high precision or Trotter scans
     - Optional diagnostics on high-memory machines

Three Solvation Modes
---------------------

.. list-table::
   :header-rows: 1

   * - Mode
     - Energy model
     - Purpose
   * - ``fixed``
     - ``E_QPE(H_vac) + E_MM``
     - Fast compile-once baseline
   * - ``hf_corrected``
     - ``E_HF(R) + E_MM`` with interval QPE diagnostics
     - Intermediate mode for throughput
   * - ``dynamic``
     - ``E_QPE(H_eff) + E_MM``
     - Runtime diagonal MM embedding

q2m3 uses the difference between fixed and dynamic correlation behavior to study
``delta_corr-pol``, the correlation-polarization coupling term.

The current dynamic runtime coefficient path updates diagonal coefficients. It
does not accept ``embedding_mode="full_oneelectron"`` because off-diagonal
one-electron terms can change the operator support that Catalyst compiled for a
given circuit.
