Architecture
============

Package boundaries
------------------

q2m3 has separate entry points for QPE, solvation, SQD, and fragment interaction energies.
The diagram shows their main computational dependencies.

.. mermaid::

   flowchart TB
       accTitle: q2m3 package architecture
       accDescr: Fragment interaction workflows call the integral SQD solver and core resource estimates. Solvation retains a separate Catalyst and Monte Carlo path.

       user["User scripts"]
       fragments["fragments<br/>counterpoise, AVAS, references, reports"]
       sqd["sqd<br/>sampling and subspace diagonalization"]
       core["core<br/>QPE, QM/MM, RDM, resource estimates"]
       interfaces["interfaces<br/>PySCF and PennyLane conversion"]
       solvation["solvation<br/>Catalyst circuits and MC orchestration"]
       sampling["sampling<br/>water geometry, MC moves, MM terms"]
       profiling["profiling<br/>memory, timing, IR diagnostics"]
       utils["utils<br/>I/O and plotting"]

       user --> fragments
       user --> sqd
       user --> core
       user --> solvation
       user --> profiling
       user --> utils
       fragments --> sqd
       fragments --> core
       sqd --> interfaces
       core --> interfaces
       solvation --> core
       solvation --> interfaces
       solvation --> sampling
       solvation --> profiling

.. list-table::
   :header-rows: 1

   * - Package
     - Responsibility
   * - ``q2m3.fragments``
     - Fragment inputs, counterpoise subsystems, active spaces, interaction energies, and reports.
   * - ``q2m3.sqd``
     - LUCJ sampling, subspace solvers, reference tiers, input provenance, and resource guards.
   * - ``q2m3.core``
     - QPE circuits, QM/MM orchestration, RDMs, quantum resource estimates, and device selection.
   * - ``q2m3.interfaces``
     - Chemistry conversion, fixed-MO embedding, and density matrices.
   * - ``q2m3.solvation``
     - Catalyst circuit construction, IR caching, MC execution, and solvation analysis.
   * - ``q2m3.sampling``
     - Classical MC moves, water geometry, and MM force-field helpers.
   * - ``q2m3.profiling``
     - Runtime, memory, and compilation measurements.
   * - ``q2m3.utils``
     - File I/O and plotting.

Fragment workflow
-----------------

q2m3 exports ``q2m3.run`` and ``q2m3.Molecule`` lazily.
They resolve to ``q2m3.fragments.workflow.run`` and ``q2m3.fragments.molecule.Molecule``.
The workflow accepts geometry with an optional two-fragment partition.
It does not call ``QuantumQMMM`` or the solvation orchestrator.

For a dimer, counterpoise constructs ``AB``, ``A``, and ``B`` in the dimer basis.
Each monomer retains the other fragment's basis functions as ghost atoms.
A single molecule uses subsystem ``M``.

.. mermaid::

   flowchart TD
       accTitle: Fragment interaction energy workflow
       accDescr: Both modes run RHF, validate AVAS spaces, and compute MP2. Full mode adds budgeted CCSD(T), integral SQD, and resource estimates.

       molecule["Molecule + AVAS labels and threshold"]
       cp["RHF: AB, A, B or M"]
       avas["AVAS selection<br/>ghost checks and preflight<br/>dimer-monomer correspondence"]
       classical["HF and frozen-core MP2"]
       mode{"mode"}
       full["budgeted frozen-core CCSD(T)<br/>active-space integrals and context"]
       child["one child interpreter<br/>serial CCSD seeds and integral SQD jobs"]
       resources["QPE cost and SQD resource estimates"]
       result["SubsystemResult or FragmentInteractionResult"]
       report["render_report<br/>derived Markdown report"]

       molecule --> cp --> avas --> classical --> mode
       mode -->|classical_only| result
       mode -->|full| full --> child --> resources --> result
       result --> report

Both modes check AVAS spaces before correlated calculations.
Dimer runs also check whether monomer spaces correspond to the dimer space.
A mismatch raises ``ActiveSpaceMismatchError``.

``classical_only`` computes HF and frozen-core MP2 without constructing an SQD runner.
``full`` requires the SQD extra. It adds:

1. A frozen-core CCSD(T) reference when its predicted cost fits ``CCSDTBudget``.
2. Active-space integrals and an ``IntegralContext`` for each subsystem.
3. A CCSD seed and SQD calculation for each subsystem.
4. QPE cost estimates and SQD workload and memory estimates from those integrals.

A budget-rejected CCSD(T) reference remains absent with a reason.
The QPE estimate uses double factorization. It does not execute a ``QPEEngine`` circuit.

Execution and result boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SubprocessSQDRunner`` starts one child interpreter for all subsystem jobs.
The child runs jobs serially. Its OpenMP, MKL, OpenBLAS, and Rayon thread counts are one.
The parent retains its thread settings.

The child builds each CCSD seed before calling ``run_sqd_from_integrals``.
SQD applies its own supervisor, resource caps, and reference policy.
The first job failure stops later jobs. The parent propagates the failure and cleans up stage processes.

For each energy layer, the dimer result computes ``E(AB) - E(A) - E(B)`` in Hartree.
A missing subsystem energy makes that interaction energy absent.
Only a T0 ``exact_casci`` baseline supplies ``e_casci``.
The result retains each complete ``SQDResult``, provenance, warnings, and absence reasons.

Error decomposition separates algorithm, active-space, basis, and model differences.
The same-basis reference uses computed CCSD(T), or an imported reference when CCSD(T) is absent.
Missing inputs leave the corresponding error absent.
``render_report`` derives a Markdown report from the results.
Imported experimental Ks values appear beside the results without statistical conclusions.

Optional fragment tools
~~~~~~~~~~~~~~~~~~~~~~~

These tools require explicit calls. ``run`` does not invoke them:

- ``structure`` reads experimental dimers, normalizes X-H bonds, and supports DFT-D3(BJ) geometry optimization.
- ``external.psi4`` runs an external Psi4 interpreter for an HF counterpoise comparison.
- ``diagnostics.entropy`` computes DMRG orbital entropies and mutual information.

``importers`` reads CSV or QCSchema reference energies.
``run(references=...)`` invokes this importer when callers supply reference paths.

These workflows do not implement polarizable MM or an SQD-MC loop.
Fragment interaction energies are not solvation free energies.

QPE and solvation
-----------------

``QuantumQMMM.compute_ground_state`` constructs a Hamiltonian through the PySCF/PennyLane converter.
It calls ``QPEEngine`` for energy estimation and can request RDM measurements.
A classical fallback exercises the QM/MM pipeline when no PennyLane Hamiltonian is available.

Solvation uses a separate orchestrator with ``fixed``, ``hf_corrected``, and ``dynamic`` modes.
Catalyst compiles circuit structure for repeated execution.
The IR cache keys include molecule, basis, active space, estimation wires, Trotter depth, circuit style, and embedding mode.
Dynamic updates change diagonal coefficients.
Full-one-electron embedding requires a fixed Hamiltonian because off-diagonal terms can change operator support.
See :doc:`user-guide/core-concepts` for the energy models and physical limits.

``device_type="auto"`` selects an available PennyLane device.
Catalyst execution depends separately on the JAX/Catalyst backend.
A Lightning GPU device name alone does not establish GPU execution for a compiled circuit.

SQD entry points
----------------

The geometry entry constructs a vacuum RHF frame, active-space integrals, and a CCSD seed.
The integral entry accepts those data with an authenticated ``IntegralContext`` and ``CCSDSeed``.
Fragments uses the integral entry with its AVAS-selected subsystem Hamiltonians.
Both entries share the SQD executor.

The executor checks frame and Hamiltonian hashes, tensor shapes, symmetries, constants, energies, and residuals.
It applies a shared wall deadline and polls process-tree RSS.
Preprocessing performed before ``run_sqd_from_integrals`` is outside that call's resource measurement.
See :doc:`user-guide/sqd` for platform and calibration limits.

Source map
----------

Paths below are relative to ``src/q2m3``.

.. list-table::
   :header-rows: 1

   * - Responsibility
     - Source
   * - Public fragment entry and orchestration
     - ``__init__.py``, ``fragments/workflow.py``
   * - Geometry, counterpoise, and AVAS checks
     - ``fragments/molecule.py``, ``fragments/counterpoise.py``, ``fragments/active_space.py``
   * - Integral assembly and classical references
     - ``fragments/integrals.py``, ``fragments/references.py``
   * - SQD child process and transport
     - ``fragments/sqd_stage.py``, ``fragments/_sqd_child.py``, ``sqd/transport.py``
   * - Results, imported references, and reports
     - ``fragments/result.py``, ``fragments/importers.py``, ``fragments/report.py``
   * - Resource estimates
     - ``fragments/resources.py``, ``core/resource_estimation.py``, ``sqd/resources.py``
   * - QPE and solvation entry points
     - ``core/quantum_qmmm.py``, ``solvation/orchestrator.py``
