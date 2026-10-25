Sample-based quantum diagonalization (SQD)
==========================================

SQD samples a CCSD-seeded LUCJ state with ffsim.
It solves the recovered subspace with qiskit-addon-sqd.
Both public entry points return an ``SQDResult`` with comparisons for the same Hamiltonian.
This classical simulation does not demonstrate quantum advantage or hardware execution.

Installation and platform
-------------------------

Use this source checkout on Linux x86_64 for the SQD implementation and examples:

.. code-block:: bash

   uv sync --frozen --extra sqd
   export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
   export NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu

The public supervisor requires Linux x86_64 because it uses ``/proc`` and
``multiprocessing`` with ``fork``. The supervisor rejects other platforms before
execution.

The ``sqd`` extra installs ffsim and qiskit-addon-sqd.
Catalyst is optional. Use ``uv.lock`` to reproduce the dependency versions.

``src/q2m3/sqd/resource_calibration.json`` defines the measured resource model and dependency versions.
The model does not cover other platforms or arbitrary dependency versions.
Installed Catalyst must match the calibration version. Catalyst may be absent.

Use :doc:`/tutorials/sqd-showcase` for a sequence of example runs.
See :doc:`/api-reference/sqd` for signatures and configuration types.

Ordinary ``import q2m3`` still imports PennyLane. The SQD algorithm does not
call PennyLane or Catalyst. SQD workflow exports load on access.
Without the extra, they raise ``ImportError`` with the installation command. To
check availability, test ``"run_sqd" in q2m3.__all__``. Do not use ``hasattr``,
because it can trigger that error. Configuration/result types do not import
ffsim or Qiskit.

Two runnable public entry points
--------------------------------

Save either Python block as a file. After the installation above, run
``uv run --no-sync python FILE.py``. Both use H₂ at 0.74 Å/STO-3G, explicitly
``(2 electrons, 2 spatial orbitals)``, or four system qubits. The small
128-shot smoke case uses seed 31 and two LUCJ repetitions. It reaches the
full four-determinant space. Agreement with CASCI is a regression check,
not evidence that sampling outperforms classical solvers.

The geometry entry point builds integrals and a same-frame CCSD seed:

.. code-block:: python

   import numpy as np
   from q2m3 import run_sqd
   from q2m3.sqd import LUCJConfig

   result = run_sqd(
       ["H", "H"],
       np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
       active_electrons=2,
       active_orbitals=2,
       basis="sto-3g",
       lucj=LUCJConfig(n_reps=2, shots=128),
       seed=31,
       verbose=False,
   )
   result.validate()
   assert result.baseline_tier == "T0"
   assert abs(result.sqd_energy - result.baseline_energy) < 1e-10
   print(result.sqd_energy, result.baseline_method, result.delta_mHa)

The integral entry point accepts already assembled chemist-order tensors.
This example uses the guarded producers to supply authenticated frame and
CCSD data. ``host_available_mb`` is a conservative memory cap, not a request
to allocate that much memory. Here, currently available Linux memory bounds
it.

.. code-block:: python

   from pathlib import Path

   from q2m3.molecule import MoleculeConfig
   from q2m3.sqd import LUCJConfig, run_sqd_from_integrals
   from q2m3.sqd.ansatz import build_ccsd_seed
   from q2m3.sqd.integrals import build_integrals

   available_kib = next(
       int(line.split()[1])
       for line in Path("/proc/meminfo").read_text().splitlines()
       if line.startswith("MemAvailable:")
   )
   cap_mb = min(4096.0, available_kib * 1024 / 1e6)
   molecule = MoleculeConfig(
       name="H2", symbols=["H", "H"],
       coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
       charge=0, active_electrons=2, active_orbitals=2,
   )
   data = build_integrals(molecule, host_available_mb=cap_mb)
   seed_data = build_ccsd_seed(data, host_available_mb=cap_mb)
   result = run_sqd_from_integrals(
       data.h1, data.h2, data.e_core,
       norb=data.norb, nelec=data.nelec,
       context=data.context, seed_data=seed_data,
       lucj=LUCJConfig(n_reps=2, shots=128), seed=31,
   )
   result.validate()
   assert result.baseline_tier == "T0"
   assert abs(result.sqd_energy - result.baseline_energy) < 1e-10
   print(result.sqd_energy, result.baseline_method, result.delta_mHa)

Both examples yield approximately −1.13728383449 Ha in the measured profile.
``run_sqd_from_integrals`` requires ``context``. Bare ``t1``/``t2`` arrays are not an
accepted substitute for ``CCSDSeed``. The solver checks these inputs:

- shape
- finite values
- balanced electron counts
- Hamiltonian/frame identity
- HF/CCSD energy
- CCSD residuals

Failed or unconverged seeds raise an error. For a deliberate
baseline-only run, set ``mode="reference_only"`` (and ``seed_data=None`` at the
integral entry). The workflow never silently substitutes HF for a failed SQD run.

Fixed orbital frame and MM embedding
------------------------------------

Coordinates, including ``mm_coords``, are in Å. ``mm_charges`` are in elementary
charges. Energies and integral constants are Hartree. The high-level entry
accepts both MM arrays and ``embedding_mode="diagonal"`` (default) or
``"full_oneelectron"``. The latter requires MM charges. The vacuum MO frame
and two-electron tensor stay fixed. Diagonal mode adds only diagonal active
one-electron terms. Full mode also retains off-diagonal terms.

All solver arms share the same active span, Hamiltonian, frame and core
constant. ``e_core`` already includes the MM nuclear contribution: do not add
it again at the low-level entry. The low-level ``h2`` is chemist ordered and
is not transposed again. In an embedded fixed frame, ``hf_energy`` is the
reference determinant expectation, not a relaxed embedded SCF minimum.
Tests cover static MM and the H₂O adapter. SQD MC integration,
correlated RDM feedback and polarizable MM are outside this delivery.

Reference ladder and result consumers
-------------------------------------

``ReferenceConfig`` defaults to 900 seconds and 8192 decimal MB. The reference
ladder shares the remaining budget across attempts:

.. list-table::
   :header-rows: 1

   * - Tier
     - Method
     - Meaning
   * - T0
     - ``exact_casci``
     - Exact within this finite active Hamiltonian
   * - T1
     - Selected CI
     - Tight-cutoff result. Cutoff differences do not bound its error
   * - T1+
     - Explicitly enabled plugin
     - Solver-specific approximation and uncertainty
   * - T2
     - CCSD(T)
     - Approximate fallback with diagnostics and trust flags

T0 cannot be disabled to force a downgrade. Only explicit unavailability,
budget exhaustion or controlled timeout permits trying a lower tier.
The workflow must report numerical failures. Every non-T0 result carries
``baseline_downgrade_reason`` and a warning.

If uncertainty is unknown, the result records ``None`` and a reason. It must
not record zero. T2 reports the T1 diagnostic, convergence,
triples correction and triples/correlation ratio. Flags can mark it
untrustworthy. Zero T0 uncertainty excludes basis/model/experimental errors.

Use ``baseline_energy``, ``baseline_method``, ``baseline_tier``,
``baseline_uncertainty_mHa``, ``baseline_downgrade_reason``, ``warnings`` and
``null_reasons``. Output schema ``sqd.result.v1`` has 40 fields and writes only
canonical ``baseline_*`` keys. An old ``shci_energy`` number alone cannot tell
whether it came from CASCI or SHCI. Readers must require explicit method,
tier, units and uncertainty semantics. They must reject conflicting aliases.

``delta_mHa = 1000 * (sqd_energy - baseline_energy)`` and
``delta_vs_sci_mHa = 1000 * (sqd_energy - iso_ndet_sci_energy)``:
positive means SQD is worse. ``ratio_sqd_over_sci`` is defined only against
T0 with a valid nonzero SCI error. SCI and random arms match each spin's
subspace dimension. ``subspace_dim`` is their Cartesian product.
``unique_dets_vs_shots`` counts sampled pairs and is a different quantity.

Reference-only outputs have ``None`` and explicit reasons for unexecuted
sampling/comparison fields. ``save_json_results`` from ``q2m3.utils.io`` accepts
the complete dataclass and preserves these fields. It rejects nonfinite JSON
values.

Resource limits
----------------

The calibrated Linux model requires serial execution and balanced closed shells.
Its domain permits:

- 2–10 active orbitals.
- At most 252 determinants per spin and 63504 determinant pairs.
- At most 36 UCJ factors, two batches, two iterations, and 100000 shots.

Other stage-specific input/domain checks still apply. Public diagonalization
uses two batches, two iterations and ``min(300, shots)`` samples per batch.
Unknown/out-of-domain bounds raise ``ResourceModelDomainError`` before the
allocation. ``allow_large`` cannot bypass the domain, host or hard cap.

The allocation policy warns at 2048 MB and normally rejects at 8192 MB.
Its absolute ceiling is 12288 MB.
The public executor applies a separate 8192 MB component cap.
The effective limit is the smallest component, host, or user cap.
A prediction equal to the effective limit rejects before allocation.
``allow_large`` cannot bypass that limit.

All limits use decimal MB.
The historical benchmark criterion is ``peak_bytes < 2_000_000_000``.

See :doc:`sqd-validation` for measured results and research status.
See :doc:`/api-reference/sqd-results` for all result fields.
