SQD measured results
====================

These are historical measurements from the SQD delivery.
They are not accuracy or memory guarantees for new calculations.
The measured environment used Linux x86_64, Python 3.12.3, NumPy 2.4.1, SciPy 1.16.3,
ffsim 0.0.84, Qiskit 2.5.2, and qiskit-addon-sqd 0.13.1.

The approved default benchmark scope used reps 2/shots 100000. All eight
runs used T0, and their measured whole-process-tree peaks were below
2 billion bytes (maximum 1,021,018,112 bytes). Sampling alone peaked at
732,467,200 bytes. Measurements used 10 ms polling and do not guarantee
capture of every instantaneous peak or certify arbitrary geometries,
MM systems, plugins or T2 paths.

.. list-table::
   :header-rows: 1

   * - Case
     - Active space
     - SQD − T0 (mHa)
     - Interpretation
   * - H₂/STO-3G
     - 2e2o
     - 0
     - Full space 4/4. Convention regression
   * - H₃O⁺/STO-3G
     - 4e4o
     - 0
     - Full space 36/36. Convention regression
   * - Glycine/STO-3G, seeds 0–4
     - 6e6o
     - mean 0.525399, sample std 0.111683, range 0.325821–0.579774
     - Subspaces 64–90/400. All five worse than CCSD. Mixed against SCI
   * - N₂, 1.1 Å/cc-pVDZ, seed 0
     - 10e10o
     - 18.782945
     - Subspace 342/63504. Worse than SCI by 8.918280 mHa and CCSD by 16.801641 mHa

These deltas use each run's own Hamiltonian/frame/core reference. Historical
absolute energies must not be mixed into them. The calibration rejected historical glycine runs with reps 4/shots 200000.
It also rejected N₂ runs with reps 4/shots 500000.
Their energy and full-run memory benchmarks remain unmeasured. Rejecting a run
before execution does not constitute a successful memory benchmark.

E1 (four orbital bases) and E6 (restricted connectivity) remain
**inconclusive** because their original numerical matrices exceed the
certified domain. E1 requires all per-seed and mean ranges to lie strictly
on one side of 2 mHa for a directional conclusion. Missing points cannot
pass. E6 compares restricted-minus-full errors per seed against 5 mHa.
Missing points or threshold crossings are inconclusive.

H₃O⁺ checks show that the operator uses the supplied connectivity.
They do not establish better energies or hardware readiness.
The checks do not certify hardware layout, SWAP overhead, noise, or ancillas.
The delivery did not run E3. Its status remains ``not_run``.

.. list-table::
   :header-rows: 1

   * - Study
     - Status
     - Evidence boundary
   * - E1
     - inconclusive
     - Original four-basis numerical matrix exceeds the certified domain
   * - E3
     - not_run
     - Not executed in this delivery
   * - E6
     - inconclusive
     - Original restricted/full numerical matrix is incomplete

The :doc:`/tutorials/sqd-showcase` identifies the maintained example commands.
Historical reports under ``tmp/`` and evidence under ``.plan/`` are not installed package assets.
Retain the original reports when citing these historical measurements.
