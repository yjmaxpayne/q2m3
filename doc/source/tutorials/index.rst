Tutorials
=========

The tutorials are ordered from lightweight validation toward heavier solvation
diagnostics. Run the H2 tutorials first before attempting H3O+ or high-memory
profiling scripts.

.. toctree::
   :maxdepth: 1

   h2-qpe-validation
   h2-resource-estimation
   h2-mc-solvation
   three-mode-solvation

Optional Diagnostic Scripts
---------------------------

The following examples are intentionally not first-run tutorials:

* ``examples/h3o_mc_solvation.py``: H3O+ ionic solvation, 16 GB+ RAM recommended.
* ``examples/h3o_8bit_qpe_benchmark.py``: high-precision H3O+ benchmark with
  fallback options, 30 GB+ RAM recommended.
* ``examples/h3o_dynamic_trotter_oom_scan.py``: memory-guarded dynamic Trotter
  scaling scan.
* ``examples/qpe_memory_profile.py``: compilation memory profiler.
