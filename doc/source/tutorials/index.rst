Tutorials
=========

The tutorials are ordered from lightweight validation toward larger SQD and
solvation workflows. Run the H2 paths first before attempting glycine, H3O+, or
high-memory profiling scripts.

.. toctree::
   :maxdepth: 1

   sqd-showcase
   h2-qpe-validation
   h2-resource-estimation
   full-oneelectron-embedding
   h2-mc-solvation
   three-mode-solvation

Optional Diagnostic Scripts
---------------------------

The following examples are intentionally not first-run tutorials:

* ``examples/qmmm/h3o_mc_solvation.py``: H3O+ ionic solvation, 16 GB+ RAM recommended.
* ``examples/qpe/h3o_8bit_qpe_benchmark.py``: high-precision H3O+ benchmark with
  fallback options, 30 GB+ RAM recommended.
* ``examples/performance/h3o_dynamic_trotter_oom_scan.py``: memory-guarded dynamic Trotter
  scaling scan.
* ``examples/performance/qpe_memory_profile.py``: compilation memory profiler.
