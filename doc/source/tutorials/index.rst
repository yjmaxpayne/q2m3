Tutorials
=========

The tutorials cover small validation runs and larger SQD and solvation
workflows. Run the H2 examples before glycine, H3O+, or profiling scripts
that require large amounts of memory.

For fragment interactions, start with :doc:`fragment-interaction` to calculate
the water-dimer interaction energy. Continue with :doc:`fragment-report` to
compare the methods and interpret the resource estimates.

.. toctree::
   :maxdepth: 1

   sqd-showcase
   h2-qpe-validation
   h2-resource-estimation
   full-oneelectron-embedding
   fragment-interaction
   fragment-report
   h2-mc-solvation
   three-mode-solvation

Optional Diagnostic Scripts
---------------------------

Run these diagnostics after the first-run tutorials:

* ``examples/qmmm/h3o_mc_solvation.py``: H3O+ ionic solvation, 16 GB+ RAM recommended.
* ``examples/qpe/h3o_8bit_qpe_benchmark.py``: high-precision H3O+ benchmark with
  fallback options, 30 GB+ RAM recommended.
* ``examples/performance/h3o_dynamic_trotter_oom_scan.py``: memory-guarded dynamic Trotter
  scaling scan.
* ``examples/performance/qpe_memory_profile.py``: compilation memory profiler.
