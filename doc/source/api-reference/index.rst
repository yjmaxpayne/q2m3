API Reference
=============

The API reference is grouped by package boundary. Start with the package-level
exports, then use the module pages for implementation-specific entry points.

.. toctree::
   :maxdepth: 1

   package
   core
   interfaces
   solvation
   sampling
   profiling
   utils

Package Groups
--------------

.. list-table::
   :header-rows: 1

   * - Page
     - Scope
   * - :doc:`package`
     - Top-level exports, constants, and molecule configuration.
   * - :doc:`core`
     - QPE, QM/MM orchestration, RDM, resource estimation, and device utilities.
   * - :doc:`interfaces`
     - PySCF/PennyLane conversion and density matrix bridge.
   * - :doc:`solvation`
     - MC solvation workflow, Catalyst circuit bundles, IR cache, and analysis.
   * - :doc:`sampling`
     - Classical MC moves, water geometry, force-field helpers, and Metropolis sampling.
   * - :doc:`profiling`
     - Memory, timing, Catalyst IR, and QPE compilation profiling.
   * - :doc:`utils`
     - I/O and plotting helpers.
