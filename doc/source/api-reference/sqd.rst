SQD API
=======

The SQD API is optional. Install it with ``uv sync --frozen --extra sqd``.
Configuration and result contracts remain importable without the optional
sampling backend; workflow functions are exposed lazily when ``ffsim`` and
``qiskit-addon-sqd`` are installed.

Public Workflows
----------------

.. autofunction:: q2m3.sqd.orchestrator.run_sqd

.. autofunction:: q2m3.sqd.orchestrator.run_sqd_from_integrals

Configuration Contracts
-----------------------

.. autoclass:: q2m3.sqd.LUCJConfig
   :members:

.. autoclass:: q2m3.sqd.ReferenceConfig
   :members:

.. autoclass:: q2m3.sqd.SQDConfig
   :members:

.. autoclass:: q2m3.sqd.IntegralContext
   :members:

.. autoclass:: q2m3.sqd.CCSDSeed
   :members:

Result Contracts
----------------

.. autoclass:: q2m3.sqd.SQDResult
   :members:

.. autoclass:: q2m3.sqd.ReferenceResult
   :members:

.. autoclass:: q2m3.sqd.ReferenceAttempt
   :members:

.. autoclass:: q2m3.sqd.T1Residual
   :members:

.. autoclass:: q2m3.sqd.T2Diagnostics
   :members:

Authenticated Integral Assembly
-------------------------------

These lower-level producers create the provenance records required by
``run_sqd_from_integrals``. The integral tensor uses real chemist ordering and
``e_core`` is added exactly once.

.. autoclass:: q2m3.sqd.integrals.IntegralData
   :members:

.. autofunction:: q2m3.sqd.integrals.build_integrals

.. autofunction:: q2m3.sqd.integrals.hamiltonian_id

.. autofunction:: q2m3.sqd.ansatz.build_ccsd_seed

Resource Policy
---------------

.. autofunction:: q2m3.sqd.resources.estimate_rss_mb

.. autofunction:: q2m3.sqd.resources.guard_allocation

.. autofunction:: q2m3.sqd.resources.load_resource_model

Exceptions
----------

.. automodule:: q2m3.sqd.exceptions
   :members:
   :show-inheritance:
