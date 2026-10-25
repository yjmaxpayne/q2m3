Getting started
===============

Install q2m3 from source with Python 3.11 or newer and ``uv``.
Run the H2 examples before larger SQD or Catalyst calculations.

Install
-------

.. code-block:: bash

   git clone https://github.com/yjmaxpayne/q2m3.git
   cd q2m3
   uv sync --frozen

``uv run`` uses the project's ``.venv``. Activation is optional.
The core dependencies include PySCF, PennyLane, NumPy, SciPy, Matplotlib, and Rich.

Add extras for the workflow you need:

.. list-table::
   :header-rows: 1

   * - Workflow
     - Command
   * - Development
     - ``uv sync --frozen --extra dev``
   * - SQD
     - ``uv sync --frozen --extra sqd``
   * - Catalyst solvation
     - ``uv sync --frozen --extra catalyst --extra solvation``
   * - Documentation
     - ``uv sync --frozen --extra docs --extra sqd --extra catalyst --extra solvation --extra viz``

The ``gpu`` extra requires compatible NVIDIA drivers and CUDA runtime support.
CPU examples do not require it.

Run H2 validation
-----------------

.. code-block:: bash

   uv run python examples/qpe/h2_qpe_validation.py
   uv run python examples/resources/h2_resource_estimation.py

The first script compares vacuum and MM-embedded HF and QPE energies.
Inspect every ``[OK]`` and ``[FAIL]`` line. Its exit status does not report failed checks.
The second script estimates quantum hardware resources.
See :doc:`tutorials/h2-qpe-validation` and :doc:`tutorials/h2-resource-estimation` for interpretation.

Energies use Hartree internally. Convert explicitly before reporting kcal/mol:

.. doctest::

   >>> round(HARTREE_TO_KCAL_MOL * KCAL_TO_HARTREE, 12)
   1.0

Choose the next workflow
------------------------

.. list-table::
   :header-rows: 1

   * - Workflow
     - Requirements and scope
   * - :doc:`tutorials/h2-mc-solvation`
     - Catalyst and solvation extras. 8 GB+ RAM recommended.
   * - :doc:`tutorials/sqd-showcase`
     - SQD extra, Linux x86_64, and serial native threads.
   * - :doc:`tutorials/fragment-interaction`
     - Core install for ``classical_only``. SQD extra and Linux x86_64 for ``full``.
   * - H3O+ MC diagnostics
     - Catalyst and solvation extras. 16 GB+ RAM recommended.
   * - 8-bit QPE and dynamic-Trotter diagnostics
     - 30 GB+ RAM recommended. Use the scripts' memory guards.

The memory recommendations are not measured upper bounds.
See :doc:`tutorials/index` for diagnostic script paths.
Catalyst compilation depends on estimation wires, Trotter depth, and Hamiltonian term count.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1

   * - Symptom
     - Action
   * - Missing ``catalyst`` or ``jax``
     - Install the ``catalyst`` and ``solvation`` extras.
   * - Missing ``ffsim`` or ``qiskit_addon_sqd``
     - Install the ``sqd`` extra.
   * - SQD rejects the platform, threads, or memory request
     - Check :doc:`user-guide/sqd`. ``allow_large`` cannot bypass domain or hard limits.
   * - Slow first MC run
     - Allow compilation to finish. Later runs can reuse compatible IR.
   * - H3O+ run exceeds memory
     - Start with H2. Reduce Trotter depth or use more memory.
   * - GPU device is unavailable
     - Check ``q2m3.core.device_utils``. CPU options include ``lightning.qubit`` and
       ``default.qubit``.
   * - Large QPE–HF difference
     - Check estimation wires, ``base_time``, energy shift, and Trotter depth.

Read :doc:`user-guide/core-concepts` before interpreting energy differences as chemistry.
