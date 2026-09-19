Package API
===========

Top-Level Package
-----------------

.. automodule:: q2m3

Primary Exports
---------------

* :class:`q2m3.QuantumQMMM`
* :class:`q2m3.QPEEngine`
* :class:`q2m3.QMMMSystem`
* :class:`q2m3.PySCFPennyLaneConverter`
* :class:`q2m3.UnifiedDensityMatrix`
* :class:`q2m3.MoleculeConfig`
* :func:`q2m3.load_xyz`
* :func:`q2m3.save_json_results`
* :func:`q2m3.run_sqd` (available with the ``sqd`` extra)

Optional SQD Exports
--------------------

``q2m3.run_sqd`` is resolved lazily and appears in ``q2m3.__all__`` only when
both SQD backend modules can be located. The integral entry point and typed
contracts live under :mod:`q2m3.sqd`; see :doc:`sqd`.

Constants
---------

.. automodule:: q2m3.constants
   :members:

Molecule Configuration
----------------------

.. automodule:: q2m3.molecule
   :members:
   :show-inheritance:
