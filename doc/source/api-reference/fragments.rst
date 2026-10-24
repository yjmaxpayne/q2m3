Fragment Interaction API
========================

The fragment API computes counterpoise-corrected interaction energies of
two-fragment dimers. ``run`` and ``Molecule`` are also top-level exports of
``q2m3``. ``mode="classical_only"`` needs only the core install.
``mode="full"`` needs the ``sqd`` extra and Linux x86_64. See
:doc:`../tutorials/fragment-interaction` for a complete run.

Workflow
--------

.. autofunction:: q2m3.fragments.run

.. autoclass:: q2m3.fragments.Molecule
   :members: from_xyz, charge, geometry_sha256

Result Contracts
----------------

.. autoclass:: q2m3.fragments.result.FragmentInteractionResult
   :members: validate

.. autoclass:: q2m3.fragments.result.SubsystemResult
   :members: validate

References And Report
---------------------

.. autofunction:: q2m3.fragments.import_references

.. autoclass:: q2m3.fragments.result.ExternalReference

.. autofunction:: q2m3.fragments.render_report

Exceptions
----------

.. automodule:: q2m3.fragments.errors
   :members:
