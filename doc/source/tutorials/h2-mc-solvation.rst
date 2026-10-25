H2 MC Solvation
===============

Run fixed-mode H2 Monte Carlo solvation with ``q2m3.solvation``.
Install the ``catalyst`` and ``solvation`` extras first.

Run The Script
--------------

.. code-block:: bash

   uv run python examples/qmmm/h2_mc_solvation.py

The script builds a ``SolvationConfig``, runs ``run_solvation()``, and reports
whether the Catalyst IR cache was a hit or miss.

Configuration Pattern
---------------------

.. code-block:: python

   from q2m3.solvation import MoleculeConfig, QPEConfig, SolvationConfig, run_solvation

   h2 = MoleculeConfig(
       name="H2",
       symbols=["H", "H"],
       coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
       charge=0,
       active_electrons=2,
       active_orbitals=2,
       basis="sto-3g",
   )

   config = SolvationConfig(
       molecule=h2,
       qpe_config=QPEConfig(
           n_estimation_wires=4,
           n_trotter_steps=10,
           n_shots=50,
           qpe_interval=10,
           target_resolution=0.003,
           energy_range=0.2,
       ),
       hamiltonian_mode="fixed",
       n_waters=10,
       n_mc_steps=100,
       temperature=300.0,
       random_seed=42,
   )

   result = run_solvation(config, show_plots=False)

Fixed Mode
----------

``fixed`` mode uses a vacuum QPE Hamiltonian.
It adds MM energy terms during the MC loop.
Use this example to check Catalyst, JAX, PySCF, and the IR cache together.

The first run can compile Catalyst IR. Later runs can reuse cached IR if the
circuit structure is unchanged.

Output Fields
-------------

The result dictionary includes energy trajectories, acceptance statistics,
timing data, circuit metadata, and optional cache statistics. Diagnostics can
add result keys. Code that reads results should use ``.get()`` for optional
metadata.

Memory Notes
------------

Use H2 fixed mode for the first MC run. H3O+ and dynamic mode increase Hamiltonian
term count, circuit size, and Catalyst compile cost. Treat them as diagnostics
until the H2 run passes its checks.
