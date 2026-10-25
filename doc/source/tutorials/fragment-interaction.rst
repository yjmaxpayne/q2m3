Fragment Interaction Energy
===========================

The interaction between two water molecules changes their total energy.
This tutorial calculates that change for the S22 water dimer. You will define
the two fragments, choose an active space, and compare five electronic
structure methods in one calculation.

From Fragments To Interaction Energy
------------------------------------

Each water molecule is one fragment. We call them A and B, and call the
complete dimer AB. For each method, q2m3 calculates the interaction energy as

.. math::

   \Delta E = E(AB) - E(A) - E(B).

A negative value means that the dimer has lower energy than the separate
fragments at the same geometry. All three calculations use the dimer basis.
For each monomer, the partner's atoms retain their basis functions but
contribute no nuclei or electrons. These are *ghost atoms*. This procedure
gives the counterpoise-corrected interaction energy.

The workflow first calculates Hartree–Fock (HF) and second-order
Møller–Plesset perturbation theory (MP2) energies. It then compares two
methods within a selected active space: complete active-space configuration
interaction (CASCI) and sample-based quantum diagonalization (SQD).
A frozen-core CCSD(T) calculation supplies a reference from the full orbital
space when its estimated cost fits the calculation budget.

Prepare The Calculation
-----------------------

The example uses the default ``mode="full"`` and the cc-pVDZ basis.
Install the optional SQD backend from the repository root:

.. code-block:: bash

   uv sync --frozen --extra sqd

The SQD backend supports Linux x86_64. For a calculation with only core
dependencies, use ``mode="classical_only"`` in the ``run`` call below. This mode
returns HF and MP2 energies and the selected active spaces. It skips CASCI,
SQD, CCSD(T), and resource estimates.

Download :download:`water-dimer.xyz` into your working directory.
The coordinates use Angstrom. Atom indices start at zero, so atoms 0–2 form
fragment A and atoms 3–5 form fragment B.

.. literalinclude:: water-dimer.xyz
   :language: text

Choose The Active Space
-----------------------

CASCI and SQD treat the same selected electrons and orbitals explicitly.
The atomic valence active space (AVAS) procedure selects these orbitals from
the atomic shells you specify. The ``avas_labels`` parameter identifies the
shells, and ``avas_threshold`` controls the selection. Both parameters require
an explicit value.

For this example, use ``avas_labels=["O 2p", "H 1s"]`` and
``avas_threshold=0.7``. The resulting spaces are:

.. list-table::
   :header-rows: 1

   * - Subsystem
     - Active electrons
     - Active orbitals
     - SQD qubits
   * - Dimer AB
     - 12
     - 8
     - 16
   * - Monomer A
     - 6
     - 4
     - 8
   * - Monomer B
     - 6
     - 4
     - 8

An element label such as ``"O 2p"`` selects that shell on every oxygen atom.
An indexed label such as ``"0 O 2p"`` refers only to atom 0 in the full
geometry. q2m3 applies the same labels to all three subsystems and retains
only labels on real atoms in each monomer.

The current workflow accepts at most 10 active orbitals per subsystem.
For these labels, the PySCF default threshold of 0.2 selects 12 orbitals for
the dimer. The example uses 0.7 to obtain the smaller spaces above.
q2m3 also checks that the monomer active spaces correspond to the dimer
active space before it calculates correlated energies.

Calculate And Display The Energies
----------------------------------

Run the following Python code from the directory containing
``water-dimer.xyz``. ``Molecule.from_xyz`` assigns the atoms to the two
fragments. ``run`` calculates their interaction energies, and ``render_report``
formats the result as Markdown.

.. testsetup::

   import os

   # The documentation test starts from the repository root.
   _cwd = os.getcwd()
   os.chdir("doc/source/tutorials")

.. testcleanup::

   os.chdir(_cwd)

.. doctest::

   >>> from q2m3 import Molecule, run
   >>> from q2m3.fragments import render_report
   >>> dimer = Molecule.from_xyz("water-dimer.xyz", fragments=[(0, 1, 2), (3, 4, 5)], name="water-dimer")
   >>> result = run(dimer, avas_labels=["O 2p", "H 1s"], avas_threshold=0.7)
   >>> print(render_report([result]))  # doctest: +ELLIPSIS
   # 片段相互作用能报告
   ...
   ## 片段 water-dimer（模式 full，基组 cc-pvdz）
   ...
   | HF | -0.005868 | -3.682 | |
   | MP2 | -0.006272 | -3.935 | |
   | CASCI | -0.005695 | -3.574 | |
   | SQD | -0.005669 | -3.557 | |
   | CCSD(T) | -0.00589... | -3.699 | |
   ...

The report currently uses Chinese headings. Its energy columns give
:math:`\Delta E` in Hartree (Ha) and kcal/mol. In this example, every method gives
a negative interaction energy. SQD gives about −3.557 kcal/mol, close to
the CASCI value of −3.574 kcal/mol within the same active space.

The ``result`` object also stores the energies, active spaces, and resource
estimates for AB, A, and B. Use ``result.delta_e`` to access the interaction
energies in Hartree. Continue with :doc:`Read A Fragment Interaction Report <fragment-report>` to interpret the method differences and resource
estimates.

.. _fragment-adapt-calculation:

Adapt The Calculation
---------------------

For another dimer, replace the geometry and fragment indices first.
Then choose AVAS labels for the shells relevant to that interaction.
Inspect the selected spaces before you compare energies across geometries.

If the selected monomer and dimer spaces do not correspond, q2m3 raises
``ActiveSpaceMismatchError``. Revisit the labels and threshold for the whole
dimer. ``FragmentWorkflowError`` can also indicate an unsupported active space,
such as one without an active virtual orbital or with more than 10 orbitals.
The exception message identifies the subsystem and the failed condition.

To compare against a reference calculation, add
``references=["refs.csv"]`` to the ``run`` call. The CSV needs these columns:

.. list-table::
   :header-rows: 1

   * - Column
     - Content
   * - ``fragment_id``
     - The molecule name, ``water-dimer`` in this example
   * - ``method``
     - The reference method
   * - ``basis``
     - The reference basis
   * - ``delta_e``
     - The reference interaction energy
   * - ``unit``
     - ``Ha``, ``kcal/mol``, or ``kJ/mol``

Use a reference for the same geometry and interaction-energy definition.
q2m3 classifies a row with the calculation basis as a same-basis reference.
It classifies a row with another basis as an external reference.
Supply at most one reference in each category.
The next tutorial explains how these references enter the energy comparisons.

Keep the geometry and calculation settings with your result.
``result.provenance`` records the geometry hash, basis, AVAS settings, seed,
package versions, and thread settings. Use these records when you repeat
the calculation or compare results from different environments.
