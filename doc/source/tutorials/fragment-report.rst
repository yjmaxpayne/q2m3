Read A Fragment Interaction Report
==================================

The water-dimer calculation gives five interaction energies. Reading them
together shows how the method and active space affect the result.
This tutorial explains the energy differences and the associated resource
estimates.

You can read the tables below without running Python. To reproduce them,
follow :doc:`Fragment Interaction Energy <fragment-interaction>` with cc-pVDZ
and ``mode="full"``. The generated report uses Chinese headings. The
explanations below follow its table order.

Start With The Interaction Energy
---------------------------------

The first table compares five methods for the same water-dimer geometry.
Each row gives the counterpoise-corrected interaction energy,
:math:`\Delta E = E(AB) - E(A) - E(B)`. AB denotes the dimer, and A and B denote
the two water molecules. All three energies use the dimer basis.

.. list-table::
   :header-rows: 1

   * - Method
     - Calculation
     - ΔE (kcal/mol)
   * - HF
     - Hartree–Fock
     - −3.682
   * - MP2
     - Second-order Møller–Plesset perturbation theory with a frozen core
     - −3.935
   * - CASCI
     - Exact diagonalization within the selected active space
     - −3.574
   * - SQD
     - Sample-based quantum diagonalization within the same active space
     - −3.557
   * - CCSD(T)
     - Coupled cluster with perturbative triples and a frozen core
     - −3.699

All five values are negative. Each method therefore places the dimer below
the separate fragments at this geometry. A more negative value represents
a stronger calculated interaction. The spread from −3.935 to −3.557 kcal/mol
shows how the result changes with the electronic structure method.

The report shows each energy in Ha and kcal/mol, with six and three
decimal places, respectively. A dash means that the calculation has no
value for that entry. The adjacent column explains why. For example,
``classical_only`` mode supplies HF and MP2 energies. In ``full`` mode, CCSD(T)
runs when its estimated cost fits the calculation budget.

Compare SQD With CASCI
----------------------

CASCI provides an exact solution within the selected active space.
SQD approximates that solution in the same space. Their difference therefore
measures the SQD algorithm error for this calculation:

.. math::

   \epsilon_{\mathrm{algorithm}} = \Delta E_{\mathrm{SQD}} - \Delta E_{\mathrm{CASCI}}.

The example report gives +0.016 kcal/mol. SQD predicts a slightly less
negative interaction energy than CASCI. This comparison isolates the
algorithm's contribution because both methods use the same active space.

The report calculates differences before rounding the energies.
Subtracting the rounded table entries can therefore give a slightly
different last digit.

Compare The Active Space With A Larger Calculation
--------------------------------------------------

The next comparison asks how much the selected active space changes the
interaction energy. In this example, the reference is frozen-core CCSD(T)
in the full orbital space with the same cc-pVDZ basis:

.. math::

   \epsilon_{\mathrm{active\ space}} = \Delta E_{\mathrm{CASCI}} - \Delta E_{\mathrm{CCSD(T)}}.

This difference is +0.125 kcal/mol. The active-space CASCI calculation
predicts a less negative interaction energy than the CCSD(T) reference.
Here, this difference exceeds the SQD–CASCI difference.
Choosing the active space therefore deserves attention alongside the SQD
settings when you refine this calculation.

The report calls this quantity the *active-space error*. It measures the
difference from the chosen reference, which here also changes the
correlation method. CCSD(T) remains an approximate reference.
The source label ``pyscf_ccsd_t`` identifies the CCSD(T) calculation from
the same run. If that calculation has no value, q2m3 can use an imported
reference with the same basis.

Add An External Reference
-------------------------

A reference with another basis extends the comparison beyond the current
calculation. The report combines it with the same-basis reference to give
two further differences:

.. list-table::
   :header-rows: 1

   * - Quantity
     - Definition
   * - Basis error
     - Same-basis reference − external reference
   * - Model error
     - CASCI − external reference

Each term uses interaction energies. When all references are available,
the model error equals the active-space error plus the basis error.
The report shows each difference with its sign and its absolute value,
in Ha and kcal/mol.

The water-dimer example supplies no external reference, so these two entries
show a dash and ``missing external reference``. To fill them, supply a
reference file as described in :ref:`Adapt The Calculation <fragment-adapt-calculation>`.
Choose a reference with the same geometry and energy definition.
If its method also differs, the reported basis error includes that method
difference.

Relate The Active Space To Computational Cost
---------------------------------------------

The final two tables help plan calculations for the dimer and its monomers.
They describe QPE resources and SQD workload separately.

The QPE table estimates the cost of quantum phase estimation with a
double-factorized active-space Hamiltonian. It lists the Hamiltonian norm
parameter λ in Ha, logical qubits, and Toffoli gates for AB, A, and B.
These estimates describe a possible QPE calculation. The energies above
come from the electronic structure methods in the first table.

The SQD table connects the active-space size to the simulated workload:

.. list-table::
   :header-rows: 1

   * - Subsystem
     - SQD qubits
     - Full CI dimension
     - Prepare RSS (MB)
     - Sample RSS (MB)
   * - AB
     - 16
     - 784
     - 1031.7
     - 1031.7
   * - A
     - 8
     - 16
     - 1024.5
     - 1024.5
   * - B
     - 8
     - 16
     - 1024.5
     - 1024.5

SQD uses two qubits per active orbital. The full configuration-interaction
(CI) dimension counts the determinants for the active space and its spin populations.
RSS means resident set size. The two RSS columns predict peak memory for
the prepare and sample stages in decimal MB.

Read these estimates when you choose a larger active space or plan a series
of calculations. For an energy comparison, start with the interaction
energies and their differences from the selected references.

Include Experimental Ks Values
------------------------------

If you also have experimental Ks values, pass a mapping of labels to values
as ``ks_table`` in ``render_report``. The report appends them in the supplied
order, so collaborators can read the measurements alongside the calculations.
Statistical analysis of their relationship to ΔE remains a separate step.
The water-dimer example has no Ks table.
