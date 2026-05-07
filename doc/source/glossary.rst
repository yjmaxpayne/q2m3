Glossary
========

.. glossary::

   QM/MM
      Quantum mechanics/molecular mechanics. q2m3 treats a small QM region
      with PySCF and PennyLane while representing the solvent environment with
      classical point charges and MM force-field terms.

   QPE
      Quantum phase estimation. q2m3 uses QPE circuits to estimate molecular
      energies from a time-evolution phase.

   EFTQC
      Early fault-tolerant quantum computer. In q2m3, this term is used for
      resource-estimation studies such as logical qubit and Toffoli counts.

   RDM
      Reduced density matrix. The 1-RDM is used for Mulliken population
      analysis and quantum-to-classical observable transfer.

   active space
      The selected electrons and orbitals included in a quantum calculation.
      q2m3 documents active spaces as electrons/orbitals and the resulting
      system qubit count.

   Trotterization
      Approximation of Hamiltonian time evolution by decomposing it into
      products of simpler exponentials. Trotter depth affects QPE accuracy and
      Catalyst compile size.

   delta_corr-pol
      Correlation-polarization coupling: the change in electronic correlation
      energy caused by the MM environment.

   Catalyst
      The PennyLane compilation stack used for ``@qjit`` workflows. q2m3 uses
      it where compile-once/reuse-many execution is intended.

   IR cache
      A cache of Catalyst intermediate representation artifacts used to avoid
      repeating expensive compilation stages for unchanged circuit structure.

   Hartree
      Internal energy unit used by q2m3.

   kcal/mol
      Reporting unit used only after explicit conversion from Hartree.

   Hamiltonian 1-norm
      Sum-of-coefficients measure used by the EFTQC resource-estimation path.

   MM embedding
      Electrostatic coupling of classical point charges into the QM
      Hamiltonian.

   phase wrapping
      Ambiguity caused when the energy phase exceeds the finite QPE register
      range and wraps modulo one.
