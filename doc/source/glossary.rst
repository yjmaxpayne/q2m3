Glossary
========

.. glossary::

   QM/MM
      Quantum mechanics/molecular mechanics. q2m3 treats a small QM region
      with PySCF and PennyLane. It represents the solvent environment with
      classical point charges and MM force-field terms.

   QPE
      Quantum phase estimation. q2m3 uses QPE circuits to estimate molecular
      energies from a time-evolution phase.

   SQD
      Sample-based quantum diagonalization. q2m3 prepares a CCSD-seeded LUCJ
      state in ffsim, samples fixed-particle-number determinants, and solves the
      Hamiltonian in the recovered subspace with qiskit-addon-sqd.

   HF
      Hartree-Fock. A mean-field electronic reference.
      RHF denotes restricted Hartree-Fock for the closed-shell workflows.

   CCSD
      Coupled cluster with single and double excitations.
      CCSD(T) adds a perturbative triples correction.

   CASCI
      Complete active-space configuration interaction.
      The SQD T0 reference solves the finite active-space Hamiltonian exactly.

   SCI
      Selected configuration interaction. The calculation retains selected determinants.

   AVAS
      Atomic valence active space. The fragment workflow uses AO labels to select
      active orbitals. AO means atomic orbital.

   MO
      Molecular orbital. Fixed-MO embedding retains the vacuum orbital frame.

   LUCJ
      Local unitary cluster Jastrow ansatz used by the SQD workflow. Its
      repetition count and optional spin-channel connectivity are explicit run
      inputs.

   reference tier
      The SQD baseline policy: T0 exact active-space CASCI, T1 selected CI,
      explicitly registered T1+ plugins, or diagnostic T2 CCSD(T). A tier
      records method and uncertainty semantics. It is not an accuracy label by
      itself.

   EFTQC
      Early fault-tolerant quantum computer. q2m3 uses this term for
      resource-estimation studies such as logical qubit and Toffoli counts.

   RDM
      Reduced density matrix. q2m3 uses the 1-RDM for Mulliken population
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
      it for repeated execution of compiled circuits.

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
      Ambiguity from measuring phase modulo one. Different energies can map
      to the same phase at a given evolution time.
