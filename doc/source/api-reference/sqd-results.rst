SQD result fields
=================

The following 40 names match the ``sqd.result.v1`` contract. JSON preserves
these names and uses ``null`` for Python ``None``. Every absent top-level value
requires a ``null_reasons`` entry. Required metadata is not omitted. Nested
objects retain their own units and reasons. A consumer must keep the method,
tier, downgrade reason and warnings beside the reference energy.

.. list-table::
   :header-rows: 1

   * - Field
     - Unit
     - Meaning and absence rule
   * - ``schema_version``
     - metadata
     - Required, ``sqd.result.v1``
   * - ``status``
     - metadata
     - Required, ``completed`` or ``reference_only``
   * - ``sqd_energy``
     - Ha
     - SQD energy. Absent in reference-only mode
   * - ``hf_energy``
     - Ha
     - Required same-Hamiltonian determinant expectation
   * - ``hf_reference_kind``
     - metadata
     - Required ``canonical_rhf`` or ``fixed_frame_determinant``
   * - ``baseline_energy``
     - Ha
     - Required selected reference energy
   * - ``iso_active_space_ccsd_energy``
     - Ha
     - Same-space CCSD. May be absent if not run in reference-only mode
   * - ``iso_ndet_sci_energy``
     - Ha
     - Matched-spin-dimension SCI. Absent in reference-only mode
   * - ``iso_ndet_random_energy``
     - Ha
     - Matched random comparison. Absent in reference-only mode or with an explicit
       unavailable/budget reason
   * - ``baseline_tier``
     - metadata
     - Required T0/T1/T1+/T2. Never infer from a legacy alias
   * - ``baseline_method``
     - metadata
     - Required actual method. CASCI stays ``exact_casci``
   * - ``baseline_uncertainty_mHa``
     - mHa
     - T0 is zero. Unknown approximate uncertainty is absent with reason
   * - ``baseline_uncertainty_kind``
     - metadata
     - Required ``exact_active_space``, ``solver_estimate`` or ``unknown``
   * - ``baseline_downgrade_reason``
     - metadata
     - Absent for T0. Required nonempty text for every lower tier
   * - ``baseline_untrustworthy``
     - boolean
     - Required trust flag. Does not turn an approximation into T0
   * - ``baseline_t1_residual``
     - structured metadata
     - T1 cutoff/energy record. Absent when inapplicable. Energies are Ha, loose residual is mHa,
       and cannot bound the tight energy error
   * - ``t1_diagnostic``
     - dimensionless
     - T2 CCSD amplitude diagnostic. Absent when T2 is not used
   * - ``t2_diagnostics``
     - structured metadata
     - T2 flags, CCSD convergence, energies in Ha and dimensionless ratios. Absent when inapplicable
   * - ``reference_attempts``
     - structured metadata
     - Required ordered attempt records, including reasons, wall seconds and decimal MB peak where
       known
   * - ``delta_mHa``
     - mHa
     - SQD minus baseline. Absent in reference-only mode
   * - ``delta_vs_sci_mHa``
     - mHa
     - SQD minus same-size SCI. Absent in reference-only mode
   * - ``ratio_sqd_over_sci``
     - dimensionless
     - T0-relative error ratio. Absent for nonexact baseline, invalid denominator or reference-only
       mode, with reason
   * - ``unique_dets_vs_shots``
     - counts
     - Cumulative ``(shots, unique sampled pairs)`` curve. Absent in reference-only mode
   * - ``active_space``
     - counts
     - Required ``(electrons, spatial orbitals)``
   * - ``n_reps``
     - count
     - Executed LUCJ repetitions. Absent in reference-only mode
   * - ``shots``
     - count
     - Executed sampling shots. Absent in reference-only mode
   * - ``seed``
     - integer
     - Required actual random seed, including reference-only runs
   * - ``subspace_dim``
     - count
     - Product of alpha/beta dimensions. Absent in reference-only mode
   * - ``subspace_dims``
     - counts
     - ``(d_alpha, d_beta)``. Absent in reference-only mode
   * - ``full_ci_dim``
     - count
     - Required full fixed-particle-number space dimension
   * - ``embedding_mode``
     - metadata
     - Required actual embedding mode
   * - ``two_electron_tensor_fixed``
     - boolean
     - Required fixed-tensor assumption
   * - ``fixed_mo``
     - boolean
     - Required orbital-frame assumption
   * - ``backend``
     - metadata
     - Executed sampling backend. Absent in reference-only mode
   * - ``versions``
     - mapping
     - Required exact dependency version snapshot
   * - ``diagnostics``
     - mapping
     - Required additional diagnostics. Retain embedded flags/reasons
   * - ``provenance``
     - mapping
     - Required immutable input/frame/Hamiltonian/seed source snapshot
   * - ``timings_s``
     - s
     - Required geometry/integrals/ccsd/prepare/sample/diagonalize/reference/comparison/total keys.
       Unexecuted stages are absent with reasons, not fabricated zero timings
   * - ``null_reasons``
     - mapping
     - Required reasons for missing values, including ``reference_only``, ``not_applicable`` or the
       actual failure-to-define reason
   * - ``warnings``
     - metadata
     - Required warning collection. Every non-T0 baseline has a visible warning
