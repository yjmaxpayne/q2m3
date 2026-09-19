# SQD end to end: glycine active spaces

The learning path is H₂ → glycine → active-space scan → explicit integrals → fixed
point-charge embedding. Install `uv sync --frozen --extra sqd --extra dev` from the
checkout root and set native thread counts before Python starts:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
uv run --no-sync python -m examples.sqd.h2_ground_state
uv run --no-sync python -m examples.sqd.glycine_ground_state --active-space 6
uv run --no-sync python -m examples.sqd.glycine_ground_state --active-space 8
uv run --no-sync python -m examples.sqd.glycine_ground_state
uv run --no-sync python -m examples.sqd.glycine_active_space_scan
uv run --no-sync python -m examples.sqd.glycine_from_integrals
uv run --no-sync python -m examples.qmmm.glycine_sqd_embedding
```

Glycine uses the repository's neutral STO-3G geometry. CAS(6e,6o), CAS(8e,8o) and
CAS(10e,10o) use 12, 16 and 20 system qubits and full fixed-particle determinant
spaces of 400, 4,900 and 63,504. Actual MO indices are printed and saved. The
production engine builds vacuum RHF orbitals, active integrals and a CCSD seed;
two LUCJ layers prepare the sampling state, then 100,000 occupation samples feed
SQD. Seed 0 is the single-run default; the scan executes seeds 0–4 independently.

Read the terminal's energy table alongside the saved `summary.csv`. HF, CCSD,
same-size SCI and random subspaces remain visible even when they outperform SQD.
`energies.png/svg` separates total energies across spaces from errors against
that space's own T0/exact CASCI reference. T1/T1+/T2 results retain their actual
method, uncertainty and downgrade reason; none enter the exact-error curve.
Finite-seed spread is not a confidence interval. H₂'s full-space recovery checks
conventions; it does not show a sampling advantage.

`cost.png/svg` displays subspace fractions, stage times and internal polled RSS.
The subspace fraction is not a simulator-memory speedup: ffsim still represents
the fixed-particle-number state. Internal RSS covers parent plus simultaneous
descendants through the start of final result construction; it excludes imports
and figures. Explicit-integral preprocessing is recorded separately. Independent
complete-process measurement is available via `python -m tools.sqd.verify_showcase`.

Every run writes a new directory under `data/output/examples/`, or the supplied
`--output`. Full `sqd.result.v1` JSON accompanies the table, plots and input data.
The scan launches one fresh interpreter per point, saves each immediately and
returns nonzero on any failed point. No point is silently reduced or filled from
historical data. Each engine call has a 900 s/8192 decimal MB default budget;
allow minutes for the scan, and up to 15 such engine budgets on slower machines.

The integral example explicitly builds chemist integrals and the authenticated
same-frame seed, then compares `run_sqd_from_integrals` with `run_sqd`. This tests
adapter consistency with a shared solver, not independent numerical correctness.
The glycine embedding example uses two artificially placed neutral TIP3P waters,
compares vacuum/diagonal/full-one-electron treatments in the same vacuum MO frame,
and reports reference shifts separately from SQD residual changes. It has no
orbital relaxation, MM polarization, solvent sampling or solvation free energy.

For argument details and artifact layout, read `examples/sqd/README.md` and
`examples/qmmm/README.md` in the checkout. Calibration and historical audits live
under `tools/sqd/`. Measured example results are included in `examples/sqd/README.md`.
