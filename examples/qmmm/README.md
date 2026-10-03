# Fixed-MO embedding and QM/MM workflows

Follow this learning order from the checkout root.

```bash
uv sync --frozen --extra dev --extra sqd --extra catalyst --extra solvation
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

1. [Fixed-MO one-electron embedding: diagonal vs full matrix](full_oneelectron_embedding.py):
   `uv run --no-sync python -m examples.qmmm.full_oneelectron_embedding`
2. [Water CAS(4e,4o) SQD with asymmetric fixed charges](h2o_sqd_validation.py):
   `uv run --no-sync python -m examples.qmmm.h2o_sqd_validation`
3. [Glycine CAS(10e,10o): vacuum, diagonal, full_oneelectron](glycine_sqd_embedding.py):
   `uv run --no-sync python -m examples.qmmm.glycine_sqd_embedding`
4. [H₂ QPE-driven MC introduction](h2_mc_solvation.py):
   `uv run --no-sync python -m examples.qmmm.h2_mc_solvation`
5. [H₃O⁺ QPE-driven MC](h3o_mc_solvation.py):
   `uv run --no-sync python -m examples.qmmm.h3o_mc_solvation`
6. [Existing three-mode QPE–MC comparison](h2_three_mode_comparison.py):
   `uv run --no-sync python -m examples.qmmm.h2_three_mode_comparison`

SQD entries require only the `sqd` extra. MC entries require the `catalyst`
and `solvation` extras. These examples do not add SQD–MC integration. The
scripts state their built-in geometries, charges, and mode settings explicitly.

The glycine example accepts `--active-space 6|8|10`, `--seed`, `--shots`, and
`--output`. It saves complete per-mode results, summaries, and PNG/SVG figures in
`data/output/examples/`. Its two artificial TIP3P waters contain six charges with
a net charge of zero. They are a fixed environment, not a sampled solvent.

Vacuum MOs and two-electron tensors remain fixed. The model excludes
polarization and orbital relaxation. Separate reference energy shifts from
changes in the SQD solver residual. These are not solvation free energies.

Each glycine mode has a budget of 900 s and 8192 decimal MB. The water example
keeps its existing budget of 120 s and 8192 MB. MC and Catalyst compilation can
be substantially more expensive. Start with H₂. Legacy MC outputs keep their
existing locations.

Return to the [capability map](../README.md).
