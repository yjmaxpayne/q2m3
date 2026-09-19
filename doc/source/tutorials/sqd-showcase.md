# SQD end to end: H₂, glycine, integrals, and embedding

This tutorial follows every public path added for sample-based quantum
diagonalization (SQD). You will run a full-space H₂ regression, a sparse glycine
calculation, a five-seed scan, the authenticated integral adapter, and two
fixed-point-charge embedding examples.

After completing it, you should be able to:

- choose the geometry or integral entry point;
- read SQD errors without mixing active spaces or reference tiers;
- preserve the complete `sqd.result.v1` record;
- distinguish the engine's resource audit from whole-process measurement; and
- state the physical limits of the fixed-MO embedding examples.

## Prerequisites

Run from a Linux x86_64 checkout with Python 3.11 or newer. The production
supervisor uses `/proc` and `fork`; its calibrated resource model does not cover
other platforms. The published benchmark profile used Python 3.12. Install the
locked SQD and development environments:

```bash
uv sync --frozen --extra sqd --extra dev
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
```

Set the thread variables before Python starts. SQD is CPU-based and does not
need Catalyst or a GPU. The commands below use 512 shots for a quick walkthrough;
the measured benchmark in `examples/sqd/README.md` uses 100,000 shots.

```{note}
Each command creates a new directory under `data/output/examples/` and prints
its path. Use `--output NEW_PATH` when you need a predictable location; the path
must not already exist.
```

## 1. Establish the H₂ convention

Start with the smallest complete active space:

```bash
uv run --no-sync python -m examples.sqd.h2_ground_state --shots 512
```

The script calls `q2m3.sqd.run_sqd` for H₂/STO-3G with CAS(2e,2o), four
system qubits, two LUCJ repetitions, and seed 31. Its four sampled determinants
cover the full fixed-particle-number space. Expect a completed T0
`exact_casci` reference and SQD agreement near numerical precision.

This result checks tensor conventions, energy constants, sampling conversion,
and result serialization. Full-space recovery does not demonstrate a sampling
advantage.

## 2. Run sparse glycine SQD

Use the smallest glycine space for the walkthrough:

```bash
uv run --no-sync python -m examples.sqd.glycine_ground_state \
  --active-space 6 --shots 512
```

The geometry entry performs the complete production flow:

1. build vacuum RHF molecular orbitals and choose the requested active indices;
2. assemble real chemist-order active-space integrals and `e_core`;
3. solve same-frame RCCSD and authenticate the amplitudes;
4. prepare a two-layer spin-balanced LUCJ state and sample occupations;
5. recover and diagonalize sampled determinant subspaces;
6. run the reference ladder and matched-size controls; and
7. return an immutable `SQDResult` with provenance, timings, and resource data.

The available glycine choices are CAS(6e,6o), CAS(8e,8o), and CAS(10e,10o),
which use 12, 16, and 20 system qubits. Their full closed-shell determinant
spaces contain 400, 4,900, and 63,504 alpha/beta pairs. Each choice constructs
a different Hamiltonian and reference, so total energies across rows do not
measure SQD convergence.

## 3. Measure seed spread

Run five consecutive seeds for CAS(6e,6o):

```bash
uv run --no-sync python -m examples.sqd.glycine_active_space_scan \
  --active-spaces 6 --shots 512
```

The scan launches each point in a fresh interpreter and runs them serially. It
saves every successful point immediately, records failures, and exits nonzero if
any requested point fails. The default command without `--active-spaces` runs
all three spaces and 15 calculations.

Use `statistics.json` for the mean, sample standard deviation, and range of the
executed T0 rows. These are descriptive values for five deterministic seed
choices, not confidence intervals. Missing or downgraded points never enter an
exact-reference curve.

## 4. Verify the integral adapter

The lower-level entry accepts an already assembled Hamiltonian only with its
authenticated frame and seed:

```bash
uv run --no-sync python -m examples.sqd.glycine_from_integrals \
  --active-space 6 --shots 512
```

The example first creates `IntegralData` with `build_integrals`, then creates a
same-frame `CCSDSeed` with `build_ccsd_seed`. It passes `h1`, chemist-order `h2`,
`e_core`, `IntegralContext`, and the seed to `run_sqd_from_integrals`. The script
runs the geometry entry as well and requires six reported energies plus the
frame, Hamiltonian, and active-index identities to agree within 1e-8 Ha.

Both entries share the same SQD solver. Agreement verifies adapter and assembly
consistency; it is not an independent numerical oracle. Do not transpose `h2`
or add `e_core` again. Bare `t1` and `t2` arrays cannot replace `CCSDSeed`.

Integral and CCSD preprocessing occurs before the lower-level engine call and
is therefore outside that call's `timings_s["total"]` and embedded peak RSS.
Use the complete-process monitor when that boundary matters:

```bash
uv run --no-sync python -m tools.sqd.verify_showcase \
  --output data/output/examples/tutorial-monitor \
  -- uv run --no-sync python -m examples.sqd.glycine_ground_state \
  --active-space 6 --shots 512
```

Choose a new monitor path for every run. Its 10 ms `/proc` polling covers the
command from interpreter launch through exit, including descendants, imports,
serialization, and figures. It remains an observation and can miss brief peaks.

## 5. Compare fixed-MO embedding modes

The water validation uses two asymmetric synthetic point charges and compares
diagonal with full one-electron embedding:

```bash
uv run --no-sync python -m examples.qmmm.h2o_sqd_validation --shots 512
```

The glycine example compares vacuum, diagonal, and full-one-electron treatments
in one vacuum MO frame:

```bash
uv run --no-sync python -m examples.qmmm.glycine_sqd_embedding \
  --active-space 6 --shots 512
```

For embedded runs, coordinates are Å, point charges are elementary-charge units,
and energies are Hartree. Both modes hold the vacuum MO frame and two-electron
tensor fixed. `diagonal` adds only active-space `Delta h_pp`; `full_oneelectron`
retains the complete `Delta h_pq` perturbation. `e_core` already contains the
MM-induced nuclear and frozen-core correction.

Read two effects separately:

- `baseline_energy(mode) - baseline_energy(vacuum)` is the fixed-Hamiltonian
  reference shift, to the quality of each reference;
- `delta_mHa(mode) - delta_mHa(vacuum)` is the change in SQD solver residual.

The charges are fixed test environments. These examples do not include orbital
relaxation, MM polarization, solvent sampling, correlated RDM feedback, an SQD-MC
loop, or a solvation free energy.

## Read the artifacts

The H₂, glycine, integral, and glycine-embedding runs write `input.json`, a
complete result JSON, `summary.json`, `summary.csv`, and energy/cost plots in PNG
and SVG. The H₂O validation instead writes `report.json`, per-mode result and
downstream-wrapper JSON files, summaries, and comparison plots. Scans add one
subdirectory and log per point plus `statistics.json`; the integral example adds
`assembly.json` and `consistency.json`. Failures preserve completed artifacts and
write `failure.json`.

The key result fields are:

| Field | Interpretation |
| --- | --- |
| `status` | `completed` or the explicitly requested `reference_only` mode |
| `sqd_energy` | Sampled-subspace energy in Hartree; absent for reference-only runs |
| `baseline_tier`, `baseline_method` | The actual reference path; only T0 is exact in the active space |
| `delta_mHa` | `1000 * (E_SQD - E_baseline)`; positive means SQD is higher |
| `subspace_dim`, `full_ci_dim` | Sampled Cartesian determinant space and complete fixed-particle space |
| `reference_attempts` | Ordered solver selection/execution audit |
| `provenance` | Configuration, orbital frame, Hamiltonian, seed, and input sources |
| `timings_s` | Stage timings with explicit reasons for stages that did not run |
| `diagnostics.resources` | Polled engine-process RSS scope, cap, and timing boundary |
| `null_reasons`, `warnings` | Required explanations for absent or downgraded values |

Preserve the complete `SQDResult` or its serialized `sqd.result.v1` object. An
energy without its method, tier, active space, frame, Hamiltonian, units, and
uncertainty semantics is not an auditable result.

## Resource failures and reference-only runs

`ReferenceConfig` sets one shared wall deadline and decimal-MB RSS budget. The
executor takes the tightest of the user budget, currently available host memory,
and the 8192 MB hard component cap. Predictions at a cap reject before allocation;
observed process-tree RSS at a cap terminates the run. `allow_large=True` can
cross only a soft policy boundary and cannot bypass the calibrated domain, host
availability, or hard caps.

For an intentional baseline without sampling, pass `mode="reference_only"`.
The result records absent sampling fields and their reasons. A failed full run
never silently turns into reference-only or substitutes HF for SQD.

See [](../sqd.md) for the reference ladder, full 40-field result contract,
calibrated domain, measured outcomes, and research-status boundaries. The
checkout guides `examples/sqd/README.md`, `examples/qmmm/README.md`, and
`tools/sqd/README.md` document every CLI argument and audit tool.
