# Fragment interaction examples

[English](README.md) | [简体中文](README_CN.md)

These examples connect the [calculation tutorial](../../doc/source/tutorials/fragment-interaction.md)
to the [report tutorial](../../doc/source/tutorials/fragment-report.md).
Run each module from the checkout root.
The examples use public q2m3 APIs and bundled input files.
They require no downloads at runtime.

## Learning order

| Step | Module | Learn | Default work |
|---|---|---|---|
| 1 | `water_dimer_classical` | Fragments, ghost atoms, and counterpoise | HF and frozen-core MP2 |
| 2 | `water_dimer_full` | Method differences, reference tiers, and resources | Full mode |
| 3 | `crystal_dimer_prepare` | CIF import and X–H normalization | Structure only |
| 4 | `crystal_dimer_interaction` | Interaction energies from a crystal structure | Classical mode |
| 5 | `fragment_report` | Two systems, imported references, and optional Ks | Sequential classical calculations |

## Code layout

The five modules above are executable examples.
Their calculation functions call `q2m3.fragments.run` directly.
Three private modules share example setup and output handling:

| Module | Responsibility |
| --- | --- |
| [`_inputs.py`](_inputs.py) | Bundled inputs, fixed settings, and structure exports |
| [`_common.py`](_common.py) | CLI arguments, run directories, parameter mapping, and result files |
| [`_accuracy.py`](_accuracy.py) | Acceptance thresholds and error summaries for these two examples |

These helpers belong to the examples, not the installed library API.
They do not implement energy solvers or change computed energies.
The library implements structure normalization and energy calculations.
The accuracy thresholds apply only to the declared examples and active spaces.

## Installation and execution

Core dependencies support step 1. Install ASE through `viz` for the crystal input.
Full mode also requires `sqd`, which supports Linux x86_64.
Use this installation for all five examples and their tests:

```bash
uv sync --frozen --extra dev --extra viz --extra sqd
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
```

Set these variables before Python starts. SQD requires serial native libraries.
For classical calculations and structure preparation, omit `--extra sqd`.
If worktrees share a virtual environment, set `PYTHONPATH` to the current checkout:

```bash
export PYTHONPATH="$PWD/src"
```

Run the examples in this order:

```bash
uv run --no-sync python -m examples.fragments.water_dimer_classical
uv run --no-sync python -m examples.fragments.water_dimer_full
uv run --no-sync python -m examples.fragments.crystal_dimer_prepare
uv run --no-sync python -m examples.fragments.crystal_dimer_interaction
uv run --no-sync python -m examples.fragments.fragment_report
```

Use `--help` on any entry to see its arguments.
Both crystal interaction and aggregate report entries accept `--mode full`:

```bash
uv run --no-sync python -m examples.fragments.crystal_dimer_interaction --mode full
uv run --no-sync python -m examples.fragments.fragment_report --mode full
```

## Fixed inputs

The water input is the tutorial's S22 water dimer, system 2.
Fragment A contains atoms 0–2. Fragment B contains atoms 3–5.
The crystal input is COD 2010678: 4-nitrophenol with 4-methylpyridine.
Fragment A contains 4-nitrophenol. Fragment B contains 4-methylpyridine.
See [data attribution](data/README.md) for the input sources.

| System | Basis | AVAS labels | Threshold | AB | A | B |
|---|---|---|---|---|---|---|
| Water dimer | cc-pVDZ | `O 2p`, `H 1s` | 0.7 | CAS(12e,8o) | CAS(6e,4o) | CAS(6e,4o) |
| Crystal dimer | STO-3G | `0 O 2p`, `4 N 2p` | 0.1 | CAS(12e,9o) | CAS(6e,4o) | CAS(6e,5o) |

CAS(ne,no) specifies active electrons and spatial orbitals.
SQD uses two qubits per active orbital.
AVAS selects active orbitals through projection onto atomic valence orbitals.
Indexed AVAS labels refer to the original atom order, with indices starting at zero.
Both modes select active spaces and check their correspondence.

The CIF reader uses only the asymmetric unit.
It does not expand symmetry operations or periodic images.
The crystal energy examples use normalized coordinates.
Normalization moves H atoms along their original bonds.
It preserves heavy-atom coordinates and fragment membership.
The targets are C–H = 1.083 Å, N–H = 1.009 Å, and O–H = 0.983 Å.
The [structure API](../../src/q2m3/fragments/structure.py) records the target sources.
Normalization is not geometry optimization.
It does not model hydrogen-bond polarization or identify the true hydrogen positions.

STO-3G keeps the crystal example affordable for teaching.
This minimal basis limits quantitative interaction energies.
The chosen orbitals also restrict the correlated active-space calculation.
These settings do not establish convergence with basis or active-space size.

## Outputs and failure evidence

Every entry creates a new directory under `data/output/examples/`.
Use `--output PATH` to specify a new directory.
An existing directory causes failure before calculation starts.
The repository does not track generated results.

| File | Content |
|---|---|
| `run.json` | Requested arguments, start time, status, and elapsed seconds |
| `input.json` | Exact geometry, fragments, geometry hash, and calculation settings |
| Original XYZ or CIF | Copy of the bundled input |
| `result.json` | Every public result field, including warnings and null reasons |
| `summary.csv` | Five method rows with interaction energies in Ha and kcal/mol |
| `report.md` | Library report with Chinese headings and retained warnings |
| `accuracy.json` | Full-mode SQD acceptance, subsystem errors, and interaction error |
| `failure.json` | Exception type, message, and traceback after failure |

Structure preparation saves `original.xyz`, `normalized.xyz`, `fragments.json`,
and `xh_bonds.csv`. The CSV records each H atom, its parent, and both bond lengths.
This entry does not calculate energies or create an energy report.
Crystal energy runs also retain these structure files.

The aggregate entry creates one subdirectory per system.
It saves `results.json`, `summary.csv`, and `report.md` at the aggregate level.
Its CSV identifies each system's basis and result path.
After an item fails, it continues with the next item.
The aggregate report identifies failed items and preserves successful results.
Any failed item makes the command return a nonzero exit code.
An invalid reference or Ks file stops the batch before calculations start.

## Full-mode controls

Full mode defaults to seed 0 and 100,000 shots per subsystem.
The water example uses 24 UCJ factor slots. The crystal example uses 36.
UCJ means unitary cluster Jastrow.
These counts cover `2 * n_occupied * n_virtual` for each dimer's real CCSD doubles matrix.
The same capacity covers both monomers. ffsim pads unused slots with identity operators.
The library's general default remains two factors. These examples explicitly override it.
The preparation uses CCSD amplitudes without variational optimization.
CASCI coefficients do not select sampled configurations or change SQD energies.
Classical mode does not use the following SQD or CCSD(T) controls.

| Argument | Default | Library setting |
|---|---|---|
| `--seed` | 0 | `run(seed=...)` |
| `--shots` | 100000 | `LUCJConfig.shots` |
| `--wall-budget` | 900 s | `ReferenceConfig.wall_budget_s`, per SQD subsystem call |
| `--rss-budget` | 8192 MB | `ReferenceConfig.rss_budget_mb`, per SQD subsystem call |
| `--ccsd-t-wall-budget` | 7200 s | `CCSDTBudget.wall_s`, predicted sum across AB, A, and B |
| `--ccsd-t-rss-budget` | 20000 MB | `CCSDTBudget.rss_mb`, predicted stage peak |

Memory budgets use decimal MB. Budget arguments must be finite and positive.
These arguments retain the library's budget behavior.
They do not impose a deadline on the entire example.
The SQD reference machinery can downgrade its reference when budgets restrict it.
Inspect each subsystem's `sqd.baseline_tier`, `baseline_downgrade_reason`,
`reference_attempts`, `null_reasons`, and `warnings` in `result.json`.

Full-mode acceptance requires a trustworthy T0 reference for AB, A, and B.
Each SQD energy must lie within 0.5 mHa above its CASCI reference.
A 1e-6 Ha numerical tolerance applies to the variational lower bound.
The absolute SQD interaction-energy error must not exceed 0.25 kcal/mol.
These example thresholds do not establish full-system or basis convergence.
Missing T0 references, inadequate factor capacity, or excessive errors make the command fail.
The output retains the original energies, reports, and acceptance evidence before failure.
Reducing shots can cause failure. There is no automatic retry or substitution of CASCI for SQD.

The CCSD(T) budget gate uses predicted cost before running the stage.
Its model uses cc-pVDZ calibration at eight threads.
Read the cost provenance and warnings when conditions differ.
Rejecting that stage leaves CCSD(T) null with a reason.
Other full-mode stages still run.
This command exercises that path:

```bash
uv run --no-sync python -m examples.fragments.water_dimer_full --ccsd-t-wall-budget 0.000001
```

## Imported references and experimental Ks

Supply your own measurements and reference energies:

```bash
uv run --no-sync python -m examples.fragments.fragment_report \
  --references /path/to/references.csv --ks-table /path/to/ks.json
```

The CSV must cover both `water-dimer` and `nitrophenol-methylpyridine`.
It requires these columns:

```csv
fragment_id,method,basis,delta_e,unit
```

Use `Ha`, `kcal/mol`, or `kJ/mol` for `unit`.
The importer classifies each row relative to that system's calculation basis.
Supply at most one same-basis reference and one external reference per system.
Reference geometries must match the saved input, including crystal X–H normalization.
Reference interaction energies must use the same counterpoise definition.
The importer cannot establish these scientific conditions from a CSV.

The Ks file must be a JSON object mapping labels to finite numbers.
Include the experimental conditions and units in the labels when needed.
Labels must be nonempty single lines without a vertical bar.
The aggregate directory retains both imported files.
Ks values appear alongside energies without statistical conclusions.
The examples supply no reference energies or experimental measurements.

## Read the results

All interaction energies use `ΔE = E(AB) − E(A) − E(B)`.
Each subsystem uses the dimer basis.
Ghost atoms retain basis functions without nuclei or electrons.
`result.json` records the real and ghost atom indices for each subsystem.
Negative ΔE places the dimer below the separated fragments at the specified geometry.

Classical mode supplies HF and frozen-core MP2.
Full mode also attempts frozen-core CCSD(T), active-space references, and SQD.
CASCI is exact only within the selected active space.
The SQD–CASCI difference measures the algorithm error in that space.
The CASCI–CCSD(T) difference also contains differences between correlation methods.
CCSD(T) remains approximate.
If an imported reference changes method, the reported basis difference includes that change too.

Null entries retain their reasons in JSON, CSV, and Markdown.
An empty CSV energy cell and a Markdown dash do not mean zero.
Use unrounded JSON or CSV energies when calculating differences.
The report rounds energies after calculating differences.

The QPE table estimates resources for the active-space Hamiltonian.
These examples do not execute QPE.
The SQD table reports workload and predicted memory.
Predicted memory is not a measured whole-process peak.
The two dimers use different bases, so the aggregate table is not a controlled ranking of interaction strengths.

## Measured example results

Full-mode measurements on **2026-10-25** use Python 3.11.17, PySCF 2.14.0, NumPy 2.4.1, ffsim 0.0.84,
qiskit-addon-sqd 0.13.1, and PennyLane 0.44.0.
Each native library uses one thread.
Full runs use seed 0 and 100,000 shots.
Times below describe local validation runs, not performance guarantees.

| Run | Elapsed time | Scope |
|---|---|---|
| Crystal preparation | 2.36 s | Includes initial library imports |
| Classical aggregate | 10.24 s | Both dimers and aggregate outputs |
| Water classical item | 0.30 s | Within aggregate, after initial imports |
| Crystal classical item | 7.64 s | Within aggregate, after initial imports |
| Water full item | 13.44 s | Within the full aggregate, after initial imports |
| Crystal full item | 197.58 s | Within the full aggregate |
| Full aggregate | 214.74 s | Both dimers, acceptance, and aggregate outputs |
| Water full, CCSD(T) budget rejected | 14.05 s | Remaining full-mode stages complete |

Some validation commands ran concurrently. The table reports elapsed wall time from `run.json`.
They exclude interpreter startup before the example creates its output directory.
Both full runs selected T0 references for AB, A, and B.
The library recorded its eight-thread CCSD(T) calibration warning.
Subsystem SQD records also retained the JAX/fork warning from this environment.

| Method | Water ΔE (kcal/mol) | Crystal ΔE (kcal/mol) |
|---|---|---|
| HF | −3.682 | −4.754 |
| MP2 | −3.935 | −5.251 |
| CASCI | −3.574 | −4.844 |
| SQD | −3.530 | −4.743 |
| CCSD(T) | −3.699 | −3.557 |

The classical and CASCI settings match the tutorial.
SQD retains the complete real-amplitude factor capacity for the declared active spaces.
At seed 0, SQD differs from CASCI by +0.043327 kcal/mol for water and +0.101587 kcal/mol for the crystal.
Across seeds 0–4, water errors range from +0.020358 to +0.068442 kcal/mol.
Crystal errors range from +0.101587 to +0.183548 kcal/mol.
The largest subsystem errors are 0.109069 mHa for water and 0.300421 mHa for the crystal.

An exactly disconnected pair of crystal A Hamiltonians has zero cross-fragment integrals.
Its CASCI interaction is zero within 1e-8 Ha.
The SQD interaction is +0.006214 kcal/mol at seed 0.
The regression limit is 0.05 kcal/mol.
These results establish finite-shot accuracy for the tested examples and seeds only.

The acceptance report prevents error cancellation from hiding inaccurate subsystem energies.

## Verification

Run the example and fragment tests with serial native libraries:

```bash
uv run --no-sync pytest tests/examples/test_fragment_examples.py \
  tests/examples/test_fragment_sqd_accuracy.py tests/fragments -n 0
uv run --no-sync pytest tests/examples/test_fragment_sqd_numerics.py -n 0
uv run --no-sync ruff check src/ tests/ examples/fragments/
uv run --no-sync black --check src/ tests/ examples/fragments/ --line-length 100
```

The focused tests check CLI defaults, exclusive outputs, failure evidence,
reference coverage, report values, structure preservation, and both real classical calculations.
Full-mode validation also runs both dimers and the CCSD(T) budget rejection path.

The numerical regressions use five seeds for each system through the public SQD integral workflow.
They also test doubles reconstruction and disconnected-fragment additivity.
The bounded resource probe is reproducible with this command:

```bash
uv run --no-sync python tests/sqd/ucj_profile_probe.py /tmp/ucj-profile.json 100
```

The resource profile supports at most 36 factor slots and 100,000 shots.
It supports balanced closed shells with 2–10 active orbitals, at most 252 determinants per spin, and 63504 total.
It preserves the original solver settings.
Its import/native reserve is 1024 MB per stage.
The probe measures preparation and sampling in 27 isolated processes with fixed thread counts.
Production calls retain their process-tree RSS caps and wall deadlines.

Return to the [example guide](../README.md).
