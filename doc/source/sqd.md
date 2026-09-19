# Sample-based quantum diagonalization (SQD)

SQD prepares a CCSD-seeded LUCJ state with ffsim, samples determinants, and
solves the Hamiltonian in the recovered subspace with qiskit-addon-sqd. The
public workflows return an `SQDResult` with same-Hamiltonian reference and
comparison energies. This is a classical simulator implementation; it does
not demonstrate quantum advantage or execution on quantum hardware.

## Installation and platform

Use this source checkout on Linux x86_64 for the SQD implementation and examples:

```bash
uv sync --frozen --extra sqd
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
```

The public supervisor requires Linux x86_64 because it uses `/proc` and
`multiprocessing` with `fork`; other platforms are rejected before execution.
SQD requires ffsim and qiskit-addon-sqd; Catalyst is optional. Add
`--extra catalyst` only when also using Catalyst workflows. The unified
`uv.lock` resolves PySCF 2.14.0 even for a core-only locked checkout; the prior
core environment used 2.11.0. This affects shared QPE/MM dependencies too,
so an old core environment is not equivalent to a fresh locked install.
The measured profile is Linux x86_64, Python 3.12.3, NumPy 2.4.1, SciPy
1.16.3, ffsim 0.0.84, Qiskit 2.5.2 and qiskit-addon-sqd 0.13.1. Resource
certification does not extend to other platforms or arbitrary versions.
Catalyst may be absent; when installed, its version must still match the
calibration environment. All required numerical packages remain version checked.

For a guided progression through every maintained example, use the
[](tutorials/sqd-showcase.md). Public signatures and typed contracts are listed
in [](api-reference/sqd.rst).

Ordinary `import q2m3` retains the existing PennyLane import cost. SQD's
algorithm does not call PennyLane or Catalyst. Its workflow exports are lazy;
missing the extra raises `ImportError` with the installation command. Check
`"run_sqd" in q2m3.__all__` for availability, rather than `hasattr`, which can
trigger that error. Configuration/result types do not import ffsim or Qiskit.

## Two runnable public entry points

Save either Python block as a file and run `uv run --no-sync python FILE.py`
after the installation above. Both use H₂ at 0.74 Å/STO-3G, explicitly
`(2 electrons, 2 spatial orbitals)`, or four system qubits. The small
128-shot smoke case uses seed 31 and two LUCJ repetitions. It reaches the
full four-determinant space; agreement with CASCI is a regression check,
not evidence that sampling outperforms classical solvers.

The geometry entry point builds integrals and a same-frame CCSD seed:

```python
import numpy as np
from q2m3 import run_sqd
from q2m3.sqd import LUCJConfig

result = run_sqd(
    ["H", "H"],
    np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
    active_electrons=2,
    active_orbitals=2,
    basis="sto-3g",
    lucj=LUCJConfig(n_reps=2, shots=128),
    seed=31,
    verbose=False,
)
result.validate()
assert result.baseline_tier == "T0"
assert abs(result.sqd_energy - result.baseline_energy) < 1e-10
print(result.sqd_energy, result.baseline_method, result.delta_mHa)
```

The integral entry point accepts already assembled chemist-order tensors.
This example uses the guarded producers to supply authenticated frame and
CCSD data. `host_available_mb` is a conservative memory cap, not a request
to allocate that much memory; it is bounded here by currently available
Linux memory.

```python
from pathlib import Path

from q2m3.molecule import MoleculeConfig
from q2m3.sqd import LUCJConfig, run_sqd_from_integrals
from q2m3.sqd.ansatz import build_ccsd_seed
from q2m3.sqd.integrals import build_integrals

available_kib = next(
    int(line.split()[1])
    for line in Path("/proc/meminfo").read_text().splitlines()
    if line.startswith("MemAvailable:")
)
cap_mb = min(4096.0, available_kib * 1024 / 1e6)
molecule = MoleculeConfig(
    name="H2", symbols=["H", "H"],
    coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    charge=0, active_electrons=2, active_orbitals=2,
)
data = build_integrals(molecule, host_available_mb=cap_mb)
seed_data = build_ccsd_seed(data, host_available_mb=cap_mb)
result = run_sqd_from_integrals(
    data.h1, data.h2, data.e_core,
    norb=data.norb, nelec=data.nelec,
    context=data.context, seed_data=seed_data,
    lucj=LUCJConfig(n_reps=2, shots=128), seed=31,
)
result.validate()
assert result.baseline_tier == "T0"
assert abs(result.sqd_energy - result.baseline_energy) < 1e-10
print(result.sqd_energy, result.baseline_method, result.delta_mHa)
```

Both yield approximately −1.13728383449 Ha in the measured profile.
`run_sqd_from_integrals` requires `context`; bare `t1`/`t2` arrays are not an
accepted substitute for `CCSDSeed`. Shape, finite values, balanced electron
counts, Hamiltonian/frame identity, HF/CCSD energy and CCSD residuals are
checked. Failed or unconverged seeds raise an error. For a deliberate
baseline-only run, set `mode="reference_only"` (and `seed_data=None` at the
integral entry). It never silently substitutes HF for a failed SQD run.

## Fixed orbital frame and MM embedding

Coordinates, including `mm_coords`, are in Å; `mm_charges` are in elementary
charges. Energies and integral constants are Hartree. The high-level entry
accepts both MM arrays and `embedding_mode="diagonal"` (default) or
`"full_oneelectron"`. The latter requires MM charges. The vacuum MO frame
and two-electron tensor stay fixed; diagonal mode adds only diagonal active
one-electron terms, while full mode retains off-diagonal terms too.

All solver arms share the same active span, Hamiltonian, frame and core
constant. `e_core` already includes the MM nuclear contribution: do not add
it again at the low-level entry. The low-level `h2` is chemist ordered and
is not transposed again. In an embedded fixed frame, `hf_energy` is the
reference determinant expectation, not a relaxed embedded SCF minimum.
Static MM and the H₂O adapter have been verified. SQD MC integration,
correlated RDM feedback and polarizable MM are outside this delivery.

## Reference ladder and result consumers

`ReferenceConfig` defaults to 900 seconds and 8192 decimal MB. The reference
ladder shares the remaining budget across attempts:

| Tier | Method | Meaning |
| --- | --- | --- |
| T0 | `exact_casci` | Exact within this finite active Hamiltonian |
| T1 | Selected CI | Tight-cutoff result; cutoff differences do not bound its error |
| T1+ | Explicitly enabled plugin | Solver-specific approximation and uncertainty |
| T2 | CCSD(T) | Approximate fallback with diagnostics and trust flags |

T0 cannot be disabled to force a downgrade. Only explicit unavailability,
budget exhaustion or controlled timeout permits trying a lower tier;
numerical failure must surface. Every non-T0 result carries
`baseline_downgrade_reason` and a warning. Unknown uncertainty is `None`
with a reason, never zero. T2 reports the T1 diagnostic, convergence,
triples correction and triples/correlation ratio; flags can mark it
untrustworthy. Zero T0 uncertainty excludes basis/model/experimental errors.

Use `baseline_energy`, `baseline_method`, `baseline_tier`,
`baseline_uncertainty_mHa`, `baseline_downgrade_reason`, `warnings` and
`null_reasons`. Output schema `sqd.result.v1` has 40 fields and writes only
canonical `baseline_*` keys. An old `shci_energy` number alone cannot tell
whether it came from CASCI or SHCI; readers must require explicit method,
tier, units and uncertainty semantics and reject conflicting aliases.
Future simple-input consumers should retain the complete result under
`sqd`; future MM callbacks should retain the entire result, not only energy.
Those downstream consumers are not claimed as implemented here.

`delta_mHa = 1000 * (sqd_energy - baseline_energy)` and
`delta_vs_sci_mHa = 1000 * (sqd_energy - iso_ndet_sci_energy)`:
positive means SQD is worse. `ratio_sqd_over_sci` is defined only against
T0 with a valid nonzero SCI error. SCI and random arms match each spin's
subspace dimension. `subspace_dim` is their Cartesian product;
`unique_dets_vs_shots` counts sampled pairs and is a different quantity.
Reference-only outputs have `None` and explicit reasons for unexecuted
sampling/comparison fields. `save_json_results` from `q2m3.utils.io` accepts
the complete dataclass and preserves these fields; nonfinite JSON values
are rejected.

## Resource limits and measured outcomes

The calibrated single-thread Linux model covers balanced closed shells,
2–10 active orbitals, at most 252 determinants per spin and 63504 total,
repetitions/batches/iterations at most 2, and shots at most 100000.
Other stage-specific input/domain checks still apply. Public diagonalization
uses two batches, two iterations and `min(300, shots)` samples per batch.
Unknown/out-of-domain bounds raise `ResourceModelDomainError` before the
allocation. `allow_large` cannot bypass the domain, host or hard cap.
Predicted RSS at 2048 MB warns, at 8192 MB normally rejects, and at
12288 MB always rejects. Equality to the host or a tighter user cap also
rejects. These are decimal MB, separate from the benchmark criterion
`peak_bytes < 2_000_000_000`.

The approved default benchmark scope used reps 2/shots 100000. All eight
runs used T0, and their measured whole-process-tree peaks were below
2 billion bytes (maximum 1,021,018,112 bytes). Sampling alone peaked at
732,467,200 bytes. Measurements used 10 ms polling and do not guarantee
capture of every instantaneous peak or certify arbitrary geometries,
MM systems, plugins or T2 paths.

| Case | Active space | SQD − T0 (mHa) | Interpretation |
| --- | --- | --- | --- |
| H₂/STO-3G | 2e2o | 0 | Full space 4/4; convention regression |
| H₃O⁺/STO-3G | 4e4o | 0 | Full space 36/36; convention regression |
| Glycine/STO-3G, seeds 0–4 | 6e6o | mean 0.525399; sample std 0.111683; range 0.325821–0.579774 | Subspaces 64–90/400; all five worse than CCSD; mixed against SCI |
| N₂, 1.1 Å/cc-pVDZ, seed 0 | 10e10o | 18.782945 | Subspace 342/63504; worse than SCI by 8.918280 mHa and CCSD by 16.801641 mHa |

These deltas use each run's own Hamiltonian/frame/core reference; historical
absolute energies must not be mixed into them. Historical Glycine
reps 4/shots 200000 and N₂ reps 4/shots 500000 are rejected as out of domain.
Their energy and full-run memory benchmarks remain unmeasured; a cheap
rejection is not a successful memory benchmark.

E1 (four orbital bases) and E6 (restricted connectivity) remain
**inconclusive** because their original numerical matrices exceed the
certified domain. E1 requires all per-seed and mean ranges to lie strictly
on one side of 2 mHa for a directional conclusion; missing points cannot
pass. E6 compares restricted-minus-full errors per seed against 5 mHa;
missing points or threshold crossings are inconclusive. Verified nontrivial
H₃O⁺ connectivity consumption proves the operator uses its inputs, not
energy superiority or hardware readiness. No hardware layout, SWAP, noise
or ancilla certification is implied. E3 was not run in this delivery and must
not be labeled complete.

| Study | Status | Evidence boundary |
| --- | --- | --- |
| E1 | inconclusive | Original four-basis numerical matrix exceeds the certified domain |
| E3 | not_run | Not executed in this delivery |
| E6 | inconclusive | Original restricted/full numerical matrix is incomplete |

The versioned benchmark scripts listed in the example guide regenerate
inputs and reports in a fresh output directory. Historical local reports
under `tmp/` and planning evidence under `.plan/` are not installed package
assets; reproduce results with those scripts or restore the explicitly
provided handoff bundle.

## Complete result field reference

The following 40 names match the `sqd.result.v1` contract. JSON preserves
these names and uses `null` for Python `None`. Every absent top-level value
requires a `null_reasons` entry; required metadata is not omitted. Nested
objects retain their own units and reasons. A consumer must keep the method,
tier, downgrade reason and warnings beside the reference energy.

| Field | Unit | Meaning and absence rule |
| --- | --- | --- |
| `schema_version` | metadata | Required, `sqd.result.v1` |
| `status` | metadata | Required, `completed` or `reference_only` |
| `sqd_energy` | Ha | SQD energy; absent in reference-only mode |
| `hf_energy` | Ha | Required same-Hamiltonian determinant expectation |
| `hf_reference_kind` | metadata | Required `canonical_rhf` or `fixed_frame_determinant` |
| `baseline_energy` | Ha | Required selected reference energy |
| `iso_active_space_ccsd_energy` | Ha | Same-space CCSD; may be absent if not run in reference-only mode |
| `iso_ndet_sci_energy` | Ha | Matched-spin-dimension SCI; absent in reference-only mode |
| `iso_ndet_random_energy` | Ha | Matched random comparison; absent in reference-only mode or with an explicit unavailable/budget reason |
| `baseline_tier` | metadata | Required T0/T1/T1+/T2; never infer from a legacy alias |
| `baseline_method` | metadata | Required actual method; CASCI stays `exact_casci` |
| `baseline_uncertainty_mHa` | mHa | T0 is zero; unknown approximate uncertainty is absent with reason |
| `baseline_uncertainty_kind` | metadata | Required `exact_active_space`, `solver_estimate` or `unknown` |
| `baseline_downgrade_reason` | metadata | Absent for T0; required nonempty text for every lower tier |
| `baseline_untrustworthy` | boolean | Required trust flag; does not turn an approximation into T0 |
| `baseline_t1_residual` | structured metadata | T1 cutoff/energy record; absent when inapplicable. Energies are Ha, loose residual is mHa, and cannot bound the tight energy error |
| `t1_diagnostic` | dimensionless | T2 CCSD amplitude diagnostic; absent when T2 is not used |
| `t2_diagnostics` | structured metadata | T2 flags, CCSD convergence, energies in Ha and dimensionless ratios; absent when inapplicable |
| `reference_attempts` | structured metadata | Required ordered attempt records, including reasons, wall seconds and decimal MB peak where known |
| `delta_mHa` | mHa | SQD minus baseline; absent in reference-only mode |
| `delta_vs_sci_mHa` | mHa | SQD minus same-size SCI; absent in reference-only mode |
| `ratio_sqd_over_sci` | dimensionless | T0-relative error ratio; absent for nonexact baseline, invalid denominator or reference-only mode, with reason |
| `unique_dets_vs_shots` | counts | Cumulative `(shots, unique sampled pairs)` curve; absent in reference-only mode |
| `active_space` | counts | Required `(electrons, spatial orbitals)` |
| `n_reps` | count | Executed LUCJ repetitions; absent in reference-only mode |
| `shots` | count | Executed sampling shots; absent in reference-only mode |
| `seed` | integer | Required actual random seed, including reference-only runs |
| `subspace_dim` | count | Product of alpha/beta dimensions; absent in reference-only mode |
| `subspace_dims` | counts | `(d_alpha, d_beta)`; absent in reference-only mode |
| `full_ci_dim` | count | Required full fixed-particle-number space dimension |
| `embedding_mode` | metadata | Required actual embedding mode |
| `two_electron_tensor_fixed` | boolean | Required fixed-tensor assumption |
| `fixed_mo` | boolean | Required orbital-frame assumption |
| `backend` | metadata | Executed sampling backend; absent in reference-only mode |
| `versions` | mapping | Required exact dependency version snapshot |
| `diagnostics` | mapping | Required additional diagnostics; retain embedded flags/reasons |
| `provenance` | mapping | Required immutable input/frame/Hamiltonian/seed source snapshot |
| `timings_s` | s | Required geometry/integrals/ccsd/prepare/sample/diagonalize/reference/comparison/total keys; unexecuted stages are absent with reasons, not fabricated zero timings |
| `null_reasons` | mapping | Required reasons for missing values, including `reference_only`, `not_applicable` or the actual failure-to-define reason |
| `warnings` | metadata | Required warning collection; every non-T0 baseline has a visible warning |
