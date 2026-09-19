# SQD: from four determinants to 63,504

These tutorials call the public production API directly. Energies are Hartree,
geometry is Angstrom, and signed solver differences are mHa. Install from a fresh
checkout with `uv sync --frozen --extra sqd --extra dev` (Linux, Python 3.11+).
The frozen dependencies match the current calibrated domain; no Catalyst extra
is required. Set threads before Python starts:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
```

## Learning order

1. **H₂**: read `h2_ground_state.py`, then run
   `uv run --no-sync python -m examples.sqd.h2_ground_state`.
   STO-3G, CAS(2e,2o), 4 system qubits, 128 shots, seed 31. Its 4/4 determinant
   space is a full-space convention regression, not evidence of a sampling advantage.
2. **Glycine end to end**: run
   `uv run --no-sync python -m examples.sqd.glycine_ground_state`.
   The existing neutral NH₂–CH₂–COOH geometry is converted to Cartesian coordinates;
   vacuum RHF orbitals define an active space, CCSD amplitudes initialize two LUCJ
   layers, ffsim samples occupations, and SQD diagonalizes the sampled subspace.
   Default CAS(10e,10o), 20 system qubits, 100,000 shots, seed 0. Use
   `--active-space 6` or `--active-space 8` to work up to the largest case.
3. **Scale and seed spread**: run
   `uv run --no-sync python -m examples.sqd.glycine_active_space_scan`.
   This performs 15 actual calculations: CAS(6,6)/(8,8)/(10,10) × seeds 0–4,
   each in a fresh interpreter, serially. `--seed N` starts five consecutive seeds;
   `--active-spaces 6 8` explicitly requests a smaller study. Failed points are
   retained, successful points are saved immediately, and any failure causes a
   nonzero exit. No historical data fills gaps.
4. **Integral adapter**: run
   `uv run --no-sync python -m examples.sqd.glycine_from_integrals`.
   Inspect explicit real chemist `h1`, `h2`, `e_core`, authenticated `IntegralContext`
   and same-Hamiltonian CCSD seed assembly. `run_sqd_from_integrals` needs no tensor
   transpose; `e_core` is added once. A second geometry run checks adapter agreement
   within 1e-8 Ha. Both entries share a solver, so this is not an independent oracle.
5. Continue to [fixed-environment glycine](../qmmm/README.md).

## Read the result

Terminal output follows system → method → all energy controls → subspace/cost →
interpretation. It includes HF, active-space CCSD, same-size SCI, same-size random
subspace, SQD and the selected reference, including controls that outperform SQD.
Actual zero-based MO indices and frozen-core count are in the full result context.
Closed-shell full determinant counts are `comb(norb, nelec/2)**2`: 400, 4,900 and
63,504 for the glycine ladder. These use 12, 16 and 20 system qubits, no QPE ancillas.

Each active space has **its own Hamiltonian and reference**. The total-energy plot
shows changes in recovered correlation across active spaces; the error plot shows
`1000*(E_method-E_reference)` within each space. Only T0/exact CASCI points enter
the exact-error panel. Downgrades (including T2/CCSD(T)), unknown uncertainty,
reference attempts and warnings remain in JSON/CSV and terminal output. Seed
scatter is descriptive, not a confidence interval or guaranteed monotonic convergence.

The sampled determinant fraction is not a memory speedup factor: ffsim still
simulates the entire fixed-particle-number state space. Internal RSS is a polled
parent-plus-descendants measurement ending at the start of final result construction.
The total engine timer uses that boundary too. It excludes import and plot costs.
The integral tutorial separately records preprocessing outside the engine window.
For independent monitoring of the complete process, including these costs:

```bash
uv run --no-sync python -m tools.sqd.verify_showcase \
  --output data/output/examples/glycine-monitor \
  -- uv run --no-sync python -m examples.sqd.glycine_ground_state \
  --output data/output/examples/glycine-run
```

## Example results

A run on 2026-09-19 used Python 3.12.3, NumPy 2.4.1, SciPy 1.16.3,
PySCF 2.14.0, ffsim 0.0.84 and qiskit-addon-sqd 0.13.1 from the frozen
installation above, with OMP/MKL/OpenBLAS thread counts set to 1.
The default glycine scan used two LUCJ layers, 100,000 shots and seeds 0–4.
All 15 calculations completed with their own T0/exact CASCI reference.

| Active space | System qubits | Full determinants | Mean signed SQD error / mHa | Seed range / mHa | Sample standard deviation / mHa |
|---|---:|---:|---:|---:|---:|
| CAS(6e,6o) | 12 | 400 | 0.525399 | 0.325821–0.579774 | 0.111683 |
| CAS(8e,8o) | 16 | 4,900 | 0.968885 | 0.855738–1.302400 | 0.187935 |
| CAS(10e,10o) | 20 | 63,504 | 2.526072 | 2.268914–2.772343 | 0.227651 |

Errors are `1000*(E_SQD-E_CASCI)` for each active space. The five-seed spread
is descriptive, not a confidence interval. Here, larger spaces had larger SQD
errors; CCSD or same-size SCI sometimes gave energies closer to CASCI than SQD.
The complete outputs retain those controls alongside HF and random subspaces.

H₂ recovered all four determinants and agreed with independently assembled
PySCF CASCI within 1e-8 Ha. The glycine integral example matched all six
geometry-entry energies within 1e-8 Ha, with identical frame and Hamiltonian
identifiers; that comparison checks adapter consistency with a shared solver.

The entire serial scan took about 67 s on the test machine, with a complete-process
polled peak RSS of 913 decimal MB. This external measurement includes process
launch, simultaneous descendants, serialization and figures; its 10 ms polling
can miss short-lived peaks. These are example observations, not runtime guarantees
or an extension of the resource model's certified domain.

## Inputs, artifacts and cost

All single-system entries accept `--output`, `--seed`, `--shots`, `--wall-budget`
and `--rss-budget`; glycine entries also accept `--active-space 6|8|10`.
Defaults are 900 seconds and 8192 **decimal MB per engine call**. This is an upper
budget, not a runtime prediction; allow minutes for the full scan on a workstation
and at most 15 × 900 s for its engine calls. The integral tutorial calls the engine
twice; embedding calls it three times. Avoid parallel scans. Unsupported profiles
are rejected by the existing resource model rather than silently reduced.

Outputs go to a unique `data/output/examples/<name>-<time>-<id>/`:

- `input.json`: geometry, active space, shots, seed and budgets;
- complete `result.json` (or named mode/entry files), unchanged `sqd.result.v1`;
- `statistics.json`: descriptive mean, sample standard deviation and range of actual T0 runs, with failed/non-T0 counts;
- `summary.json` and `summary.csv`: identical flattened energy, reference and cost rows;
- `energies.png/svg` and `cost.png/svg`: headless figures, individual runs and explicit units;
- scan subdirectories with full results and logs for each requested point;
- `failure.json` on errors, with preserved earlier artifacts and a nonzero exit;
- integral `assembly.json` and `consistency.json` describe preprocessing and adapter agreement.

Missing extras provide an installation hint. Budget failure and unsupported inputs
remain failures; no problem-size reduction or chemical-accuracy promise is made.
