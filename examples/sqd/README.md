# SQD: from four determinants to 63,504

These tutorials call the public production API directly. Energies are in
Hartree, geometry is in Angstrom, and signed solver differences are in mHa. To
install from a fresh checkout, run `uv sync --frozen --extra sqd --extra dev`
(Linux x86_64, Python 3.11+). The supervisor rejects other platforms because its
resource accounting uses `/proc` and `fork`. The frozen dependencies match the
current calibrated domain. SQD does not require the Catalyst extra. Set the
thread variables before Python starts:

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
   The script converts the existing neutral NH₂–CH₂–COOH geometry to Cartesian
   coordinates. Then the workflow runs these steps:
   - Vacuum RHF orbitals define an active space.
   - CCSD amplitudes initialize two LUCJ layers.
   - ffsim samples occupations.
   - SQD diagonalizes the sampled subspace.

   Defaults: CAS(10e,10o), 20 system qubits, 100,000 shots, seed 0. Use
   `--active-space 6` or `--active-space 8` to work up to the largest case.
3. **Scale and seed spread**: run
   `uv run --no-sync python -m examples.sqd.glycine_active_space_scan`.
   This scan performs 15 actual calculations: CAS(6,6)/(8,8)/(10,10) × seeds 0–4.
   Each calculation runs serially in a fresh interpreter. `--seed N` starts five
   consecutive seeds. `--active-spaces 6 8` explicitly requests a smaller study.
   The scan keeps failed points and saves successful points immediately. Any
   failure causes a nonzero exit. No historical data fills gaps.
4. **Integral adapter**: run
   `uv run --no-sync python -m examples.sqd.glycine_from_integrals`.
   Inspect the explicit real chemist `h1`, `h2`, and `e_core`, the authenticated
   `IntegralContext`, and the same-Hamiltonian CCSD seed assembly.
   `run_sqd_from_integrals` needs no tensor transpose. `e_core` is added once. A
   second geometry run checks adapter agreement within 1e-8 Ha. Both entries share
   a solver, so this check is not an independent oracle.
5. Continue to [fixed-environment glycine](../qmmm/README.md).

## Read the result

Terminal output follows this order: system → method → all energy controls →
subspace/cost → interpretation. It includes HF, active-space CCSD, same-size SCI,
a same-size random subspace, SQD, and the selected reference. It also includes
controls that outperform SQD. The full result context contains the actual
zero-based MO indices and the frozen-core count. Closed-shell full determinant
counts are `comb(norb, nelec/2)**2`: 400, 4,900, and 63,504 for the glycine
ladder. These cases use 12, 16, and 20 system qubits, with no QPE ancillas.

Each active space has **its own Hamiltonian and reference**. The total-energy plot
shows changes in recovered correlation across active spaces. The error plot shows
`1000*(E_method-E_reference)` within each space. Only T0/exact CASCI points enter
the exact-error panel. Downgrades (including T2/CCSD(T)), unknown uncertainty,
reference attempts, and warnings remain in the JSON/CSV and terminal output. Seed
scatter is descriptive. It is not a confidence interval, and it does not
guarantee monotonic convergence.

The sampled determinant fraction is not a memory speedup factor. ffsim still
simulates the entire fixed-particle-number state space. Internal RSS is a polled
parent-plus-descendants measurement. This measurement ends at the start of final
result construction. The total engine timer uses the same boundary. It excludes
import and plot costs. The integral tutorial records preprocessing separately,
outside the engine window. To monitor the complete process independently,
including these costs, run:

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
is descriptive, not a confidence interval. In this run, larger spaces had larger
SQD errors. CCSD or same-size SCI sometimes gave energies closer to CASCI than
SQD. The complete outputs keep those controls together with HF and random
subspaces.

H₂ recovered all four determinants and agreed with independently assembled
PySCF CASCI within 1e-8 Ha. The glycine integral example matched all six
geometry-entry energies within 1e-8 Ha, with identical frame and Hamiltonian
identifiers. That comparison checks adapter consistency with a shared solver.

The entire serial scan took about 67 s on the test machine. The complete-process
polled peak RSS was 913 decimal MB. This external measurement includes process
start-up, simultaneous descendants, serialization, and figures. Its 10 ms polling
can miss short-lived peaks. These are example observations. They are not runtime
guarantees, and they do not extend the certified domain of the resource model.

## Inputs, artifacts and cost

All single-system entries accept `--output`, `--seed`, `--shots`, `--wall-budget`,
and `--rss-budget`. Glycine entries also accept `--active-space 6|8|10`.
The defaults are 900 seconds and 8192 **decimal MB per engine call**. These values
are an upper budget, not a runtime prediction. Allow minutes for the full scan on
a workstation. Its engine calls take at most 15 × 900 s. The integral tutorial
calls the engine twice. The embedding example calls it three times. Do not run
scans in parallel. The existing resource model rejects unsupported profiles. It
does not reduce them silently.

Outputs go to a unique `data/output/examples/<name>-<time>-<id>/`:

- `input.json`: geometry, active space, shots, seed, and budgets
- complete `result.json` (or named mode/entry files) in the unchanged `sqd.result.v1` format
- `statistics.json`: descriptive mean, sample standard deviation, and range of actual T0 runs, with failed/non-T0 counts
- `summary.json` and `summary.csv`: identical flattened energy, reference, and cost rows
- `energies.png/svg` and `cost.png/svg`: headless figures with individual runs and explicit units
- scan subdirectories with full results and logs for each requested point
- `failure.json` on errors, with preserved earlier artifacts and a nonzero exit
- integral `assembly.json` and `consistency.json`: preprocessing and adapter agreement

A missing extra gives an installation hint. Budget failures and unsupported
inputs remain failures. The scripts do not reduce the problem size, and they do
not promise chemical accuracy.
