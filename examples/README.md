# q2m3 Examples

> Hybrid Quantum-Classical QM/MM for Early Fault-Tolerant Quantum Computers (EFTQC)

---

This directory contains maintained examples for the q2m3 package. The scripts
here are runnable, inspectable entry points for the public APIs and diagnostics
that are useful to package users.

## Runtime And Memory Tiers

| Tier | Scripts | Expected environment |
|------|---------|----------------------|
| First run | `h2_qpe_validation.py`, `h2_resource_estimation.py`, `full_oneelectron_embedding.py` | CPU laptop or workstation |
| Standard MC | `h2_mc_solvation.py` | Catalyst/JAX installed; 8 GB+ RAM recommended |
| H3O+ MC | `h3o_mc_solvation.py` | Catalyst/JAX installed; 16 GB+ RAM recommended |
| High-memory diagnostics | `h3o_8bit_qpe_benchmark.py`, `h3o_dynamic_trotter_oom_scan.py`, `qpe_memory_profile.py` | 30 GB+ RAM recommended; use provided guards/options |

Do not start by running every script in this directory. Catalyst compile memory
scales with estimation wires, Trotter depth, and Hamiltonian term count. H3O+
examples are intentionally separated from the H2 first-run path.

## Chapter 1: Static QPE (`q2m3.core` API)

Single-configuration QPE studies. Validates algorithm correctness and hardware resource estimates.

| Example | Description | Lines |
|---------|-------------|-------|
| `h2_qpe_validation.py` | QPE correctness: vacuum vs MM-embedded H2 solvation | 517 |
| `h2_resource_estimation.py` | EFTQC hardware resource estimation (Toffoli, qubits, 1-norm) | 95 |
| `full_oneelectron_embedding.py` | H2 + one TIP3P water fixed-MO diagonal vs full one-electron resource rows | 120 |

```bash
uv run python examples/h2_qpe_validation.py
uv run python examples/h2_resource_estimation.py
uv run python examples/full_oneelectron_embedding.py
```

**Verified H2 results from the current scripts**:
- QPE-HF energy gap: `0.0174 Ha` (`10.9 kcal/mol` correlation energy)
- QPE solvation stabilization: `-0.0543 kcal/mol` for 2 TIP3P waters
- PennyLane `<HF|H|HF>` vs PySCF HF agreement: `<= 0.0001 kcal/mol`
- H2 EFTQC resource estimate: `1,224,608` Toffoli gates and `115` logical qubits

**Fixed-MO full-one-electron embedding**:

`full_oneelectron_embedding.py` compares three resource rows:

| Row | Meaning |
|-----|---------|
| `vacuum` | No MM point charges |
| `diagonal` | Adds only active-space `Delta h_pp` point-charge terms |
| `full_oneelectron` | Adds the full fixed-MO active-space `Delta h_pq` matrix |

The script prints `delta_h_offdiag_fro`, Hamiltonian `lambda`, Toffoli gates,
and logical qubits. It is not a relaxed solvation energy calculation; the MO
frame and two-electron tensor stay fixed at their vacuum values.

---

## Chapter 2: MC Dynamics (`q2m3.solvation` API)

Monte Carlo solvation sampling. Each step changes the MM environment -> different Hamiltonian.

| Example | Description | Lines |
|---------|-------------|-------|
| `h2_mc_solvation.py` | H2 fixed-mode MC with compile-once QPE | 90 |
| `h3o_mc_solvation.py` | H3O+ hf_corrected MC with safe default `n_trotter_steps=3` | 98 |
| `h2_three_mode_comparison.py` | H2 fixed / hf_corrected / dynamic comparison and δ_corr-pol analysis | 494 |

```bash
# Recommended MC smoke path
uv run python examples/h2_mc_solvation.py

# Optional: heavier ionic example; use a 16 GB+ RAM machine
uv run python examples/h3o_mc_solvation.py

# Longer comparison run
uv run python examples/h2_three_mode_comparison.py
```

`h3o_mc_solvation.py` uses `n_trotter_steps=3` by default. Raising H3O+
Trotter depth can substantially increase Catalyst IR size and memory pressure.
Use `h3o_dynamic_trotter_oom_scan.py` before increasing this value on shared or
low-memory machines.

**Three QPE modes** (in `h2_three_mode_comparison.py`):

| Mode | Energy Formula | Physics |
|------|---------------|---------|
| `fixed` | E_QPE(H_vac) + E_MM | Approximate (ignores δ_corr-pol) |
| `hf_corrected` | E_HF(R) + E_MM | Intermediate (HF-level MM embedding) |
| `dynamic` | E_QPE(H_eff with MM) + E_MM | Most rigorous (runtime coefficient parameterization) |

**Key finding**: `dynamic` mode uses JAX-traceable Hamiltonian coefficients via
`TrotterProduct(..., check_hermitian=False)` to solve the Catalyst recompilation bottleneck.
Compile once, run many times.

Runtime coefficient updates remain diagonal-only. Full-one-electron embedding
can change fixed operator support, so it is exposed through the fixed-MO
resource and fixed-Hamiltonian paths rather than the dynamic coefficient path.

---

## Resource And IR-QRE Studies

EFTQC resource estimation and compile-IR ↔ quantum-resource correlation studies.
These scripts generate local `data/output/qre_*` and `data/output/ir_qre_*`
artifacts. Generated outputs are not part of the public examples contract and
should be reproduced locally when needed.

| Example | Description | Lines |
|---------|-------------|-------|
| `resource_estimation_survey.py` | Small-molecule QRE matrix; 13 runnable rows plus an explicit Glycine skip row | 474 |
| `h2_8bit_qpe_benchmark.py` | H2 4/8-bit QPE resolution benchmark | 140 |
| `h3o_8bit_qpe_benchmark.py` | H3O+ 4/8-bit fixed-mode benchmark with 6-bit fallback support | 288 |
| `ir_qre_trotter5_compile_survey.py` | Standardized 4-bit, trotter=5 Catalyst compile survey | 403 |
| `ir_qre_correlation_analysis.py` | Joins QRE and compile survey outputs; D1-D4 regression report | 625 |
| `h3o_dynamic_trotter_oom_scan.py` | Memory-guarded H3O+ dynamic Trotter scaling scan | 459 |

```bash
uv run python examples/resource_estimation_survey.py
OMP_NUM_THREADS=2 uv run python examples/ir_qre_trotter5_compile_survey.py
uv run python examples/ir_qre_correlation_analysis.py

# High-memory diagnostics. Prefer a 30 GB+ RAM machine.
OMP_NUM_THREADS=4 uv run python examples/h2_8bit_qpe_benchmark.py
OMP_NUM_THREADS=8 uv run python examples/h3o_8bit_qpe_benchmark.py --skip-8bit
OMP_NUM_THREADS=8 uv run python examples/h3o_dynamic_trotter_oom_scan.py
```

**Key artifacts**:
- `data/output/qre_survey.csv` and `.json` — QRE summary with success/skip rows
- `data/output/ir_qre_trotter5_compile_survey.csv` and `.json` — measured compile rows
- `data/output/ir_qre_correlation.csv` — currently 6 measured joined rows
- `data/output/ir_qre_correlation_stats.csv` — D1-D4 fit statistics
- `data/output/ir_qre_correlation_report.md` — text report

Current checked-in local outputs show a strong descriptive D1 association
(`R² ≈ 0.957`) and weak/null D2-D4 fits (`R² ≈ 0.072`, `0.145`, `0.002`).
Treat these correlations as diagnostic evidence, not as predictive scaling laws.

---

## Profiling And Tools

| Example | Description | Lines |
|---------|-------------|-------|
| `catalyst_benchmark.py` | Catalyst JIT performance benchmark | 248 |
| `qpe_memory_profile.py` | QPE compilation memory profiling for fixed/dynamic Hamiltonian modes | 572 |

```bash
uv run python examples/catalyst_benchmark.py
uv run python examples/qpe_memory_profile.py --mode fixed
```

`qpe_memory_profile.py --mode both` and `--sweep` can become memory intensive.
Use the narrower `--mode fixed` or `--mode dynamic` commands first.

---

## Scientific Background

### The Missing Physics: δ_corr-pol

The `fixed` and `hf_corrected` modes assume:
$$E_{\mathrm{corr}}(H_{\mathrm{eff}}) \approx E_{\mathrm{corr}}(H_{\mathrm{vac}})$$

The correlation-polarization coupling term:
$$\delta_{\mathrm{corr-pol}} \equiv E_{\mathrm{corr}}(H_{\mathrm{eff}}) - E_{\mathrm{corr}}(H_{\mathrm{vac}})$$

is non-zero and can be comparable to the total solvation-energy scale. The
`dynamic` mode samples this quantity via `h2_three_mode_comparison.py`.

### References

- [PennyLane QPE Tutorial](https://pennylane.ai/qml/demos/tutorial_qpe/)
- [PennyLane Catalyst Documentation](https://docs.pennylane.ai/projects/catalyst/)
