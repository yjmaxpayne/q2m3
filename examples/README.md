# q2m3 Examples

> Hybrid Quantum-Classical QM/MM for Early Fault-Tolerant Quantum Computers (EFTQC)

---

## Chapter 1: Static QPE (`q2m3.core` API)

Single-configuration QPE studies. Validates algorithm correctness and hardware resource estimates.

| Example | Description | Lines |
|---------|-------------|-------|
| `h2_qpe_validation.py` | QPE correctness: vacuum vs MM-embedded H2 solvation | ~520 |
| `h2_resource_estimation.py` | EFTQC hardware resource estimation (Toffoli, qubits, 1-norm) | ~95 |

```bash
uv run python examples/h2_qpe_validation.py
uv run python examples/h2_resource_estimation.py
```

**Key results (H2, STO-3G)**:
- QPE-HF energy gap: ~0.017 Ha (10.9 kcal/mol correlation energy)
- MM solvation stabilization: -0.054 kcal/mol (2 TIP3P waters)
- PL↔PySCF agreement: < 0.001 kcal/mol

---

## Chapter 2: MC Dynamics (`q2m3.solvation` API)

Monte Carlo solvation sampling. Each step changes the MM environment → different Hamiltonian.

| Example | Description | Lines |
|---------|-------------|-------|
| `h2_mc_solvation.py` | MC entry point: H2 with fixed-mode QPE (pre-compiled) | ~90 |
| `h3o_mc_solvation.py` | Ionic solvation: H3O+ with hf_corrected mode | ~99 |
| `h2_three_mode_comparison.py` | Full three-mode comparison + δ_corr-pol analysis | ~470 |

```bash
uv run python examples/h2_mc_solvation.py
uv run python examples/h3o_mc_solvation.py
uv run python examples/h2_three_mode_comparison.py
```

**Three QPE modes** (in `h2_three_mode_comparison.py`):

| Mode | Energy Formula | Physics |
|------|---------------|---------|
| `fixed` | E_QPE(H_vac) + E_MM | Approximate (ignores δ_corr-pol) |
| `hf_corrected` | E_HF(R) + E_MM | Intermediate (HF-level MM embedding) |
| `dynamic` | E_QPE(H_eff with MM) + E_MM | Most rigorous (runtime coefficient parameterization) |

**Key finding**: `dynamic` mode uses JAX-traceable Hamiltonian coefficients via
`TrotterProduct(..., check_hermitian=False)` to solve the Catalyst recompilation bottleneck.
Compile once, run many times.

---

## Profiling & Tools

| Example | Description | Lines |
|---------|-------------|-------|
| `catalyst_benchmark.py` | Catalyst JIT performance benchmark (single vs multi-execution) | ~250 |
| `qpe_memory_profile.py` | QPE compilation memory profiling | ~570 |

```bash
uv run python examples/catalyst_benchmark.py
uv run python examples/qpe_memory_profile.py
```

**Catalyst speedup summary**:

| Scenario | Expected Speedup |
|----------|-----------------|
| Single QPE execution | ~1x (JIT overhead ≈ execution) |
| Multi-execution (100+ iters) | 5–50x |
| `qml.for_loop()` VQE-style | 10–100x |

---

## Archived Code

See [`_archived/README.md`](./_archived/README.md) for code superseded by the
production `q2m3.solvation` and `q2m3.core` packages.

Notable archived files:
- `h3op_demo/` → functionality migrated to `q2m3.profiling.timing` and `q2m3.core.resource_estimation`
- `h3o_qpe_full_demo.py` → replaced by `h3o_mc_solvation.py`
- `h2_three_mode_qpe_mc.py` (890 lines) → replaced by `h2_three_mode_comparison.py` + `q2m3.solvation.analysis`

---

## Scientific Background

### The Missing Physics: δ_corr-pol

The `fixed` and `hf_corrected` modes assume:
$$E_{\mathrm{corr}}(H_{\mathrm{eff}}) \approx E_{\mathrm{corr}}(H_{\mathrm{vac}})$$

The correlation-polarization coupling term:
$$\delta_{\mathrm{corr-pol}} \equiv E_{\mathrm{corr}}(H_{\mathrm{eff}}) - E_{\mathrm{corr}}(H_{\mathrm{vac}})$$

is non-zero (~10–30% of E_corr, 0.6–3 kcal/mol for H3O+) and comparable to the
total solvation energy. The `dynamic` mode directly measures this quantity via
`h2_three_mode_comparison.py`.

**Measured (H2, STO-3G, 500 MC steps, qpe_interval=1)**:
- ⟨δ_corr-pol⟩ ≈ -0.10 kcal/mol
- σ ≈ 7.14 kcal/mol (dominated by 4-bit QPE bin width ~12 mHa)

### References

- [PennyLane QPE Tutorial](https://pennylane.ai/qml/demos/tutorial_qpe/)
- [PennyLane Catalyst Documentation](https://docs.pennylane.ai/projects/catalyst/)
- [q2m3 Technical Overview](../TECHNICAL_OVERVIEW.md)
