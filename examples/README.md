# q2m3 Examples

## Before running

Use the [source installation](../README.md#from-source) and run commands from
the repository root. The H₂ starters use core dependencies; MC and Catalyst
profiling require the `solvation` extra (which includes Catalyst and JAX).
GPU and visualization extras are optional for the commands below.

## Start with H₂

```bash
uv run python examples/h2_qpe_validation.py
uv run python examples/h2_resource_estimation.py
uv run python examples/full_oneelectron_embedding.py
```

These cover vacuum/MM QPE, EFTQC resource estimates, and fixed-MO diagonal
versus full one-electron embedding. H₂/STO-3G uses `(2e, 2o)`, or four system
qubits; the QPE validation adds four estimation wires. Check its printed
validation results as well as the exit status.

For the H₂ MC example (ten waters, 100 steps, `fixed` mode):

```bash
uv sync --extra solvation
uv run python examples/h2_mc_solvation.py
```

## Script index

| Category | Script | Purpose |
| --- | --- | --- |
| H₂ starter | [h2_qpe_validation.py](h2_qpe_validation.py) | Vacuum/MM QPE integration checks |
| H₂ starter | [h2_resource_estimation.py](h2_resource_estimation.py) | Vacuum/MM resource comparison |
| H₂ starter | [full_oneelectron_embedding.py](full_oneelectron_embedding.py) | Fixed-MO embedding resource rows |
| MC | [h2_mc_solvation.py](h2_mc_solvation.py) | H₂ fixed-Hamiltonian sampling |
| MC | [h3o_mc_solvation.py](h3o_mc_solvation.py) | H₃O⁺ `(4e, 4o)`, `hf_corrected`, three Trotter steps |
| MC comparison | [h2_three_mode_comparison.py](h2_three_mode_comparison.py) | Three modes and correlation–polarization diagnostics |
| Resources | [resource_estimation_survey.py](resource_estimation_survey.py) | Small-molecule EFTQC survey |
| QPE benchmark | [h2_8bit_qpe_benchmark.py](h2_8bit_qpe_benchmark.py) | H₂ 4/8-bit resolution and sampling study |
| QPE benchmark | [h3o_8bit_qpe_benchmark.py](h3o_8bit_qpe_benchmark.py) | H₃O⁺ resolution study; skip/fallback options |
| Compile survey | [ir_qre_trotter5_compile_survey.py](ir_qre_trotter5_compile_survey.py) | Four estimation wires, five Trotter steps |
| Analysis | [ir_qre_correlation_analysis.py](ir_qre_correlation_analysis.py) | Join resource and compile survey outputs |
| Memory scan | [h3o_dynamic_trotter_oom_scan.py](h3o_dynamic_trotter_oom_scan.py) | H₃O⁺ Trotter scan with memory guard |
| Profiling | [catalyst_benchmark.py](catalyst_benchmark.py) | Catalyst compilation/execution comparison |
| Profiling | [qpe_memory_profile.py](qpe_memory_profile.py) | Fixed/dynamic QPE compilation memory |

The [sqd/ probes](sqd/README.md) provide dependency/integration evidence;
they are not a production SQD engine.

## Scientific and runtime boundaries

For the default `embedding_mode="diagonal"`, MC acceptance uses:

| Mode | Acceptance energy | Hamiltonian behavior |
| --- | --- | --- |
| `fixed` | `E_QPE(H_vac) + E_MM` | Vacuum coefficients remain fixed |
| `hf_corrected` | `E_HF(R) + E_MM` | Embedded HF each step; vacuum QPE at diagnostic intervals |
| `dynamic` | `E_QPE(H_diag(R)) + E_MM` | Update diagonal MM terms and nuclear constant each step |

Here `R` is the solvent configuration and `E_MM` is solvent–solvent energy.
Catalyst circuits are compiled and reused across steps. Dynamic updates keep
the vacuum MO frame and two-electron tensor fixed. Full one-electron
`Delta h_pq` embedding is available for resource estimates and fixed
Hamiltonians; it is not supported by the dynamic coefficient-update path.
These models do not include polarizable MM or correlated orbital relaxation.

QPE–HF gaps are not direct correlation-energy measurements: finite phase
resolution, Trotter approximation, and sampling can affect them. Energies are
in Hartree internally; kcal/mol values require explicit conversion.
See the [H₂ tutorial](../doc/source/tutorials/h2-qpe-validation.md) for checks
and sign conventions, and the [PennyLane QPE introduction](https://pennylane.ai/demos/tutorial_qpe/).

Run larger MC, benchmark, and profiling scripts individually after inspecting
their configuration and available guards. Compile time and memory depend on
the system and circuit settings; no fixed RAM budget guarantees success.
Surveys generate local files under `data/output/`; the correlation analysis
requires both resource and compile survey inputs. Generated files are not
versioned reference results or evidence for predictive scaling laws.
