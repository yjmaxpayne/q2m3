# Three-Mode Solvation Comparison

The three-mode comparison estimates how much the solvent environment changes
the electronic correlation contribution. It is more expensive than the fixed
H2 smoke test and should be run after `h2_mc_solvation.py`.

## Run The Script

```bash
uv run python examples/h2_three_mode_comparison.py
```

## Modes

| Mode | Formula | Role |
| --- | --- | --- |
| `fixed` | `E_QPE(H_vac) + E_MM` | Fast baseline; ignores solvent changes in correlation |
| `hf_corrected` | `E_HF(R) + E_MM` | HF-level MM embedding with interval QPE diagnostics |
| `dynamic` | `E_QPE(H_eff) + E_MM` | Runtime MM-embedded QPE coefficients |

The comparison delegates analysis to `q2m3.solvation.analysis.run_mode_comparison()`.

## Compile-Once Idea

```{mermaid}
flowchart LR
    accTitle: Three Mode Execution
    accDescr: Fixed and dynamic modes compile reusable QPE structures while the MC loop updates solvent coordinates and energies.

    mc["MC solvent states"]
    fixed["fixed<br/>vacuum Hamiltonian"]
    hf["hf_corrected<br/>HF energy path"]
    dynamic["dynamic<br/>runtime coefficients"]
    analysis["mode comparison<br/>delta_corr-pol"]

    mc --> fixed
    mc --> hf
    mc --> dynamic
    fixed --> analysis
    hf --> analysis
    dynamic --> analysis
```

Dynamic mode uses runtime-traceable Hamiltonian coefficients so that the circuit
structure can be compiled once while coefficients change across MC steps.

## Reading The Result

The key scientific quantity is `delta_corr-pol`, the correlation-polarization
coupling term. For small H2 runs, it is a diagnostic of whether the QPE energy
changes materially when the MM embedding enters the Hamiltonian rather than
only the classical correction.

## Optional H3O+ Diagnostics

H3O+ workflows use a larger active space and many more Hamiltonian terms. The
public H3O+ scripts are useful for ionic-solvation diagnostics, but they are not
the default tutorial path. Use a 16 GB+ RAM machine for `h3o_mc_solvation.py`
and a 30 GB+ RAM machine for 8-bit or dynamic Trotter diagnostics.
