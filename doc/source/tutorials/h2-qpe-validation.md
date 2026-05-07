# H2 QPE Validation

This tutorial validates the small H2 path used as the first q2m3 smoke test.
It compares vacuum and MM-embedded H2 Hamiltonians and reports stabilization
in kcal/mol after explicit conversion from Hartree.

## Run The Script

```bash
uv run python examples/h2_qpe_validation.py
```

The script performs four steps:

1. Build vacuum and solvated PennyLane Hamiltonians from PySCF data.
2. Run classical Hartree-Fock references for vacuum and MM-embedded systems.
3. Execute QPE for both Hamiltonians.
4. Compare HF and QPE stabilization energies.

## What The Example Builds

| Component | Value |
| --- | --- |
| Molecule | H2 |
| Bond length | 0.74 Angstrom |
| Basis | STO-3G |
| Active space | 2 electrons, 2 orbitals |
| System qubits | 4 |
| MM environment | 2 TIP3P waters as point charges |
| Default QPE register | 4 estimation wires |

The MM point charges are placed about 3 Angstrom from H2. This keeps the
example small while still exercising the MM embedding path.

## Interpreting Results

The maintained example README records these current H2 checks:

| Quantity | Current reference |
| --- | --- |
| QPE-HF signed energy offset | `-0.0174 Ha` (QPE lower than HF) |
| Absolute offset equivalent | `10.9 kcal/mol` |
| QPE solvation stabilization | `-0.0543 kcal/mol` for 2 TIP3P waters |
| PennyLane HF vs PySCF HF agreement | `<= 0.0001 kcal/mol` |

Treat the H2 script as an integration validation, not as a broad chemistry
benchmark. The thresholds are example-specific and intentionally relaxed for
the POC scale.

## Common Adjustments

Use more estimation wires for better phase resolution, and increase Trotter
steps only after confirming memory and runtime are acceptable. If the energy
appears to wrap or jump by a large amount, inspect the base time and shifted
QPE parameters before interpreting the difference physically.
