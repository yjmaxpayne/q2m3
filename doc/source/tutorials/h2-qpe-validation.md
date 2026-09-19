# H2 QPE Validation

This tutorial validates the small H2 path used as the first q2m3 smoke test.
It compares vacuum and MM-embedded H2 Hamiltonians and reports stabilization
in kcal/mol after explicit conversion from Hartree.

## Run The Script

```bash
uv run python examples/qpe/h2_qpe_validation.py
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

Use the current script's `run_validation_checks` and printed results as the
integration check. Its example-specific thresholds are `0.1 kcal/mol` for
PennyLane HF expectation values versus PySCF HF, and `2.0 Ha` for absolute
QPE–HF differences. These deliberately loose POC checks do not establish
chemical accuracy. The script prints failures without setting a failing exit
status, so inspect every `[OK]` / `[FAIL]` line.

The signed offset is `E_QPE - E_HF`; it includes finite phase resolution,
Trotter error, and sampling effects and must not be identified directly with
correlation energy. The default run uses 100 shots, so QPE results can vary.
The script defines stabilization as `(E_vacuum - E_solvated)` converted to
kcal/mol: a positive value means the solvated system has lower energy.
Its sign check is bypassed when the absolute QPE stabilization is at most
`0.01 kcal/mol`.

For a regression comparison, save the output together with the source
revision, dependency versions, device, and QPE parameters. Historical README
numbers are not acceptance thresholds.

## Common Adjustments

Use more estimation wires for better phase resolution, and increase Trotter
steps only after confirming memory and runtime are acceptable. If the energy
appears to wrap or jump by a large amount, inspect the base time and shifted
QPE parameters before interpreting the difference physically.
