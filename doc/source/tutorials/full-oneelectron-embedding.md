# Full-One-Electron Fixed-MO Embedding

This tutorial runs a small H2/STO-3G active-space resource estimate in vacuum
and near one TIP3P water shell. It demonstrates the public
`embedding_mode="diagonal"` and `embedding_mode="full_oneelectron"` resource
rows.

## Run The Script

```bash
uv run python examples/full_oneelectron_embedding.py
```

The script prints Hamiltonian `lambda`, Toffoli count, logical qubits, and
`delta_h_offdiag_fro` for:

| Row | Meaning |
| --- | --- |
| `vacuum` | H2 without MM point charges |
| `diagonal` | Fixed-MO point-charge perturbation with only active-space `Delta h_pp` |
| `full_oneelectron` | Fixed-MO point-charge perturbation with full active-space `Delta h_pq` |

## Minimal API Pattern

```python
import numpy as np

from q2m3.core import estimate_resources

h2_symbols = ["H", "H"]
h2_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

tip3p_charges = np.array([-0.834, 0.417, 0.417])
tip3p_coords = np.array(
    [
        [3.0, 0.0, 0.0],
        [3.5, 0.8, 0.0],
        [3.5, -0.8, 0.0],
    ]
)

common = dict(
    symbols=h2_symbols,
    coords=h2_coords,
    basis="sto-3g",
    active_electrons=2,
    active_orbitals=2,
)

diagonal = estimate_resources(
    **common,
    mm_charges=tip3p_charges,
    mm_coords=tip3p_coords,
    embedding_mode="diagonal",
)
full = estimate_resources(
    **common,
    mm_charges=tip3p_charges,
    mm_coords=tip3p_coords,
    embedding_mode="full_oneelectron",
)

print(diagonal.embedding_mode, diagonal.hamiltonian_1norm)
print(full.embedding_diagnostics.delta_h_offdiag_fro)
```

## Interpretation Boundaries

`full_oneelectron` means the full fixed-MO active-space one-electron
perturbation is included in the resource row. It does not mean relaxed orbital
optimization, a polarizable MM force field, or a relaxed solvation energy.

The current dynamic runtime coefficient workflow remains diagonal-update only.
Full-one-electron embedding is available for resource estimates and fixed
Hamiltonian/operator paths because off-diagonal one-electron terms can change
the compiled operator support.
