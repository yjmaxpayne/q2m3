# H2 Resource Estimation

This tutorial uses the `q2m3.core.resource_estimation` API to estimate EFTQC
hardware resources for H2 and to compare vacuum versus MM-embedded Hamiltonians.

## Run The Script

```bash
uv run python examples/h2_resource_estimation.py
```

## Minimal API Pattern

```python
import numpy as np

from q2m3.core import compare_vacuum_solvated, estimate_resources

h2_symbols = ["H", "H"]
h2_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

vacuum = estimate_resources(symbols=h2_symbols, coords=h2_coords, basis="sto-3g")
print(vacuum.logical_qubits)
print(vacuum.toffoli_gates)

comparison = compare_vacuum_solvated(
    symbols=h2_symbols,
    coords=h2_coords,
    mm_charges=np.array([-0.834, 0.417, 0.417, -0.834, 0.417, 0.417]),
    mm_coords=np.array(
        [
            [3.0, 0.0, 0.0],
            [3.5, 0.8, 0.0],
            [3.5, -0.8, 0.0],
            [-3.0, 0.0, 0.0],
            [-3.5, 0.8, 0.0],
            [-3.5, -0.8, 0.0],
        ]
    ),
)
print(comparison.delta_lambda_percent)
```

## Current H2 Reference Values

The maintained example reports:

| Metric | H2/STO-3G reference |
| --- | --- |
| Logical qubits | `115` |
| Toffoli gates | `1,224,608` |
| Target error | Chemical-accuracy scale by default |

The comparison is expected to show only a small resource change from MM
embedding because point charges primarily modify one-electron terms. For this
small H2 example, two-electron integrals dominate the resource estimate.

## Interpretation Boundaries

Resource estimates describe an EFTQC algorithmic cost model. They do not
predict local Catalyst compile memory, host RAM usage, or wall-clock runtime
for PennyLane simulation.
