# Fault-tolerant resource estimates

Follow this learning order from the checkout root.

```bash
uv sync --frozen --extra dev
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

1. [H₂ vacuum and point-charge resource comparison](h2_resource_estimation.py):
   `uv run --no-sync python -m examples.resources.h2_resource_estimation`
2. [Multimolecule resource survey](resource_estimation_survey.py):
   `uv run --no-sync python -m examples.resources.resource_estimation_survey`

These examples produce resource estimates from built-in geometries and active
spaces. They do not execute on quantum hardware. The H₂ example prints
estimates. The survey writes `data/output/qre_survey.json` and plots. Allow
seconds to minutes for the classical integrals and factorization. Larger
molecules cost more.

The [fragment examples](../fragments/README.md) report QPE estimates and SQD workloads for AB, A, and B.
The report identifies each subsystem's basis and active space.
Its QPE table does not represent an executed QPE calculation.

## Minimal Python API

This snippet estimates resources for H₂/STO-3G. The active space has two
electrons in two spatial orbitals, which gives four Jordan–Wigner system qubits.
Coordinates are in Å. The target energy error is in Hartree.
Run the snippet with `python` in the PyPI environment, or with `uv run python`
in the source checkout.

```python
import numpy as np

from q2m3.core import estimate_resources

resources = estimate_resources(
    symbols=["H", "H"],
    coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
    basis="sto-3g",
    active_electrons=2,
    active_orbitals=2,
    target_error=0.0016,
)
print("System qubits:", resources.n_system_qubits)
print("Logical qubits:", resources.logical_qubits)
print("Toffoli gates:", resources.toffoli_gates)
```

`estimate_resources` returns an `EFTQCResources` object. The logical-qubit and
Toffoli-gate counts come from an algorithmic cost model. They do not predict
local simulator runtime or Catalyst compilation memory.

Return to the [capability map](../README.md).
