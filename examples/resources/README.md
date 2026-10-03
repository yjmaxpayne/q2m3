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

Return to the [capability map](../README.md).
