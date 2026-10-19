# QPE validation and resolution

Follow this learning order from the checkout root.

```bash
uv sync --frozen --extra dev --extra catalyst --extra solvation
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

1. [H₂ vacuum and embedded correctness check](h2_qpe_validation.py):
   `uv run --no-sync python -m examples.qpe.h2_qpe_validation`
2. [H₂ 4/6/8-bit resolution comparison](h2_8bit_qpe_benchmark.py):
   `uv run --no-sync python -m examples.qpe.h2_8bit_qpe_benchmark`
3. [H₃O⁺ resolution comparison with documented memory fallback](h3o_8bit_qpe_benchmark.py):
   `uv run --no-sync python -m examples.qpe.h3o_8bit_qpe_benchmark`

Each script contains its molecular geometries and QPE grids. The validation
script prints energies. The resolution benchmarks keep their established JSON
output conventions.

Start with H₂. Higher-bit Catalyst compilation may take minutes and many GB of
memory. H₃O⁺ 8-bit compilation can exceed workstation RAM. Phase resolution and
Trotter error are not correlation energy.

Return to the [capability map](../README.md).
