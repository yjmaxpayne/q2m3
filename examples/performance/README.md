# Performance and compiler studies

Follow this learning order from the checkout root.

```bash
uv sync --frozen --extra dev --extra catalyst --extra solvation
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

1. [Catalyst compile/reuse benchmark](catalyst_benchmark.py):
   `uv run --no-sync python -m examples.performance.catalyst_benchmark`
2. [QPE compilation memory and phase timing](qpe_memory_profile.py):
   `uv run --no-sync python -m examples.performance.qpe_memory_profile`
3. [Memory-guarded dynamic Trotter scan](h3o_dynamic_trotter_oom_scan.py):
   `uv run --no-sync python -m examples.performance.h3o_dynamic_trotter_oom_scan`
4. [Standardized Trotter-5 IR compilation survey](ir_qre_trotter5_compile_survey.py):
   `uv run --no-sync python -m examples.performance.ir_qre_trotter5_compile_survey`
5. [Correlate measured IR data with resource estimates](ir_qre_correlation_analysis.py):
   `uv run --no-sync python -m examples.performance.ir_qre_correlation_analysis`

Read each script’s --help/docstring before running. Inputs are built-in system grids; correlation analysis consumes data/output/qre_survey.json and data/output/ir_qre_trotter5_compile_survey.json. Existing JSON/CSV/IR/figure locations are retained under data/output or the selected IR directory. Compilation may consume many GB and minutes per point: run serially, begin with H₂ and small phase registers. Catalyst is useful for compile-once/reuse-many workloads. The full grids are research workloads, not smoke tests.

Return to the [capability map](../README.md).
