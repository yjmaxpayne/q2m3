# Archived POC Code

This directory contains proof-of-concept code that has been superseded by the
production `q2m3.solvation` package.

## Contents

- **mc_solvation_poc/**: Original MC solvation POC package. Replaced by
  `src/q2m3/solvation/` which provides the same workflow with cleaner
  architecture, IR caching, and three-mode QPE support.

- **h2_cached_qpe_driven_mc.py**: Experimental IR caching driver. Functionality
  now integrated into `q2m3.solvation.ir_cache`.

- **qpe_compile_cache_verify.py**: Catalyst compilation cache verification
  script. No longer needed with the production IR cache infrastructure.

## Production Alternatives

| POC Script | Production Replacement |
|-----------|----------------------|
| `mc_solvation_poc/orchestrator.py` | `q2m3.solvation.orchestrator.run_solvation()` |
| `mc_solvation_poc/energy.py` | `q2m3.solvation.energy` |
| `mc_solvation_poc/mc_loop.py` | `q2m3.solvation.mc_loop` |
| `mc_solvation_poc/plotting.py` | `q2m3.solvation.plotting` + `q2m3.utils.plotting` |
| `h2_cached_qpe_driven_mc.py` | `q2m3.solvation.ir_cache` |

## Example Usage (Production)

```bash
uv run python examples/h2_mc_solvation.py
uv run python examples/h3o_mc_solvation.py
uv run python examples/h2_three_mode_comparison.py
```

---

## h3op_demo/ (2026-02-28 归档)

H3O+ 完整 QPE 演示包（6 模块）。有价值的功能已迁移到生产代码：
- `profiling.py` → `q2m3.profiling.timing`
- `computation.py` (资源估算) → `q2m3.core.resource_estimation`

**被替代**: `h3o_mc_solvation.py` 使用 `q2m3.solvation` API 覆盖此功能。

## h3o_qpe_full_demo.py (2026-02-28 归档)

H3O+ QPE 全流程演示。依赖 h3op_demo/ 包。

**被替代**: `h3o_mc_solvation.py`

## h2_three_mode_qpe_mc.py (2026-02-28 归档)

三模式 QPE 对比实验原版（890 行）。分析逻辑已提取到生产代码。

**被替代**: `h2_three_mode_comparison.py` + `q2m3.solvation.analysis`
