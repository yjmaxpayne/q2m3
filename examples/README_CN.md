# 按功能学习示例

[English](README.md) | [简体中文](README_CN.md)

从仓库根目录运行所有命令。
先运行小体系，再按各章节的顺序学习。
SQD 和片段示例使用 `python -m examples.…` 入口。
各章节说明输入、输出、依赖和实测成本。
仓库不提交运行时输出的数值结果。

| 功能 | 起点 | 后续内容 |
| --- | --- | --- |
| [QPE](qpe/README.md) | H₂ 检查 | H₂ 和 H₃O⁺ 分辨率基准 |
| [SQD](sqd/README.md) | H₂ 基态 | 甘氨酸活性空间、五个种子、积分入口 |
| [QM/MM](qmmm/README.md) | 固定分子轨道嵌入 | 水和甘氨酸 SQD、现有 QPE–MC 流程 |
| [片段相互作用能](fragments/README_CN.md) | 水二聚体 HF/MP2 | 完整计算、CIF 准备、晶体能量、汇总报告 |
| [资源估计](resources/README.md) | H₂ 资源估计 | 多分子比较 |
| [性能](performance/README.md) | Catalyst 基准 | QPE 内存、Trotter 扫描、编译分析 |

## 运行片段示例

五个入口依次为：

```bash
uv sync --frozen --extra dev --extra viz --extra sqd
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
uv run --no-sync python -m examples.fragments.water_dimer_classical
uv run --no-sync python -m examples.fragments.water_dimer_full
uv run --no-sync python -m examples.fragments.crystal_dimer_prepare
uv run --no-sync python -m examples.fragments.crystal_dimer_interaction
uv run --no-sync python -m examples.fragments.fragment_report
```

在启动 Python 前设置线程变量。
核心依赖支持水二聚体经典计算。
晶体输入还要求 `viz` 提供的 ASE。
完整模式要求 Linux x86_64 和 `sqd` 依赖。
QPE–MC 和 JIT 示例另需 `catalyst` 与 `solvation`。

晶体能量入口和汇总入口默认采用经典模式。
用 `--mode full` 启用完整模式。
完整模式检查各子系统及相互作用能的 SQD 精度。
精度不达标时，命令返回非零退出码。
[片段章节](fragments/README_CN.md)列出验收限值和科学边界。

SQD 和片段示例默认在 `data/output/examples/` 下创建独占目录。
`--output` 必须指定新目录。
已有目录会使命令失败。
原有 QPE、MC、资源及性能脚本沿用各自的输出位置。

## 数据与证据

[片段实测结果](fragments/README_CN.md#实测结果)标注 2026-10-25 的环境和输入。
水与晶体采用不同基组，不能据此作受控的相互作用强度排序。
QPE 资源表不表示运行过 QPE。
缺失值、降级原因和警告必须保留。
外部参考必须匹配几何结构与相互作用能定义。

[SQD 章节](sqd/README.md#example-results)另列 2026-09-19 的历史甘氨酸结果。
这些历史结果只描述原始输入和版本。
共享 CI 不保证复算这些研究。
