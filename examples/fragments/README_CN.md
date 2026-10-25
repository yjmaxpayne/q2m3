# 片段相互作用能示例

[English](README.md) | [简体中文](README_CN.md)

本章衔接[计算教案](../../doc/source/tutorials/fragment-interaction.md)和[报告教案](../../doc/source/tutorials/fragment-report.md)。
示例直接调用公开 API。
输入文件随示例提供，运行时不下载数据。
从仓库根目录运行各模块。

## 学习顺序

| 顺序 | 模块 | 内容 | 默认计算 |
| --- | --- | --- | --- |
| 1 | `water_dimer_classical` | 片段、ghost 原子、counterpoise | HF 和冻结核 MP2 |
| 2 | `water_dimer_full` | 方法差异、参考层级、资源估计 | 完整模式 |
| 3 | `crystal_dimer_prepare` | CIF 导入、X–H 键长归一化 | 只处理结构 |
| 4 | `crystal_dimer_interaction` | 晶体片段相互作用能 | 经典模式 |
| 5 | `fragment_report` | 两个体系、外部参考、可选 Ks | 按顺序运行经典计算 |

## 代码结构

上述五个模块是可运行示例。
计算函数直接调用 `q2m3.fragments.run`。
三个私有模块共享示例准备和输出代码：

| 模块 | 职责 |
| --- | --- |
| [`_inputs.py`](_inputs.py) | 随附输入、固定参数、结构导出 |
| [`_common.py`](_common.py) | 命令行参数、运行目录、参数映射、结果文件 |
| [`_accuracy.py`](_accuracy.py) | 两个示例的精度限值和误差摘要 |

这些辅助模块属于示例，不属于安装包的公开 API。
它们不实现能量求解算法，也不修改计算能量。
库负责结构归一化和能量计算。
精度限值只适用于本章的示例及其活性空间。

## 安装与运行

核心依赖支持第一例。
晶体输入还要求 ASE，可通过 `viz` 安装。
完整模式还要求 `sqd`，支持平台为 Linux x86_64。
以下安装包含五个示例及其测试所需的依赖：

```bash
uv sync --frozen --extra dev --extra viz --extra sqd
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
```

在启动 Python 前设置这些变量。
SQD 要求原生库采用串行线程。
仅运行经典计算或结构准备时，可省略 `--extra sqd`。
如果多个工作树共享虚拟环境，请设置当前源码路径：

```bash
export PYTHONPATH="$PWD/src"
```

按以下顺序运行示例：

```bash
uv run --no-sync python -m examples.fragments.water_dimer_classical
uv run --no-sync python -m examples.fragments.water_dimer_full
uv run --no-sync python -m examples.fragments.crystal_dimer_prepare
uv run --no-sync python -m examples.fragments.crystal_dimer_interaction
uv run --no-sync python -m examples.fragments.fragment_report
```

所有入口都支持 `--help` 和 `--output`。
晶体能量入口和汇总入口还支持完整模式：

```bash
uv run --no-sync python -m examples.fragments.crystal_dimer_interaction --mode full
uv run --no-sync python -m examples.fragments.fragment_report --mode full
```

## 固定输入

水输入采用教案中的 S22 水二聚体，即体系 2。
片段 A 包含原子 0–2，片段 B 包含原子 3–5。
晶体输入来自 COD 2010678。
片段 A 为 4-硝基苯酚，片段 B 为 4-甲基吡啶。
[数据归属说明](data/README.md)记录来源及原始论文。

| 体系 | 基组 | AVAS 标签 | 阈值 | AB | A | B |
| --- | --- | --- | --- | --- | --- | --- |
| 水二聚体 | cc-pVDZ | `O 2p`, `H 1s` | 0.7 | CAS(12e,8o) | CAS(6e,4o) | CAS(6e,4o) |
| 晶体二聚体 | STO-3G | `0 O 2p`, `4 N 2p` | 0.1 | CAS(12e,9o) | CAS(6e,4o) | CAS(6e,5o) |

AVAS 按原子价轨道投影选择活性空间。
CAS(ne,no) 表示活性电子数和空间轨道数。
SQD 每个活性空间轨道使用两个量子比特。
带索引的 AVAS 标签采用原始原子顺序，索引从零开始。
经典模式与完整模式都检查活性空间的对应关系。

CIF 流程只读取非对称单元。
该流程不展开对称性或周期镜像。
晶体能量示例使用 X–H 键长归一化后的坐标。
归一化只沿原键方向移动 H 原子。
重原子坐标和片段归属保持不变。

| 键 | 目标键长（Å） |
| --- | ---: |
| C–H | 1.083 |
| N–H | 1.009 |
| O–H | 0.983 |

[结构 API](../../src/q2m3/fragments/structure.py)记录目标值的来源。
归一化不是几何优化，也不确定真实氢原子位置。
归一化不模拟氢键极化。
STO-3G 降低了教学计算的成本。
该基组和所选活性空间限制了定量解释。
这些设置不能证明基组或活性空间已收敛。

## 输出与失败证据

每个入口在 `data/output/examples/` 下创建独占运行目录。
用 `--output PATH` 指定其他新目录。
已有目录会在计算开始前使命令失败。
仓库不提交计算结果。

| 文件 | 内容 |
| --- | --- |
| `run.json` | 请求参数、开始时间、状态、耗时 |
| `input.json` | 几何结构、片段、几何哈希、计算设置 |
| 原始 XYZ 或 CIF | 示例输入的副本 |
| `result.json` | 完整结果、警告、缺失值原因 |
| `summary.csv` | 五种方法的相互作用能，单位为 Ha 和 kcal/mol |
| `report.md` | 库输出的中文报告、警告、精度验收 |
| `accuracy.json` | 完整模式的各子系统误差与相互作用能误差 |
| `failure.json` | 失败时的异常类型、信息、调用栈 |

结构准备保存 `original.xyz`、`normalized.xyz` 和 `fragments.json`。
`xh_bonds.csv` 记录各 H 原子及其键长变化。
结构准备不计算能量。
晶体能量入口也保留这些结构文件。

汇总入口为每个体系创建子目录。
汇总目录保存 `results.json`、`summary.csv` 和 `report.md`。
CSV 标注各体系的基组和结果路径。
单项失败后，汇总入口继续计算下一项。
任何单项失败都会使最终命令返回非零退出码。
参考文件或 Ks 文件无效时，入口会在计算前终止。

## 完整模式与精度验收

完整模式默认使用 seed 0。
每个子系统默认采样 100,000 次。
UCJ 指幺正簇 Jastrow 算符。
水示例预留 24 个 UCJ 分解项，晶体示例预留 36 项。
这些数量覆盖各二聚体的 `2 * n_occupied * n_virtual`。
该上界覆盖实数 CCSD 双激发振幅的完整分解。

ffsim 用恒等算符补足单体未使用的位置。
库的通用默认值仍为两项，片段示例显式设置更多项。
态制备使用 CCSD 振幅，不做变分优化。
CASCI 系数不参与采样行列式的选择。
示例不使用 CASCI 能量替代 SQD 能量。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--seed` | 0 | 随机种子 |
| `--shots` | 100000 | 每个子系统的采样次数 |
| `--wall-budget` | 900 s | 每次 SQD 调用的时间预算 |
| `--rss-budget` | 8192 MB | 每次 SQD 调用的内存预算 |
| `--ccsd-t-wall-budget` | 7200 s | AB、A、B 的 CCSD(T) 预测总耗时上限 |
| `--ccsd-t-rss-budget` | 20000 MB | CCSD(T) 的预测峰值内存上限 |

MB 采用十进制定义。
预算必须是有限正数。
上述预算不限制整个示例的总运行时间。
SQD 参考计算可因预算限制降级。
结果会保留参考层级、降级原因、尝试记录和警告。

完整模式必须满足以下条件：

1. AB、A、B 均有可信的 T0 参考。
2. UCJ 分解项数覆盖各子系统所需的完整分解。
3. 各子系统的 SQD − CASCI 误差不超过 +0.5 mHa。
4. 变分下界最多允许 −1e-6 Ha 数值偏差。
5. 相互作用能误差的绝对值不超过 0.25 kcal/mol。

T0 指选定活性空间内的精确 CASCI 参考。
验收失败时，示例保留结果和 `accuracy.json`。
命令随后返回非零退出码。
减少采样次数可能导致验收失败。
这些限值不证明完整体系的精度。

CCSD(T) 在运行前检查预测成本。
其成本模型采用 cc-pVDZ、八线程校准。
实际条件不同时，请阅读结果中的成本警告。
预算拒绝后，CCSD(T) 值为空，其他完整模式阶段继续运行。
以下命令演示该路径：

```bash
uv run --no-sync python -m examples.fragments.water_dimer_full --ccsd-t-wall-budget 0.000001
```

## 外部参考与 Ks

汇总入口可导入用户提供的参考文件：

```bash
uv run --no-sync python -m examples.fragments.fragment_report \
  --references /path/to/references.csv --ks-table /path/to/ks.json
```

参考 CSV 必须覆盖 `water-dimer` 和 `nitrophenol-methylpyridine`。
必需列为：

```csv
fragment_id,method,basis,delta_e,unit
```

`unit` 支持 `Ha`、`kcal/mol` 或 `kJ/mol`。
导入器按每个体系的计算基组分类。
每个体系最多有一项同基组参考和一项外部参考。
参考几何必须匹配保存的输入，包括晶体 X–H 归一化。
参考能量必须采用同一 counterpoise 定义。
仅凭 CSV 内容，导入器无法证明这两个科学条件成立。

Ks 文件采用 JSON 标签映射，值必须是有限数值。
标签应说明实验条件和单位。
标签必须非空，不能包含换行或竖线。
汇总目录保留两个导入文件。
报告只并列展示 Ks，不提供统计结论。
示例不提供虚构参考值或实验数据。

## 结果解释

所有相互作用能采用 `ΔE = E(AB) − E(A) − E(B)`。
counterpoise 用 ghost 原子保留二聚体基组。
ghost 原子只提供基函数，不提供原子核或电子。
负值表示该几何下的二聚体能量低于两个单体之和。
经典模式提供 HF 和冻结核 MP2。

CASCI 只在选定活性空间内精确。
SQD − CASCI 衡量该空间内的算法误差。
CASCI 与 CCSD(T) 的差异也包含相关方法差异。
CCSD(T) 本身仍是近似方法。
外部参考若改变方法，报告中的基组差异也包含方法差异。

空值不表示零。
JSON、CSV 和 Markdown 都保留缺失值原因。
计算差值时，应使用未舍入的 JSON 或 CSV 数据。
报告在计算差值后再舍入。
两个体系的基组不同，不能据此作受控的相互作用强度排序。

QPE 表提供活性空间哈密顿量的资源估计。
这些示例不运行 QPE。
SQD 表记录工作量和预测内存。
预测内存不等于整个进程的实测峰值。

## 实测结果

以下完整模式结果来自 **2026-10-25** 的本地运行。
运行使用 Python 3.11.17、PySCF 2.14.0 和 NumPy 2.4.1。
SQD 使用 ffsim 0.0.84 和 qiskit-addon-sqd 0.13.1。
PennyLane 版本为 0.44.0。
各原生库采用单线程。
两个体系均使用 seed 0，每个子系统采样 100,000 次。

| 方法 | 水 ΔE（kcal/mol） | 晶体 ΔE（kcal/mol） |
| --- | ---: | ---: |
| HF | −3.682 | −4.754 |
| MP2 | −3.935 | −5.251 |
| CASCI | −3.574 | −4.844 |
| SQD | −3.530 | −4.743 |
| CCSD(T) | −3.699 | −3.557 |

| SQD − CASCI 相互作用能误差 | 水（kcal/mol） | 晶体（kcal/mol） |
| --- | ---: | ---: |
| seed 0 | +0.043327 | +0.101587 |
| seeds 0–4 范围 | +0.020358 至 +0.068442 | +0.101587 至 +0.183548 |

五个种子中的最大子系统误差为水 0.109069 mHa、晶体 0.300421 mHa。
该范围只描述这些种子，不是置信区间。
它也不能保证其他体系或采样预算达到同样精度。

非相互作用测试采用两个晶体 A 片段哈密顿量。
片段间积分严格为零。
CASCI 相互作用能在 1e-8 Ha 内为零。
seed 0 的 SQD 残差为 +0.006214 kcal/mol。
该回归测试的限值为 0.05 kcal/mol。

| 运行 | 实测耗时 | 范围 |
| --- | ---: | --- |
| 晶体结构准备 | 2.36 s | 包含初次导入 |
| 经典汇总 | 10.24 s | 两个体系及汇总输出 |
| 水完整计算子项 | 13.44 s | 完整汇总内，初次导入之后 |
| 晶体完整计算子项 | 197.58 s | 完整汇总内 |
| 完整汇总 | 214.74 s | 两个体系、精度验收、汇总输出 |
| 水完整计算，CCSD(T) 预算拒绝 | 14.05 s | 其余完整模式阶段完成 |

时间来自 `run.json`，不包含创建输出目录前的解释器启动时间。
部分检查命令曾并行运行。
这些时间不是性能保证。
结果保留了八线程 CCSD(T) 校准警告。
结果也保留了本环境的 JAX/fork 警告。

## 检查与复算

采用前述线程设置，运行以下命令：

```bash
uv run --no-sync pytest tests/examples/test_fragment_examples.py \
  tests/examples/test_fragment_sqd_accuracy.py tests/fragments -n 0
uv run --no-sync pytest tests/examples/test_fragment_sqd_numerics.py -n 0
uv run --no-sync ruff check src/ tests/ examples/fragments/
uv run --no-sync black --check src/ tests/ examples/fragments/ --line-length 100
uv run --no-sync python tests/sqd/ucj_profile_probe.py /tmp/ucj-profile.json 100
```

数值测试通过公开 SQD 积分入口运行两个体系及五个种子。
测试还检查双激发振幅重构和非相互作用片段的能量可加性。
资源模型最多支持 36 个分解项和 100,000 次采样。
模型支持闭壳层体系的 2–10 个活性空间轨道。
每个自旋最多有 252 个行列式，总数最多为 63504。
求解器设置沿用原模型。
各阶段的导入与原生库内存预留为 1024 MB。

资源探针在 27 个独立进程中测量态制备和采样。
该探针固定原生线程数。
生产调用保留进程树 RSS 上限和时间预算。
探针不能证明任意瞬时内存峰值都在预算内。

返回[示例索引](../README_CN.md)。
