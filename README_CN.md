# q2m3：量子经典混合 QM/MM 模拟框架

[English](README.md) | [简体中文](README_CN.md)

<p align="center">
  <img src="doc/source/_static/logo.svg" alt="q2m3 logo" width="360">
</p>

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://github.com/yjmaxpayne/q2m3/actions/workflows/docs.yml/badge.svg)](https://github.com/yjmaxpayne/q2m3/actions/workflows/docs.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://yjmaxpayne.github.io/q2m3/zh_CN/)
[![codecov](https://codecov.io/gh/yjmaxpayne/q2m3/graph/badge.svg)](https://codecov.io/gh/yjmaxpayne/q2m3)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20114945.svg)](https://doi.org/10.5281/zenodo.20114945)
[![PennyLane](https://img.shields.io/badge/PennyLane-%3E%3D0.44.0-01A982)](https://pennylane.ai/)
[![Catalyst](https://img.shields.io/badge/Catalyst-%3E%3D0.14.0-01A982)](https://github.com/PennyLaneAI/catalyst)
[![PySCF](https://img.shields.io/badge/PySCF-%3E%3D2.0.0-blue)](https://pyscf.org/)

q2m3 是量子经典混合 QM/MM（量子力学 / 分子力学）研究框架。
它将 PySCF 分子模型连接到 PennyLane 量子算法和资源估计。

项目目前处于 alpha 概念验证阶段。
现有小体系检查不足以证明其具备生产级化学精度。
解释结果前，请阅读各示例的适用条件。

## 安装

q2m3 要求 Python 3.11+。
使用库时，请在虚拟环境中安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install q2m3
```

运行示例时，请使用源码仓库和 [uv](https://docs.astral.sh/uv/getting-started/installation/)：

```bash
git clone https://github.com/yjmaxpayne/q2m3.git
cd q2m3
uv sync
```

各示例指南列出了可选依赖和运行命令。

## 示例与文档

请从[示例指南](examples/README_CN.md)开始。
各章节说明输入、输出和科学边界。

| 主题 | 指南 |
| --- | --- |
| 量子相位估计（QPE） | [QPE 示例](examples/qpe/README.md) |
| 基于采样的量子对角化（SQD） | [SQD 示例](examples/sqd/README.md) |
| QM/MM 与溶剂化 | [QM/MM 示例](examples/qmmm/README.md) |
| 片段相互作用 | [片段示例](examples/fragments/README_CN.md) |
| 量子资源估计 | [资源示例](examples/resources/README.md) |
| 编译与运行成本 | [性能示例](examples/performance/README.md) |

[文档站点](https://yjmaxpayne.github.io/q2m3/zh_CN/)提供教程、模型假设和 API 参考。

## 开发、引用与许可证

[开发指南](doc/source/development.md)列出了测试、代码检查和文档构建命令。
请遵循[贡献约定](doc/source/development.md#contribution-notes)。
不要提交生成的输出文件。

研究工作应引用实际使用的版本。
[CITATION.cff](CITATION.cff)提供引用元数据。
q2m3 采用 [MIT 许可证](LICENSE)。
