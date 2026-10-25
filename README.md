# q2m3: A Hybrid Quantum-Classical QM/MM Simulation Framework

[English](README.md) | [简体中文](README_CN.md)

<p align="center">
  <img src="doc/source/_static/logo.svg" alt="q2m3 logo" width="360">
</p>

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://github.com/yjmaxpayne/q2m3/actions/workflows/docs.yml/badge.svg)](https://github.com/yjmaxpayne/q2m3/actions/workflows/docs.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://yjmaxpayne.github.io/q2m3/)
[![codecov](https://codecov.io/gh/yjmaxpayne/q2m3/graph/badge.svg)](https://codecov.io/gh/yjmaxpayne/q2m3)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20114945.svg)](https://doi.org/10.5281/zenodo.20114945)
[![PennyLane](https://img.shields.io/badge/PennyLane-%3E%3D0.44.0-01A982)](https://pennylane.ai/)
[![Catalyst](https://img.shields.io/badge/Catalyst-%3E%3D0.14.0-01A982)](https://github.com/PennyLaneAI/catalyst)
[![PySCF](https://img.shields.io/badge/PySCF-%3E%3D2.0.0-blue)](https://pyscf.org/)

q2m3 is a research framework for hybrid quantum-classical QM/MM
(quantum mechanics / molecular mechanics). It connects PySCF molecular models
to PennyLane quantum algorithms and resource estimates.

The project is an alpha proof of concept. Its small-system checks do not
establish production chemistry accuracy. Read each example's assumptions before
interpreting its results.

## Install

q2m3 requires Python 3.11+. To use the library, install it in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install q2m3
```

To run examples, use a source checkout with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/yjmaxpayne/q2m3.git
cd q2m3
uv sync
```

Each example guide lists its optional dependencies and execution commands.

## Examples and documentation

Start with the [example guide](examples/README.md).
Choose a chapter for runnable examples, inputs, outputs, and scientific limits.

| Topic | Guide |
| --- | --- |
| Quantum phase estimation (QPE) | [QPE examples](examples/qpe/README.md) |
| Sample-based quantum diagonalization (SQD) | [SQD examples](examples/sqd/README.md) |
| QM/MM and solvation | [QM/MM examples](examples/qmmm/README.md) |
| Fragment interactions | [Fragment examples](examples/fragments/README.md) |
| Quantum resource estimates | [Resource examples](examples/resources/README.md) |
| Compilation and runtime | [Performance examples](examples/performance/README.md) |

See the [documentation site](https://yjmaxpayne.github.io/q2m3/) for tutorials,
model assumptions, and API references.

## Development, citation, and license

See the [development guide](doc/source/development.md) for tests, code checks,
and documentation builds. Follow the [contribution notes](doc/source/development.md#contribution-notes).
Do not commit generated outputs.

For research use, cite the release that produced your results.
[CITATION.cff](CITATION.cff) contains the citation metadata.
q2m3 uses the [MIT License](LICENSE).
