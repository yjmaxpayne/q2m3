# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Quantum-QM/MM POC Framework

A hybrid quantum-classical framework for QM/MM calculations using QPE algorithms.
"""

__version__ = "0.1.0"
__author__ = "Ye Jun <yjmaxpayne@hotmail.com>"

from .core import QMMMSystem, QPEEngine, QuantumQMMM
from .interfaces import PySCFPennyLaneConverter, UnifiedDensityMatrix
from .utils import load_xyz, save_json_results

__all__ = [
    "QuantumQMMM",
    "QPEEngine",
    "QMMMSystem",
    "PySCFPennyLaneConverter",
    "UnifiedDensityMatrix",
    "load_xyz",
    "save_json_results",
]
