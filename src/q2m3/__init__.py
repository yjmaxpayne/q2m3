# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Quantum-QM/MM POC Framework

A hybrid quantum-classical framework for QM/MM calculations using QPE algorithms.
"""

from .version import __version__

__author__ = "Ye Jun <yjmaxpayne@hotmail.com>"

from .constants import (
    ANGSTROM_TO_BOHR,
    CHEMICAL_ACCURACY_HA,
    HARTREE_TO_KCAL_MOL,
    KCAL_TO_HARTREE,
    TIP3P_HYDROGEN_CHARGE,
    TIP3P_OXYGEN_CHARGE,
)
from .core import (
    CATALYST_VERSION,
    HAS_CATALYST,
    HAS_JAX_CUDA,
    HAS_LIGHTNING_GPU,
    HAS_LIGHTNING_QUBIT,
    JAX_DEFAULT_BACKEND,
    QMMMSystem,
    QPEEngine,
    QuantumQMMM,
    get_best_available_device,
    get_catalyst_effective_backend,
)
from .interfaces import (
    FixedMOEmbeddingDiagnostics,
    FixedMOEmbeddingResult,
    PySCFPennyLaneConverter,
    UnifiedDensityMatrix,
    build_fixed_mo_embedding_integrals,
)
from .molecule import MoleculeConfig
from .utils import load_xyz, save_json_results

__all__ = [
    "__version__",
    "QuantumQMMM",
    "QPEEngine",
    "QMMMSystem",
    "FixedMOEmbeddingDiagnostics",
    "FixedMOEmbeddingResult",
    "PySCFPennyLaneConverter",
    "UnifiedDensityMatrix",
    "build_fixed_mo_embedding_integrals",
    "load_xyz",
    "save_json_results",
    # Molecule config
    "MoleculeConfig",
    # Constants
    "HARTREE_TO_KCAL_MOL",
    "KCAL_TO_HARTREE",
    "ANGSTROM_TO_BOHR",
    "CHEMICAL_ACCURACY_HA",
    "TIP3P_OXYGEN_CHARGE",
    "TIP3P_HYDROGEN_CHARGE",
    # Device utilities
    "HAS_LIGHTNING_GPU",
    "HAS_LIGHTNING_QUBIT",
    "HAS_JAX_CUDA",
    "JAX_DEFAULT_BACKEND",
    "HAS_CATALYST",
    "CATALYST_VERSION",
    "get_best_available_device",
    "get_catalyst_effective_backend",
]
