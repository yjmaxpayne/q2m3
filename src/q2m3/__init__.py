# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Hybrid quantum-classical chemistry framework.

q2m3 provides QPE, QM/MM solvation, resource-estimation, and optional
sample-based quantum diagonalization workflows.
"""

from typing import Any as _Any

from q2m3._lazy import available_exports as _available_exports
from q2m3._lazy import lazy_getattr as _lazy_getattr

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

_LAZY_EXPORTS = {
    "run_solvation": ("q2m3.solvation.orchestrator", "catalyst"),
    "run_sqd": ("q2m3.sqd.orchestrator", "sqd"),
}
__all__ += _available_exports(_LAZY_EXPORTS)


def __getattr__(name: str) -> _Any:
    return _lazy_getattr(__name__, globals(), _LAZY_EXPORTS, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
