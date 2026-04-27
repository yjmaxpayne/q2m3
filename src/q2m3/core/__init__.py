# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Core module for Quantum-QM/MM calculations.

Contains the main computational engines and system builders.
"""

from .device_utils import (
    CATALYST_VERSION,
    HAS_CATALYST,
    HAS_JAX_CUDA,
    HAS_LIGHTNING_GPU,
    HAS_LIGHTNING_QUBIT,
    JAX_DEFAULT_BACKEND,
    get_best_available_device,
    get_catalyst_effective_backend,
    select_device,
)
from .hamiltonian_utils import build_operator_index_map, decompose_hamiltonian
from .qmmm_system import QMMMSystem
from .qpe import QPEEngine
from .quantum_qmmm import QuantumQMMM
from .rdm import RDMEstimator, measure_rdm_from_qpe_state
from .resource_estimation import (
    EFTQCResources,
    ResourceComparisonResult,
    compare_vacuum_solvated,
    derive_t_resources,
    estimate_eftqc_runtime,
    estimate_resources,
)

__all__ = [
    "decompose_hamiltonian",
    "build_operator_index_map",
    # Resource estimation
    "EFTQCResources",
    "ResourceComparisonResult",
    "estimate_resources",
    "compare_vacuum_solvated",
    "derive_t_resources",
    "estimate_eftqc_runtime",
    "QPEEngine",
    "QMMMSystem",
    "QuantumQMMM",
    "RDMEstimator",
    "measure_rdm_from_qpe_state",
    # Device utilities
    "select_device",
    "get_best_available_device",
    "get_catalyst_effective_backend",
    "HAS_LIGHTNING_GPU",
    "HAS_LIGHTNING_QUBIT",
    "HAS_JAX_CUDA",
    "JAX_DEFAULT_BACKEND",
    "HAS_CATALYST",
    "CATALYST_VERSION",
]
