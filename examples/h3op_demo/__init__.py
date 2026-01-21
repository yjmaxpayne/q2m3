# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
H3O+ QPE Demo Package

Modular components for the H3O+ Quantum Phase Estimation demonstration.
This package provides a structured approach to hybrid quantum-classical
QM/MM calculations targeting early fault-tolerant quantum computers (EFTQC).

Submodules:
    config: Constants, device detection, and configuration functions
    output: Print and display functions for demo output
    analysis: Solvation effect analysis functions
    computation: Resource estimation and Catalyst analysis
    data_builder: Output data construction for JSON export
    profiling: Performance profiling utilities
"""

from examples.h3op_demo.analysis import (
    analyze_qpe_solvation_effect,
    analyze_solvation_effect,
)
from examples.h3op_demo.computation import (
    run_catalyst_analysis,
    run_resource_estimation,
)
from examples.h3op_demo.config import (
    CATALYST_VERSION,
    CHEMICAL_ACCURACY_ERROR,
    ENERGY_CONSISTENCY_THRESHOLD,
    HARTREE_TO_KCAL_MOL,
    HAS_CATALYST,
    HAS_JAX_CUDA,
    HAS_LIGHTNING_GPU,
    HAS_LIGHTNING_QUBIT,
    JAX_DEFAULT_BACKEND,
    MM_STABILIZATION_THRESHOLD,
    RELAXED_ACCURACY_ERROR,
    create_h3o_geometry,
    get_best_available_device,
    get_catalyst_effective_backend,
    get_qpe_config,
)
from examples.h3op_demo.data_builder import build_output_data
from examples.h3op_demo.output import (
    print_comparison,
    print_header,
    print_hf_solvation_effect,
    print_qpe_solvation_effect,
    print_resource_estimation,
    print_section,
    print_summary,
    print_system_info,
)
from examples.h3op_demo.profiling import profile_function, profile_section

__all__ = [
    # Constants
    "HARTREE_TO_KCAL_MOL",
    "MM_STABILIZATION_THRESHOLD",
    "ENERGY_CONSISTENCY_THRESHOLD",
    "CHEMICAL_ACCURACY_ERROR",
    "RELAXED_ACCURACY_ERROR",
    # Device detection
    "HAS_CATALYST",
    "CATALYST_VERSION",
    "HAS_LIGHTNING_GPU",
    "HAS_LIGHTNING_QUBIT",
    "HAS_JAX_CUDA",
    "JAX_DEFAULT_BACKEND",
    # Config functions
    "get_best_available_device",
    "get_catalyst_effective_backend",
    "create_h3o_geometry",
    "get_qpe_config",
    # Output functions
    "print_header",
    "print_section",
    "print_system_info",
    "print_hf_solvation_effect",
    "print_resource_estimation",
    "print_qpe_solvation_effect",
    "print_comparison",
    "print_summary",
    # Analysis functions
    "analyze_solvation_effect",
    "analyze_qpe_solvation_effect",
    # Computation functions
    "run_resource_estimation",
    "run_catalyst_analysis",
    # Data builder
    "build_output_data",
    # Profiling
    "profile_section",
    "profile_function",
]
