#!/usr/bin/env python
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

# Ensure project root is in sys.path for running from examples/ directory
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

"""
H3O+ Quantum Phase Estimation (QPE) Demo

This script demonstrates q2m3's core capabilities for hybrid quantum-classical
QM/MM calculations targeting early fault-tolerant quantum computers (EFTQC).

Key features demonstrated:
1. PySCF to PennyLane Hamiltonian conversion
2. Standard QPE circuit (HF state prep -> Trotter evolution -> inverse QFT)
3. Catalyst @qjit JIT compilation for circuit optimization
4. QM/MM system setup with TIP3P water solvation

IMPORTANT: GPU Support Architecture
------------------------------------
q2m3 has TWO separate GPU support systems:

1. PennyLane Lightning GPU (for standard QPE):
   - Device: lightning.gpu
   - Requirements: pennylane-lightning[gpu], cuQuantum, CUDA
   - Status checked by: HAS_LIGHTNING_GPU

2. JAX/Catalyst GPU (for @qjit compiled circuits):
   - Backend: JAX with CUDA support
   - Requirements: jax[cuda12] or jax[cuda11]
   - Status checked by: HAS_JAX_CUDA

When Catalyst @qjit is enabled, execution backend is determined by JAX,
NOT PennyLane device selection. If JAX lacks CUDA support, Catalyst
runs on CPU regardless of lightning.gpu availability.
"""

import warnings

# Analysis module exports
from examples.h3op_demo.analysis import (
    analyze_qpe_solvation_effect,
    analyze_solvation_effect,
)

# Computation module exports
from examples.h3op_demo.computation import (
    run_catalyst_analysis,
    run_resource_estimation,
)

# Config module exports
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

# Data builder module exports
from examples.h3op_demo.data_builder import build_output_data

# Output module exports
from examples.h3op_demo.output import (
    print_comparison,
    print_header,
    print_hf_solvation_effect,
    print_profiling_report,
    print_qpe_solvation_effect,
    print_resource_estimation,
    print_section,
    print_summary,
    print_system_info,
)

# Profiling module exports (new)
from examples.h3op_demo.profiling import profile_function, profile_section
from q2m3.core import QuantumQMMM
from q2m3.utils import save_json_results

# =============================================================================
# Re-export all public API from submodules for backward compatibility
# =============================================================================


# =============================================================================
# All public API
# =============================================================================

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
    "print_profiling_report",
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
    # Main
    "main",
]


def main():
    """Main demo execution with performance profiling."""
    # Initialize profiling data collection
    profiling_data = {}

    with profile_section("Total Demo Execution", verbose=False) as total_timing:
        print_header()

        # Step 1: System configuration
        print_section("System Configuration", step=1)
        h3o_atoms = create_h3o_geometry()
        mm_waters = 8
        qpe_config = get_qpe_config(device_type="auto")

        print_system_info(qpe_config, mm_waters)
        print()
        selected_device = get_best_available_device()
        device_note = " (GPU detected)" if HAS_LIGHTNING_GPU else ""
        print(f"Device Selection: auto -> {selected_device}{device_note}")

        # Step 1.5: Circuit visualization (show circuit structure before calculations)
        print_section("Circuit Visualization (PennyLane)", step=1.5)
        print("Generating QPE + RDM circuit diagrams...")
        print()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            viz_qmmm = QuantumQMMM(
                qm_atoms=h3o_atoms,
                mm_waters=mm_waters,
                qpe_config=qpe_config,
                use_catalyst=False,
            )
            circuits = viz_qmmm.draw_circuits()

        print("QPE Circuit (Standard Phase Estimation):")
        print("-" * 60)
        print(circuits["qpe"])
        print()
        print("RDM Measurement Circuit (Pauli Expectation Values):")
        print("-" * 60)
        print(circuits["rdm"])
        print()

        # Step 2: EFTQC Resource Estimation (with profiling)
        print_section("EFTQC Resource Estimation (Vacuum vs Solvated)", step=2)
        with profile_section("Resource Estimation") as resource_timing:
            eftqc_data = run_resource_estimation(h3o_atoms, mm_waters)
        profiling_data["resource_estimation"] = resource_timing
        print_resource_estimation(eftqc_data, mm_waters)

        # Step 3: Solvation effect analysis (classical HF level, with profiling)
        print_section("Solvation Effect Analysis (Classical HF)", step=3)
        print("Comparing H3O+ energy in vacuum vs. explicit TIP3P water environment...")
        print("This validates that MM embedding correctly polarizes the QM electron density.")
        print()

        with profile_section("HF Solvation Analysis") as hf_timing:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solvation_data = analyze_solvation_effect(h3o_atoms, mm_waters)
        profiling_data["hf_solvation"] = hf_timing
        print_hf_solvation_effect(solvation_data)

        # Step 4: QPE solvation effect analysis (Standard QPE, with profiling)
        print_section("Standard QPE Solvation Effect Analysis (Quantum Level)", step=4)
        print("Comparing QPE energies: vacuum vs. explicit TIP3P solvation...")
        print("This validates MM embedding is correctly included in the quantum Hamiltonian.")
        print()

        with profile_section("Standard QPE (total)") as standard_timing:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                qpe_solvation_data = analyze_qpe_solvation_effect(
                    h3o_atoms, mm_waters, qpe_config, use_catalyst=False
                )
        profiling_data["standard_qpe"] = standard_timing
        print_qpe_solvation_effect(qpe_solvation_data, solvation_data, label="Standard QPE")

        # Step 5: Catalyst @qjit QPE Solvation Effect Analysis (with profiling)
        print_section("Catalyst @qjit QPE Solvation Effect Analysis", step=5)
        with profile_section("Catalyst QPE (total)") as catalyst_timing:
            catalyst_solvation_data = run_catalyst_analysis(h3o_atoms, mm_waters, solvation_data)
        profiling_data["catalyst_qpe"] = catalyst_timing

        # Step 6: Results comparison
        print_section("Results Comparison", step=6)
        print_comparison(qpe_solvation_data, catalyst_solvation_data)

        # Step 7: Save results
        print_section("Save Results", step=7)
        output_data = build_output_data(
            h3o_atoms,
            mm_waters,
            qpe_config,
            solvation_data,
            qpe_solvation_data,
            catalyst_solvation_data,
            eftqc_data,
        )

        output_file = "data/output/h3o_quantum_qpe_results.json"
        save_json_results(output_data, output_file)
        print(f"Results saved to: {output_file}")

        # Summary
        print_section("Demo Summary")
        print_summary(solvation_data, qpe_solvation_data, catalyst_solvation_data, eftqc_data)

    # Store total timing
    profiling_data["total"] = total_timing

    # Print profiling report (key insight for jit + lightning.gpu bottleneck analysis)
    print_profiling_report(profiling_data)


if __name__ == "__main__":
    main()
