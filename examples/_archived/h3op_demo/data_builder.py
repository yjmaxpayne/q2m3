# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Data builder module for H3O+ QPE Demo.

Contains output data construction for JSON export.
"""

from datetime import datetime

from examples.h3op_demo.config import (
    CATALYST_VERSION,
    HAS_CATALYST,
    HAS_JAX_CUDA,
    HAS_LIGHTNING_GPU,
    JAX_DEFAULT_BACKEND,
    MM_STABILIZATION_THRESHOLD,
    get_best_available_device,
    get_catalyst_effective_backend,
)
from q2m3.core.qmmm_system import Atom


def build_output_data(
    h3o_atoms: list[Atom],
    mm_waters: int,
    qpe_config: dict,
    solvation_data: dict,
    qpe_solvation_data: dict,
    catalyst_solvation_data: dict | None,
    eftqc_data: dict,
) -> dict:
    """Build output data dictionary for JSON export."""
    actual_device = get_best_available_device()
    eftqc_vac_chem = eftqc_data["vacuum_chemical"]
    eftqc_vac_relax = eftqc_data["vacuum_relaxed"]
    eftqc_sol_chem = eftqc_data["solvated_chemical"]
    eftqc_sol_relax = eftqc_data["solvated_relaxed"]

    return {
        "timestamp": datetime.now().isoformat(),
        "catalyst_available": HAS_CATALYST,
        "catalyst_version": CATALYST_VERSION if HAS_CATALYST else None,
        "lightning_gpu_available": HAS_LIGHTNING_GPU,
        "jax_cuda_available": HAS_JAX_CUDA,  # Separate from Lightning GPU!
        "jax_backend": JAX_DEFAULT_BACKEND,  # Actual JAX execution backend
        "system": {
            "qm_region": "H3O+",
            "n_atoms": len(h3o_atoms),
            "total_charge": 1,
            "mm_waters": mm_waters,
        },
        "qpe_config": qpe_config,
        "quantum_resources": {
            "active_electrons": qpe_config["active_electrons"],
            "active_orbitals": qpe_config["active_orbitals"],
            "system_qubits": qpe_config["active_orbitals"] * 2,
            "estimation_qubits": qpe_config["n_estimation_wires"],
            "total_qubits": qpe_config["active_orbitals"] * 2 + qpe_config["n_estimation_wires"],
        },
        "solvation_effect_hf": {
            "energy_vacuum_hartree": solvation_data["energy_vacuum"],
            "energy_solvated_hartree": solvation_data["energy_solvated"],
            "stabilization_hartree": solvation_data["stabilization_hartree"],
            "stabilization_kcal_mol": solvation_data["stabilization_kcal_mol"],
            "mm_embedding_active": bool(
                solvation_data["stabilization_hartree"] > MM_STABILIZATION_THRESHOLD
            ),
        },
        "solvation_effect_standard_qpe": {
            "device": actual_device,
            "energy_vacuum_hartree": qpe_solvation_data["energy_vacuum"],
            "energy_solvated_hartree": qpe_solvation_data["energy_solvated"],
            "energy_hf_solvated": qpe_solvation_data["energy_hf_solvated"],
            "stabilization_hartree": qpe_solvation_data["stabilization_hartree"],
            "stabilization_kcal_mol": qpe_solvation_data["stabilization_kcal_mol"],
            "mm_embedding_active": bool(
                qpe_solvation_data["stabilization_hartree"] > MM_STABILIZATION_THRESHOLD
            ),
            "charges_vacuum": qpe_solvation_data["charges_vacuum"],
            "charges_solvated": qpe_solvation_data["charges_solvated"],
            "execution_time_vacuum_s": qpe_solvation_data["time_vacuum_s"],
            "execution_time_solvated_s": qpe_solvation_data["time_solvated_s"],
            "execution_time_total_s": qpe_solvation_data["time_total_s"],
        },
        "solvation_effect_catalyst_qpe": (
            {
                "pennylane_device": get_best_available_device(),  # PennyLane device name
                "jax_backend": JAX_DEFAULT_BACKEND,  # Actual JAX execution backend
                "effective_backend": get_catalyst_effective_backend(),  # Human-readable
                "has_jax_cuda": HAS_JAX_CUDA,  # Whether Catalyst uses GPU
                "energy_vacuum_hartree": catalyst_solvation_data["energy_vacuum"],
                "energy_solvated_hartree": catalyst_solvation_data["energy_solvated"],
                "energy_hf_solvated": catalyst_solvation_data["energy_hf_solvated"],
                "stabilization_hartree": catalyst_solvation_data["stabilization_hartree"],
                "stabilization_kcal_mol": catalyst_solvation_data["stabilization_kcal_mol"],
                "mm_embedding_active": bool(
                    catalyst_solvation_data["stabilization_hartree"] > MM_STABILIZATION_THRESHOLD
                ),
                "charges_vacuum": catalyst_solvation_data["charges_vacuum"],
                "charges_solvated": catalyst_solvation_data["charges_solvated"],
                "execution_time_vacuum_s": catalyst_solvation_data["time_vacuum_s"],
                "execution_time_solvated_s": catalyst_solvation_data["time_solvated_s"],
                "execution_time_total_s": catalyst_solvation_data["time_total_s"],
            }
            if catalyst_solvation_data is not None
            else None
        ),
        "eftqc_resources": {
            "vacuum": {
                "chemical_accuracy": {
                    "target_error_hartree": eftqc_vac_chem["target_error"],
                    "hamiltonian_1norm": eftqc_vac_chem["hamiltonian_1norm"],
                    "logical_qubits": eftqc_vac_chem["logical_qubits"],
                    "toffoli_gates": eftqc_vac_chem["toffoli_gates"],
                    "qpe_iterations": eftqc_vac_chem["qpe_iterations"],
                    "trotter_steps": eftqc_vac_chem["trotter_steps"],
                },
                "relaxed_accuracy": {
                    "target_error_hartree": eftqc_vac_relax["target_error"],
                    "hamiltonian_1norm": eftqc_vac_relax["hamiltonian_1norm"],
                    "logical_qubits": eftqc_vac_relax["logical_qubits"],
                    "toffoli_gates": eftqc_vac_relax["toffoli_gates"],
                    "qpe_iterations": eftqc_vac_relax["qpe_iterations"],
                    "trotter_steps": eftqc_vac_relax["trotter_steps"],
                },
            },
            "solvated": {
                "n_mm_charges": eftqc_sol_chem["n_mm_charges"],
                "chemical_accuracy": {
                    "target_error_hartree": eftqc_sol_chem["target_error"],
                    "hamiltonian_1norm": eftqc_sol_chem["hamiltonian_1norm"],
                    "logical_qubits": eftqc_sol_chem["logical_qubits"],
                    "toffoli_gates": eftqc_sol_chem["toffoli_gates"],
                    "qpe_iterations": eftqc_sol_chem["qpe_iterations"],
                    "trotter_steps": eftqc_sol_chem["trotter_steps"],
                },
                "relaxed_accuracy": {
                    "target_error_hartree": eftqc_sol_relax["target_error"],
                    "hamiltonian_1norm": eftqc_sol_relax["hamiltonian_1norm"],
                    "logical_qubits": eftqc_sol_relax["logical_qubits"],
                    "toffoli_gates": eftqc_sol_relax["toffoli_gates"],
                    "qpe_iterations": eftqc_sol_relax["qpe_iterations"],
                    "trotter_steps": eftqc_sol_relax["trotter_steps"],
                },
            },
            "analysis": {
                "mm_embedding_1norm_increase_percent": eftqc_data["delta_lambda"],
                "error_relaxation_gate_reduction_percent": eftqc_data["gate_reduction"],
            },
        },
    }
