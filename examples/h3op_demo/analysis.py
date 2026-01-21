# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Analysis module for H3O+ QPE Demo.

Contains solvation effect analysis functions.
"""

import time

import numpy as np
from pyscf import gto

from examples.h3op_demo.config import HARTREE_TO_KCAL_MOL
from q2m3.core import QuantumQMMM
from q2m3.core.qmmm_system import Atom, QMMMSystem
from q2m3.interfaces import PySCFPennyLaneConverter


def analyze_solvation_effect(h3o_atoms: list[Atom], mm_waters: int) -> dict:
    """
    Analyze the solvation effect on H3O+ energy at the classical HF level.

    Compares vacuum vs solvated H3O+ to demonstrate MM embedding is working.

    Args:
        h3o_atoms: H3O+ atomic geometry
        mm_waters: Number of TIP3P water molecules

    Returns:
        Dictionary with vacuum energy, solvated energy, and stabilization energy
    """
    # Build QM/MM system (MM environment set up automatically in __init__)
    qmmm_system = QMMMSystem(qm_atoms=h3o_atoms, num_waters=mm_waters)
    mol_dict = qmmm_system.to_pyscf_mol()

    # Create PySCF molecule
    mol = gto.M(**mol_dict)

    # Get MM embedding potential
    mm_charges, mm_coords = qmmm_system.get_embedding_potential()

    converter = PySCFPennyLaneConverter()

    # Calculate energy in vacuum (no MM)
    result_vacuum = converter.build_qmmm_hamiltonian(mol, np.array([]), np.array([]).reshape(0, 3))
    energy_vacuum = result_vacuum["energy_hf"]

    # Calculate energy with MM embedding
    result_solvated = converter.build_qmmm_hamiltonian(mol, mm_charges, mm_coords)
    energy_solvated = result_solvated["energy_hf"]

    # Stabilization energy (positive = MM stabilizes the system)
    stabilization = energy_vacuum - energy_solvated
    stabilization_kcal = stabilization * HARTREE_TO_KCAL_MOL

    return {
        "energy_vacuum": energy_vacuum,
        "energy_solvated": energy_solvated,
        "stabilization_hartree": stabilization,
        "stabilization_kcal_mol": stabilization_kcal,
        "n_mm_atoms": len(mm_charges),
        "n_mm_waters": mm_waters,
    }


def analyze_qpe_solvation_effect(
    h3o_atoms: list[Atom], mm_waters: int, qpe_config: dict, use_catalyst: bool = False
) -> dict:
    """
    Analyze solvation effect at the QPE quantum level.

    Compares vacuum vs solvated QPE energies to verify MM embedding
    is correctly included in the quantum Hamiltonian.

    Args:
        h3o_atoms: H3O+ atomic geometry
        mm_waters: Number of TIP3P water molecules
        qpe_config: QPE configuration dictionary
        use_catalyst: Enable Catalyst @qjit compilation

    Returns:
        Dictionary with vacuum QPE, solvated QPE, comparison metrics, and timing
    """
    # QPE in vacuum (no MM waters)
    start_vacuum = time.perf_counter()
    qmmm_vacuum = QuantumQMMM(
        qm_atoms=h3o_atoms,
        mm_waters=0,
        qpe_config=qpe_config,
        use_catalyst=use_catalyst,
    )
    result_vacuum = qmmm_vacuum.compute_ground_state()
    time_vacuum = time.perf_counter() - start_vacuum

    # QPE with MM solvation
    start_solvated = time.perf_counter()
    qmmm_solvated = QuantumQMMM(
        qm_atoms=h3o_atoms,
        mm_waters=mm_waters,
        qpe_config=qpe_config,
        use_catalyst=use_catalyst,
    )
    result_solvated = qmmm_solvated.compute_ground_state()
    time_solvated = time.perf_counter() - start_solvated

    # Compute stabilization
    stabilization = result_vacuum["energy"] - result_solvated["energy"]
    stabilization_kcal = stabilization * HARTREE_TO_KCAL_MOL

    # Extract fine-grained timing from compute_ground_state results
    timing_vacuum = result_vacuum.get("timing", {})
    timing_solvated = result_solvated.get("timing", {})

    return {
        "energy_vacuum": result_vacuum["energy"],
        "energy_solvated": result_solvated["energy"],
        "energy_hf_vacuum": result_vacuum.get("energy_hf", None),
        "energy_hf_solvated": result_solvated.get("energy_hf", None),
        "stabilization_hartree": stabilization,
        "stabilization_kcal_mol": stabilization_kcal,
        "charges_vacuum": result_vacuum["atomic_charges"],
        "charges_solvated": result_solvated["atomic_charges"],
        "convergence_solvated": result_solvated["convergence"],
        "rdm_source_solvated": result_solvated.get("rdm_source", "unknown"),
        "time_vacuum_s": time_vacuum,
        "time_solvated_s": time_solvated,
        "time_total_s": time_vacuum + time_solvated,
        # Fine-grained timing breakdown for bottleneck analysis
        "timing_vacuum": timing_vacuum,
        "timing_solvated": timing_solvated,
    }
