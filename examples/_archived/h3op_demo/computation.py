# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Computation module for H3O+ QPE Demo.

Contains resource estimation and Catalyst analysis functions.
"""

import warnings

import numpy as np

from examples.h3op_demo.analysis import analyze_qpe_solvation_effect
from examples.h3op_demo.config import (
    CHEMICAL_ACCURACY_ERROR,
    HAS_CATALYST,
    RELAXED_ACCURACY_ERROR,
    get_qpe_config,
)
from examples.h3op_demo.output import print_qpe_solvation_effect
from q2m3.core.qmmm_system import Atom, QMMMSystem
from q2m3.interfaces import PySCFPennyLaneConverter


def run_resource_estimation(h3o_atoms: list[Atom], mm_waters: int) -> dict:
    """Run EFTQC resource estimation for vacuum and solvated systems.

    Returns:
        Dictionary containing vacuum and solvated resource estimates,
        and derived analysis metrics (delta_lambda, gate_reduction).
    """
    h3o_symbols = [atom.symbol for atom in h3o_atoms]
    h3o_coords = np.array([atom.position for atom in h3o_atoms])

    qmmm_system = QMMMSystem(qm_atoms=h3o_atoms, num_waters=mm_waters)
    mm_charges, mm_coords = qmmm_system.get_embedding_potential()

    converter = PySCFPennyLaneConverter(basis="sto-3g", mapping="jordan_wigner")

    eftqc_vac_chem = converter.estimate_qpe_resources(
        symbols=h3o_symbols, coords=h3o_coords, charge=1, target_error=CHEMICAL_ACCURACY_ERROR
    )
    eftqc_vac_relax = converter.estimate_qpe_resources(
        symbols=h3o_symbols, coords=h3o_coords, charge=1, target_error=RELAXED_ACCURACY_ERROR
    )
    eftqc_sol_chem = converter.estimate_qpe_resources(
        symbols=h3o_symbols,
        coords=h3o_coords,
        charge=1,
        mm_charges=mm_charges,
        mm_coords=mm_coords,
        target_error=CHEMICAL_ACCURACY_ERROR,
    )
    eftqc_sol_relax = converter.estimate_qpe_resources(
        symbols=h3o_symbols,
        coords=h3o_coords,
        charge=1,
        mm_charges=mm_charges,
        mm_coords=mm_coords,
        target_error=RELAXED_ACCURACY_ERROR,
    )

    delta_lambda = (
        (eftqc_sol_chem["hamiltonian_1norm"] - eftqc_vac_chem["hamiltonian_1norm"])
        / eftqc_vac_chem["hamiltonian_1norm"]
        * 100
    )
    gate_reduction = (1 - eftqc_vac_relax["toffoli_gates"] / eftqc_vac_chem["toffoli_gates"]) * 100

    return {
        "vacuum_chemical": eftqc_vac_chem,
        "vacuum_relaxed": eftqc_vac_relax,
        "solvated_chemical": eftqc_sol_chem,
        "solvated_relaxed": eftqc_sol_relax,
        "delta_lambda": delta_lambda,
        "gate_reduction": gate_reduction,
        "n_mm_charges": len(mm_charges),
    }


def run_catalyst_analysis(
    h3o_atoms: list[Atom], mm_waters: int, solvation_data: dict
) -> dict | None:
    """Run Catalyst @qjit QPE solvation effect analysis.

    Returns:
        Catalyst solvation data dict if Catalyst is available, None otherwise.
    """
    if not HAS_CATALYST:
        print("WARNING: pennylane-catalyst is not installed.")
        print("To enable Catalyst support, install with:")
        print("  pip install pennylane-catalyst")
        print()
        print("Skipping Catalyst solvation effect analysis...")
        return None

    print("Comparing Catalyst QPE energies: vacuum vs. explicit TIP3P solvation...")
    print("This validates MM embedding works correctly with Catalyst JIT compilation.")
    print()

    # Use auto device selection (now supports lightning.gpu as of PennyLane 0.44.0)
    qpe_config_catalyst = get_qpe_config(device_type="auto")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        catalyst_solvation_data = analyze_qpe_solvation_effect(
            h3o_atoms, mm_waters, qpe_config_catalyst, use_catalyst=True
        )

    print_qpe_solvation_effect(catalyst_solvation_data, solvation_data, label="Catalyst QPE")
    return catalyst_solvation_data
