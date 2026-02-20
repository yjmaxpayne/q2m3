#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
H3O+ + TIP3P Water: Modular MC Solvation with QPE Validation

Hydronium ion (H3O+) solvation simulation using the modular mc_solvation framework.

Key differences from H2:
    - Charged system (+1)
    - Minimal (2e, 2o) active space configured for Catalyst compilation speed
    - Stronger solute-solvent interactions (ion-dipole)

Note: Physical H3O+ warrants a larger active space (e.g., 4e, 4o = 10 qubits),
but this example uses (2e, 2o) = 4 qubits for faster compilation benchmarking.
Uses energy-shifted QPE for high-precision measurements within the
limited ancilla qubit range.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import warnings

warnings.filterwarnings("ignore")

from examples.mc_solvation import (
    MoleculeConfig,
    QPEConfig,
    SolvationConfig,
    run_solvation,
)

# =============================================================================
# H3O+ Molecule Configuration
# =============================================================================

# Pyramidal geometry (C3v symmetry)
H3OP_MOLECULE = MoleculeConfig(
    name="H3O+",
    symbols=["O", "H", "H", "H"],
    coords=[
        [0.0000, 0.0000, 0.1173],
        [0.0000, 0.9572, -0.4692],
        [0.8286, -0.4786, -0.4692],
        [-0.8286, -0.4786, -0.4692],
    ],
    charge=1,  # Cation
    active_electrons=2,  # Minimal HOMO-LUMO active space for faster Catalyst compilation
    active_orbitals=2,  # 4 system qubits (Jordan-Wigner: 2 orbitals × 2 spin = 4 qubits)
    basis="sto-3g",
)


# =============================================================================
# Main
# =============================================================================


def main():
    """Run H3O+ MC solvation simulation with QPE validation."""
    # Configure simulation
    config = SolvationConfig(
        molecule=H3OP_MOLECULE,
        qpe_config=QPEConfig(
            n_estimation_wires=4,
            n_trotter_steps=10,
            n_shots=50,
            qpe_interval=10,
            target_resolution=0.003,  # ~2 kcal/mol
            energy_range=0.2,  # ±0.1 Ha for MM effects
            use_catalyst=True,
        ),
        qpe_mode="vacuum_correction",
        n_waters=10,
        n_mc_steps=100,
        temperature=300.0,
        translation_step=0.3,
        rotation_step=0.2618,
        initial_water_distance=3.5,  # Closer for ion-dipole interaction
        random_seed=42,
        verbose=True,
    )

    # Run simulation
    result = run_solvation(config, show_plots=False)

    return result


if __name__ == "__main__":
    main()
