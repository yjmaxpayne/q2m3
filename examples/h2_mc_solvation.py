#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
H2 + TIP3P Water: Modular MC Solvation with QPE Validation

This example demonstrates the modular mc_solvation framework for
hybrid quantum-classical QM/MM Monte Carlo solvation simulations.

Uses the refactored module structure:
    - mc_solvation.config: Configuration dataclasses
    - mc_solvation.solvent: TIP3P water model
    - mc_solvation.energy: PySCF HF + MM energy computation
    - mc_solvation.orchestrator: Main workflow with qpe_mode switching

Features:
    - QJIT-compiled MC loop with Catalyst
    - Pre-compiled QPE circuit (vacuum_correction mode)
    - Rich console output with timing statistics
    - Energy trajectory plotting
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
# H2 Molecule Configuration
# =============================================================================

H2_MOLECULE = MoleculeConfig(
    name="H2",
    symbols=["H", "H"],
    coords=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.74],  # 0.74 Angstrom bond length
    ],
    charge=0,
    active_electrons=2,
    active_orbitals=2,
    basis="sto-3g",
)


# =============================================================================
# Main
# =============================================================================


def main():
    """Run H2 MC solvation simulation with QPE validation."""
    # Configure simulation
    config = SolvationConfig(
        molecule=H2_MOLECULE,
        qpe_config=QPEConfig(
            n_estimation_wires=4,
            n_trotter_steps=10,
            n_shots=50,
            qpe_interval=10,
            target_resolution=0.003,
            energy_range=0.2,
            use_catalyst=True,
        ),
        qpe_mode="vacuum_correction",
        n_waters=10,
        n_mc_steps=100,
        temperature=300.0,
        translation_step=0.3,
        rotation_step=0.2618,  # ~15 degrees
        initial_water_distance=4.0,
        random_seed=42,
        verbose=True,
    )

    # Run simulation
    result = run_solvation(config, show_plots=False)

    return result


if __name__ == "__main__":
    main()
