# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Physical Constants and TIP3P Water Model Parameters

This module centralizes all physical constants used in MC solvation simulations.
All values are Python floats (NOT JAX scalars) for compatibility with both
pure Python and @qjit compiled code.
"""

# =============================================================================
# Unit Conversion Constants
# =============================================================================
HARTREE_TO_KCAL_MOL = 627.5094
KCAL_TO_HARTREE = 1.0 / HARTREE_TO_KCAL_MOL
ANGSTROM_TO_BOHR = 1.8897259886
BOLTZMANN_CONSTANT = 3.1668114e-6  # Hartree/K

# =============================================================================
# TIP3P Water Model Parameters
# Reference: Jorgensen et al., J. Chem. Phys. 79, 926 (1983)
# =============================================================================
TIP3P_OH_BOND_LENGTH = 0.9572  # Angstrom
TIP3P_HOH_ANGLE = 104.52  # degrees
TIP3P_OXYGEN_CHARGE = -0.834  # e
TIP3P_HYDROGEN_CHARGE = 0.417  # e
TIP3P_SIGMA_OO = 3.15061  # Angstrom (Lennard-Jones sigma)
TIP3P_EPSILON_OO = 0.152  # kcal/mol (Lennard-Jones epsilon)
COULOMB_CONSTANT = 332.0637  # kcal/mol * Angstrom / e^2

# =============================================================================
# Default Solvation Parameters
# =============================================================================
DEFAULT_N_WATERS = 10
DEFAULT_N_MC_STEPS = 1000
DEFAULT_TEMPERATURE = 300.0  # Kelvin
DEFAULT_TRANSLATION_STEP = 0.3  # Angstrom
DEFAULT_ROTATION_STEP = 0.2618  # ~15 degrees in radians
DEFAULT_INITIAL_WATER_DISTANCE = 4.0  # Angstrom

# =============================================================================
# Default QPE Parameters
# =============================================================================
DEFAULT_QPE_INTERVAL = 10  # Run QPE every N MC steps
DEFAULT_N_ESTIMATION_WIRES = 4
DEFAULT_N_TROTTER_STEPS = 10
DEFAULT_N_QPE_SHOTS = 50
