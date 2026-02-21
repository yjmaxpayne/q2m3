# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Physical Constants and MC Solvation Parameters

Re-exports core physical constants from q2m3.constants and defines
MC-specific defaults. All values are Python floats (NOT JAX scalars)
for compatibility with both pure Python and @qjit compiled code.
"""

from q2m3.constants import (  # noqa: F401 — re-exported for package consumers
    ANGSTROM_TO_BOHR,
    BOLTZMANN_CONSTANT,
    COULOMB_CONSTANT,
    HARTREE_TO_KCAL_MOL,
    KCAL_TO_HARTREE,
    TIP3P_EPSILON_OO,
    TIP3P_HOH_ANGLE,
    TIP3P_HYDROGEN_CHARGE,
    TIP3P_OH_BOND_LENGTH,
    TIP3P_OXYGEN_CHARGE,
    TIP3P_SIGMA_OO,
)

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
