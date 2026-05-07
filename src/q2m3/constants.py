# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Centralized physical constants for quantum chemistry and QM/MM simulations.

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
COULOMB_CONSTANT = 332.0637  # kcal/mol * Angstrom / e^2

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

# =============================================================================
# Accuracy Thresholds
# =============================================================================
CHEMICAL_ACCURACY_HA = 0.0016  # ~1 kcal/mol
RELAXED_ACCURACY_HA = 0.016  # ~10 kcal/mol
