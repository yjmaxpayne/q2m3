# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
TIP3P force field for water-water interactions.

This module provides energy calculations for TIP3P water molecules,
including Lennard-Jones and Coulomb interactions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .water_molecule import WaterMolecule

# Physical constants
COULOMB_CONSTANT = 332.0637  # kcal/mol * Angstrom / e^2
KCAL_TO_HARTREE = 1.0 / 627.5094  # 1 kcal/mol = 0.00159 Hartree

# TIP3P Lennard-Jones parameters (O-O interaction only)
# sigma = 3.15061 Angstrom, epsilon = 0.6364 kJ/mol = 0.152 kcal/mol
TIP3P_SIGMA_OO = 3.15061  # Angstrom
TIP3P_EPSILON_OO = 0.152  # kcal/mol


class TIP3PForceField:
    """
    TIP3P water model force field for MM energy calculations.

    Computes:
    - Lennard-Jones energy (O-O interactions only)
    - Coulomb energy (all atom pairs)

    All energies are returned in Hartree for compatibility with QM calculations.
    """

    def __init__(
        self,
        sigma_oo: float = TIP3P_SIGMA_OO,
        epsilon_oo: float = TIP3P_EPSILON_OO,
    ):
        """
        Initialize TIP3P force field.

        Args:
            sigma_oo: LJ sigma parameter for O-O interaction (Angstrom)
            epsilon_oo: LJ epsilon parameter for O-O interaction (kcal/mol)
        """
        self.sigma_oo = sigma_oo
        self.epsilon_oo = epsilon_oo

    def compute_lj_energy(self, waters: list[WaterMolecule]) -> float:
        """
        Compute Lennard-Jones energy between all water pairs.

        Only O-O interactions are included (TIP3P convention).

        Args:
            waters: List of WaterMolecule objects

        Returns:
            Total LJ energy in Hartree
        """
        n_waters = len(waters)
        if n_waters < 2:
            return 0.0

        e_lj_kcal = 0.0

        for i in range(n_waters):
            for j in range(i + 1, n_waters):
                # O-O distance
                o_i = waters[i].oxygen_position
                o_j = waters[j].oxygen_position
                r_oo = np.linalg.norm(o_j - o_i)

                # LJ potential: 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
                sigma_r = self.sigma_oo / r_oo
                sigma_r_6 = sigma_r**6
                sigma_r_12 = sigma_r_6**2
                e_lj_kcal += 4.0 * self.epsilon_oo * (sigma_r_12 - sigma_r_6)

        return e_lj_kcal * KCAL_TO_HARTREE

    def compute_coulomb_energy(self, waters: list[WaterMolecule]) -> float:
        """
        Compute Coulomb energy between all water pairs.

        All atom-atom interactions between different molecules are included.

        Args:
            waters: List of WaterMolecule objects

        Returns:
            Total Coulomb energy in Hartree
        """
        n_waters = len(waters)
        if n_waters < 2:
            return 0.0

        e_coulomb_kcal = 0.0

        for i in range(n_waters):
            for j in range(i + 1, n_waters):
                coords_i = waters[i].get_atom_coords()  # (3, 3)
                charges_i = waters[i].get_charges()  # (3,)
                coords_j = waters[j].get_atom_coords()
                charges_j = waters[j].get_charges()

                # All atom pairs between molecules i and j
                for ai in range(3):
                    for aj in range(3):
                        r_ij = np.linalg.norm(coords_j[aj] - coords_i[ai])
                        q_i = charges_i[ai]
                        q_j = charges_j[aj]
                        e_coulomb_kcal += COULOMB_CONSTANT * q_i * q_j / r_ij

        return e_coulomb_kcal * KCAL_TO_HARTREE

    def compute_mm_energy(self, waters: list[WaterMolecule]) -> float:
        """
        Compute total MM energy (LJ + Coulomb) between all water pairs.

        Args:
            waters: List of WaterMolecule objects

        Returns:
            Total MM energy in Hartree
        """
        return self.compute_lj_energy(waters) + self.compute_coulomb_energy(waters)
