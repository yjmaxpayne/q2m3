# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
QM/MM system builder for molecular simulations.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Atom:
    """Represents an atom with position and properties."""

    symbol: str
    position: np.ndarray  # 3D coordinates in Angstrom
    charge: float = 0.0
    is_qm: bool = True


class QMMMSystem:
    """
    Build and manage QM/MM system for H3O+ in water environment.

    Handles the partitioning of QM and MM regions, and sets up
    the embedding potential for quantum calculations.
    """

    def __init__(
        self,
        qm_atoms: list[Atom],
        mm_model: str = "tip3p",
        solvation_shells: int = 1,
        num_waters: int = 8,
    ):
        """
        Initialize QM/MM system.

        Args:
            qm_atoms: List of atoms in QM region (H3O+)
            mm_model: Water model for MM region
            solvation_shells: Number of solvation shells
            num_waters: Number of water molecules in MM region
        """
        self.qm_atoms = qm_atoms
        self.mm_model = mm_model
        self.solvation_shells = solvation_shells
        self.num_waters = num_waters
        self.mm_atoms = []

        # Initialize system
        self._setup_mm_environment()

    def _setup_mm_environment(self) -> None:
        """
        Generate MM water molecules around QM region.

        Creates a simple spherical solvation shell around the QM region
        using TIP3P water geometry.
        """
        if self.num_waters == 0:
            return

        # TIP3P water geometry constants (Angstrom)
        OH_BOND_LENGTH = 0.9572
        SOLVATION_RADIUS = 3.0

        # Calculate center of QM region
        qm_center = self._calculate_qm_center()

        for i in range(self.num_waters):
            water_atoms = self._create_water_molecule(
                i, qm_center, SOLVATION_RADIUS, OH_BOND_LENGTH
            )
            self.mm_atoms.extend(water_atoms)

    def _create_water_molecule(
        self, index: int, center: np.ndarray, radius: float, bond_length: float
    ) -> list[Atom]:
        """
        Create a single water molecule at spherical position.

        Args:
            index: Water molecule index
            center: Center point for placement
            radius: Distance from center
            bond_length: O-H bond length

        Returns:
            List of three atoms (O, H, H)
        """
        # TIP3P partial charges
        OXYGEN_CHARGE = -0.834
        HYDROGEN_CHARGE = 0.417

        # Calculate spherical position
        theta = np.pi * (index + 0.5) / self.num_waters
        phi = 2 * np.pi * index / self.num_waters

        o_pos = center + radius * np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

        # Create water atoms (simplified orientation)
        o_atom = Atom("O", o_pos, charge=OXYGEN_CHARGE, is_qm=False)
        h1_pos = o_pos + np.array([bond_length, 0, 0])
        h2_pos = o_pos + np.array([-bond_length, 0, 0])

        h1_atom = Atom("H", h1_pos, charge=HYDROGEN_CHARGE, is_qm=False)
        h2_atom = Atom("H", h2_pos, charge=HYDROGEN_CHARGE, is_qm=False)

        return [o_atom, h1_atom, h2_atom]

    def _calculate_qm_center(self) -> np.ndarray:
        """Calculate geometric center of QM region."""
        positions = np.array([atom.position for atom in self.qm_atoms])
        return np.mean(positions, axis=0)

    def get_embedding_potential(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate embedding potential from MM charges.

        Returns:
            Tuple of (charges, coordinates) for MM point charges
        """
        if not self.mm_atoms:
            return np.array([]), np.array([])

        charges = np.array([atom.charge for atom in self.mm_atoms])
        coords = np.array([atom.position for atom in self.mm_atoms])

        return charges, coords

    def get_total_charge(self) -> int:
        """Get total system charge."""
        return sum(atom.charge for atom in self.qm_atoms)

    def get_qm_coords(self) -> np.ndarray:
        """Get QM atom coordinates as numpy array."""
        return np.array([atom.position for atom in self.qm_atoms])

    def to_pyscf_mol(self) -> dict[str, Any]:
        """
        Convert to PySCF molecule format.

        Returns:
            Dictionary with atom positions and basis set info
        """
        atom_str = []
        for atom in self.qm_atoms:
            atom_str.append(
                f"{atom.symbol} {atom.position[0]:.6f} {atom.position[1]:.6f} {atom.position[2]:.6f}"
            )

        return {
            "atom": "\n".join(atom_str),
            "charge": self.get_total_charge(),
            "basis": "sto-3g",
            "unit": "Angstrom",
        }
