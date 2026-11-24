# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
QM/MM system builder for molecular simulations.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from q2m3.constants import TIP3P_HOH_ANGLE, TIP3P_OH_BOND_LENGTH


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

    This class partitions the QM and MM regions and prepares
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

        # Solvation shell radius (Angstrom)
        SOLVATION_RADIUS = 3.0

        # Calculate center of QM region
        qm_center = self._calculate_qm_center()

        for i in range(self.num_waters):
            water_atoms = self._create_water_molecule(
                i, qm_center, SOLVATION_RADIUS, TIP3P_OH_BOND_LENGTH
            )
            self.mm_atoms.extend(water_atoms)

    def _create_water_molecule(
        self, index: int, center: np.ndarray, radius: float, bond_length: float
    ) -> list[Atom]:
        """
        Create a single TIP3P water molecule at spherical position.

        The oxygen sits on a sphere of ``radius`` around ``center``. Both hydrogens
        are ``bond_length`` from the oxygen, with the TIP3P H-O-H angle. The H-O-H
        bisector points radially outward, so both hydrogens face away from
        ``center``. The H-H axis lies along ``cross(r_hat, z_hat)``, where
        ``r_hat`` is the unit vector from ``center`` to the oxygen. When ``r_hat``
        is parallel to z, the x axis replaces z.

        Args:
            index: Water molecule index
            center: Center point for placement (Angstrom)
            radius: Distance from center to the oxygen (Angstrom)
            bond_length: O-H bond length (Angstrom)

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

        # Hydrogens: bisector along r_hat (outward), spread along p_hat
        r_hat = (o_pos - center) / np.linalg.norm(o_pos - center)
        perp = np.cross(r_hat, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(perp) < 1e-8:
            perp = np.cross(r_hat, np.array([1.0, 0.0, 0.0]))
        p_hat = perp / np.linalg.norm(perp)
        half = np.radians(TIP3P_HOH_ANGLE / 2)

        o_atom = Atom("O", o_pos, charge=OXYGEN_CHARGE, is_qm=False)
        h1_pos = o_pos + bond_length * (np.cos(half) * r_hat + np.sin(half) * p_hat)
        h2_pos = o_pos + bond_length * (np.cos(half) * r_hat - np.sin(half) * p_hat)

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
