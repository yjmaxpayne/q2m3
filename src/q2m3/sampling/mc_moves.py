# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Monte Carlo move generator for water molecules.

This module provides random translation and rotation moves for
Monte Carlo sampling of water configurations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .water_molecule import WaterMolecule

# Default MC step sizes
DEFAULT_TRANSLATION_STEP = 0.3  # Angstrom
DEFAULT_ROTATION_STEP = np.radians(15.0)  # radians (~15 degrees)


@dataclass
class MCMoveGenerator:
    """
    Generator for Monte Carlo moves (translation and rotation).

    Attributes:
        translation_step: Maximum translation in each direction (Angstrom)
        rotation_step: Maximum rotation in each Euler angle (radians)
    """

    translation_step: float = DEFAULT_TRANSLATION_STEP
    rotation_step: float = DEFAULT_ROTATION_STEP

    def propose_translation(self, water: WaterMolecule) -> WaterMolecule:
        """
        Propose a random translation move.

        Args:
            water: Original water molecule

        Returns:
            New water molecule with translated position
        """
        # Random displacement in [-step, +step] for each dimension
        displacement = np.random.uniform(-self.translation_step, self.translation_step, size=3)
        new_position = water.position + displacement

        return WaterMolecule(
            position=new_position,
            euler_angles=water.euler_angles.copy(),
        )

    def propose_rotation(self, water: WaterMolecule) -> WaterMolecule:
        """
        Propose a random rotation move.

        Args:
            water: Original water molecule

        Returns:
            New water molecule with rotated orientation
        """
        # Random rotation in [-step, +step] for each Euler angle
        delta_angles = np.random.uniform(-self.rotation_step, self.rotation_step, size=3)
        new_euler_angles = water.euler_angles + delta_angles

        return WaterMolecule(
            position=water.position.copy(),
            euler_angles=new_euler_angles,
        )

    def propose_move(self, water: WaterMolecule) -> WaterMolecule:
        """
        Propose either a translation or rotation move with equal probability.

        Args:
            water: Original water molecule

        Returns:
            New water molecule with proposed move applied
        """
        if np.random.random() < 0.5:
            return self.propose_translation(water)
        else:
            return self.propose_rotation(water)

    def propose_move_for_system(
        self, waters: list[WaterMolecule]
    ) -> tuple[list[WaterMolecule], int]:
        """
        Propose a move for one randomly selected water in the system.

        Args:
            waters: List of water molecules

        Returns:
            Tuple of (new_waters, moved_index):
                - new_waters: Copy of waters list with one water moved
                - moved_index: Index of the water that was moved
        """
        n_waters = len(waters)
        if n_waters == 0:
            raise ValueError("Cannot propose move for empty water list")

        # Select random water
        idx = np.random.randint(n_waters)

        # Create new list with copies
        new_waters = [w.copy() for w in waters]

        # Apply move to selected water
        new_waters[idx] = self.propose_move(waters[idx])

        return new_waters, idx
