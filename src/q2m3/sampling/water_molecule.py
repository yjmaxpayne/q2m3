# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Water molecule representation for Monte Carlo sampling.

This module provides a WaterMolecule class that represents a TIP3P water
molecule with position and orientation (Euler angles).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# TIP3P water model parameters
TIP3P_OH_BOND_LENGTH = 0.9572  # Angstrom
TIP3P_HOH_ANGLE = 104.52  # degrees
TIP3P_OXYGEN_CHARGE = -0.834  # e
TIP3P_HYDROGEN_CHARGE = 0.417  # e


def _rotation_matrix_from_euler(euler_angles: np.ndarray) -> np.ndarray:
    """
    Create rotation matrix from Euler angles (ZYX convention).

    Args:
        euler_angles: [roll, pitch, yaw] in radians
            - roll: rotation around x-axis
            - pitch: rotation around y-axis
            - yaw: rotation around z-axis

    Returns:
        3x3 rotation matrix
    """
    roll, pitch, yaw = euler_angles

    # Rotation around z-axis (yaw)
    Rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )

    # Rotation around y-axis (pitch)
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    # Rotation around x-axis (roll)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )

    # Combined rotation: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx


@dataclass
class WaterMolecule:
    """
    TIP3P water molecule with position and orientation.

    The water molecule is defined by:
    - position: 3D coordinates of oxygen atom (center of molecule)
    - euler_angles: [roll, pitch, yaw] rotation in radians

    Default orientation has hydrogens in the xy-plane, symmetric about x-axis.

    Attributes:
        position: Oxygen atom position in Angstrom
        euler_angles: Euler angles [roll, pitch, yaw] in radians
    """

    position: np.ndarray
    euler_angles: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        """Validate inputs and ensure numpy arrays."""
        self.position = np.asarray(self.position, dtype=float)
        self.euler_angles = np.asarray(self.euler_angles, dtype=float)

        if self.position.shape != (3,):
            raise ValueError(f"position must be shape (3,), got {self.position.shape}")
        if self.euler_angles.shape != (3,):
            raise ValueError(f"euler_angles must be shape (3,), got {self.euler_angles.shape}")

    @property
    def oxygen_position(self) -> np.ndarray:
        """Return oxygen atom position (same as molecule position)."""
        return self.position.copy()

    def get_atom_coords(self) -> np.ndarray:
        """
        Get coordinates of all three atoms (O, H1, H2).

        Returns:
            Array of shape (3, 3) with [O, H1, H2] coordinates in Angstrom.
        """
        # Default H positions in molecule frame (xy-plane)
        half_angle_rad = np.radians(TIP3P_HOH_ANGLE / 2)

        # H1 at positive angle from x-axis
        h1_local = np.array(
            [
                TIP3P_OH_BOND_LENGTH * np.cos(half_angle_rad),
                TIP3P_OH_BOND_LENGTH * np.sin(half_angle_rad),
                0.0,
            ]
        )

        # H2 at negative angle from x-axis
        h2_local = np.array(
            [
                TIP3P_OH_BOND_LENGTH * np.cos(half_angle_rad),
                -TIP3P_OH_BOND_LENGTH * np.sin(half_angle_rad),
                0.0,
            ]
        )

        # Apply rotation
        R = _rotation_matrix_from_euler(self.euler_angles)
        h1_rotated = R @ h1_local
        h2_rotated = R @ h2_local

        # Translate to world coordinates
        o_pos = self.position
        h1_pos = self.position + h1_rotated
        h2_pos = self.position + h2_rotated

        return np.array([o_pos, h1_pos, h2_pos])

    def get_charges(self) -> np.ndarray:
        """
        Get TIP3P partial charges for all atoms.

        Returns:
            Array of shape (3,) with [O_charge, H1_charge, H2_charge].
        """
        return np.array([TIP3P_OXYGEN_CHARGE, TIP3P_HYDROGEN_CHARGE, TIP3P_HYDROGEN_CHARGE])

    def copy(self) -> WaterMolecule:
        """Create an independent copy of this water molecule."""
        return WaterMolecule(
            position=self.position.copy(),
            euler_angles=self.euler_angles.copy(),
        )
