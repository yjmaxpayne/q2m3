# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Molecular system configuration.

Provides a generic MoleculeConfig dataclass for configuring QM region
parameters in hybrid quantum-classical simulations.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MoleculeConfig:
    """
    Generic molecular system configuration.

    This class supports any molecular system by specifying atomic symbols,
    coordinates, charge, and active space parameters.

    Attributes:
        name: Human-readable identifier for the molecule
        symbols: List of atomic symbols (e.g., ["O", "H", "H"])
        coords: Atomic coordinates in Angstrom, shape (n_atoms, 3)
        charge: Total molecular charge (0 for neutral, +1 for cation, etc.)
        active_electrons: Number of active electrons for QPE active space
        active_orbitals: Number of active orbitals for QPE active space
        basis: Basis set name (default: "sto-3g")

    Example:
        # Water molecule
        water = MoleculeConfig(
            name="H2O",
            symbols=["O", "H", "H"],
            coords=[[0.0, 0.0, 0.117], [-0.756, 0.0, -0.469], [0.756, 0.0, -0.469]],
            charge=0,
            active_electrons=4,
            active_orbitals=4,
        )
    """

    name: str
    symbols: list[str]
    coords: list[list[float]]
    charge: int
    active_electrons: int
    active_orbitals: int
    basis: str = "sto-3g"

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return len(self.symbols)

    @property
    def coords_array(self) -> np.ndarray:
        """Coordinates as numpy array."""
        return np.array(self.coords)

    @property
    def center(self) -> np.ndarray:
        """Geometric center of the molecule."""
        return np.mean(self.coords_array, axis=0)

    def validate(self) -> None:
        """Validate configuration consistency."""
        if len(self.symbols) != len(self.coords):
            raise ValueError(
                f"Number of symbols ({len(self.symbols)}) must match "
                f"number of coordinate sets ({len(self.coords)})"
            )
        for i, coord in enumerate(self.coords):
            if len(coord) != 3:
                raise ValueError(f"Coordinate {i} must have 3 components, got {len(coord)}")
