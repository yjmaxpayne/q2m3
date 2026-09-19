"""Reproducible molecular inputs; no solver or audit-tool dependencies."""

from __future__ import annotations

import numpy as np

# The repository's original neutral NH2-CH2-COOH Z matrix, in Angstrom/degrees.
GLYCINE_ZMATRIX = """
C
C 1 1.52
N 1 1.47 2 110.0
O 2 1.21 1 125.0 3 180.0
O 2 1.35 1 111.0 3 0.0
H 1 1.09 2 108.0 3 120.0
H 1 1.09 2 108.0 3 -120.0
H 3 1.01 1 110.0 2 60.0
H 3 1.01 1 110.0 2 -60.0
H 5 0.97 2 107.0 1 0.0
"""


def glycine_geometry() -> tuple[list[str], np.ndarray]:
    """Return the existing glycine geometry as Cartesian Angstrom coordinates."""
    from pyscf import gto

    mol = gto.M(atom=GLYCINE_ZMATRIX, basis="sto-3g", charge=0, verbose=0)
    return [mol.atom_symbol(i) for i in range(mol.natm)], mol.atom_coords(unit="Angstrom")


def fixed_waters() -> tuple[np.ndarray, np.ndarray]:
    """Return six neutral TIP3P point charges and artificial fixed positions (Å).

    Each water has O–H=0.9572 Å and H–O–H=104.52 degrees. These are manually
    placed external charges, not an equilibrated solvent configuration.
    """
    angle = np.deg2rad(104.52)
    water = np.array(
        [[0, 0, 0], [0.9572, 0, 0], [0.9572 * np.cos(angle), 0.9572 * np.sin(angle), 0]]
    )
    coordinates = np.vstack((water + [4.0, 2.0, 1.0], -water + [-3.5, -2.0, 1.0]))
    return np.tile([-0.834, 0.417, 0.417], 2), coordinates
