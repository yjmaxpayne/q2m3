# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Explicit Solvent Models for QM/MM Simulations

This module provides data-driven solvent model definitions for use with
PySCF's qmmm module. It focuses on:
1. Solvent geometry and partial charges (for pyscf.qmmm.mm_charge)
2. Rigid body coordinate transformations
3. Classical MM energy for solvent-solvent interactions (MC acceptance)

Design Philosophy:
    - Data-driven: Solvent models defined as dataclasses, easy to extend
    - PySCF-compatible: Output coordinates in Angstrom for qmmm.mm_charge
    - Minimal: Only implement what PySCF doesn't provide directly

Usage with PySCF:
    from pyscf import qmmm

    model = TIP3P_WATER
    molecules = [SolventMolecule(model, pos, angles) for ...]
    mm_coords, mm_charges = get_mm_embedding_data(molecules)

    # Convert to Bohr for PySCF
    mm_coords_bohr = mm_coords * ANGSTROM_TO_BOHR
    mf = qmmm.mm_charge(mf, mm_coords_bohr, mm_charges)
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .constants import ANGSTROM_TO_BOHR, COULOMB_CONSTANT, KCAL_TO_HARTREE

# =============================================================================
# Solvent Model Definition (Data-Driven)
# =============================================================================


@dataclass(frozen=True)
class SolventAtom:
    """
    Single atom in a solvent model.

    Attributes:
        symbol: Atomic symbol (e.g., "O", "H")
        local_coord: Position in molecule-fixed frame (Angstrom)
        charge: Partial charge (elementary charge units)
        lj_sigma: Lennard-Jones sigma parameter (Angstrom), 0 if no LJ
        lj_epsilon: Lennard-Jones epsilon parameter (kcal/mol), 0 if no LJ
    """

    symbol: str
    local_coord: tuple[float, float, float]
    charge: float
    lj_sigma: float = 0.0
    lj_epsilon: float = 0.0

    @property
    def local_coord_array(self) -> np.ndarray:
        return np.array(self.local_coord)


@dataclass(frozen=True)
class SolventModel:
    """
    Solvent model definition containing geometry, charges, and LJ parameters.

    This is an immutable data structure that defines a solvent type.
    Actual solvent molecule instances are represented by SolventMolecule.

    Attributes:
        name: Model identifier (e.g., "TIP3P", "SPC/E")
        atoms: Tuple of SolventAtom defining the molecule structure
        reference: Literature reference for the model parameters
    """

    name: str
    atoms: tuple[SolventAtom, ...]
    reference: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def symbols(self) -> list[str]:
        return [atom.symbol for atom in self.atoms]

    @property
    def charges(self) -> np.ndarray:
        return np.array([atom.charge for atom in self.atoms])

    @property
    def local_coords(self) -> np.ndarray:
        """Local coordinates as (n_atoms, 3) array."""
        return np.array([atom.local_coord for atom in self.atoms])

    def get_lj_params(self, atom_idx: int) -> tuple[float, float]:
        """Get (sigma, epsilon) for atom at given index."""
        atom = self.atoms[atom_idx]
        return (atom.lj_sigma, atom.lj_epsilon)


# =============================================================================
# Predefined Solvent Models
# =============================================================================

# TIP3P Water Model
# Reference: Jorgensen et al., J. Chem. Phys. 79, 926 (1983)
_TIP3P_HALF_ANGLE = np.radians(104.52 / 2)
_TIP3P_OH_LENGTH = 0.9572

TIP3P_WATER = SolventModel(
    name="TIP3P",
    atoms=(
        SolventAtom("O", (0.0, 0.0, 0.0), -0.834, lj_sigma=3.15061, lj_epsilon=0.152),
        SolventAtom(
            "H",
            (
                _TIP3P_OH_LENGTH * np.cos(_TIP3P_HALF_ANGLE),
                _TIP3P_OH_LENGTH * np.sin(_TIP3P_HALF_ANGLE),
                0.0,
            ),
            0.417,
        ),
        SolventAtom(
            "H",
            (
                _TIP3P_OH_LENGTH * np.cos(_TIP3P_HALF_ANGLE),
                -_TIP3P_OH_LENGTH * np.sin(_TIP3P_HALF_ANGLE),
                0.0,
            ),
            0.417,
        ),
    ),
    reference="Jorgensen et al., J. Chem. Phys. 79, 926 (1983)",
)

# SPC/E Water Model
# Reference: Berendsen et al., J. Phys. Chem. 91, 6269 (1987)
_SPCE_HALF_ANGLE = np.radians(109.47 / 2)  # Tetrahedral angle
_SPCE_OH_LENGTH = 1.0

SPC_E_WATER = SolventModel(
    name="SPC/E",
    atoms=(
        SolventAtom("O", (0.0, 0.0, 0.0), -0.8476, lj_sigma=3.166, lj_epsilon=0.1554),
        SolventAtom(
            "H",
            (
                _SPCE_OH_LENGTH * np.cos(_SPCE_HALF_ANGLE),
                _SPCE_OH_LENGTH * np.sin(_SPCE_HALF_ANGLE),
                0.0,
            ),
            0.4238,
        ),
        SolventAtom(
            "H",
            (
                _SPCE_OH_LENGTH * np.cos(_SPCE_HALF_ANGLE),
                -_SPCE_OH_LENGTH * np.sin(_SPCE_HALF_ANGLE),
                0.0,
            ),
            0.4238,
        ),
    ),
    reference="Berendsen et al., J. Phys. Chem. 91, 6269 (1987)",
)

# Registry of available models
SOLVENT_MODELS: dict[str, SolventModel] = {
    "TIP3P": TIP3P_WATER,
    "SPC/E": SPC_E_WATER,
}


# =============================================================================
# Solvent Molecule Instance
# =============================================================================


@dataclass
class SolventMolecule:
    """
    A solvent molecule instance with position and orientation.

    Attributes:
        model: The solvent model (defines geometry and parameters)
        position: Center of mass position in Angstrom
        euler_angles: Orientation as (roll, pitch, yaw) in radians
    """

    model: SolventModel
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    euler_angles: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.euler_angles = np.asarray(self.euler_angles, dtype=float)

    def get_atom_coords(self) -> np.ndarray:
        """
        Get global coordinates of all atoms.

        Returns:
            Array of shape (n_atoms, 3) in Angstrom
        """
        R = _euler_to_rotation_matrix(self.euler_angles)
        local = self.model.local_coords
        return self.position + (R @ local.T).T

    def get_charges(self) -> np.ndarray:
        """Get partial charges for all atoms."""
        return self.model.charges

    def to_state_vector(self) -> np.ndarray:
        """Convert to state vector [x, y, z, roll, pitch, yaw]."""
        return np.concatenate([self.position, self.euler_angles])

    @classmethod
    def from_state_vector(cls, model: SolventModel, state: np.ndarray) -> "SolventMolecule":
        """Create molecule from state vector."""
        return cls(model=model, position=state[:3], euler_angles=state[3:])


def _euler_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
    """
    Create rotation matrix from Euler angles (ZYX convention).

    Args:
        euler_angles: [roll, pitch, yaw] in radians

    Returns:
        3x3 rotation matrix
    """
    roll, pitch, yaw = euler_angles

    Rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )

    return Rz @ Ry @ Rx


# =============================================================================
# Functions for PySCF Integration
# =============================================================================


def get_mm_embedding_data(
    molecules: Sequence[SolventMolecule],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract MM embedding data for pyscf.qmmm.mm_charge().

    Args:
        molecules: Sequence of SolventMolecule instances

    Returns:
        Tuple of:
            - coords: Atom coordinates in Angstrom, shape (n_total_atoms, 3)
            - charges: Partial charges, shape (n_total_atoms,)

    Usage with PySCF:
        coords, charges = get_mm_embedding_data(molecules)
        coords_bohr = coords * ANGSTROM_TO_BOHR
        mf = qmmm.mm_charge(mf, coords_bohr, charges)
    """
    all_coords = []
    all_charges = []

    for mol in molecules:
        all_coords.append(mol.get_atom_coords())
        all_charges.append(mol.get_charges())

    if not all_coords:
        return np.array([]).reshape(0, 3), np.array([])

    return np.vstack(all_coords), np.concatenate(all_charges)


def get_mm_embedding_data_bohr(
    molecules: Sequence[SolventMolecule],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get MM embedding data with coordinates in Bohr (ready for PySCF).

    This is a convenience function that converts coordinates to Bohr,
    which is the unit expected by pyscf.qmmm.mm_charge().

    Returns:
        Tuple of (coords_bohr, charges)
    """
    coords, charges = get_mm_embedding_data(molecules)
    return coords * ANGSTROM_TO_BOHR, charges


# =============================================================================
# MM Energy Calculation (for MC acceptance criterion)
# =============================================================================


def compute_mm_energy(molecules: Sequence[SolventMolecule]) -> float:
    """
    Compute classical MM energy between solvent molecules.

    This calculates the Lennard-Jones and Coulomb interactions between
    all pairs of solvent molecules. Used for the MC acceptance criterion.

    Note: This is a simple pairwise implementation. For production use
    with large systems, consider using OpenMM or other optimized libraries.

    Args:
        molecules: Sequence of SolventMolecule instances

    Returns:
        Total MM energy in Hartree
    """
    n_mol = len(molecules)
    if n_mol < 2:
        return 0.0

    # Precompute all atom coordinates and charges
    all_coords = [mol.get_atom_coords() for mol in molecules]
    all_charges = [mol.get_charges() for mol in molecules]

    e_lj_kcal = 0.0
    e_coulomb_kcal = 0.0

    for i in range(n_mol):
        model_i = molecules[i].model
        coords_i = all_coords[i]
        charges_i = all_charges[i]

        for j in range(i + 1, n_mol):
            model_j = molecules[j].model
            coords_j = all_coords[j]
            charges_j = all_charges[j]

            # LJ interactions (only between atoms with LJ parameters)
            for ai in range(model_i.n_atoms):
                sigma_i, eps_i = model_i.get_lj_params(ai)
                if eps_i == 0:
                    continue

                for aj in range(model_j.n_atoms):
                    sigma_j, eps_j = model_j.get_lj_params(aj)
                    if eps_j == 0:
                        continue

                    r = np.linalg.norm(coords_j[aj] - coords_i[ai])

                    # Lorentz-Berthelot combining rules
                    sigma = (sigma_i + sigma_j) / 2
                    epsilon = np.sqrt(eps_i * eps_j)

                    sigma_r = sigma / r
                    sigma_r_6 = sigma_r**6
                    e_lj_kcal += 4.0 * epsilon * (sigma_r_6**2 - sigma_r_6)

            # Coulomb interactions (all atom pairs)
            for ai in range(model_i.n_atoms):
                for aj in range(model_j.n_atoms):
                    r = np.linalg.norm(coords_j[aj] - coords_i[ai])
                    e_coulomb_kcal += COULOMB_CONSTANT * charges_i[ai] * charges_j[aj] / r

    return (e_lj_kcal + e_coulomb_kcal) * KCAL_TO_HARTREE


# =============================================================================
# Initialization Utilities
# =============================================================================


def initialize_solvent_ring(
    model: SolventModel,
    n_molecules: int,
    center: np.ndarray,
    radius: float,
    random_seed: int = 42,
) -> list[SolventMolecule]:
    """
    Initialize solvent molecules in a ring arrangement.

    Molecules are placed equidistant in a ring in the x-y plane,
    with random initial orientations.

    Args:
        model: Solvent model to use
        n_molecules: Number of molecules
        center: Center of the ring (typically solute center)
        radius: Ring radius in Angstrom
        random_seed: Random seed for reproducible orientations

    Returns:
        List of SolventMolecule instances
    """
    np.random.seed(random_seed)
    center = np.asarray(center)

    molecules = []
    for i in range(n_molecules):
        angle = 2 * np.pi * i / n_molecules
        position = np.array(
            [
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
                center[2],
            ]
        )
        euler_angles = np.random.uniform(-np.pi, np.pi, size=3)

        molecules.append(SolventMolecule(model, position, euler_angles))

    return molecules


def molecules_to_state_array(molecules: Sequence[SolventMolecule]) -> np.ndarray:
    """Convert list of molecules to state array of shape (n_mol, 6)."""
    return np.array([mol.to_state_vector() for mol in molecules])


def state_array_to_molecules(model: SolventModel, states: np.ndarray) -> list[SolventMolecule]:
    """Convert state array to list of molecules."""
    return [SolventMolecule.from_state_vector(model, state) for state in states]
