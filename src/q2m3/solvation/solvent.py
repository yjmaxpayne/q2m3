# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
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

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from q2m3.constants import (
    ANGSTROM_TO_BOHR,
    COULOMB_CONSTANT,
    KCAL_TO_HARTREE,
    TIP3P_EPSILON_OO,
    TIP3P_HOH_ANGLE,
    TIP3P_OH_BOND_LENGTH,
    TIP3P_SIGMA_OO,
)

_MIN_MM_DISTANCE_ANGSTROM = 1e-12

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


def _lj_pair_kcal(sigma_i: float, eps_i: float, sigma_j: float, eps_j: float, r: float) -> float:
    """Lennard-Jones energy (kcal/mol) of one atom pair with Lorentz-Berthelot rules."""
    sigma = (sigma_i + sigma_j) / 2
    epsilon = np.sqrt(eps_i * eps_j)

    sigma_r = sigma / r
    sigma_r_6 = sigma_r**6
    return 4.0 * epsilon * (sigma_r_6**2 - sigma_r_6)


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
        Total MM energy in Hartree. Exact intermolecular atom overlap is
        treated as a hard-core invalid configuration and returns +inf.
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
                    if r <= _MIN_MM_DISTANCE_ANGSTROM:
                        return float("inf")

                    e_lj_kcal += _lj_pair_kcal(sigma_i, eps_i, sigma_j, eps_j, r)

            # Coulomb interactions (all atom pairs)
            for ai in range(model_i.n_atoms):
                for aj in range(model_j.n_atoms):
                    r = np.linalg.norm(coords_j[aj] - coords_i[ai])
                    if r <= _MIN_MM_DISTANCE_ANGSTROM:
                        return float("inf")
                    e_coulomb_kcal += COULOMB_CONSTANT * charges_i[ai] * charges_j[aj] / r

    return (e_lj_kcal + e_coulomb_kcal) * KCAL_TO_HARTREE


# =============================================================================
# MM Geometry Gate and QM-MM Lennard-Jones Repulsion
# =============================================================================

_WATER_SYMBOLS = ("O", "H", "H")


@dataclass(frozen=True)
class MMGeometryReport:
    """Result of the MM geometry gate.

    Attributes:
        oh_max_dev_A: Largest absolute O-H bond deviation from TIP3P (Angstrom),
            over water molecules only. NaN when there is no water molecule.
        hoh_max_dev_deg: Largest absolute H-O-H angle deviation from TIP3P (degrees),
            over water molecules only. NaN when there is no water molecule.
        min_qm_mm_A: Measured minimum distance between any QM atom and any MM atom
            (Angstrom). NaN when a coordinate is not finite.
        passed: True only when ``violations`` is empty.
        violations: Violated rules in the fixed order ``"non_water_model"``,
            ``"oh_bond"``, ``"hoh_angle"``, ``"qm_mm_distance"``.
    """

    oh_max_dev_A: float
    hoh_max_dev_deg: float
    min_qm_mm_A: float
    passed: bool
    violations: tuple[str, ...]


def _water_deviations(coords: np.ndarray) -> tuple[float, float]:
    """Return the largest O-H deviation (Angstrom) and the H-O-H deviation (degrees)."""
    oh_1 = coords[1] - coords[0]
    oh_2 = coords[2] - coords[0]
    len_1 = np.linalg.norm(oh_1)
    len_2 = np.linalg.norm(oh_2)
    oh_dev = np.max(np.abs(np.array([len_1, len_2]) - TIP3P_OH_BOND_LENGTH))
    cos_angle = np.clip(np.dot(oh_1, oh_2) / (len_1 * len_2), -1.0, 1.0)
    hoh_dev = abs(np.degrees(np.arccos(cos_angle)) - TIP3P_HOH_ANGLE)
    return float(oh_dev), float(hoh_dev)


def check_mm_geometry(
    molecules: Sequence[SolventMolecule],
    qm_coords: np.ndarray,
    *,
    min_qm_mm_A: float = 1.5,
    oh_tol_A: float = 0.01,
    hoh_tol_deg: float = 0.5,
) -> MMGeometryReport:
    """Check MM water geometry and QM-MM separation without raising.

    A rule passes only when its measured value lies within its tolerance, so a
    NaN value (for example from a non-finite coordinate) counts as a violation.
    Molecules whose atom symbols are not ``("O", "H", "H")`` are flagged as
    ``"non_water_model"`` and are left out of the bond and angle deviations.

    Args:
        molecules: MM solvent molecules.
        qm_coords: QM atom coordinates of shape (n_qm, 3) in Angstrom.
        min_qm_mm_A: Threshold (Angstrom). The measured minimum QM-MM distance
            must be strictly greater than this value. The report field of the same
            name holds the measured distance, not this threshold.
        oh_tol_A: Largest allowed absolute O-H deviation from TIP3P (Angstrom).
        hoh_tol_deg: Largest allowed absolute H-O-H deviation from TIP3P (degrees).

    Returns:
        MMGeometryReport with the measured values and the violated rules.
    """
    oh_devs: list[float] = []
    hoh_devs: list[float] = []
    has_non_water = False
    for mol in molecules:
        if tuple(mol.model.symbols) != _WATER_SYMBOLS:
            has_non_water = True
            continue
        oh_dev, hoh_dev = _water_deviations(mol.get_atom_coords())
        oh_devs.append(oh_dev)
        hoh_devs.append(hoh_dev)

    oh_max_dev = float(np.max(oh_devs)) if oh_devs else float("nan")
    hoh_max_dev = float(np.max(hoh_devs)) if hoh_devs else float("nan")

    qm = np.asarray(qm_coords, dtype=float).reshape(-1, 3)
    mm = np.vstack([mol.get_atom_coords() for mol in molecules]) if molecules else np.empty((0, 3))
    distances = np.linalg.norm(qm[:, None, :] - mm[None, :, :], axis=-1)
    min_distance = float(np.min(distances)) if distances.size else float("inf")

    checks = (
        ("non_water_model", not has_non_water),
        ("oh_bond", oh_max_dev <= oh_tol_A),
        ("hoh_angle", hoh_max_dev <= hoh_tol_deg),
        ("qm_mm_distance", min_distance > min_qm_mm_A),
    )
    violations = tuple(name for name, ok in checks if not ok)
    return MMGeometryReport(
        oh_max_dev_A=oh_max_dev,
        hoh_max_dev_deg=hoh_max_dev,
        min_qm_mm_A=min_distance,
        passed=violations == (),
        violations=violations,
    )


# Element -> (sigma in Angstrom, epsilon in kcal/mol) for QM atoms in QM-MM LJ.
QM_MM_LJ_PARAMETERS: dict[str, tuple[float, float]] = {
    "O": (TIP3P_SIGMA_OO, TIP3P_EPSILON_OO),
    "H": (0.0, 0.0),
}


def validate_qm_mm_lj_elements(qm_symbols: Sequence[str]) -> None:
    """Check that QM-MM Lennard-Jones parameters cover the solute.

    Args:
        qm_symbols: Element symbols of the QM atoms.

    Raises:
        ValueError: If an element is not in ``QM_MM_LJ_PARAMETERS``, or if no QM
            atom has epsilon > 0 (for example H2).
    """
    unknown = sorted({s for s in qm_symbols if s not in QM_MM_LJ_PARAMETERS})
    if unknown:
        names = ", ".join(repr(s) for s in unknown)
        raise ValueError(
            f"QM-MM Lennard-Jones parameters are missing for element(s) {names}. "
            f"Known elements: {sorted(QM_MM_LJ_PARAMETERS)}."
        )
    if not any(QM_MM_LJ_PARAMETERS[s][1] > 0 for s in qm_symbols):
        raise ValueError(
            "The QM region has no atom with Lennard-Jones epsilon > 0. "
            "QM-MM Lennard-Jones repulsion cannot keep the solvent away from it."
        )


def qm_mm_lj_energy(
    qm_symbols: Sequence[str], qm_coords: np.ndarray, molecules: Sequence[SolventMolecule]
) -> float:
    """Compute the QM-MM Lennard-Jones energy with Lorentz-Berthelot rules.

    Only atom pairs with epsilon > 0 on both the QM and the MM side contribute.
    QM parameters come from ``QM_MM_LJ_PARAMETERS``.

    Args:
        qm_symbols: Element symbols of the QM atoms.
        qm_coords: QM atom coordinates of shape (n_qm, 3) in Angstrom.
        molecules: MM solvent molecules.

    Returns:
        QM-MM Lennard-Jones energy in Hartree. Exact overlap of a contributing
        atom pair returns +inf.

    Raises:
        ValueError: If ``validate_qm_mm_lj_elements`` rejects ``qm_symbols``, if
            the shape of ``qm_coords`` does not match ``qm_symbols``, or if a QM
            or MM coordinate is not finite.
    """
    validate_qm_mm_lj_elements(qm_symbols)
    qm = np.asarray(qm_coords, dtype=float)
    if qm.shape != (len(qm_symbols), 3):
        raise ValueError(
            f"qm_coords must have shape ({len(qm_symbols)}, 3), but has shape {qm.shape}."
        )
    mm_sites: list[tuple[np.ndarray, float, float]] = []
    for mol in molecules:
        coords = mol.get_atom_coords()
        mm_sites.extend(
            (coords[aj], *mol.model.get_lj_params(aj)) for aj in range(mol.model.n_atoms)
        )
    if not np.all(np.isfinite(qm)) or not all(np.all(np.isfinite(c)) for c, _, _ in mm_sites):
        raise ValueError("QM and MM coordinates must be finite.")

    e_lj_kcal = 0.0
    for symbol, qm_coord in zip(qm_symbols, qm, strict=True):
        sigma_i, eps_i = QM_MM_LJ_PARAMETERS[symbol]
        if eps_i == 0:
            continue
        for mm_coord, sigma_j, eps_j in mm_sites:
            if eps_j == 0:
                continue
            r = np.linalg.norm(mm_coord - qm_coord)
            if r <= _MIN_MM_DISTANCE_ANGSTROM:
                return float("inf")
            e_lj_kcal += _lj_pair_kcal(sigma_i, eps_i, sigma_j, eps_j, r)

    return e_lj_kcal * KCAL_TO_HARTREE


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
