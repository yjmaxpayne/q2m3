# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Energy Computation Module for QM/MM MC Solvation

This module provides energy computation functions using PySCF for QM calculations
and the solvent module for MM interactions. All functions are designed to be
compatible with Catalyst @qjit via pure_callback.

Energy Decomposition Strategies:
    1. vacuum_correction: E_total = E_QPE(vacuum) + ΔE_MM(HF)
       - Fast: QPE circuit can be pre-compiled
       - Approximate: Ignores correlation-polarization coupling

    2. mm_embedded: E_total = E_QPE(with_MM_embedding)
       - Rigorous: MM charges in QPE Hamiltonian
       - Slow: Requires dynamic Hamiltonian construction

Both strategies use pyscf.qmmm.mm_charge() for electrostatic embedding.
"""

from typing import Sequence

import numpy as np
from pyscf import gto, qmmm, scf

from .config import MoleculeConfig, SolvationConfig
from .constants import ANGSTROM_TO_BOHR
from .solvent import SolventMolecule, compute_mm_energy, get_mm_embedding_data


def compute_hf_energy_vacuum(molecule: MoleculeConfig) -> float:
    """
    Compute vacuum HF energy for the QM region.

    Args:
        molecule: Molecular system configuration

    Returns:
        Vacuum HF energy in Hartree
    """
    atom_str = "; ".join(
        f"{s} {c[0]} {c[1]} {c[2]}" for s, c in zip(molecule.symbols, molecule.coords, strict=True)
    )

    mol = gto.M(
        atom=atom_str,
        basis=molecule.basis,
        charge=molecule.charge,
        unit="Angstrom",
    )

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    return float(mf.e_tot)


def compute_hf_energy_solvated(
    molecule: MoleculeConfig,
    solvent_molecules: Sequence[SolventMolecule],
) -> float:
    """
    Compute solvated HF energy with MM electrostatic embedding.

    Uses pyscf.qmmm.mm_charge() to embed MM point charges in the QM calculation.

    Args:
        molecule: QM region molecular configuration
        solvent_molecules: List of solvent molecules (MM region)

    Returns:
        Solvated HF energy in Hartree
    """
    atom_str = "; ".join(
        f"{s} {c[0]} {c[1]} {c[2]}" for s, c in zip(molecule.symbols, molecule.coords, strict=True)
    )

    mol = gto.M(
        atom=atom_str,
        basis=molecule.basis,
        charge=molecule.charge,
        unit="Angstrom",
    )

    mf = scf.RHF(mol)
    mf.verbose = 0

    # Add MM embedding if solvent molecules present
    if solvent_molecules:
        mm_coords, mm_charges = get_mm_embedding_data(solvent_molecules)
        mm_coords_bohr = mm_coords * ANGSTROM_TO_BOHR
        mf = qmmm.mm_charge(mf, mm_coords_bohr, mm_charges)

    mf.run()
    return float(mf.e_tot)


def compute_total_energy(
    molecule: MoleculeConfig,
    solvent_molecules: Sequence[SolventMolecule],
) -> float:
    """
    Compute total QM/MM energy: E_HF(solvated) + E_MM(solvent-solvent).

    This is the energy used for MC acceptance criterion.

    Args:
        molecule: QM region molecular configuration
        solvent_molecules: List of solvent molecules

    Returns:
        Total energy in Hartree
    """
    e_qm = compute_hf_energy_solvated(molecule, solvent_molecules)
    e_mm = compute_mm_energy(solvent_molecules)
    return e_qm + e_mm


def compute_mm_correction(
    molecule: MoleculeConfig,
    solvent_molecules: Sequence[SolventMolecule],
    e_vacuum: float,
) -> float:
    """
    Compute HF-level MM correction: ΔE_MM = E_HF(solvated) - E_HF(vacuum).

    This correction is used in the vacuum_correction QPE mode to account
    for the electrostatic effect of MM charges on the QM region.

    Args:
        molecule: QM region molecular configuration
        solvent_molecules: List of solvent molecules
        e_vacuum: Pre-computed vacuum HF energy

    Returns:
        MM correction in Hartree
    """
    e_solvated = compute_hf_energy_solvated(molecule, solvent_molecules)
    return e_solvated - e_vacuum


# =============================================================================
# Callback-Compatible Functions for @qjit
# =============================================================================
# These functions accept raw numpy arrays and are suitable for pure_callback.


def _compute_hf_energy_vacuum_impl(
    symbols: list[str],
    coords_flat: np.ndarray,
    charge: int,
    basis: str,
) -> float:
    """
    Pure callback implementation for vacuum HF energy.

    Args:
        symbols: Atomic symbols
        coords_flat: Flattened coordinates array
        charge: Molecular charge
        basis: Basis set name

    Returns:
        Vacuum HF energy in Hartree
    """
    coords = coords_flat.reshape(-1, 3)
    atom_str = "; ".join(f"{s} {c[0]} {c[1]} {c[2]}" for s, c in zip(symbols, coords, strict=True))

    mol = gto.M(atom=atom_str, basis=basis, charge=charge, unit="Angstrom")
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()

    return np.float64(mf.e_tot)


def _compute_hf_energy_solvated_impl(
    symbols: list[str],
    coords_flat: np.ndarray,
    charge: int,
    basis: str,
    mm_coords_flat: np.ndarray,
    mm_charges: np.ndarray,
) -> float:
    """
    Pure callback implementation for solvated HF energy.

    Args:
        symbols: Atomic symbols for QM region
        coords_flat: Flattened QM coordinates
        charge: QM region charge
        basis: Basis set name
        mm_coords_flat: Flattened MM coordinates in Angstrom
        mm_charges: MM partial charges

    Returns:
        Solvated HF energy in Hartree
    """
    coords = coords_flat.reshape(-1, 3)
    atom_str = "; ".join(f"{s} {c[0]} {c[1]} {c[2]}" for s, c in zip(symbols, coords, strict=True))

    mol = gto.M(atom=atom_str, basis=basis, charge=charge, unit="Angstrom")
    mf = scf.RHF(mol)
    mf.verbose = 0

    if len(mm_charges) > 0:
        mm_coords = mm_coords_flat.reshape(-1, 3)
        mm_coords_bohr = mm_coords * ANGSTROM_TO_BOHR
        mf = qmmm.mm_charge(mf, mm_coords_bohr, mm_charges)

    mf.run()
    return np.float64(mf.e_tot)


def _compute_total_energy_impl(
    symbols: list[str],
    qm_coords_flat: np.ndarray,
    charge: int,
    basis: str,
    solvent_states: np.ndarray,
    solvent_model_name: str,
) -> float:
    """
    Pure callback implementation for total QM/MM energy.

    This function reconstructs SolventMolecule objects from state arrays
    and computes the total energy. Designed for use with pure_callback.

    Args:
        symbols: QM atomic symbols
        qm_coords_flat: Flattened QM coordinates
        charge: QM charge
        basis: Basis set
        solvent_states: Solvent state array of shape (n_mol, 6)
        solvent_model_name: Name of solvent model (e.g., "TIP3P")

    Returns:
        Total energy in Hartree
    """
    from .solvent import SOLVENT_MODELS, state_array_to_molecules

    model = SOLVENT_MODELS[solvent_model_name]
    solvent_molecules = state_array_to_molecules(model, solvent_states)

    # Get MM embedding data
    mm_coords, mm_charges = get_mm_embedding_data(solvent_molecules)

    # Compute HF energy with embedding
    e_qm = _compute_hf_energy_solvated_impl(
        symbols,
        qm_coords_flat,
        charge,
        basis,
        mm_coords.flatten(),
        mm_charges,
    )

    # Compute MM energy
    e_mm = compute_mm_energy(solvent_molecules)

    return np.float64(e_qm + e_mm)


def create_energy_callback(config: SolvationConfig):
    """
    Create a closure for energy computation that captures configuration.

    This is used to create pure_callback compatible functions that have
    access to the molecular configuration.

    Args:
        config: Solvation configuration

    Returns:
        Function that takes (qm_coords_flat, solvent_states) and returns energy
    """
    symbols = config.molecule.symbols
    charge = config.molecule.charge
    basis = config.molecule.basis
    solvent_model_name = "TIP3P"  # TODO: Get from config

    def energy_callback(qm_coords_flat: np.ndarray, solvent_states: np.ndarray) -> float:
        return _compute_total_energy_impl(
            symbols,
            qm_coords_flat,
            charge,
            basis,
            solvent_states,
            solvent_model_name,
        )

    return energy_callback
