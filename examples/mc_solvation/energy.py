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
       - Diagonal MM embedding: includes diagonal one-electron MM corrections
         (delta_h1e[p,p]); neglects off-diagonal h1e terms and two-electron modifications
       - Slow: Requires dynamic Hamiltonian construction

Both strategies use pyscf.qmmm.mm_charge() for electrostatic embedding.
"""

from collections.abc import Callable, Sequence

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


def compute_mulliken_charges(
    molecule: MoleculeConfig,
    solvent_states: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute Mulliken charges for vacuum or solvated geometry.

    Args:
        molecule: QM region molecular configuration
        solvent_states: Optional solvent state array (n_mol, 6); None for vacuum

    Returns:
        Dictionary {atom_label: mulliken_charge}, e.g. {"O0": -0.45, "H1": +0.22}
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

    if solvent_states is not None:
        from .solvent import (
            SOLVENT_MODELS,
            get_mm_embedding_data,
            state_array_to_molecules,
        )

        model = SOLVENT_MODELS["TIP3P"]
        solvent_molecules = state_array_to_molecules(model, np.asarray(solvent_states))
        mm_coords, mm_charges = get_mm_embedding_data(solvent_molecules)
        mm_coords_bohr = mm_coords * ANGSTROM_TO_BOHR
        mf = qmmm.mm_charge(mf, mm_coords_bohr, mm_charges)

    mf.run()
    _, charges = mf.mulliken_pop(verbose=0)

    return {f"{sym}{i}": float(charges[i]) for i, sym in enumerate(molecule.symbols)}


# =============================================================================
# Hamiltonian Coefficient Decomposition (for mm_embedded mode)
# =============================================================================


def precompute_vacuum_cache(config: SolvationConfig) -> dict:
    """
    Pre-compute and cache vacuum SCF data (runs once at initialization).

    Eliminates redundant vacuum SCF calls in MC loop callbacks by caching
    all vacuum-derived quantities needed for MM coefficient patching.

    Args:
        config: Solvation configuration with molecule definition

    Returns:
        Dictionary containing cached vacuum SCF data:
            - h_core_vac: Vacuum core Hamiltonian (AO basis)
            - mo_coeff: Full MO coefficient matrix
            - mo_coeff_active: Active-space MO coefficients (AO x active_orbitals)
            - energy_nuc_vac: Vacuum nuclear repulsion energy
            - e_vacuum: Total vacuum HF energy
            - active_idx: List of active orbital indices
    """
    symbols = config.molecule.symbols
    coords = config.molecule.coords
    charge = config.molecule.charge
    basis = config.molecule.basis
    active_electrons = config.molecule.active_electrons
    active_orbitals = config.molecule.active_orbitals

    # Build PySCF molecule
    atom_str = "; ".join(f"{s} {c[0]} {c[1]} {c[2]}" for s, c in zip(symbols, coords, strict=True))
    mol = gto.M(atom=atom_str, basis=basis, charge=charge, unit="Angstrom")

    # Run vacuum RHF SCF
    mf_vac = scf.RHF(mol)
    mf_vac.verbose = 0
    mf_vac.run()

    # Active space indices
    n_electrons = mol.nelectron
    n_core = (n_electrons - active_electrons) // 2
    active_idx = list(range(n_core, n_core + active_orbitals))

    return {
        "h_core_vac": mf_vac.get_hcore(),
        "mo_coeff": mf_vac.mo_coeff,
        "mo_coeff_active": mf_vac.mo_coeff[:, active_idx],
        "energy_nuc_vac": mol.energy_nuc(),
        "e_vacuum": float(mf_vac.e_tot),
        "active_idx": active_idx,
    }


def create_coeff_callback(
    config: SolvationConfig,
    base_coeffs: np.ndarray,
    op_index_map: dict,
    active_orbitals: int,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Factory: create a callback that computes updated Hamiltonian coefficients
    for a given solvent configuration.

    The callback runs PySCF vacuum + solvated SCF to compute delta_h1e_mo
    and delta_nuc, then patches the base_coeffs (which include vacuum H +
    energy shift) with MM corrections on Identity and Z(wire) terms.

    Physics:
        H_mm = H_vacuum + delta_nuc * I + sum_p delta_h1e[p,p] * n_p
        where n_p = (I - Z_p) / 2 for each spin orbital p.
        Expanding: Identity += delta_nuc + sum_p delta_h1e[p,p]
                   Z(2p+s)  -= delta_h1e[p,p] / 2    (for s in {0,1})

    Args:
        config: Solvation configuration
        base_coeffs: Coefficient array including vacuum H + energy shift
        op_index_map: From build_operator_index_map()
        active_orbitals: Number of active spatial orbitals

    Returns:
        Callable: (solvent_states, qm_coords_flat) -> np.ndarray[n_terms]
    """
    from .solvent import SOLVENT_MODELS, get_mm_embedding_data, state_array_to_molecules

    symbols = config.molecule.symbols
    charge = config.molecule.charge
    basis = config.molecule.basis
    active_electrons = config.molecule.active_electrons

    identity_idx = op_index_map["identity_idx"]
    z_wire_idx = op_index_map["z_wire_idx"]

    base_coeffs_copy = np.array(base_coeffs, dtype=np.float64)

    def compute_mm_coefficients(
        solvent_states: np.ndarray, qm_coords_flat: np.ndarray
    ) -> np.ndarray:
        """Compute Hamiltonian coefficients with MM embedding for current solvent config."""
        # Reconstruct solvent molecules from state array
        model = SOLVENT_MODELS["TIP3P"]
        solvent_molecules = state_array_to_molecules(model, np.asarray(solvent_states))
        mm_coords, mm_charges = get_mm_embedding_data(solvent_molecules)

        if len(mm_charges) == 0:
            return base_coeffs_copy.copy()

        # Build PySCF molecule
        coords = np.asarray(qm_coords_flat).reshape(-1, 3)
        atom_str = "; ".join(
            f"{s} {c[0]} {c[1]} {c[2]}" for s, c in zip(symbols, coords, strict=True)
        )
        mol = gto.M(atom=atom_str, basis=basis, charge=charge, unit="Angstrom")

        # Vacuum SCF (for MO coefficients)
        mf_vac = scf.RHF(mol)
        mf_vac.verbose = 0
        mf_vac.run()

        # Solvated SCF (for MM-modified integrals)
        mf_sol = scf.RHF(mol)
        mf_sol.verbose = 0
        mm_coords_bohr = mm_coords * ANGSTROM_TO_BOHR
        mf_sol = qmmm.mm_charge(mf_sol, mm_coords_bohr, mm_charges)
        mf_sol.run()

        # Active space selection
        n_electrons = mol.nelectron
        n_core = (n_electrons - active_electrons) // 2
        active_idx = list(range(n_core, n_core + active_orbitals))

        # MM effect on single-electron integrals (MO basis)
        mo_coeff_active = mf_vac.mo_coeff[:, active_idx]
        delta_h1e_ao = mf_sol.get_hcore() - mf_vac.get_hcore()
        delta_h1e_mo = mo_coeff_active.T @ delta_h1e_ao @ mo_coeff_active

        # MM effect on nuclear energy
        delta_nuc = mf_sol.energy_nuc() - mol.energy_nuc()

        # Patch coefficients
        new_coeffs = base_coeffs_copy.copy()

        # Identity: += delta_nuc + sum_p delta_h1e_mo[p,p]
        identity_correction = delta_nuc
        for p in range(active_orbitals):
            identity_correction += delta_h1e_mo[p, p]
        new_coeffs[identity_idx] += identity_correction

        # Z(wire): -= delta_h1e_mo[p,p] / 2  for each spin orbital
        for p in range(active_orbitals):
            for spin in [0, 1]:
                wire = 2 * p + spin
                if wire in z_wire_idx:
                    new_coeffs[z_wire_idx[wire]] -= delta_h1e_mo[p, p] / 2

        return new_coeffs

    return compute_mm_coefficients


def create_fused_qpe_callback(
    config: SolvationConfig,
    base_coeffs: np.ndarray,
    op_index_map: dict,
    active_orbitals: int,
    vacuum_cache: dict,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Factory: fused callback for qpe_driven mode.

    Single solvated SCF per call (using cached vacuum data).
    Returns flat array: [coeffs(n_terms), e_mm_sol_sol, e_hf_solvated]

    Args:
        config: Solvation configuration
        base_coeffs: Coefficient array including vacuum H + energy shift
        op_index_map: From build_operator_index_map()
        active_orbitals: Number of active spatial orbitals
        vacuum_cache: Pre-computed vacuum data from precompute_vacuum_cache()

    Returns:
        Callable: (solvent_states, qm_coords_flat) -> np.ndarray[n_terms + 2]
    """
    from .solvent import SOLVENT_MODELS, get_mm_embedding_data, state_array_to_molecules

    symbols = config.molecule.symbols
    charge = config.molecule.charge
    basis = config.molecule.basis

    identity_idx = op_index_map["identity_idx"]
    z_wire_idx = op_index_map["z_wire_idx"]

    base_coeffs_copy = np.array(base_coeffs, dtype=np.float64)

    # Unpack vacuum cache
    h_core_vac = vacuum_cache["h_core_vac"]
    mo_coeff_active = vacuum_cache["mo_coeff_active"]
    energy_nuc_vac = vacuum_cache["energy_nuc_vac"]

    def fused_callback(solvent_states: np.ndarray, qm_coords_flat: np.ndarray) -> np.ndarray:
        """Compute fused Hamiltonian coefficients + energies for qpe_driven mode."""
        # Reconstruct solvent molecules from state array
        model = SOLVENT_MODELS["TIP3P"]
        solvent_molecules = state_array_to_molecules(model, np.asarray(solvent_states))
        mm_coords, mm_charges = get_mm_embedding_data(solvent_molecules)

        if len(mm_charges) == 0:
            e_mm = compute_mm_energy(solvent_molecules)
            return np.concatenate([base_coeffs_copy.copy(), [e_mm, 0.0]])

        # Build PySCF molecule
        coords = np.asarray(qm_coords_flat).reshape(-1, 3)
        atom_str = "; ".join(
            f"{s} {c[0]} {c[1]} {c[2]}" for s, c in zip(symbols, coords, strict=True)
        )
        mol = gto.M(atom=atom_str, basis=basis, charge=charge, unit="Angstrom")

        # Solvated SCF only (vacuum data from cache)
        mf_sol = scf.RHF(mol)
        mf_sol.verbose = 0
        mm_coords_bohr = mm_coords * ANGSTROM_TO_BOHR
        mf_sol = qmmm.mm_charge(mf_sol, mm_coords_bohr, mm_charges)
        mf_sol.run()

        # MM effect on single-electron integrals (MO basis) using cached vacuum data
        delta_h1e_ao = mf_sol.get_hcore() - h_core_vac
        delta_h1e_mo = mo_coeff_active.T @ delta_h1e_ao @ mo_coeff_active

        # MM effect on nuclear energy using cached vacuum data
        delta_nuc = mf_sol.energy_nuc() - energy_nuc_vac

        # Patch coefficients (same logic as create_coeff_callback)
        new_coeffs = base_coeffs_copy.copy()

        # Identity: += delta_nuc + sum_p delta_h1e_mo[p,p]
        identity_correction = delta_nuc
        for p in range(active_orbitals):
            identity_correction += delta_h1e_mo[p, p]
        new_coeffs[identity_idx] += identity_correction

        # Z(wire): -= delta_h1e_mo[p,p] / 2  for each spin orbital
        for p in range(active_orbitals):
            for spin in [0, 1]:
                wire = 2 * p + spin
                if wire in z_wire_idx:
                    new_coeffs[z_wire_idx[wire]] -= delta_h1e_mo[p, p] / 2

        # Append additional energies for qpe_driven mode
        e_mm_sol_sol = compute_mm_energy(solvent_molecules)
        e_hf_solvated = float(mf_sol.e_tot)

        return np.concatenate([new_coeffs, [e_mm_sol_sol, e_hf_solvated]])

    return fused_callback


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


def create_qpe_step_callback(
    fused_callback: Callable[[np.ndarray, np.ndarray], np.ndarray],
    compiled_circuit: Callable,
    base_time: float,
    n_estimation_wires: int,
    energy_shift: float,
    n_terms: int,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Create a consolidated callback for one QPE-driven MC step.

    Wraps fused_callback + compiled_circuit + phase_extraction into a single
    callable that is opaque to Catalyst @qjit via pure_callback. This prevents
    the QPE circuit IR from being inlined into the MC loop MLIR, reducing
    compilation memory from ~16G to <1G.

    The callback executes:
        1. fused_callback → coeffs, e_mm, e_hf  (PySCF + MM)
        2. compiled_circuit(coeffs) → probs      (pre-compiled QPE)
        3. phase extraction → e_qpe              (expected value)

    Args:
        fused_callback: From create_fused_qpe_callback, returns [coeffs(n_terms), e_mm, e_hf]
        compiled_circuit: Pre-compiled @qjit QPE circuit accepting coeffs array
        base_time: Base evolution time for QPE phase-to-energy conversion
        n_estimation_wires: Number of QPE estimation qubits
        energy_shift: Energy shift applied to Hamiltonian (for un-shifting)
        n_terms: Number of Hamiltonian coefficient terms

    Returns:
        Callable: (solvent_states, qm_coords_flat) -> np.ndarray[5]
            Output: [e_qpe, e_mm, e_hf, cb_elapsed, qpe_elapsed]
    """
    import time as _time

    import jax.numpy as jnp

    n_bins = 2**n_estimation_wires

    def compute_step(solvent_states: np.ndarray, qm_coords_flat: np.ndarray) -> np.ndarray:
        # Step 1: fused callback (PySCF + MM energy)
        cb_start = _time.perf_counter()
        fused_result = fused_callback(solvent_states, qm_coords_flat)
        cb_elapsed = _time.perf_counter() - cb_start

        coeffs = fused_result[:n_terms]
        e_mm = float(fused_result[n_terms])
        e_hf = float(fused_result[n_terms + 1])

        # Step 2: QPE circuit (already @qjit compiled, just call it)
        qpe_start = _time.perf_counter()
        probs = compiled_circuit(jnp.array(coeffs))
        qpe_elapsed = _time.perf_counter() - qpe_start

        # Step 3: phase extraction (expected value mode)
        probs_np = np.asarray(probs)
        expected_bin = sum(float(probs_np[k]) * k for k in range(n_bins))
        phase = expected_bin / n_bins
        delta_e = -2.0 * np.pi * phase / base_time
        e_qpe = delta_e + energy_shift

        return np.array([e_qpe, e_mm, e_hf, cb_elapsed, qpe_elapsed], dtype=np.float64)

    return compute_step


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
