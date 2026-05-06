"""
Energy computation module for MC solvation simulations.

Provides vacuum caching, three-mode coefficient callbacks, and step callback
factories for QPE-driven Monte Carlo solvation energy evaluation.

Three Hamiltonian modes:
    - hf_corrected: HF energy for MC acceptance, interval-based QPE diagnostics
    - fixed: Compile-once vacuum coefficients, QPE every step
    - dynamic: Per-step MM-embedded Hamiltonian, QPE every step

All step callbacks are pure Python (not @qjit); they invoke pre-compiled
@qjit circuits internally (ADR-003).
"""

from __future__ import annotations

import time as _time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from pyscf import gto, qmmm, scf

from q2m3.constants import ANGSTROM_TO_BOHR

from .config import MoleculeConfig, SolvationConfig
from .phase_extraction import extract_energy_from_probs, extract_energy_from_shots
from .solvent import SolventMolecule, compute_mm_energy, get_mm_embedding_data

_MIN_QMMM_DISTANCE_ANGSTROM = 1e-12


@dataclass(frozen=True)
class StepResult:
    """Result from one MC step's energy computation."""

    e_qpe: float  # QPE energy estimate (NaN if not evaluated)
    e_mm_sol_sol: float  # Solvent-solvent MM energy
    e_hf_ref: float  # HF reference energy (diagnostic)
    callback_time: float  # PySCF computation time (seconds)
    qpe_time: float  # QPE circuit execution time (seconds)


# =============================================================================
# Vacuum Cache
# =============================================================================


def precompute_vacuum_cache(config: SolvationConfig) -> dict:
    """
    Run vacuum SCF once and cache results for MC loop.

    Args:
        config: Solvation configuration with molecule definition

    Returns:
        Dictionary with cached vacuum SCF data:
            h_core_vac, mo_coeff, mo_coeff_active,
            energy_nuc_vac, e_vacuum, active_idx
    """
    mol_cfg = config.molecule
    atom_str = _build_atom_string(mol_cfg)

    mol = gto.M(atom=atom_str, basis=mol_cfg.basis, charge=mol_cfg.charge, unit="Angstrom")

    mf_vac = scf.RHF(mol)
    mf_vac.verbose = 0
    mf_vac.run()

    n_electrons = mol.nelectron
    n_core = (n_electrons - mol_cfg.active_electrons) // 2
    active_idx = list(range(n_core, n_core + mol_cfg.active_orbitals))

    return {
        "h_core_vac": mf_vac.get_hcore(),
        "mo_coeff": mf_vac.mo_coeff,
        "mo_coeff_active": mf_vac.mo_coeff[:, active_idx],
        "energy_nuc_vac": mol.energy_nuc(),
        "e_vacuum": float(mf_vac.e_tot),
        "active_idx": active_idx,
    }


# =============================================================================
# Coefficient Callbacks
# =============================================================================


def create_coefficient_callback(
    config: SolvationConfig,
    circuit_bundle: QPECircuitBundle,  # noqa: F821 — forward ref to avoid circular import
    vacuum_cache: dict,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Factory: create coefficient update callback for the configured mode.

    For "hf_corrected" / "fixed": returns base_coeffs unchanged.
    For "dynamic": runs solvated SCF, computes delta, patches coefficients.

    Args:
        config: Solvation configuration
        circuit_bundle: QPE circuit bundle with base_coeffs and op_index_map
        vacuum_cache: Pre-computed vacuum data

    Returns:
        Callable: (solvent_states, qm_coords_flat) -> np.ndarray[n_terms]
    """
    mode = config.hamiltonian_mode
    base_coeffs = np.array(circuit_bundle.base_coeffs, dtype=np.float64)

    if mode in ("hf_corrected", "fixed"):

        def _fixed_callback(solvent_states: np.ndarray, qm_coords_flat: np.ndarray) -> np.ndarray:
            return base_coeffs.copy()

        return _fixed_callback

    # Dynamic mode: patch coefficients with MM delta
    return _build_dynamic_callback(config, circuit_bundle, vacuum_cache)


def _build_dynamic_callback(
    config: SolvationConfig,
    circuit_bundle: QPECircuitBundle,  # noqa: F821
    vacuum_cache: dict,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Build dynamic mode coefficient callback using cached vacuum data."""
    from .solvent import SOLVENT_MODELS, get_mm_embedding_data, state_array_to_molecules

    mol_cfg = config.molecule
    base_coeffs = np.array(circuit_bundle.base_coeffs, dtype=np.float64)
    identity_idx = circuit_bundle.op_index_map["identity_idx"]
    z_wire_idx = circuit_bundle.op_index_map["z_wire_idx"]
    active_orbitals = circuit_bundle.active_orbitals

    h_core_vac = vacuum_cache["h_core_vac"]
    mo_coeff_active = vacuum_cache["mo_coeff_active"]
    energy_nuc_vac = vacuum_cache["energy_nuc_vac"]

    # Mutable state: stores e_hf_solvated as byproduct for step callback
    _state = {"e_hf_solvated": vacuum_cache["e_vacuum"]}

    def _dynamic_callback(solvent_states: np.ndarray, qm_coords_flat: np.ndarray) -> np.ndarray:
        model = SOLVENT_MODELS["TIP3P"]
        solvent_molecules = state_array_to_molecules(model, np.asarray(solvent_states))
        mm_coords, mm_charges = get_mm_embedding_data(solvent_molecules)

        if len(mm_charges) == 0:
            _state["e_hf_solvated"] = vacuum_cache["e_vacuum"]
            return base_coeffs.copy()

        # Build PySCF molecule from current coordinates
        coords = np.asarray(qm_coords_flat).reshape(-1, 3)
        atom_str = "; ".join(
            f"{s} {c[0]} {c[1]} {c[2]}" for s, c in zip(mol_cfg.symbols, coords, strict=True)
        )
        mol = gto.M(atom=atom_str, basis=mol_cfg.basis, charge=mol_cfg.charge, unit="Angstrom")

        # Solvated SCF only (vacuum data from cache)
        mf_sol = scf.RHF(mol)
        mf_sol.verbose = 0
        mm_coords_bohr = mm_coords * ANGSTROM_TO_BOHR
        mf_sol = qmmm.mm_charge(mf_sol, mm_coords_bohr, mm_charges)
        mf_sol.run()

        _state["e_hf_solvated"] = float(mf_sol.e_tot)

        # MM effect on single-electron integrals (MO basis)
        delta_h1e_ao = mf_sol.get_hcore() - h_core_vac
        delta_h1e_mo = mo_coeff_active.T @ delta_h1e_ao @ mo_coeff_active

        # MM effect on nuclear energy
        delta_nuc = mf_sol.energy_nuc() - energy_nuc_vac

        # Patch coefficients
        new_coeffs = base_coeffs.copy()

        # Identity: += delta_nuc + sum_p delta_h1e_mo[p,p]
        identity_correction = delta_nuc
        for p in range(active_orbitals):
            identity_correction += delta_h1e_mo[p, p]
        new_coeffs[identity_idx] += identity_correction

        # Z(wire): -= delta_h1e_mo[p,p] / 2  for each spin orbital
        for p in range(active_orbitals):
            for spin in (0, 1):
                wire = 2 * p + spin
                if wire in z_wire_idx:
                    new_coeffs[z_wire_idx[wire]] -= delta_h1e_mo[p, p] / 2

        return new_coeffs

    # Expose internal state for step callback consumption
    _dynamic_callback._state = _state
    return _dynamic_callback


# =============================================================================
# Step Callbacks
# =============================================================================


def create_step_callback(
    circuit_bundle: QPECircuitBundle,  # noqa: F821
    config: SolvationConfig,
    vacuum_cache: dict,
) -> Callable[[np.ndarray, np.ndarray], StepResult]:
    """
    Factory: create per-step energy computation callback (fixed/dynamic modes).

    Internal pipeline:
    1. coefficient_callback → new_coeffs
    2. compiled_circuit(jnp.array(new_coeffs)) → measurement_result
    3. phase_extraction(measurement_result) → e_qpe
    4. compute_mm_energy(solvents) → e_mm_sol_sol
    5. Return StepResult

    Args:
        circuit_bundle: QPE circuit bundle with compiled circuit
        config: Solvation configuration
        vacuum_cache: Pre-computed vacuum data

    Returns:
        Callable: (solvent_states, qm_coords_flat) -> StepResult
    """
    import jax.numpy as jnp

    from .solvent import SOLVENT_MODELS, state_array_to_molecules

    coeff_callback = create_coefficient_callback(config, circuit_bundle, vacuum_cache)
    compiled_circuit = circuit_bundle.compiled_circuit
    mode = config.hamiltonian_mode

    base_time = circuit_bundle.base_time
    energy_shift = circuit_bundle.energy_shift
    n_estimation_wires = circuit_bundle.n_estimation_wires
    measurement_mode = circuit_bundle.measurement_mode

    is_fixed = circuit_bundle.is_fixed_circuit

    def _step(solvent_states: np.ndarray, qm_coords_flat: np.ndarray) -> StepResult:
        if is_fixed:
            # Fixed mode: zero-arg circuit, no coefficient update needed
            cb_elapsed = 0.0
            qpe_start = _time.perf_counter()
            measurement_result = compiled_circuit()
            qpe_elapsed = _time.perf_counter() - qpe_start
        else:
            # Dynamic mode: compute new coefficients, pass as JAX array
            cb_start = _time.perf_counter()
            new_coeffs = coeff_callback(solvent_states, qm_coords_flat)
            cb_elapsed = _time.perf_counter() - cb_start

            qpe_start = _time.perf_counter()
            measurement_result = compiled_circuit(jnp.array(new_coeffs))
            qpe_elapsed = _time.perf_counter() - qpe_start

        # Step 3: Phase extraction
        result_np = np.asarray(measurement_result)
        if measurement_mode == "probs":
            e_qpe = extract_energy_from_probs(
                result_np, base_time, energy_shift, n_estimation_wires
            )
        else:
            e_qpe = extract_energy_from_shots(
                result_np, base_time, energy_shift, n_estimation_wires
            )

        # Step 4: MM energy
        model = SOLVENT_MODELS["TIP3P"]
        solvent_molecules = state_array_to_molecules(model, np.asarray(solvent_states))
        e_mm_sol_sol = compute_mm_energy(solvent_molecules)

        # Step 5: e_hf_ref depends on mode
        if mode == "dynamic" and hasattr(coeff_callback, "_state"):
            e_hf_ref = coeff_callback._state["e_hf_solvated"]
        else:
            e_hf_ref = vacuum_cache["e_vacuum"]

        return StepResult(
            e_qpe=e_qpe,
            e_mm_sol_sol=e_mm_sol_sol,
            e_hf_ref=e_hf_ref,
            callback_time=cb_elapsed,
            qpe_time=qpe_elapsed,
        )

    return _step


def create_hf_corrected_step_callback(
    config: SolvationConfig,
    vacuum_cache: dict,
    qm_coords: np.ndarray,
    e_vacuum: float,
    circuit_bundle: QPECircuitBundle | None = None,  # noqa: F821
) -> Callable[[np.ndarray, np.ndarray], StepResult]:
    """
    Factory: hf_corrected mode step callback with deferred compilation.

    Every step: PySCF QMMM → e_hf_ref; compute_mm_energy → e_mm.
    Every qpe_interval steps: additionally run QPE circuit for diagnostics.

    If circuit_bundle is None, QPE circuit is lazily built on the first
    QPE step (deferred compilation — avoids startup compilation cost).

    Non-QPE steps return StepResult(e_qpe=NaN, qpe_time=0.0).

    Args:
        config: Solvation configuration
        vacuum_cache: Pre-computed vacuum data
        qm_coords: QM coordinates array for circuit building
        e_vacuum: Vacuum HF energy for energy shift
        circuit_bundle: Optional pre-built circuit bundle (None = deferred)

    Returns:
        Callable: (solvent_states, qm_coords_flat) -> StepResult
    """
    import jax.numpy as jnp

    from .solvent import SOLVENT_MODELS, state_array_to_molecules

    qpe_interval = config.qpe_config.qpe_interval
    mol_cfg = config.molecule

    # Mutable state: step counter + lazy bundle
    _state = {"step_counter": 0, "bundle": circuit_bundle}

    def _step(solvent_states: np.ndarray, qm_coords_flat: np.ndarray) -> StepResult:
        current_step = _state["step_counter"]
        _state["step_counter"] += 1
        is_qpe_step = current_step % qpe_interval == 0

        # Always: compute HF solvated energy and MM energy
        cb_start = _time.perf_counter()
        model = SOLVENT_MODELS["TIP3P"]
        solvent_molecules = state_array_to_molecules(model, np.asarray(solvent_states))
        e_mm_sol_sol = compute_mm_energy(solvent_molecules)

        e_hf_ref = compute_hf_energy_solvated(mol_cfg, solvent_molecules)
        cb_elapsed = _time.perf_counter() - cb_start

        if not is_qpe_step:
            return StepResult(
                e_qpe=float("nan"),
                e_mm_sol_sol=e_mm_sol_sol,
                e_hf_ref=e_hf_ref,
                callback_time=cb_elapsed,
                qpe_time=0.0,
            )

        # Lazy circuit build on first QPE step
        bundle = _state["bundle"]
        if bundle is None:
            from .circuit_builder import build_qpe_circuit

            qm_coords_2d = np.asarray(qm_coords).reshape(-1, 3)
            bundle = build_qpe_circuit(config, qm_coords_2d, e_vacuum)
            if config.ir_cache_enabled:
                from .ir_cache import resolve_compiled_circuit

                bundle, _cache_stats = resolve_compiled_circuit(config, bundle)
                _state["cache_stats"] = _cache_stats
            _state["bundle"] = bundle

        # QPE step: run circuit with vacuum coefficients (dynamic-style)
        qpe_start = _time.perf_counter()
        measurement_result = bundle.compiled_circuit(jnp.array(bundle.base_coeffs))
        qpe_elapsed = _time.perf_counter() - qpe_start

        result_np = np.asarray(measurement_result)
        if bundle.measurement_mode == "probs":
            e_corr = extract_energy_from_probs(
                result_np, bundle.base_time, bundle.energy_shift, bundle.n_estimation_wires
            )
        else:
            e_corr = extract_energy_from_shots(
                result_np, bundle.base_time, bundle.energy_shift, bundle.n_estimation_wires
            )

        # QPE energy: E_corr + e_vacuum + delta_e_mm
        delta_e_mm = e_hf_ref - vacuum_cache["e_vacuum"]
        e_qpe = e_corr + vacuum_cache["e_vacuum"] + delta_e_mm

        return StepResult(
            e_qpe=e_qpe,
            e_mm_sol_sol=e_mm_sol_sol,
            e_hf_ref=e_hf_ref,
            callback_time=cb_elapsed,
            qpe_time=qpe_elapsed,
        )

    _step._state = _state  # Expose internal state for orchestrator post-MC inspection
    return _step


# =============================================================================
# Helper PySCF Functions
# =============================================================================


def compute_hf_energy_vacuum(molecule: MoleculeConfig) -> float:
    """
    Compute vacuum HF energy for the QM region.

    Args:
        molecule: Molecular system configuration

    Returns:
        Vacuum HF energy in Hartree
    """
    atom_str = _build_atom_string(molecule)
    mol = gto.M(atom=atom_str, basis=molecule.basis, charge=molecule.charge, unit="Angstrom")

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

    Args:
        molecule: QM region molecular configuration
        solvent_molecules: List of solvent molecules (MM region)

    Returns:
        Solvated HF energy in Hartree
    """
    atom_str = _build_atom_string(molecule)
    mol = gto.M(atom=atom_str, basis=molecule.basis, charge=molecule.charge, unit="Angstrom")

    mf = scf.RHF(mol)
    mf.verbose = 0

    if solvent_molecules:
        mm_coords, mm_charges = get_mm_embedding_data(solvent_molecules)
        if _has_qmmm_point_charge_overlap(mm_coords, np.asarray(molecule.coords, dtype=float)):
            return float("inf")
        mm_coords_bohr = mm_coords * ANGSTROM_TO_BOHR
        mf = qmmm.mm_charge(mf, mm_coords_bohr, mm_charges)

    mf.run()
    return float(mf.e_tot)


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
    atom_str = _build_atom_string(molecule)
    mol = gto.M(atom=atom_str, basis=molecule.basis, charge=molecule.charge, unit="Angstrom")

    mf = scf.RHF(mol)
    mf.verbose = 0

    if solvent_states is not None:
        from .solvent import SOLVENT_MODELS, get_mm_embedding_data, state_array_to_molecules

        model = SOLVENT_MODELS["TIP3P"]
        solvent_molecules = state_array_to_molecules(model, np.asarray(solvent_states))
        mm_coords, mm_charges = get_mm_embedding_data(solvent_molecules)
        mm_coords_bohr = mm_coords * ANGSTROM_TO_BOHR
        mf = qmmm.mm_charge(mf, mm_coords_bohr, mm_charges)

    mf.run()
    _, charges = mf.mulliken_pop(verbose=0)

    return {f"{sym}{i}": float(charges[i]) for i, sym in enumerate(molecule.symbols)}


# =============================================================================
# Internal Helpers
# =============================================================================


def _build_atom_string(molecule: MoleculeConfig) -> str:
    """Build PySCF atom specification string."""
    return "; ".join(
        f"{s} {c[0]} {c[1]} {c[2]}" for s, c in zip(molecule.symbols, molecule.coords, strict=True)
    )


def _has_qmmm_point_charge_overlap(mm_coords: np.ndarray, qm_coords: np.ndarray) -> bool:
    """Return True when an MM point charge is exactly on a QM atom."""
    if mm_coords.size == 0 or qm_coords.size == 0:
        return False

    deltas = mm_coords[:, np.newaxis, :] - qm_coords[np.newaxis, :, :]
    distances_sq = np.einsum("...i,...i->...", deltas, deltas)
    return bool(np.any(distances_sq <= _MIN_QMMM_DISTANCE_ANGSTROM**2))
