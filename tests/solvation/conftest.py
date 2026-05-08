# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures for q2m3.solvation tests."""

import numpy as np
import pennylane as qml
import pytest

from q2m3.solvation import TIP3P_WATER, MoleculeConfig, QPEConfig, SolvationConfig
from q2m3.solvation.circuit_builder import QPECircuitBundle
from q2m3.solvation.solvent import initialize_solvent_ring

# ============================================================================
# Molecule Data Fixtures (pure data, no solvation dependency)
# ============================================================================


@pytest.fixture
def h2_molecule_data():
    """H2 molecule raw data for testing."""
    return {
        "name": "H2",
        "symbols": ["H", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        "charge": 0,
        "active_electrons": 2,
        "active_orbitals": 2,
        "basis": "sto-3g",
    }


@pytest.fixture
def h2_hf_energy():
    """Pre-computed H2/STO-3G HF energy."""
    return -1.1175  # Hartree


# ============================================================================
# Solvation Module Fixtures
# ============================================================================


@pytest.fixture
def h2_molecule_config():
    """H2 MoleculeConfig for solvation tests."""
    return MoleculeConfig(
        name="H2",
        symbols=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        charge=0,
        active_electrons=2,
        active_orbitals=2,
    )


@pytest.fixture
def solvation_config_minimal(h2_molecule_config):
    """Minimal SolvationConfig for fast tests."""
    return SolvationConfig(
        molecule=h2_molecule_config,
        qpe_config=QPEConfig(n_estimation_wires=3, n_trotter_steps=2),
        hamiltonian_mode="fixed",
        n_waters=3,
        n_mc_steps=10,
        verbose=False,
    )


@pytest.fixture
def water_molecules_3():
    """3 TIP3P water molecules in a ring."""
    return initialize_solvent_ring(
        model=TIP3P_WATER,
        n_molecules=3,
        center=np.array([0.0, 0.0, 0.0]),
        radius=4.0,
    )


def _fake_probs_from_coeffs(
    coeffs: np.ndarray,
    base_coeffs: np.ndarray,
    n_estimation_wires: int,
) -> np.ndarray:
    """Return deterministic QPE-like probabilities without compiling a circuit."""
    n_bins = 2**n_estimation_wires
    probs = np.zeros(n_bins, dtype=float)
    if n_bins == 1:
        probs[0] = 1.0
        return probs

    delta = float(np.asarray(coeffs, dtype=float)[0] - base_coeffs[0])
    frac = min(0.9, abs(delta) * 50.0)
    probs[0] = 1.0 - frac
    probs[1] = frac
    return probs


@pytest.fixture
def fast_qpe_bundle_factory():
    """Build QPECircuitBundle test doubles for solvation integration tests."""

    def _factory(config: SolvationConfig, qm_coords: np.ndarray, hf_energy: float, **_kwargs):
        qpe = config.qpe_config
        n_estimation_wires = max(1, qpe.n_estimation_wires)
        n_system_qubits = 2 * config.molecule.active_orbitals
        base_coeffs = np.array([hf_energy, 0.05, -0.05, 0.02, -0.02], dtype=np.float64)
        ops = [
            qml.Identity(wires=list(range(n_system_qubits))),
            qml.PauliZ(0),
            qml.PauliZ(1),
            qml.PauliZ(2),
            qml.PauliZ(3),
        ][: len(base_coeffs)]
        is_fixed = config.hamiltonian_mode == "fixed"
        measurement_mode = "probs" if qpe.n_shots == 0 else "shots"

        def compiled_circuit(coeffs_arr=None):
            coeffs = base_coeffs if coeffs_arr is None else np.asarray(coeffs_arr, dtype=float)
            if measurement_mode == "shots":
                return np.zeros((qpe.n_shots, n_estimation_wires), dtype=np.int64)
            return _fake_probs_from_coeffs(coeffs, base_coeffs, n_estimation_wires)

        return QPECircuitBundle(
            compiled_circuit=compiled_circuit,
            base_coeffs=base_coeffs,
            ops=ops,
            base_time=1.0,
            op_index_map={"identity_idx": 0, "z_wire_idx": {0: 1, 1: 2, 2: 3, 3: 4}},
            energy_shift=hf_energy,
            n_estimation_wires=n_estimation_wires,
            n_system_qubits=n_system_qubits,
            active_orbitals=config.molecule.active_orbitals,
            n_trotter_steps=qpe.n_trotter_steps,
            measurement_mode=measurement_mode,
            is_fixed_circuit=is_fixed,
        )

    return _factory


@pytest.fixture
def fast_solvation_qpe(monkeypatch, fast_qpe_bundle_factory):
    """Replace repeated compile/PySCF runtime work with deterministic doubles."""
    from q2m3.solvation import circuit_builder, ir_cache, orchestrator
    from q2m3.solvation.energy import StepResult

    def fake_vacuum_cache(config: SolvationConfig):
        return {
            "h_core_vac": np.zeros((2, 2)),
            "mo_coeff": np.eye(2),
            "mo_coeff_active": np.eye(2),
            "energy_nuc_vac": 0.0,
            "e_vacuum": -1.1175,
            "active_idx": [0, 1],
        }

    def fake_step_result(config: SolvationConfig, solvent_states: np.ndarray, *, qpe: bool = True):
        states = np.asarray(solvent_states, dtype=float)
        displacement = float(states[0, 0] - config.initial_water_distance)
        dynamic_shift = -0.01 * displacement if config.hamiltonian_mode == "dynamic" else 0.0
        e_qpe = -1.1175 + dynamic_shift if qpe else float("nan")
        return StepResult(
            e_qpe=e_qpe,
            e_mm_sol_sol=-0.001 * config.n_waters,
            e_hf_ref=-1.1175 + dynamic_shift,
            callback_time=0.0,
            qpe_time=0.0 if not qpe else 0.001,
        )

    def fake_create_step_callback(_bundle, config: SolvationConfig, _vacuum_cache):
        def _step(solvent_states, _qm_coords_flat):
            return fake_step_result(config, solvent_states, qpe=True)

        return _step

    def fake_create_hf_corrected_step_callback(
        config: SolvationConfig,
        _vacuum_cache,
        qm_coords,
        e_vacuum,
        circuit_bundle=None,
    ):
        state = {
            "step_counter": 0,
            "bundle": (
                circuit_bundle
                if circuit_bundle is not None
                else fast_qpe_bundle_factory(config, qm_coords, e_vacuum)
            ),
        }

        def _step(solvent_states, _qm_coords_flat):
            current_step = state["step_counter"]
            state["step_counter"] += 1
            return fake_step_result(
                config,
                solvent_states,
                qpe=current_step % config.qpe_config.qpe_interval == 0,
            )

        _step._state = state
        return _step

    monkeypatch.setattr(circuit_builder, "build_qpe_circuit", fast_qpe_bundle_factory)
    monkeypatch.setattr(orchestrator, "build_qpe_circuit", fast_qpe_bundle_factory)
    monkeypatch.setattr(orchestrator, "compute_hf_energy_vacuum", lambda _mol: -1.1175)
    monkeypatch.setattr(orchestrator, "precompute_vacuum_cache", fake_vacuum_cache)
    monkeypatch.setattr(orchestrator, "compute_mulliken_charges", lambda _mol: {"H0": 0.0})
    monkeypatch.setattr(orchestrator, "create_step_callback", fake_create_step_callback)
    monkeypatch.setattr(
        orchestrator,
        "create_hf_corrected_step_callback",
        fake_create_hf_corrected_step_callback,
    )
    monkeypatch.setattr(
        ir_cache,
        "resolve_compiled_circuit",
        lambda _config, bundle: (bundle, {"is_cache_hit": False, "test_double": True}),
    )


@pytest.fixture
def fast_ir_cache_runtime(monkeypatch, fast_qpe_bundle_factory, fast_solvation_qpe):
    """Exercise cache hit/miss semantics without launching Phase-A compilation."""
    from q2m3.solvation import ir_cache

    def fake_resolve(config: SolvationConfig, bundle: QPECircuitBundle):
        cache_path = ir_cache.cache_path_for_config(config)
        hit = ir_cache.is_cache_available(config)
        if not hit and config.ir_cache_enabled:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text("fake LLVM IR for test cache")
        return bundle, {"is_cache_hit": hit, "test_double": True}

    monkeypatch.setattr(ir_cache, "resolve_compiled_circuit", fake_resolve)
