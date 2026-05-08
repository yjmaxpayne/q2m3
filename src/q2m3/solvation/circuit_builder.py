# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
QPE circuit builder for MC solvation simulations.

Builds parameterized @qjit-compiled QPE circuits that accept Hamiltonian
coefficients as runtime arguments. Compile once, call with different
coefficients each MC step — avoids 67s recompilation per step.

Key Catalyst constraints (POC-validated):
- Use X gates instead of qml.BasisState (Catalyst @qjit + ctrl() bug)
- TrotterProduct requires check_hermitian=False (JAX tracer incompatible)
- MAX_TROTTER_STEPS_RUNTIME caps Trotter steps to avoid MLIR OOM
- MSB-first qubit ordering (matching PennyLane QFT convention)
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pennylane as qml
from catalyst import qjit

from q2m3.core.device_utils import select_device as _select_device
from q2m3.core.hamiltonian_utils import (  # re-export for convenience
    build_operator_index_map,
    decompose_hamiltonian,
)
from q2m3.core.qpe import QPEEngine
from q2m3.interfaces import PySCFPennyLaneConverter

from .config import SolvationConfig

# Re-export helpers so callers can import from circuit_builder
__all__ = [
    "QPECircuitBundle",
    "MAX_TROTTER_STEPS_RUNTIME",
    "build_qpe_circuit",
    "decompose_hamiltonian",
    "build_operator_index_map",
]

# OOM protection: cap Trotter steps for runtime-parameterized circuits.
# Catalyst MLIR cannot constant-fold JAX-traced coefficients,
# so symbolic IR scales as n_estimation × n_trotter × n_hamiltonian_terms.
# Benchmark: n_est=2, n_trotter=3 → ~24s (H2), ~96s (H3O+).
MAX_TROTTER_STEPS_RUNTIME: int = 20


@dataclass(frozen=True)
class QPECircuitBundle:
    """All artifacts from QPE circuit compilation.

    Attributes:
        compiled_circuit: @qjit function: () → result (fixed) or (coeffs_arr) → result (dynamic)
        base_coeffs: Initial vacuum + shift coefficients
        ops: PennyLane operators (compile-time constants)
        base_time: Evolution time for phase-to-energy conversion
        op_index_map: {"identity_idx": int, "z_wire_idx": {wire: idx}}
        energy_shift: E_HF shift applied to Hamiltonian
        n_estimation_wires: Number of QPE estimation qubits
        n_system_qubits: Number of system qubits (= 2 * active_orbitals)
        active_orbitals: Number of active spatial orbitals
        n_trotter_steps: Actual Trotter steps (may be capped)
        measurement_mode: "probs" or "shots"
        is_fixed_circuit: True = zero-arg circuit (fixed mode); False = takes coeffs_arr
        embedding_mode: MM embedding mode used to build the fixed operator support
    """

    compiled_circuit: Callable
    base_coeffs: np.ndarray
    ops: list
    base_time: float
    op_index_map: dict
    energy_shift: float
    n_estimation_wires: int
    n_system_qubits: int
    active_orbitals: int
    n_trotter_steps: int
    measurement_mode: str
    is_fixed_circuit: bool = False
    embedding_mode: str = "diagonal"


def build_qpe_circuit(
    config: SolvationConfig,
    qm_coords: np.ndarray,
    hf_energy: float,
    *,
    _keep_intermediate: bool = False,
) -> QPECircuitBundle:
    """Build parameterized QPE circuit accepting runtime coefficients.

    The circuit builder constructs the base Hamiltonian, decomposes it into
    coefficients and PennyLane operators, applies the HF energy shift, derives
    shifted QPE parameters, caps runtime Trotter depth when needed, and returns
    a ``QPECircuitBundle``. In diagonal mode the base Hamiltonian is vacuum
    operator support; in ``full_oneelectron`` mode it is a fixed MM operator.

    Args:
        config: Solvation configuration
        qm_coords: QM region coordinates in Angstrom, shape (n_atoms, 3)
        hf_energy: HF energy for energy shift and base_time calculation

    Returns:
        QPECircuitBundle with compiled circuit and metadata
    """
    config.validate()
    mol = config.molecule
    qpe = config.qpe_config

    # Step 1: Build base Hamiltonian
    converter = PySCFPennyLaneConverter(basis=mol.basis, mapping="jordan_wigner")
    if config.embedding_mode == "full_oneelectron":
        mm_coords, mm_charges = _initial_solvent_mm_embedding(config, qm_coords)
        H, n_qubits, hf_state = converter.pyscf_to_pennylane_hamiltonian_with_mm(
            symbols=mol.symbols,
            coords=qm_coords,
            charge=mol.charge,
            mm_charges=mm_charges,
            mm_coords=mm_coords,
            active_electrons=mol.active_electrons,
            active_orbitals=mol.active_orbitals,
            embedding_mode="full_oneelectron",
        )
    else:
        H, n_qubits, hf_state = converter.pyscf_to_pennylane_hamiltonian(
            symbols=mol.symbols,
            coords=qm_coords,
            charge=mol.charge,
            active_electrons=mol.active_electrons,
            active_orbitals=mol.active_orbitals,
        )

    # Step 2: Decompose H into coefficients and operators
    coeffs, ops = decompose_hamiltonian(H)

    # Step 3: Build operator index map (may extend with missing Z terms)
    op_index_map, coeffs, ops = build_operator_index_map(ops, n_qubits, coeffs)

    # Step 4: Compute shifted QPE parameters
    params = QPEEngine.compute_shifted_qpe_params(
        target_resolution=qpe.target_resolution,
        energy_range=qpe.energy_range,
    )
    base_time = params["base_time"]
    energy_shift = hf_energy

    n_estimation_wires = qpe.n_estimation_wires
    if n_estimation_wires <= 0:
        n_estimation_wires = params["n_estimation_wires"]

    # Apply energy shift to Identity coefficient
    coeffs[op_index_map["identity_idx"]] -= energy_shift
    base_coeffs = np.array(coeffs, dtype=np.float64)

    # Step 5: Cap Trotter steps for OOM protection (runtime-parameterized circuits only)
    is_fixed = config.hamiltonian_mode == "fixed"
    n_trotter = qpe.n_trotter_steps
    if not is_fixed and n_trotter > MAX_TROTTER_STEPS_RUNTIME:
        warnings.warn(
            f"n_trotter_steps={n_trotter} exceeds runtime circuit ceiling "
            f"({MAX_TROTTER_STEPS_RUNTIME}). Capping to avoid OOM.",
            stacklevel=2,
        )
        n_trotter = MAX_TROTTER_STEPS_RUNTIME

    # Step 6: Wire layout
    n_system = n_qubits
    system_wires = list(range(n_system))
    est_wires = list(range(n_system, n_system + n_estimation_wires))
    total_wires = n_system + n_estimation_wires

    # Step 7: Build @qjit circuit (mode-dependent)
    _qjit_deco = qjit(keep_intermediate=True) if _keep_intermediate else qjit

    if is_fixed:
        # H_fixed: build Hamiltonian OUTSIDE @qjit with Python floats.
        # Catalyst can constant-fold these → smaller IR, faster compilation.
        compiled_circuit, measurement_mode = _build_fixed_circuit(
            _qjit_deco,
            base_coeffs,
            ops,
            hf_state,
            system_wires,
            est_wires,
            total_wires,
            n_estimation_wires,
            base_time,
            n_trotter,
            qpe.n_shots,
            qpe.device_seed,
        )
    else:
        # H_dynamic: coefficients are JAX runtime parameters.
        # Used by hf_corrected and dynamic modes.
        compiled_circuit, measurement_mode = _build_dynamic_circuit(
            _qjit_deco,
            ops,
            hf_state,
            system_wires,
            est_wires,
            total_wires,
            n_estimation_wires,
            base_time,
            n_trotter,
            qpe.n_shots,
            qpe.device_seed,
        )

    return QPECircuitBundle(
        compiled_circuit=compiled_circuit,
        base_coeffs=base_coeffs,
        ops=ops,
        base_time=base_time,
        op_index_map=op_index_map,
        energy_shift=energy_shift,
        n_estimation_wires=n_estimation_wires,
        n_system_qubits=n_system,
        active_orbitals=mol.active_orbitals,
        n_trotter_steps=n_trotter,
        measurement_mode=measurement_mode,
        is_fixed_circuit=is_fixed,
        embedding_mode=config.embedding_mode,
    )


def _initial_solvent_mm_embedding(config: SolvationConfig, qm_coords: np.ndarray):
    """Return deterministic initial TIP3P MM embedding data for fixed Hamiltonian mode."""
    from .solvent import TIP3P_WATER, get_mm_embedding_data, initialize_solvent_ring

    qm_center = np.asarray(qm_coords, dtype=float).reshape(-1, 3).mean(axis=0)
    solvent_molecules = initialize_solvent_ring(
        model=TIP3P_WATER,
        n_molecules=config.n_waters,
        center=qm_center,
        radius=config.initial_water_distance,
        random_seed=config.random_seed,
    )
    return get_mm_embedding_data(solvent_molecules)


def _build_fixed_circuit(
    _qjit_deco,
    base_coeffs,
    ops,
    hf_state,
    system_wires,
    est_wires,
    total_wires,
    n_estimation_wires,
    base_time,
    n_trotter,
    n_shots,
    device_seed,
):
    """Build H_fixed zero-arg circuit (compile-time constant coefficients)."""
    # Hamiltonian built with Python floats BEFORE @qjit — enables constant-fold
    H_fixed = qml.dot(list(base_coeffs), ops)

    if n_shots == 0:
        dev = _select_device("lightning.qubit", total_wires, use_catalyst=True)

        @_qjit_deco
        def compiled_circuit():
            @qml.qnode(dev)
            def qnode():
                for wire, occ in zip(system_wires, hf_state, strict=True):
                    if occ == 1:
                        qml.PauliX(wires=wire)
                for w in est_wires:
                    qml.Hadamard(wires=w)
                for k, ew in enumerate(est_wires):
                    t = (2 ** (n_estimation_wires - 1 - k)) * base_time
                    qml.ctrl(
                        qml.adjoint(qml.TrotterProduct(H_fixed, time=t, n=n_trotter, order=2)),
                        control=ew,
                    )
                qml.adjoint(qml.QFT)(wires=est_wires)
                return qml.probs(wires=est_wires)

            return qnode()

        return compiled_circuit, "probs"
    else:
        dev = _select_device("lightning.qubit", total_wires, use_catalyst=True, seed=device_seed)

        @_qjit_deco
        def compiled_circuit():
            @qml.set_shots(n_shots)
            @qml.qnode(dev)
            def qnode():
                for wire, occ in zip(system_wires, hf_state, strict=True):
                    if occ == 1:
                        qml.PauliX(wires=wire)
                for w in est_wires:
                    qml.Hadamard(wires=w)
                for k, ew in enumerate(est_wires):
                    t = (2 ** (n_estimation_wires - 1 - k)) * base_time
                    qml.ctrl(
                        qml.adjoint(qml.TrotterProduct(H_fixed, time=t, n=n_trotter, order=2)),
                        control=ew,
                    )
                qml.adjoint(qml.QFT)(wires=est_wires)
                return qml.sample(wires=est_wires)

            return qnode()

        return compiled_circuit, "shots"


def _build_dynamic_circuit(
    _qjit_deco,
    ops,
    hf_state,
    system_wires,
    est_wires,
    total_wires,
    n_estimation_wires,
    base_time,
    n_trotter,
    n_shots,
    device_seed,
):
    """Build H_dynamic parameterized circuit (runtime JAX coefficient array)."""
    if n_shots == 0:
        dev = _select_device("lightning.qubit", total_wires, use_catalyst=True)

        @_qjit_deco
        def compiled_circuit(coeffs_arr):
            H_runtime = qml.dot(coeffs_arr, ops)

            @qml.qnode(dev)
            def qnode():
                for wire, occ in zip(system_wires, hf_state, strict=True):
                    if occ == 1:
                        qml.PauliX(wires=wire)
                for w in est_wires:
                    qml.Hadamard(wires=w)
                for k, ew in enumerate(est_wires):
                    t = (2 ** (n_estimation_wires - 1 - k)) * base_time
                    qml.ctrl(
                        qml.adjoint(
                            qml.TrotterProduct(
                                H_runtime, time=t, n=n_trotter, order=2, check_hermitian=False
                            )
                        ),
                        control=ew,
                    )
                qml.adjoint(qml.QFT)(wires=est_wires)
                return qml.probs(wires=est_wires)

            return qnode()

        return compiled_circuit, "probs"
    else:
        dev = _select_device("lightning.qubit", total_wires, use_catalyst=True, seed=device_seed)

        @_qjit_deco
        def compiled_circuit(coeffs_arr):
            H_runtime = qml.dot(coeffs_arr, ops)

            @qml.set_shots(n_shots)
            @qml.qnode(dev)
            def qnode():
                for wire, occ in zip(system_wires, hf_state, strict=True):
                    if occ == 1:
                        qml.PauliX(wires=wire)
                for w in est_wires:
                    qml.Hadamard(wires=w)
                for k, ew in enumerate(est_wires):
                    t = (2 ** (n_estimation_wires - 1 - k)) * base_time
                    qml.ctrl(
                        qml.adjoint(
                            qml.TrotterProduct(
                                H_runtime, time=t, n=n_trotter, order=2, check_hermitian=False
                            )
                        ),
                        control=ew,
                    )
                qml.adjoint(qml.QFT)(wires=est_wires)
                return qml.sample(wires=est_wires)

            return qnode()

        return compiled_circuit, "shots"
