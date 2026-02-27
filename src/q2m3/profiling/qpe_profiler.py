# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Three-phase QPE compilation profiling workflow.

Provides profile_* functions for measuring memory usage and timing across:
  - Phase A: PySCF → PennyLane Hamiltonian construction
  - Phase B: @qjit QPE circuit compilation (H_dynamic and H_fixed modes)
  - Phase C: Repeated execution of already-compiled circuit

Catalyst is an optional dependency. profile_qjit_compilation[_fixed] raise
ImportError at call time if catalyst is not installed.
"""

import logging
import time
import tracemalloc
from typing import Any

import numpy as np
import pennylane as qml

from q2m3.core.device_utils import select_device as _select_device
from q2m3.core.hamiltonian_utils import build_operator_index_map, decompose_hamiltonian
from q2m3.core.qpe import QPEEngine
from q2m3.interfaces.pyscf_pennylane import PySCFPennyLaneConverter
from q2m3.molecule import MoleculeConfig
from q2m3.profiling.catalyst_ir import analyze_ir_stages, ir_output_dir
from q2m3.profiling.memory import MemorySnapshot, MemoryTimeline, take_snapshot

try:
    from catalyst import qjit as _qjit
except ImportError:
    _qjit = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def profile_hamiltonian_build(
    mol: MoleculeConfig,
    n_est: int,
    n_trotter: int,
) -> tuple[MemorySnapshot, list, list[float], np.ndarray, dict]:
    """Profile Phase A: PySCF → PennyLane Hamiltonian construction.

    Returns:
        Tuple of (snapshot, ops, coeffs, hf_state, circuit_params)
    """
    tracemalloc.reset_peak()
    snap_before = take_snapshot("A:before")
    t0 = time.monotonic()

    # Build vacuum Hamiltonian
    converter = PySCFPennyLaneConverter(basis=mol.basis, mapping="jordan_wigner")
    H, n_qubits, hf_state = converter.pyscf_to_pennylane_hamiltonian(
        symbols=mol.symbols,
        coords=np.array(mol.coords),
        charge=mol.charge,
        active_electrons=mol.active_electrons,
        active_orbitals=mol.active_orbitals,
    )

    # Decompose into coefficients and operators
    coeffs, ops = decompose_hamiltonian(H)

    # Build operator index map (may extend coeffs/ops with missing Z terms)
    op_index_map, coeffs, ops = build_operator_index_map(ops, n_qubits, coeffs)

    # Compute shifted QPE parameters (use HF energy as shift reference)
    from pyscf import gto, scf

    pyscf_mol = gto.M(
        atom=[(s, c) for s, c in zip(mol.symbols, mol.coords, strict=True)],
        basis=mol.basis,
        charge=mol.charge,
        unit="Angstrom",
    )
    mf = scf.RHF(pyscf_mol)
    mf.verbose = 0
    hf_energy = mf.kernel()
    logger.debug("Phase A: HF energy = %.6f Ha", hf_energy)

    params = QPEEngine.compute_shifted_qpe_params(
        target_resolution=0.003,
        energy_range=0.2,
    )
    base_time = params["base_time"]
    energy_shift = hf_energy

    # Apply energy shift to Identity coefficient
    identity_idx = op_index_map["identity_idx"]
    coeffs[identity_idx] -= energy_shift
    base_coeffs = np.array(coeffs, dtype=np.float64)

    elapsed = time.monotonic() - t0
    snap_after = take_snapshot("A:after")
    snap_after.elapsed_s = elapsed
    logger.debug(
        "Phase A: elapsed = %.2fs, RSS delta = %.1f MB",
        elapsed,
        snap_after.rss_mb - snap_before.rss_mb,
    )

    circuit_params = {
        "n_system_qubits": n_qubits,
        "n_estimation_wires": n_est,
        "n_trotter": n_trotter,
        "n_terms": len(ops),
        "base_time": base_time,
        "energy_shift": energy_shift,
        "hf_state": hf_state,
        "base_coeffs": base_coeffs,
        "op_index_map": op_index_map,
    }

    result_snap = MemorySnapshot(
        label="Phase A: Hamiltonian Build",
        rss_mb=snap_after.rss_mb - snap_before.rss_mb,
        vm_peak_mb=snap_after.vm_peak_mb,
        maxrss_mb=snap_after.maxrss_mb,
        maxrss_children_mb=snap_after.maxrss_children_mb,
        tracemalloc_peak_mb=snap_after.tracemalloc_peak_mb,
        tracemalloc_current_mb=snap_after.tracemalloc_current_mb
        - snap_before.tracemalloc_current_mb,
        elapsed_s=elapsed,
    )

    return result_snap, ops, coeffs, hf_state, circuit_params


def profile_qjit_compilation(
    ops: list,
    coeffs: list[float],
    hf_state: np.ndarray,
    circuit_params: dict,
    ir_dir: str | None = None,
    keep_intermediate: bool = True,
) -> tuple[MemorySnapshot, MemoryTimeline, list, Any]:
    """Profile Phase B: H_dynamic mode @qjit QPE circuit compilation.

    This is the critical phase — first call triggers MLIR→LLVM compilation.

    IMPORTANT: @qjit is applied functionally INSIDE ir_output_dir context,
    because Catalyst captures os.getcwd() at decoration time to determine
    the IR workspace location.

    Args:
        ir_dir: If provided, IR files are preserved at this path.
                If None, a tempdir is used and auto-cleaned after IR analysis.
        keep_intermediate: If True, retain all 6 IR stages in memory for analysis.
                          If False, only final stage is kept (tests memory impact).

    Returns:
        Tuple of (snapshot, timeline, ir_analysis, compiled_fn)
    """
    if _qjit is None:
        raise ImportError(
            "catalyst is required for profile_qjit_compilation. "
            "Install with: pip install pennylane-catalyst"
        )

    n_system = circuit_params["n_system_qubits"]
    n_est = circuit_params["n_estimation_wires"]
    n_trotter = circuit_params["n_trotter"]
    base_time = circuit_params["base_time"]

    system_wires = list(range(n_system))
    est_wires = list(range(n_system, n_system + n_est))
    total_wires = n_system + n_est

    dev = _select_device("lightning.qubit", total_wires, use_catalyst=True)

    # Bare function — NO @qjit decorator here (applied functionally below)
    def qpe_profiled(coeffs_arr):
        H_runtime = qml.dot(coeffs_arr, ops)

        @qml.qnode(dev)
        def qnode():
            # HF state preparation via X gates (Catalyst-compatible)
            for wire, occ in zip(system_wires, hf_state, strict=True):
                if occ == 1:
                    qml.PauliX(wires=wire)
            # Hadamard on estimation qubits
            for w in est_wires:
                qml.Hadamard(wires=w)
            # Controlled time evolutions (MSB-first convention)
            for k, ew in enumerate(est_wires):
                t = (2 ** (n_est - 1 - k)) * base_time
                qml.ctrl(
                    qml.adjoint(
                        qml.TrotterProduct(
                            H_runtime, time=t, n=n_trotter, order=2, check_hermitian=False
                        )
                    ),
                    control=ew,
                )
            # Inverse QFT
            qml.adjoint(qml.QFT)(wires=est_wires)
            return qml.probs(wires=est_wires)

        return qnode()

    # Profile compilation (triggered by first call)
    tracemalloc.reset_peak()
    snap_before = take_snapshot("B:before")
    timeline = MemoryTimeline(interval_s=0.1)

    with timeline:
        t0 = time.monotonic()
        coeffs_jax = np.array(coeffs, dtype=np.float64)
        # Apply @qjit INSIDE ir_output_dir so Catalyst captures correct cwd
        with ir_output_dir(ir_dir):
            compiled = _qjit(keep_intermediate=keep_intermediate)(qpe_profiled)
            _result = compiled(coeffs_jax)  # Triggers compilation
            # Analyze IR stages BEFORE context exit (tempdir may be cleaned)
            ir_analysis = analyze_ir_stages(compiled) if keep_intermediate else []
        elapsed = time.monotonic() - t0

    logger.debug("Phase B (dynamic): elapsed = %.2fs", elapsed)
    snap_after = take_snapshot("B:after")

    result_snap = MemorySnapshot(
        label="Phase B: @qjit Compilation",
        rss_mb=snap_after.rss_mb - snap_before.rss_mb,
        vm_peak_mb=snap_after.vm_peak_mb,
        maxrss_mb=snap_after.maxrss_mb,
        maxrss_children_mb=snap_after.maxrss_children_mb,
        tracemalloc_peak_mb=snap_after.tracemalloc_peak_mb,
        tracemalloc_current_mb=snap_after.tracemalloc_current_mb
        - snap_before.tracemalloc_current_mb,
        elapsed_s=elapsed,
    )

    return result_snap, timeline, ir_analysis, compiled


def profile_qjit_compilation_fixed(
    ops: list,
    coeffs: list[float],
    hf_state: np.ndarray,
    circuit_params: dict,
    ir_dir: str | None = None,
    keep_intermediate: bool = True,
) -> tuple[MemorySnapshot, MemoryTimeline, list, Any]:
    """Profile Phase B for H_fixed mode: Hamiltonian built OUTSIDE @qjit.

    Coefficients are Python floats → Catalyst can constant-fold them into MLIR.
    The compiled function takes zero arguments.

    IMPORTANT: @qjit is applied functionally INSIDE ir_output_dir context,
    because Catalyst captures os.getcwd() at decoration time.

    Args:
        ir_dir: If provided, IR files are preserved at this path.
                If None, a tempdir is used and auto-cleaned after IR analysis.
        keep_intermediate: If True, retain all 6 IR stages for analysis.

    Returns:
        Tuple of (snapshot, timeline, ir_analysis, compiled_fn)
    """
    if _qjit is None:
        raise ImportError(
            "catalyst is required for profile_qjit_compilation_fixed. "
            "Install with: pip install pennylane-catalyst"
        )

    n_system = circuit_params["n_system_qubits"]
    n_est = circuit_params["n_estimation_wires"]
    n_trotter = circuit_params["n_trotter"]
    base_time = circuit_params["base_time"]

    system_wires = list(range(n_system))
    est_wires = list(range(n_system, n_system + n_est))
    total_wires = n_system + n_est

    dev = _select_device("lightning.qubit", total_wires, use_catalyst=True)

    # H_fixed: build concrete Hamiltonian BEFORE @qjit (Python floats, not JAX tracers)
    H_fixed = qml.dot(list(coeffs), ops)

    # Bare function — NO @qjit decorator here (applied functionally below)
    def qpe_fixed():  # Zero arguments — all coefficients are compile-time constants
        @qml.qnode(dev)
        def qnode():
            # HF state preparation via X gates (Catalyst-compatible)
            for wire, occ in zip(system_wires, hf_state, strict=True):
                if occ == 1:
                    qml.PauliX(wires=wire)
            # Hadamard on estimation qubits
            for w in est_wires:
                qml.Hadamard(wires=w)
            # Controlled time evolutions (MSB-first convention)
            for k, ew in enumerate(est_wires):
                t = (2 ** (n_est - 1 - k)) * base_time
                qml.ctrl(
                    qml.adjoint(qml.TrotterProduct(H_fixed, time=t, n=n_trotter, order=2)),
                    control=ew,
                )
            # Inverse QFT
            qml.adjoint(qml.QFT)(wires=est_wires)
            return qml.probs(wires=est_wires)

        return qnode()

    # Profile compilation (triggered by first call)
    tracemalloc.reset_peak()
    snap_before = take_snapshot("B:before")
    timeline = MemoryTimeline(interval_s=0.1)

    with timeline:
        t0 = time.monotonic()
        # Apply @qjit INSIDE ir_output_dir so Catalyst captures correct cwd
        with ir_output_dir(ir_dir):
            compiled = _qjit(keep_intermediate=keep_intermediate)(qpe_fixed)
            _result = compiled()  # Zero-arg call triggers compilation
            # Analyze IR stages BEFORE context exit (tempdir may be cleaned)
            ir_analysis = analyze_ir_stages(compiled) if keep_intermediate else []
        elapsed = time.monotonic() - t0

    logger.debug("Phase B (fixed): elapsed = %.2fs", elapsed)
    snap_after = take_snapshot("B:after")

    result_snap = MemorySnapshot(
        label="Phase B: @qjit Compilation (H_fixed)",
        rss_mb=snap_after.rss_mb - snap_before.rss_mb,
        vm_peak_mb=snap_after.vm_peak_mb,
        maxrss_mb=snap_after.maxrss_mb,
        maxrss_children_mb=snap_after.maxrss_children_mb,
        tracemalloc_peak_mb=snap_after.tracemalloc_peak_mb,
        tracemalloc_current_mb=snap_after.tracemalloc_current_mb
        - snap_before.tracemalloc_current_mb,
        elapsed_s=elapsed,
    )

    return result_snap, timeline, ir_analysis, compiled


def profile_execution(
    compiled_fn: Any,
    coeffs: list[float],
    n_calls: int = 5,
    is_fixed: bool = False,
) -> tuple[MemorySnapshot, float]:
    """Profile Phase C: repeated execution of already-compiled circuit.

    Returns:
        Tuple of (snapshot, prob_sum_from_last_call)
    """
    tracemalloc.reset_peak()
    snap_before = take_snapshot("C:before")
    t0 = time.monotonic()

    coeffs_arr = np.array(coeffs, dtype=np.float64)
    result = None
    for _ in range(n_calls):
        result = compiled_fn() if is_fixed else compiled_fn(coeffs_arr)

    elapsed = time.monotonic() - t0
    logger.debug("Phase C: %d calls in %.2fs", n_calls, elapsed)
    snap_after = take_snapshot("C:after")

    prob_sum = float(np.sum(result)) if result is not None else 0.0

    result_snap = MemorySnapshot(
        label=f"Phase C: Execution ({n_calls}x)",
        rss_mb=snap_after.rss_mb - snap_before.rss_mb,
        vm_peak_mb=snap_after.vm_peak_mb,
        maxrss_mb=snap_after.maxrss_mb,
        maxrss_children_mb=snap_after.maxrss_children_mb,
        tracemalloc_peak_mb=snap_after.tracemalloc_peak_mb,
        tracemalloc_current_mb=snap_after.tracemalloc_current_mb
        - snap_before.tracemalloc_current_mb,
        elapsed_s=elapsed,
    )

    return result_snap, prob_sum
