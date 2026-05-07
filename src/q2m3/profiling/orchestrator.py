# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Subprocess orchestration and parameter sweep for QPE compilation profiling.

Provides functions to run profiling passes in isolated subprocesses (avoiding
ru_maxrss accumulation) and to sweep across parameter grids.

All progress reporting uses an injectable ``on_progress`` callback — no direct
console output (Rich, print, etc.) so callers can plug in any UI layer.
"""

import logging
import multiprocessing
import tracemalloc
from collections.abc import Callable

from q2m3.molecule import MoleculeConfig
from q2m3.profiling.memory import ParentSideMonitor, ProfileResult
from q2m3.profiling.qpe_profiler import (
    profile_execution,
    profile_hamiltonian_build,
    profile_qjit_compilation,
    profile_qjit_compilation_fixed,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Molecule presets
# =============================================================================

MOLECULES: dict[str, MoleculeConfig] = {
    "h2": MoleculeConfig(
        name="H2",
        symbols=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        charge=0,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
    ),
    "h3o": MoleculeConfig(
        name="H3O+",
        symbols=["O", "H", "H", "H"],
        coords=[
            [0.0000, 0.0000, 0.1173],
            [0.0000, 0.9572, -0.4692],
            [0.8286, -0.4786, -0.4692],
            [-0.8286, -0.4786, -0.4692],
        ],
        charge=1,
        active_electrons=4,
        active_orbitals=4,
        basis="sto-3g",
    ),
}


# =============================================================================
# Parameter sweep grid
# =============================================================================

H2_SWEEP_GRID: list[tuple[int, int]] = [
    (2, 1),
    (2, 3),
    (2, 5),
    (3, 3),
    (3, 5),
    (4, 3),
    (4, 5),
    (4, 10),
]


# =============================================================================
# Private helpers
# =============================================================================


def _spawn_context() -> multiprocessing.context.SpawnContext:
    """Get 'spawn' multiprocessing context to avoid CUDA fork issues."""
    return multiprocessing.get_context("spawn")


# =============================================================================
# Core profiling functions
# =============================================================================


def run_single_profile(
    mol: MoleculeConfig,
    n_est: int,
    n_trotter: int,
    mode: str = "both",
    ir_dir: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> ProfileResult | tuple[ProfileResult, ProfileResult]:
    """Execute all three profiling phases for one parameter combination.

    Args:
        mol: Molecule configuration.
        n_est: Number of estimation wires.
        n_trotter: Trotter decomposition order.
        mode: "dynamic", "fixed", or "both".
        ir_dir: Directory to preserve IR files. None uses a tempdir.
        on_progress: Optional callback for progress messages.

    Returns:
        A single ProfileResult for "dynamic"/"fixed", or a tuple of
        (fixed_result, dynamic_result) when mode="both".
    """
    is_fixed = mode == "fixed"

    if on_progress:
        on_progress(
            f"Phase A: building Hamiltonian for {mol.name} (n_est={n_est}, n_trotter={n_trotter})"
        )

    # Phase A: Hamiltonian build (shared between modes)
    snap_a, ops, coeffs, hf_state, circuit_params = profile_hamiltonian_build(mol, n_est, n_trotter)
    circuit_params["n_estimation_wires"] = n_est
    circuit_params["n_trotter"] = n_trotter

    n_terms = circuit_params["n_terms"]
    ir_scale = n_est * n_trotter * n_terms

    if on_progress:
        on_progress(f"Phase B: @qjit compilation (mode={mode})")

    # Phase B: @qjit compilation (mode-dependent)
    if is_fixed:
        snap_b, timeline, ir_analysis, compiled_fn = profile_qjit_compilation_fixed(
            ops,
            coeffs,
            hf_state,
            circuit_params,
            ir_dir=ir_dir,
        )
    else:
        snap_b, timeline, ir_analysis, compiled_fn = profile_qjit_compilation(
            ops,
            coeffs,
            hf_state,
            circuit_params,
            ir_dir=ir_dir,
        )

    if on_progress:
        on_progress("Phase C: execution measurement")

    # Phase C: Execution
    snap_c, prob_sum = profile_execution(compiled_fn, coeffs, is_fixed=is_fixed)

    return ProfileResult(
        molecule=mol.name,
        n_system_qubits=circuit_params["n_system_qubits"],
        n_estimation_wires=n_est,
        n_trotter=n_trotter,
        n_terms=n_terms,
        ir_scale=ir_scale,
        mode=mode,
        phase_a=snap_a,
        phase_b=snap_b,
        phase_c=snap_c,
        timeline_peak_mb=timeline.peak_rss_mb,
        timeline_samples=timeline.samples,
        ir_analysis=ir_analysis,
        prob_sum=prob_sum,
    )


def run_single_profile_in_subprocess(
    mol_key: str,
    n_est: int,
    n_trotter: int,
    queue: multiprocessing.Queue,
    mode: str = "dynamic",
    ir_dir: str | None = None,
) -> None:
    """Run a single profiling pass inside a subprocess (avoids ru_maxrss accumulation).

    Results are placed into *queue* — this function does not return the result.
    """
    try:
        tracemalloc.start()
        mol = MOLECULES[mol_key]
        result = run_single_profile(
            mol,
            n_est,
            n_trotter,
            mode=mode,
            ir_dir=ir_dir,
        )
        tracemalloc.stop()
        queue.put(result)
    except Exception as e:
        queue.put(
            ProfileResult(
                molecule=mol_key,
                n_system_qubits=0,
                n_estimation_wires=n_est,
                n_trotter=n_trotter,
                n_terms=0,
                ir_scale=0,
                mode=mode,
                error=str(e),
            )
        )


def _run_mode_in_subprocess(
    mol_key: str,
    n_est: int,
    n_trotter: int,
    mode: str,
    ir_dir: str | None = None,
) -> tuple[ProfileResult, dict]:
    """Run a single mode in a subprocess with parent-side RSS monitoring.

    Returns:
        Tuple of (ProfileResult, parent_monitor_data_dict).
    """
    ctx = _spawn_context()
    queue = ctx.Queue()
    proc = ctx.Process(
        target=run_single_profile_in_subprocess,
        args=(mol_key, n_est, n_trotter, queue, mode, ir_dir),
    )
    proc.start()

    # Start parent-side monitoring of child process
    monitor = ParentSideMonitor(proc.pid, interval_s=0.1)
    monitor.start()

    proc.join(timeout=600)
    monitor.stop()

    # Collect parent-observed data
    parent_data = {
        "peak_rss_mb": monitor.peak_rss_mb,
        "peak_hwm_mb": monitor.peak_hwm_mb,
        "peak_smaps": monitor.peak_smaps,
        "n_samples": len(monitor.samples),
    }

    if proc.is_alive():
        proc.kill()
        proc.join()
        return (
            ProfileResult(
                molecule=mol_key,
                n_system_qubits=0,
                n_estimation_wires=n_est,
                n_trotter=n_trotter,
                n_terms=0,
                ir_scale=0,
                mode=mode,
                error="timeout",
            ),
            parent_data,
        )
    elif not queue.empty():
        return queue.get(), parent_data
    else:
        return (
            ProfileResult(
                molecule=mol_key,
                n_system_qubits=0,
                n_estimation_wires=n_est,
                n_trotter=n_trotter,
                n_terms=0,
                ir_scale=0,
                mode=mode,
                error="no result from subprocess",
            ),
            parent_data,
        )


def run_both_modes(
    mol_key: str,
    n_est: int,
    n_trotter: int,
    ir_dir: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[ProfileResult, ProfileResult, dict, dict]:
    """Run fixed and dynamic modes in isolated subprocesses with parent-side monitoring.

    Returns:
        Tuple of (result_fixed, result_dynamic, parent_data_fixed, parent_data_dynamic).
    """
    if on_progress:
        on_progress("Running H_fixed mode...")

    result_fixed, parent_fixed = _run_mode_in_subprocess(
        mol_key,
        n_est,
        n_trotter,
        "fixed",
        ir_dir=ir_dir,
    )

    if on_progress:
        if result_fixed.error:
            on_progress(f"H_fixed ERROR: {result_fixed.error}")
        elif result_fixed.phase_b is not None:
            on_progress(
                f"H_fixed: self={result_fixed.phase_b.maxrss_mb:.0f} MB, "
                f"parent-observed={parent_fixed['peak_rss_mb']:.0f} MB, "
                f"compile={result_fixed.phase_b.elapsed_s:.1f}s"
            )

    if on_progress:
        on_progress("Running H_dynamic mode...")

    result_dynamic, parent_dynamic = _run_mode_in_subprocess(
        mol_key,
        n_est,
        n_trotter,
        "dynamic",
        ir_dir=ir_dir,
    )

    if on_progress:
        if result_dynamic.error:
            on_progress(f"H_dynamic ERROR: {result_dynamic.error}")
        elif result_dynamic.phase_b is not None:
            on_progress(
                f"H_dynamic: self={result_dynamic.phase_b.maxrss_mb:.0f} MB, "
                f"parent-observed={parent_dynamic['peak_rss_mb']:.0f} MB, "
                f"compile={result_dynamic.phase_b.elapsed_s:.1f}s"
            )

    return result_fixed, result_dynamic, parent_fixed, parent_dynamic


def run_sweep(
    mol_key: str = "h2",
    mode: str = "dynamic",
    grid: list[tuple[int, int]] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[tuple[int, int], ProfileResult]:
    """Run parameter sweep with subprocess isolation.

    Args:
        mol_key: Key into MOLECULES dict.
        mode: "dynamic" or "fixed".
        grid: List of (n_est, n_trotter) pairs. Defaults to H2_SWEEP_GRID.
        on_progress: Optional callback for progress messages.

    Returns:
        Dict mapping (n_est, n_trotter) to ProfileResult for successful runs.
        Failed grid points are silently skipped (logged at warning level).
    """
    if grid is None:
        grid = H2_SWEEP_GRID

    results: dict[tuple[int, int], ProfileResult] = {}

    for i, (n_est, n_trotter) in enumerate(grid):
        if on_progress:
            on_progress(
                f"[{i + 1}/{len(grid)}] n_est={n_est}, n_trotter={n_trotter} (H_{mode}) ..."
            )

        try:
            result, _parent_data = _run_mode_in_subprocess(
                mol_key,
                n_est,
                n_trotter,
                mode,
            )
            results[(n_est, n_trotter)] = result

            if on_progress:
                if result.error:
                    on_progress(f"  ERROR: {result.error}")
                elif result.phase_b is not None:
                    on_progress(
                        f"  peak={result.phase_b.maxrss_mb:.0f} MB, "
                        f"compile={result.phase_b.elapsed_s:.1f}s"
                    )
        except Exception as e:
            logger.warning("Sweep grid point (%d, %d) failed: %s", n_est, n_trotter, e)
            if on_progress:
                on_progress(f"  SKIPPED: {e}")

    return results
