#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
QPE Circuit Compilation Memory Profiler

Measures real peak memory consumption during Catalyst @qjit compilation of
QPE circuits in two Hamiltonian modes:

  - H_fixed:   Coefficients are Python floats, built OUTSIDE @qjit.
                Catalyst can constant-fold them → smaller IR, less memory.
  - H_dynamic: Coefficients are JAX-traced runtime parameters INSIDE @qjit.
                Full symbolic graph preserved → larger IR, more memory.

Designed to answer Xanadu's question:

    "4.2k IR operations is relatively small and isn't expected to drive
     such large memory requirements."

Key fact: The ~16GB figure in energy.py:660-661 describes an OLD architecture
bug (QPE IR double-inlined into MC loop MLIR). Current code pre-compiles QPE
separately (orchestrator.py:656-673). This script measures ACTUAL memory.

Three measurement layers:
  - resource.getrusage(RUSAGE_SELF).ru_maxrss → process lifetime peak RSS
    (captures C++ MLIR compiler allocations)
  - /proc/self/status VmRSS/VmPeak → current/peak RSS (Linux kernel level)
  - tracemalloc → Python heap only (PennyLane IR construction, JAX tracing)

Usage:
    uv run python examples/qpe_memory_profile.py                          # both modes
    uv run python examples/qpe_memory_profile.py --mode fixed             # H_fixed only
    uv run python examples/qpe_memory_profile.py --mode dynamic           # H_dynamic only
    uv run python examples/qpe_memory_profile.py --mode both --n-est 2    # compare modes
    uv run python examples/qpe_memory_profile.py --sweep                  # parameter sweep
    uv run python examples/qpe_memory_profile.py --sweep --mode fixed     # sweep H_fixed
    uv run python examples/qpe_memory_profile.py --ir-dir ./tmp           # preserve IR files
"""

import argparse
import multiprocessing
import os
import resource
import shutil
import sys
import tempfile
import threading
import time
import tracemalloc
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np  # noqa: E402
import pennylane as qml  # noqa: E402
from catalyst import qjit  # noqa: E402
from catalyst.debug import get_compilation_stage  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from examples.mc_solvation.config import MoleculeConfig  # noqa: E402
from examples.mc_solvation.energy import (  # noqa: E402
    build_operator_index_map,
    decompose_hamiltonian,
)
from q2m3.core import QPEEngine  # noqa: E402
from q2m3.core.device_utils import CATALYST_VERSION  # noqa: E402
from q2m3.core.device_utils import select_device as _select_device  # noqa: E402
from q2m3.interfaces import PySCFPennyLaneConverter  # noqa: E402

console = Console()


# =============================================================================
# Molecule Presets
# =============================================================================

MOLECULES = {
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
# Step 1-2: Memory Measurement Infrastructure
# =============================================================================


@dataclass
class MemorySnapshot:
    """Multi-layer memory snapshot at a single point in time."""

    label: str
    rss_mb: float  # /proc/self/status VmRSS
    vm_peak_mb: float  # /proc/self/status VmPeak
    maxrss_mb: float  # resource.getrusage (cumulative peak)
    tracemalloc_peak_mb: float  # Python heap peak since last reset
    tracemalloc_current_mb: float  # Python heap current
    elapsed_s: float = 0.0


@dataclass
class ProfileResult:
    """Complete profiling result for one parameter combination."""

    molecule: str
    n_system_qubits: int
    n_estimation_wires: int
    n_trotter: int
    n_terms: int
    ir_scale: int  # n_est × n_trotter × n_terms
    mode: str = "dynamic"  # "fixed" or "dynamic"
    phase_a: MemorySnapshot | None = None
    phase_b: MemorySnapshot | None = None
    phase_c: MemorySnapshot | None = None
    timeline_peak_mb: float = 0.0
    timeline_samples: list = field(default_factory=list)
    ir_analysis: list = field(default_factory=list)  # list of (stage, size_kb, lines)
    prob_sum: float = 0.0
    error: str | None = None


def read_proc_status() -> dict[str, float]:
    """Parse /proc/self/status for VmRSS and VmPeak (Linux only)."""
    result = {"VmRSS": 0.0, "VmPeak": 0.0}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                for key in result:
                    if line.startswith(key + ":"):
                        # Format: "VmRSS:    123456 kB"
                        result[key] = float(line.split()[1]) / 1024.0  # kB → MB
    except FileNotFoundError:
        pass  # Non-Linux platform
    return result


def take_snapshot(label: str) -> MemorySnapshot:
    """Take a comprehensive memory snapshot from all measurement layers."""
    proc = read_proc_status()
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in KB on Linux
    maxrss_mb = ru.ru_maxrss / 1024.0

    tm_current, tm_peak = tracemalloc.get_traced_memory()

    return MemorySnapshot(
        label=label,
        rss_mb=proc["VmRSS"],
        vm_peak_mb=proc["VmPeak"],
        maxrss_mb=maxrss_mb,
        tracemalloc_peak_mb=tm_peak / (1024 * 1024),
        tracemalloc_current_mb=tm_current / (1024 * 1024),
    )


@contextmanager
def ir_output_dir(path: str | None = None):
    """Manage Catalyst IR workspace directory.

    Must wrap @qjit decoration (not just the call), because Catalyst captures
    os.getcwd() at decoration time to determine the IR output location.

    Args:
        path: If provided, IR files are written here and preserved after exit.
              If None, a temporary directory is used and auto-cleaned on exit.

    Yields:
        The workspace directory path (user-specified or auto-created tempdir).
    """
    original_cwd = os.getcwd()
    if path is not None:
        os.makedirs(path, exist_ok=True)
        workspace = path
    else:
        workspace = tempfile.mkdtemp(prefix="qpe_ir_")
    os.chdir(workspace)
    try:
        yield workspace
    finally:
        os.chdir(original_cwd)
        if path is None:
            shutil.rmtree(workspace, ignore_errors=True)


class MemoryTimeline:
    """Background daemon thread sampling /proc/self/status VmRSS at 100ms intervals."""

    def __init__(self, interval_s: float = 0.1):
        self._interval = interval_s
        self._samples: list[tuple[float, float]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    def __enter__(self):
        self._start_time = time.monotonic()
        self._samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _sample_loop(self):
        while not self._stop_event.is_set():
            elapsed = time.monotonic() - self._start_time
            rss = read_proc_status()["VmRSS"]
            self._samples.append((elapsed, rss))
            self._stop_event.wait(self._interval)

    @property
    def peak_rss_mb(self) -> float:
        if not self._samples:
            return 0.0
        return max(s[1] for s in self._samples)

    @property
    def samples(self) -> list[tuple[float, float]]:
        return list(self._samples)


# =============================================================================
# Step 3: Phase A — Hamiltonian Construction Profiler
# =============================================================================


def profile_hamiltonian_build(
    mol: MoleculeConfig, n_est: int, n_trotter: int
) -> tuple[MemorySnapshot, list, list[float], np.ndarray, dict]:
    """
    Profile Phase A: PySCF → PennyLane Hamiltonian construction.

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

    # Report delta
    result_snap = MemorySnapshot(
        label="Phase A: Hamiltonian Build",
        rss_mb=snap_after.rss_mb - snap_before.rss_mb,
        vm_peak_mb=snap_after.vm_peak_mb,
        maxrss_mb=snap_after.maxrss_mb,
        tracemalloc_peak_mb=snap_after.tracemalloc_peak_mb,
        tracemalloc_current_mb=snap_after.tracemalloc_current_mb
        - snap_before.tracemalloc_current_mb,
        elapsed_s=elapsed,
    )

    return result_snap, ops, coeffs, hf_state, circuit_params


# =============================================================================
# Step 4: Phase B — @qjit Compilation Profiler (core)
# =============================================================================


def profile_qjit_compilation(
    ops: list,
    coeffs: list[float],
    hf_state: np.ndarray,
    circuit_params: dict,
    ir_dir: str | None = None,
) -> tuple[MemorySnapshot, MemoryTimeline, list, object]:
    """
    Profile Phase B: @qjit(keep_intermediate=True) QPE circuit compilation.

    This is the critical phase — first call triggers MLIR→LLVM compilation.

    IMPORTANT: @qjit is applied functionally INSIDE ir_output_dir context,
    because Catalyst captures os.getcwd() at decoration time to determine
    the IR workspace location.

    Args:
        ir_dir: If provided, IR files are preserved at this path.
                If None, a tempdir is used and auto-cleaned after IR analysis.

    Returns:
        Tuple of (snapshot, timeline, ir_analysis, compiled_fn)
    """
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
            compiled = qjit(keep_intermediate=True)(qpe_profiled)
            _result = compiled(coeffs_jax)  # Triggers compilation
            # Analyze IR stages BEFORE context exit (tempdir may be cleaned)
            ir_analysis = analyze_ir_stages(compiled)
        elapsed = time.monotonic() - t0

    snap_after = take_snapshot("B:after")

    result_snap = MemorySnapshot(
        label="Phase B: @qjit Compilation",
        rss_mb=snap_after.rss_mb - snap_before.rss_mb,
        vm_peak_mb=snap_after.vm_peak_mb,
        maxrss_mb=snap_after.maxrss_mb,
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
) -> tuple[MemorySnapshot, MemoryTimeline, list, object]:
    """
    Profile Phase B for H_fixed mode: Hamiltonian built OUTSIDE @qjit.

    Coefficients are Python floats → Catalyst can constant-fold them into MLIR.
    The compiled function takes zero arguments.

    IMPORTANT: @qjit is applied functionally INSIDE ir_output_dir context,
    because Catalyst captures os.getcwd() at decoration time.

    Args:
        ir_dir: If provided, IR files are preserved at this path.
                If None, a tempdir is used and auto-cleaned after IR analysis.
    """
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
            compiled = qjit(keep_intermediate=True)(qpe_fixed)
            _result = compiled()  # Zero-arg call triggers compilation
            # Analyze IR stages BEFORE context exit (tempdir may be cleaned)
            ir_analysis = analyze_ir_stages(compiled)
        elapsed = time.monotonic() - t0

    snap_after = take_snapshot("B:after")

    result_snap = MemorySnapshot(
        label="Phase B: @qjit Compilation (H_fixed)",
        rss_mb=snap_after.rss_mb - snap_before.rss_mb,
        vm_peak_mb=snap_after.vm_peak_mb,
        maxrss_mb=snap_after.maxrss_mb,
        tracemalloc_peak_mb=snap_after.tracemalloc_peak_mb,
        tracemalloc_current_mb=snap_after.tracemalloc_current_mb
        - snap_before.tracemalloc_current_mb,
        elapsed_s=elapsed,
    )

    return result_snap, timeline, ir_analysis, compiled


# =============================================================================
# Step 5: Phase C — Execution Profiler
# =============================================================================


def profile_execution(
    compiled_fn, coeffs: list[float], n_calls: int = 5, is_fixed: bool = False
) -> tuple[MemorySnapshot, float]:
    """
    Profile Phase C: repeated execution of already-compiled circuit.

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
    snap_after = take_snapshot("C:after")

    prob_sum = float(np.sum(result)) if result is not None else 0.0

    result_snap = MemorySnapshot(
        label=f"Phase C: Execution ({n_calls}x)",
        rss_mb=snap_after.rss_mb - snap_before.rss_mb,
        vm_peak_mb=snap_after.vm_peak_mb,
        maxrss_mb=snap_after.maxrss_mb,
        tracemalloc_peak_mb=snap_after.tracemalloc_peak_mb,
        tracemalloc_current_mb=snap_after.tracemalloc_current_mb
        - snap_before.tracemalloc_current_mb,
        elapsed_s=elapsed,
    )

    return result_snap, prob_sum


# =============================================================================
# Step 6: Catalyst IR Analysis
# =============================================================================

COMPILATION_STAGES = [
    "mlir",
    "QuantumCompilationStage",
    "HLOLoweringStage",
    "BufferizationStage",
    "MLIRToLLVMDialectConversion",
    "LLVMIRTranslation",
]


def analyze_ir_stages(compiled_fn) -> list[tuple[str, float, int]]:
    """
    Export IR text from each compilation stage and measure size/lines.

    Returns:
        List of (stage_name, size_kb, n_lines)
    """
    results = []
    for stage in COMPILATION_STAGES:
        try:
            ir_text = get_compilation_stage(compiled_fn, stage)
            size_kb = len(ir_text.encode("utf-8")) / 1024.0
            n_lines = ir_text.count("\n")
            results.append((stage, size_kb, n_lines))
        except Exception:
            # Stage may not exist for this compilation
            pass
    return results


# =============================================================================
# Step 7: Parameter Sweep (subprocess isolation)
# =============================================================================


def _spawn_context():
    """Get 'spawn' multiprocessing context to avoid CUDA fork issues."""
    return multiprocessing.get_context("spawn")


H2_SWEEP_GRID = [
    (2, 1),
    (2, 3),
    (2, 5),
    (3, 3),
    (3, 5),
    (4, 3),
    (4, 5),
    (4, 10),
]


def run_single_profile_in_subprocess(
    mol_key: str,
    n_est: int,
    n_trotter: int,
    queue,
    mode: str = "dynamic",
    ir_dir: str | None = None,
):
    """Run a single profiling pass inside a subprocess (avoids ru_maxrss accumulation)."""
    try:
        tracemalloc.start()
        result = run_single_profile(mol_key, n_est, n_trotter, mode=mode, ir_dir=ir_dir)
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


def run_single_profile(
    mol_key: str,
    n_est: int,
    n_trotter: int,
    mode: str = "dynamic",
    ir_dir: str | None = None,
) -> ProfileResult:
    """Execute all three profiling phases for one parameter combination."""
    mol = MOLECULES[mol_key]
    is_fixed = mode == "fixed"

    # Phase A: Hamiltonian build (shared between modes)
    snap_a, ops, coeffs, hf_state, circuit_params = profile_hamiltonian_build(mol, n_est, n_trotter)
    circuit_params["n_estimation_wires"] = n_est
    circuit_params["n_trotter"] = n_trotter

    n_terms = circuit_params["n_terms"]
    ir_scale = n_est * n_trotter * n_terms

    # Phase B: @qjit compilation (mode-dependent)
    if is_fixed:
        snap_b, timeline, ir_analysis, compiled_fn = profile_qjit_compilation_fixed(
            ops, coeffs, hf_state, circuit_params, ir_dir=ir_dir
        )
    else:
        snap_b, timeline, ir_analysis, compiled_fn = profile_qjit_compilation(
            ops, coeffs, hf_state, circuit_params, ir_dir=ir_dir
        )

    # Phase C: Execution
    snap_c, prob_sum = profile_execution(compiled_fn, coeffs, is_fixed=is_fixed)

    return ProfileResult(
        molecule=mol_key,
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


def run_sweep(
    mol_key: str, mode: str = "dynamic", ir_dir: str | None = None
) -> list[ProfileResult]:
    """Run parameter sweep with subprocess isolation."""
    results = []
    grid = H2_SWEEP_GRID  # Only H2 sweep is practical

    for i, (n_est, n_trotter) in enumerate(grid):
        console.print(
            f"  [{i + 1}/{len(grid)}] n_est={n_est}, n_trotter={n_trotter} " f"(H_{mode}) ...",
            end=" ",
        )
        ctx = _spawn_context()
        queue = ctx.Queue()
        proc = ctx.Process(
            target=run_single_profile_in_subprocess,
            args=(mol_key, n_est, n_trotter, queue, mode, ir_dir),
        )
        proc.start()
        proc.join(timeout=600)  # 10 min timeout per config

        if proc.is_alive():
            proc.kill()
            proc.join()
            console.print("[red]TIMEOUT[/red]")
            results.append(
                ProfileResult(
                    molecule=mol_key,
                    n_system_qubits=0,
                    n_estimation_wires=n_est,
                    n_trotter=n_trotter,
                    n_terms=0,
                    ir_scale=0,
                    error="timeout",
                )
            )
        elif not queue.empty():
            result = queue.get()
            if result.error:
                console.print(f"[red]ERROR: {result.error}[/red]")
            else:
                console.print(
                    f"[green]peak={result.phase_b.maxrss_mb:.0f} MB, "
                    f"compile={result.phase_b.elapsed_s:.1f}s[/green]"
                )
            results.append(result)
        else:
            console.print("[red]NO RESULT[/red]")
            results.append(
                ProfileResult(
                    molecule=mol_key,
                    n_system_qubits=0,
                    n_estimation_wires=n_est,
                    n_trotter=n_trotter,
                    n_terms=0,
                    ir_scale=0,
                    error="no result from subprocess",
                )
            )
    return results


def _run_mode_in_subprocess(
    mol_key: str, n_est: int, n_trotter: int, mode: str, ir_dir: str | None = None
):
    """Run a single mode in a subprocess and return ProfileResult."""
    ctx = _spawn_context()
    queue = ctx.Queue()
    proc = ctx.Process(
        target=run_single_profile_in_subprocess,
        args=(mol_key, n_est, n_trotter, queue, mode, ir_dir),
    )
    proc.start()
    proc.join(timeout=600)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return ProfileResult(
            molecule=mol_key,
            n_system_qubits=0,
            n_estimation_wires=n_est,
            n_trotter=n_trotter,
            n_terms=0,
            ir_scale=0,
            mode=mode,
            error="timeout",
        )
    elif not queue.empty():
        return queue.get()
    else:
        return ProfileResult(
            molecule=mol_key,
            n_system_qubits=0,
            n_estimation_wires=n_est,
            n_trotter=n_trotter,
            n_terms=0,
            ir_scale=0,
            mode=mode,
            error="no result from subprocess",
        )


def run_both_modes(
    mol_key: str, n_est: int, n_trotter: int, ir_dir: str | None = None
) -> tuple[ProfileResult, ProfileResult]:
    """Run fixed and dynamic modes in isolated subprocesses (avoids ru_maxrss pollution)."""
    console.print("\n[bold]Running H_fixed mode...[/bold]")
    result_fixed = _run_mode_in_subprocess(mol_key, n_est, n_trotter, "fixed", ir_dir=ir_dir)
    if result_fixed.error:
        console.print(f"  [red]H_fixed ERROR: {result_fixed.error}[/red]")
    else:
        console.print(
            f"  [green]H_fixed: peak={result_fixed.phase_b.maxrss_mb:.0f} MB, "
            f"compile={result_fixed.phase_b.elapsed_s:.1f}s[/green]"
        )

    console.print("\n[bold]Running H_dynamic mode...[/bold]")
    result_dynamic = _run_mode_in_subprocess(mol_key, n_est, n_trotter, "dynamic", ir_dir=ir_dir)
    if result_dynamic.error:
        console.print(f"  [red]H_dynamic ERROR: {result_dynamic.error}[/red]")
    else:
        console.print(
            f"  [green]H_dynamic: peak={result_dynamic.phase_b.maxrss_mb:.0f} MB, "
            f"compile={result_dynamic.phase_b.elapsed_s:.1f}s[/green]"
        )

    return result_fixed, result_dynamic


def print_mode_comparison(fixed: ProfileResult, dynamic: ProfileResult):
    """Print side-by-side comparison table of H_fixed vs H_dynamic."""
    table = Table(title="H_fixed vs H_dynamic Comparison", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("H_fixed", justify="right", style="cyan")
    table.add_column("H_dynamic", justify="right", style="magenta")
    table.add_column("Ratio (dyn/fix)", justify="right", style="yellow")

    def _ratio(a: float, b: float) -> str:
        if a > 0:
            return f"{b / a:.2f}×"
        return "—"

    # Compilation time
    fix_t = fixed.phase_b.elapsed_s if fixed.phase_b else 0
    dyn_t = dynamic.phase_b.elapsed_s if dynamic.phase_b else 0
    table.add_row("Compile Time (s)", f"{fix_t:.1f}", f"{dyn_t:.1f}", _ratio(fix_t, dyn_t))

    # Peak RSS
    fix_rss = fixed.phase_b.maxrss_mb if fixed.phase_b else 0
    dyn_rss = dynamic.phase_b.maxrss_mb if dynamic.phase_b else 0
    table.add_row("Peak RSS (MB)", f"{fix_rss:.0f}", f"{dyn_rss:.0f}", _ratio(fix_rss, dyn_rss))

    # Timeline peak
    table.add_row(
        "Timeline Peak (MB)",
        f"{fixed.timeline_peak_mb:.0f}",
        f"{dynamic.timeline_peak_mb:.0f}",
        _ratio(fixed.timeline_peak_mb, dynamic.timeline_peak_mb),
    )

    # Python heap peak
    fix_heap = fixed.phase_b.tracemalloc_peak_mb if fixed.phase_b else 0
    dyn_heap = dynamic.phase_b.tracemalloc_peak_mb if dynamic.phase_b else 0
    table.add_row("Python Heap Peak (MB)", f"{fix_heap:.1f}", f"{dyn_heap:.1f}", "")

    # Execution time
    fix_exec = fixed.phase_c.elapsed_s if fixed.phase_c else 0
    dyn_exec = dynamic.phase_c.elapsed_s if dynamic.phase_c else 0
    table.add_row("Exec Time (5x, s)", f"{fix_exec:.3f}", f"{dyn_exec:.3f}", "")

    # IR sizes (largest stage)
    def _largest_ir(analysis):
        if not analysis:
            return 0.0, "N/A"
        stage, size, _ = max(analysis, key=lambda x: x[1])
        return size, stage

    fix_ir, fix_stage = _largest_ir(fixed.ir_analysis)
    dyn_ir, dyn_stage = _largest_ir(dynamic.ir_analysis)
    table.add_row(
        "Largest IR (KB)",
        f"{fix_ir:.0f} ({fix_stage})",
        f"{dyn_ir:.0f} ({dyn_stage})",
        _ratio(fix_ir, dyn_ir),
    )

    # Prob sum sanity
    table.add_row("Prob Sum", f"{fixed.prob_sum:.6f}", f"{dynamic.prob_sum:.6f}", "")

    console.print(table)


# =============================================================================
# Step 8: Output Formatting
# =============================================================================


def print_system_info():
    """Print system information panel."""
    import platform

    total_ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)

    info_lines = [
        f"Python:     {platform.python_version()}",
        f"PennyLane:  {qml.__version__}",
        f"Catalyst:   {CATALYST_VERSION}",
        f"Platform:   {platform.system()} {platform.machine()}",
        f"Total RAM:  {total_ram_gb:.1f} GB",
    ]
    try:
        import jax

        info_lines.append(f"JAX:        {jax.__version__} (backend: {jax.default_backend()})")
    except ImportError:
        info_lines.append("JAX:        not installed")

    console.print(Panel("\n".join(info_lines), title="System Information", border_style="blue"))


def print_circuit_params(result: ProfileResult):
    """Print circuit parameters panel."""
    mode_label = (
        "H_fixed (compile-time constants)"
        if result.mode == "fixed"
        else "H_dynamic (runtime coefficients)"
    )
    lines = [
        f"Molecule:          {result.molecule}",
        f"Hamiltonian mode:  {mode_label}",
        f"System qubits:     {result.n_system_qubits}",
        f"Estimation wires:  {result.n_estimation_wires}",
        f"Total qubits:      {result.n_system_qubits + result.n_estimation_wires}",
        f"Trotter steps:     {result.n_trotter}",
        f"Hamiltonian terms: {result.n_terms}",
        f"IR scale estimate: {result.ir_scale} "
        f"({result.n_estimation_wires}×{result.n_trotter}×{result.n_terms})",
    ]
    console.print(Panel("\n".join(lines), title="Circuit Parameters", border_style="cyan"))


def print_memory_table(result: ProfileResult):
    """Print Phase A/B/C memory comparison table."""
    table = Table(title="Memory Profile by Phase", show_lines=True)
    table.add_column("Phase", style="bold")
    table.add_column("RSS Δ (MB)", justify="right")
    table.add_column("VmPeak (MB)", justify="right")
    table.add_column("ru_maxrss (MB)", justify="right")
    table.add_column("Python Heap Peak (MB)", justify="right")
    table.add_column("Wall Time (s)", justify="right")

    for snap in [result.phase_a, result.phase_b, result.phase_c]:
        if snap is None:
            continue
        table.add_row(
            snap.label,
            f"{snap.rss_mb:+.1f}",
            f"{snap.vm_peak_mb:.1f}",
            f"{snap.maxrss_mb:.1f}",
            f"{snap.tracemalloc_peak_mb:.1f}",
            f"{snap.elapsed_s:.2f}",
        )

    console.print(table)


def print_ir_analysis(ir_analysis: list[tuple[str, float, int]]):
    """Print Catalyst IR stage analysis table."""
    if not ir_analysis:
        console.print("[yellow]No IR analysis data available.[/yellow]")
        return

    table = Table(title="Catalyst IR Stage Analysis", show_lines=True)
    table.add_column("Compilation Stage", style="bold")
    table.add_column("IR Size (KB)", justify="right")
    table.add_column("IR Lines", justify="right")
    table.add_column("Amplification", justify="right", style="yellow")

    prev_size = None
    for stage, size_kb, n_lines in ir_analysis:
        amp = f"{size_kb / prev_size:.1f}×" if prev_size and prev_size > 0 else "—"
        table.add_row(stage, f"{size_kb:.1f}", f"{n_lines:,}", amp)
        prev_size = size_kb

    console.print(table)

    # Total amplification
    if len(ir_analysis) >= 2:
        first_size = ir_analysis[0][1]
        last_size = ir_analysis[-1][1]
        if first_size > 0:
            console.print(
                f"  Total IR amplification: {first_size:.1f} KB → {last_size:.1f} KB "
                f"([bold]{last_size / first_size:.1f}×[/bold])"
            )


def print_memory_timeline(samples: list[tuple[float, float]]):
    """Print ASCII timeline of RSS during compilation."""
    if len(samples) < 3:
        return

    console.print("\n[bold]Memory Timeline (RSS during compilation):[/bold]")
    chart_width = 60
    rss_values = [s[1] for s in samples]
    min_rss = min(rss_values)
    max_rss = max(rss_values)
    rss_range = max_rss - min_rss

    if rss_range < 1:
        console.print("  (RSS variation < 1 MB — flat)")
        return

    # Downsample to ~20 rows
    n_rows = min(20, len(samples))
    step = max(1, len(samples) // n_rows)

    for i in range(0, len(samples), step):
        t, rss = samples[i]
        bar_len = int((rss - min_rss) / rss_range * chart_width)
        bar = "█" * bar_len
        console.print(f"  {t:6.1f}s │{bar} {rss:.0f} MB")

    console.print(f"  {'':6s} └{'─' * chart_width}")
    console.print(f"  Peak RSS: {max_rss:.0f} MB  |  Range: {min_rss:.0f}–{max_rss:.0f} MB")


def print_sweep_table(results: list[ProfileResult]):
    """Print parameter sweep results table."""
    table = Table(title="Parameter Sweep: Memory Scaling", show_lines=True)
    table.add_column("Mode", justify="center")
    table.add_column("n_est", justify="center")
    table.add_column("n_trotter", justify="center")
    table.add_column("IR Scale", justify="right")
    table.add_column("Peak RSS (MB)", justify="right", style="bold")
    table.add_column("Compile Time (s)", justify="right")
    table.add_column("Prob Sum", justify="right")
    table.add_column("Status", justify="center")

    for r in results:
        mode_tag = "fixed" if r.mode == "fixed" else "dyn"
        if r.error:
            table.add_row(
                mode_tag,
                str(r.n_estimation_wires),
                str(r.n_trotter),
                str(r.ir_scale),
                "—",
                "—",
                "—",
                f"[red]{r.error}[/red]",
            )
        else:
            table.add_row(
                mode_tag,
                str(r.n_estimation_wires),
                str(r.n_trotter),
                str(r.ir_scale),
                f"{r.phase_b.maxrss_mb:.0f}" if r.phase_b else "—",
                f"{r.phase_b.elapsed_s:.1f}" if r.phase_b else "—",
                f"{r.prob_sum:.6f}",
                "[green]OK[/green]" if abs(r.prob_sum - 1.0) < 0.01 else "[red]BAD[/red]",
            )

    console.print(table)


def print_summary(result: ProfileResult):
    """Print summary panel answering Xanadu's question."""
    if result.phase_b is None:
        return

    peak_rss = result.phase_b.maxrss_mb
    compile_time = result.phase_b.elapsed_s
    n_terms = result.n_terms
    ir_scale = result.ir_scale
    mode_label = "H_fixed" if result.mode == "fixed" else "H_dynamic"

    # Find largest IR stage
    largest_stage = "N/A"
    largest_size = 0
    for stage, size_kb, _ in result.ir_analysis:
        if size_kb > largest_size:
            largest_size = size_kb
            largest_stage = stage

    lines = [
        f"[bold]Molecule:[/bold] {result.molecule}",
        f"[bold]Hamiltonian Mode:[/bold] {mode_label}",
        f"[bold]Peak Process RSS:[/bold] {peak_rss:.0f} MB ({peak_rss / 1024:.2f} GB)",
        f"[bold]Compilation Time:[/bold] {compile_time:.1f}s",
        f"[bold]Hamiltonian Terms:[/bold] {n_terms}",
        f"[bold]IR Scale:[/bold] {ir_scale} ({result.n_estimation_wires}×{result.n_trotter}×{n_terms})",
        f"[bold]Largest IR Stage:[/bold] {largest_stage} ({largest_size:.0f} KB)",
        f"[bold]Prob Sum (sanity):[/bold] {result.prob_sum:.6f}",
        "",
        "[bold]Key Finding:[/bold]",
    ]

    if peak_rss < 2048:  # < 2 GB
        lines.append(
            f"  Peak RSS = {peak_rss:.0f} MB — well below the 16 GB figure from the old "
            "double-inlining bug. The pre-compilation fix in orchestrator.py:656-673 "
            "is effective."
        )
    else:
        lines.append(
            f"  Peak RSS = {peak_rss:.0f} MB — significant memory consumption. "
            "Investigate IR amplification stages above."
        )

    console.print(Panel("\n".join(lines), title=f"Summary ({mode_label})", border_style="green"))


# =============================================================================
# Step 9: Main + CLI
# =============================================================================


def _print_single_result(result: ProfileResult):
    """Print full profiling report for a single mode result."""
    print_circuit_params(result)
    print_memory_table(result)
    print_ir_analysis(result.ir_analysis)
    print_memory_timeline(result.timeline_samples)
    print_summary(result)


def main():
    parser = argparse.ArgumentParser(
        description="QPE Circuit Compilation Memory Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--molecule",
        choices=["h2", "h3o"],
        default="h2",
        help="Target molecule (default: h2)",
    )
    parser.add_argument(
        "--mode",
        choices=["fixed", "dynamic", "both"],
        default="both",
        help="Hamiltonian mode: fixed (compile-time), dynamic (runtime), both (default: both)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run parameter scaling sweep (H2 only)",
    )
    parser.add_argument("--n-est", type=int, default=4, help="Estimation wires (default: 4)")
    parser.add_argument("--n-trotter", type=int, default=10, help="Trotter steps (default: 10)")
    parser.add_argument(
        "--ir-dir",
        type=str,
        default=None,
        help="Directory to save Catalyst IR stages (default: auto-cleanup after analysis)",
    )
    args = parser.parse_args()

    console.print(
        Panel(
            "[bold]QPE Circuit Compilation Memory Profiler[/bold]\n"
            "Measuring real peak memory during Catalyst @qjit compilation\n"
            "of QPE circuits (H_fixed and/or H_dynamic modes).",
            border_style="magenta",
        )
    )

    print_system_info()

    if args.sweep:
        console.print(f"\n[bold]Running parameter sweep (mode={args.mode})...[/bold]")
        if args.mode == "both":
            results_fixed = run_sweep(args.molecule, mode="fixed", ir_dir=args.ir_dir)
            results_dynamic = run_sweep(args.molecule, mode="dynamic", ir_dir=args.ir_dir)
            print_sweep_table(results_fixed + results_dynamic)
        else:
            results = run_sweep(args.molecule, mode=args.mode, ir_dir=args.ir_dir)
            print_sweep_table(results)
            for r in reversed(results):
                if not r.error:
                    print_ir_analysis(r.ir_analysis)
                    print_summary(r)
                    break
    elif args.mode == "both":
        console.print(
            f"\n[bold]Profiling {args.molecule.upper()}: "
            f"n_est={args.n_est}, n_trotter={args.n_trotter} (both modes)[/bold]"
        )
        result_fixed, result_dynamic = run_both_modes(
            args.molecule, args.n_est, args.n_trotter, ir_dir=args.ir_dir
        )

        # Print detailed reports for each mode
        for result in [result_fixed, result_dynamic]:
            if not result.error:
                console.print(f"\n{'=' * 70}")
                _print_single_result(result)

        # Print comparison if both succeeded
        if not result_fixed.error and not result_dynamic.error:
            console.print(f"\n{'=' * 70}")
            print_mode_comparison(result_fixed, result_dynamic)
    else:
        # Single mode (in-process)
        console.print(
            f"\n[bold]Profiling {args.molecule.upper()}: "
            f"n_est={args.n_est}, n_trotter={args.n_trotter} "
            f"(H_{args.mode})[/bold]"
        )

        tracemalloc.start()
        result = run_single_profile(
            args.molecule, args.n_est, args.n_trotter, mode=args.mode, ir_dir=args.ir_dir
        )
        tracemalloc.stop()

        if result.error:
            console.print(f"[red]ERROR: {result.error}[/red]")
            sys.exit(1)

        _print_single_result(result)


if __name__ == "__main__":
    main()
