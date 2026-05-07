#!/usr/bin/env python3
# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Catalyst @qjit Performance Benchmark

Demonstrates when Catalyst JIT compilation provides speedup:
- Single execution: JIT overhead can dominate (no speedup expected)
- Multi-execution: compile once, reuse many times
- VQE-style loops: amortize compilation across repeated circuit evaluations

Key finding: For single QPE executions, JIT compilation overhead often
equals or exceeds execution time. Significant speedup only appears in
multi-iteration workflows (VQE optimization, MC loops).

Usage:
    uv run python examples/catalyst_benchmark.py
"""

import time

import numpy as np

from q2m3.core import QuantumQMMM
from q2m3.core.device_utils import (
    HAS_CATALYST,
    HAS_JAX_CUDA,
    HAS_LIGHTNING_GPU,
    get_best_available_device,
    get_catalyst_effective_backend,
)

# QPE parameters (same defaults as h2_qpe_validation.py)
DEFAULT_N_ESTIMATION_WIRES = 4
DEFAULT_N_TROTTER_STEPS = 20
DEFAULT_N_SHOTS = 100

# H2 geometry (bond length 0.74 Angstrom)
H2_COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


def _build_qmmm(device_type: str, use_catalyst: bool) -> QuantumQMMM:
    """Create a QuantumQMMM instance for H2 with 2 TIP3P waters."""
    from q2m3.core.qmmm_system import Atom

    qm_atoms = [Atom("H", H2_COORDS[0]), Atom("H", H2_COORDS[1])]
    return QuantumQMMM(
        qm_atoms=qm_atoms,
        mm_waters=2,
        qpe_config={
            "use_real_qpe": True,
            "n_estimation_wires": DEFAULT_N_ESTIMATION_WIRES,
            "base_time": "auto",
            "n_trotter_steps": DEFAULT_N_TROTTER_STEPS,
            "n_shots": DEFAULT_N_SHOTS,
            "active_electrons": 2,
            "active_orbitals": 2,
            "device_type": device_type,
            "use_catalyst": use_catalyst,
        },
    )


def benchmark_single(device_type: str) -> dict:
    """Part A: Single execution comparison (demonstrates JIT overhead).

    Catalyst's compilation cost is ~equal to execution time for small circuits,
    so single execution shows no speedup. This is expected behavior.
    """
    print("[Part A] Single Execution (demonstrates JIT overhead)")
    print("-" * 70)
    print(f"Device: {device_type}")
    print()

    # A1: Standard QPE (baseline, no JIT)
    print("  A1. Standard QPE (no @qjit)...")
    start = time.perf_counter()
    result_standard = _build_qmmm(device_type, use_catalyst=False).compute_ground_state()
    time_standard = time.perf_counter() - start
    print(f"      Energy: {result_standard['energy']:.6f} Ha, Time: {time_standard:.3f} s")

    # A2: Catalyst QPE (includes JIT compilation time in first call)
    catalyst_backend = get_catalyst_effective_backend()
    print("  A2. Catalyst QPE (@qjit, includes compilation)...")
    start = time.perf_counter()
    qmmm_catalyst = _build_qmmm(device_type, use_catalyst=True)
    result_catalyst = qmmm_catalyst.compute_ground_state()
    time_catalyst_first = time.perf_counter() - start
    print(f"      Energy: {result_catalyst['energy']:.6f} Ha, Time: {time_catalyst_first:.3f} s")
    print()

    speedup = time_standard / time_catalyst_first if time_catalyst_first > 0 else 0
    print("  Single Execution Result:")
    print(f"    Standard:     {time_standard:.3f} s")
    print(f"    Catalyst:     {time_catalyst_first:.3f} s (includes JIT compilation)")
    if speedup >= 1:
        print(f"    Speedup:      {speedup:.2f}x")
    else:
        print(f"    Slowdown:     {1 / speedup:.2f}x (expected due to JIT overhead)")
    print()

    return {
        "device_type": device_type,
        "catalyst_backend": catalyst_backend,
        "time_standard": time_standard,
        "time_catalyst_first": time_catalyst_first,
        "single_speedup": speedup,
        "energy_standard": result_standard["energy"],
        "energy_catalyst": result_catalyst["energy"],
    }


def benchmark_multi(device_type: str, n_iterations: int = 5) -> dict:
    """Part B: Multi-execution comparison (demonstrates Catalyst advantage).

    Catalyst compiles once, then reuses the compiled circuit for subsequent
    calls. This amortizes JIT cost over many executions.
    """
    print(f"[Part B] Multi-Execution Benchmark ({n_iterations} runs, reuses compiled circuit)")
    print("-" * 70)
    print("  This demonstrates Catalyst's advantage: compile once, execute many times.")
    print()

    # B1: Standard QPE — each execution is fully independent
    print(f"  B1. Standard QPE x{n_iterations} (each run is independent)...")
    start = time.perf_counter()
    for _ in range(n_iterations):
        qm_atoms_std = _build_qmmm(device_type, use_catalyst=False)
        qm_atoms_std.compute_ground_state()
    time_multi_standard = time.perf_counter() - start
    print(
        f"      Total time: {time_multi_standard:.3f} s "
        f"({time_multi_standard / n_iterations:.3f} s/iter)"
    )

    # B2: Catalyst QPE — first call compiles, subsequent calls reuse
    print(f"  B2. Catalyst QPE x{n_iterations} (reuses compiled circuit)...")
    start = time.perf_counter()
    qmmm_cat = _build_qmmm(device_type, use_catalyst=True)
    qmmm_cat.compute_ground_state()  # First call: includes JIT compilation
    for _ in range(n_iterations - 1):
        qmmm_cat.compute_ground_state()  # Subsequent calls: uses cached compilation
    time_multi_catalyst = time.perf_counter() - start
    print(
        f"      Total time: {time_multi_catalyst:.3f} s "
        f"({time_multi_catalyst / n_iterations:.3f} s/iter)"
    )
    print()

    multi_speedup = time_multi_standard / time_multi_catalyst if time_multi_catalyst > 0 else 0
    print("  Multi-Execution Result:")
    print(f"    Standard:     {time_multi_standard:.3f} s total")
    print(f"    Catalyst:     {time_multi_catalyst:.3f} s total")
    if multi_speedup >= 1:
        print(f"    Speedup:      {multi_speedup:.2f}x faster with Catalyst")
    else:
        print(f"    Ratio:        {1 / multi_speedup:.2f}x")
    print()

    return {
        "n_iterations": n_iterations,
        "time_multi_standard": time_multi_standard,
        "time_multi_catalyst": time_multi_catalyst,
        "multi_speedup": multi_speedup,
    }


def print_summary(single: dict, multi: dict) -> None:
    """Part C: Summary and interpretation."""
    print("[Part C] Summary and Interpretation")
    print("-" * 70)

    print("  Why no speedup for single execution?")
    print("  ------------------------------------")
    print("  Catalyst's JIT compilation has overhead that ~equals execution time for")
    print("  small circuits. The speedup appears when:")
    print("    1. Reusing compiled circuits across multiple executions")
    print("    2. Running optimization loops with qml.for_loop()")
    print("    3. Computing gradients with catalyst.value_and_grad()")
    print()
    print("  For VQE-style workloads (100+ iterations), expect 10-50x speedup.")
    print("  Reference: https://pennylane.ai/qml/demos/how_to_catalyst_lightning_gpu")
    print()

    # Energy comparison (verify Catalyst gives same result)
    e_diff = abs(single["energy_standard"] - single["energy_catalyst"])
    print("  Energy Comparison (Standard vs Catalyst):")
    print(f"    Standard QPE:  {single['energy_standard']:.6f} Ha")
    print(f"    Catalyst QPE:  {single['energy_catalyst']:.6f} Ha")
    print(f"    Difference:    {e_diff:.6f} Ha")
    if e_diff < 0.01:
        print("    Status: Results consistent (diff < 0.01 Ha)")
    else:
        print("    Status: Results differ (stochastic QPE sampling)")
    print()

    print("  Performance Summary:")
    print(f"    Device:           {single['device_type']}")
    print(f"    Catalyst Backend: {single['catalyst_backend']}")
    has_gpu = HAS_LIGHTNING_GPU and HAS_JAX_CUDA
    print(f"    GPU Acceleration: {'enabled' if has_gpu else 'disabled (CPU fallback)'}")
    print(
        f"    Single speedup:   {single['single_speedup']:.2f}x "
        f"(includes JIT compilation overhead)"
    )
    print(
        f"    Multi speedup:    {multi['multi_speedup']:.2f}x "
        f"({multi['n_iterations']} iterations, compile-once pattern)"
    )


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    print("=" * 70)
    print("       Catalyst @qjit QPE Performance Benchmark")
    print("=" * 70)
    print()

    if not HAS_CATALYST:
        print("WARNING: pennylane-catalyst is not installed.")
        print("To enable Catalyst support, install with:")
        print("  pip install pennylane-catalyst")
        print()
        print("Skipping benchmark...")
        return

    device_type = get_best_available_device()
    if not HAS_JAX_CUDA and HAS_LIGHTNING_GPU:
        print(f"Note: Catalyst runs on CPU (JAX lacks CUDA) with device: {device_type}")
    print()

    # Part A: Single execution (shows JIT overhead)
    single = benchmark_single(device_type)

    # Part B: Multi-execution (shows compile-once advantage)
    multi = benchmark_multi(device_type)

    # Part C: Summary
    print_summary(single, multi)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
