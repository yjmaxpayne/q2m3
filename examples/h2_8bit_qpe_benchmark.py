#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
H2 QPE Resolution Benchmark: 4-bit vs 8-bit

Compares QPE estimation register size (4 vs 8 bits) across:
  - Compilation time (Phase A/B IR cache)
  - Per-QPE execution time
  - Phase resolution (bin width)
  - σ_QPE shot noise characterization

Findings:
  - 8-bit fixed: 13.3x per-QPE slowdown, 16x resolution gain
  - 8-bit dynamic: safe at n_trotter ≤ 5 (OOM at ≥ 7 on 30GB)
  - σ_QPE reduction requires shots ∝ 2^n_estimation_wires

Usage:
    OMP_NUM_THREADS=4 uv run python examples/h2_8bit_qpe_benchmark.py
    OMP_NUM_THREADS=4 uv run python examples/h2_8bit_qpe_benchmark.py --json
"""

import argparse
import json
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from q2m3.solvation import MoleculeConfig, QPEConfig, SolvationConfig, run_solvation  # noqa: E402

H2 = MoleculeConfig(
    name="H2",
    symbols=["H", "H"],
    coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    charge=0,
    active_electrons=2,
    active_orbitals=2,
    basis="sto-3g",
)


def run_benchmark(
    n_est: int, n_mc: int, n_shots: int, mode: str = "fixed", n_trotter: int = 10
) -> dict:
    """Run a single benchmark configuration and return metrics."""
    config = SolvationConfig(
        molecule=H2,
        qpe_config=QPEConfig(
            n_estimation_wires=n_est,
            n_trotter_steps=n_trotter,
            n_shots=n_shots,
            target_resolution=0.003 if n_est == 4 else 0.0002,
            energy_range=0.2,
        ),
        hamiltonian_mode=mode,
        n_mc_steps=n_mc,
        n_waters=10,
        temperature=300.0,
        random_seed=42,
        verbose=False,
    )
    t0 = time.perf_counter()
    result = run_solvation(config, show_plots=False)
    wall = time.perf_counter() - t0

    timing = result.get("timing")
    qt = timing.quantum_times if timing else np.array([])
    q_e = np.array(result.get("quantum_energies", []))
    valid = q_e[~np.isnan(q_e)]
    cache = result.get("cache_stats", {})

    return {
        "n_est": n_est,
        "n_trotter": n_trotter,
        "mode": mode,
        "n_mc": n_mc,
        "n_shots": n_shots,
        "wall_s": round(wall, 2),
        "compile_s": round(timing.quantum_compile_time, 2) if timing else 0,
        "per_qpe_ms": round(float(np.mean(qt)) * 1000, 2) if len(qt) > 0 else 0,
        "mean_energy": round(float(np.mean(valid)), 6) if len(valid) > 0 else None,
        "std_energy": round(float(np.std(valid)), 6) if len(valid) > 0 else None,
        "sigma_kcal": round(float(np.std(valid)) * 627.509, 4) if len(valid) > 0 else None,
        "resolution_mha": round(0.2 * 1000 / (2**n_est), 1),
        "n_bins": 2**n_est,
        "cache_hit": cache.get("is_cache_hit", False),
    }


def main():
    parser = argparse.ArgumentParser(description="4-bit vs 8-bit QPE benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    parser.add_argument("--mc-steps", type=int, default=50, help="MC steps per run")
    args = parser.parse_args()

    results = []

    # 1. Analytical mode comparison (n_shots=0)
    print("Phase 1: Analytical mode (n_shots=0)", file=sys.stderr)
    results.append(run_benchmark(4, args.mc_steps, 0, "fixed"))
    results.append(run_benchmark(8, args.mc_steps, 0, "fixed"))

    # 2. Shots mode comparison (n_shots=50)
    print("Phase 2: Shots mode (n_shots=50)", file=sys.stderr)
    results.append(run_benchmark(4, args.mc_steps, 50, "fixed"))
    results.append(run_benchmark(8, args.mc_steps, 50, "fixed"))

    # 3. Dynamic mode (8-bit, n_trotter=5)
    print("Phase 3: Dynamic mode (8-bit, n_trotter=5)", file=sys.stderr)
    results.append(run_benchmark(8, args.mc_steps, 0, "dynamic", n_trotter=5))

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    else:
        print(f"\n{'='*72}")
        print("  H2 QPE RESOLUTION BENCHMARK: 4-bit vs 8-bit")
        print(f"{'='*72}")
        hdr = f"{'bits':>4} {'mode':<8} {'shots':>5} {'trotter':>7} {'per_qpe':>9} {'σ_QPE':>10} {'res':>8} {'bins':>5}"
        print(hdr)
        print("-" * 72)
        for r in results:
            sigma = f"{r['sigma_kcal']:.2f}" if r["sigma_kcal"] and r["sigma_kcal"] > 0 else "0.00"
            print(
                f"{r['n_est']:>4} {r['mode']:<8} {r['n_shots']:>5} {r['n_trotter']:>7} "
                f"{r['per_qpe_ms']:>8.1f}ms {sigma:>8} {r['resolution_mha']:>6.1f} {r['n_bins']:>5}"
            )

        print("\nKey: σ_QPE in kcal/mol, res in mHa/bin")
        print("Note: σ_QPE with shots requires n_shots ∝ 2^n_est for fair comparison")


if __name__ == "__main__":
    main()
