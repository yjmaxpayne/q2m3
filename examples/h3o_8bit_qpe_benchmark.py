#!/usr/bin/env python3
# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
H3O+ QPE Resolution Benchmark: 4-bit vs 8-bit (fixed mode)

Companion experiment to h2_8bit_qpe_benchmark.py, scaling from H2 to the
larger active-space H3O+ Hamiltonian to validate that the 8-bit resolution
gain transfers to a charged ionic system.

Configuration:
  - Molecule: H3O+ (STO-3G), active space (4e, 4o) → 8 system qubits
  - Solvent: 10 TIP3P waters, ion-dipole interaction (initial_water_distance=3.5 Å)
  - MC: 50 steps (lighter than the production 100 to keep wall time bounded)
  - Mode: fixed (Catalyst-compatible Hamiltonian; runtime coefficients deferred
          per discovery in dev/wiki/catalyst-runtime-coefficients.md)

Metrics captured per configuration:
  - Wall clock time (s)
  - Compile time (Phase A IR build + Phase B JIT) — from TimingData
  - Per-QPE execution time (mean of step times)
  - σ_QPE (energy stddev across MC steps, kcal/mol) — proxy for shot noise
  - Peak RSS (process self + Catalyst children, MB) via /proc and rusage
  - Resolution (mHa/bin) and bin count (2^n_estimation_wires)

OOM safety:
  - 8-bit run wrapped in try/except. If RSS > 22 GB during compile or any
    MemoryError / subprocess failure → automatic downgrade to 6-bit.
  - The script reports measured RSS so memory behavior comes from the run.

Usage:
    OMP_NUM_THREADS=8 uv run python examples/h3o_8bit_qpe_benchmark.py
    OMP_NUM_THREADS=8 uv run python examples/h3o_8bit_qpe_benchmark.py --json
    OMP_NUM_THREADS=8 uv run python examples/h3o_8bit_qpe_benchmark.py --skip-8bit  # debug
"""

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from typing import Literal, cast

os.environ.setdefault("OMP_NUM_THREADS", "8")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from q2m3.profiling.memory import read_proc_status, take_snapshot  # noqa: E402
from q2m3.solvation import (  # noqa: E402
    MoleculeConfig,
    QPEConfig,
    SolvationConfig,
    run_solvation,
)

# H3O+ pyramidal geometry (C3v) — same as production h3o_mc_solvation.py
H3OP = MoleculeConfig(
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
)

# Memory safety threshold (MB) — abort 8-bit if combined physical RSS crosses this.
# Use self+children RSS (RUSAGE), NOT VmPeak (virtual): LLVM JIT routinely
# reserves 20+ GB of virtual mmap on a 30 GB system without using it.
RSS_ABORT_MB = 22 * 1024  # 22 GB physical, leaves ~8 GB headroom on 30 GB box


def run_one(
    n_est: int,
    n_mc: int,
    n_shots: int,
    mode: str = "fixed",
    n_trotter: int = 10,
    label: str = "",
) -> dict:
    """Run one benchmark config; capture timing + memory + statistics."""
    config = SolvationConfig(
        molecule=H3OP,
        qpe_config=QPEConfig(
            n_estimation_wires=n_est,
            n_trotter_steps=n_trotter,
            n_shots=n_shots,
            qpe_interval=10,
            target_resolution=0.003 if n_est == 4 else 0.0002,
            energy_range=0.2,
        ),
        hamiltonian_mode=cast(Literal["hf_corrected", "fixed", "dynamic"], mode),
        n_mc_steps=n_mc,
        n_waters=10,
        temperature=300.0,
        translation_step=0.3,
        rotation_step=0.2618,
        initial_water_distance=3.5,
        random_seed=42,
        verbose=False,
    )

    snap_pre = take_snapshot(f"{label}_pre")
    t0 = time.perf_counter()
    result = run_solvation(config, show_plots=False)
    wall = time.perf_counter() - t0
    snap_post = take_snapshot(f"{label}_post")

    timing = result.get("timing")
    qt = timing.quantum_times if timing else np.array([])
    qt_active = qt[qt > 0] if len(qt) > 0 else qt
    q_e = np.array(result.get("quantum_energies", []), dtype=float)
    valid = q_e[~np.isnan(q_e)] if q_e.size else q_e
    cache = result.get("cache_stats", {})

    return {
        "label": label,
        "n_est": n_est,
        "n_trotter": n_trotter,
        "mode": mode,
        "n_mc": n_mc,
        "n_shots": n_shots,
        "wall_s": round(wall, 2),
        "compile_s": round(timing.quantum_compile_time, 2) if timing else 0.0,
        "per_qpe_ms": round(float(np.mean(qt_active)) * 1000, 2) if qt_active.size else 0.0,
        "n_qpe_steps": int(qt_active.size),
        "mean_energy_ha": round(float(np.mean(valid)), 6) if valid.size else None,
        "std_energy_ha": round(float(np.std(valid)), 6) if valid.size else None,
        "sigma_kcal": round(float(np.std(valid)) * 627.509, 4) if valid.size else None,
        "resolution_mha": round(0.2 * 1000 / (2**n_est), 4),
        "n_bins": 2**n_est,
        "cache_hit": cache.get("is_cache_hit", False),
        # Memory: peak across self + children (catalyst child handles MLIR→LLVM)
        "vm_peak_mb": round(snap_post.vm_peak_mb, 1),
        "rss_self_peak_mb": round(snap_post.maxrss_mb, 1),
        "rss_children_peak_mb": round(snap_post.maxrss_children_mb, 1),
        "rss_total_peak_mb": round(snap_post.maxrss_mb + snap_post.maxrss_children_mb, 1),
        "rss_delta_mb": round(snap_post.rss_mb - snap_pre.rss_mb, 1),
    }


def safe_run(label: str, **kwargs) -> dict:
    """Run with OOM/error guard. Returns a result dict or an error stub."""
    print(f"\n[{label}] starting…", file=sys.stderr, flush=True)
    pre = read_proc_status()
    print(
        f"[{label}] pre-run VmRSS={pre['VmRSS']:.0f} MB, VmPeak={pre['VmPeak']:.0f} MB",
        file=sys.stderr,
        flush=True,
    )
    try:
        rec = run_one(label=label, **kwargs)
        print(
            f"[{label}] done: wall={rec['wall_s']}s, compile={rec['compile_s']}s, "
            f"per_qpe={rec['per_qpe_ms']}ms, peak_rss={rec['vm_peak_mb']:.0f}MB",
            file=sys.stderr,
            flush=True,
        )
        if rec["rss_total_peak_mb"] > RSS_ABORT_MB:
            rec["warning"] = f"physical RSS (self+children) exceeded {RSS_ABORT_MB}MB threshold"
            print(f"[{label}] WARN: {rec['warning']}", file=sys.stderr, flush=True)
        return rec
    except (MemoryError, RuntimeError, Exception) as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"[{label}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return {
            "label": label,
            "n_est": kwargs.get("n_est"),
            "mode": kwargs.get("mode"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": tb.splitlines()[-5:],
        }


def main():
    parser = argparse.ArgumentParser(description="H3O+ 4-bit vs 8-bit QPE benchmark")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    parser.add_argument(
        "--out",
        default="data/output/h3o_8bit_qpe_benchmark.json",
        help="Persist JSON results here (relative to project root)",
    )
    parser.add_argument(
        "--mc-steps", type=int, default=50, help="MC steps per run (default for all)"
    )
    parser.add_argument(
        "--mc-steps-8bit",
        type=int,
        default=0,
        help="Override MC steps for 8-bit only (0 = use --mc-steps)",
    )
    parser.add_argument(
        "--skip-4bit", action="store_true", help="Skip 4-bit (when reusing existing data)"
    )
    parser.add_argument("--skip-8bit", action="store_true", help="Skip 8-bit (smoke test only)")
    parser.add_argument(
        "--run-6bit", action="store_true", help="Always run 6-bit fallback datapoint"
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Do not auto-downgrade 8-bit failure to 6-bit",
    )
    args = parser.parse_args()

    results: list[dict] = []

    # 4-bit baseline (fixed mode, analytical n_shots=0)
    if not args.skip_4bit:
        results.append(safe_run("4bit_fixed_analytical", n_est=4, n_mc=args.mc_steps, n_shots=0))

    if not args.skip_8bit:
        # 8-bit experiment (may use a smaller mc-steps for smoke test)
        n_mc_8bit = args.mc_steps_8bit if args.mc_steps_8bit else args.mc_steps
        rec_8 = safe_run("8bit_fixed_analytical", n_est=8, n_mc=n_mc_8bit, n_shots=0)
        results.append(rec_8)
        if "error" in rec_8 and not args.no_fallback:
            print("[fallback] 8-bit failed, retrying with 6-bit", file=sys.stderr, flush=True)
            results.append(
                safe_run("6bit_fixed_analytical", n_est=6, n_mc=args.mc_steps, n_shots=0)
            )

    if args.run_6bit:
        # Always-on 6-bit datapoint (used when 8-bit OOM is suspected and we
        # need a fallback record without waiting for 8-bit failure)
        results.append(safe_run("6bit_fixed_analytical", n_est=6, n_mc=args.mc_steps, n_shots=0))

    # Persist
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "molecule": "H3O+",
                "active_space": "(4e, 4o) → 8 system qubits",
                "n_waters": 10,
                "n_mc_steps": args.mc_steps,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n[persist] wrote {out_path}", file=sys.stderr)

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
        return

    # Pretty table
    print(f"\n{'=' * 92}")
    print("  H3O+ QPE RESOLUTION BENCHMARK: 4-bit vs 8-bit (fixed mode)")
    print(f"{'=' * 92}")
    hdr = (
        f"{'bits':>4} {'mode':<7} {'wall':>7} {'compile':>8} {'per_qpe':>9} "
        f"{'σ_QPE':>9} {'resol':>7} {'bins':>5} {'VmPeak':>9} {'children':>9}"
    )
    print(hdr)
    print("-" * 92)
    for r in results:
        if "error" in r:
            print(f"{r['n_est']:>4} {r['mode']:<7} ERROR: {r['error']}")
            continue
        sigma = f"{r['sigma_kcal']:.2f}" if r["sigma_kcal"] and r["sigma_kcal"] > 0 else "0.00"
        print(
            f"{r['n_est']:>4} {r['mode']:<7} {r['wall_s']:>6.1f}s {r['compile_s']:>7.1f}s "
            f"{r['per_qpe_ms']:>8.1f}ms {sigma:>7} kcal {r['resolution_mha']:>5.2f}m "
            f"{r['n_bins']:>5} {r['vm_peak_mb']:>7.0f}MB {r['rss_children_peak_mb']:>7.0f}MB"
        )

    print("\nKey:")
    print("  per_qpe   = mean QPE eval time across MC steps with QPE measurement")
    print("  σ_QPE     = std of QPE energies (kcal/mol) — proxy for shot noise + MC drift")
    print("  resol     = phase resolution (mHa/bin) = energy_range / 2^n_est")
    print("  VmPeak    = process peak virtual memory (Linux /proc)")
    print("  children  = peak RSS of Catalyst MLIR→LLVM child compilers (rusage)")


if __name__ == "__main__":
    main()
