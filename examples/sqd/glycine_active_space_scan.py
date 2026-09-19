"""Run CAS(6,6), (8,8), (10,10), each with five seeds in serial fresh interpreters."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from examples.sqd._presentation import ROOT, execute, save_summary, write_json


def calculate(args, directory) -> None:
    """Retain every requested point, including failures, before returning status."""
    rows = []
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        JAX_PLATFORMS="cpu",
    )
    request = dict(
        active_spaces=args.active_spaces,
        seeds=list(range(args.seed, args.seed + 5)),
        shots=args.shots,
        n_reps=2,
        wall_budget_s=args.wall_budget,
        rss_budget_decimal_mb=args.rss_budget,
        schedule="serial fresh interpreter",
    )
    write_json(directory / "input.json", request)
    for active in args.active_spaces:
        for seed in request["seeds"]:
            label = f"cas{active}-seed{seed}"
            target = directory / label
            command = [
                sys.executable,
                "-m",
                "examples.sqd.glycine_ground_state",
                "--active-space",
                str(active),
                "--seed",
                str(seed),
                "--shots",
                str(args.shots),
                "--wall-budget",
                str(args.wall_budget),
                "--rss-budget",
                str(args.rss_budget),
                "--output",
                str(target),
                "--no-plots",
            ]
            print(f"Running {label} ({len(rows)+1}/{5*len(args.active_spaces)})", flush=True)
            with (directory / f"{label}.log").open("w") as log:
                process = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            if process.returncode == 0 and (target / "summary.json").is_file():
                row = json.loads((target / "summary.json").read_text())[0]
                row["label"] = label
                row["result_file"] = f"{label}/result.json"
                print(
                    f"  {row['baseline_tier']} / {row['baseline_method']}: "
                    f"SQD-reference={row['delta_mHa']:.6f} mHa; "
                    f"space {row['subspace_dim']}/{row['full_ci_dim']}",
                    flush=True,
                )
            else:
                failure = target / "failure.json"
                reason = (
                    json.loads(failure.read_text())["reason"]
                    if failure.exists()
                    else (f"worker exit {process.returncode}; inspect {label}.log")
                )
                row = dict(
                    label=label,
                    status="failed",
                    active_electrons=active,
                    active_orbitals=active,
                    seed=seed,
                    shots=args.shots,
                    reason=reason,
                )
                print(f"  FAILED: {reason}", flush=True)
            rows.append(row)
            save_summary(directory, rows, plots=False)
    save_summary(directory, rows)
    print(
        "All points are actual runs; seed spread is descriptive, not a confidence interval. "
        "Each active space has its own Hamiltonian/reference. Errors need not decrease with size."
    )
    if any(row["status"] != "completed" for row in rows):
        raise RuntimeError("Scan incomplete; partial results and failure rows retained")


def main() -> int:
    """Parse the scan request; --seed chooses the first of five consecutive seeds."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shots", type=int, default=100_000)
    p.add_argument("--active-spaces", type=int, nargs="+", choices=(6, 8, 10), default=[6, 8, 10])
    p.add_argument("--wall-budget", type=float, default=900.0)
    p.add_argument("--rss-budget", type=float, default=8192.0)
    args = p.parse_args()
    if args.seed < 0 or args.shots < 1 or len(set(args.active_spaces)) != len(args.active_spaces):
        p.error("Use a nonnegative seed, positive shots and distinct active spaces")
    return execute(args, "glycine-scan", calculate)


if __name__ == "__main__":
    raise SystemExit(main())
