#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Collect H2 MC convergence and finite-shell structure-analysis tables.

This script reuses existing H2 trajectory and three-mode delta-corr-pol CSV
artifacts. It does not rerun QPE. Outputs are intended as manuscript support
data for MC convergence, finite-shell radial density, coordination, and
representative solute + solvent-shell snapshots.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from q2m3.solvation.config import MoleculeConfig
from q2m3.solvation.structure_analysis import (
    coordination_by_cutoff,
    load_state_trajectory_csv,
    radial_density_profile,
    radial_profile_to_rows,
    select_random_snapshot_indices,
    shell_counts_per_frame,
    validate_state_trajectory,
    write_xyz_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_TRAJECTORY_CSV = OUTPUT_DIR / "h2_trotter5_dynamic_mc_trajectory.csv"
DEFAULT_DELTA_CSV = OUTPUT_DIR / "h2_three_mode_delta_corr_pol.csv"
DEFAULT_ANALYSIS_DIR = OUTPUT_DIR / "h2_mc_structure_analysis"

H2_MOLECULE = MoleculeConfig(
    name="H2",
    symbols=["H", "H"],
    coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    charge=0,
    active_electrons=2,
    active_orbitals=2,
    basis="sto-3g",
)


@dataclass(frozen=True)
class DeltaTrajectory:
    """Aligned fixed/dynamic/delta energy trajectory."""

    steps: np.ndarray
    fixed_energy_ha: np.ndarray | None
    dynamic_energy_ha: np.ndarray | None
    delta_corr_pol_ha: np.ndarray


def _parse_csv_floats(value: str) -> list[float]:
    """Parse a comma-separated float list."""
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _first_present(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    """Return the first present non-empty value for a set of possible CSV names."""
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    return None


def load_delta_trajectory(path: Path) -> DeltaTrajectory:
    """Load fixed/dynamic and delta-corr-pol data from a flexible CSV schema."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} contains no delta-corr-pol rows")

    steps: list[int] = []
    fixed_values: list[float] = []
    dynamic_values: list[float] = []
    delta_values: list[float] = []
    has_fixed = True
    has_dynamic = True
    has_delta = True

    fixed_names = (
        "fixed_energy_ha",
        "qpe_fixed_ha",
        "fixed_qpe_energy_ha",
        "quantum_energy_fixed_ha",
        "fixed",
    )
    dynamic_names = (
        "dynamic_energy_ha",
        "qpe_dynamic_ha",
        "dynamic_qpe_energy_ha",
        "quantum_energy_dynamic_ha",
        "dynamic",
    )
    delta_names = ("delta_corr_pol_ha", "delta_ha", "delta", "dynamic_minus_fixed_ha")

    for index, row in enumerate(rows):
        step_raw = _first_present(row, ("step", "mc_step", "qpe_step", "index"))
        steps.append(int(float(step_raw)) if step_raw is not None else index)

        fixed_raw = _first_present(row, fixed_names)
        dynamic_raw = _first_present(row, dynamic_names)
        delta_raw = _first_present(row, delta_names)

        if fixed_raw is None:
            has_fixed = False
            fixed_values.append(float("nan"))
        else:
            fixed_values.append(float(fixed_raw))
        if dynamic_raw is None:
            has_dynamic = False
            dynamic_values.append(float("nan"))
        else:
            dynamic_values.append(float(dynamic_raw))
        if delta_raw is None:
            has_delta = False
            delta_values.append(float("nan"))
        else:
            delta_values.append(float(delta_raw))

    fixed = np.array(fixed_values, dtype=float) if has_fixed else None
    dynamic = np.array(dynamic_values, dtype=float) if has_dynamic else None
    delta = np.array(delta_values, dtype=float) if has_delta else None
    if delta is None:
        if fixed is None or dynamic is None:
            raise ValueError(
                f"{path} must contain delta_corr_pol_ha or both fixed and dynamic energies"
            )
        delta = dynamic - fixed

    return DeltaTrajectory(
        steps=np.array(steps, dtype=int),
        fixed_energy_ha=fixed,
        dynamic_energy_ha=dynamic,
        delta_corr_pol_ha=delta,
    )


def _running_mean(values: np.ndarray) -> np.ndarray:
    """Return the cumulative running mean of finite values."""
    finite = np.isfinite(values)
    cumulative = np.cumsum(np.where(finite, values, 0.0))
    counts = np.cumsum(finite.astype(int))
    out = np.full_like(values, np.nan, dtype=float)
    mask = counts > 0
    out[mask] = cumulative[mask] / counts[mask]
    return out


def running_mean_rows(delta: DeltaTrajectory) -> list[dict[str, Any]]:
    """Return CSV rows for fixed/dynamic/delta running means."""
    delta_running = _running_mean(delta.delta_corr_pol_ha)
    fixed_running = (
        _running_mean(delta.fixed_energy_ha) if delta.fixed_energy_ha is not None else None
    )
    dynamic_running = (
        _running_mean(delta.dynamic_energy_ha) if delta.dynamic_energy_ha is not None else None
    )
    rows = []
    for i, step in enumerate(delta.steps):
        row: dict[str, Any] = {
            "step": int(step),
            "delta_corr_pol_ha": float(delta.delta_corr_pol_ha[i]),
            "delta_running_mean_ha": float(delta_running[i]),
        }
        if delta.fixed_energy_ha is not None and fixed_running is not None:
            row["fixed_energy_ha"] = float(delta.fixed_energy_ha[i])
            row["fixed_running_mean_ha"] = float(fixed_running[i])
        if delta.dynamic_energy_ha is not None and dynamic_running is not None:
            row["dynamic_energy_ha"] = float(delta.dynamic_energy_ha[i])
            row["dynamic_running_mean_ha"] = float(dynamic_running[i])
        rows.append(row)
    return rows


def block_statistics_rows(
    delta: DeltaTrajectory,
    *,
    burn_in: int,
    n_blocks: int,
) -> list[dict[str, Any]]:
    """Return post-burn block mean rows for delta-corr-pol."""
    values = delta.delta_corr_pol_ha[burn_in:]
    steps = delta.steps[burn_in:]
    if len(values) == 0:
        raise ValueError("burn-in removes all delta-corr-pol samples")
    n_blocks = max(1, min(n_blocks, len(values)))
    value_blocks = np.array_split(values, n_blocks)
    step_blocks = np.array_split(steps, n_blocks)
    rows = []
    for block_id, (block, block_steps) in enumerate(zip(value_blocks, step_blocks, strict=True)):
        std = float(np.std(block, ddof=1)) if len(block) > 1 else 0.0
        rows.append(
            {
                "block_id": block_id,
                "step_start": int(block_steps[0]),
                "step_end": int(block_steps[-1]),
                "n_samples": int(len(block)),
                "mean_delta_ha": float(np.mean(block)),
                "std_delta_ha": std,
                "sem_delta_ha": float(std / np.sqrt(len(block))) if len(block) > 0 else 0.0,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV using union field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    trajectory_csv: Path,
    delta_csv: Path,
    output_dir: Path,
    *,
    burn_in_fraction: float = 0.2,
    burn_in_steps: int | None = None,
    bin_width: float = 0.25,
    r_max: float = 8.0,
    coordination_cutoffs: list[float] | None = None,
    n_blocks: int = 5,
    n_snapshots: int = 3,
    snapshot_cutoff: float = 3.5,
    snapshot_tail_window_steps: int = 300,
    snapshot_random_seed: int = 42,
) -> dict[str, Any]:
    """Run H2 MC structure analysis and write all manuscript-support outputs."""
    states = validate_state_trajectory(load_state_trajectory_csv(trajectory_csv))
    delta = load_delta_trajectory(delta_csv)
    burn_in = (
        int(burn_in_steps)
        if burn_in_steps is not None
        else int(np.floor(states.shape[0] * burn_in_fraction))
    )
    burn_in = max(0, min(burn_in, states.shape[0] - 1))
    cutoffs = np.array(coordination_cutoffs or [2.5, 3.0, 3.5, 4.0, 4.5, 5.0], dtype=float)

    output_dir.mkdir(parents=True, exist_ok=True)
    running_path = output_dir / "h2_delta_running_mean.csv"
    block_path = output_dir / "h2_delta_block_stats.csv"
    radial_path = output_dir / "h2_shell_radial_profile.csv"
    coordination_path = output_dir / "h2_coordination_cutoff.csv"
    summary_path = output_dir / "h2_mc_structure_analysis_summary.json"

    _write_csv(running_path, running_mean_rows(delta))
    blocks = block_statistics_rows(
        delta, burn_in=min(burn_in, len(delta.steps) - 1), n_blocks=n_blocks
    )
    _write_csv(block_path, blocks)

    bin_edges = np.arange(0.0, r_max + 0.5 * bin_width, bin_width)
    profile = radial_density_profile(states, H2_MOLECULE.coords_array, bin_edges, start=burn_in)
    _write_csv(radial_path, radial_profile_to_rows(profile))

    coordination = coordination_by_cutoff(states, H2_MOLECULE.coords_array, cutoffs, start=burn_in)
    coordination_rows = [
        {"cutoff_a": float(cutoff), "mean_coordination": float(count)}
        for cutoff, count in zip(cutoffs, coordination, strict=True)
    ]
    _write_csv(coordination_path, coordination_rows)

    snapshot_start = max(burn_in, states.shape[0] - max(1, snapshot_tail_window_steps))
    snapshot_indices = select_random_snapshot_indices(
        states,
        n_snapshots=n_snapshots,
        start=snapshot_start,
        stop=states.shape[0],
        random_seed=snapshot_random_seed,
    )
    snapshot_paths = []
    shell_counts = shell_counts_per_frame(
        states, H2_MOLECULE.coords_array, snapshot_cutoff, start=0
    )
    for old_path in output_dir.glob("h2_shell_snapshot_step*.xyz"):
        old_path.unlink()
    for index in snapshot_indices:
        path = output_dir / f"h2_shell_snapshot_step{index:04d}.xyz"
        write_xyz_snapshot(
            path,
            H2_MOLECULE.symbols,
            H2_MOLECULE.coords_array,
            states[index],
            comment=(
                f"H2 finite-shell snapshot step={index}; "
                f"shell_count_{snapshot_cutoff:.2f}A={int(shell_counts[index])}"
            ),
        )
        snapshot_paths.append(path)

    post_delta = delta.delta_corr_pol_ha[min(burn_in, len(delta.steps) - 1) :]
    summary = {
        "metadata": {
            "schema": "h2_mc_structure_analysis.v1",
            "trajectory_csv": str(trajectory_csv),
            "delta_csv": str(delta_csv),
            "note": "Finite-shell radial density/RDF-like profile; no bulk periodic density assumed.",
            "snapshot_selection": (
                "random_tail_window"
                f"(tail_window_steps={snapshot_tail_window_steps}, seed={snapshot_random_seed})"
            ),
        },
        "n_steps": int(states.shape[0]),
        "n_waters": int(states.shape[1]),
        "burn_in_steps": int(burn_in),
        "snapshot_tail_window_steps": int(snapshot_tail_window_steps),
        "snapshot_random_seed": int(snapshot_random_seed),
        "delta_post_burn_mean_ha": float(np.mean(post_delta)),
        "delta_post_burn_sem_ha": float(
            np.std(post_delta, ddof=1) / np.sqrt(len(post_delta)) if len(post_delta) > 1 else 0.0
        ),
        "snapshot_indices": [int(index) for index in snapshot_indices],
        "outputs": {
            "running_mean_csv": str(running_path),
            "block_stats_csv": str(block_path),
            "radial_profile_csv": str(radial_path),
            "coordination_csv": str(coordination_path),
            "snapshots_xyz": [str(path) for path in snapshot_paths],
        },
        "blocks": blocks,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    return {**summary, "summary_path": str(summary_path)}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-csv", type=Path, default=DEFAULT_TRAJECTORY_CSV)
    parser.add_argument("--delta-csv", type=Path, default=DEFAULT_DELTA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--burn-in-fraction", type=float, default=0.2)
    parser.add_argument("--burn-in-steps", type=int, default=None)
    parser.add_argument("--bin-width", type=float, default=0.25)
    parser.add_argument("--r-max", type=float, default=8.0)
    parser.add_argument("--coordination-cutoffs", default="2.5,3.0,3.5,4.0,4.5,5.0")
    parser.add_argument("--n-blocks", type=int, default=5)
    parser.add_argument("--n-snapshots", type=int, default=3)
    parser.add_argument("--snapshot-cutoff", type=float, default=3.5)
    parser.add_argument("--snapshot-tail-window-steps", type=int, default=300)
    parser.add_argument("--snapshot-random-seed", type=int, default=42)
    args = parser.parse_args(argv)

    summary = run_analysis(
        args.trajectory_csv,
        args.delta_csv,
        args.output_dir,
        burn_in_fraction=args.burn_in_fraction,
        burn_in_steps=args.burn_in_steps,
        bin_width=args.bin_width,
        r_max=args.r_max,
        coordination_cutoffs=_parse_csv_floats(args.coordination_cutoffs),
        n_blocks=args.n_blocks,
        n_snapshots=args.n_snapshots,
        snapshot_cutoff=args.snapshot_cutoff,
        snapshot_tail_window_steps=args.snapshot_tail_window_steps,
        snapshot_random_seed=args.snapshot_random_seed,
    )
    print(f"wrote {summary['summary_path']}")
    for key, value in summary["outputs"].items():
        if key == "snapshots_xyz":
            for path in value:
                print(f"wrote {path}")
        else:
            print(f"wrote {value}")


if __name__ == "__main__":
    main()
