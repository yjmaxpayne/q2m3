# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Tests for examples/collect_h2_mc_structure_analysis.py."""

import csv
import json

import numpy as np

from q2m3.solvation.structure_analysis import write_state_trajectory_csv, xyz_atom_count


def _write_trajectory(tmp_path):
    """Write a small H2 solvent trajectory fixture."""
    states = np.zeros((6, 2, 6), dtype=float)
    states[:, 0, 0] = np.linspace(1.2, 1.8, 6)
    states[:, 1, 0] = np.linspace(3.0, 3.8, 6)
    path = tmp_path / "h2_trotter5_dynamic_mc_trajectory.csv"
    write_state_trajectory_csv(path, states)
    return path


def _write_delta(tmp_path):
    """Write a paired fixed/dynamic delta fixture."""
    path = tmp_path / "h2_three_mode_delta_corr_pol.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "fixed_energy_ha", "dynamic_energy_ha"],
        )
        writer.writeheader()
        for step in range(6):
            writer.writerow(
                {
                    "step": step,
                    "fixed_energy_ha": -1.10 - 0.001 * step,
                    "dynamic_energy_ha": -1.095 - 0.0015 * step,
                }
            )
    return path


def test_main_writes_structure_analysis_tables_and_xyz(tmp_path):
    """Collector writes convergence, RDF-like shell, coordination, and XYZ outputs."""
    from examples.collect_h2_mc_structure_analysis import main

    out_dir = tmp_path / "analysis"
    main(
        [
            "--trajectory-csv",
            str(_write_trajectory(tmp_path)),
            "--delta-csv",
            str(_write_delta(tmp_path)),
            "--output-dir",
            str(out_dir),
            "--burn-in-steps",
            "1",
            "--bin-width",
            "1.0",
            "--r-max",
            "6.0",
            "--n-snapshots",
            "2",
            "--snapshot-tail-window-steps",
            "3",
            "--snapshot-random-seed",
            "11",
        ]
    )

    running = out_dir / "h2_delta_running_mean.csv"
    blocks = out_dir / "h2_delta_block_stats.csv"
    radial = out_dir / "h2_shell_radial_profile.csv"
    coordination = out_dir / "h2_coordination_cutoff.csv"
    summary = out_dir / "h2_mc_structure_analysis_summary.json"
    snapshots = sorted(out_dir.glob("h2_shell_snapshot_step*.xyz"))

    assert running.exists()
    assert blocks.exists()
    assert radial.exists()
    assert coordination.exists()
    assert summary.exists()
    assert len(snapshots) == 2
    assert xyz_atom_count(snapshots[0]) == 8

    with radial.open(newline="", encoding="utf-8") as f:
        radial_rows = list(csv.DictReader(f))
    assert "radial_density_per_a3" in radial_rows[0]
    assert "cumulative_coordination" in radial_rows[0]

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["metadata"]["schema"] == "h2_mc_structure_analysis.v1"
    assert payload["burn_in_steps"] == 1
    assert payload["snapshot_indices"]
    assert payload["snapshot_tail_window_steps"] == 3
    assert payload["snapshot_random_seed"] == 11
    assert all(index >= 3 for index in payload["snapshot_indices"])
