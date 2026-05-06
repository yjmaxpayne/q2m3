# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Tests for examples/collect_h2_depth5_replay_dataset.py."""

import csv
import json

import numpy as np


def test_main_writes_depth5_replay_dataset(tmp_path, monkeypatch):
    """The collector writes the accepted-path trajectory and paired replay CSVs."""
    from examples import collect_h2_depth5_replay_dataset as script

    trajectory = np.zeros((4, 2, 6), dtype=float)
    trajectory[:, 0, 0] = [1.0, 1.1, 1.2, 1.3]
    trajectory[:, 1, 0] = [3.0, 3.1, 3.2, 3.3]

    def fake_run_solvation(config, show_plots=False):
        return {
            "trajectory_solvent_states": trajectory,
            "acceptance_rate": 0.5,
        }

    def fake_replay(config, states):
        is_dynamic = config.hamiltonian_mode == "dynamic"
        return {
            "quantum_energies": np.array([-1.0, -1.1, -1.2, -1.3]) + (0.01 if is_dynamic else 0.0),
            "hf_energies": np.array([-0.9, -0.91, -0.92, -0.93]),
        }

    monkeypatch.setattr(script, "run_solvation", fake_run_solvation)
    monkeypatch.setattr(script, "replay_quantum_trajectory", fake_replay)

    trajectory_csv = tmp_path / "trajectory.csv"
    delta_csv = tmp_path / "delta.csv"
    summary_json = tmp_path / "summary.json"
    script.main(
        [
            "--n-mc-steps",
            "4",
            "--n-waters",
            "2",
            "--trajectory-csv",
            str(trajectory_csv),
            "--delta-csv",
            str(delta_csv),
            "--summary-json",
            str(summary_json),
        ]
    )

    assert trajectory_csv.exists()
    assert delta_csv.exists()
    assert summary_json.exists()

    with delta_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["delta_corr_pol_ha"] == "0.01"
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["metadata"]["n_trotter_steps"] == 5
