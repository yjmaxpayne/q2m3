# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Tests for examples/collect_h2_water_count_convergence.py."""

import csv
import json

import numpy as np


def _fake_runner(**kwargs):
    """Return deterministic lightweight sampling output without PySCF/Catalyst."""
    n_waters = kwargs["n_waters"]
    n_steps = kwargs["n_mc_steps"]
    states = np.zeros((n_steps, n_waters, 6), dtype=float)
    for step in range(n_steps):
        for water in range(n_waters):
            states[step, water, 0] = 1.5 + 0.2 * water + 0.01 * step
    hf = -1.0 - 0.0005 * n_waters - np.linspace(0.0, 0.001, n_steps)
    total = hf + 0.0001 * n_waters
    return {
        "n_waters": n_waters,
        "n_mc_steps": n_steps,
        "random_seed": kwargs["random_seed"],
        "e_vacuum_ha": -1.0,
        "hf_energies": hf,
        "mm_energies": np.full(n_steps, 0.0001 * n_waters),
        "total_energies": total,
        "trajectory_solvent_states": states,
        "acceptance_rate": 0.5,
        "n_accepted": n_steps // 2,
    }


def test_main_writes_classical_qmmm_water_count_outputs(tmp_path):
    """Water-count collector writes explicit classical QM/MM convergence data."""
    from examples.collect_h2_water_count_convergence import main

    out_dir = tmp_path / "water_counts"
    main(
        [
            "--output-dir",
            str(out_dir),
            "--n-waters",
            "2,4",
            "--n-mc-steps",
            "6",
            "--write-trajectories",
        ],
        runner=_fake_runner,
    )

    csv_path = out_dir / "h2_water_count_convergence.csv"
    json_path = out_dir / "h2_water_count_convergence.json"
    trajectories = sorted(out_dir.glob("h2_hfmm_*waters_trajectory.csv"))

    assert csv_path.exists()
    assert json_path.exists()
    assert len(trajectories) == 2

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(row["n_waters"]) for row in rows] == [2, 4]
    assert rows[0]["data_class"] == "classical_qmmm_hf_mm"
    assert float(rows[1]["late_hf_solvation_shift_mha"]) < 0.0

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "does not run or claim full dynamic QPE" in payload["metadata"]["note"]
