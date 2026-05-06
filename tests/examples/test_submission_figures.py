# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Smoke tests for manuscript figure scripts."""

import csv

import numpy as np

from q2m3.solvation.structure_analysis import write_xyz_snapshot


def _assert_nonempty(path):
    """Assert generated figure exists and is non-empty."""
    assert path.exists()
    assert path.stat().st_size > 1000


def test_figure2_mc_shell_analysis_smoke(tmp_path):
    """Figure 2 script writes a non-empty PDF from compact CSV fixtures."""
    from examples.figure2_mc_shell_analysis import main

    running = tmp_path / "running.csv"
    with running.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "delta_running_mean_ha",
                "fixed_running_mean_ha",
                "dynamic_running_mean_ha",
            ],
        )
        writer.writeheader()
        for step in range(8):
            writer.writerow(
                {
                    "step": step,
                    "delta_running_mean_ha": 0.001 * (step + 1),
                    "fixed_running_mean_ha": -1.0 - 0.0002 * step,
                    "dynamic_running_mean_ha": -0.999 + 0.0001 * step,
                }
            )

    blocks = tmp_path / "blocks.csv"
    with blocks.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["block_id", "mean_delta_ha", "sem_delta_ha"],
        )
        writer.writeheader()
        for block in range(4):
            writer.writerow(
                {
                    "block_id": block,
                    "mean_delta_ha": 0.001 + 0.0002 * block,
                    "sem_delta_ha": 0.00005,
                }
            )

    radial = tmp_path / "radial.csv"
    with radial.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bin_center_a", "radial_density_per_a3", "cumulative_coordination"],
        )
        writer.writeheader()
        for i in range(1, 7):
            writer.writerow(
                {
                    "bin_center_a": i,
                    "radial_density_per_a3": 0.01 * i,
                    "cumulative_coordination": 0.5 * i,
                }
            )

    output = tmp_path / "figure2.pdf"
    main(
        [
            "--running-csv",
            str(running),
            "--block-csv",
            str(blocks),
            "--radial-csv",
            str(radial),
            "--output",
            str(output),
        ]
    )
    _assert_nonempty(output)


def test_figure3_solvent_shell_snapshots_smoke(tmp_path):
    """Figure 3 script writes a non-empty PDF with ASE rendering."""
    from examples.figure3_solvent_shell_snapshots import main

    for idx in range(2):
        states = np.zeros((2, 6), dtype=float)
        states[0, :3] = [1.5 + idx, 0.0, 0.0]
        states[1, :3] = [3.0 + idx, 0.5, 0.0]
        write_xyz_snapshot(
            tmp_path / f"h2_shell_snapshot_step{idx:04d}.xyz",
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
            states,
            comment=f"snapshot {idx}",
        )

    output = tmp_path / "figure3.pdf"
    main(
        [
            "--input-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--renderer",
            "ase",
        ]
    )
    _assert_nonempty(output)


def test_figure3_projection_legend_layout():
    """Figure 3 projection legend stays shared and horizontal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from examples.figure3_solvent_shell_snapshots import _plot_projection_legend

    fig, ax = plt.subplots()
    _plot_projection_legend(ax)

    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["water O", "H2 centroid"]
    assert legend._ncols == 2
    assert not ax.axison

    plt.close(fig)


def test_figure4_catalyst_ion_matrix_smoke(tmp_path):
    """Figure 4 script writes a non-empty PDF from a compact matrix fixture."""
    from examples.figure4_catalyst_ion_matrix import main

    matrix = tmp_path / "ion_catalyst_matrix.csv"
    fieldnames = [
        "label",
        "reference_mc_steps",
        "fixed_status",
        "dynamic_status",
        "fixed_compile_rss_gb",
        "dynamic_compile_rss_gb",
        "fixed_compile_time_s",
        "dynamic_compile_time_s",
        "timing_n_iterations",
        "fixed_no_catalyst_repeated_qpe_s",
        "fixed_catalyst_repeated_qpe_s",
        "dynamic_no_catalyst_repeated_qpe_s",
        "dynamic_catalyst_repeated_qpe_s",
        "fixed_no_catalyst_reference_mc_s",
        "fixed_compile_once_catalyst_reference_mc_s",
        "fixed_compile_once_speedup_reference_mc",
        "dynamic_no_catalyst_reference_mc_s",
        "dynamic_compile_once_catalyst_reference_mc_s",
        "dynamic_compile_once_speedup_reference_mc",
        "boundary_note",
    ]
    with matrix.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "label": "H2",
                    "reference_mc_steps": 1000,
                    "fixed_status": "measured",
                    "dynamic_status": "measured",
                    "fixed_compile_rss_gb": 1.0,
                    "dynamic_compile_rss_gb": 3.0,
                    "fixed_compile_time_s": 10.0,
                    "dynamic_compile_time_s": 30.0,
                    "timing_n_iterations": 5,
                    "fixed_no_catalyst_repeated_qpe_s": 12.0,
                    "fixed_catalyst_repeated_qpe_s": 4.0,
                    "dynamic_no_catalyst_repeated_qpe_s": 18.0,
                    "dynamic_catalyst_repeated_qpe_s": 5.0,
                    "fixed_no_catalyst_reference_mc_s": 2400.0,
                    "fixed_compile_once_catalyst_reference_mc_s": 22.0,
                    "fixed_compile_once_speedup_reference_mc": 109.1,
                    "dynamic_no_catalyst_reference_mc_s": 3600.0,
                    "dynamic_compile_once_catalyst_reference_mc_s": 40.0,
                    "dynamic_compile_once_speedup_reference_mc": 90.0,
                    "boundary_note": "",
                },
                {
                    "label": "H3O+",
                    "reference_mc_steps": 1000,
                    "fixed_status": "measured_h3o_mc",
                    "dynamic_status": "boundary_unmeasured",
                    "fixed_compile_rss_gb": 5.0,
                    "dynamic_compile_rss_gb": "",
                    "fixed_compile_time_s": 2.0,
                    "dynamic_compile_time_s": "",
                    "timing_n_iterations": "",
                    "fixed_no_catalyst_repeated_qpe_s": "",
                    "fixed_catalyst_repeated_qpe_s": "",
                    "dynamic_no_catalyst_repeated_qpe_s": "",
                    "dynamic_catalyst_repeated_qpe_s": "",
                    "fixed_no_catalyst_reference_mc_s": "",
                    "fixed_compile_once_catalyst_reference_mc_s": "",
                    "fixed_compile_once_speedup_reference_mc": "",
                    "dynamic_no_catalyst_reference_mc_s": "",
                    "dynamic_compile_once_catalyst_reference_mc_s": "",
                    "dynamic_compile_once_speedup_reference_mc": "",
                    "boundary_note": "boundary",
                },
            ]
        )

    output = tmp_path / "figure4.pdf"
    main(["--matrix-csv", str(matrix), "--output", str(output)])
    _assert_nonempty(output)
