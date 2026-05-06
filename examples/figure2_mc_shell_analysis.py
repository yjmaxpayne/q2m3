#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Generate manuscript Figure 2: MC convergence and finite-shell analysis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "output" / "h2_mc_structure_analysis"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "figure2_mc_shell_analysis.pdf"


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _col(rows: list[dict[str, str]], name: str) -> np.ndarray:
    """Extract a numeric column."""
    return np.array([float(row[name]) for row in rows], dtype=float)


def _configure_style() -> None:
    """Apply compact publication-style matplotlib settings."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_figure(
    *,
    running_csv: Path,
    block_csv: Path,
    radial_csv: Path,
    output: Path,
) -> Path:
    """Create the MC shell analysis figure."""
    _configure_style()
    import matplotlib.pyplot as plt

    running = _read_rows(running_csv)
    blocks = _read_rows(block_csv)
    radial = _read_rows(radial_csv)
    if not running or not blocks or not radial:
        raise ValueError("running, block, and radial CSV inputs must be non-empty")

    fig = plt.figure(figsize=(7.1, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.36)
    ax_run = fig.add_subplot(gs[0, :])
    ax_block = fig.add_subplot(gs[1, 0])
    ax_radial = fig.add_subplot(gs[1, 1])

    steps = _col(running, "step")
    delta_running_mha = _col(running, "delta_running_mean_ha") * 1000.0
    ax_run.plot(steps, delta_running_mha, color="#0072B2", lw=1.5, label="delta running mean")
    if "fixed_running_mean_ha" in running[0] and "dynamic_running_mean_ha" in running[0]:
        fixed = _col(running, "fixed_running_mean_ha")
        dynamic = _col(running, "dynamic_running_mean_ha")
        ax_run.plot(
            steps,
            (fixed - fixed[0]) * 1000.0,
            color="#D55E00",
            lw=1.0,
            ls="--",
            label="fixed shift",
        )
        ax_run.plot(
            steps,
            (dynamic - dynamic[0]) * 1000.0,
            color="#009E73",
            lw=1.0,
            ls=":",
            label="dynamic shift",
        )
    ax_run.axhline(0.0, color="#555555", lw=0.7)
    ax_run.set_xlabel("MC step")
    ax_run.set_ylabel("Energy running mean (mHa)")
    ax_run.set_title("A  H2 MC running mean")
    ax_run.grid(True, color="#dddddd", lw=0.6)
    ax_run.legend(frameon=False, ncols=3, loc="best")

    block_ids = _col(blocks, "block_id")
    means = _col(blocks, "mean_delta_ha") * 1000.0
    sem = _col(blocks, "sem_delta_ha") * 1000.0
    ax_block.errorbar(
        block_ids,
        means,
        yerr=sem,
        fmt="o-",
        color="#CC79A7",
        ecolor="#555555",
        capsize=3,
        lw=1.2,
        ms=4.0,
    )
    ax_block.axhline(float(np.mean(means)), color="#000000", lw=0.8, ls="--")
    ax_block.set_xlabel("Post-burn block")
    ax_block.set_ylabel("delta-corr-pol (mHa)")
    ax_block.set_title("B  Block means")
    ax_block.grid(True, color="#dddddd", lw=0.6)

    centers = _col(radial, "bin_center_a")
    density = _col(radial, "radial_density_per_a3")
    coordination = _col(radial, "cumulative_coordination")
    ax_radial.plot(centers, density, color="#0072B2", lw=1.5, label="radial density")
    ax_radial.set_xlabel("H2 center to water O distance (A)")
    ax_radial.set_ylabel("Finite-shell density (1/A^3)", color="#0072B2")
    ax_radial.tick_params(axis="y", labelcolor="#0072B2")
    ax_radial.grid(True, color="#dddddd", lw=0.6)
    ax_coord = ax_radial.twinx()
    ax_coord.plot(centers, coordination, color="#D55E00", lw=1.4, label="coordination")
    ax_coord.set_ylabel("Cumulative coordination", color="#D55E00")
    ax_coord.tick_params(axis="y", labelcolor="#D55E00")
    ax_radial.set_title("C  Finite-shell radial profile")

    for ax in (ax_run, ax_block, ax_radial):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_coord.spines["top"].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--running-csv", type=Path, default=None)
    parser.add_argument("--block-csv", type=Path, default=None)
    parser.add_argument("--radial-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    output = plot_figure(
        running_csv=args.running_csv or args.input_dir / "h2_delta_running_mean.csv",
        block_csv=args.block_csv or args.input_dir / "h2_delta_block_stats.csv",
        radial_csv=args.radial_csv or args.input_dir / "h2_shell_radial_profile.csv",
        output=args.output,
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
