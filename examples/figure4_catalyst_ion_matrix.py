#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Generate manuscript Figure 4: ion Catalyst compile matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "output" / "ion_catalyst_matrix" / "ion_catalyst_matrix.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "output" / "ion_catalyst_matrix" / "figure4_catalyst_ion_matrix.pdf"
)
MODES = ("fixed", "dynamic")


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: str | None) -> float:
    """Convert CSV values to float, returning NaN for blanks."""
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _configure_style() -> None:
    """Apply compact publication style."""
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


def _matrix(rows: list[dict[str, str]], suffix: str) -> np.ndarray:
    """Build a system x mode matrix for a numeric suffix."""
    return np.array(
        [[_to_float(row.get(f"{mode}_{suffix}")) for mode in MODES] for row in rows],
        dtype=float,
    )


def _statuses(rows: list[dict[str, str]]) -> list[list[str]]:
    """Build system x mode status strings."""
    return [[row.get(f"{mode}_status", "") for mode in MODES] for row in rows]


def _status_label(status: str) -> str:
    """Compact status label for heatmap annotation."""
    if status == "measured":
        return "meas."
    if status == "measured_h3o_mc":
        return "meas.*"
    if "boundary" in status:
        return "bound."
    if status == "not_measured":
        return "n.m."
    return status[:7] if status else "n/a"


def _plot_heatmap(ax, matrix: np.ndarray, rows: list[dict[str, str]], title: str, cbar_label: str):
    """Plot one annotated heatmap."""
    import matplotlib.pyplot as plt

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#E6E6E6")
    finite = matrix[np.isfinite(matrix)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(MODES)), MODES)
    ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
    ax.set_title(title, loc="left")
    statuses = _statuses(rows)
    for i in range(len(rows)):
        for j in range(len(MODES)):
            value = matrix[i, j]
            text = f"{value:.2g}" if np.isfinite(value) else _status_label(statuses[i][j])
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if np.isfinite(value) else "#333333",
            )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(cbar_label)


def _mode_bar_rows(
    rows: list[dict[str, str]],
    *,
    no_catalyst_suffix: str,
    catalyst_suffix: str,
) -> list[dict[str, float | str]]:
    """Flatten rows into per-mode bar data for timing/model plots."""
    plot_rows: list[dict[str, float | str]] = []
    for row in rows:
        for mode in MODES:
            no_catalyst = _to_float(row.get(f"{mode}_{no_catalyst_suffix}"))
            catalyst = _to_float(row.get(f"{mode}_{catalyst_suffix}"))
            if np.isfinite(no_catalyst) and np.isfinite(catalyst):
                plot_rows.append(
                    {
                        "label": row["label"],
                        "mode": mode,
                        "tick": f'{row["label"]}\n{mode}',
                        "no_catalyst": no_catalyst,
                        "catalyst": catalyst,
                        "speedup": (
                            _to_float(row.get(f"{mode}_compile_once_speedup_reference_mc"))
                            if catalyst_suffix == "compile_once_catalyst_reference_mc_s"
                            else _to_float(row.get(f"{mode}_catalyst_speedup"))
                        ),
                    }
                )
    return plot_rows


def _plot_grouped_bars(
    ax,
    plot_rows: list[dict[str, float | str]],
    *,
    title: str,
    catalyst_label: str,
    ylabel: str,
    annotate_speedup: bool = True,
) -> None:
    """Plot paired no-Catalyst / Catalyst grouped bars."""
    ax.set_title(title, loc="left")
    if not plot_rows:
        ax.text(
            0.5,
            0.5,
            "optional timing rows\nnot measured",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#555555",
        )
        ax.set_axis_off()
        return

    labels = [str(row["tick"]) for row in plot_rows]
    x = np.arange(len(labels))
    width = 0.35
    standard = np.array([float(row["no_catalyst"]) for row in plot_rows])
    catalyst = np.array([float(row["catalyst"]) for row in plot_rows])
    ax.bar(x - width / 2, standard, width, color="#D55E00", label="no Catalyst")
    ax.bar(x + width / 2, catalyst, width, color="#0072B2", label=catalyst_label)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color="#dddddd", lw=0.6)
    if not annotate_speedup:
        return
    ymax = max(np.max(standard), np.max(catalyst)) if len(plot_rows) else 0.0
    for xpos, row in zip(x, plot_rows, strict=True):
        speedup = float(row["speedup"])
        if np.isfinite(speedup):
            ax.text(xpos, ymax * 1.02, f"{speedup:.1f}x", ha="center", va="bottom", fontsize=6.5)
    ax.set_ylim(top=ymax * 1.18 if ymax > 0.0 else 1.0)


def _plot_timing(ax, rows: list[dict[str, str]]) -> None:
    """Plot repeated-QPE timing controls on the standardized circuit."""
    plot_rows = _mode_bar_rows(
        rows,
        no_catalyst_suffix="no_catalyst_repeated_qpe_s",
        catalyst_suffix="catalyst_repeated_qpe_s",
    )
    n_iterations = next(
        (
            int(float(row["timing_n_iterations"]))
            for row in rows
            if row.get("timing_n_iterations") not in (None, "")
        ),
        0,
    )
    title = f"C  {n_iterations}x repeated QPE" if n_iterations else "C  Repeated QPE"
    _plot_grouped_bars(
        ax,
        plot_rows,
        title=title,
        catalyst_label="Catalyst post-compile",
        ylabel="Wall time (s)",
    )


def _plot_mc_model(ax, rows: list[dict[str, str]]) -> None:
    """Plot 1000-step MC wall-time model from no-Catalyst and compile-once Catalyst totals."""
    plot_rows = _mode_bar_rows(
        rows,
        no_catalyst_suffix="no_catalyst_reference_mc_s",
        catalyst_suffix="compile_once_catalyst_reference_mc_s",
    )
    reference_mc_steps = next(
        (
            int(float(row["reference_mc_steps"]))
            for row in rows
            if row.get("reference_mc_steps") not in (None, "")
        ),
        0,
    )
    title = (
        f"D  {reference_mc_steps}-step MC model" if reference_mc_steps else "D  MC wall-time model"
    )
    _plot_grouped_bars(
        ax,
        plot_rows,
        title=title,
        catalyst_label="Catalyst compile-once",
        ylabel="Modeled wall time (s)",
    )


def plot_figure(matrix_csv: Path, output: Path) -> Path:
    """Generate the Catalyst ion matrix figure."""
    rows = _read_rows(matrix_csv)
    if not rows:
        raise ValueError(f"{matrix_csv} contains no rows")
    _configure_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.3, 6.1))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38)
    ax_rss = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_timing = fig.add_subplot(gs[1, 0])
    ax_mc = fig.add_subplot(gs[1, 1])

    _plot_heatmap(ax_rss, _matrix(rows, "compile_rss_gb"), rows, "A  Compile RSS", "GB")
    _plot_heatmap(ax_time, _matrix(rows, "compile_time_s"), rows, "B  Compile time", "s")
    _plot_timing(ax_timing, rows)
    _plot_mc_model(ax_mc, rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = plot_figure(args.matrix_csv, args.output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
