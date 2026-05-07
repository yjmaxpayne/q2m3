#!/usr/bin/env python3
# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""IR-QRE correlation analysis using the standardized trotter-5 compile survey.

This script joins measured Catalyst/MLIR compilation-resource data from
``ir_qre_trotter5_compile_survey.json`` with fault-tolerant quantum resource
estimates from ``qre_survey.json``. All compile rows must use 4 estimation wires
and ``n_trotter=5``; older fixed-10 and reduced-shape dynamic rows are rejected.

Usage:
    OMP_NUM_THREADS=4 uv run python examples/ir_qre_correlation_analysis.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QRE_JSON = PROJECT_ROOT / "data" / "output" / "qre_survey.json"
DEFAULT_COMPILE_JSON = PROJECT_ROOT / "data" / "output" / "ir_qre_trotter5_compile_survey.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
STANDARD_N_TROTTER = 5


@dataclass(frozen=True)
class Dimension:
    """One planned IR-QRE correlation dimension."""

    dimension_id: str
    title: str
    x_key: str
    y_key: str
    x_label: str
    y_label: str
    filename_slug: str
    log_axes: bool = True


@dataclass(frozen=True)
class DimensionStats:
    """Regression and correlation statistics for one dimension."""

    dimension_id: str
    title: str
    n_points: int
    power_law_formula: str
    power_law_alpha: float | None
    power_law_exponent: float | None
    power_law_r_squared: float | None
    power_law_p_value: float | None
    pearson_r: float | None
    pearson_p_value: float | None
    spearman_r: float | None
    spearman_p_value: float | None
    conclusion: str
    outlier_note: str


DIMENSIONS = [
    Dimension(
        dimension_id="D1",
        title="IR ops vs Toffoli gates",
        x_key="ir_ops_lower_bound",
        y_key="toffoli_gates",
        x_label="IR ops lower bound (n_est x 5 trotter x n_terms)",
        y_label="Toffoli gates",
        filename_slug="ir_ops_vs_toffoli",
    ),
    Dimension(
        dimension_id="D2",
        title="Compile RSS vs logical qubits",
        x_key="compile_rss_gb",
        y_key="logical_qubits",
        x_label="Compile RSS peak (GB)",
        y_label="Logical qubits",
        filename_slug="compile_rss_vs_logical_qubits",
    ),
    Dimension(
        dimension_id="D3",
        title="Compile time vs T-depth",
        x_key="compile_time_s",
        y_key="t_depth",
        x_label="Compile time (s)",
        y_label="T-depth",
        filename_slug="compile_time_vs_t_depth",
    ),
    Dimension(
        dimension_id="D4",
        title="Bufferization amplification vs Hamiltonian 1-norm",
        x_key="bufferization_amp",
        y_key="hamiltonian_1norm_Ha",
        x_label="BufferizationStage amplification ratio",
        y_label="Hamiltonian 1-norm lambda (Ha)",
        filename_slug="bufferization_amp_vs_lambda",
    ),
]


def load_qre_records(path: Path) -> dict[str, dict[str, Any]]:
    """Load successful QRE records keyed by label."""
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    records: dict[str, dict[str, Any]] = {}
    for record in payload.get("systems", []):
        if record.get("toffoli_gates") is None:
            continue
        records[record["label"]] = record
    return records


def _validate_trotter5(record: dict[str, Any], path: Path) -> None:
    """Reject nonstandard compile rows instead of silently mixing shapes."""
    n_trotter = record.get("n_trotter")
    if n_trotter != STANDARD_N_TROTTER:
        label = record.get("label", "<unknown>")
        raise ValueError(f"{path}:{label} must use n_trotter=5; got {n_trotter}")


def load_compile_rows(path: Path = DEFAULT_COMPILE_JSON) -> list[dict[str, Any]]:
    """Load measured rows from the standardized trotter-5 compile survey."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    metadata_trotter = payload.get("metadata", {}).get("n_trotter")
    if metadata_trotter is not None and metadata_trotter != STANDARD_N_TROTTER:
        raise ValueError(f"{path} metadata must use n_trotter=5; got {metadata_trotter}")

    rows: list[dict[str, Any]] = []
    for record in payload.get("systems", []):
        _validate_trotter5(record, path)
        if record.get("classical_status") != "measured":
            continue
        rows.append(
            {
                "label": record["label"],
                "classical_case": record["classical_case"],
                "classical_mode": record["classical_mode"],
                "classical_status": "measured",
                "n_system_qubits": record.get("n_system_qubits"),
                "n_terms": record["n_terms"],
                "n_estimation_wires": record["n_estimation_wires"],
                "n_trotter": record["n_trotter"],
                "total_qubits": record.get("total_qubits"),
                "ir_ops_lower_bound": record["ir_ops_lower_bound"],
                "compile_rss_gb": record["compile_rss_gb"],
                "compile_time_s": record["compile_time_s"],
                "bufferization_amp": record.get("bufferization_amp"),
                "classical_provenance": f"{path}:{record['classical_case']}",
            }
        )
    return rows


def _join_classical_qre(
    classical_rows: list[dict[str, Any]], qre_json: Path
) -> list[dict[str, Any]]:
    """Join measured compile rows with successful QRE records."""
    qre_records = load_qre_records(qre_json)
    rows: list[dict[str, Any]] = []
    for classical in classical_rows:
        qre = qre_records.get(classical["label"])
        if qre is None:
            continue
        rows.append(
            {
                **classical,
                "basis": qre.get("basis"),
                "active_electrons": qre.get("active_electrons"),
                "active_orbitals": qre.get("active_orbitals"),
                "n_system_qubits": qre.get("n_system_qubits", classical.get("n_system_qubits")),
                "logical_qubits": qre.get("logical_qubits"),
                "toffoli_gates": qre.get("toffoli_gates"),
                "t_depth": qre.get("t_depth"),
                "hamiltonian_1norm_Ha": qre.get("hamiltonian_1norm_Ha"),
                "qre_provenance": str(qre_json),
            }
        )
    return rows


def build_correlation_rows(
    qre_json: Path = DEFAULT_QRE_JSON,
    compile_json: Path = DEFAULT_COMPILE_JSON,
) -> list[dict[str, Any]]:
    """Join measured trotter-5 compilation rows with QRE records."""
    return _join_classical_qre(load_compile_rows(compile_json), qre_json)


def build_case_rows(
    qre_json: Path = DEFAULT_QRE_JSON,
    compile_json: Path = DEFAULT_COMPILE_JSON,
) -> list[dict[str, Any]]:
    """Return case-level rows; currently identical to the measured summary."""
    return build_correlation_rows(qre_json, compile_json)


def _finite_pairs(
    rows: list[dict[str, Any]], dimension: Dimension
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Extract finite x/y pairs for a dimension."""
    valid_rows: list[dict[str, Any]] = []
    x_vals: list[float] = []
    y_vals: list[float] = []
    for row in rows:
        x_raw = row.get(dimension.x_key)
        y_raw = row.get(dimension.y_key)
        if x_raw is None or y_raw is None:
            continue
        x = float(x_raw)
        y = float(y_raw)
        if math.isfinite(x) and math.isfinite(y):
            valid_rows.append(row)
            x_vals.append(x)
            y_vals.append(y)
    return np.array(x_vals, dtype=float), np.array(y_vals, dtype=float), valid_rows


def _safe_ols(
    x: np.ndarray, y: np.ndarray
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return slope/intercept/R2/p for transformed-space OLS."""
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None, None, None, None
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    ss_x = float(np.sum(x_centered**2))
    ss_y = float(np.sum(y_centered**2))
    if ss_x == 0.0 or ss_y == 0.0:
        return None, None, None, None

    ss_xy = float(np.sum(x_centered * y_centered))
    slope = ss_xy / ss_x
    intercept = float(np.mean(y)) - slope * float(np.mean(x))
    r_value = ss_xy / math.sqrt(ss_x * ss_y)
    r_squared = max(0.0, min(1.0, r_value**2))
    if r_squared >= 1.0:
        p_value = 0.0
    else:
        degrees_freedom = len(x) - 2
        t_stat = abs(r_value) * math.sqrt(degrees_freedom / (1.0 - r_squared))
        p_value = float(2.0 * scipy_stats.t.sf(t_stat, degrees_freedom))
    return float(slope), float(intercept), float(r_squared), p_value


def _safe_power_law(
    x: np.ndarray, y: np.ndarray
) -> tuple[float | None, float | None, float | None, float | None]:
    """Fit y = alpha * x ** beta in log-log space."""
    mask = (x > 0) & (y > 0)
    x_pos = x[mask]
    y_pos = y[mask]
    if len(x_pos) < 3 or np.unique(x_pos).size < 2 or np.unique(y_pos).size < 2:
        return None, None, None, None
    beta, log_alpha, r_squared, p_value = _safe_ols(np.log10(x_pos), np.log10(y_pos))
    if beta is None or log_alpha is None:
        return None, None, None, None
    return 10**log_alpha, beta, r_squared, p_value


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float | None, float | None]:
    """Return Pearson or Spearman correlation, tolerating constant inputs."""
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None, None
    if method == "pearson":
        result = scipy_stats.pearsonr(x, y)
    elif method == "spearman":
        result = scipy_stats.spearmanr(x, y)
    else:
        raise ValueError(f"unknown correlation method: {method}")
    return float(result.statistic), float(result.pvalue)


def _format_float(value: float | None, precision: int = 3) -> str:
    """Format optional floats for report text."""
    if value is None:
        return "n/a"
    if value == 0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 10000:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def _power_formula(alpha: float | None, exponent: float | None) -> str:
    """Format y = a * x^b with concrete fitted coefficients."""
    if alpha is None or exponent is None:
        return "n/a"
    return f"y = {_format_float(alpha)} * x^{_format_float(exponent)}"


def _conclusion(r_squared: float | None, p_value: float | None, n_points: int) -> str:
    """Classify association strength with explicit small-sample caveat."""
    if r_squared is None or p_value is None:
        return "undefined statistic; zero variance or insufficient independent variation"
    if r_squared >= 0.80 and p_value < 0.05:
        return f"strong descriptive association (n={n_points}); treat as hypothesis-generating"
    if r_squared >= 0.40:
        return f"partial/weak association (n={n_points}); mode/provenance effects remain important"
    return f"null/weak result (n={n_points}); QRE alone is not predictive"


def _power_outlier_note(
    x: np.ndarray,
    y: np.ndarray,
    rows: list[dict[str, Any]],
    alpha: float | None,
    exponent: float | None,
) -> str:
    """Identify large residuals in the power fit and summarize likely cause."""
    if alpha is None or exponent is None or len(x) < 3:
        return "No residual outlier test; power fit is undefined for this dimension."
    mask = (x > 0) & (y > 0)
    x_pos = x[mask]
    y_pos = y[mask]
    rows_pos = [row for row, keep in zip(rows, mask, strict=False) if bool(keep)]
    predicted = alpha * np.power(x_pos, exponent)
    residuals = np.log10(y_pos) - np.log10(predicted)
    std = float(np.std(residuals))
    if std == 0.0:
        return "No residual outliers; all points lie on the power fit."
    z_scores = np.abs(residuals / std)
    idx = int(np.argmax(z_scores))
    row = rows_pos[idx]
    return (
        f"Largest residual is {row['classical_case']} "
        f"({row['classical_status']}, z={z_scores[idx]:.2f})."
    )


def compute_dimension_stats(rows: list[dict[str, Any]]) -> list[DimensionStats]:
    """Compute power-law, Pearson, and Spearman statistics on measured rows."""
    stats_rows: list[DimensionStats] = []
    measured_rows = [row for row in rows if row.get("classical_status") == "measured"]
    for dimension in DIMENSIONS:
        x, y, valid_rows = _finite_pairs(measured_rows, dimension)
        alpha, exponent, power_r2, power_p = _safe_power_law(x, y)
        pearson_r, pearson_p = _safe_corr(x, y, "pearson")
        spearman_r, spearman_p = _safe_corr(x, y, "spearman")
        stats_rows.append(
            DimensionStats(
                dimension_id=dimension.dimension_id,
                title=dimension.title,
                n_points=len(valid_rows),
                power_law_formula=_power_formula(alpha, exponent),
                power_law_alpha=alpha,
                power_law_exponent=exponent,
                power_law_r_squared=power_r2,
                power_law_p_value=power_p,
                pearson_r=pearson_r,
                pearson_p_value=pearson_p,
                spearman_r=spearman_r,
                spearman_p_value=spearman_p,
                conclusion=_conclusion(power_r2, power_p, len(valid_rows)),
                outlier_note=_power_outlier_note(x, y, valid_rows, alpha, exponent),
            )
        )
    return stats_rows


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write joined IR-QRE rows to CSV."""
    fieldnames = [
        "label",
        "basis",
        "classical_case",
        "classical_mode",
        "classical_status",
        "active_electrons",
        "active_orbitals",
        "n_system_qubits",
        "n_terms",
        "n_estimation_wires",
        "n_trotter",
        "total_qubits",
        "ir_ops_lower_bound",
        "compile_rss_gb",
        "compile_time_s",
        "bufferization_amp",
        "logical_qubits",
        "toffoli_gates",
        "t_depth",
        "hamiltonian_1norm_Ha",
        "classical_provenance",
        "qre_provenance",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_stats_csv(stats_rows: list[DimensionStats], path: Path) -> None:
    """Write dimension statistics to CSV."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(stats_rows[0]).keys()))
        writer.writeheader()
        for stat in stats_rows:
            writer.writerow(asdict(stat))


def _case_colors(rows: list[dict[str, Any]]) -> list[str]:
    """Colorblind-friendly colors by compilation mode."""
    palette = {"dynamic": "#0072B2", "fixed": "#D55E00"}
    return [palette.get(row.get("classical_mode", ""), "#009E73") for row in rows]


def _short_case_label(row: dict[str, Any]) -> str:
    """Compact label for dense scatter annotations."""
    mode = "dyn" if row.get("classical_mode") == "dynamic" else "fix"
    return f"{row['label']}\n{mode}"


def plot_dimensions(
    rows: list[dict[str, Any]], stats_rows: list[DimensionStats], output_dir: Path
) -> list[Path]:
    """Generate one 300-dpi scatter plot per planned dimension."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stat_by_id = {stat.dimension_id: stat for stat in stats_rows}
    figure_paths: list[Path] = []
    for dimension in DIMENSIONS:
        x, y, valid_rows = _finite_pairs(rows, dimension)
        stat = stat_by_id[dimension.dimension_id]
        fig, ax = plt.subplots(figsize=(6.5, 4.8))

        plotted_labels: set[str] = set()
        for xi, yi, row, color in zip(x, y, valid_rows, _case_colors(valid_rows), strict=False):
            legend_label = row["classical_mode"]
            ax.scatter(
                [xi],
                [yi],
                c=[color],
                marker="o",
                s=68,
                edgecolor="black",
                linewidth=1.0,
                alpha=0.88,
                label=legend_label if legend_label not in plotted_labels else None,
            )
            plotted_labels.add(legend_label)
            ax.annotate(
                _short_case_label(row),
                (xi, yi),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=6.3,
                alpha=0.76,
            )

        if dimension.log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")

        if len(x) > 0 and stat.power_law_alpha is not None and stat.power_law_exponent is not None:
            x_pos = x[x > 0]
            if len(x_pos) > 0:
                x_line = np.geomspace(float(np.min(x_pos)), float(np.max(x_pos)), 100)
                y_line = stat.power_law_alpha * np.power(x_line, stat.power_law_exponent)
                ax.plot(
                    x_line,
                    y_line,
                    color="#000000",
                    linewidth=1.3,
                    label=(
                        "power fit: "
                        f"{_power_formula(stat.power_law_alpha, stat.power_law_exponent)}; "
                        f"R2={_format_float(stat.power_law_r_squared, 2)}"
                    ),
                )

        ax.set_title(f"{dimension.dimension_id}: {dimension.title}", fontsize=12)
        ax.set_xlabel(dimension.x_label)
        ax.set_ylabel(dimension.y_label)
        ax.grid(True, which="both", color="#dddddd", linewidth=0.7, alpha=0.8)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=7.5, loc="best", ncols=1)

        fig_path = (
            output_dir
            / f"ir_qre_correlation_{dimension.dimension_id}_{dimension.filename_slug}.png"
        )
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(fig_path)
    return figure_paths


def write_report(
    rows: list[dict[str, Any]],
    stats_rows: list[DimensionStats],
    figure_paths: list[Path],
    cases_path: Path,
    path: Path,
) -> None:
    """Write a concise Markdown analysis report."""
    systems = sorted({row["label"] for row in rows})
    lines = [
        "# IR-QRE Correlation Analysis",
        "",
        "## Scope and data provenance",
        "",
        (
            f"This report joins {len(rows)} measured compilation-case rows across "
            f"{len(systems)} QRE systems: {', '.join(systems)}."
        ),
        (
            "All compile rows follow the trotter-5 standard: 4 estimation wires, "
            "n_trotter=5, and measured Catalyst compilation metrics. Historical "
            "fixed-10, dynamic-2/1, and projected rows are excluded from this analysis."
        ),
        (
            "The primary CSV and D1-D4 figures are case-level measured rows. Detailed "
            f"case provenance is kept in `{cases_path.name}`."
        ),
        "",
        "## Dimension statistics",
        "",
        "Power fits use the concrete form `y = a * x^b` on measured log-log positive values.",
        "",
        "| Dim | Pair | n | Power fit | a | b | R2 | p-value | Pearson r | Spearman rho | Conclusion |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for stat in stats_rows:
        lines.append(
            f"| {stat.dimension_id} | {stat.title} | {stat.n_points} | "
            f"`{stat.power_law_formula}` | "
            f"{_format_float(stat.power_law_alpha)} | "
            f"{_format_float(stat.power_law_exponent)} | "
            f"{_format_float(stat.power_law_r_squared)} | "
            f"{_format_float(stat.power_law_p_value)} | "
            f"{_format_float(stat.pearson_r)} | "
            f"{_format_float(stat.spearman_r)} | {stat.conclusion} |"
        )

    lines.extend(["", "## Outlier review", ""])
    for stat in stats_rows:
        lines.append(f"- **{stat.dimension_id} {stat.title}**: {stat.outlier_note}")

    lines.extend(["", "## Figures", ""])
    for fig_path in figure_paths:
        lines.append(f"- `{fig_path.name}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The standardized matrix tests whether IR-scale and compiler-cost metrics "
                "track QRE metrics when Trotter count is held fixed. The result should be "
                "read as descriptive engineering evidence, not as a predictive model."
            ),
            "",
            "## Measured compile-data status",
            "",
            (
                "Only rows from `ir_qre_trotter5_compile_survey.json` are accepted. This "
                "keeps total-qubit coverage comparable while preventing old nonstandard "
                "measurements from entering the fitted statistics."
            ),
            "",
            "## Null-result handling",
            "",
            (
                "A weak or undefined statistic is reported as an engineering result, not "
                "hidden. Compile memory/time must still be modelled with compilation mode "
                "and coefficient mutability as first-class factors."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qre-json", type=Path, default=DEFAULT_QRE_JSON)
    parser.add_argument("--compile-json", type=Path, default=DEFAULT_COMPILE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_correlation_rows(args.qre_json, args.compile_json)
    if not rows:
        raise ValueError(f"No joined IR-QRE rows produced from {args.compile_json}")

    stats_rows = compute_dimension_stats(rows)
    case_rows = build_case_rows(args.qre_json, args.compile_json)
    summary_path = args.output_dir / "ir_qre_correlation.csv"
    cases_path = args.output_dir / "ir_qre_correlation_cases.csv"
    stats_path = args.output_dir / "ir_qre_correlation_stats.csv"
    report_path = args.output_dir / "ir_qre_correlation_report.md"

    write_summary_csv(rows, summary_path)
    write_summary_csv(case_rows, cases_path)
    write_stats_csv(stats_rows, stats_path)
    figure_paths = plot_dimensions(rows, stats_rows, args.output_dir)
    write_report(rows, stats_rows, figure_paths, cases_path, report_path)

    print(f"wrote {summary_path}")
    print(f"wrote {cases_path}")
    print(f"wrote {stats_path}")
    print(f"wrote {report_path}")
    for fig_path in figure_paths:
        print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
