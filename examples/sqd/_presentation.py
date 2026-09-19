"""Input/output and interpretation for examples; all solvers live in q2m3."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "data/output/examples"
ENERGIES = {
    "HF": "hf_energy",
    "CCSD": "iso_active_space_ccsd_energy",
    "SCI (same size)": "iso_ndet_sci_energy",
    "Random (same size)": "iso_ndet_random_energy",
    "SQD": "sqd_energy",
}
RESOURCE_SCOPE = "parent + simultaneous descendants; through start of final result construction"


def parser(description: str, *, h2: bool = False) -> argparse.ArgumentParser:
    """Build consistent example arguments without importing optional solvers."""
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--output", type=Path, help="New exclusive run directory")
    result.add_argument("--seed", type=int, default=31 if h2 else 0)
    result.add_argument("--shots", type=int, default=128 if h2 else 100_000)
    if not h2:
        result.add_argument("--active-space", type=int, choices=(6, 8, 10), default=10)
    result.add_argument("--wall-budget", type=float, default=900.0, help="Seconds per engine call")
    result.add_argument(
        "--rss-budget", type=float, default=8192.0, help="Decimal MB per engine call"
    )
    result.add_argument("--no-plots", action="store_true", help=argparse.SUPPRESS)
    return result


def output_directory(path: Path | None, name: str) -> Path:
    """Create an exclusive run directory, including for default invocations."""
    path = path or OUTPUT_ROOT / (f"{name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}")
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, value: Any) -> None:
    """Write finite JSON, atomically replacing only artifacts in this run."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_result(directory: Path, name: str, result: Any) -> dict:
    """Preserve every public result field using the production serializer."""
    from q2m3.utils.io import save_json_results

    result.validate()
    path = directory / f"{name}.json"
    save_json_results(result, path)
    return json.loads(path.read_text())


def summary_row(payload: dict, label: str, result_file: str) -> dict:
    """Derive one CSV/plot row without inventing absent values or uncertainty."""
    electrons, orbitals = payload["active_space"]
    context = payload["provenance"]["context"]
    energy, baseline = payload["sqd_energy"], payload["baseline_energy"]
    if energy is not None:
        if not math.isfinite(energy) or not math.isclose(
            payload["delta_mHa"], 1000 * (energy - baseline), abs_tol=1e-8
        ):
            raise ValueError("Invalid SQD-minus-reference error or nonfinite energy")
    size, full = payload["subspace_dim"], payload["full_ci_dim"]
    if size is not None and not 0 < size <= full:
        raise ValueError("Illegal subspace dimension")
    resources = payload["diagnostics"].get("resources", {})
    row = dict(
        label=label,
        status=payload["status"],
        result_file=result_file,
        active_electrons=electrons,
        active_orbitals=orbitals,
        system_qubits=2 * orbitals,
        seed=payload["seed"],
        shots=payload["shots"],
        n_reps=payload["n_reps"],
        active_indices=json.dumps(context["active_indices"]),
        embedding_mode=payload["embedding_mode"],
        baseline_tier=payload["baseline_tier"],
        baseline_method=payload["baseline_method"],
        baseline_energy=baseline,
        baseline_uncertainty_mHa=payload["baseline_uncertainty_mHa"],
        baseline_uncertainty_kind=payload["baseline_uncertainty_kind"],
        baseline_downgrade_reason=payload["baseline_downgrade_reason"],
        baseline_untrustworthy=payload["baseline_untrustworthy"],
        delta_mHa=payload["delta_mHa"],
        exact_error_mHa=payload["delta_mHa"] if payload["baseline_tier"] == "T0" else None,
        subspace_dim=size,
        full_ci_dim=full,
        subspace_fraction=None if size is None else size / full,
        internal_peak_rss_mb=resources.get("peak_rss_mb"),
        rss_scope=RESOURCE_SCOPE,
        warnings=json.dumps(payload["warnings"]),
        null_reasons=json.dumps(payload["null_reasons"]),
    )
    row.update({field: payload[field] for field in ENERGIES.values()})
    row.update({f"time_{stage}_s": value for stage, value in payload["timings_s"].items()})
    return row


def describe_input(symbols, coordinates, active: int, args) -> dict:
    """Print the system and sampling workflow and return reproducible input metadata."""
    data = dict(
        symbols=symbols,
        coordinates_angstrom=coordinates.tolist(),
        basis="sto-3g",
        charge=0,
        active_space=[active, active],
        system_qubits=2 * active,
        seed=args.seed,
        shots=args.shots,
        n_reps=2,
        wall_budget_s=args.wall_budget,
        rss_budget_decimal_mb=args.rss_budget,
    )
    print(
        f"System: {' '.join(symbols)}, STO-3G, CAS({active}e,{active}o), "
        f"{2 * active} system qubits, 0 estimation qubits",
        flush=True,
    )
    print("Geometry (Angstrom):", flush=True)
    for symbol, xyz in zip(symbols, coordinates, strict=True):
        print(f"  {symbol:2s} {xyz[0]:10.6f} {xyz[1]:10.6f} {xyz[2]:10.6f}")
    print(
        f"Method: vacuum RHF orbitals → active integrals → CCSD amplitudes → "
        f"2-layer LUCJ → {args.shots:,} samples (seed {args.seed}) → SQD",
        flush=True,
    )
    print(f"Per-engine budget: {args.wall_budget:g} s / {args.rss_budget:g} decimal MB", flush=True)
    return data


def print_result(payload: dict, label: str) -> None:
    """Explain reference quality, competing methods and measured cost in plain text."""
    row = summary_row(payload, label, f"{label}.json")
    print(
        f"\n{label}: MO indices (zero-based) {row['active_indices']}; "
        f"reference {row['baseline_tier']}/{row['baseline_method']}"
    )
    print(f"{'Method':22s} {'Energy / Ha':>18s} {'Difference / mHa':>20s}")
    print(f"{'Reference':22s} {row['baseline_energy']:18.10f} {0.0:20.6f}")
    for name, field in ENERGIES.items():
        value = row[field]
        if value is None:
            print(f"{name:22s} unavailable: {payload['null_reasons'].get(field, 'unknown')}")
        else:
            print(f"{name:22s} {value:18.10f} {1000*(value-row['baseline_energy']):20.6f}")
    print(
        f"Subspace: {row['subspace_dim']} / {row['full_ci_dim']} determinants; "
        f"fraction {row['subspace_fraction']}; alpha/beta sizes {payload['subspace_dims']}"
    )
    print(
        "Stage wall times (s): "
        + ", ".join(
            f"{key}={value:.3f}" if value is not None else f"{key}=not executed"
            for key, value in payload["timings_s"].items()
        )
    )
    print(f"Internal polled peak RSS: {row['internal_peak_rss_mb']} decimal MB; {RESOURCE_SCOPE}.")
    print(
        "Subspace fraction is not a simulator-memory speedup: ffsim still represents the "
        "fixed-particle-number state space."
    )
    if row["baseline_tier"] != "T0":
        print(
            f"Reference downgraded: {row['baseline_downgrade_reason']}. "
            f"Uncertainty: {row['baseline_uncertainty_kind']} "
            f"({payload['null_reasons'].get('baseline_uncertainty_mHa')}). "
            "Difference is excluded from exact-reference error plots."
        )
    else:
        print("Signed differences use this active space's own exact CASCI Hamiltonian reference.")
    if row["subspace_dim"] == row["full_ci_dim"]:
        print(
            "Full-space regression checks conventions; it does not demonstrate sampling advantage."
        )
    for warning in payload["warnings"]:
        print(f"Warning: {warning}")
    print(flush=True)


def save_summary(directory: Path, rows: list[dict], *, plots: bool = True) -> None:
    """Save matching JSON and CSV, plus noninteractive plots from those same rows."""
    write_json(directory / "summary.json", rows)
    write_json(directory / "statistics.json", seed_statistics(rows))
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    if plots:
        plot_results(directory, rows)


def seed_statistics(rows: list[dict]) -> list[dict]:
    """Summarize actual T0 seeds, explicitly counting failures and downgraded rows."""
    groups = {}
    for row in rows:
        key = (row["active_electrons"], row["active_orbitals"], row.get("embedding_mode", "vacuum"))
        groups.setdefault(key, []).append(row)
    summaries = []
    for (nelec, norb, mode), group in groups.items():
        errors = [r["exact_error_mHa"] for r in group if r.get("exact_error_mHa") is not None]
        summaries.append(
            dict(
                active_space=[nelec, norb],
                embedding_mode=mode,
                requested_points=len(group),
                t0_points=len(errors),
                failed_points=sum(r["status"] == "failed" for r in group),
                non_t0_points=sum(r.get("baseline_tier") not in (None, "T0") for r in group),
                mean_signed_error_mHa=mean(errors) if errors else None,
                sample_std_mHa=stdev(errors) if len(errors) > 1 else None,
                min_signed_error_mHa=min(errors) if errors else None,
                max_signed_error_mHa=max(errors) if errors else None,
                scope="available T0 individual runs only; descriptive spread, not a confidence interval",
            )
        )
    return summaries


def plot_results(directory: Path, rows: list[dict]) -> None:
    """Plot every actual seed; exact-error panels include T0 references only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [row for row in rows if row["status"] in ("completed", "reference_only")]
    if not valid:
        return
    labels = [
        f"CAS({r['active_electrons']},{r['active_orbitals']})\n"
        f"{r['label']} / seed {r['seed']}\n{r['baseline_tier']}/{r['baseline_method']}"
        for r in valid
    ]
    x = list(range(len(valid)))
    width = max(10, len(valid) * 0.9)
    fig, axes = plt.subplots(2, 1, figsize=(width, 9), layout="constrained")
    for name, field in {"Reference": "baseline_energy", **ENERGIES}.items():
        y = [r[field] if r[field] is not None else float("nan") for r in valid]
        axes[0].plot(x, y, "o", label=name, markersize=4)
        if name != "Reference":
            error = [
                (
                    1000 * (r[field] - r["baseline_energy"])
                    if r[field] is not None and r["baseline_tier"] == "T0"
                    else float("nan")
                )
                for r in valid
            ]
            axes[1].plot(x, error, "o", label=name, markersize=4)
    axes[0].set(ylabel="Total energy / Ha", title="Total energies across spaces / environments")
    axes[1].set(ylabel="Signed error / mHa", title="Within-space error vs T0 / exact CASCI only")
    axes[1].axhline(0, color="black", linewidth=0.6)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8, ncols=3)
        ax.grid(alpha=0.2)
    excluded = sum(r["baseline_tier"] != "T0" for r in valid)
    failed = len(rows) - len(valid)
    fig.suptitle(
        f"Individual runs; no confidence intervals | {failed} failed, "
        f"{excluded} non-T0 excluded from exact errors"
    )
    _save_figure(fig, directory, "energies")
    plt.close(fig)
    fig, panels = plt.subplots(4, 1, figsize=(width, 14), layout="constrained")
    panels[0].plot(x, [r["full_ci_dim"] for r in valid], "o", label="Full fixed-particle space")
    panels[0].plot(x, [r["subspace_dim"] or float("nan") for r in valid], "o", label="SQD subspace")
    panels[0].set(yscale="log", ylabel="Determinants (log scale)", title="Active-space growth")
    panels[0].legend(fontsize=8)
    axes = panels[1:]
    axes[0].bar(x, [r["subspace_fraction"] or 0 for r in valid])
    axes[0].set(
        ylabel="Subspace / full dimension", title="Determinant compression (not RSS speedup)"
    )
    bottoms = [0.0] * len(valid)
    for stage in (
        "geometry",
        "integrals",
        "ccsd",
        "prepare",
        "sample",
        "diagonalize",
        "reference",
        "comparison",
    ):
        times = [r.get(f"time_{stage}_s") or 0 for r in valid]
        axes[1].bar(x, times, bottom=bottoms, label=stage)
        bottoms = [a + b for a, b in zip(bottoms, times, strict=True)]
    axes[1].set(ylabel="Wall time / s", title="Disjoint engine stages (unexecuted stages omitted)")
    axes[1].legend(ncols=4, fontsize=8)
    axes[2].bar(x, [r["internal_peak_rss_mb"] or 0 for r in valid])
    axes[2].set(
        ylabel="Polled RSS / decimal MB",
        title="Engine window: through start of final result construction",
    )
    for ax in panels:
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, directory, "cost")
    plt.close(fig)


def _save_figure(fig, directory, name):
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{name}.{extension}", dpi=180)


def execute(args, name, calculation) -> int:
    """Keep failed runs inspectable and propagate failure as a nonzero exit."""
    directory = output_directory(args.output, name)
    try:
        calculation(args, directory)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, ImportError):
            message += "; install from checkout: uv sync --frozen --extra sqd"
        write_json(directory / "failure.json", dict(status="failed", reason=message))
        print(f"Run failed: {message}\nSaved: {directory}", file=sys.stderr)
        return 1
    print(f"Saved complete results, summary and figures: {directory}", flush=True)
    return 0
