"""Check saved showcase results against independent arithmetic and artifact contracts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def validate_run(directory: Path, *, require_t0: bool = False) -> list[dict]:
    """Reject missing points, invalid errors/dimensions and JSON/CSV disagreement.

    This checks saved evidence; it does not substitute for running the solver.
    """
    rows = json.loads((directory / "summary.json").read_text())
    with (directory / "summary.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    if not rows or len(rows) != len(csv_rows):
        raise ValueError("Summary row inventory mismatch")
    for row, csv_row in zip(rows, csv_rows, strict=True):
        if row["status"] != "completed":
            raise ValueError("A requested calculation did not complete")
        for key, value in row.items():
            if csv_row[key] != ("" if value is None else str(value)):
                raise ValueError(f"CSV mismatch: {key}")
        data = json.loads((directory / row["result_file"]).read_text())
        if len(data) != 40 or data["schema_version"] != "sqd.result.v1":
            raise ValueError("Incomplete public result")
        for key in (
            "baseline_tier",
            "baseline_method",
            "baseline_uncertainty_kind",
            "baseline_uncertainty_mHa",
            "baseline_downgrade_reason",
            "seed",
            "shots",
            "n_reps",
            "embedding_mode",
        ):
            if row[key] != data[key]:
                raise ValueError(f"Result metadata mismatch: {key}")
        if require_t0 and data["baseline_tier"] != "T0":
            raise ValueError("Acceptance requires T0 for every requested point")
        for key in (
            "sqd_energy",
            "hf_energy",
            "baseline_energy",
            "iso_active_space_ccsd_energy",
            "iso_ndet_sci_energy",
            "iso_ndet_random_energy",
        ):
            if row[key] != data[key] or (data[key] is not None and not math.isfinite(data[key])):
                raise ValueError(f"Energy summary mismatch: {key}")
        expected = 1000 * (data["sqd_energy"] - data["baseline_energy"])
        if not math.isclose(row["delta_mHa"], expected, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Signed mHa error mismatch")
        exact = row["exact_error_mHa"]
        if (data["baseline_tier"] != "T0" and exact is not None) or (
            data["baseline_tier"] == "T0" and exact != row["delta_mHa"]
        ):
            raise ValueError("Exact-error series includes a non-exact reference")
        nelec, norb = data["active_space"]
        full = math.comb(norb, nelec // 2) ** 2
        if not (
            data["full_ci_dim"] == row["full_ci_dim"] == full
            and data["subspace_dim"] == row["subspace_dim"]
            and 0 < data["subspace_dim"] <= full
            and math.prod(data["subspace_dims"]) == data["subspace_dim"]
        ):
            raise ValueError("Illegal determinant dimensions")
        if row["subspace_fraction"] != data["subspace_dim"] / full:
            raise ValueError("Incorrect subspace fraction")
    statistics = json.loads((directory / "statistics.json").read_text())
    groups = {
        (r["active_electrons"], r["active_orbitals"], r.get("embedding_mode", "vacuum"))
        for r in rows
    }
    reported = {(*item["active_space"], item["embedding_mode"]) for item in statistics}
    if len(statistics) != len(groups) or reported != groups:
        raise ValueError("Statistics group inventory mismatch")
    for stats in statistics:
        selected = [
            r
            for r in rows
            if [r["active_electrons"], r["active_orbitals"]] == stats["active_space"]
            and r.get("embedding_mode", "vacuum") == stats["embedding_mode"]
        ]
        errors = [r["delta_mHa"] for r in selected if r.get("baseline_tier") == "T0"]
        average = sum(errors) / len(errors) if errors else None
        if stats["requested_points"] != len(selected) or stats["t0_points"] != len(errors):
            raise ValueError("Statistics point count mismatch")
        if average is not None and not math.isclose(
            stats["mean_signed_error_mHa"], average, rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError("Statistics mean mismatch")
        if len(errors) > 1:
            deviation = math.sqrt(sum((e - average) ** 2 for e in errors) / (len(errors) - 1))
            if not math.isclose(stats["sample_std_mHa"], deviation, rel_tol=0, abs_tol=1e-8):
                raise ValueError("Statistics sample deviation mismatch")
    for name in ("energies.png", "energies.svg", "cost.png", "cost.svg"):
        if (directory / name).stat().st_size < 1000:
            raise ValueError(f"Missing/empty figure: {name}")
    return rows


def validate_scan(directory: Path) -> None:
    """Require all 15 default points exactly once, with their own exact reference."""
    rows = validate_run(directory, require_t0=True)
    expected = {(n, seed) for n in (6, 8, 10) for seed in range(5)}
    if len(rows) != 15 or {(r["active_orbitals"], r["seed"]) for r in rows} != expected:
        raise ValueError("Default scan must have exactly 15 distinct requested points")


def validate_embedding(directory: Path) -> None:
    """Check fixed frames and separate Hamiltonian shifts from solver residuals."""
    rows = validate_run(directory, require_t0=True)
    by_mode = {r["embedding_mode"]: r for r in rows}
    if len(rows) != 3 or set(by_mode) != {"vacuum", "diagonal", "full_oneelectron"}:
        raise ValueError("Three distinct embedding modes are required")
    frames = {
        json.loads((directory / r["result_file"]).read_text())["provenance"]["context"]["frame_id"]
        for r in rows
    }
    if len(frames) != 1:
        raise ValueError("Embedding frame mismatch")
    vacuum = by_mode["vacuum"]
    for row in rows:
        shift = 1000 * (row["baseline_energy"] - vacuum["baseline_energy"])
        residual_change = row["delta_mHa"] - vacuum["delta_mHa"]
        sqd_shift = 1000 * (row["sqd_energy"] - vacuum["sqd_energy"])
        if not all(
            math.isclose(a, b, abs_tol=1e-8)
            for a, b in (
                (shift, row["reference_shift_vs_vacuum_mHa"]),
                (residual_change, row["solver_error_change_mHa"]),
                (sqd_shift, row["sqd_shift_vs_vacuum_mHa"]),
                (sqd_shift, shift + residual_change),
            )
        ):
            raise ValueError("Embedding energy decomposition mismatch")


def validate_integrals(directory: Path) -> None:
    """Recompute adapter agreement from preserved results, not a saved success flag."""
    rows = validate_run(directory, require_t0=True)
    if len(rows) != 2 or {row["label"] for row in rows} != {"integrals", "geometry"}:
        raise ValueError("Both integral and geometry results required")
    low, high = [json.loads((directory / row["result_file"]).read_text()) for row in rows]
    for key in ("frame_id", "hamiltonian_id", "active_indices"):
        if low["provenance"]["context"][key] != high["provenance"]["context"][key]:
            raise ValueError("Integral adapter context mismatch")
    for key in (
        "sqd_energy",
        "baseline_energy",
        "hf_energy",
        "iso_active_space_ccsd_energy",
        "iso_ndet_sci_energy",
        "iso_ndet_random_energy",
    ):
        if low[key] is None or high[key] is None:
            matched = low[key] == high[key]
        else:
            matched = math.isclose(low[key], high[key], rel_tol=0, abs_tol=1e-8)
        if not matched:
            raise ValueError("Integral adapter energy mismatch")
    if not json.loads((directory / "consistency.json").read_text())["matched"]:
        raise ValueError("Integral adapter consistency failed")


def main() -> int:
    """Validate the real scan, integral and embedding deliverables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--integrals", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    args = parser.parse_args()
    validate_scan(args.scan)
    validate_embedding(args.embedding)
    validate_integrals(args.integrals)
    print("PASS: 15 T0 scan points, fixed-frame embedding, integral adapter, CSV/JSON and figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
