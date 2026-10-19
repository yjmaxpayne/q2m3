"""Scientific interpretation, artifact consistency and failure retention for tutorials."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.sqd import _presentation as display
from examples.sqd._inputs import fixed_waters
from tests.sqd.test_result import result_data


def _validate_saved_run(directory: Path, *, require_t0: bool = False) -> list[dict]:
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


def _validate_embedding(directory: Path) -> None:
    rows = _validate_saved_run(directory, require_t0=True)
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


def _validate_integrals(directory: Path) -> None:
    rows = _validate_saved_run(directory, require_t0=True)
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


@pytest.fixture
def payload():
    data = result_data()
    data["provenance"] = {"context": {"active_indices": [0, 1]}}
    data["diagnostics"] = {"resources": {"peak_rss_mb": 123.0}}
    return data


def test_summary_uses_signed_mha_and_keeps_all_controls(payload, tmp_path):
    row = display.summary_row(payload, "test", "result.json")
    assert row["delta_mHa"] == pytest.approx(20)
    assert row["exact_error_mHa"] == pytest.approx(20)
    assert row["subspace_fraction"] == 1
    display.save_summary(tmp_path, [row])
    assert json.loads((tmp_path / "summary.json").read_text()) == [row]
    with (tmp_path / "summary.csv").open() as stream:
        csv_row = next(csv.DictReader(stream))
    for key in (*display.ENERGIES.values(), "delta_mHa", "internal_peak_rss_mb"):
        assert float(csv_row[key]) == row[key]
    for name in ("energies", "cost"):
        assert (tmp_path / f"{name}.png").stat().st_size > 1000
        assert "<svg" in (tmp_path / f"{name}.svg").read_text()


@pytest.mark.parametrize("corruption", ["units", "nonfinite", "dimension"])
def test_summary_rejects_corrupt_scientific_claims(payload, corruption):
    if corruption == "units":
        payload["delta_mHa"] = 0.02
    elif corruption == "nonfinite":
        payload["sqd_energy"] = float("nan")
    else:
        payload["subspace_dim"] = 5
    with pytest.raises(ValueError):
        display.summary_row(payload, "bad", "result.json")


def test_downgrade_is_not_exact_error_and_is_explained(payload, capsys, monkeypatch, tmp_path):
    payload.update(
        baseline_tier="T2",
        baseline_method="ccsd_t",
        baseline_uncertainty_mHa=None,
        baseline_uncertainty_kind="unknown",
        baseline_downgrade_reason="budget",
    )
    payload["null_reasons"]["baseline_uncertainty_mHa"] = "unknown_for_t2"
    row = display.summary_row(payload, "T2", "result.json")
    assert row["exact_error_mHa"] is None
    assert row["baseline_uncertainty_mHa"] is None
    figures = []
    monkeypatch.setattr(display, "_save_figure", lambda fig, *_: figures.append(fig))
    display.plot_results(tmp_path, [row])
    for line in figures[0].axes[1].lines[:-1]:
        assert np.isnan(line.get_ydata()).all(), "T2 entered exact-reference curve"
    display.print_result(payload, "T2")
    output = capsys.readouterr().out
    assert "Reference downgraded: budget" in output
    assert "unknown_for_t2" in output
    assert "not a simulator-memory speedup" in output


def test_fixed_waters_are_neutral_with_tip3p_geometry():
    charges, coordinates = fixed_waters()
    assert charges.shape == (6,)
    assert sum(charges) == pytest.approx(0)
    for start in (0, 3):
        bonds = coordinates[start + 1 : start + 3] - coordinates[start]
        assert np.linalg.norm(bonds, axis=1) == pytest.approx([0.9572, 0.9572])
        angle = np.degrees(np.arccos(np.dot(*bonds) / 0.9572**2))
        assert angle == pytest.approx(104.52)


def test_partial_scan_retains_each_requested_point(payload, tmp_path, monkeypatch):
    from examples.sqd import glycine_active_space_scan as scan

    calls = []

    def worker(command, **kwargs):
        calls.append(command)
        seed = int(command[command.index("--seed") + 1])
        target = Path(command[command.index("--output") + 1])
        target.mkdir()
        if seed == 2:
            display.write_json(target / "failure.json", {"reason": "test budget exceeded"})
            return SimpleNamespace(returncode=1)
        data = copy.deepcopy(payload)
        data["seed"] = seed
        row = display.summary_row(data, "mock", "result.json")
        display.write_json(target / "summary.json", [row])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(scan.subprocess, "run", worker)
    monkeypatch.setattr(display, "plot_results", lambda *_: None)
    args = SimpleNamespace(
        active_spaces=[6],
        seed=0,
        shots=100000,
        wall_budget=900,
        rss_budget=8192,
        output=tmp_path / "scan",
    )
    assert display.execute(args, "scan", scan.calculate) == 1
    rows = json.loads((args.output / "summary.json").read_text())
    assert len(rows) == len(calls) == 5
    assert [row["seed"] for row in rows] == list(range(5))
    assert rows[2]["status"] == "failed"
    assert rows[2]["reason"] == "test budget exceeded"
    assert sum(row["status"] == "completed" for row in rows) == 4
    assert (args.output / "failure.json").exists()
    assert all(command[1:3] == ["-m", "examples.sqd.glycine_ground_state"] for command in calls)


def test_failure_hint_and_exclusive_outputs(tmp_path, capsys):
    args = SimpleNamespace(output=tmp_path / "missing")

    def missing(*_):
        raise ImportError("ffsim is missing")

    assert display.execute(args, "missing", missing) == 1
    assert "uv sync --frozen --extra sqd" in capsys.readouterr().err
    assert "ffsim" in (args.output / "failure.json").read_text()
    with pytest.raises(FileExistsError):
        display.execute(args, "missing", missing)


@pytest.mark.sqd
def test_real_h2_tutorial_matches_independent_casci(tmp_path):
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit_addon_sqd")
    from pyscf import gto, mcscf, scf

    directory = tmp_path / "h2"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    process = subprocess.run(
        [sys.executable, "-m", "examples.sqd.h2_ground_state", "--output", str(directory)],
        cwd=display.ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads((directory / "result.json").read_text())
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    exact = float(mcscf.CASCI(scf.RHF(mol).run(), 2, 2).kernel()[0])
    assert len(result) == 40
    assert result["baseline_tier"] == "T0"
    assert result["subspace_dim"] == result["full_ci_dim"] == 4
    assert result["sqd_energy"] == pytest.approx(exact, abs=1e-8)
    assert result["baseline_energy"] == pytest.approx(exact, abs=1e-8)
    assert "Full-space regression" in process.stdout
    assert (
        json.loads((directory / "summary.json").read_text())[0]["sqd_energy"]
        == result["sqd_energy"]
    )


@pytest.mark.parametrize(
    "mutation", ["none", "csv", "energy", "dimensions", "missing_field", "statistics"]
)
def test_saved_artifact_acceptance_rejects_corruption(payload, tmp_path, mutation):
    display.write_json(tmp_path / "result.json", payload)
    row = display.summary_row(payload, "test", "result.json")
    display.save_summary(tmp_path, [row])
    if mutation == "csv":
        path = tmp_path / "summary.csv"
        path.write_text(path.read_text().replace("-1.12", "-100.12"))
    elif mutation == "statistics":
        display.write_json(tmp_path / "statistics.json", [])
    elif mutation in ("energy", "dimensions", "missing_field"):
        data = json.loads((tmp_path / "result.json").read_text())
        if mutation == "energy":
            data["sqd_energy"] += 0.1
        elif mutation == "dimensions":
            data["subspace_dims"] = [1, 1]
        else:
            del data["warnings"]
        display.write_json(tmp_path / "result.json", data)
    if mutation == "none":
        assert _validate_saved_run(tmp_path, require_t0=True) == [row]
    else:
        with pytest.raises(ValueError):
            _validate_saved_run(tmp_path, require_t0=True)


def test_saved_showcase_directories():
    root = os.environ.get("Q2M3_SHOWCASE_ROOT")
    if root is None:
        pytest.skip("Q2M3_SHOWCASE_ROOT is not set")
    base = Path(root)
    _validate_saved_run(base / "ci-scan", require_t0=True)
    _validate_embedding(base / "ci-embedding")
    _validate_integrals(base / "ci-integrals")
