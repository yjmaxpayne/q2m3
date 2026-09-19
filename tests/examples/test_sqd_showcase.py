"""Scientific interpretation, artifact consistency and failure retention for tutorials."""

from __future__ import annotations

import copy
import csv
import json
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
    from tools.sqd.check_showcase import validate_run

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
        assert validate_run(tmp_path, require_t0=True) == [row]
    else:
        with pytest.raises(ValueError):
            validate_run(tmp_path, require_t0=True)


def test_external_monitor_records_timeout_and_nonzero_exit(tmp_path):
    from tools.sqd.verify_showcase import measure

    timed = measure(
        [sys.executable, "-c", "import time; time.sleep(5)"], tmp_path / "timeout", wall_s=0.2
    )
    assert timed["reason"] == "wall_budget_exceeded"
    assert timed["exit_code"] != 0
    assert timed["wall_s"] < 4
    failed = measure([sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "exit")
    assert failed["reason"] is None
    assert failed["exit_code"] == 3
