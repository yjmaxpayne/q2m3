"""Real connectivity consumption and honest paired missing-data reporting."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.sqd


def load_example():
    path = Path(__file__).resolve().parents[2] / "tools/sqd/connectivity_comparison.py"
    spec = importlib.util.spec_from_file_location("connectivity_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_in_fresh_process(node):
    """Keep native thread-pool state out of the suite's later fork workers."""
    if os.environ.get("Q2M3_CONNECTIVITY_TEST_NODE") == node:
        return False
    result = subprocess.run(
        [sys.executable, "-m", "pytest", f"{__file__}::{node}", "-o", "addopts=", "-n", "0", "-q"],
        env=dict(os.environ, Q2M3_CONNECTIVITY_TEST_NODE=node),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return True


def test_real_report_keeps_original_workloads_and_missing_pairs(tmp_path):
    if run_in_fresh_process("test_real_report_keeps_original_workloads_and_missing_pairs"):
        return
    example = load_example()
    report = example.run_comparison(tmp_path / "run")
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    specs = manifest["systems"]
    assert [(s["n_reps"], s["shots"], s["num_batches"], s["max_iterations"]) for s in specs] == [
        (2, 50000, 3, 5),
        (2, 50000, 3, 5),
        (4, 200000, 5, 6),
        (4, 500000, 2, 2),
    ]
    assert [s["active_space"] for s in specs] == [[2, 2], [4, 4], [6, 6], [10, 10]]
    assert specs[2]["geometry_format"] == "zmatrix"
    assert "C 1 1.52" in specs[2]["atom"]
    assert manifest["seeds"] == [0, 1, 2]
    assert report["capability"]["generator_available"] is True
    assert len(report["records"]) == 12
    assert {(r["system"], r["seed"]) for r in report["records"]} == {
        (name, seed) for name in ("h2", "h3o", "glycine", "n2") for seed in range(3)
    }
    for row in report["records"]:
        assert row["D_mHa"] is None
        assert row["full_energy_ha"] is row["restricted_energy_ha"] is None
        assert row["status"] == "inconclusive"
        assert row["null_reason"] == "original_workload_outside_calibration"
        assert row["molecular_pair_executed"] is False
    assert all(s["conclusion"] == "inconclusive" for s in report["statistics"].values())
    assert report["statistics"]["h2"]["mean_mHa"] is None
    for name in ("h2", "h3o"):
        assert set(report["profile_rejections"][name]) == {"num_batches", "max_iterations"}
    assert set(report["profile_rejections"]["glycine"]) == {
        "n_reps",
        "shots",
        "num_batches",
        "max_iterations",
    }
    assert set(report["profile_rejections"]["n2"]) == {"n_reps", "shots"}
    witness = report["operator_witness"]
    assert witness["scope"] == "h3o_operator_only_no_sqd_energy_comparison"
    assert witness["forbidden_max_abs"] == [0.0, 0.0]
    assert witness["matrix_max_difference"] > 1e-8
    assert witness["state_infidelity"] > 1e-8
    assert witness["restricted_pairs"] == [[[0, 1], [1, 2], [2, 3]], [[0, 0]]]
    arrays = np.load(tmp_path / "run/operator-witness.npz")
    for channel in range(2):
        mask = np.zeros((4, 4), dtype=bool)
        for i, j in witness["restricted_pairs"][channel]:
            mask[i, j] = mask[j, i] = True
        assert np.all(arrays["restricted_mats"][:, channel, ~mask] == 0)
        assert np.any(np.abs(arrays["full_mats"][:, channel, ~mask]) > 1e-8)
    assert example.verify_artifacts(tmp_path / "run") == report
    with pytest.raises(FileExistsError):
        example.run_comparison(tmp_path / "run")
    p = tmp_path / "run/manifest.json"
    p.chmod(0o644)
    p.write_text(p.read_text().replace("500000", "100000"))
    with pytest.raises(ValueError, match="hash"):
        example.verify_artifacts(tmp_path / "run")


def test_real_operator_receives_pairs_and_bypass_fails(monkeypatch):
    if run_in_fresh_process("test_real_operator_receives_pairs_and_bypass_fails"):
        return
    import ffsim

    example = load_example()
    original = ffsim.UCJOpSpinBalanced.from_t_amplitudes
    captured = []

    def capture(*args, **kwargs):
        captured.append(kwargs["interaction_pairs"])
        return original(*args, **kwargs)

    monkeypatch.setattr(ffsim.UCJOpSpinBalanced, "from_t_amplitudes", capture)
    example.operator_witness()
    assert captured == [None, ([(0, 1), (1, 2), (2, 3)], [(0, 0)])]

    def bypass(*args, **kwargs):
        kwargs["interaction_pairs"] = None
        return original(*args, **kwargs)

    monkeypatch.setattr(ffsim.UCJOpSpinBalanced, "from_t_amplitudes", bypass)
    with pytest.raises(ValueError, match="forbidden"):
        example.operator_witness()


@pytest.mark.parametrize(
    "values,expected",
    [
        ([6.0, 7.0, 8.0], "negative"),
        ([5.0, -2.0, 4.0], "not_refuted_in_sample"),
        ([4.9, 5.1, 6.0], "inconclusive"),
        ([None, 8.0, 9.0], "inconclusive"),
    ],
)
def test_paired_threshold_does_not_hide_seeds(values, expected):
    example = load_example()
    rows = [{"seed": i, "D_mHa": v} for i, v in enumerate(values)]
    stats = example.summarize_pairs(rows)
    assert stats["conclusion"] == expected
    assert stats["mean_mHa"] == (None if None in values else np.mean(values))


def test_pair_summary_rejects_duplicate_and_nonfinite_values():
    example = load_example()
    for rows in (
        [{"seed": 0, "D_mHa": 6.0}] * 3,
        [{"seed": s, "D_mHa": float("nan")} for s in range(3)],
    ):
        with pytest.raises(ValueError):
            example.summarize_pairs(rows)


def test_missing_generator_is_explicit(monkeypatch):
    from ffsim.variational import util

    example = load_example()
    monkeypatch.delattr(util, "interaction_pairs_spin_balanced")
    probe = example.connectivity_capability()
    assert probe["generator_available"] is False
    assert probe["pairs"] is None
    assert probe["reason"] == "generator_missing_in_installed_version"
    assert probe["version"] == "0.0.84"


def test_unrelated_resource_errors_propagate(monkeypatch):
    from q2m3.sqd import resources
    from q2m3.sqd.exceptions import ResourceModelDomainError

    example = load_example()
    original = resources.load_resource_model

    def broken(*args, **kwargs):
        if kwargs.get("profile"):
            raise ResourceModelDomainError("dependency versions differ")
        return original(*args, **kwargs)

    monkeypatch.setattr(resources, "load_resource_model", broken)
    with pytest.raises(ResourceModelDomainError, match="versions"):
        example.profile_rejections(example.SYSTEMS[0])


def test_newly_reachable_workload_requires_numerical_experiment(monkeypatch, tmp_path):
    example = load_example()
    monkeypatch.setattr(example, "profile_rejections", lambda spec: {})
    with pytest.raises(RuntimeError, match="numerical paired experiment required"):
        example.run_comparison(tmp_path / "accepted")
    assert not (tmp_path / "accepted/report.json").exists()
