"""Scientific report audits use real public results and deliberate corruptions."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.sqd


def example():
    path = Path(__file__).resolve().parents[2] / "tools/sqd/molecular_benchmark.py"
    spec = importlib.util.spec_from_file_location("sqd_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    module = example()
    root = tmp_path_factory.mktemp("benchmark") / "h2"
    return module, root, module.run_benchmark("h2", root)


def test_real_h2_complete_result_and_monitor(measured):
    module, root, report = measured
    assert module.verify_artifacts(root) == report
    row = report["records"][0]
    result = row["sqd"]
    assert len(result) == 40
    assert result["active_space"] == [2, 2]
    assert result["baseline_tier"] == "T0"
    assert (
        abs(result["delta_mHa"] - 1000 * (result["sqd_energy"] - result["baseline_energy"])) < 1e-9
    )
    assert result["iso_active_space_ccsd_energy"] is not None
    assert row["interpretation"] == "convention_regression_not_accuracy_advantage"
    assert row["resources"]["stage_peaks_bytes"]["sample"] > 0
    assert row["resources"]["peak_tree_bytes"] >= row["resources"]["stage_peaks_bytes"]["sample"]
    assert row["audit"]["sampling_integrity"] is True
    assert row["audit"]["four_arm_spin_fairness"] is True
    assert row["resources"]["stage_complete"]["sample"] is True
    assert set(row["resources"]["g2b_stage_checks"]) == {
        "integrals",
        "ccsd",
        "prepare",
        "sample",
        "diagonalize",
        "reference",
        "comparison",
    }
    assert report["manifest"]["profiles"]["default"] == {"n_reps": 2, "shots": 100000}
    with pytest.raises(FileExistsError):
        module.run_benchmark("h2", root)


@pytest.mark.parametrize(
    "corruption", ["units", "ccsd", "full_accuracy", "sample_as_total", "spin", "curve"]
)
def test_report_rejects_misleading_science(measured, corruption):
    module, _, report = measured
    row = copy.deepcopy(report["records"][0])
    if corruption == "units":
        row["sqd"]["delta_mHa"] += 1.0
    elif corruption == "ccsd":
        del row["sqd"]["iso_active_space_ccsd_energy"]
    elif corruption == "full_accuracy":
        row["interpretation"] = "accuracy_advantage"
    elif corruption == "sample_as_total":
        row["resources"]["peak_tree_bytes"] = row["resources"]["stage_peaks_bytes"]["sample"] - 1
    elif corruption == "spin":
        row["sqd"]["diagnostics"]["comparison_ci_strings"]["sci"][0].append(0)
    else:
        row["sqd"]["unique_dets_vs_shots"][-1][1] += 1
    with pytest.raises((ValueError, KeyError)):
        module.audit_record(row)


def test_threshold_is_decimal_strict_and_requires_both_windows():
    module = example()
    assert module.memory_verdict(100, 1_999_999_999) == "pass"
    assert module.memory_verdict(100, 2_000_000_000) == "fail"
    assert module.memory_verdict(None, 100) == "inconclusive"
    assert module.memory_verdict(None, 2_000_000_000) == "fail"


def test_statistics_require_five_distinct_nonfull_seeds():
    module = example()
    rows = [
        {"seed": i, "delta_mHa": float(i), "subspace_dim": 3, "full_ci_dim": 4} for i in range(5)
    ]
    stats = module.glycine_statistics(rows)
    assert stats["mean_mHa"] == 2
    assert stats["sample_std_mHa"] == pytest.approx(2.5**0.5)
    assert stats["max_abs_error_mHa"] == 4
    with pytest.raises(ValueError):
        module.glycine_statistics(rows[:-1])
    rows[0]["subspace_dim"] = 4
    with pytest.raises(ValueError):
        module.glycine_statistics(rows)


def test_artifact_hash_detects_tampering(measured):
    module, root, _ = measured
    path = root / "manifest.json"
    original = path.read_bytes()
    path.chmod(0o644)
    path.write_text("{}")
    try:
        with pytest.raises(ValueError, match="hash"):
            module.verify_artifacts(root)
    finally:
        path.write_bytes(original)
        path.chmod(0o444)


def test_entrypoints_exist_and_name_system():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [sys.executable, str(root / "tools/sqd/molecular_benchmark.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--system {h2,h3o,glycine,n2}" in process.stdout
