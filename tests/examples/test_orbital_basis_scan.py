"""An unreachable scan must retain its inputs and never manufacture orbital results."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.sqd


def load_example():
    path = Path(__file__).resolve().parents[2] / "tools/sqd/orbital_basis_scan.py"
    spec = importlib.util.spec_from_file_location("orbital_scan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_preflight_and_complete_report(tmp_path):
    scan = load_example()
    report = scan.run_scan(tmp_path / "scan")
    manifest = json.loads((tmp_path / "scan/manifest.json").read_text())
    assert manifest["experiment"]["n_reps"] == 4
    assert manifest["experiment"]["shots"] == 500000
    assert manifest["experiment"]["seeds"] == [0, 1, 2]
    assert manifest["experiment"]["active_space"] == [10, 10]
    assert report["conclusion"] == "inconclusive"
    assert report["scope"] == "shared_preflight_only"
    assert len(report["entry_probes"]) == 3
    assert {p["seed"] for p in report["entry_probes"]} == {0, 1, 2}
    assert set(report["profile_probes"]) == {"n_reps", "shots"}
    assert all(p["exception"] == "ResourceModelDomainError" for p in report["entry_probes"])
    assert len(report["records"]) == 12
    assert {(r["basis"], r["seed"]) for r in report["records"]} == {
        (b, s) for b in ("canonical_rhf", "natural", "casscf", "localized") for s in range(3)
    }
    for row in report["records"]:
        assert row["status"] == "blocked_before_orbital_construction"
        assert row["orbital_method_consumed"] is False
        assert row["sqd_result"] is None
        assert row["frame_id"] is None
        assert row["hamiltonian_id"] is None
        assert all(v is None for v in row["energies_ha"].values())
        assert row["delta_mHa"] is None
        assert row["null_reason"] == "uncalibrated_workload"
    assert report["statistics"]["paired_ranges_mHa"] == [None, None, None]
    assert report["statistics"]["mean_range_mHa"] is None
    assert scan.verify_artifacts(tmp_path / "scan") == report
    before = (tmp_path / "scan/manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        scan.run_scan(tmp_path / "scan")
    assert (tmp_path / "scan/manifest.json").read_bytes() == before


@pytest.mark.parametrize("error", [ValueError("broken"), RuntimeError("solver bug")])
def test_unexpected_failure_propagates(error):
    scan = load_example()

    def fail():
        raise error

    with pytest.raises(type(error), match="broken|solver bug"):
        scan.expect_domain_rejection(fail, "n_reps=4")


def test_acceptance_and_unrelated_domain_failure_are_not_inconclusive():
    from q2m3.sqd.exceptions import ResourceModelDomainError

    scan = load_example()
    with pytest.raises(RuntimeError, match="accepted"):
        scan.expect_domain_rejection(lambda: object(), "n_reps=4")

    def mismatch():
        raise ResourceModelDomainError("dependency versions differ")

    with pytest.raises(ResourceModelDomainError, match="versions"):
        scan.expect_domain_rejection(mismatch, "n_reps=4")


def test_mutated_manifest_is_rejected(tmp_path):
    scan = load_example()
    out = tmp_path / "scan"
    scan.run_scan(out)
    path = out / "manifest.json"
    path.chmod(0o644)
    path.write_text(path.read_text().replace("500000", "100000"))
    with pytest.raises(ValueError, match="hash"):
        scan.verify_artifacts(out)
