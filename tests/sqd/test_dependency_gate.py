"""Fail-closed checks for dependency isolation and numerical comparison."""

from __future__ import annotations

import importlib

import pytest

from tests.sqd import dependency_gate as gate


def test_core_profile_accepts_export_and_installed_inventory():
    gate.assert_core_profile("numpy==2.0\npyscf==2.11\n", [{"name": "numpy"}])


@pytest.mark.parametrize("name", ["ffsim", "qiskit", "Qiskit_Addon_SQD", "qiskit-aer"])
@pytest.mark.parametrize("location", ["export", "installed"])
def test_core_profile_rejects_optional_quantum_packages(name, location):
    exported = f"{name}==1.0\n" if location == "export" else "numpy==2.0\n"
    installed = [{"name": name if location == "installed" else "numpy"}]
    with pytest.raises(AssertionError, match="optional SQD"):
        gate.assert_core_profile(exported, installed)


def test_catalyst_gate_requires_actual_import(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ModuleNotFoundError, match="catalyst"):
        gate.require_catalyst()


def test_catalyst_gate_accepts_successful_import(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", lambda name: object())
    assert gate.require_catalyst()["imported"] is True


def test_numeric_comparison_rejects_identical_wrong_science():
    report = {"input": {}, "energies_ha": {"fci": -1.0, "ccsd": -0.9}}
    with pytest.raises(AssertionError, match="oracle"):
        gate.compare_numerics(report, report)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.00000001])
def test_numeric_comparison_rejects_drift_or_nonfinite(bad):
    baseline = valid_report()
    candidate = valid_report()
    candidate["energies_ha"]["fci"] = bad
    with pytest.raises(AssertionError):
        gate.compare_numerics(baseline, candidate)


def test_numeric_comparison_accepts_equal_valid_reports():
    assert gate.compare_numerics(valid_report(), valid_report())["max_energy_delta_ha"] == 0


def test_numeric_comparison_rejects_different_inputs():
    candidate = valid_report()
    candidate["input"] = {"basis": "different"}
    with pytest.raises(AssertionError, match="input"):
        gate.compare_numerics(valid_report(), candidate)


def valid_report():
    return {
        "input": {"basis": "sto-3g"},
        "energies_ha": {name: -1.0 for name in ("fci", "ccsd", "casci", "selected_ci")},
        "ao2mo_max_error": 0.0,
        "arrays": {"h1": [[1.0]], "h2": [[[[0.1]]]]},
    }


@pytest.mark.parametrize("exported,installed", [("", [{"name": "numpy"}]), ("numpy==2", [])])
def test_core_profile_rejects_empty_evidence(exported, installed):
    with pytest.raises(AssertionError, match="Empty"):
        gate.assert_core_profile(exported, installed)


def test_numeric_comparison_rejects_changed_integral_shape():
    baseline, candidate = valid_report(), valid_report()
    candidate["arrays"]["h1"] = [1.0]
    with pytest.raises(AssertionError, match="shape"):
        gate.compare_numerics(baseline, candidate)


def test_numeric_comparison_rejects_wrong_ao_transform():
    report = valid_report()
    report["ao2mo_max_error"] = 1e-8
    with pytest.raises(AssertionError, match="AO transform"):
        gate.compare_numerics(report, report)
