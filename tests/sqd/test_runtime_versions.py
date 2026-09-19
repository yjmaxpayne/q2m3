"""Example provenance distinguishes absent optional packages from broken installs."""

from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest


@pytest.fixture
def calibration():
    path = Path(__file__).resolve().parents[2] / "tools/sqd/calibrate_resources.py"
    spec = importlib.util.spec_from_file_location("calibration_versions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_absent_optional_catalyst_is_reported_honestly(calibration, monkeypatch):
    def version(name):
        if name == "pennylane-catalyst":
            raise PackageNotFoundError(name)
        return "installed-version"

    monkeypatch.setattr(calibration, "version", version)
    versions = calibration.installed_versions()
    assert versions["pennylane-catalyst"] == "not-installed"
    assert versions["ffsim"] == "installed-version"


def test_absent_required_dependency_is_not_hidden(calibration, monkeypatch):
    def version(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(calibration, "version", version)
    with pytest.raises(PackageNotFoundError):
        calibration.installed_versions()


def test_installed_optional_version_is_preserved(calibration, monkeypatch):
    monkeypatch.setattr(calibration, "version", lambda name: "actual-version")
    assert calibration.installed_versions()["pennylane-catalyst"] == "actual-version"
