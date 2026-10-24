"""``external`` tests skip themselves until ``Q2M3_PSI4_PYTHON`` names the Psi4 interpreter."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

CONFTEST = Path(__file__).with_name("conftest.py")
PSI4_PYTHON = "Q2M3_PSI4_PYTHON"
SAMPLE = """
import pytest


@pytest.mark.external
def test_marked():
    pass


def test_plain():
    pass
"""


def test_external_marker_is_registered(pytestconfig):
    assert any(line.startswith("external:") for line in pytestconfig.getini("markers"))


@pytest.fixture
def isolated(pytester, monkeypatch):
    """Run one marked and one unmarked test under a copy of the real fragments conftest."""
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pytester.makeconftest(CONFTEST.read_text())
    pytester.makeini("[pytest]\nmarkers =\n    external\n")
    pytester.makepyfile(test_psi4=SAMPLE)
    return pytester


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_external_test_skips_with_reason_without_psi4_python(isolated, monkeypatch, value):
    if value is None:
        monkeypatch.delenv(PSI4_PYTHON, raising=False)
    else:
        monkeypatch.setenv(PSI4_PYTHON, value)
    result = isolated.runpytest_subprocess("-rs")
    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.fnmatch_lines([f"SKIPPED*{PSI4_PYTHON}*"])


def test_external_test_runs_when_psi4_python_is_set(isolated, monkeypatch):
    monkeypatch.setenv(PSI4_PYTHON, "/opt/psi4/bin/python")
    isolated.runpytest_subprocess().assert_outcomes(passed=2)
