"""Dependency profiles must fail closed before pytest can skip missing extras."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[2] / "tools/sqd/ci_profile.py"
    assert path.is_file(), "CI profile runner must be shipped in the checkout"
    spec = importlib.util.spec_from_file_location("ci_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("profile,missing", [("sqd", "ffsim"), ("sqd-catalyst", "catalyst")])
def test_missing_required_extra_fails_before_pytest(runner, monkeypatch, profile, missing):
    def importing(name):
        if name == missing:
            raise ImportError(f"missing {name}")
        return SimpleNamespace()

    monkeypatch.setattr(runner.importlib, "import_module", importing)
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match=missing):
        runner.preflight(profile)


def test_core_profile_rejects_accidentally_installed_sqd(runner, monkeypatch):
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(RuntimeError, match="ffsim"):
        runner.preflight("core")


def test_sqd_profile_rejects_accidentally_installed_catalyst(runner, monkeypatch):
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(RuntimeError, match="catalyst"):
        runner.preflight("sqd")


def test_collection_exclusions_preserve_pure_sqd_tests(runner):
    args = runner.pytest_args("core")
    assert "--ignore=tests/solvation" in args
    assert "--ignore=tests/sqd/test_sampling.py" in args
    assert "--ignore=tests/sqd" not in args
    for name in ("test_config.py", "test_result.py", "test_resources.py"):
        assert f"--ignore=tests/sqd/{name}" not in args
    assert "not sqd" in args[args.index("-m") + 1]
    assert args[args.index("-n") + 1] == "0"


def test_full_profile_keeps_science_and_catalyst(runner):
    args = runner.pytest_args("sqd-catalyst")
    assert not any(arg.startswith("--ignore") for arg in args)
    assert args[args.index("-m") + 1] == "not slow and not gpu"


def test_all_skipped_sqd_is_not_success(runner):
    gate = runner.ExecutionGate("sqd")
    session = SimpleNamespace(
        exitstatus=0, config=SimpleNamespace(option=SimpleNamespace(collectonly=False))
    )
    gate.pytest_sessionfinish(session, 0)
    assert session.exitstatus != 0


@pytest.mark.parametrize("profile", ["sqd", "sqd-catalyst"])
def test_only_mock_or_import_tests_cannot_satisfy_execution_gate(runner, profile):
    gate = runner.ExecutionGate(profile)
    nodes = [
        "tests/sqd/test_imports.py::test_public_import_profile[blocked0]",
        "tests/solvation/test_circuit_builder.py::TestBuildQPECircuit::test_h2_probs_mode",
    ]
    # These broad tags qualified in the old gate, even without a real solver/JIT.
    if hasattr(gate, "pytest_collection_modifyitems"):
        gate.pytest_collection_modifyitems(
            [
                SimpleNamespace(nodeid=node, keywords={"sqd", "catalyst", "solvation"})
                for node in nodes
            ]
        )
    for node in nodes:
        gate.pytest_runtest_logreport(SimpleNamespace(nodeid=node, when="call", passed=True))
    session = SimpleNamespace(
        exitstatus=0, config=SimpleNamespace(option=SimpleNamespace(collectonly=False))
    )
    gate.pytest_sessionfinish(session, 0)
    assert session.exitstatus != 0


@pytest.mark.parametrize("profile", ["sqd", "sqd-catalyst"])
def test_real_smoke_nodes_satisfy_execution_gate(runner, profile):
    gate = runner.ExecutionGate(profile)
    nodes = [
        "tests/sqd/test_orchestrator.py::test_full_high_low_same_frame_seed",
        "tests/examples/test_sqd_showcase.py::test_real_h2_tutorial_matches_independent_casci",
    ]
    if profile == "sqd-catalyst":
        nodes.append("tests/test_qpe_circuit.py::TestQPECatalyst::test_qpe_with_catalyst_h2")
    for node in nodes:
        gate.pytest_runtest_logreport(SimpleNamespace(nodeid=node, when="call", passed=True))
    session = SimpleNamespace(
        exitstatus=0, config=SimpleNamespace(option=SimpleNamespace(collectonly=False))
    )
    gate.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 0


def test_sqd_success_without_real_catalyst_smoke_is_failure(runner):
    gate = runner.ExecutionGate("sqd-catalyst")
    gate.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid="tests/sqd/test_orchestrator.py::test_full_high_low_same_frame_seed",
            when="call",
            passed=True,
        )
    )
    session = SimpleNamespace(
        exitstatus=0, config=SimpleNamespace(option=SimpleNamespace(collectonly=False))
    )
    gate.pytest_sessionfinish(session, 0)
    assert session.exitstatus != 0


def test_collection_only_does_not_claim_execution(runner):
    gate = runner.ExecutionGate("sqd-catalyst")
    session = SimpleNamespace(
        exitstatus=0, config=SimpleNamespace(option=SimpleNamespace(collectonly=True))
    )
    gate.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 0
    assert gate.passed == set()


def test_setup_pass_is_not_scientific_execution(runner):
    gate = runner.ExecutionGate("sqd")
    gate.pytest_runtest_logreport(SimpleNamespace(nodeid="science", when="setup", passed=True))
    assert not gate.passed
