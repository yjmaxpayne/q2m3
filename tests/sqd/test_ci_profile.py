"""Dependency profiles must fail closed before pytest can skip missing extras."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[2] / "tests/conftest.py"
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
    settings = runner.profile_settings("core")
    assert "tests/solvation" in settings["ignore"]
    assert "tests/sqd/test_sampling.py" in settings["ignore"]
    assert "tests/sqd" not in settings["ignore"]
    for name in ("test_config.py", "test_result.py", "test_resources.py"):
        assert f"tests/sqd/{name}" not in settings["ignore"]
    assert "not sqd" in settings["markexpr"]


def test_full_profile_keeps_science_and_catalyst(runner):
    settings = runner.profile_settings("sqd-catalyst")
    assert settings == {"ignore": [], "markexpr": "not slow and not gpu"}
    assert runner.profile_settings("sqd-catalyst", include_slow=True)["markexpr"] == "not gpu"


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


pytest_plugins = ["pytester"]


@pytest.fixture
def isolated(pytester, monkeypatch):
    """Copy the real hooks and control only dependency discovery in the child process."""
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    source = (Path(__file__).resolve().parents[1] / "conftest.py").read_text()
    pytester.makeconftest(
        source + "\nimport importlib.util\nimportlib.util.find_spec = lambda name: None\n"
    )
    pytester.makeini(
        "[pytest]\nmarkers =\n    slow\n    gpu\n    catalyst\n    solvation\n    sqd\n"
    )
    pytester.makepyfile(test_smoke="def test_pass(): pass")
    return pytester


def run_profile(isolated, *args):
    return isolated.runpytest_subprocess("-p", "xdist.plugin", "-n", "0", *args)


def test_profile_rejects_unknown_name(isolated):
    result = run_profile(isolated, "--q2m3-profile=unknown")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*invalid choice*unknown*"])


def test_include_slow_requires_profile(isolated):
    result = run_profile(isolated, "--q2m3-include-slow")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--q2m3-include-slow requires --q2m3-profile*"])


def test_profile_rejects_parallel_execution(isolated):
    for workers in ("2", "auto"):
        result = run_profile(isolated, "--q2m3-profile=core", "-n", workers)
        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines(["*profile requires serial execution*"])


def test_profile_combines_explicit_markers(isolated):
    isolated.makepyfile(test_selection="""
        import pytest
        @pytest.mark.sqd
        def test_sqd(): assert False
        @pytest.mark.slow
        def test_slow(): pass
        @pytest.mark.gpu
        def test_gpu(): assert False
    """)
    result = run_profile(isolated, "--q2m3-profile=core", "-m", "slow or sqd or gpu")
    assert result.ret != 0
    result.assert_outcomes(deselected=4)
    result = run_profile(
        isolated, "--q2m3-profile=core", "--q2m3-include-slow", "-m", "slow or sqd or gpu"
    )
    assert result.ret == 0
    result.assert_outcomes(passed=1, deselected=3)


def test_profile_excludes_before_module_import(isolated):
    for path in ("tests/solvation/conftest.py", "tests/sqd/test_sampling.py", "custom/test_bad.py"):
        target = isolated.path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("raise AssertionError('excluded module imported')")
    result = run_profile(isolated, "--q2m3-profile=core", "--ignore=custom")
    assert result.ret == 0
    result.assert_outcomes(passed=1)


def test_collection_only_still_checks_dependencies(isolated):
    conftest = isolated.path / "conftest.py"
    source = conftest.read_text()
    conftest.write_text(source + "\nimportlib.util.find_spec = lambda name: object()\n")
    result = run_profile(isolated, "--q2m3-profile=core", "--collect-only")
    assert result.ret != 0
    result.stderr.fnmatch_lines(["*core profile requires ffsim to be absent*"])
    conftest.write_text(source)
    result = run_profile(isolated, "--q2m3-profile=core", "--collect-only")
    assert result.ret == 0
    assert "CI profile did not execute" not in result.stdout.str()


def test_missing_required_node_fails(isolated):
    conftest = isolated.path / "conftest.py"
    conftest.write_text(conftest.read_text() + "\ndef preflight(profile): pass\n")
    result = run_profile(isolated, "--q2m3-profile=sqd")
    assert result.ret != 0
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(
        ["*CI profile did not execute required tests:*test_full_high_low_same_frame_seed*"]
    )


def test_xfail_and_xpass_do_not_satisfy_gate(isolated):
    isolated.makepyfile(test_smoke="""
        import pytest
        @pytest.mark.xfail
        def test_xfail(): assert False
        @pytest.mark.xfail
        def test_xpass(): pass
    """)
    result = run_profile(isolated, "--q2m3-profile=core")
    result.assert_outcomes(xfailed=1, xpassed=1)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*CI profile did not execute required tests:*any*"])


def test_gate_state_is_isolated_between_sessions(runner, monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda profile: None)

    def config():
        return pytest.Config.fromdictargs({}, [])

    first, second = config(), config()
    try:
        for current in (first, second):
            current.option.q2m3_profile = "sqd-catalyst"
            current.option.q2m3_include_slow = False
            current.option.numprocesses = 0
            runner.pytest_configure(current)
        gate1 = first.pluginmanager.get_plugin("q2m3-execution-gate")
        gate2 = second.pluginmanager.get_plugin("q2m3-execution-gate")
        assert gate1 is not gate2
        gate1.pytest_runtest_logreport(SimpleNamespace(nodeid="smoke", when="call", passed=True))
        assert gate1.passed
        assert not gate2.passed
    finally:
        first._ensure_unconfigure()
        second._ensure_unconfigure()
