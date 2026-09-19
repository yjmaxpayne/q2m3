"""Run explicit installed-dependency CI profiles, failing before missing-extra skips.

Run from the checkout root with ``PYTHONPATH=src python tools/sqd/ci_profile.py
--profile core|sqd|sqd-catalyst``. The core profile permits development tooling,
but no SQD/Catalyst extras. Use a separate environment for each profile.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path

PROFILES = ("core", "sqd", "sqd-catalyst")
SQD_MODULES = ("ffsim", "qiskit", "qiskit_addon_sqd")
# These modules contain only SQD scientific tests. Mixed modules (including
# imports, configuration, result and dependency probes) stay collected in core.
SQD_SCIENCE = (
    "tests/sqd/test_ansatz.py",
    "tests/sqd/test_sampling.py",
    "tests/sqd/test_diagonalize.py",
    "tests/sqd/test_reference.py",
    "tests/sqd/test_observables.py",
    "tests/sqd/test_orchestrator.py",
    "tests/sqd/test_embedding.py",
    "tests/examples/test_sqd_benchmarks.py",
    "tests/examples/test_orbital_basis_scan.py",
    "tests/examples/test_connectivity_comparison.py",
)


def preflight(profile: str) -> None:
    """Import required extras and reject contaminated absence profiles.

    Raises:
        ImportError: A required dependency cannot actually be imported.
        RuntimeError: An absent-extra profile contains the extra.
    """
    required = SQD_MODULES if profile != "core" else ()
    if profile == "sqd-catalyst":
        required += ("catalyst",)
    forbidden = SQD_MODULES + ("catalyst",) if profile == "core" else ()
    if profile == "sqd":
        forbidden = ("catalyst",)
    for name in required:
        importlib.import_module(name)
    for name in forbidden:
        if importlib.util.find_spec(name) is not None:
            raise RuntimeError(f"{profile} profile requires {name} to be absent")


def pytest_args(profile: str, *, include_slow: bool = False) -> list[str]:
    """Build serial test selection, excluding unavailable imports before collection."""
    markers = ["not gpu"] if include_slow else ["not slow", "not gpu"]
    ignored = []
    if profile != "sqd-catalyst":
        # Its conftest imports circuit_builder before marker deselection happens.
        ignored.append("tests/solvation")
        markers.extend(("not catalyst", "not solvation"))
    if profile == "core":
        ignored.extend(SQD_SCIENCE)
        markers.append("not sqd")
    return [
        "tests/",
        "-n",
        "0",
        "-ra",
        *[f"--ignore={path}" for path in ignored],
        "-m",
        " and ".join(markers),
    ]


class ExecutionGate:
    """Require successful real solver/JIT smoke nodes, not merely broad markers.

    The SQD node runs both real entry points and compares energies and JSON.
    The Catalyst node builds and executes an H2 QPE circuit with use_catalyst=True;
    unlike the solvation builder unit tests, it does not replace qjit with a mock.
    Renaming either test requires deliberately updating this fail-closed contract.
    """

    def __init__(self, profile: str) -> None:
        self.required = {"any"}
        if profile != "core":
            self.required.add("tests/sqd/test_orchestrator.py::test_full_high_low_same_frame_seed")
            self.required.add(
                "tests/examples/test_sqd_showcase.py::test_real_h2_tutorial_matches_independent_casci"
            )
        if profile == "sqd-catalyst":
            self.required.add(
                "tests/test_qpe_circuit.py::TestQPECatalyst::test_qpe_with_catalyst_h2"
            )
        self.passed: set[str] = set()

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call" and report.passed:
            self.passed.update(("any", report.nodeid))

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        if not session.config.option.collectonly and self.required - self.passed:
            print(
                f"CI profile did not execute required tests: {sorted(self.required - self.passed)}"
            )
            session.exitstatus = 1


def main() -> int:
    """Validate installed dependencies and run the selected test profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--include-slow", action="store_true")
    parser.add_argument("--coverage", action="store_true")
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    preflight(args.profile)
    import pytest

    options = pytest_args(args.profile, include_slow=args.include_slow)
    if args.collect_only:
        options += ["--collect-only", "--no-cov"]
    elif args.coverage:
        options += ["--cov=src/q2m3", "--cov-report=xml", "--cov-report=term"]
    else:
        options += ["--no-cov"]
    return int(pytest.main(options, plugins=[ExecutionGate(args.profile)]))


if __name__ == "__main__":
    raise SystemExit(main())
