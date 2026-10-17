# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Shared pytest fixtures for q2m3 tests.
"""

import importlib
import importlib.util
import os

# Keep JAX-based tests on CPU even when q2m3[gpu] is installed; explicit
# lightning.gpu tests still exercise the PennyLane GPU backend.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pennylane as qml
import pytest

from q2m3.interfaces import PySCFPennyLaneConverter

# ============================================================================
# Pytest Configuration Hooks
# ============================================================================


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


def profile_settings(profile: str, include_slow: bool = False) -> dict:
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
    return {"ignore": ignored, "markexpr": " and ".join(markers)}


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
        if report.when == "call" and report.passed and not hasattr(report, "wasxfail"):
            self.passed.update(("any", report.nodeid))

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        if not session.config.option.collectonly and self.required - self.passed:
            print(
                f"CI profile did not execute required tests: {sorted(self.required - self.passed)}"
            )
            if not exitstatus:
                session.exitstatus = 1


def pytest_addoption(parser):
    """Add optional CI profiles to the normal pytest entry point."""
    parser.addoption("--q2m3-profile", choices=PROFILES, default=None)
    parser.addoption("--q2m3-include-slow", action="store_true", default=False)


def pytest_configure(config):
    """Check dependencies and apply profile restrictions before collection."""
    profile = config.getoption("q2m3_profile")
    include_slow = config.getoption("q2m3_include_slow")
    if profile is None:
        if include_slow:
            raise pytest.UsageError("--q2m3-include-slow requires --q2m3-profile")
        return
    if getattr(config.option, "numprocesses", None) not in (None, 0):
        raise pytest.UsageError("The profile requires serial execution. Use -n 0.")
    preflight(profile)
    settings = profile_settings(profile, include_slow)
    config.option.ignore = list(dict.fromkeys((config.option.ignore or []) + settings["ignore"]))
    selected = config.option.markexpr
    config.option.markexpr = (
        f"({selected}) and ({settings['markexpr']})" if selected else settings["markexpr"]
    )
    config.pluginmanager.register(ExecutionGate(profile), "q2m3-execution-gate")


def pytest_collection_modifyitems(config, items):
    """Skip catalyst/gpu tests if dependencies are not available."""
    # Check Catalyst availability
    try:
        import catalyst  # noqa: F401

        has_catalyst = True
    except ImportError:
        has_catalyst = False

    # Check lightning.gpu availability
    has_lightning_gpu = False
    try:
        _test_dev = qml.device("lightning.gpu", wires=1)
        del _test_dev
        has_lightning_gpu = True
    except Exception:
        # Any exception means lightning.gpu is not available; safe to ignore for test skipping.
        pass

    # Apply skip markers
    for item in items:
        if "catalyst" in item.keywords and not has_catalyst:
            item.add_marker(pytest.mark.skip(reason="pennylane-catalyst not installed"))
        if "gpu" in item.keywords and not has_lightning_gpu:
            item.add_marker(pytest.mark.skip(reason="lightning.gpu not available"))


# ============================================================================
# Molecule Data Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def h2_molecule_data():
    """H2 molecule data for basic validation."""
    return {
        "symbols": ["H", "H"],
        "coords": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),  # Angstrom
        "charge": 0,
        "n_electrons": 2,
        "expected_hf_energy": -1.117,  # Hartree (STO-3G)
    }


# ============================================================================
# PennyLane Hamiltonian Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def h2_hamiltonian(h2_molecule_data):
    """Generate PennyLane Hamiltonian for H2."""
    converter = PySCFPennyLaneConverter()
    H, n_qubits, hf_state = converter.pyscf_to_pennylane_hamiltonian(
        symbols=h2_molecule_data["symbols"],
        coords=h2_molecule_data["coords"],
        charge=h2_molecule_data["charge"],
    )
    return {
        "hamiltonian": H,
        "n_qubits": n_qubits,
        "hf_state": hf_state,
        "molecule_data": h2_molecule_data,
    }


# ============================================================================
# QPE Configuration Fixtures
# ============================================================================


@pytest.fixture
def qpe_config_basic():
    """Basic QPE configuration for quick tests.

    Uses lightning.qubit for faster test execution.
    Optimized for test speed.
    """
    return {
        "n_estimation_wires": 3,  # Reduced from 4 to 3
        "n_trotter_steps": 2,  # Reduced from 3 to 2
        "base_time": 0.5,
        "n_shots": 5,  # Reduced default shots
        "device_type": "lightning.qubit",
    }


@pytest.fixture
def qpe_config_accurate():
    """More accurate QPE configuration.

    Uses lightning.qubit for faster test execution.
    """
    return {
        "n_estimation_wires": 6,
        "n_trotter_steps": 5,
        "base_time": 0.3,
        "device_type": "lightning.qubit",
    }


# ============================================================================
# Simple Test Hamiltonians
# ============================================================================


@pytest.fixture
def simple_hamiltonian():
    """Simple 2-qubit Hamiltonian for unit testing."""
    coeffs = [0.5, 0.3, 0.2]
    ops = [qml.PauliZ(0), qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)]
    return qml.Hamiltonian(coeffs, ops)


@pytest.fixture
def pauli_z_hamiltonian():
    """Single PauliZ Hamiltonian for phase testing.

    Ground state energy: -1.0 (|1⟩ state)
    Excited state energy: +1.0 (|0⟩ state)
    """
    return qml.Hamiltonian([1.0], [qml.PauliZ(0)])


# ============================================================================
# QPE Integration Test Configurations
# ============================================================================


@pytest.fixture
def h2_qpe_config():
    """H2 QPE configuration for fast tests.

    Uses lightning.qubit for faster test execution.
    Optimized for test speed: minimal shots, low iterations.
    """
    return {
        "algorithm": "standard",
        "iterations": 3,  # Reduced from 4 to 3
        "mapping": "jordan_wigner",
        "system_qubits": 4,
        "error_tolerance": 0.02,  # Relaxed from 0.01 to 0.02
        "use_real_qpe": True,
        "n_estimation_wires": 3,  # Reduced from 4 to 3
        "base_time": 0.5,
        "n_trotter_steps": 2,  # Reduced from 3 to 2
        "n_shots": 5,  # Reduced from 10 to 5
        "energy_warning_threshold": 5.0,  # Relaxed threshold
        "device_type": "lightning.qubit",
    }


@pytest.fixture
def h2_classical_config():
    """H2 classical (HF) configuration for fallback tests."""
    return {
        "algorithm": "standard",
        "iterations": 4,
        "mapping": "jordan_wigner",
        "system_qubits": 4,
        "error_tolerance": 0.01,
        "use_real_qpe": False,  # Use classical HF
    }
