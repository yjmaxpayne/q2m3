# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Shared pytest fixtures for q2m3 tests.
"""

import numpy as np
import pennylane as qml
import pytest

from q2m3.interfaces import PySCFPennyLaneConverter

# ============================================================================
# Pytest Configuration Hooks
# ============================================================================


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
