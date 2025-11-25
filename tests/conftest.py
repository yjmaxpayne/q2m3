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
    """Skip catalyst tests if pennylane-catalyst is not installed."""
    try:
        import catalyst  # noqa: F401

        # Catalyst is available, run all tests
    except ImportError:
        skip_catalyst = pytest.mark.skip(reason="pennylane-catalyst not installed")
        for item in items:
            if "catalyst" in item.keywords:
                item.add_marker(skip_catalyst)


# ============================================================================
# Molecule Data Fixtures
# ============================================================================


@pytest.fixture
def h2_molecule_data():
    """H2 molecule data for basic validation."""
    return {
        "symbols": ["H", "H"],
        "coords": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),  # Angstrom
        "charge": 0,
        "n_electrons": 2,
        "expected_hf_energy": -1.117,  # Hartree (STO-3G)
    }


@pytest.fixture
def h3o_molecule_data():
    """H3O+ hydronium ion data for target system validation."""
    return {
        "symbols": ["O", "H", "H", "H"],
        "coords": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.48, 0.83, 0.0],
                [-0.48, -0.83, 0.0],
            ]
        ),  # Angstrom
        "charge": 1,
        "n_electrons": 10,  # O(8) + 3*H(1) - charge(1)
        "expected_hf_energy": -75.3,  # Hartree (STO-3G, approximate)
    }


# ============================================================================
# PennyLane Hamiltonian Fixtures
# ============================================================================


@pytest.fixture
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


@pytest.fixture
def h3o_hamiltonian(h3o_molecule_data):
    """Generate PennyLane Hamiltonian for H3O+ (full space - expensive)."""
    converter = PySCFPennyLaneConverter()
    H, n_qubits, hf_state = converter.pyscf_to_pennylane_hamiltonian(
        symbols=h3o_molecule_data["symbols"],
        coords=h3o_molecule_data["coords"],
        charge=h3o_molecule_data["charge"],
    )
    return {
        "hamiltonian": H,
        "n_qubits": n_qubits,
        "hf_state": hf_state,
        "molecule_data": h3o_molecule_data,
    }


@pytest.fixture
def h3o_hamiltonian_active_space(h3o_molecule_data):
    """Generate PennyLane Hamiltonian for H3O+ with active space (cheaper).

    Uses 4 active electrons in 4 active orbitals = 8 qubits.
    This makes QPE simulation feasible on typical hardware.
    """
    converter = PySCFPennyLaneConverter()
    H, n_qubits, hf_state = converter.pyscf_to_pennylane_hamiltonian(
        symbols=h3o_molecule_data["symbols"],
        coords=h3o_molecule_data["coords"],
        charge=h3o_molecule_data["charge"],
        active_electrons=4,
        active_orbitals=4,
    )
    return {
        "hamiltonian": H,
        "n_qubits": n_qubits,
        "hf_state": hf_state,
        "molecule_data": h3o_molecule_data,
        "active_space": {"electrons": 4, "orbitals": 4},
    }


# ============================================================================
# QPE Configuration Fixtures
# ============================================================================


@pytest.fixture
def qpe_config_basic():
    """Basic QPE configuration for quick tests."""
    return {
        "n_estimation_wires": 4,
        "n_trotter_steps": 3,
        "base_time": 0.5,
    }


@pytest.fixture
def qpe_config_accurate():
    """More accurate QPE configuration."""
    return {
        "n_estimation_wires": 6,
        "n_trotter_steps": 5,
        "base_time": 0.3,
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
def h3o_qpe_config():
    """H3O+ QPE configuration with active space via fixture."""
    return {
        "algorithm": "standard",
        "iterations": 8,
        "mapping": "jordan_wigner",
        "system_qubits": 8,
        "error_tolerance": 0.005,
        "use_real_qpe": True,
        "n_estimation_wires": 3,
        "base_time": 0.3,
        "n_trotter_steps": 2,
        "n_shots": 10,
        "active_electrons": 4,
        "active_orbitals": 4,
        "energy_warning_threshold": 5.0,  # Relaxed threshold for testing
    }


@pytest.fixture
def h2_qpe_config():
    """H2 QPE configuration for fast tests."""
    return {
        "algorithm": "standard",
        "iterations": 4,
        "mapping": "jordan_wigner",
        "system_qubits": 4,
        "error_tolerance": 0.01,
        "use_real_qpe": True,
        "n_estimation_wires": 4,
        "base_time": 0.5,
        "n_trotter_steps": 3,
        "n_shots": 10,
        "energy_warning_threshold": 2.0,
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
