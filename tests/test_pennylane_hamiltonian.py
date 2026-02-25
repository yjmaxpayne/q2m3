# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
TDD tests for PennyLane Hamiltonian generation.

Phase 1: pyscf_to_pennylane_hamiltonian() implementation
"""

import numpy as np
import pennylane as qml
import pytest

from q2m3.interfaces import PySCFPennyLaneConverter


class TestPennyLaneHamiltonianBasic:
    """P0: Basic Hamiltonian generation tests."""

    def test_h2_hamiltonian_basic(self):
        """Test Hamiltonian generation for H2 molecule."""
        converter = PySCFPennyLaneConverter()
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])  # Angstrom

        H, n_qubits, hf_state = converter.pyscf_to_pennylane_hamiltonian(symbols, coords, charge=0)

        # Hamiltonian should be valid
        assert H is not None
        assert isinstance(H, qml.Hamiltonian | qml.ops.Sum)

        # H2 STO-3G: 2 spatial orbitals * 2 spins = 4 qubits
        assert n_qubits == 4

        # HF state should have correct length and electron count
        assert len(hf_state) == n_qubits
        assert sum(hf_state) == 2  # 2 electrons in H2

    def test_hamiltonian_return_types(self):
        """Test correct return types from pyscf_to_pennylane_hamiltonian."""
        converter = PySCFPennyLaneConverter()
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        result = converter.pyscf_to_pennylane_hamiltonian(symbols, coords, charge=0)

        # Should return tuple of 3 elements
        assert isinstance(result, tuple)
        assert len(result) == 3

        H, n_qubits, hf_state = result

        # Type checks
        assert isinstance(n_qubits, int)
        assert isinstance(hf_state, np.ndarray)

    def test_hf_state_correctness(self):
        """Test HF state has correct electron occupation."""
        converter = PySCFPennyLaneConverter()
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        _, n_qubits, hf_state = converter.pyscf_to_pennylane_hamiltonian(symbols, coords, charge=0)

        # HF state should be binary (0 or 1)
        assert all(x in [0, 1] for x in hf_state)

        # First 2 qubits should be occupied (Jordan-Wigner, H2 has 2 electrons)
        expected_hf = np.array([1, 1, 0, 0])
        np.testing.assert_array_equal(hf_state, expected_hf)


class TestPennyLaneHamiltonianValidation:
    """P2: Input validation and edge cases."""

    def test_invalid_symbols_coords_mismatch(self):
        """Test error when symbols and coords don't match."""
        converter = PySCFPennyLaneConverter()
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0]])  # Only 1 coord for 2 symbols

        with pytest.raises(ValueError, match="symbols.*coords"):
            converter.pyscf_to_pennylane_hamiltonian(symbols, coords, charge=0)

    def test_invalid_coords_shape(self):
        """Test error for invalid coordinate shape."""
        converter = PySCFPennyLaneConverter()
        symbols = ["H", "H"]
        coords = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.74])  # 1D array

        # Should handle 1D or raise error
        # Actually, we should accept 1D if length is correct (n_atoms * 3)
        result = converter.pyscf_to_pennylane_hamiltonian(symbols, coords, charge=0)
        assert result is not None  # Should work with flattened coords


class TestPennyLaneHamiltonianEnergy:
    """P2: Energy sanity checks."""

    def test_h2_hamiltonian_energy_bounds(self):
        """Test H2 Hamiltonian gives reasonable energy."""
        converter = PySCFPennyLaneConverter()
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        H, n_qubits, hf_state = converter.pyscf_to_pennylane_hamiltonian(symbols, coords, charge=0)

        # Create device and compute HF energy expectation
        dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def hf_energy():
            qml.BasisState(hf_state, wires=range(n_qubits))
            return qml.expval(H)

        energy = hf_energy()

        # H2 HF/STO-3G energy should be around -1.1 Hartree
        assert -2.0 < energy < 0.0
        assert abs(energy - (-1.117)) < 0.1  # Allow some tolerance
