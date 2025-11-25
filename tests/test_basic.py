# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Basic tests for Quantum-QM/MM framework.
"""

import numpy as np
import pytest

from q2m3.core import QMMMSystem, QPEEngine
from q2m3.core.qmmm_system import Atom
from q2m3.interfaces import UnifiedDensityMatrix


class TestQMMMSystem:
    """Test QM/MM system builder."""

    def test_system_initialization(self):
        """Test basic system initialization."""
        # Create H3O+ atoms
        atoms = [
            Atom("O", np.array([0.0, 0.0, 0.0]), charge=-2.0),
            Atom("H", np.array([0.96, 0.0, 0.0]), charge=1.0),
            Atom("H", np.array([-0.48, 0.83, 0.0]), charge=1.0),
            Atom("H", np.array([-0.48, -0.83, 0.0]), charge=1.0),
        ]

        system = QMMMSystem(qm_atoms=atoms, num_waters=8)

        assert len(system.qm_atoms) == 4
        assert system.get_total_charge() == 1  # H3O+ is +1
        assert system.num_waters == 8

    def test_pyscf_conversion(self):
        """Test conversion to PySCF format."""
        atoms = [
            Atom("O", np.array([0.0, 0.0, 0.0])),
            Atom("H", np.array([0.96, 0.0, 0.0])),
        ]

        system = QMMMSystem(qm_atoms=atoms, num_waters=0)
        mol_dict = system.to_pyscf_mol()

        assert "atom" in mol_dict
        assert mol_dict["basis"] == "sto-3g"
        assert "O 0.000000" in mol_dict["atom"]


class TestQPEEngine:
    """Test QPE engine."""

    def test_engine_initialization(self):
        """Test QPE engine initialization."""
        engine = QPEEngine(n_qubits=12, n_iterations=8, mapping="jordan_wigner")

        assert engine.n_qubits == 12
        assert engine.n_iterations == 8
        assert engine.mapping == "jordan_wigner"

    def test_estimate_ground_state_energy(self):
        """Test QPE ground state energy estimation with classical simulation."""
        from pyscf import gto

        from q2m3.interfaces import PySCFPennyLaneConverter

        # Create simple H2 molecule for testing
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="Angstrom")

        engine = QPEEngine(n_qubits=4, n_iterations=5, mapping="jordan_wigner")
        converter = PySCFPennyLaneConverter()

        # Build hamiltonian using converter
        hamiltonian_data = converter.build_qmmm_hamiltonian(mol, np.array([]), np.array([]))

        # Test that energy estimation returns a reasonable value
        result = engine.estimate_ground_state_energy(hamiltonian_data)

        assert "energy" in result
        assert "convergence" in result
        assert isinstance(result["energy"], float)
        assert result["convergence"]["converged"] is True
        assert result["convergence"]["iterations"] <= 5
        # Energy should be negative for bonded H2 (around -1.1 Hartree)
        assert -2.0 < result["energy"] < 0.0

    def test_invalid_mapping(self):
        """Test that invalid mapping raises error."""
        # This test would be implemented when validation is added
        pass


class TestUnifiedDensityMatrix:
    """Test unified density matrix interface."""

    def test_density_matrix_validation(self):
        """Test density matrix validation."""
        # Valid Hermitian matrix
        dm = np.array([[1.0, 0.5j], [-0.5j, 0.0]])

        udm = UnifiedDensityMatrix(dm)
        assert udm.source == "pyscf"

    def test_invalid_density_matrix(self):
        """Test that non-Hermitian matrix raises error."""
        # Non-Hermitian matrix
        dm = np.array([[1.0, 1.0], [0.0, 1.0]])

        with pytest.raises(ValueError, match="Hermitian"):
            UnifiedDensityMatrix(dm)

    def test_from_quantum_state(self):
        """Test density matrix creation from quantum state."""
        state = np.array([1.0, 0.0])  # |0> state

        udm = UnifiedDensityMatrix(np.zeros((2, 2)))
        udm.from_quantum_state(state)

        expected = np.outer(state, state)
        np.testing.assert_allclose(udm.dm_data, expected)
        assert udm.source == "pennylane"


@pytest.mark.parametrize(
    "n_qubits,iterations",
    [
        (12, 5),
        (16, 8),
        (20, 10),
    ],
)
def test_qpe_configurations(n_qubits, iterations):
    """Test various QPE configurations for POC."""
    engine = QPEEngine(n_qubits=n_qubits, n_iterations=iterations)
    assert engine.n_qubits == n_qubits
    assert engine.n_iterations == iterations


class TestQuantumQMMM:
    """Test main QuantumQMMM interface."""

    def test_compute_ground_state(self):
        """Test full QM/MM ground state computation."""
        from q2m3.core import QuantumQMMM

        # Create H3O+ atoms
        atoms = [
            Atom("O", np.array([0.0, 0.0, 0.0]), charge=-2.0),
            Atom("H", np.array([0.96, 0.0, 0.0]), charge=1.0),
            Atom("H", np.array([-0.48, 0.83, 0.0]), charge=1.0),
            Atom("H", np.array([-0.48, -0.83, 0.0]), charge=1.0),
        ]

        qpe_config = {
            "algorithm": "iterative",
            "iterations": 5,
            "mapping": "jordan_wigner",
            "system_qubits": 12,
            "error_tolerance": 0.005,
            "use_real_qpe": False,  # Use classical HF for fast test
        }

        qmmm = QuantumQMMM(qm_atoms=atoms, mm_waters=4, qpe_config=qpe_config)
        results = qmmm.compute_ground_state()

        # Verify result structure
        assert "energy" in results
        assert "density_matrix" in results
        assert "atomic_charges" in results
        assert "convergence" in results

        # Verify convergence info
        assert results["convergence"]["converged"] is True
        assert results["convergence"]["iterations"] > 0

        # Energy should be reasonable for H3O+
        assert isinstance(results["energy"], float)
        assert results["energy"] < 0  # Should be negative

        # Atomic charges should sum to +1 (H3O+ total charge)
        if results["atomic_charges"]:
            total_charge = sum(results["atomic_charges"].values())
            assert abs(total_charge - 1.0) < 0.5  # Allow some tolerance


class TestPySCFPennyLaneConverter:
    """Test PySCF to PennyLane converter."""

    def test_build_qmmm_hamiltonian(self):
        """Test QM/MM Hamiltonian construction."""
        from pyscf import gto

        from q2m3.interfaces import PySCFPennyLaneConverter

        # Create simple water molecule
        mol = gto.M(
            atom="O 0 0 0; H 0.96 0 0; H -0.24 0.93 0", basis="sto-3g", unit="Angstrom", charge=0
        )

        converter = PySCFPennyLaneConverter()

        # No MM charges for basic test
        hamiltonian_data = converter.build_qmmm_hamiltonian(mol, np.array([]), np.array([]))

        # Should return a dictionary with necessary data
        assert hamiltonian_data is not None
        assert "mol" in hamiltonian_data
        assert "energy_nuc" in hamiltonian_data
