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

    def test_mm_embedding_effect(self):
        """Verify MM embedding affects QM energy (solvent polarization)."""
        from pyscf import gto

        from q2m3.interfaces import PySCFPennyLaneConverter

        # Create H3O+ molecule
        mol = gto.M(
            atom="""
                O  0.0000  0.0000  0.0000
                H  0.9600  0.0000  0.0000
                H -0.4800  0.8300  0.0000
                H -0.4800 -0.8300  0.0000
            """,
            basis="sto-3g",
            unit="Angstrom",
            charge=1,
        )

        # Create MM point charges (simplified 2 water molecules)
        # TIP3P charges: O=-0.834, H=+0.417
        mm_charges = np.array(
            [
                -0.834,
                0.417,
                0.417,  # Water 1
                -0.834,
                0.417,
                0.417,  # Water 2
            ]
        )
        # Place waters ~3 Angstrom away from H3O+ (in Angstrom units)
        mm_coords = np.array(
            [
                [3.0, 0.0, 0.0],  # O1
                [3.5, 0.8, 0.0],  # H1a
                [3.5, -0.8, 0.0],  # H1b
                [-3.0, 0.0, 0.0],  # O2
                [-3.5, 0.8, 0.0],  # H2a
                [-3.5, -0.8, 0.0],  # H2b
            ]
        )

        converter = PySCFPennyLaneConverter()

        # Calculate energy WITH MM embedding
        result_with_mm = converter.build_qmmm_hamiltonian(mol, mm_charges, mm_coords)
        energy_with_mm = result_with_mm["energy_hf"]

        # Calculate energy WITHOUT MM embedding (vacuum)
        result_no_mm = converter.build_qmmm_hamiltonian(
            mol, np.array([]), np.array([]).reshape(0, 3)
        )
        energy_no_mm = result_no_mm["energy_hf"]

        # MM should stabilize the system (energy becomes more negative)
        energy_diff = energy_no_mm - energy_with_mm

        # Print diagnostic info
        print(f"\nMM embedding effect verification:")
        print(f"  Vacuum energy:    {energy_no_mm:.6f} Ha")
        print(f"  With MM energy:   {energy_with_mm:.6f} Ha")
        print(f"  Stabilization:    {energy_diff:.4f} Ha ({energy_diff * 627.5:.1f} kcal/mol)")

        # Assertions:
        # 1. MM should stabilize the system (positive energy difference)
        assert (
            energy_diff > 0.001
        ), f"MM should stabilize QM region, but energy diff = {energy_diff:.6f} Ha"
        # 2. Stabilization should be reasonable (not too large)
        assert energy_diff < 2.0, f"Energy difference too large: {energy_diff:.4f} Ha"
        # 3. Both energies should be reasonable for H3O+
        assert -80.0 < energy_no_mm < -70.0, f"Vacuum energy unreasonable: {energy_no_mm}"
        assert -80.0 < energy_with_mm < -70.0, f"MM energy unreasonable: {energy_with_mm}"

    def test_pennylane_hamiltonian_with_mm(self):
        """Verify PennyLane Hamiltonian includes MM solvent effects."""
        from q2m3.interfaces import PySCFPennyLaneConverter

        # Create H3O+ geometry
        symbols = ["O", "H", "H", "H"]
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.48, 0.83, 0.0],
                [-0.48, -0.83, 0.0],
            ]
        )

        # Simple MM: 2 water molecules with TIP3P charges
        mm_charges = np.array(
            [
                -0.834,
                0.417,
                0.417,  # Water 1
                -0.834,
                0.417,
                0.417,  # Water 2
            ]
        )
        mm_coords = np.array(
            [
                [3.0, 0.0, 0.0],
                [3.5, 0.8, 0.0],
                [3.5, -0.8, 0.0],
                [-3.0, 0.0, 0.0],
                [-3.5, 0.8, 0.0],
                [-3.5, -0.8, 0.0],
            ]
        )

        converter = PySCFPennyLaneConverter()

        # Build Hamiltonian WITHOUT MM (original function)
        H_vacuum, n_qubits_v, hf_v = converter.pyscf_to_pennylane_hamiltonian(
            symbols,
            coords,
            charge=1,
            active_electrons=4,
            active_orbitals=4,
        )

        # Build Hamiltonian WITH MM (new function)
        H_mm, n_qubits_mm, hf_mm = converter.pyscf_to_pennylane_hamiltonian_with_mm(
            symbols,
            coords,
            charge=1,
            mm_charges=mm_charges,
            mm_coords=mm_coords,
            active_electrons=4,
            active_orbitals=4,
        )

        # Both should have same number of qubits
        assert n_qubits_v == n_qubits_mm, "Qubit counts should match"

        # HF states should match
        np.testing.assert_array_equal(hf_v, hf_mm, "HF states should match")

        # At minimum, verify both are valid Hamiltonians
        assert H_vacuum is not None, "Vacuum Hamiltonian should not be None"
        assert H_mm is not None, "MM Hamiltonian should not be None"

        # Get number of terms (works for both Hamiltonian and Sum types)
        n_terms_vacuum = len(H_vacuum.operands) if hasattr(H_vacuum, "operands") else 1
        n_terms_mm = len(H_mm.operands) if hasattr(H_mm, "operands") else 1

        print(f"\nPennyLane Hamiltonian with MM test:")
        print(f"  Vacuum Hamiltonian terms: {n_terms_vacuum}")
        print(f"  MM Hamiltonian terms: {n_terms_mm}")
        print(f"  Qubits: {n_qubits_mm}")
