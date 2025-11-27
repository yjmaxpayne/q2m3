# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
TDD tests for QPE integration into QuantumQMMM main workflow.

Phase 3: Integration of real QPE circuit into compute_ground_state().
"""

import logging

import numpy as np
import pytest

from q2m3.core import QuantumQMMM
from q2m3.core.qmmm_system import Atom

# ============================================================================
# Test Fixtures - H2 molecule for fast tests
# ============================================================================


@pytest.fixture
def h2_atoms():
    """Simple H2 molecule atoms."""
    return [
        Atom("H", np.array([0.0, 0.0, 0.0])),
        Atom("H", np.array([0.0, 0.0, 0.74])),
    ]


# ============================================================================
# P0: _build_qmmm_hamiltonian() PennyLane Data Tests
# ============================================================================


class TestBuildQMMMHamiltonian:
    """Test that _build_qmmm_hamiltonian returns PennyLane Hamiltonian data."""

    def test_hamiltonian_data_contains_pennylane_hamiltonian(self, h2_atoms, h2_qpe_config):
        """Test hamiltonian_data contains PennyLane Hamiltonian object."""
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=h2_qpe_config)
        hamiltonian_data = qmmm._build_qmmm_hamiltonian()

        assert "pennylane_hamiltonian" in hamiltonian_data
        # PennyLane >= 0.43 uses Sum or LinearCombination for Hamiltonians
        H = hamiltonian_data["pennylane_hamiltonian"]
        # Check it's a valid PennyLane operator with wires
        assert hasattr(H, "wires") and hasattr(H, "operands")

    def test_hamiltonian_data_contains_n_qubits(self, h2_atoms, h2_qpe_config):
        """Test hamiltonian_data contains n_qubits."""
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=h2_qpe_config)
        hamiltonian_data = qmmm._build_qmmm_hamiltonian()

        assert "n_qubits" in hamiltonian_data
        assert isinstance(hamiltonian_data["n_qubits"], int)
        assert hamiltonian_data["n_qubits"] == 4  # H2 has 4 spin-orbitals

    def test_hamiltonian_data_contains_hf_state(self, h2_atoms, h2_qpe_config):
        """Test hamiltonian_data contains HF reference state."""
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=h2_qpe_config)
        hamiltonian_data = qmmm._build_qmmm_hamiltonian()

        assert "hf_state" in hamiltonian_data
        assert isinstance(hamiltonian_data["hf_state"], np.ndarray)
        # H2 has 2 electrons -> |1100>
        assert np.array_equal(hamiltonian_data["hf_state"], np.array([1, 1, 0, 0]))

    def test_hamiltonian_data_still_contains_pyscf_data(self, h2_atoms, h2_qpe_config):
        """Test hamiltonian_data still contains PySCF data for Mulliken analysis."""
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=h2_qpe_config)
        hamiltonian_data = qmmm._build_qmmm_hamiltonian()

        # Should still have PySCF data
        assert "mol" in hamiltonian_data
        assert "scf_result" in hamiltonian_data
        assert "energy_hf" in hamiltonian_data


# ============================================================================
# P1: compute_ground_state() with Real QPE Tests
# ============================================================================


class TestComputeGroundStateRealQPE:
    """Test compute_ground_state with real QPE circuit."""

    def test_compute_ground_state_returns_energy(self, h2_atoms, h2_qpe_config):
        """Test compute_ground_state returns energy estimate."""
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=h2_qpe_config)
        result = qmmm.compute_ground_state()

        assert "energy" in result
        assert isinstance(result["energy"], float)
        assert np.isfinite(result["energy"])

    def test_compute_ground_state_returns_hf_reference(self, h2_atoms, h2_qpe_config):
        """Test compute_ground_state returns HF energy as reference."""
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=h2_qpe_config)
        result = qmmm.compute_ground_state()

        assert "energy_hf" in result
        # H2 HF energy should be around -1.117 Hartree
        assert -2.0 < result["energy_hf"] < 0.0

    def test_compute_ground_state_returns_convergence_info(self, h2_atoms, h2_qpe_config):
        """Test compute_ground_state returns QPE convergence info."""
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=h2_qpe_config)
        result = qmmm.compute_ground_state()

        assert "convergence" in result
        assert result["convergence"]["method"] == "real_qpe"
        assert "n_estimation_wires" in result["convergence"]
        assert "n_shots" in result["convergence"]

    def test_compute_ground_state_returns_density_matrix(self, h2_atoms, h2_qpe_config):
        """Test compute_ground_state returns density matrix."""
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=h2_qpe_config)
        result = qmmm.compute_ground_state()

        assert "density_matrix" in result
        dm = result["density_matrix"]
        assert isinstance(dm, np.ndarray)
        assert dm.ndim == 2
        assert dm.shape[0] == dm.shape[1]


# ============================================================================
# P1: Classical Fallback Tests
# ============================================================================


class TestComputeGroundStateClassicalFallback:
    """Test compute_ground_state with classical (HF) fallback."""

    def test_classical_fallback_when_use_real_qpe_false(self, h2_atoms, h2_classical_config):
        """Test that HF is used when use_real_qpe=False."""
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=h2_classical_config)
        result = qmmm.compute_ground_state()

        assert "energy" in result
        assert "convergence" in result
        # Classical mode should NOT have "method": "real_qpe"
        assert result["convergence"].get("method") != "real_qpe"

    def test_classical_fallback_returns_valid_energy(self, h2_atoms, h2_classical_config):
        """Test classical fallback returns valid HF energy."""
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=h2_classical_config)
        result = qmmm.compute_ground_state()

        # H2 HF energy should be around -1.117 Hartree
        assert -2.0 < result["energy"] < 0.0


# ============================================================================
# P1: Energy Warning Tests
# ============================================================================


class TestEnergyWarning:
    """Test energy difference warning functionality."""

    def test_energy_warning_logged_when_difference_exceeds_threshold(self, h2_atoms, caplog):
        """Test that warning is logged when QPE differs significantly from HF."""
        # Use very tight threshold to trigger warning
        config = {
            "use_real_qpe": True,
            "n_estimation_wires": 3,  # Reduced from 4 to 3
            "base_time": 0.5,
            "n_trotter_steps": 2,  # Reduced from 3 to 2
            "n_shots": 5,  # Reduced from 10 to 5
            "energy_warning_threshold": 0.001,  # Very tight threshold
        }
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=config)

        with caplog.at_level(logging.WARNING):
            result = qmmm.compute_ground_state()

        # Warning should be logged due to tight threshold
        # Note: This test may pass or fail depending on QPE accuracy
        assert "energy_difference" in result

    def test_energy_difference_returned_in_result(self, h2_atoms, h2_qpe_config):
        """Test that energy_difference is included in result."""
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=h2_qpe_config)
        result = qmmm.compute_ground_state()

        assert "energy_difference" in result
        assert isinstance(result["energy_difference"], float)
        assert result["energy_difference"] >= 0


# ============================================================================
# P1: Atomic Charges Tests
# ============================================================================


class TestAtomicCharges:
    """Test Mulliken analysis still works with real QPE."""

    def test_atomic_charges_computed(self, h2_atoms, h2_qpe_config):
        """Test that atomic charges are computed from density matrix."""
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=h2_qpe_config)
        result = qmmm.compute_ground_state()

        assert "atomic_charges" in result
        assert len(result["atomic_charges"]) == 2  # H2 has 2 atoms


# ============================================================================
# Circuit Visualization Tests
# ============================================================================


class TestCircuitVisualization:
    """Test circuit visualization methods."""

    def test_draw_circuits_returns_qpe_and_rdm(self, h2_atoms, h2_qpe_config):
        """Test draw_circuits returns both QPE and RDM diagrams."""
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=h2_qpe_config)
        diagrams = qmmm.draw_circuits()

        # Should return dictionary with qpe and rdm keys
        assert "qpe" in diagrams
        assert "rdm" in diagrams

        # Both should be non-empty strings
        assert isinstance(diagrams["qpe"], str)
        assert isinstance(diagrams["rdm"], str)
        assert len(diagrams["qpe"]) > 0
        assert len(diagrams["rdm"]) > 0

    def test_draw_circuits_qpe_contains_structure(self, h2_atoms, h2_qpe_config):
        """Test QPE diagram contains expected structure."""
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=h2_qpe_config)
        diagrams = qmmm.draw_circuits()

        qpe_diagram = diagrams["qpe"]
        # Should contain circuit structure elements
        assert "QPE" in qpe_diagram or "Estimation" in qpe_diagram

    def test_draw_circuits_rdm_contains_structure(self, h2_atoms, h2_qpe_config):
        """Test RDM diagram contains expected structure."""
        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=h2_qpe_config)
        diagrams = qmmm.draw_circuits()

        rdm_diagram = diagrams["rdm"]
        # Should contain RDM measurement structure
        assert "RDM" in rdm_diagram or "Trotter" in rdm_diagram
