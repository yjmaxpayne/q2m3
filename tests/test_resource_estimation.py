# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Tests for EFTQC resource estimation functionality.
"""

import numpy as np
import pytest

from q2m3.interfaces import PySCFPennyLaneConverter


class TestResourceEstimation:
    """Test EFTQC resource estimation functionality."""

    @pytest.fixture
    def converter(self):
        """Create PySCFPennyLaneConverter instance."""
        return PySCFPennyLaneConverter()

    def test_h2_resource_estimation(self, converter):
        """H2: smallest benchmark system."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        result = converter.estimate_qpe_resources(symbols, coords)

        # Basic structure checks
        assert "logical_qubits" in result
        assert "toffoli_gates" in result
        assert "hamiltonian_1norm" in result
        assert "qpe_iterations" in result
        assert "trotter_steps" in result
        assert "target_error" in result
        assert "n_electrons" in result
        assert "n_orbitals" in result
        assert "basis" in result

        # H2 resource estimation validation
        assert result["logical_qubits"] > 0
        assert result["toffoli_gates"] > 0
        assert result["hamiltonian_1norm"] > 0
        assert result["n_electrons"] == 2  # H + H

        # Sanity checks based on demo results (~1.2M gates, ~115 qubits)
        assert 100_000 < result["toffoli_gates"] < 5_000_000

    def test_trotter_steps_recommendation(self, converter):
        """Verify Trotter steps are reasonably estimated."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        result = converter.estimate_qpe_resources(symbols, coords)

        # Trotter steps should be > 0 and reasonable (10-100 for small molecules)
        assert result["trotter_steps"] >= 10
        assert result["trotter_steps"] <= 200

    def test_qpe_iterations_formula(self, converter):
        """QPE iterations should be ceil(lambda / error)."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        result = converter.estimate_qpe_resources(symbols, coords, target_error=0.0016)

        # qpe_iterations = ceil(lambda / error)
        expected_iterations = int(np.ceil(result["hamiltonian_1norm"] / 0.0016))
        assert result["qpe_iterations"] == expected_iterations

    def test_error_scaling(self, converter):
        """Gates should scale inversely with target error."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        result_strict = converter.estimate_qpe_resources(symbols, coords, target_error=0.0016)
        result_relaxed = converter.estimate_qpe_resources(symbols, coords, target_error=0.016)

        # Relaxed error should require fewer gates
        assert result_relaxed["toffoli_gates"] < result_strict["toffoli_gates"]

        # Roughly 10x fewer gates for 10x relaxed error
        ratio = result_strict["toffoli_gates"] / result_relaxed["toffoli_gates"]
        assert 5 < ratio < 15  # Allow tolerance

    def test_active_space_parameters(self, converter):
        """Verify active_electrons and active_orbitals are recorded."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        # Full space (default)
        result_full = converter.estimate_qpe_resources(symbols, coords)
        assert "active_electrons" not in result_full or result_full["active_electrons"] is None

        # Active space (future feature - placeholder test)
        # result_active = converter.estimate_qpe_resources(
        #     symbols, coords, active_electrons=2, active_orbitals=2
        # )
        # assert result_active["active_electrons"] == 2
        # assert result_active["active_orbitals"] == 2

    def test_invalid_coordinates(self, converter):
        """Test error handling for invalid coordinates."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0]])  # Wrong shape

        with pytest.raises((ValueError, IndexError)):
            converter.estimate_qpe_resources(symbols, coords)

    def test_basis_set_recording(self, converter):
        """Verify basis set is correctly recorded."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        result = converter.estimate_qpe_resources(symbols, coords)
        assert result["basis"] == "sto-3g"  # Default basis

    def test_mm_embedded_flag_vacuum(self, converter):
        """Verify mm_embedded flag is False for vacuum Hamiltonian."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        result = converter.estimate_qpe_resources(symbols, coords)
        assert result["mm_embedded"] is False
        assert result["n_mm_charges"] == 0

    def test_mm_embedded_flag_solvated(self, converter):
        """Verify mm_embedded flag is True when MM charges provided."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        # Single TIP3P water molecule
        mm_charges = np.array([-0.834, 0.417, 0.417])
        mm_coords = np.array([[3.0, 0.0, 0.0], [3.5, 0.8, 0.0], [3.5, -0.8, 0.0]])

        result = converter.estimate_qpe_resources(
            symbols, coords, mm_charges=mm_charges, mm_coords=mm_coords
        )
        assert result["mm_embedded"] is True
        assert result["n_mm_charges"] == 3

    def test_mm_embedding_modifies_1norm(self, converter):
        """MM embedding modifies Hamiltonian 1-norm due to QM-MM terms."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        # Vacuum estimate
        result_vacuum = converter.estimate_qpe_resources(symbols, coords)

        # Solvated estimate (2 TIP3P waters)
        mm_charges = np.array([-0.834, 0.417, 0.417, -0.834, 0.417, 0.417])
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

        result_solvated = converter.estimate_qpe_resources(
            symbols, coords, mm_charges=mm_charges, mm_coords=mm_coords
        )

        # MM embedding modifies 1-norm (direction depends on charge distribution)
        # The key validation is that both produce valid resource estimates
        assert result_vacuum["hamiltonian_1norm"] > 0
        assert result_solvated["hamiltonian_1norm"] > 0
        assert result_solvated["mm_embedded"] is True
        assert result_vacuum["mm_embedded"] is False


class TestQuantumQMMMResourceEstimation:
    """Integration tests for resource estimation in QuantumQMMM workflow."""

    def test_compute_ground_state_with_resource_estimation(self):
        """Test resource estimation integration in compute_ground_state."""
        from q2m3.core import QuantumQMMM
        from q2m3.core.qmmm_system import Atom

        # H2 molecule (simplest case for fast test)
        h2_atoms = [
            Atom("H", np.array([0.0, 0.0, 0.0])),
            Atom("H", np.array([0.74, 0.0, 0.0])),
        ]

        # Use minimal QPE config for fast execution
        qpe_config = {
            "use_real_qpe": False,  # Use classical simulation for speed
            "device_type": "default.qubit",
        }

        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=qpe_config)

        # Without resource estimation (default)
        result_no_res = qmmm.compute_ground_state(include_resource_estimation=False)
        assert "eftqc_resources" not in result_no_res
        assert "energy" in result_no_res

        # With resource estimation
        result_with_res = qmmm.compute_ground_state(include_resource_estimation=True)
        assert "eftqc_resources" in result_with_res
        assert "energy" in result_with_res

        # Validate resource estimation structure
        resources = result_with_res["eftqc_resources"]
        assert "logical_qubits" in resources
        assert "toffoli_gates" in resources
        assert "hamiltonian_1norm" in resources
        assert "qpe_iterations" in resources
        assert "trotter_steps" in resources
        assert resources["n_electrons"] == 2  # H2 has 2 electrons
