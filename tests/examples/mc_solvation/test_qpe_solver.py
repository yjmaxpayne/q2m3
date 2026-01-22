# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Unit tests for mc_solvation.qpe_solver module.

Tests verify QPE solver configuration and energy extraction logic.
Note: Full QPE circuit tests are in tests/test_qpe_circuit.py.
"""

import numpy as np
import pennylane as qml
import pytest

from examples.mc_solvation.qpe_solver import (
    QPESolver,
    QPESolverConfig,
    extract_energy_from_samples_impl,
)
from examples.mc_solvation.quantum_solver import SolverResult


class TestQPESolverConfig:
    """Tests for QPESolverConfig dataclass."""

    def test_default_values(self):
        """Should use sensible default values."""
        config = QPESolverConfig()
        assert config.n_estimation_wires == 4
        assert config.n_trotter_steps == 10
        assert config.n_shots == 50
        assert config.use_catalyst is True
        assert config.target_resolution == 0.003
        assert config.energy_range == 0.2

    def test_custom_values(self):
        """Should accept custom values."""
        config = QPESolverConfig(
            n_estimation_wires=6,
            n_trotter_steps=20,
            n_shots=100,
            use_catalyst=False,
            target_resolution=0.001,
            energy_range=0.5,
        )
        assert config.n_estimation_wires == 6
        assert config.n_trotter_steps == 20
        assert config.n_shots == 100
        assert config.use_catalyst is False
        assert config.target_resolution == 0.001
        assert config.energy_range == 0.5


class TestQPESolver:
    """Tests for QPESolver class."""

    def test_solver_name(self):
        """Solver name should be 'QPE'."""
        config = QPESolverConfig()
        solver = QPESolver(config)
        assert solver.name == "QPE"

    def test_initial_state(self):
        """Solver should start without compiled circuit."""
        config = QPESolverConfig()
        solver = QPESolver(config)
        assert solver._engine is None
        assert solver._compiled_circuit is None
        assert solver.get_compiled_circuit() is None

    def test_config_stored(self):
        """Config should be stored in solver."""
        config = QPESolverConfig(n_shots=99)
        solver = QPESolver(config)
        assert solver.config.n_shots == 99


class TestExtractEnergyFromSamples:
    """Tests for energy extraction from QPE samples."""

    def test_single_sample_zero_phase(self):
        """All-zero sample should give zero phase."""
        samples = np.array([[0, 0, 0, 0]])  # 4-bit, all zeros
        base_time = 1.0
        e_ref = 0.0

        energy = extract_energy_from_samples_impl(samples, base_time, e_ref)

        # Phase index 0 -> phase 0 -> delta_e = -2*pi*0/t = 0
        assert abs(energy - e_ref) < 1e-10

    def test_single_sample_with_reference(self):
        """Reference energy should be added to result."""
        samples = np.array([[0, 0, 0, 0]])
        base_time = 1.0
        e_ref = -1.0

        energy = extract_energy_from_samples_impl(samples, base_time, e_ref)

        assert abs(energy - (-1.0)) < 1e-10

    def test_mode_selection(self):
        """Most frequent measurement should be selected."""
        # 4-bit samples where 0001 appears most frequently
        samples = np.array(
            [
                [0, 0, 0, 1],  # index 1
                [0, 0, 0, 1],  # index 1
                [0, 0, 0, 1],  # index 1
                [0, 0, 1, 0],  # index 2
            ]
        )
        base_time = 1.0
        e_ref = 0.0

        energy = extract_energy_from_samples_impl(samples, base_time, e_ref)

        # Mode is index 1 -> phase = 1/16 -> delta_e = -2*pi*(1/16)/1.0
        expected_delta = -2 * np.pi * (1 / 16) / 1.0
        assert abs(energy - expected_delta) < 1e-10

    def test_1d_sample_handling(self):
        """Should handle 1D sample array."""
        # Single shot as 1D array
        sample = np.array([1, 0, 0, 0])  # index 8 for 4-bit
        base_time = 2.0
        e_ref = -75.0

        energy = extract_energy_from_samples_impl(sample, base_time, e_ref)

        # Phase = 8/16 = 0.5 -> delta_e = -2*pi*0.5/2 = -pi/2
        expected_delta = -2 * np.pi * 0.5 / 2.0
        expected_energy = expected_delta + e_ref
        assert abs(energy - expected_energy) < 1e-10

    def test_different_bit_lengths(self):
        """Should work with different numbers of estimation bits."""
        # 3-bit samples
        samples_3bit = np.array([[0, 0, 1]])  # index 1 of 8
        energy_3bit = extract_energy_from_samples_impl(samples_3bit, 1.0, 0.0)
        expected_3bit = -2 * np.pi * (1 / 8) / 1.0

        # 5-bit samples
        samples_5bit = np.array([[0, 0, 0, 0, 1]])  # index 1 of 32
        energy_5bit = extract_energy_from_samples_impl(samples_5bit, 1.0, 0.0)
        expected_5bit = -2 * np.pi * (1 / 32) / 1.0

        assert abs(energy_3bit - expected_3bit) < 1e-10
        assert abs(energy_5bit - expected_5bit) < 1e-10

    def test_base_time_scaling(self):
        """Energy should scale inversely with base_time."""
        samples = np.array([[0, 1, 0, 0]])  # index 4 of 16
        e_ref = 0.0

        energy_t1 = extract_energy_from_samples_impl(samples, 1.0, e_ref)
        energy_t2 = extract_energy_from_samples_impl(samples, 2.0, e_ref)

        # delta_e ∝ 1/t, so energy_t1 = 2 * energy_t2
        assert abs(energy_t1 - 2 * energy_t2) < 1e-10


class TestQPESolverIntegration:
    """Integration tests for QPESolver with simple Hamiltonians.

    Note: Full QPE circuit tests require specific Hamiltonian structures
    compatible with the QPEEngine. These tests are skipped as they require
    molecular Hamiltonians from pyscf_to_pennylane_hamiltonian.
    """

    @pytest.fixture
    def simple_pauli_z(self):
        """Simple Pauli Z Hamiltonian for testing."""
        # H = Z with eigenvalues +1 (|0⟩) and -1 (|1⟩)
        return qml.Hamiltonian([1.0], [qml.PauliZ(0)])

    @pytest.mark.skip(reason="QPE circuit requires molecular Hamiltonian structure")
    def test_solve_pauli_z_ground_state(self, simple_pauli_z):
        """Should estimate ground state of simple Z Hamiltonian."""
        config = QPESolverConfig(
            n_estimation_wires=3,
            n_trotter_steps=1,
            n_shots=10,
            use_catalyst=False,
        )
        solver = QPESolver(config)

        # HF state |1⟩ (ground state of Z)
        hf_state = np.array([1])

        result = solver.solve(
            hamiltonian=simple_pauli_z,
            hf_state=hf_state,
            n_qubits=1,
            e_ref=0.0,
        )

        assert isinstance(result, SolverResult)
        assert result.method == "QPE"
        assert result.converged is True
        # Energy should be close to -1 (ground state of Z)
        # Note: QPE resolution limited, so we use loose tolerance
        assert -2.0 < result.energy < 0.0

    @pytest.mark.skip(reason="QPE circuit requires molecular Hamiltonian structure")
    def test_precompile_then_run(self, simple_pauli_z):
        """Should support precompile and run_precompiled workflow."""
        config = QPESolverConfig(
            n_estimation_wires=3,
            n_trotter_steps=1,
            n_shots=5,
            use_catalyst=False,
        )
        solver = QPESolver(config)

        hf_state = np.array([1])

        # Precompile
        solver.precompile(simple_pauli_z, hf_state, n_qubits=1)

        # Compiled circuit should exist
        assert solver.get_compiled_circuit() is not None

        # Run precompiled
        result = solver.run_precompiled(e_ref=0.0)

        assert result.converged is True
        assert result.metadata.get("precompiled") is True

    def test_run_precompiled_without_precompile_raises(self):
        """Should raise if run_precompiled called without precompile."""
        config = QPESolverConfig()
        solver = QPESolver(config)

        with pytest.raises(RuntimeError, match="not pre-compiled"):
            solver.run_precompiled()
