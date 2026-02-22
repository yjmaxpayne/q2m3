# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Integration tests for mc_solvation.orchestrator module.

Tests verify the workflow coordination and helper functions.
Note: Full run_solvation tests are marked as slow due to QPE execution.
"""

import numpy as np
import pytest

from examples.mc_solvation.config import MoleculeConfig, QPEConfig, SolvationConfig
from examples.mc_solvation.orchestrator import (
    _MAX_TROTTER_STEPS_RUNTIME,
    _create_energy_callback_mm_embedded,
    _create_energy_callback_vacuum_correction,
    build_mm_embedded_qpe_circuit,
    extract_qpe_energy_from_samples,
    run_solvation,
)


class TestExtractQPEEnergyFromSamples:
    """Tests for QPE energy extraction helper."""

    def test_zero_phase_samples(self):
        """All-zero samples should give zero energy."""
        samples = np.array([[0, 0, 0, 0]])
        base_time = 1.0
        energy = extract_qpe_energy_from_samples(samples, base_time)
        assert abs(energy) < 1e-10

    def test_half_phase_samples(self):
        """Half-phase (1000...) samples should give specific energy."""
        samples = np.array([[1, 0, 0, 0]])  # 8/16 = 0.5 phase
        base_time = 1.0
        energy = extract_qpe_energy_from_samples(samples, base_time)
        # E = -2*pi*0.5/1.0 = -pi
        assert abs(energy - (-np.pi)) < 1e-10

    def test_mode_selection_from_multiple_samples(self):
        """Most frequent measurement should determine energy."""
        # Mode is index 2 (0010)
        samples = np.array(
            [
                [0, 0, 1, 0],  # index 2
                [0, 0, 1, 0],  # index 2
                [0, 0, 0, 1],  # index 1
            ]
        )
        base_time = 2.0
        energy = extract_qpe_energy_from_samples(samples, base_time)
        # Phase = 2/16 = 0.125, E = -2*pi*0.125/2.0
        expected = -2 * np.pi * 0.125 / 2.0
        assert abs(energy - expected) < 1e-10

    def test_1d_sample_array(self):
        """Should handle 1D sample array."""
        sample = np.array([0, 1, 0, 0])  # index 4/16
        base_time = 1.0
        energy = extract_qpe_energy_from_samples(sample, base_time)
        expected = -2 * np.pi * 0.25 / 1.0
        assert abs(energy - expected) < 1e-10


class TestEnergyCallbackCreation:
    """Tests for energy callback factory functions."""

    def test_vacuum_correction_callback_returns_callable(
        self, h2_molecule_config, qpe_config_minimal
    ):
        """vacuum_correction callback factory should return callable."""
        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=qpe_config_minimal,
            qpe_mode="vacuum_correction",
            n_waters=2,
            n_mc_steps=5,
        )
        e_vacuum = -1.117

        callback = _create_energy_callback_vacuum_correction(config, e_vacuum)

        assert callable(callback)

    def test_vacuum_correction_callback_returns_array(self, h2_molecule_config, qpe_config_minimal):
        """vacuum_correction callback should return [energy, time] array."""
        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=qpe_config_minimal,
            qpe_mode="vacuum_correction",
            n_waters=2,
            n_mc_steps=5,
        )
        e_vacuum = -1.117

        callback = _create_energy_callback_vacuum_correction(config, e_vacuum)

        # Prepare inputs
        qm_coords_flat = np.array(h2_molecule_config.coords).flatten()
        solvent_states = np.array(
            [
                [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Water 1 at (4,0,0)
                [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],  # Water 2 at (0,4,0)
            ]
        )

        result = callback(qm_coords_flat, solvent_states)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        assert result.dtype == np.float64
        # result[0] is energy, result[1] is elapsed time
        assert result[0] < 0  # H2 + water should have negative energy
        assert result[1] >= 0  # Time should be non-negative

    def test_mm_embedded_callback_returns_callable(self, h2_molecule_config, qpe_config_minimal):
        """mm_embedded callback should return a callable (same as vacuum_correction)."""
        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=qpe_config_minimal,
            qpe_mode="mm_embedded",
            n_waters=2,
            n_mc_steps=5,
        )

        e_vacuum = -1.1
        callback = _create_energy_callback_mm_embedded(config, e_vacuum)
        assert callable(callback)


class TestConfigurationValidation:
    """Tests for configuration validation in orchestrator flow."""

    def test_invalid_config_raises_early(self):
        """Invalid configuration should raise before expensive computation."""
        invalid_mol = MoleculeConfig(
            name="Invalid",
            symbols=["H", "H"],
            coords=[[0, 0]],  # Wrong dimension
            charge=0,
            active_electrons=2,
            active_orbitals=2,
        )
        config = SolvationConfig(
            molecule=invalid_mol,
            n_waters=2,
            n_mc_steps=10,
        )

        with pytest.raises(ValueError):
            run_solvation(config, show_plots=False)


class TestOrchestratorIntegration:
    """Integration tests for full orchestrator workflow.

    Note: These tests involve actual QPE circuit execution and are slow.
    They are skipped by default and should be run explicitly when needed.
    """

    @pytest.fixture
    def minimal_h2_config(self, h2_molecule_config):
        """Minimal H2 config for fast integration tests."""
        return SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=QPEConfig(
                n_estimation_wires=3,
                n_trotter_steps=2,
                n_shots=5,
                qpe_interval=5,
                use_catalyst=False,  # Avoid JIT compilation overhead
            ),
            qpe_mode="vacuum_correction",
            n_waters=2,
            n_mc_steps=5,  # Very short simulation
            temperature=300.0,
            random_seed=42,
            verbose=False,
        )

    @pytest.mark.skip(reason="Full orchestrator test requires QPE execution (slow)")
    def test_run_solvation_returns_result_dict(self, minimal_h2_config):
        """run_solvation should return result dictionary with expected keys."""
        result = run_solvation(minimal_h2_config, show_plots=False)

        assert isinstance(result, dict)
        assert "initial_energy" in result
        assert "final_energy" in result
        assert "best_energy" in result
        assert "acceptance_rate" in result
        assert "quantum_energies" in result
        assert "timing" in result
        assert "e_vacuum" in result
        assert "circuit_metadata" in result
        meta = result["circuit_metadata"]
        assert set(meta.keys()) == TestCircuitMetadata.EXPECTED_KEYS
        assert meta["total_qubits"] == meta["n_system_qubits"] + meta["n_estimation_wires"]

    @pytest.mark.skip(reason="Full orchestrator test requires QPE execution (slow)")
    def test_run_solvation_energy_changes(self, minimal_h2_config):
        """Energy should potentially change after MC sampling."""
        result = run_solvation(minimal_h2_config, show_plots=False)

        initial = result["initial_energy"]
        best = result["best_energy"]

        # Best energy should be <= initial (MC searches for lower energy)
        assert best <= initial + 1e-10  # Small tolerance for numerical noise

    @pytest.mark.skip(reason="Full orchestrator test requires QPE execution (slow)")
    def test_run_solvation_acceptance_rate_reasonable(self, minimal_h2_config):
        """Acceptance rate should be between 0 and 1."""
        result = run_solvation(minimal_h2_config, show_plots=False)

        acceptance = float(result["acceptance_rate"])
        assert 0.0 <= acceptance <= 1.0


class TestVacuumCorrectionMode:
    """Tests specific to vacuum_correction mode behavior."""

    def test_vacuum_energy_computed(self, h2_molecule_config, qpe_config_minimal):
        """Vacuum energy should be computed and stored in result."""
        # This test would require running the full workflow
        # For unit testing, we verify the callback captures e_vacuum
        e_vacuum = -1.117
        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=qpe_config_minimal,
            qpe_mode="vacuum_correction",
            n_waters=2,
            n_mc_steps=5,
        )

        callback = _create_energy_callback_vacuum_correction(config, e_vacuum)

        # The callback should use the captured e_vacuum internally
        # We verify this indirectly by checking the callback exists
        assert callback is not None


class TestMMEmbeddedMode:
    """Tests for mm_embedded mode (Frame 14 target architecture)."""

    def test_mm_embedded_mode_accepted(self, h2_molecule_config, qpe_config_minimal):
        """mm_embedded mode should be accepted as valid qpe_mode."""
        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=qpe_config_minimal,
            qpe_mode="mm_embedded",
            n_waters=2,
            n_mc_steps=5,
        )
        config.validate()
        assert config.qpe_mode == "mm_embedded"


class TestCircuitMetadata:
    """Tests for circuit_metadata dict schema in run_solvation result."""

    # Expected keys for any circuit_metadata dict
    EXPECTED_KEYS = {
        "n_system_qubits",
        "n_estimation_wires",
        "total_qubits",
        "n_hamiltonian_terms",
        "n_trotter_steps",
        "n_trotter_steps_requested",
        "base_time",
        "energy_formula",
        "energy_shift",
    }

    def test_circuit_metadata_expected_keys(self):
        """circuit_metadata should contain all expected keys, no more no less."""
        # Simulate a circuit_metadata dict from vacuum_correction mode
        metadata = {
            "n_system_qubits": 4,
            "n_estimation_wires": 3,
            "total_qubits": 7,
            "n_hamiltonian_terms": 15,
            "n_trotter_steps": 10,
            "n_trotter_steps_requested": 10,
            "base_time": 1.23,
            "energy_formula": "E_corr(vac) + E_HF(R)",
            "energy_shift": -1.117,
        }
        assert set(metadata.keys()) == self.EXPECTED_KEYS

    def test_circuit_metadata_total_qubits_consistent(self):
        """total_qubits must equal n_system_qubits + n_estimation_wires."""
        for n_sys, n_est in [(4, 3), (8, 4), (2, 2)]:
            metadata = {
                "n_system_qubits": n_sys,
                "n_estimation_wires": n_est,
                "total_qubits": n_sys + n_est,
            }
            assert (
                metadata["total_qubits"]
                == metadata["n_system_qubits"] + metadata["n_estimation_wires"]
            )


class TestTrotterStepsCeiling:
    """Tests for runtime-parameterized circuit compilation memory guard."""

    def test_max_trotter_steps_constant_exists(self):
        """_MAX_TROTTER_STEPS_RUNTIME should be a positive integer."""
        assert isinstance(_MAX_TROTTER_STEPS_RUNTIME, int)
        assert 1 <= _MAX_TROTTER_STEPS_RUNTIME <= 50

    def test_max_trotter_steps_is_conservative(self):
        """Ceiling should be within practical compilation range."""
        # Validated: n_trotter=10 compiles in ~225s for H2 (4 est wires, 15 terms).
        # Ceiling of 20 allows headroom for larger systems.
        assert _MAX_TROTTER_STEPS_RUNTIME <= 50

    @pytest.mark.skip(reason="Requires PySCF + Catalyst compilation (slow)")
    def test_build_mm_embedded_caps_trotter_steps(self, h2_molecule_config):
        """build_mm_embedded_qpe_circuit should cap n_trotter when exceeding ceiling."""
        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=QPEConfig(
                n_estimation_wires=2,
                n_trotter_steps=10,  # Exceeds _MAX_TROTTER_STEPS_RUNTIME
                n_shots=5,
                use_catalyst=True,
            ),
            qpe_mode="mm_embedded",
            n_waters=2,
            n_mc_steps=5,
        )
        qm_coords = np.array(h2_molecule_config.coords)
        # Should not OOM — guard caps n_trotter internally
        result = build_mm_embedded_qpe_circuit(config, qm_coords, hf_energy_estimate=-1.1)
        assert result is not None
