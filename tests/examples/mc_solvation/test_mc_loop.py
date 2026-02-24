# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Unit tests for MC Loop module.

Tests QJIT compilation and MC sampling functionality.
"""

import numpy as np
import pytest

from examples.mc_solvation.config import MoleculeConfig, QPEConfig, SolvationConfig


class TestMCLoopCreation:
    """Test mc_loop factory function can be created without errors."""

    @pytest.fixture
    def minimal_config(self):
        """Minimal configuration for testing."""
        molecule = MoleculeConfig(
            name="H2",
            symbols=["H", "H"],
            coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            charge=0,
            active_electrons=2,
            active_orbitals=2,
            basis="sto-3g",
        )
        return SolvationConfig(
            molecule=molecule,
            qpe_config=QPEConfig(
                n_estimation_wires=3,
                n_trotter_steps=2,
                n_shots=10,
                qpe_interval=5,
                use_catalyst=False,  # Disable for faster tests
            ),
            qpe_mode="vacuum_correction",
            n_waters=2,
            n_mc_steps=5,
            temperature=300.0,
            random_seed=42,
        )

    def test_mc_loop_accepts_numpy_arrays(self, minimal_config):
        """
        BUG-REGRESSION: @qjit mc_loop must accept numpy arrays.

        Root cause: Type annotations with np.ndarray caused Catalyst AOT
        compilation to fail with "Argument type <class 'numpy.ndarray'>
        is not a valid JAX type."

        Expected: mc_loop creation should succeed and function should be callable.
        """
        from examples.mc_solvation.mc_loop import create_mc_loop

        # Mock quantum circuit (returns probability distribution for 3 estimation wires)
        def dummy_circuit():
            probs = np.zeros(2**3, dtype=np.float64)
            probs[0] = 1.0  # All probability on bin 0 -> phase=0
            return probs

        # Mock energy computation
        def dummy_energy_impl(qm_coords, solvent_states):
            return np.array([-1.0, 0.01], dtype=np.float64)

        # This should NOT raise TypeError about numpy.ndarray
        mc_loop = create_mc_loop(
            config=minimal_config,
            compiled_circuit=dummy_circuit,
            compute_energy_impl=dummy_energy_impl,
            base_time=1.0,
            n_estimation_wires=3,
            energy_shift=-1.0,
        )

        # Verify the function was created successfully
        assert callable(mc_loop)

    @pytest.mark.skip(reason="Requires full quantum compilation, slow")
    def test_mc_loop_execution(self, minimal_config):
        """Test full MC loop execution (slow, requires Catalyst)."""
        pass


class TestQPEDrivenMCLoopCreation:
    """Test create_qpe_driven_mc_loop factory function."""

    @pytest.fixture
    def qpe_driven_config(self):
        """Configuration for qpe_driven mode."""
        molecule = MoleculeConfig(
            name="H2",
            symbols=["H", "H"],
            coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            charge=0,
            active_electrons=2,
            active_orbitals=2,
            basis="sto-3g",
        )
        return SolvationConfig(
            molecule=molecule,
            qpe_config=QPEConfig(
                n_estimation_wires=3,
                n_trotter_steps=2,
                n_shots=0,
                qpe_interval=1,
                use_catalyst=False,
            ),
            qpe_mode="qpe_driven",
            n_waters=2,
            n_mc_steps=5,
            temperature=300.0,
            random_seed=42,
        )

    def test_factory_returns_callable(self, qpe_driven_config):
        """create_qpe_driven_mc_loop returns a callable @qjit function."""
        from examples.mc_solvation.mc_loop import create_qpe_driven_mc_loop

        n_terms = 15

        # Mock compiled circuit: accepts coeffs array, returns probs
        def dummy_circuit(coeffs):
            probs = np.zeros(2**3, dtype=np.float64)
            probs[0] = 0.8
            probs[1] = 0.2
            return probs

        # Mock fused callback: returns [coeffs(n_terms), e_mm, e_hf]
        def dummy_fused_impl(solvent_states, qm_coords_flat):
            result = np.zeros(n_terms + 2, dtype=np.float64)
            result[:n_terms] = np.random.default_rng(42).random(n_terms)
            result[n_terms] = 0.001  # e_mm_sol_sol
            result[n_terms + 1] = -1.1  # e_hf_solvated
            return result

        mc_loop = create_qpe_driven_mc_loop(
            config=qpe_driven_config,
            compiled_circuit=dummy_circuit,
            compute_fused_impl=dummy_fused_impl,
            base_time=1.0,
            n_estimation_wires=3,
            energy_shift=-1.0,
            n_terms=n_terms,
        )

        assert callable(mc_loop)

    def test_factory_signature_matches_spec(self, qpe_driven_config):
        """Factory function accepts all documented parameters."""
        import inspect

        from examples.mc_solvation.mc_loop import create_qpe_driven_mc_loop

        sig = inspect.signature(create_qpe_driven_mc_loop)
        param_names = list(sig.parameters.keys())

        # Required parameters from task spec
        assert "config" in param_names
        assert "compiled_circuit" in param_names
        assert "compute_fused_impl" in param_names
        assert "base_time" in param_names
        assert "n_estimation_wires" in param_names
        assert "energy_shift" in param_names
        assert "n_terms" in param_names
        assert "n_shots" in param_names


class TestQPEDrivenMCLoopExecution:
    """Integration tests for qpe_driven MC loop execution with @qjit."""

    @pytest.fixture
    def qpe_driven_config(self):
        """Minimal config for fast execution."""
        molecule = MoleculeConfig(
            name="H2",
            symbols=["H", "H"],
            coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            charge=0,
            active_electrons=2,
            active_orbitals=2,
            basis="sto-3g",
        )
        return SolvationConfig(
            molecule=molecule,
            qpe_config=QPEConfig(
                n_estimation_wires=3,
                n_trotter_steps=2,
                n_shots=0,
                qpe_interval=1,
                use_catalyst=False,
            ),
            qpe_mode="qpe_driven",
            n_waters=2,
            n_mc_steps=5,
            temperature=300.0,
            random_seed=42,
        )

    @pytest.fixture
    def mc_loop_and_inputs(self, qpe_driven_config):
        """Create mc_loop with mock callbacks and inputs."""
        from examples.mc_solvation.mc_loop import create_qpe_driven_mc_loop

        n_terms = 15
        n_estimation_wires = 3
        energy_shift = -1.1

        # Deterministic mock circuit: slight energy variation based on coeffs
        def mock_circuit(coeffs):
            probs = np.zeros(2**n_estimation_wires, dtype=np.float64)
            # Put most probability in bin 1, with small variation from coeffs
            probs[0] = 0.1
            probs[1] = 0.7
            probs[2] = 0.15
            probs[3] = 0.05
            return probs

        # Deterministic fused callback with slight variation
        step_counter = {"n": 0}

        def mock_fused_impl(solvent_states, qm_coords_flat):
            step_counter["n"] += 1
            result = np.zeros(n_terms + 2, dtype=np.float64)
            # Coefficients with small variation per call
            rng = np.random.default_rng(42 + step_counter["n"])
            result[:n_terms] = rng.random(n_terms) * 0.1
            result[n_terms] = 0.002  # e_mm_sol_sol
            result[n_terms + 1] = -1.05  # e_hf_solvated (reference)
            return result

        mc_loop = create_qpe_driven_mc_loop(
            config=qpe_driven_config,
            compiled_circuit=mock_circuit,
            compute_fused_impl=mock_fused_impl,
            base_time=1.0,
            n_estimation_wires=n_estimation_wires,
            energy_shift=energy_shift,
            n_terms=n_terms,
        )

        n_waters = qpe_driven_config.n_waters
        initial_solvents = np.random.default_rng(42).random((n_waters, 6))
        qm_coords_flat = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.74])

        return mc_loop, initial_solvents, qm_coords_flat

    def test_return_dict_has_all_required_keys(self, mc_loop_and_inputs):
        """Return dict contains all keys specified in task 2.3."""
        mc_loop, solvents, qm_coords = mc_loop_and_inputs
        result = mc_loop(solvents, qm_coords, 42)

        required_keys = [
            "initial_energy",
            "final_energy",
            "best_energy",
            "best_solvent_states",
            "final_solvent_states",
            "acceptance_rate",
            "avg_energy",
            "n_accepted",
            "quantum_energies",
            "hf_energies",
            "n_quantum_evaluations",
            "hf_times",
            "quantum_times",
            "best_qpe_energy",
            "best_qpe_solvent_states",
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_quantum_energies_shape_and_nonzero(self, mc_loop_and_inputs):
        """quantum_energies has shape (n_mc_steps,) and all elements are non-zero."""
        mc_loop, solvents, qm_coords = mc_loop_and_inputs
        result = mc_loop(solvents, qm_coords, 42)

        qe = np.asarray(result["quantum_energies"])
        assert qe.shape == (5,), f"Expected (5,), got {qe.shape}"
        # All elements should be non-zero (QPE runs every step)
        assert np.all(qe != 0.0), f"Some quantum_energies are zero: {qe}"

    def test_hf_energies_recorded_every_step(self, mc_loop_and_inputs):
        """hf_energies records HF reference energy at each step."""
        mc_loop, solvents, qm_coords = mc_loop_and_inputs
        result = mc_loop(solvents, qm_coords, 42)

        hf_e = np.asarray(result["hf_energies"])
        assert hf_e.shape == (5,), f"Expected (5,), got {hf_e.shape}"
        # All should be the mock value (-1.05) since our mock returns constant
        assert np.all(hf_e != 0.0), f"Some hf_energies are zero: {hf_e}"

    def test_n_quantum_evaluations_equals_n_mc_steps(self, mc_loop_and_inputs):
        """n_quantum_evaluations should equal n_mc_steps for qpe_driven."""
        mc_loop, solvents, qm_coords = mc_loop_and_inputs
        result = mc_loop(solvents, qm_coords, 42)

        assert int(result["n_quantum_evaluations"]) == 5

    def test_acceptance_rate_in_valid_range(self, mc_loop_and_inputs):
        """Acceptance rate should be between 0 and 1."""
        mc_loop, solvents, qm_coords = mc_loop_and_inputs
        result = mc_loop(solvents, qm_coords, 42)

        rate = float(result["acceptance_rate"])
        assert 0.0 <= rate <= 1.0, f"Invalid acceptance rate: {rate}"
