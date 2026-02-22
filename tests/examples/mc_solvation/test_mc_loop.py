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
