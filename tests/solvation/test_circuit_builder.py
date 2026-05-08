# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Tests for q2m3.solvation.circuit_builder module."""

import warnings

import numpy as np
import pennylane as qml
import pytest

# ============================================================================
# Unit Tests: QPECircuitBundle dataclass
# ============================================================================


class TestQPECircuitBundle:
    """Test QPECircuitBundle frozen dataclass."""

    def test_creation(self):
        """QPECircuitBundle stores all circuit compilation artifacts."""
        from q2m3.solvation.circuit_builder import QPECircuitBundle

        bundle = QPECircuitBundle(
            compiled_circuit=lambda x: x,
            base_coeffs=np.array([1.0, 0.5, -0.3]),
            ops=[qml.Identity(wires=[0, 1]), qml.PauliZ(0), qml.PauliZ(1)],
            base_time=3.14,
            op_index_map={"identity_idx": 0, "z_wire_idx": {0: 1, 1: 2}},
            energy_shift=-1.1,
            n_estimation_wires=4,
            n_system_qubits=4,
            active_orbitals=2,
            n_trotter_steps=3,
            measurement_mode="probs",
            is_fixed_circuit=False,
        )

        assert bundle.base_time == 3.14
        assert bundle.energy_shift == -1.1
        assert bundle.n_estimation_wires == 4
        assert bundle.n_system_qubits == 4
        assert bundle.active_orbitals == 2
        assert bundle.n_trotter_steps == 3
        assert bundle.measurement_mode == "probs"
        assert bundle.is_fixed_circuit is False
        assert len(bundle.base_coeffs) == 3
        assert len(bundle.ops) == 3

    def test_immutable(self):
        """QPECircuitBundle is frozen — fields cannot be reassigned."""
        from q2m3.solvation.circuit_builder import QPECircuitBundle

        bundle = QPECircuitBundle(
            compiled_circuit=lambda x: x,
            base_coeffs=np.array([1.0]),
            ops=[qml.Identity(wires=0)],
            base_time=1.0,
            op_index_map={"identity_idx": 0, "z_wire_idx": {}},
            energy_shift=0.0,
            n_estimation_wires=2,
            n_system_qubits=2,
            active_orbitals=1,
            n_trotter_steps=1,
            measurement_mode="probs",
            is_fixed_circuit=False,
        )

        with pytest.raises(AttributeError):
            bundle.base_time = 99.0


# ============================================================================
# Unit Tests: MAX_TROTTER_STEPS_RUNTIME constant
# ============================================================================


class TestMaxTrotterSteps:
    """Test OOM protection constant."""

    def test_value(self):
        """MAX_TROTTER_STEPS_RUNTIME is 20 (benchmark-validated ceiling)."""
        from q2m3.solvation.circuit_builder import MAX_TROTTER_STEPS_RUNTIME

        assert MAX_TROTTER_STEPS_RUNTIME == 20


# ============================================================================
# Unit Tests: Hamiltonian decomposition helpers (re-exported from core)
# ============================================================================


class TestDecomposeHamiltonian:
    """Test decompose_hamiltonian via circuit_builder module."""

    def test_simple_hamiltonian(self):
        """Decompose a 3-term Hamiltonian into coefficients and operators."""
        from q2m3.solvation.circuit_builder import decompose_hamiltonian

        # Build H = 0.5 * I + 0.3 * Z(0) - 0.2 * Z(1)
        H = qml.dot(
            [0.5, 0.3, -0.2],
            [qml.Identity(wires=[0, 1]), qml.PauliZ(0), qml.PauliZ(1)],
        )

        coeffs, ops = decompose_hamiltonian(H)

        assert len(coeffs) == 3
        assert len(ops) == 3
        assert np.isclose(coeffs[0], 0.5)
        assert np.isclose(coeffs[1], 0.3)
        assert np.isclose(coeffs[2], -0.2)


class TestBuildOperatorIndexMap:
    """Test build_operator_index_map via circuit_builder module."""

    def test_finds_identity_and_z(self):
        """Index map correctly locates Identity and single-Z operators."""
        from q2m3.solvation.circuit_builder import build_operator_index_map

        ops = [qml.Identity(wires=[0, 1]), qml.PauliZ(0), qml.PauliZ(1)]
        coeffs = [0.5, 0.3, -0.2]

        index_map, ext_coeffs, ext_ops = build_operator_index_map(ops, 2, coeffs)

        assert index_map["identity_idx"] == 0
        assert 0 in index_map["z_wire_idx"]
        assert 1 in index_map["z_wire_idx"]

    def test_extends_missing_z_terms(self):
        """Missing Z(wire) terms are appended with coefficient 0.0."""
        from q2m3.solvation.circuit_builder import build_operator_index_map

        # Only Z(0) present, missing Z(1) for 2-qubit system
        ops = [qml.Identity(wires=[0, 1]), qml.PauliZ(0)]
        coeffs = [0.5, 0.3]

        index_map, ext_coeffs, ext_ops = build_operator_index_map(ops, 2, coeffs)

        # Z(1) should be appended
        assert len(ext_coeffs) == 3
        assert ext_coeffs[2] == 0.0
        assert index_map["z_wire_idx"][1] == 2


# ============================================================================
# Integration Tests: build_qpe_circuit (real PySCF; qjit compilation disabled)
# ============================================================================


@pytest.mark.solvation
class TestBuildQPECircuit:
    """Integration tests for build_qpe_circuit with a real H2 system."""

    @pytest.fixture(autouse=True)
    def _disable_qjit_compile(self, monkeypatch):
        """Exercise builder/QNode behavior without paying Catalyst compile cost."""
        from q2m3.solvation import circuit_builder as cb

        def identity_qjit(*args, **_kwargs):
            if args and callable(args[0]):
                return args[0]

            def decorator(fn):
                return fn

            return decorator

        monkeypatch.setattr(cb, "qjit", identity_qjit)

    def test_h2_probs_mode(self, h2_molecule_config, h2_hf_energy):
        """build_qpe_circuit builds H2/STO-3G circuit in probs mode."""
        from q2m3.solvation.circuit_builder import QPECircuitBundle, build_qpe_circuit
        from q2m3.solvation.config import QPEConfig, SolvationConfig

        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=QPEConfig(
                n_estimation_wires=2,
                n_trotter_steps=1,
                n_shots=0,  # probs mode
            ),
            n_waters=1,
            n_mc_steps=2,
        )

        qm_coords = h2_molecule_config.coords_array
        bundle = build_qpe_circuit(config, qm_coords, h2_hf_energy)

        # Verify bundle type and fields
        assert isinstance(bundle, QPECircuitBundle)
        assert bundle.measurement_mode == "probs"
        assert bundle.n_estimation_wires == 2
        assert bundle.n_system_qubits > 0
        assert bundle.active_orbitals == 2
        assert bundle.base_time > 0
        assert len(bundle.base_coeffs) == len(bundle.ops)

        # Verify compiled circuit returns valid probability distribution
        import jax.numpy as jnp

        result = bundle.compiled_circuit(jnp.array(bundle.base_coeffs))
        probs = np.asarray(result)

        expected_bins = 2**bundle.n_estimation_wires
        assert probs.shape == (expected_bins,)
        assert np.isclose(probs.sum(), 1.0, atol=1e-6)
        assert np.all(probs >= 0)

    def test_h2_shots_mode(self, h2_molecule_config, h2_hf_energy):
        """build_qpe_circuit builds H2/STO-3G circuit in shots mode."""
        from q2m3.solvation.circuit_builder import build_qpe_circuit
        from q2m3.solvation.config import QPEConfig, SolvationConfig

        n_shots = 8
        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=QPEConfig(
                n_estimation_wires=2,
                n_trotter_steps=1,
                n_shots=n_shots,
            ),
            n_waters=1,
            n_mc_steps=2,
        )

        qm_coords = h2_molecule_config.coords_array
        bundle = build_qpe_circuit(config, qm_coords, h2_hf_energy)

        assert bundle.measurement_mode == "shots"

        # Verify compiled circuit returns shot samples
        import jax.numpy as jnp

        result = bundle.compiled_circuit(jnp.array(bundle.base_coeffs))
        samples = np.asarray(result)

        assert samples.shape == (n_shots, bundle.n_estimation_wires)
        assert set(np.unique(samples)).issubset({0, 1})

    def test_h2_shots_mode_passes_device_seed(self, h2_molecule_config, h2_hf_energy, monkeypatch):
        """Shots-mode circuit building forwards the configured device seed."""
        from q2m3.solvation import circuit_builder as cb
        from q2m3.solvation.circuit_builder import build_qpe_circuit
        from q2m3.solvation.config import QPEConfig, SolvationConfig

        captured: list[int | None] = []
        original_select_device = cb._select_device

        def recording_select_device(device_type, n_wires, use_catalyst=False, seed=None):
            captured.append(seed)
            return original_select_device(
                device_type, n_wires, use_catalyst=use_catalyst, seed=seed
            )

        monkeypatch.setattr(cb, "_select_device", recording_select_device)

        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=QPEConfig(
                n_estimation_wires=2,
                n_trotter_steps=1,
                n_shots=30,
                device_seed=123,
            ),
            n_waters=1,
            n_mc_steps=2,
        )

        qm_coords = h2_molecule_config.coords_array
        bundle = build_qpe_circuit(config, qm_coords, h2_hf_energy)

        assert bundle.measurement_mode == "shots"
        assert captured
        assert captured[-1] == 123

    def test_compiled_circuit_reusable(self, h2_molecule_config, h2_hf_energy):
        """Circuit callable can be reused with different coefficient arrays."""
        from q2m3.solvation.circuit_builder import build_qpe_circuit
        from q2m3.solvation.config import QPEConfig, SolvationConfig

        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=QPEConfig(n_estimation_wires=2, n_trotter_steps=1),
            n_waters=1,
            n_mc_steps=2,
        )

        qm_coords = h2_molecule_config.coords_array
        bundle = build_qpe_circuit(config, qm_coords, h2_hf_energy)

        import jax.numpy as jnp

        # Call with base coefficients
        result1 = bundle.compiled_circuit(jnp.array(bundle.base_coeffs))

        # Call with slightly perturbed coefficients (simulating MM correction)
        perturbed = bundle.base_coeffs.copy()
        perturbed[0] += 0.01  # Small perturbation
        result2 = bundle.compiled_circuit(jnp.array(perturbed))

        # Both should be valid probability distributions
        probs1 = np.asarray(result1)
        probs2 = np.asarray(result2)
        assert np.isclose(probs1.sum(), 1.0, atol=1e-6)
        assert np.isclose(probs2.sum(), 1.0, atol=1e-6)

        # Results should differ due to different coefficients
        assert not np.allclose(probs1, probs2)

    def test_h2_fixed_mode_zero_arg(self, h2_molecule_config, h2_hf_energy):
        """Fixed mode: circuit takes zero arguments (compile-time constant coefficients)."""
        from q2m3.solvation.circuit_builder import build_qpe_circuit
        from q2m3.solvation.config import QPEConfig, SolvationConfig

        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=QPEConfig(n_estimation_wires=2, n_trotter_steps=1, n_shots=0),
            hamiltonian_mode="fixed",
            n_waters=1,
            n_mc_steps=2,
        )

        qm_coords = h2_molecule_config.coords_array
        bundle = build_qpe_circuit(config, qm_coords, h2_hf_energy)

        # Fixed mode: is_fixed_circuit should be True
        assert bundle.is_fixed_circuit is True

        # Zero-arg invocation
        result = bundle.compiled_circuit()
        probs = np.asarray(result)
        expected_bins = 2**bundle.n_estimation_wires
        assert probs.shape == (expected_bins,)
        assert np.isclose(probs.sum(), 1.0, atol=1e-6)

    def test_h2_dynamic_mode_is_not_fixed(self, h2_molecule_config, h2_hf_energy):
        """Dynamic mode: is_fixed_circuit should be False."""
        from q2m3.solvation.circuit_builder import build_qpe_circuit
        from q2m3.solvation.config import QPEConfig, SolvationConfig

        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=QPEConfig(n_estimation_wires=2, n_trotter_steps=1),
            hamiltonian_mode="dynamic",
            n_waters=1,
            n_mc_steps=2,
        )

        qm_coords = h2_molecule_config.coords_array
        bundle = build_qpe_circuit(config, qm_coords, h2_hf_energy)

        # Dynamic mode: is_fixed_circuit should be False
        assert bundle.is_fixed_circuit is False

    def test_trotter_cap_warning(self, h2_molecule_config, h2_hf_energy):
        """Trotter steps exceeding MAX_TROTTER_STEPS_RUNTIME are capped with warning."""
        from q2m3.solvation.circuit_builder import (
            MAX_TROTTER_STEPS_RUNTIME,
            build_qpe_circuit,
        )
        from q2m3.solvation.config import QPEConfig, SolvationConfig

        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=QPEConfig(
                n_estimation_wires=2,
                n_trotter_steps=MAX_TROTTER_STEPS_RUNTIME + 5,
            ),
            n_waters=1,
            n_mc_steps=2,
        )

        qm_coords = h2_molecule_config.coords_array

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bundle = build_qpe_circuit(config, qm_coords, h2_hf_energy)

            # Should have a warning about capping
            trotter_warnings = [
                x for x in w if "n_trotter_steps" in str(x.message) and "Capping" in str(x.message)
            ]
            assert len(trotter_warnings) >= 1

        # Actual steps should be capped
        assert bundle.n_trotter_steps == MAX_TROTTER_STEPS_RUNTIME
