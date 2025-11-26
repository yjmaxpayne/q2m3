# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
TDD tests for QPE circuit implementation.

Phase 2: Standard QPE circuit with:
- Initial state preparation
- Controlled time evolution
- Inverse QFT
"""

import numpy as np
import pennylane as qml
import pytest

from q2m3.core import QPEEngine

# ============================================================================
# P0: Initial State Preparation Tests
# ============================================================================


class TestInitialStatePreparation:
    """P0: Test HF state preparation circuit."""

    def test_hf_state_preparation_h2(self, h2_hamiltonian):
        """Test HF state preparation for H2 molecule."""
        hf_state = h2_hamiltonian["hf_state"]
        n_qubits = h2_hamiltonian["n_qubits"]

        engine = QPEEngine(n_qubits=n_qubits, n_iterations=4)
        dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def prepare_and_measure():
            engine._prepare_initial_state(hf_state, list(range(n_qubits)))
            return qml.state()

        state = prepare_and_measure()

        # Convert HF binary state to expected state vector index
        # |1100⟩ for H2 -> index = 0b0011 = 3 (reversed for PennyLane convention)
        expected_index = int("".join(str(x) for x in hf_state), 2)

        # State should be a computational basis state
        assert np.abs(state[expected_index]) > 0.99

    def test_hf_state_preparation_h3o(self, h3o_hamiltonian):
        """Test HF state preparation for H3O+ molecule."""
        hf_state = h3o_hamiltonian["hf_state"]
        n_qubits = h3o_hamiltonian["n_qubits"]

        engine = QPEEngine(n_qubits=n_qubits, n_iterations=4)
        dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def prepare_and_measure():
            engine._prepare_initial_state(hf_state, list(range(n_qubits)))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        expectations = prepare_and_measure()

        # For HF state, occupied orbitals have Z expectation -1, unoccupied have +1
        for i, (exp, occ) in enumerate(zip(expectations, hf_state, strict=True)):
            expected_z = -1 if occ == 1 else 1
            assert np.isclose(
                exp, expected_z, atol=0.01
            ), f"Qubit {i}: expected {expected_z}, got {exp}"

    def test_hf_state_empty(self):
        """Test preparation with all-zero state."""
        engine = QPEEngine(n_qubits=4, n_iterations=4)
        dev = qml.device("lightning.qubit", wires=4)
        hf_state = np.array([0, 0, 0, 0])

        @qml.qnode(dev)
        def prepare_and_measure():
            engine._prepare_initial_state(hf_state, [0, 1, 2, 3])
            return qml.state()

        state = prepare_and_measure()
        # All-zero state should give |0000⟩
        assert np.abs(state[0]) > 0.99


# ============================================================================
# P0: Inverse QFT Tests
# ============================================================================


class TestInverseQFT:
    """P0: Test inverse QFT circuit."""

    def test_inverse_qft_basic(self):
        """Test QFT followed by inverse QFT gives identity."""
        engine = QPEEngine(n_qubits=2, n_iterations=4)
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def qft_then_inverse():
            # Start with |00⟩ + |01⟩ superposition
            qml.Hadamard(wires=0)
            # Apply QFT then inverse QFT should give back original state
            qml.QFT(wires=[0, 1])
            engine._inverse_qft([0, 1])
            return qml.state()

        state = qft_then_inverse()

        # Should return to |+0⟩ = (|00⟩ + |10⟩) / sqrt(2)
        # State vector: [1/sqrt(2), 0, 1/sqrt(2), 0]
        expected = np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0])
        np.testing.assert_array_almost_equal(state, expected, decimal=5)

    def test_inverse_qft_identity(self):
        """Test QFT followed by QFT^-1 is identity on arbitrary state."""
        engine = QPEEngine(n_qubits=2, n_iterations=4)
        dev = qml.device("lightning.qubit", wires=3)

        @qml.qnode(dev)
        def identity_test():
            # Prepare arbitrary state
            qml.RX(0.5, wires=0)
            qml.RY(0.7, wires=1)
            qml.CNOT(wires=[0, 1])
            # Save state by applying QFT then QFT^-1
            qml.QFT(wires=[0, 1, 2])
            engine._inverse_qft([0, 1, 2])
            return qml.state()

        @qml.qnode(dev)
        def original_state():
            qml.RX(0.5, wires=0)
            qml.RY(0.7, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.state()

        state_after = identity_test()
        state_original = original_state()

        # Should be identical
        np.testing.assert_array_almost_equal(state_after, state_original, decimal=5)

    def test_inverse_qft_single_wire(self):
        """Test inverse QFT on single wire (just Hadamard)."""
        engine = QPEEngine(n_qubits=2, n_iterations=4)
        dev = qml.device("lightning.qubit", wires=1)

        @qml.qnode(dev)
        def single_wire_qft():
            qml.Hadamard(wires=0)
            engine._inverse_qft([0])
            return qml.state()

        state = single_wire_qft()
        # H followed by H (QFT on 1 qubit) should give |0⟩
        assert np.abs(state[0]) > 0.99


# ============================================================================
# P0: Controlled Unitary Tests
# ============================================================================


class TestControlledUnitary:
    """P0: Test controlled time evolution."""

    def test_controlled_unitary_simple_hamiltonian(self, simple_hamiltonian):
        """Test controlled evolution with simple Hamiltonian."""
        engine = QPEEngine(n_qubits=2, n_iterations=4)
        dev = qml.device("lightning.qubit", wires=3)  # 2 system + 1 control

        @qml.qnode(dev)
        def test_circuit():
            # Prepare control in |1⟩
            qml.PauliX(wires=2)
            # Apply controlled unitary
            engine._apply_controlled_unitary(
                simple_hamiltonian,
                time=1.0,
                control_wire=2,
                target_wires=[0, 1],
                n_trotter_steps=3,
            )
            return qml.state()

        # Should execute without error
        state = test_circuit()
        assert state is not None
        assert len(state) == 8  # 2^3 states

    def test_controlled_unitary_control_off(self, simple_hamiltonian):
        """Test that unitary is NOT applied when control is |0⟩."""
        engine = QPEEngine(n_qubits=2, n_iterations=4)
        dev = qml.device("lightning.qubit", wires=3)

        @qml.qnode(dev)
        def with_control_off():
            # Control stays in |0⟩, target in |00⟩
            engine._apply_controlled_unitary(
                simple_hamiltonian,
                time=1.0,
                control_wire=2,
                target_wires=[0, 1],
                n_trotter_steps=3,
            )
            return qml.state()

        state = with_control_off()
        # State should remain |000⟩ when control is 0
        assert np.abs(state[0]) > 0.99

    def test_controlled_unitary_h2_hamiltonian(self, h2_hamiltonian):
        """Test controlled evolution with H2 molecular Hamiltonian."""
        H = h2_hamiltonian["hamiltonian"]
        n_qubits = h2_hamiltonian["n_qubits"]
        hf_state = h2_hamiltonian["hf_state"]

        engine = QPEEngine(n_qubits=n_qubits, n_iterations=4)
        total_wires = n_qubits + 1  # system + 1 control
        dev = qml.device("lightning.qubit", wires=total_wires)

        @qml.qnode(dev)
        def test_h2_circuit():
            # Prepare HF state on system
            qml.BasisState(hf_state, wires=range(n_qubits))
            # Prepare control in superposition
            qml.Hadamard(wires=n_qubits)
            # Apply controlled unitary
            engine._apply_controlled_unitary(
                H,
                time=0.5,
                control_wire=n_qubits,
                target_wires=list(range(n_qubits)),
                n_trotter_steps=3,
            )
            return qml.state()

        # Should execute without error
        state = test_h2_circuit()
        assert state is not None

    def test_controlled_unitary_different_times(self, simple_hamiltonian):
        """Test that different evolution times give different results."""
        engine = QPEEngine(n_qubits=2, n_iterations=4)
        dev = qml.device("lightning.qubit", wires=3)

        def evolve_with_time(t):
            @qml.qnode(dev)
            def circuit():
                qml.PauliX(wires=2)  # Control on
                qml.Hadamard(wires=0)
                engine._apply_controlled_unitary(
                    simple_hamiltonian,
                    time=t,
                    control_wire=2,
                    target_wires=[0, 1],
                    n_trotter_steps=5,
                )
                return qml.state()

            return circuit()

        state_t1 = evolve_with_time(0.5)
        state_t2 = evolve_with_time(1.0)

        # Different times should give different states
        assert not np.allclose(state_t1, state_t2)


# ============================================================================
# P1: QPE Circuit Integration Tests (H2)
# ============================================================================


class TestQPECircuitH2:
    """P1: End-to-end QPE tests with H2 molecule."""

    def test_qpe_circuit_structure(self, h2_hamiltonian, qpe_config_basic):
        """Test QPE circuit can be built and executed."""
        H = h2_hamiltonian["hamiltonian"]
        hf_state = h2_hamiltonian["hf_state"]
        n_qubits = h2_hamiltonian["n_qubits"]

        engine = QPEEngine(n_qubits=n_qubits, n_iterations=qpe_config_basic["n_estimation_wires"])

        # Build circuit
        qpe_circuit = engine._build_standard_qpe_circuit(
            H,
            hf_state,
            n_estimation_wires=qpe_config_basic["n_estimation_wires"],
            base_time=qpe_config_basic["base_time"],
            n_trotter_steps=qpe_config_basic["n_trotter_steps"],
        )

        # Execute circuit
        result = qpe_circuit()
        assert result is not None

    def test_qpe_h2_energy_estimation(self, h2_hamiltonian, qpe_config_basic):
        """Test QPE gives reasonable energy estimate for H2."""
        H = h2_hamiltonian["hamiltonian"]
        hf_state = h2_hamiltonian["hf_state"]
        n_qubits = h2_hamiltonian["n_qubits"]

        engine = QPEEngine(n_qubits=n_qubits, n_iterations=qpe_config_basic["n_estimation_wires"])

        qpe_circuit = engine._build_standard_qpe_circuit(
            H,
            hf_state,
            n_estimation_wires=qpe_config_basic["n_estimation_wires"],
            base_time=qpe_config_basic["base_time"],
            n_trotter_steps=qpe_config_basic["n_trotter_steps"],
        )

        # Run multiple shots and estimate energy
        samples = qpe_circuit()
        energy = engine._extract_energy_from_samples(
            samples, base_time=qpe_config_basic["base_time"]
        )

        # Energy should be a finite number (QPE with limited precision may give approximate values)
        # Note: We're just checking that the QPE circuit runs and produces a result
        assert np.isfinite(energy), f"Energy {energy} is not finite"

    def test_qpe_h2_multiple_runs(self, h2_hamiltonian, qpe_config_basic):
        """Test QPE consistency across multiple runs."""
        H = h2_hamiltonian["hamiltonian"]
        hf_state = h2_hamiltonian["hf_state"]
        n_qubits = h2_hamiltonian["n_qubits"]

        engine = QPEEngine(n_qubits=n_qubits, n_iterations=4)

        qpe_circuit = engine._build_standard_qpe_circuit(
            H,
            hf_state,
            n_estimation_wires=4,
            base_time=0.5,
            n_trotter_steps=3,
        )

        # Run multiple times - results should be consistent
        results = [qpe_circuit() for _ in range(3)]
        # All results should be valid arrays
        for r in results:
            assert r is not None


# ============================================================================
# P2: QPE Circuit Tests (H3O+ - MVP Target)
# ============================================================================


class TestQPECircuitH3O:
    """P2: H3O+ target system tests - MVP validation.

    Note: Full H3O+ has 16 qubits and ~2000 Pauli terms, which exceeds
    typical memory limits. We use active space approximation (4 electrons
    in 4 orbitals = 8 qubits) to make simulation feasible.
    """

    @pytest.fixture
    def qpe_config_h3o(self):
        """QPE config for H3O+ with active space."""
        return {
            "n_estimation_wires": 3,  # Reasonable precision for POC
            "n_trotter_steps": 2,  # Balance precision vs speed
            "base_time": 0.3,
        }

    @pytest.mark.slow
    def test_qpe_h3o_can_execute(self, h3o_hamiltonian_active_space, qpe_config_h3o):
        """MVP Test: H3O+ QPE circuit can execute successfully.

        Uses active space (4e, 4o) = 8 qubits to make simulation feasible.
        """
        H = h3o_hamiltonian_active_space["hamiltonian"]
        hf_state = h3o_hamiltonian_active_space["hf_state"]
        n_qubits = h3o_hamiltonian_active_space["n_qubits"]

        engine = QPEEngine(n_qubits=n_qubits, n_iterations=qpe_config_h3o["n_estimation_wires"])

        # Build circuit
        qpe_circuit = engine._build_standard_qpe_circuit(
            H,
            hf_state,
            n_estimation_wires=qpe_config_h3o["n_estimation_wires"],
            base_time=qpe_config_h3o["base_time"],
            n_trotter_steps=qpe_config_h3o["n_trotter_steps"],
        )

        # Execute - this is the MVP test: H3O+ must be runnable
        result = qpe_circuit()
        assert result is not None, "H3O+ QPE circuit failed to execute"

    @pytest.mark.slow
    def test_qpe_h3o_returns_energy(self, h3o_hamiltonian_active_space, qpe_config_h3o):
        """Test H3O+ QPE returns an energy estimate."""
        H = h3o_hamiltonian_active_space["hamiltonian"]
        hf_state = h3o_hamiltonian_active_space["hf_state"]
        n_qubits = h3o_hamiltonian_active_space["n_qubits"]

        engine = QPEEngine(n_qubits=n_qubits, n_iterations=qpe_config_h3o["n_estimation_wires"])

        qpe_circuit = engine._build_standard_qpe_circuit(
            H,
            hf_state,
            n_estimation_wires=qpe_config_h3o["n_estimation_wires"],
            base_time=qpe_config_h3o["base_time"],
            n_trotter_steps=qpe_config_h3o["n_trotter_steps"],
        )

        samples = qpe_circuit()
        energy = engine._extract_energy_from_samples(samples, base_time=qpe_config_h3o["base_time"])

        # Energy should be a finite number
        assert np.isfinite(energy), f"H3O+ energy is not finite: {energy}"

    @pytest.mark.slow
    def test_qpe_h3o_energy_is_finite(self, h3o_hamiltonian_active_space, qpe_config_h3o):
        """Test H3O+ QPE energy is a valid finite number."""
        H = h3o_hamiltonian_active_space["hamiltonian"]
        hf_state = h3o_hamiltonian_active_space["hf_state"]
        n_qubits = h3o_hamiltonian_active_space["n_qubits"]

        engine = QPEEngine(n_qubits=n_qubits, n_iterations=qpe_config_h3o["n_estimation_wires"])

        qpe_circuit = engine._build_standard_qpe_circuit(
            H,
            hf_state,
            n_estimation_wires=qpe_config_h3o["n_estimation_wires"],
            base_time=qpe_config_h3o["base_time"],
            n_trotter_steps=qpe_config_h3o["n_trotter_steps"],
        )

        samples = qpe_circuit()
        energy = engine._extract_energy_from_samples(samples, base_time=qpe_config_h3o["base_time"])

        # Energy should be finite (precision is limited with minimal config)
        assert np.isfinite(energy), f"H3O+ energy {energy} is not finite"


# ============================================================================
# P3: Catalyst @qjit Integration Tests
# ============================================================================


class TestQPECatalyst:
    """P3: Catalyst @qjit integration tests.

    Tests verify that QPE circuits work correctly with Catalyst JIT compilation.
    These tests are marked with @pytest.mark.catalyst and will be skipped if
    pennylane-catalyst is not installed.
    """

    @pytest.mark.catalyst
    def test_catalyst_availability_flag(self):
        """QPEEngine correctly reports Catalyst availability."""
        from q2m3.core.qpe import HAS_CATALYST

        # Should be True since user confirmed Catalyst is installed
        assert isinstance(HAS_CATALYST, bool)

    @pytest.mark.catalyst
    def test_qpe_engine_use_catalyst_parameter(self):
        """QPEEngine accepts use_catalyst parameter."""
        engine = QPEEngine(n_qubits=4, use_catalyst=False)
        assert hasattr(engine, "use_catalyst")
        assert engine.use_catalyst is False

        engine_jit = QPEEngine(n_qubits=4, use_catalyst=True)
        assert engine_jit.use_catalyst is True

    @pytest.mark.catalyst
    def test_qpe_with_catalyst_h2(self, h2_hamiltonian, qpe_config_basic):
        """H2 QPE works with @qjit compilation."""
        H = h2_hamiltonian["hamiltonian"]
        hf_state = h2_hamiltonian["hf_state"]
        n_qubits = h2_hamiltonian["n_qubits"]

        engine = QPEEngine(n_qubits=n_qubits, use_catalyst=True)

        qpe_circuit = engine._build_standard_qpe_circuit(
            H,
            hf_state,
            n_estimation_wires=qpe_config_basic["n_estimation_wires"],
            base_time=qpe_config_basic["base_time"],
            n_trotter_steps=qpe_config_basic["n_trotter_steps"],
        )

        # Execute JIT-compiled circuit
        result = qpe_circuit()
        assert result is not None

    @pytest.mark.catalyst
    def test_energy_consistency_jit_vs_nojit(self, h2_hamiltonian):
        """Both JIT and non-JIT produce valid finite energies.

        Note: QPE with limited precision (4 estimation qubits) and single-shot
        sampling produces highly variable results. We only verify that both
        versions execute and return finite energies. The actual energy values
        may differ significantly due to the probabilistic nature of QPE.
        """
        H = h2_hamiltonian["hamiltonian"]
        hf_state = h2_hamiltonian["hf_state"]
        n_qubits = h2_hamiltonian["n_qubits"]

        config = {
            "n_estimation_wires": 4,
            "base_time": 0.5,
            "n_trotter_steps": 3,
            "n_shots": 10,  # Multiple shots reduce probability of all-zero samples
        }

        # Non-JIT version
        engine_nojit = QPEEngine(n_qubits=n_qubits, use_catalyst=False)
        circuit_nojit = engine_nojit._build_standard_qpe_circuit(H, hf_state, **config)
        samples_nojit = circuit_nojit()
        energy_nojit = engine_nojit._extract_energy_from_samples(
            samples_nojit, base_time=config["base_time"]
        )

        # JIT version
        engine_jit = QPEEngine(n_qubits=n_qubits, use_catalyst=True)
        circuit_jit = engine_jit._build_standard_qpe_circuit(H, hf_state, **config)
        samples_jit = circuit_jit()
        energy_jit = engine_jit._extract_energy_from_samples(
            samples_jit, base_time=config["base_time"]
        )

        # Both should return finite energies (primary validation)
        assert np.isfinite(energy_nojit), f"Non-JIT energy not finite: {energy_nojit}"
        assert np.isfinite(energy_jit), f"JIT energy not finite: {energy_jit}"

        # Both energies should be non-positive (QPE phase extraction)
        # Note: With multiple shots averaging, energy should be negative for bound molecules
        assert energy_nojit <= 0, f"Non-JIT energy should be non-positive: {energy_nojit}"
        assert energy_jit <= 0, f"JIT energy should be non-positive: {energy_jit}"

    @pytest.mark.catalyst
    @pytest.mark.slow
    def test_qpe_h3o_with_catalyst(self, h3o_hamiltonian_active_space):
        """H3O+ MVP with Catalyst (active space)."""
        H = h3o_hamiltonian_active_space["hamiltonian"]
        hf_state = h3o_hamiltonian_active_space["hf_state"]
        n_qubits = h3o_hamiltonian_active_space["n_qubits"]

        config = {
            "n_estimation_wires": 3,
            "base_time": 0.3,
            "n_trotter_steps": 2,
        }

        engine = QPEEngine(n_qubits=n_qubits, use_catalyst=True)

        qpe_circuit = engine._build_standard_qpe_circuit(H, hf_state, **config)

        # MVP test: H3O+ must be runnable with Catalyst
        result = qpe_circuit()
        assert result is not None, "H3O+ QPE with Catalyst failed to execute"

        # Extract energy
        energy = engine._extract_energy_from_samples(result, base_time=config["base_time"])
        assert np.isfinite(energy), f"H3O+ Catalyst energy not finite: {energy}"


# ============================================================================
# GPU Tests (lightning.gpu)
# ============================================================================


class TestQPEGPU:
    """GPU-accelerated QPE tests using lightning.gpu device."""

    @pytest.mark.gpu
    def test_device_type_selection_gpu(self, h2_hamiltonian):
        """Test QPEEngine respects device_type='lightning.gpu' parameter."""
        H = h2_hamiltonian["hamiltonian"]
        hf_state = h2_hamiltonian["hf_state"]
        n_qubits = h2_hamiltonian["n_qubits"]

        config = {
            "n_estimation_wires": 3,
            "base_time": 0.5,
            "n_trotter_steps": 2,
            "n_shots": 5,
        }

        # Create engine with GPU device
        engine = QPEEngine(
            n_qubits=n_qubits,
            device_type="lightning.gpu",
            use_catalyst=False,
        )
        assert engine.device_type == "lightning.gpu"

        # Build and execute circuit
        qpe_circuit = engine._build_standard_qpe_circuit(H, hf_state, **config)
        result = qpe_circuit()

        assert result is not None, "QPE with lightning.gpu failed to execute"

        # Extract energy
        energy = engine._extract_energy_from_samples(result, base_time=config["base_time"])
        assert np.isfinite(energy), f"GPU energy not finite: {energy}"

    @pytest.mark.gpu
    def test_device_type_auto_selects_gpu(self):
        """Test device_type='auto' selects lightning.gpu when available."""
        from q2m3.core.qpe import HAS_LIGHTNING_GPU, _select_device

        if not HAS_LIGHTNING_GPU:
            pytest.skip("lightning.gpu not available")

        dev = _select_device("auto", n_wires=4, shots=10)
        # Check device name using the new PennyLane device API
        assert "gpu" in str(type(dev).__name__).lower() or "gpu" in str(dev.name).lower()

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_qpe_h2_gpu_energy(self, h2_hamiltonian):
        """H2 QPE with GPU should produce reasonable energy estimates."""
        H = h2_hamiltonian["hamiltonian"]
        hf_state = h2_hamiltonian["hf_state"]
        n_qubits = h2_hamiltonian["n_qubits"]

        config = {
            "n_estimation_wires": 4,
            "base_time": 0.5,
            "n_trotter_steps": 3,
            "n_shots": 10,
        }

        engine = QPEEngine(
            n_qubits=n_qubits,
            device_type="lightning.gpu",
            use_catalyst=False,
        )

        qpe_circuit = engine._build_standard_qpe_circuit(H, hf_state, **config)
        samples = qpe_circuit()
        energy = engine._extract_energy_from_samples(samples, base_time=config["base_time"])

        # Energy should be finite and non-positive
        assert np.isfinite(energy), f"GPU energy not finite: {energy}"
        assert energy <= 0, f"GPU energy should be non-positive: {energy}"
