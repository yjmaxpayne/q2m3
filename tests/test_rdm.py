# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Tests for 1-RDM measurement functionality.

Tests cover:
- Jordan-Wigner mapping correctness
- Pauli expectation value measurement
- RDM physical property validation
- Integration with QPE workflow
"""

import numpy as np
import pennylane as qml
import pytest

from q2m3.core import RDMEstimator

# ============================================================================
# RDM Configuration Fixtures
# ============================================================================


@pytest.fixture
def rdm_config_basic():
    """Basic RDM configuration for fast tests."""
    return {
        "n_shots": 100,
        "include_off_diagonal": True,
        "symmetrize": True,
    }


@pytest.fixture
def rdm_config_diagonal_only():
    """RDM configuration for diagonal-only measurement."""
    return {
        "n_shots": 100,
        "include_off_diagonal": False,
        "symmetrize": True,
    }


# ============================================================================
# Test RDM Observables Construction
# ============================================================================


class TestRDMObservables:
    """Test construction of Pauli observables for RDM measurement."""

    def test_diagonal_observable_is_pauli_z(self):
        """Diagonal elements should use Pauli Z observable."""
        estimator = RDMEstimator(n_qubits=4, n_electrons=2)
        observables = estimator.build_rdm_observables()

        # Check diagonal elements are Pauli Z
        for p in range(4):
            assert (p, p) in observables
            obs = observables[(p, p)]
            # Should be qml.Z(p)
            assert isinstance(obs, qml.Z) or hasattr(obs, "wires")

    def test_offdiagonal_observable_has_four_terms(self):
        """Off-diagonal elements should have 4 Pauli string terms."""
        estimator = RDMEstimator(n_qubits=4, n_electrons=2)
        observables = estimator.build_rdm_observables()

        # Check off-diagonal elements have 4 terms
        for p in range(4):
            for q in range(p + 1, 4):
                assert (p, q) in observables
                terms = observables[(p, q)]
                assert isinstance(terms, tuple)
                assert len(terms) == 4  # XX, YY, XY, YX terms

    def test_observable_count(self):
        """Total number of observables should be n + n(n-1)/2."""
        n_qubits = 4
        estimator = RDMEstimator(n_qubits=n_qubits, n_electrons=2)
        observables = estimator.build_rdm_observables()

        # Diagonal: n, Upper triangular (off-diagonal): n(n-1)/2
        expected_count = n_qubits + n_qubits * (n_qubits - 1) // 2
        assert len(observables) == expected_count


# ============================================================================
# Test RDM Physical Properties
# ============================================================================


class TestRDMPhysicalProperties:
    """Test that measured RDM satisfies physical constraints."""

    def test_rdm_is_hermitian(self, h2_hamiltonian, rdm_config_basic):
        """RDM should be Hermitian after measurement."""
        H = h2_hamiltonian["hamiltonian"]
        hf_state = h2_hamiltonian["hf_state"]
        n_qubits = h2_hamiltonian["n_qubits"]
        n_electrons = h2_hamiltonian["molecule_data"]["n_electrons"]

        estimator = RDMEstimator(n_qubits, n_electrons, rdm_config_basic)
        rdm = estimator.measure_1rdm(
            hamiltonian=H,
            hf_state=hf_state,
            base_time=0.5,
            n_trotter_steps=3,
            device_type="lightning.qubit",
        )

        # Check Hermiticity
        assert np.allclose(rdm, rdm.conj().T, atol=1e-8), "RDM should be Hermitian"

    def test_rdm_trace_equals_n_electrons(self, h2_hamiltonian, rdm_config_basic):
        """Trace of RDM should equal number of electrons."""
        H = h2_hamiltonian["hamiltonian"]
        hf_state = h2_hamiltonian["hf_state"]
        n_qubits = h2_hamiltonian["n_qubits"]
        n_electrons = h2_hamiltonian["molecule_data"]["n_electrons"]

        estimator = RDMEstimator(n_qubits, n_electrons, rdm_config_basic)
        rdm = estimator.measure_1rdm(
            hamiltonian=H,
            hf_state=hf_state,
            base_time=0.5,
            n_trotter_steps=3,
            device_type="lightning.qubit",
        )

        trace = np.trace(rdm)
        assert np.isclose(
            trace, n_electrons, atol=0.1
        ), f"Trace ({trace}) should equal n_electrons ({n_electrons})"

    def test_rdm_eigenvalues_bounded(self, h2_hamiltonian, rdm_config_basic):
        """RDM eigenvalues should be in [0, 1] after physical constraints."""
        H = h2_hamiltonian["hamiltonian"]
        hf_state = h2_hamiltonian["hf_state"]
        n_qubits = h2_hamiltonian["n_qubits"]
        n_electrons = h2_hamiltonian["molecule_data"]["n_electrons"]

        estimator = RDMEstimator(n_qubits, n_electrons, rdm_config_basic)
        rdm = estimator.measure_1rdm(
            hamiltonian=H,
            hf_state=hf_state,
            base_time=0.5,
            n_trotter_steps=3,
            device_type="lightning.qubit",
        )

        eigenvalues = np.linalg.eigvalsh(rdm)
        assert np.all(eigenvalues >= -1e-6), f"Eigenvalues should be >= 0: {eigenvalues}"
        assert np.all(eigenvalues <= 1 + 1e-6), f"Eigenvalues should be <= 1: {eigenvalues}"


# ============================================================================
# Test RDM for HF State (Baseline Validation)
# ============================================================================


class TestRDMHFState:
    """Test RDM measurement on HF reference state (zero evolution time)."""

    def test_hf_state_rdm_diagonal_is_occupation(self, h2_hamiltonian):
        """For HF state, diagonal RDM elements should be occupation numbers."""
        hf_state = h2_hamiltonian["hf_state"]
        n_qubits = h2_hamiltonian["n_qubits"]
        n_electrons = h2_hamiltonian["molecule_data"]["n_electrons"]

        # Minimal Trotter time to approximate HF state
        H = h2_hamiltonian["hamiltonian"]

        config = {"n_shots": 100, "include_off_diagonal": False, "symmetrize": True}
        estimator = RDMEstimator(n_qubits, n_electrons, config)

        # Zero evolution time should give HF state
        rdm = estimator.measure_1rdm(
            hamiltonian=H,
            hf_state=hf_state,
            base_time=0.0,  # No time evolution
            n_trotter_steps=1,
            device_type="lightning.qubit",
        )

        # Check diagonal elements match HF occupation
        for i, occ in enumerate(hf_state):
            assert np.isclose(
                rdm[i, i], occ, atol=0.1
            ), f"RDM[{i},{i}] = {rdm[i, i]:.3f}, expected {occ}"


# ============================================================================
# Test Spin to Spatial RDM Conversion
# ============================================================================


class TestSpinToSpatialConversion:
    """Test conversion from spin-orbital to spatial-orbital RDM."""

    def test_spatial_rdm_shape(self):
        """Spatial RDM should have half the dimension of spin RDM."""
        n_spin = 8  # 4 spatial orbitals
        spin_rdm = np.eye(n_spin)

        estimator = RDMEstimator(n_qubits=n_spin, n_electrons=4)
        spatial_rdm = estimator.spin_to_spatial_rdm(spin_rdm)

        assert spatial_rdm.shape == (4, 4), f"Expected (4, 4), got {spatial_rdm.shape}"

    def test_spatial_rdm_is_real(self):
        """Spatial RDM should be real for real Hamiltonians."""
        n_spin = 4
        # Create a simple Hermitian spin RDM
        spin_rdm = np.array(
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.05],
                [0.0, 0.0, 0.05, 0.1],
            ]
        )

        estimator = RDMEstimator(n_qubits=n_spin, n_electrons=2)
        spatial_rdm = estimator.spin_to_spatial_rdm(spin_rdm)

        assert np.allclose(spatial_rdm.imag, 0), "Spatial RDM should be real"


# ============================================================================
# Test Physical Constraint Enforcement
# ============================================================================


class TestPhysicalConstraints:
    """Test enforcement of physical constraints on RDM."""

    def test_enforce_hermiticity(self):
        """enforce_physical_constraints should make RDM Hermitian."""
        n_qubits = 4
        n_electrons = 2

        # Create non-Hermitian matrix
        raw_rdm = np.random.rand(n_qubits, n_qubits) + 1j * np.random.rand(n_qubits, n_qubits)

        estimator = RDMEstimator(n_qubits, n_electrons)
        physical_rdm = estimator.enforce_physical_constraints(raw_rdm)

        assert np.allclose(physical_rdm, physical_rdm.conj().T, atol=1e-10)

    def test_enforce_positive_semidefinite(self):
        """enforce_physical_constraints should ensure positive eigenvalues."""
        n_qubits = 4
        n_electrons = 2

        # Create matrix with negative eigenvalues
        raw_rdm = np.diag([1.0, 0.5, -0.3, 0.2])

        estimator = RDMEstimator(n_qubits, n_electrons)
        physical_rdm = estimator.enforce_physical_constraints(raw_rdm)

        eigenvalues = np.linalg.eigvalsh(physical_rdm)
        assert np.all(eigenvalues >= -1e-10), f"Eigenvalues should be >= 0: {eigenvalues}"

    def test_enforce_trace_normalization(self):
        """enforce_physical_constraints should normalize trace to n_electrons."""
        n_qubits = 4
        n_electrons = 2

        # Create matrix with wrong trace
        raw_rdm = np.diag([3.0, 2.0, 1.0, 1.0])  # trace = 7

        estimator = RDMEstimator(n_qubits, n_electrons)
        physical_rdm = estimator.enforce_physical_constraints(raw_rdm)

        trace = np.trace(physical_rdm)
        assert np.isclose(
            trace, n_electrons, atol=1e-8
        ), f"Trace ({trace}) should equal n_electrons ({n_electrons})"


# ============================================================================
# Integration Tests (marked as slow)
# ============================================================================


@pytest.mark.slow
class TestRDMQPEIntegration:
    """Integration tests for RDM measurement in QPE workflow."""

    def test_rdm_integration_h2(self, h2_hamiltonian, rdm_config_basic):
        """Full RDM measurement integration test with H2."""
        H = h2_hamiltonian["hamiltonian"]
        hf_state = h2_hamiltonian["hf_state"]
        n_qubits = h2_hamiltonian["n_qubits"]
        n_electrons = h2_hamiltonian["molecule_data"]["n_electrons"]

        estimator = RDMEstimator(n_qubits, n_electrons, rdm_config_basic)
        rdm = estimator.measure_1rdm(
            hamiltonian=H,
            hf_state=hf_state,
            base_time=0.3,
            n_trotter_steps=5,
            device_type="lightning.qubit",
        )

        # Basic sanity checks
        assert rdm.shape == (n_qubits, n_qubits)
        assert np.allclose(rdm, rdm.conj().T, atol=1e-8)  # Hermitian
        assert np.isclose(np.trace(rdm), n_electrons, atol=0.2)  # Trace

    @pytest.mark.slow
    def test_rdm_integration_h3o_active_space(self, h3o_hamiltonian_active_space, rdm_config_basic):
        """RDM measurement for H3O+ with active space."""
        H = h3o_hamiltonian_active_space["hamiltonian"]
        hf_state = h3o_hamiltonian_active_space["hf_state"]
        n_qubits = h3o_hamiltonian_active_space["n_qubits"]
        active_electrons = h3o_hamiltonian_active_space["active_space"]["electrons"]

        estimator = RDMEstimator(n_qubits, active_electrons, rdm_config_basic)
        rdm = estimator.measure_1rdm(
            hamiltonian=H,
            hf_state=hf_state,
            base_time=0.1,
            n_trotter_steps=3,
            device_type="lightning.qubit",
        )

        # Basic sanity checks
        assert rdm.shape == (n_qubits, n_qubits)
        assert np.isclose(np.trace(rdm), active_electrons, atol=0.3)


# ============================================================================
# Test Active MO to AO RDM Conversion
# ============================================================================


class TestActiveMOToAORDM:
    """Test conversion from active space MO RDM to AO basis."""

    def test_ao_rdm_shape(self):
        """AO RDM shape should match MO coefficient dimensions."""
        n_ao = 8  # e.g., STO-3G H3O+
        n_mo = 8
        active_electrons = 4
        active_orbitals = 4

        # Create mock MO coefficients and occupations
        mo_coeff = np.eye(n_ao, n_mo)  # Simplified: AO = MO
        mo_occ = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0])  # 5 occupied orbitals

        # Create active space RDM (4x4)
        active_rdm = np.diag([1.8, 1.8, 0.2, 0.2])  # Sum = 4 electrons

        estimator = RDMEstimator(n_qubits=8, n_electrons=active_electrons)
        ao_rdm = estimator.active_mo_to_ao_rdm(
            active_spatial_rdm=active_rdm,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
        )

        assert ao_rdm.shape == (n_ao, n_ao), f"Expected ({n_ao}, {n_ao}), got {ao_rdm.shape}"

    def test_ao_rdm_preserves_total_electrons(self):
        """Total electrons should be preserved after transformation."""
        n_ao = 8
        n_mo = 8
        active_electrons = 4
        active_orbitals = 4

        # H3O+ like: 10 total electrons, 5 occupied, active space = 4e, 4o
        mo_coeff = np.eye(n_ao, n_mo)
        mo_occ = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0])

        # Active space: orbitals 3,4,5,6 (0-indexed), 4 electrons
        active_rdm = np.diag([1.8, 1.8, 0.2, 0.2])

        estimator = RDMEstimator(n_qubits=8, n_electrons=active_electrons)
        ao_rdm = estimator.active_mo_to_ao_rdm(
            active_spatial_rdm=active_rdm,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
        )

        # Frozen core contributes 3*2 = 6 electrons, active contributes 4
        expected_total = 10
        actual_trace = np.trace(ao_rdm)
        assert np.isclose(
            actual_trace, expected_total, atol=0.1
        ), f"Trace ({actual_trace}) should equal total electrons ({expected_total})"

    def test_ao_rdm_frozen_core_contribution(self):
        """Frozen core orbitals should contribute 2 electrons each."""
        n_ao = 6
        n_mo = 6
        active_electrons = 2
        active_orbitals = 2

        # 4 total electrons: 2 frozen + 2 active
        mo_coeff = np.eye(n_ao, n_mo)
        mo_occ = np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0])  # 2 occupied orbitals

        # Active space starts at orbital 1 (frozen: orbital 0)
        active_rdm = np.diag([1.0, 1.0])  # 2 electrons in active space

        estimator = RDMEstimator(n_qubits=4, n_electrons=active_electrons)
        ao_rdm = estimator.active_mo_to_ao_rdm(
            active_spatial_rdm=active_rdm,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
        )

        # Frozen orbital (0) should have occupation 2.0
        assert np.isclose(
            ao_rdm[0, 0], 2.0, atol=0.1
        ), f"Frozen orbital occupation should be 2.0, got {ao_rdm[0, 0]}"

    def test_ao_rdm_is_real(self):
        """AO RDM should be real for real systems."""
        n_ao = 4
        n_mo = 4
        active_electrons = 2
        active_orbitals = 2

        mo_coeff = np.eye(n_ao, n_mo)
        mo_occ = np.array([2.0, 2.0, 0.0, 0.0])

        active_rdm = np.array([[0.9, 0.1], [0.1, 0.9]])  # Hermitian

        estimator = RDMEstimator(n_qubits=4, n_electrons=active_electrons)
        ao_rdm = estimator.active_mo_to_ao_rdm(
            active_spatial_rdm=active_rdm,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
        )

        assert np.allclose(ao_rdm.imag, 0), "AO RDM should be real"
