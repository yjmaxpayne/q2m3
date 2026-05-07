# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Tests for q2m3.solvation.energy module."""

import math
import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

from q2m3.solvation.config import QPEConfig, SolvationConfig
from q2m3.solvation.energy import (
    StepResult,
    compute_hf_energy_solvated,
    compute_hf_energy_vacuum,
    compute_mulliken_charges,
    create_coefficient_callback,
    create_hf_corrected_step_callback,
    create_step_callback,
    precompute_vacuum_cache,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def solvation_config_fixed(h2_molecule_config):
    """SolvationConfig with fixed mode."""
    return SolvationConfig(
        molecule=h2_molecule_config,
        qpe_config=QPEConfig(n_estimation_wires=3, n_trotter_steps=2),
        hamiltonian_mode="fixed",
        n_waters=3,
        n_mc_steps=10,
        verbose=False,
    )


@pytest.fixture
def solvation_config_dynamic(h2_molecule_config):
    """SolvationConfig with dynamic mode."""
    return SolvationConfig(
        molecule=h2_molecule_config,
        qpe_config=QPEConfig(n_estimation_wires=3, n_trotter_steps=2),
        hamiltonian_mode="dynamic",
        n_waters=3,
        n_mc_steps=10,
        verbose=False,
    )


@pytest.fixture
def solvation_config_hf_corrected(h2_molecule_config):
    """SolvationConfig with hf_corrected mode (qpe_interval=3 for testing)."""
    return SolvationConfig(
        molecule=h2_molecule_config,
        qpe_config=QPEConfig(n_estimation_wires=3, n_trotter_steps=2, qpe_interval=3),
        hamiltonian_mode="hf_corrected",
        n_waters=3,
        n_mc_steps=10,
        verbose=False,
    )


@pytest.fixture
def h2_vacuum_cache(solvation_config_fixed):
    """Pre-computed vacuum cache for H2."""
    return precompute_vacuum_cache(solvation_config_fixed)


@pytest.fixture
def mock_circuit_bundle(h2_vacuum_cache):
    """Mock QPECircuitBundle for testing step callbacks without Catalyst."""
    n_estimation_wires = 3
    n_bins = 2**n_estimation_wires

    # Fake probability distribution peaked around bin 3
    fake_probs = np.zeros(n_bins)
    fake_probs[3] = 0.8
    fake_probs[2] = 0.1
    fake_probs[4] = 0.1

    bundle = MagicMock()
    bundle.base_coeffs = np.array([1.0, 0.5, -0.3, 0.2, -0.1], dtype=np.float64)
    bundle.compiled_circuit = MagicMock(return_value=fake_probs)
    bundle.op_index_map = {"identity_idx": 0, "z_wire_idx": {0: 1, 1: 2, 2: 3, 3: 4}}
    bundle.energy_shift = -1.117
    bundle.n_estimation_wires = n_estimation_wires
    bundle.base_time = 1.0
    bundle.active_orbitals = 2
    bundle.measurement_mode = "probs"
    bundle.is_fixed_circuit = False  # Default: dynamic/hf_corrected style
    return bundle


# ============================================================================
# StepResult Tests
# ============================================================================


class TestStepResult:
    def test_creation(self):
        """StepResult fields are correctly set."""
        result = StepResult(
            e_qpe=-1.15,
            e_mm_sol_sol=0.002,
            e_hf_ref=-1.117,
            callback_time=0.05,
            qpe_time=0.3,
        )
        assert result.e_qpe == -1.15
        assert result.e_mm_sol_sol == 0.002
        assert result.e_hf_ref == -1.117
        assert result.callback_time == 0.05
        assert result.qpe_time == 0.3

    def test_frozen(self):
        """StepResult is immutable."""
        result = StepResult(
            e_qpe=-1.15,
            e_mm_sol_sol=0.002,
            e_hf_ref=-1.117,
            callback_time=0.05,
            qpe_time=0.3,
        )
        with pytest.raises(AttributeError):
            result.e_qpe = -2.0


# ============================================================================
# precompute_vacuum_cache Tests
# ============================================================================


class TestPrecomputeVacuumCache:
    def test_h2_vacuum_energy(self, h2_vacuum_cache):
        """H2/STO-3G vacuum HF energy is approximately -1.117 Ha."""
        assert abs(h2_vacuum_cache["e_vacuum"] - (-1.117)) < 0.01

    def test_h2_cache_keys(self, h2_vacuum_cache):
        """Cache contains all required keys."""
        expected_keys = {
            "h_core_vac",
            "mo_coeff",
            "mo_coeff_active",
            "energy_nuc_vac",
            "e_vacuum",
            "active_idx",
        }
        assert set(h2_vacuum_cache.keys()) == expected_keys

    def test_h2_active_idx(self, h2_vacuum_cache):
        """H2 (2e, 2o) active space indices are [0, 1]."""
        assert h2_vacuum_cache["active_idx"] == [0, 1]

    def test_h2_mo_coeff_shape(self, h2_vacuum_cache):
        """MO coefficient matrices have correct shapes."""
        # H2/STO-3G: 2 basis functions, 2 MOs
        assert h2_vacuum_cache["mo_coeff"].shape == (2, 2)
        assert h2_vacuum_cache["mo_coeff_active"].shape == (2, 2)

    def test_h2_h_core_is_symmetric(self, h2_vacuum_cache):
        """Core Hamiltonian is Hermitian (symmetric for real case)."""
        h_core = h2_vacuum_cache["h_core_vac"]
        np.testing.assert_allclose(h_core, h_core.T, atol=1e-12)


# ============================================================================
# create_coefficient_callback Tests
# ============================================================================


class TestCoefficientCallback:
    def test_fixed_mode_returns_base_coeffs(self, solvation_config_fixed, mock_circuit_bundle):
        """Fixed mode callback returns base_coeffs unchanged."""
        callback = create_coefficient_callback(
            solvation_config_fixed, mock_circuit_bundle, {"e_vacuum": -1.117}
        )
        dummy_solvents = np.zeros((3, 6))
        dummy_coords = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.74])

        result = callback(dummy_solvents, dummy_coords)
        np.testing.assert_array_equal(result, mock_circuit_bundle.base_coeffs)

    def test_fixed_mode_returns_copy(self, solvation_config_fixed, mock_circuit_bundle):
        """Fixed mode returns a copy, not the original array."""
        callback = create_coefficient_callback(
            solvation_config_fixed, mock_circuit_bundle, {"e_vacuum": -1.117}
        )
        result1 = callback(np.zeros((3, 6)), np.zeros(6))
        result2 = callback(np.zeros((3, 6)), np.zeros(6))
        assert result1 is not result2

    @pytest.mark.slow
    def test_dynamic_mode_modifies_coeffs(
        self, solvation_config_dynamic, mock_circuit_bundle, h2_vacuum_cache, water_molecules_3
    ):
        """Dynamic mode patches coefficients with MM delta."""
        from q2m3.solvation.solvent import molecules_to_state_array

        callback = create_coefficient_callback(
            solvation_config_dynamic, mock_circuit_bundle, h2_vacuum_cache
        )
        solvent_states = molecules_to_state_array(water_molecules_3)
        qm_coords = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.74])

        result = callback(solvent_states, qm_coords)
        # Dynamic mode should modify coefficients (not equal to base)
        assert not np.array_equal(result, mock_circuit_bundle.base_coeffs)
        # But same shape
        assert result.shape == mock_circuit_bundle.base_coeffs.shape


# ============================================================================
# create_step_callback Tests
# ============================================================================


class TestStepCallback:
    def test_returns_step_result(
        self, solvation_config_fixed, mock_circuit_bundle, h2_vacuum_cache
    ):
        """Step callback returns StepResult."""
        step_cb = create_step_callback(mock_circuit_bundle, solvation_config_fixed, h2_vacuum_cache)
        dummy_solvents = np.zeros((3, 6))
        dummy_coords = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.74])

        result = step_cb(dummy_solvents, dummy_coords)
        assert isinstance(result, StepResult)

    def test_fixed_ehf_from_vacuum(
        self, solvation_config_fixed, mock_circuit_bundle, h2_vacuum_cache
    ):
        """Fixed mode: e_hf_ref comes from vacuum cache."""
        step_cb = create_step_callback(mock_circuit_bundle, solvation_config_fixed, h2_vacuum_cache)
        result = step_cb(np.zeros((3, 6)), np.zeros(6))
        assert result.e_hf_ref == h2_vacuum_cache["e_vacuum"]

    def test_calls_compiled_circuit(
        self, solvation_config_fixed, mock_circuit_bundle, h2_vacuum_cache
    ):
        """Step callback invokes compiled_circuit."""
        step_cb = create_step_callback(mock_circuit_bundle, solvation_config_fixed, h2_vacuum_cache)
        step_cb(np.zeros((3, 6)), np.zeros(6))
        mock_circuit_bundle.compiled_circuit.assert_called_once()

    def test_qpe_time_positive(self, solvation_config_fixed, mock_circuit_bundle, h2_vacuum_cache):
        """QPE time measurement is non-negative."""
        step_cb = create_step_callback(mock_circuit_bundle, solvation_config_fixed, h2_vacuum_cache)
        result = step_cb(np.zeros((3, 6)), np.zeros(6))
        assert result.qpe_time >= 0.0


# ============================================================================
# create_hf_corrected_step_callback Tests
# ============================================================================


class TestHfCorrectedStepCallback:
    def test_non_qpe_step_returns_nan(
        self, solvation_config_hf_corrected, mock_circuit_bundle, h2_vacuum_cache
    ):
        """Non-QPE step returns NaN e_qpe and zero qpe_time."""
        qm_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        step_cb = create_hf_corrected_step_callback(
            solvation_config_hf_corrected,
            h2_vacuum_cache,
            qm_coords,
            h2_vacuum_cache["e_vacuum"],
            circuit_bundle=mock_circuit_bundle,
        )
        dummy_solvents = np.zeros((3, 6))
        dummy_coords = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.74])

        # First call is step 0 (QPE step) - skip
        _ = step_cb(dummy_solvents, dummy_coords)

        # Second call is step 1 (non-QPE step)
        result = step_cb(dummy_solvents, dummy_coords)
        assert math.isnan(result.e_qpe)
        assert result.qpe_time == 0.0

    def test_qpe_step_returns_valid_energy(
        self, solvation_config_hf_corrected, mock_circuit_bundle, h2_vacuum_cache
    ):
        """QPE step (at interval) returns valid e_qpe."""
        qm_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        step_cb = create_hf_corrected_step_callback(
            solvation_config_hf_corrected,
            h2_vacuum_cache,
            qm_coords,
            h2_vacuum_cache["e_vacuum"],
            circuit_bundle=mock_circuit_bundle,
        )
        dummy_solvents = np.zeros((3, 6))
        dummy_coords = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.74])

        # Step 0: 0 % 3 == 0 → QPE step
        result = step_cb(dummy_solvents, dummy_coords)
        assert not math.isnan(result.e_qpe)
        assert result.qpe_time > 0.0

    def test_closure_counter_increments(
        self, solvation_config_hf_corrected, mock_circuit_bundle, h2_vacuum_cache
    ):
        """Closure counter correctly tracks steps: QPE at 0, 3, 6..."""
        qm_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        step_cb = create_hf_corrected_step_callback(
            solvation_config_hf_corrected,
            h2_vacuum_cache,
            qm_coords,
            h2_vacuum_cache["e_vacuum"],
            circuit_bundle=mock_circuit_bundle,
        )
        dummy_solvents = np.zeros((3, 6))
        dummy_coords = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.74])

        # qpe_interval = 3, so QPE at steps 0, 3, 6...
        results = [step_cb(dummy_solvents, dummy_coords) for _ in range(6)]

        # Steps 0, 3 should have valid QPE; steps 1, 2, 4, 5 should have NaN
        assert not math.isnan(results[0].e_qpe)  # step 0
        assert math.isnan(results[1].e_qpe)  # step 1
        assert math.isnan(results[2].e_qpe)  # step 2
        assert not math.isnan(results[3].e_qpe)  # step 3
        assert math.isnan(results[4].e_qpe)  # step 4
        assert math.isnan(results[5].e_qpe)  # step 5

    def test_ehf_ref_is_always_set(
        self, solvation_config_hf_corrected, mock_circuit_bundle, h2_vacuum_cache
    ):
        """e_hf_ref is always valid (never NaN), even on non-QPE steps."""
        qm_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        step_cb = create_hf_corrected_step_callback(
            solvation_config_hf_corrected,
            h2_vacuum_cache,
            qm_coords,
            h2_vacuum_cache["e_vacuum"],
            circuit_bundle=mock_circuit_bundle,
        )
        dummy_solvents = np.zeros((3, 6))
        dummy_coords = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.74])

        for _ in range(4):
            result = step_cb(dummy_solvents, dummy_coords)
            assert not math.isnan(result.e_hf_ref)


# ============================================================================
# Helper PySCF Function Tests
# ============================================================================


class TestComputeHfEnergyVacuum:
    def test_h2_energy(self, h2_molecule_config):
        """H2/STO-3G vacuum HF energy is approximately -1.117 Ha."""
        e = compute_hf_energy_vacuum(h2_molecule_config)
        assert abs(e - (-1.117)) < 0.01


class TestComputeHfEnergySolvated:
    def test_h2_solvated_differs_from_vacuum(self, h2_molecule_config, water_molecules_3):
        """Solvated HF energy differs from vacuum for H2 + 3 waters."""
        e_vac = compute_hf_energy_vacuum(h2_molecule_config)
        e_sol = compute_hf_energy_solvated(h2_molecule_config, water_molecules_3)
        assert e_sol != e_vac

    def test_h2_no_solvent_equals_vacuum(self, h2_molecule_config):
        """With no solvent, solvated energy equals vacuum."""
        e_vac = compute_hf_energy_vacuum(h2_molecule_config)
        e_sol = compute_hf_energy_solvated(h2_molecule_config, [])
        assert abs(e_sol - e_vac) < 1e-10

    def test_qmmm_overlap_returns_inf_without_runtime_warning(self, h2_molecule_config):
        """MM point charges on QM nuclei are rejected before PySCF QMMM evaluation."""
        from q2m3.solvation.solvent import SOLVENT_MODELS, state_array_to_molecules

        solvent_molecules = state_array_to_molecules(SOLVENT_MODELS["TIP3P"], np.zeros((1, 6)))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            energy = compute_hf_energy_solvated(h2_molecule_config, solvent_molecules)

        assert math.isinf(energy)


class TestComputeMullikenCharges:
    def test_vacuum_charges_sum_to_total_charge(self, h2_molecule_config):
        """Mulliken charges sum to molecular charge (0 for H2)."""
        charges = compute_mulliken_charges(h2_molecule_config)
        total = sum(charges.values())
        assert abs(total - h2_molecule_config.charge) < 1e-6

    def test_vacuum_h2_symmetric(self, h2_molecule_config):
        """H2 has symmetric charges on both atoms."""
        charges = compute_mulliken_charges(h2_molecule_config)
        assert abs(charges["H0"] - charges["H1"]) < 1e-6

    def test_solvated_charges(self, h2_molecule_config, water_molecules_3):
        """Solvated Mulliken charges still sum to total charge."""
        from q2m3.solvation.solvent import molecules_to_state_array

        solvent_states = molecules_to_state_array(water_molecules_3)
        charges = compute_mulliken_charges(h2_molecule_config, solvent_states=solvent_states)
        total = sum(charges.values())
        assert abs(total - h2_molecule_config.charge) < 1e-4
