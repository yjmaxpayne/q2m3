# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Tests for solvation analysis functions.

Tests use synthetic data only — no Catalyst or PySCF required.
"""

import numpy as np
import pytest

from q2m3.solvation.analysis import (
    DeltaCorrPolResult,
    EnergyPhaseResult,
    EquilibrationResult,
    ModeComparisonResult,
    QPEHFConsistencyResult,
    analyze_energy_phases,
    compute_delta_corr_pol,
    compute_qpe_hf_consistency,
    detect_equilibration,
    run_mode_comparison,
)

# ---------------------------------------------------------------------------
# compute_delta_corr_pol
# ---------------------------------------------------------------------------


def test_compute_delta_corr_pol_basic():
    """Known delta values produce correct mean/std/SEM/t-stat."""
    fixed = np.array([1.0, 2.0, 3.0])
    dynamic = np.array([2.0, 3.0, 4.0])  # delta = [1, 1, 1]
    result = compute_delta_corr_pol(fixed, dynamic)

    assert isinstance(result, DeltaCorrPolResult)
    assert result.mean_ha == pytest.approx(1.0, abs=1e-10)
    assert result.std_ha == pytest.approx(0.0, abs=1e-10)
    assert result.sem_ha == pytest.approx(0.0, abs=1e-10)
    assert result.n_samples == 3
    assert result.is_significant is False  # t=0 < 2.0
    assert len(result.per_step_delta) == 3
    np.testing.assert_allclose(result.per_step_delta, [1.0, 1.0, 1.0])


def test_compute_delta_corr_pol_varying_delta():
    """Varying delta: t-stat computed correctly."""
    fixed = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    dynamic = np.array([2.0, 3.0, 4.0, 5.0, 6.0])  # delta = [1, 2, 3, 4, 5]
    result = compute_delta_corr_pol(fixed, dynamic)

    expected_mean = 3.0
    expected_std = np.std([1, 2, 3, 4, 5], ddof=1)
    expected_sem = expected_std / np.sqrt(5)
    expected_t = expected_mean / expected_sem

    assert result.mean_ha == pytest.approx(expected_mean, rel=1e-9)
    assert result.std_ha == pytest.approx(expected_std, rel=1e-9)
    assert result.sem_ha == pytest.approx(expected_sem, rel=1e-9)
    assert result.t_statistic == pytest.approx(expected_t, rel=1e-9)
    assert result.is_significant is True  # t >> 2.0


def test_compute_delta_corr_pol_zero_delta():
    """Identical arrays: mean=0, std=0, t=0."""
    arr = np.array([1.0, 2.0, 3.0])
    result = compute_delta_corr_pol(arr, arr)

    assert result.mean_ha == pytest.approx(0.0, abs=1e-12)
    assert result.t_statistic == pytest.approx(0.0, abs=1e-12)
    assert result.is_significant is False


def test_compute_delta_corr_pol_single_sample():
    """n=1: SEM=0 (std undefined), t=0 by convention."""
    fixed = np.array([1.0])
    dynamic = np.array([2.0])
    result = compute_delta_corr_pol(fixed, dynamic)

    assert result.n_samples == 1
    assert result.sem_ha == pytest.approx(0.0, abs=1e-12)
    assert result.t_statistic == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# analyze_energy_phases
# ---------------------------------------------------------------------------


def test_analyze_energy_phases_decreasing():
    """Monotonically decreasing energies: early_mean > late_mean."""
    energies = np.linspace(10.0, 1.0, 30)  # decreasing
    result = analyze_energy_phases(energies, n_phases=3)

    assert isinstance(result, EnergyPhaseResult)
    assert result.early_mean > result.late_mean
    assert result.n_per_phase == 10  # 30 // 3


def test_analyze_energy_phases_constant():
    """Constant energies: early == late, std=0."""
    energies = np.full(15, 5.0)
    result = analyze_energy_phases(energies, n_phases=3)

    assert result.early_mean == pytest.approx(5.0, abs=1e-12)
    assert result.late_mean == pytest.approx(5.0, abs=1e-12)
    assert result.early_std == pytest.approx(0.0, abs=1e-12)
    assert result.late_std == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# compute_qpe_hf_consistency
# ---------------------------------------------------------------------------


def test_qpe_hf_consistency_perfect_linear():
    """QPE = HF + constant offset → correlation=1.0."""
    hf = np.linspace(-1.0, 1.0, 20)
    qpe = hf + 0.05  # constant offset of 50 mHa
    result = compute_qpe_hf_consistency(qpe, hf)

    assert isinstance(result, QPEHFConsistencyResult)
    assert result.pearson_correlation == pytest.approx(1.0, abs=1e-9)
    assert result.mean_offset_mha == pytest.approx(50.0, abs=1e-6)
    assert result.n_samples == 20


def test_qpe_hf_consistency_uncorrelated():
    """Random QPE vs monotone HF → low |correlation|."""
    rng = np.random.default_rng(42)
    hf = np.linspace(-1.0, 1.0, 50)
    qpe = rng.standard_normal(50)  # no relation to hf
    result = compute_qpe_hf_consistency(qpe, hf)

    assert abs(result.pearson_correlation) < 0.4


# ---------------------------------------------------------------------------
# detect_equilibration
# ---------------------------------------------------------------------------


def test_detect_equilibration_insufficient_samples():
    """Fewer than min_samples → returns None."""
    energies = np.array([1.0, 2.0, 3.0])  # n=3 < 10
    result = detect_equilibration(energies, min_samples=10)

    assert result is None


def test_detect_equilibration_monotonic():
    """Monotonically decreasing → is_monotonic=True."""
    energies = np.linspace(5.0, 1.0, 50)  # strictly decreasing
    result = detect_equilibration(energies, min_samples=10)

    assert isinstance(result, EquilibrationResult)
    assert result.is_monotonic is True
    assert result.frac_decreasing > 0.8


def test_detect_equilibration_flat():
    """Flat (equilibrated) → is_monotonic=False."""
    rng = np.random.default_rng(0)
    energies = rng.standard_normal(50) + 0.0  # stationary around 0
    result = detect_equilibration(energies, min_samples=10)

    assert isinstance(result, EquilibrationResult)
    assert result.is_monotonic is False


# ---------------------------------------------------------------------------
# run_mode_comparison (integration)
# ---------------------------------------------------------------------------


def _make_result_dict(
    n: int = 20,
    base_energy: float = -1.0,
    offset: float = 0.0,
    seed: int = 0,
) -> dict:
    """Build synthetic run_solvation result dict for testing."""
    rng = np.random.default_rng(seed)
    q_energies = rng.standard_normal(n) * 0.01 + base_energy + offset
    hf_energies = rng.standard_normal(n) * 0.005 + base_energy - 0.05
    return {
        "quantum_energies": q_energies,
        "hf_energies": hf_energies,
    }


def test_run_mode_comparison_integration():
    """Synthetic result dicts → complete ModeComparisonResult."""
    fixed = _make_result_dict(n=20, base_energy=-1.0, seed=1)
    dynamic = _make_result_dict(n=20, base_energy=-1.0, offset=0.01, seed=2)
    e_vacuum = -1.0

    result = run_mode_comparison(fixed, dynamic, e_vacuum=e_vacuum)

    assert isinstance(result, ModeComparisonResult)
    assert isinstance(result.delta_corr_pol, DeltaCorrPolResult)
    assert isinstance(result.energy_phases, dict)
    assert "fixed" in result.energy_phases
    assert "dynamic" in result.energy_phases
    assert isinstance(result.qpe_hf_consistency, QPEHFConsistencyResult)
    assert isinstance(result.equilibration, EquilibrationResult | type(None))
    assert isinstance(result.energy_distributions, dict)
    assert "fixed" in result.energy_distributions
    assert "dynamic" in result.energy_distributions
    assert set(result.energy_distributions["fixed"].keys()) == {
        "mean",
        "std",
        "min",
        "max",
        "n_evals",
    }
    assert isinstance(result.trotter_bias_mha, float)


def test_run_mode_comparison_missing_keys():
    """result dict missing required keys raises ValueError."""
    bad_result = {"quantum_energies": np.array([1.0, 2.0])}  # no hf_energies
    fixed = _make_result_dict()
    with pytest.raises(ValueError, match="missing required keys"):
        run_mode_comparison(bad_result, fixed)


def test_run_mode_comparison_with_hf_corrected():
    """Optional hf_corrected result adds its mode to energy_distributions."""
    fixed = _make_result_dict(n=20, base_energy=-1.0, seed=1)
    dynamic = _make_result_dict(n=20, base_energy=-1.0, offset=0.01, seed=2)
    hf_corrected = _make_result_dict(n=15, base_energy=-1.0, offset=-0.005, seed=3)

    result = run_mode_comparison(fixed, dynamic, hf_corrected, e_vacuum=-1.0)

    assert "hf_corrected" in result.energy_distributions
    assert result.energy_distributions["hf_corrected"]["n_evals"] == 15
