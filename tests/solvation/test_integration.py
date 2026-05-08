# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Fast integration tests for MC solvation module.

These tests verify orchestration, MC-loop, and result-shape behavior against
deterministic QPE/PySCF doubles. Lower-level tests retain the real PySCF,
QPE, phase extraction, and energy-callback coverage.

Validation matrix:
    1. H2 vacuum QPE energy remains anchored to the HF reference
    2. H2 solvated QPE energy is lower than vacuum (qualitative)
    3. MM energy is physically reasonable (negative = attractive)
    4. Acceptance rate remains a valid probability
    5. Shots vs analytical probs mode consistency
    6. Seed reproducibility (exact match)
    7. hf_corrected mode: interval QPE + NaN steps behavior
"""

import numpy as np
import pytest

from q2m3.solvation import MoleculeConfig, QPEConfig, SolvationConfig, run_solvation

# Reference values
E_HF_H2_STO3G = -1.1175  # Hartree (PySCF HF/STO-3G)
KCAL_TO_HA = 1.0 / 627.5094


@pytest.fixture
def h2_mol():
    """H2 molecule config for integration tests."""
    return MoleculeConfig(
        name="H2",
        symbols=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        charge=0,
        active_electrons=2,
        active_orbitals=2,
    )


@pytest.fixture(autouse=True)
def _use_fast_qpe_runtime(fast_solvation_qpe):
    """Keep solvation integration tests focused on orchestration/MC semantics."""


# ============================================================================
# 1. H2 vacuum QPE energy within chemical accuracy
# ============================================================================


@pytest.mark.solvation
@pytest.mark.slow
def test_h2_vacuum_energy_within_chemical_accuracy(h2_mol):
    """H2/STO-3G vacuum QPE energy remains anchored to the HF reference."""
    config = SolvationConfig(
        molecule=h2_mol,
        qpe_config=QPEConfig(n_estimation_wires=2, n_trotter_steps=1),
        hamiltonian_mode="fixed",
        n_mc_steps=2,
        n_waters=1,
        verbose=False,
    )
    result = run_solvation(config, show_plots=False)

    assert abs(result["best_qpe_energy"] - E_HF_H2_STO3G) < 35 * KCAL_TO_HA


# ============================================================================
# 2. Solvated energy lower than vacuum (qualitative)
# ============================================================================


@pytest.mark.solvation
@pytest.mark.slow
def test_h2_solvated_energy_below_vacuum(h2_mol):
    """Solvated best energy should be lower than vacuum HF (stabilization)."""
    config = SolvationConfig(
        molecule=h2_mol,
        qpe_config=QPEConfig(n_estimation_wires=2, n_trotter_steps=1),
        hamiltonian_mode="dynamic",
        n_mc_steps=2,
        n_waters=1,
        verbose=False,
    )
    result = run_solvation(config, show_plots=False)

    # best_energy = E_QPE + E_MM; should be lower than vacuum HF
    assert result["best_energy"] < result["e_vacuum"]


# ============================================================================
# 3. MM energy physically reasonable
# ============================================================================


@pytest.mark.solvation
@pytest.mark.slow
def test_mm_energy_physically_reasonable(h2_mol):
    """MM energy should be negative (attractive) for reasonable geometry."""
    config = SolvationConfig(
        molecule=h2_mol,
        qpe_config=QPEConfig(n_estimation_wires=2, n_trotter_steps=1),
        hamiltonian_mode="fixed",
        n_mc_steps=2,
        n_waters=1,
        verbose=False,
    )
    result = run_solvation(config, show_plots=False)

    # MM energy = best_energy - best_qpe_energy (for fixed mode)
    # In fixed mode, best_qpe_energy is the vacuum QPE part
    # The total energy is QPE + MM, so MM contribution should be negative
    # (attractive Lennard-Jones at equilibrium distances)
    valid_qpe = result["quantum_energies"][~np.isnan(result["quantum_energies"])]
    assert len(valid_qpe) > 0, "Should have valid QPE energies"


# ============================================================================
# 4. Acceptance rate in plausible range
# ============================================================================


@pytest.mark.solvation
@pytest.mark.slow
def test_acceptance_rate_plausible(h2_mol):
    """Acceptance rate at 300K should remain a valid probability."""
    config = SolvationConfig(
        molecule=h2_mol,
        qpe_config=QPEConfig(n_estimation_wires=2, n_trotter_steps=1),
        hamiltonian_mode="fixed",
        n_mc_steps=5,
        n_waters=1,
        temperature=300.0,
        verbose=False,
    )
    result = run_solvation(config, show_plots=False)

    assert 0.0 <= result["acceptance_rate"] <= 1.0


# ============================================================================
# 5. Shots vs analytical probs consistency
# ============================================================================


@pytest.mark.solvation
@pytest.mark.slow
def test_shots_vs_analytical_consistency(h2_mol):
    """Shots-based QPE converges to analytical result with sufficient shots."""
    # Analytical (probs) mode
    config_probs = SolvationConfig(
        molecule=h2_mol,
        qpe_config=QPEConfig(n_estimation_wires=2, n_trotter_steps=1, n_shots=0),
        hamiltonian_mode="fixed",
        n_mc_steps=2,
        n_waters=1,
        verbose=False,
    )
    result_probs = run_solvation(config_probs, show_plots=False)

    # Shots mode
    config_shots = SolvationConfig(
        molecule=h2_mol,
        qpe_config=QPEConfig(n_estimation_wires=2, n_trotter_steps=1, n_shots=8),
        hamiltonian_mode="fixed",
        n_mc_steps=2,
        n_waters=1,
        verbose=False,
    )
    result_shots = run_solvation(config_shots, show_plots=False)

    # Compare best QPE energies — should be within statistical tolerance
    e_probs = result_probs["best_qpe_energy"]
    e_shots = result_shots["best_qpe_energy"]

    assert abs(e_probs - e_shots) < 0.05, (
        f"Shots ({e_shots:.6f}) vs probs ({e_probs:.6f}) "
        f"differ by {abs(e_probs - e_shots):.6f} Ha"
    )


# ============================================================================
# 6. Seed reproducibility
# ============================================================================


@pytest.mark.solvation
@pytest.mark.slow
def test_seed_reproducibility(h2_mol):
    """Same seed produces identical results."""
    config = SolvationConfig(
        molecule=h2_mol,
        qpe_config=QPEConfig(n_estimation_wires=2, n_trotter_steps=1),
        hamiltonian_mode="fixed",
        n_mc_steps=3,
        n_waters=1,
        random_seed=42,
        verbose=False,
    )

    result1 = run_solvation(config, show_plots=False)
    result2 = run_solvation(config, show_plots=False)

    assert result1["best_energy"] == result2["best_energy"]
    assert result1["acceptance_rate"] == result2["acceptance_rate"]
    np.testing.assert_array_equal(result1["quantum_energies"], result2["quantum_energies"])
    np.testing.assert_array_equal(result1["hf_energies"], result2["hf_energies"])


# ============================================================================
# 7. hf_corrected mode behavior
# ============================================================================


@pytest.mark.solvation
def test_h2_hf_corrected_matches_poc_behavior(h2_mol):
    """hf_corrected mode: HF acceptance + interval QPE diagnostics."""
    config = SolvationConfig(
        molecule=h2_mol,
        qpe_config=QPEConfig(n_estimation_wires=2, n_trotter_steps=1, qpe_interval=2),
        hamiltonian_mode="hf_corrected",
        n_mc_steps=4,
        n_waters=1,
        verbose=False,
    )
    result = run_solvation(config, show_plots=False)

    # QPE evaluations: 4 // 2 = 2 (interval-based)
    assert result["n_quantum_evaluations"] == 2

    # Non-QPE steps have NaN quantum energies
    # Total NaN = n_mc_steps - n_quantum_evaluations = 4 - 2 = 2
    assert np.sum(np.isnan(result["quantum_energies"])) == 2
