# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Tests for EFTQC resource estimation module (core/resource_estimation.py).

Tests the high-level structured API built on top of
PySCFPennyLaneConverter.estimate_qpe_resources().
"""

import dataclasses

import numpy as np
import pytest

from q2m3.core.resource_estimation import (
    EFTQCResources,
    EmbeddingDiagnostics,
    ResourceComparisonResult,
    compare_vacuum_solvated,
    derive_t_resources,
    estimate_eftqc_runtime,
    estimate_resources,
)

# ------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------

H2_SYMBOLS = ["H", "H"]
H2_COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

# Single TIP3P water at ~3 Å from H2
MM_CHARGES_1W = np.array([-0.834, 0.417, 0.417])
MM_COORDS_1W = np.array([[3.0, 0.0, 0.0], [3.5, 0.8, 0.0], [3.5, -0.8, 0.0]])


# ------------------------------------------------------------------
# estimate_resources() tests
# ------------------------------------------------------------------


def test_estimate_resources_returns_dataclass():
    """estimate_resources() returns a frozen EFTQCResources dataclass."""
    result = estimate_resources(H2_SYMBOLS, H2_COORDS)

    assert isinstance(result, EFTQCResources)
    assert dataclasses.is_dataclass(result)
    # Frozen dataclass: assignment should raise
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        result.logical_qubits = 0  # type: ignore[misc]


def test_estimate_resources_h2_vacuum():
    """H2 vacuum returns a structurally valid EFTQCResources."""
    result = estimate_resources(H2_SYMBOLS, H2_COORDS)

    assert result.hamiltonian_1norm > 0
    assert result.logical_qubits > 0
    assert result.toffoli_gates > 0
    assert result.target_error == pytest.approx(0.0016)
    assert result.n_system_qubits > 0
    assert result.basis == "sto-3g"
    assert result.n_mm_charges == 0
    # n_terms is None because DoubleFactorization doesn't expose term count
    assert result.n_terms is None


def test_estimate_resources_n_system_qubits_is_twice_orbitals():
    """n_system_qubits = n_orbitals * 2 (Jordan-Wigner encoding)."""
    from q2m3.interfaces import PySCFPennyLaneConverter

    converter = PySCFPennyLaneConverter()
    raw = converter.estimate_qpe_resources(H2_SYMBOLS, H2_COORDS)

    result = estimate_resources(H2_SYMBOLS, H2_COORDS)

    assert result.n_system_qubits == raw["n_orbitals"] * 2


def test_estimate_resources_with_mm_charges():
    """estimate_resources() with MM charges sets n_mm_charges correctly."""
    result = estimate_resources(
        H2_SYMBOLS,
        H2_COORDS,
        mm_charges=MM_CHARGES_1W,
        mm_coords=MM_COORDS_1W,
    )

    assert result.n_mm_charges == 3
    assert result.hamiltonian_1norm > 0
    assert result.embedding_mode == "full_oneelectron"
    assert isinstance(result.embedding_diagnostics, EmbeddingDiagnostics)
    assert result.embedding_diagnostics.delta_h_hermitian_max_abs < 1e-10


def test_estimate_resources_accepts_diagonal_embedding_mode():
    """estimate_resources() forwards diagonal embedding mode to the converter."""
    result = estimate_resources(
        H2_SYMBOLS,
        H2_COORDS,
        active_electrons=2,
        active_orbitals=2,
        mm_charges=np.array([0.25]),
        mm_coords=np.array([[2.2, 0.7, 0.3]]),
        embedding_mode="diagonal",
    )

    assert result.embedding_mode == "diagonal"
    assert result.embedding_diagnostics is not None
    assert result.embedding_diagnostics.active_indices == (0, 1)
    assert result.embedding_diagnostics.delta_h_offdiag_fro > 0.0


def test_h2_one_water_diagonal_and_full_modes_return_finite_resources():
    """H2 + one TIP3P water returns finite resources in both MM modes."""
    diagonal = estimate_resources(
        H2_SYMBOLS,
        H2_COORDS,
        active_electrons=2,
        active_orbitals=2,
        mm_charges=MM_CHARGES_1W,
        mm_coords=MM_COORDS_1W,
        embedding_mode="diagonal",
    )
    full = estimate_resources(
        H2_SYMBOLS,
        H2_COORDS,
        active_electrons=2,
        active_orbitals=2,
        mm_charges=MM_CHARGES_1W,
        mm_coords=MM_COORDS_1W,
        embedding_mode="full_oneelectron",
    )

    for result in (diagonal, full):
        assert np.isfinite(result.hamiltonian_1norm)
        assert result.logical_qubits > 0
        assert result.toffoli_gates > 0
        assert result.embedding_diagnostics is not None
        assert result.embedding_diagnostics.delta_h_hermitian_max_abs < 1e-10

    assert diagonal.embedding_mode == "diagonal"
    assert full.embedding_mode == "full_oneelectron"


# ------------------------------------------------------------------
# compare_vacuum_solvated() tests
# ------------------------------------------------------------------


def test_compare_vacuum_solvated_h2():
    """compare_vacuum_solvated() returns a ResourceComparisonResult."""
    result = compare_vacuum_solvated(
        H2_SYMBOLS,
        H2_COORDS,
        mm_charges=MM_CHARGES_1W,
        mm_coords=MM_COORDS_1W,
    )

    assert isinstance(result, ResourceComparisonResult)
    assert isinstance(result.vacuum, EFTQCResources)
    assert isinstance(result.solvated, EFTQCResources)
    assert result.vacuum.n_mm_charges == 0
    assert result.solvated.n_mm_charges == 3
    assert result.vacuum.embedding_mode == "none"
    assert result.solvated.embedding_mode == "full_oneelectron"


def test_compare_vacuum_solvated_forwards_embedding_mode():
    """compare_vacuum_solvated() can compare diagonal resource rows."""
    result = compare_vacuum_solvated(
        H2_SYMBOLS,
        H2_COORDS,
        active_electrons=2,
        active_orbitals=2,
        mm_charges=np.array([0.25]),
        mm_coords=np.array([[2.2, 0.7, 0.3]]),
        embedding_mode="diagonal",
    )

    assert result.vacuum.embedding_mode == "none"
    assert result.solvated.embedding_mode == "diagonal"
    assert result.solvated.embedding_diagnostics is not None


def test_delta_lambda_positive():
    """Explicit full-oneelectron comparison uses the documented delta formula."""
    result = compare_vacuum_solvated(
        H2_SYMBOLS,
        H2_COORDS,
        mm_charges=MM_CHARGES_1W,
        mm_coords=MM_COORDS_1W,
        embedding_mode="full_oneelectron",
    )

    assert result.delta_lambda_percent != pytest.approx(0.0)
    assert result.solvated.embedding_mode == "full_oneelectron"

    expected = (
        (result.solvated.hamiltonian_1norm - result.vacuum.hamiltonian_1norm)
        / result.vacuum.hamiltonian_1norm
        * 100
    )
    assert result.delta_lambda_percent == pytest.approx(expected)


def test_delta_gates_percent_formula():
    """delta_gates_percent formula is (solvated - vacuum) / vacuum * 100."""
    result = compare_vacuum_solvated(
        H2_SYMBOLS,
        H2_COORDS,
        mm_charges=MM_CHARGES_1W,
        mm_coords=MM_COORDS_1W,
    )

    expected = (
        (result.solvated.toffoli_gates - result.vacuum.toffoli_gates)
        / result.vacuum.toffoli_gates
        * 100
    )
    assert result.delta_gates_percent == pytest.approx(expected)


# ------------------------------------------------------------------
# Active space support tests (Cycle 1)
# ------------------------------------------------------------------


# H3O+ in approximate Cs geometry (Angstrom). Charge +1, 10 electrons.
H3OP_SYMBOLS = ["O", "H", "H", "H"]
H3OP_COORDS = np.array(
    [
        [0.0, 0.0, 0.117],
        [0.93, 0.0, -0.292],
        [-0.465, 0.806, -0.292],
        [-0.465, -0.806, -0.292],
    ]
)


def test_active_space_reduces_qubit_count():
    """Active space (4e,4o) reduces qubit count vs full STO-3G (10e,8o)."""
    full = estimate_resources(H3OP_SYMBOLS, H3OP_COORDS, charge=1, basis="sto-3g")
    active = estimate_resources(
        H3OP_SYMBOLS,
        H3OP_COORDS,
        charge=1,
        basis="sto-3g",
        active_electrons=4,
        active_orbitals=4,
    )

    # Active (4 spatial orbitals) → 8 system qubits (JW); full (8 orbitals) → 16
    assert active.n_system_qubits == 8
    assert full.n_system_qubits == 16
    assert active.n_system_qubits < full.n_system_qubits


def test_active_space_reduces_toffoli_count():
    """Active space estimate uses fewer Toffoli gates than full space."""
    full = estimate_resources(H3OP_SYMBOLS, H3OP_COORDS, charge=1, basis="sto-3g")
    active = estimate_resources(
        H3OP_SYMBOLS,
        H3OP_COORDS,
        charge=1,
        basis="sto-3g",
        active_electrons=4,
        active_orbitals=4,
    )

    assert active.toffoli_gates < full.toffoli_gates


def test_active_space_with_mm_embedding():
    """Active space + MM embedding combines without raising."""
    result = estimate_resources(
        H2_SYMBOLS,
        H2_COORDS,
        active_electrons=2,
        active_orbitals=2,
        mm_charges=MM_CHARGES_1W,
        mm_coords=MM_COORDS_1W,
    )
    assert result.n_mm_charges == 3
    assert result.n_system_qubits == 4
    assert result.toffoli_gates > 0


def test_active_space_with_mm_embedding_for_truncated_h3op_space():
    """Active-space MM embedding works when the active space is smaller than full MO space."""
    result = estimate_resources(
        H3OP_SYMBOLS,
        H3OP_COORDS,
        charge=1,
        basis="sto-3g",
        active_electrons=4,
        active_orbitals=4,
        mm_charges=MM_CHARGES_1W,
        mm_coords=MM_COORDS_1W,
    )

    assert result.n_mm_charges == 3
    assert result.n_system_qubits == 8
    assert result.toffoli_gates > 0


# ------------------------------------------------------------------
# Derived resource helpers (Cycle 2)
# ------------------------------------------------------------------


def test_derive_t_resources_uses_seven_t_per_toffoli():
    """T count = 7 * Toffoli (standard fault-tolerant decomposition)."""
    derived = derive_t_resources(toffoli_gates=1000)
    assert derived["t_count"] == 7000
    # Conservative T-depth upper bound: assume sequential Toffoli execution
    assert derived["t_depth"] >= derived["toffoli_depth"]
    assert derived["toffoli_depth"] == 1000


def test_estimate_eftqc_runtime_takes_total_toffoli_count():
    """BUG-QRE-RUNTIME-001: runtime must NOT re-multiply the DF total by ceil(lambda/eps).

    PennyLane ``DoubleFactorization.gates`` already equals
    ``estimation_cost(lambda, eps) * unitary_cost`` (whole-QPE Toffoli count).
    Passing it together with a separate iteration count double-counted the
    lambda/eps factor (x3821 for H3O+ (4e,4o)). With the physical tick model
    (Babbush 2021 Eq. 6, single CCZ factory, p=1e-3) the H3O+ row is ~0.23 h,
    not the 6.9 h the survey used to report.
    """
    # H3O+ (4e,4o) STO-3G, eps = 1.6 mHa: DF total Toffoli and logical qubits
    runtime = estimate_eftqc_runtime(toffoli_gates=6_511_100, logical_qubits=131)

    assert runtime["runtime_hours"] < 1.0
    assert runtime["code_distance"] == 23
    assert runtime["tick_microseconds"] == pytest.approx(126.5, rel=0.05)
    # Contract: runtime = N_Toffoli(total) x tick, nothing else
    assert runtime["runtime_seconds"] == pytest.approx(
        6_511_100 * runtime["tick_microseconds"] * 1e-6
    )


def test_estimate_eftqc_runtime_reproduces_lee2021_femoco():
    """Lee et al. PRX Quantum 2, 030305 (2021) FeMoco regression.

    1908 logical qubits, 6.7e9 Toffoli, p_phys=1e-3, 4 CCZ factories ->
    paper: d=31, ~40 us/Toffoli (25 kHz), 3-3.5 days, ~4e6 physical qubits.
    """
    runtime = estimate_eftqc_runtime(
        toffoli_gates=6_700_000_000,
        logical_qubits=1908,
        p_phys=1e-3,
        n_factories=4,
    )

    assert runtime["code_distance"] == 31
    assert runtime["tick_microseconds"] == pytest.approx(40.0, rel=0.2)
    assert 3.0 <= runtime["runtime_days"] <= 3.5
    assert 3.5e6 <= runtime["physical_qubits"] <= 5.0e6
    assert runtime["regime"] == "factory"


def test_estimate_eftqc_runtime_tick_is_floored_by_reaction_time():
    """Many factories cannot beat the classical reaction limit (~10 us)."""
    runtime = estimate_eftqc_runtime(
        toffoli_gates=10_000,
        logical_qubits=10,
        n_factories=64,
        t_react_microseconds=10.0,
    )

    assert runtime["regime"] == "reaction"
    assert runtime["tick_microseconds"] == pytest.approx(10.0)


def test_estimate_eftqc_runtime_scales_linearly_with_cycle():
    """In the factory-limited regime runtime scales linearly with t_cycle.

    Code distance depends only on (p_phys, logical_qubits, toffoli_gates), so
    changing t_cycle changes the tick but not d.
    """
    fast = estimate_eftqc_runtime(
        toffoli_gates=6_511_100, logical_qubits=131, t_cycle_microseconds=1.0
    )
    slow = estimate_eftqc_runtime(
        toffoli_gates=6_511_100, logical_qubits=131, t_cycle_microseconds=10.0
    )

    assert slow["code_distance"] == fast["code_distance"]
    assert slow["runtime_seconds"] == pytest.approx(fast["runtime_seconds"] * 10)


def test_estimate_resources_exposes_walk_operator_calls():
    """walk_operator_calls = DF.estimation_cost = ceil(pi*lambda/(2*eps)), informational only."""
    result = estimate_resources(symbols=H2_SYMBOLS, coords=H2_COORDS, target_error=0.0016)

    expected = int(np.ceil(np.pi * result.hamiltonian_1norm / (2 * 0.0016)))
    assert result.walk_operator_calls == expected
