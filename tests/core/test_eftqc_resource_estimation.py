# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
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


def test_delta_lambda_positive():
    """MM charges generally increase Hamiltonian 1-norm (delta_lambda_percent != 0)."""
    result = compare_vacuum_solvated(
        H2_SYMBOLS,
        H2_COORDS,
        mm_charges=MM_CHARGES_1W,
        mm_coords=MM_COORDS_1W,
    )

    # delta_lambda_percent should be non-zero when MM charges are present
    assert result.delta_lambda_percent != pytest.approx(0.0)
    # Verify the formula is correct
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


def test_estimate_eftqc_runtime_uses_cycle_time():
    """Runtime = qpe_iterations * toffoli_gates * cycle_time."""
    runtime = estimate_eftqc_runtime(
        qpe_iterations=10,
        toffoli_gates=1_000_000,
        toffoli_cycle_microseconds=1.0,
    )
    # 10 iters * 1e6 Toffoli * 1 us = 1e7 us = 10 s
    assert runtime["runtime_seconds"] == pytest.approx(10.0)
    assert runtime["runtime_hours"] == pytest.approx(10.0 / 3600)


def test_estimate_eftqc_runtime_scales_linearly_with_cycle():
    """Runtime scales linearly with toffoli_cycle_microseconds."""
    fast = estimate_eftqc_runtime(
        qpe_iterations=10, toffoli_gates=1000, toffoli_cycle_microseconds=0.1
    )
    slow = estimate_eftqc_runtime(
        qpe_iterations=10, toffoli_gates=1000, toffoli_cycle_microseconds=10.0
    )
    assert slow["runtime_seconds"] == pytest.approx(fast["runtime_seconds"] * 100)
