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
