# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Tests for fixed-MO one-electron embedding helpers."""

from __future__ import annotations

import numpy as np
import pytest

from q2m3.interfaces.fixed_mo_embedding import build_fixed_mo_embedding_integrals


@pytest.fixture(scope="module")
def h2_tip3p_embedding():
    """Build a small H2 + one TIP3P water embedding fixture."""
    symbols = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    mm_charges = np.array([-0.834, 0.417, 0.417])
    mm_coords = np.array(
        [
            [3.0, 0.0, 0.0],
            [3.5, 0.8, 0.0],
            [3.5, -0.8, 0.0],
        ]
    )

    return build_fixed_mo_embedding_integrals(
        symbols,
        coords,
        mm_charges=mm_charges,
        mm_coords=mm_coords,
        active_electrons=2,
        active_orbitals=2,
    )


def test_h2_tip3p_returns_hermitian_active_delta(h2_tip3p_embedding):
    """H2 + one TIP3P water returns a Hermitian active-space MM perturbation."""
    result = h2_tip3p_embedding

    assert result.one_electron_vacuum.shape == (2, 2)
    assert result.two_electron.shape == (2, 2, 2, 2)
    assert result.delta_h_active.shape == (2, 2)
    np.testing.assert_allclose(result.delta_h_active, result.delta_h_active.T, atol=1e-10)
    assert result.diagnostics.delta_h_hermitian_max_abs < 1e-10


def test_active_delta_splits_into_diag_and_offdiag(h2_tip3p_embedding):
    """The active-space MM perturbation is exactly diag plus offdiag parts."""
    result = h2_tip3p_embedding

    np.testing.assert_allclose(
        result.delta_h_active,
        result.delta_h_diag + result.delta_h_offdiag,
        atol=1e-12,
    )


def test_offdiag_delta_has_zero_diagonal(h2_tip3p_embedding):
    """The off-diagonal diagnostic matrix excludes diagonal terms."""
    result = h2_tip3p_embedding

    np.testing.assert_allclose(np.diag(result.delta_h_offdiag), np.zeros(2), atol=1e-12)


def test_active_indices_and_frozen_core_count_match_formula(h2_tip3p_embedding):
    """Active-space indices follow the closed-shell frozen-core convention."""
    result = h2_tip3p_embedding

    assert result.active_indices == (0, 1)
    assert result.n_core_orbitals == 0


def test_invalid_active_electrons_parity_raises():
    """Odd frozen-core electron counts are rejected."""
    symbols = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    mm_charges = np.array([-0.834, 0.417, 0.417])
    mm_coords = np.array(
        [
            [3.0, 0.0, 0.0],
            [3.5, 0.8, 0.0],
            [3.5, -0.8, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="even"):
        build_fixed_mo_embedding_integrals(
            symbols,
            coords,
            mm_charges=mm_charges,
            mm_coords=mm_coords,
            active_electrons=1,
            active_orbitals=1,
        )


def test_invalid_qm_coords_shape_raises():
    """QM coordinates must match the number of symbols."""
    with pytest.raises(ValueError, match="coords"):
        build_fixed_mo_embedding_integrals(
            ["H", "H"],
            np.array([[0.0, 0.0]]),
            mm_charges=np.array([0.1]),
            mm_coords=np.array([[3.0, 0.0, 0.0]]),
        )


def test_invalid_mm_shape_raises():
    """MM coordinates and charges must have matching leading dimensions."""
    with pytest.raises(ValueError, match="must match"):
        build_fixed_mo_embedding_integrals(
            ["H"],
            np.array([[0.0, 0.0, 0.0]]),
            mm_charges=np.array([0.1, -0.1]),
            mm_coords=np.array([[3.0, 0.0, 0.0]]),
        )


def test_partial_active_space_request_raises():
    """Active-space electron and orbital counts must be specified together."""
    with pytest.raises(ValueError, match="provided together"):
        build_fixed_mo_embedding_integrals(
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
            mm_charges=np.array([0.1]),
            mm_coords=np.array([[3.0, 0.0, 0.0]]),
            active_electrons=2,
        )


def test_top_level_public_export():
    """The public fixed-MO helper is available from the package root."""
    from q2m3 import build_fixed_mo_embedding_integrals as top_level_helper

    assert top_level_helper is build_fixed_mo_embedding_integrals
