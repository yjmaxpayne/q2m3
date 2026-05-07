# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Unit tests for q2m3.molecule module.

Tests verify MoleculeConfig dataclass behavior, property calculations,
and validation logic after promotion from examples/mc_solvation/config.py.
"""

import numpy as np
import pytest

from q2m3.molecule import MoleculeConfig


def test_molecule_config_basic():
    mol = MoleculeConfig(
        name="H2",
        symbols=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        charge=0,
        active_electrons=2,
        active_orbitals=2,
    )
    assert mol.name == "H2"
    assert mol.n_atoms == 2
    assert mol.basis == "sto-3g"


def test_molecule_config_coords_array():
    mol = MoleculeConfig(
        name="H2",
        symbols=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        charge=0,
        active_electrons=2,
        active_orbitals=2,
    )
    arr = mol.coords_array
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 3)


def test_molecule_config_center():
    mol = MoleculeConfig(
        name="H2",
        symbols=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        charge=0,
        active_electrons=2,
        active_orbitals=2,
    )
    center = mol.center
    assert abs(center[2] - 0.37) < 1e-6


def test_molecule_config_validate_ok():
    mol = MoleculeConfig(
        name="H2O",
        symbols=["O", "H", "H"],
        coords=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.9, 0.0, 0.0]],
        charge=0,
        active_electrons=4,
        active_orbitals=4,
    )
    mol.validate()  # Should not raise


def test_molecule_config_validate_symbol_mismatch():
    mol = MoleculeConfig(
        name="bad",
        symbols=["H", "H"],
        coords=[[0.0, 0.0, 0.0]],  # 1 coord but 2 symbols
        charge=0,
        active_electrons=2,
        active_orbitals=2,
    )
    with pytest.raises(ValueError, match="Number of symbols"):
        mol.validate()


def test_molecule_config_validate_bad_coord_dims():
    mol = MoleculeConfig(
        name="bad",
        symbols=["H"],
        coords=[[0.0, 0.0]],  # 2D, should be 3D
        charge=0,
        active_electrons=1,
        active_orbitals=1,
    )
    with pytest.raises(ValueError, match="3 components"):
        mol.validate()
