# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Shared pytest fixtures for mc_solvation module tests.
"""

import numpy as np
import pytest

from examples.mc_solvation import (
    SPC_E_WATER,
    TIP3P_WATER,
    MoleculeConfig,
    QPEConfig,
    SolvationConfig,
    SolventMolecule,
    initialize_solvent_ring,
)

# ============================================================================
# Molecule Configuration Fixtures
# ============================================================================


@pytest.fixture
def h2_molecule_config():
    """H2 molecule configuration for testing."""
    return MoleculeConfig(
        name="H2",
        symbols=["H", "H"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.74],
        ],
        charge=0,
        active_electrons=2,
        active_orbitals=2,
        basis="sto-3g",
    )


@pytest.fixture
def h3op_molecule_config():
    """H3O+ molecule configuration for testing."""
    return MoleculeConfig(
        name="H3O+",
        symbols=["O", "H", "H", "H"],
        coords=[
            [0.0000, 0.0000, 0.1173],
            [0.0000, 0.9572, -0.4692],
            [0.8286, -0.4786, -0.4692],
            [-0.8286, -0.4786, -0.4692],
        ],
        charge=1,
        active_electrons=4,
        active_orbitals=4,
        basis="sto-3g",
    )


# ============================================================================
# QPE Configuration Fixtures
# ============================================================================


@pytest.fixture
def qpe_config_minimal():
    """Minimal QPE configuration for fast unit tests."""
    return QPEConfig(
        n_estimation_wires=3,
        n_trotter_steps=2,
        n_shots=5,
        qpe_interval=5,
        target_resolution=0.01,
        energy_range=0.5,
        use_catalyst=False,  # Disable for unit tests
    )


@pytest.fixture
def qpe_config_standard():
    """Standard QPE configuration for integration tests."""
    return QPEConfig(
        n_estimation_wires=4,
        n_trotter_steps=10,
        n_shots=50,
        qpe_interval=10,
        target_resolution=0.003,
        energy_range=0.2,
        use_catalyst=True,
    )


# ============================================================================
# Solvation Configuration Fixtures
# ============================================================================


@pytest.fixture
def solvation_config_minimal(h2_molecule_config, qpe_config_minimal):
    """Minimal solvation configuration for fast tests."""
    return SolvationConfig(
        molecule=h2_molecule_config,
        qpe_config=qpe_config_minimal,
        qpe_mode="vacuum_correction",
        n_waters=3,
        n_mc_steps=10,
        temperature=300.0,
        translation_step=0.3,
        rotation_step=0.2618,
        initial_water_distance=4.0,
        random_seed=42,
        verbose=False,
    )


@pytest.fixture
def solvation_config_h3op(h3op_molecule_config, qpe_config_minimal):
    """H3O+ solvation configuration for tests."""
    return SolvationConfig(
        molecule=h3op_molecule_config,
        qpe_config=qpe_config_minimal,
        qpe_mode="vacuum_correction",
        n_waters=5,
        n_mc_steps=10,
        temperature=300.0,
        translation_step=0.3,
        rotation_step=0.2618,
        initial_water_distance=3.5,
        random_seed=42,
        verbose=False,
    )


# ============================================================================
# Solvent Model Fixtures
# ============================================================================


@pytest.fixture
def tip3p_model():
    """TIP3P water model."""
    return TIP3P_WATER


@pytest.fixture
def spce_model():
    """SPC/E water model."""
    return SPC_E_WATER


@pytest.fixture
def water_molecules_3():
    """3 water molecules in a ring around origin."""
    return initialize_solvent_ring(
        model=TIP3P_WATER,
        n_molecules=3,
        center=np.array([0.0, 0.0, 0.0]),
        radius=4.0,
    )


@pytest.fixture
def water_molecules_5():
    """5 water molecules in a ring around origin."""
    return initialize_solvent_ring(
        model=TIP3P_WATER,
        n_molecules=5,
        center=np.array([0.0, 0.0, 0.0]),
        radius=3.5,
    )


# ============================================================================
# State Array Fixtures
# ============================================================================


@pytest.fixture
def simple_water_state():
    """Simple water molecule state array for testing."""
    # Single TIP3P water: O at origin, H1 at (0.9572, 0, 0), H2 at (-0.2399, 0.9266, 0)
    return np.array(
        [
            [0.0, 0.0, 0.0],  # O position
            [0.9572, 0.0, 0.0],  # H1 position
            [-0.2399, 0.9266, 0.0],  # H2 position
        ]
    )


# ============================================================================
# PySCF Molecule Fixtures
# ============================================================================


@pytest.fixture
def pyscf_h2_mol():
    """PySCF molecule object for H2."""
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = [
        ["H", [0.0, 0.0, 0.0]],
        ["H", [0.0, 0.0, 0.74]],
    ]
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0
    mol.unit = "angstrom"
    mol.build()
    return mol


@pytest.fixture
def pyscf_h3op_mol():
    """PySCF molecule object for H3O+."""
    from pyscf import gto

    mol = gto.Mole()
    mol.atom = [
        ["O", [0.0000, 0.0000, 0.1173]],
        ["H", [0.0000, 0.9572, -0.4692]],
        ["H", [0.8286, -0.4786, -0.4692]],
        ["H", [-0.8286, -0.4786, -0.4692]],
    ]
    mol.basis = "sto-3g"
    mol.charge = 1
    mol.spin = 0
    mol.unit = "angstrom"
    mol.build()
    return mol
