"""Shared pytest fixtures for q2m3.solvation tests."""

import numpy as np
import pytest

from q2m3.solvation import TIP3P_WATER, MoleculeConfig, QPEConfig, SolvationConfig
from q2m3.solvation.solvent import initialize_solvent_ring

# ============================================================================
# Molecule Data Fixtures (pure data, no solvation dependency)
# ============================================================================


@pytest.fixture
def h2_molecule_data():
    """H2 molecule raw data for testing."""
    return {
        "name": "H2",
        "symbols": ["H", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        "charge": 0,
        "active_electrons": 2,
        "active_orbitals": 2,
        "basis": "sto-3g",
    }


@pytest.fixture
def h2_hf_energy():
    """Pre-computed H2/STO-3G HF energy."""
    return -1.1175  # Hartree


# ============================================================================
# Solvation Module Fixtures
# ============================================================================


@pytest.fixture
def h2_molecule_config():
    """H2 MoleculeConfig for solvation tests."""
    return MoleculeConfig(
        name="H2",
        symbols=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        charge=0,
        active_electrons=2,
        active_orbitals=2,
    )


@pytest.fixture
def solvation_config_minimal(h2_molecule_config):
    """Minimal SolvationConfig for fast tests."""
    return SolvationConfig(
        molecule=h2_molecule_config,
        qpe_config=QPEConfig(n_estimation_wires=3, n_trotter_steps=2),
        hamiltonian_mode="fixed",
        n_waters=3,
        n_mc_steps=10,
        verbose=False,
    )


@pytest.fixture
def water_molecules_3():
    """3 TIP3P water molecules in a ring."""
    return initialize_solvent_ring(
        model=TIP3P_WATER,
        n_molecules=3,
        center=np.array([0.0, 0.0, 0.0]),
        radius=4.0,
    )
