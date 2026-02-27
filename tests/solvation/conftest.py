"""Shared pytest fixtures for q2m3.solvation tests."""

import pytest

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
