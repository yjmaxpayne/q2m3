# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Unit tests for mc_solvation.energy module.

Tests verify HF and QM/MM energy calculations using PySCF.
Note: These tests involve actual quantum chemistry calculations and may be slow.
"""

import numpy as np
import pytest

from examples.mc_solvation.constants import HARTREE_TO_KCAL_MOL
from examples.mc_solvation.energy import (
    _compute_hf_energy_solvated_impl,
    _compute_hf_energy_vacuum_impl,
    compute_hf_energy_solvated,
    compute_hf_energy_vacuum,
    compute_mm_correction,
    compute_total_energy,
    create_energy_callback,
)
from examples.mc_solvation.solvent import (
    TIP3P_WATER,
    SolventMolecule,
    initialize_solvent_ring,
)


class TestHFEnergyVacuum:
    """Tests for vacuum HF energy computation."""

    def test_h2_vacuum_energy(self, h2_molecule_config):
        """H2 vacuum HF energy should be approximately -1.117 Ha."""
        energy = compute_hf_energy_vacuum(h2_molecule_config)
        # Expected: ~-1.117 Ha for H2 at STO-3G
        assert -1.2 < energy < -1.0

    def test_h3op_vacuum_energy(self, h3op_molecule_config):
        """H3O+ vacuum HF energy should be approximately -75.3 Ha."""
        energy = compute_hf_energy_vacuum(h3op_molecule_config)
        # Expected: ~-75.3 Ha for H3O+ at STO-3G
        assert -76.0 < energy < -75.0

    def test_energy_is_float(self, h2_molecule_config):
        """Energy should be returned as Python float."""
        energy = compute_hf_energy_vacuum(h2_molecule_config)
        assert isinstance(energy, float)


class TestHFEnergySolvated:
    """Tests for solvated HF energy computation."""

    def test_solvated_energy_with_waters(self, h2_molecule_config):
        """Solvated HF energy should differ from vacuum."""
        # Create a few water molecules around H2
        waters = initialize_solvent_ring(
            model=TIP3P_WATER,
            n_molecules=2,
            center=np.array([0.0, 0.0, 0.37]),  # H2 center
            radius=4.0,
            random_seed=42,
        )

        e_vacuum = compute_hf_energy_vacuum(h2_molecule_config)
        e_solvated = compute_hf_energy_solvated(h2_molecule_config, waters)

        # Energy should change with MM embedding
        assert e_vacuum != e_solvated
        # Change should be small (order of mHa)
        diff = abs(e_solvated - e_vacuum) * HARTREE_TO_KCAL_MOL
        assert diff < 50  # Less than 50 kcal/mol change

    def test_solvated_equals_vacuum_without_solvent(self, h2_molecule_config):
        """Solvated energy with empty solvent list should equal vacuum."""
        e_vacuum = compute_hf_energy_vacuum(h2_molecule_config)
        e_solvated = compute_hf_energy_solvated(h2_molecule_config, [])
        assert abs(e_vacuum - e_solvated) < 1e-10

    def test_h3op_solvated_energy(self, h3op_molecule_config):
        """H3O+ solvated energy should show ion-dipole interaction."""
        waters = initialize_solvent_ring(
            model=TIP3P_WATER,
            n_molecules=2,
            center=h3op_molecule_config.center,
            radius=3.5,
            random_seed=42,
        )

        e_vacuum = compute_hf_energy_vacuum(h3op_molecule_config)
        e_solvated = compute_hf_energy_solvated(h3op_molecule_config, waters)

        # Cation should have stronger interaction with water
        # (typically stabilizing, so e_solvated < e_vacuum)
        assert e_solvated != e_vacuum


class TestTotalEnergy:
    """Tests for total QM/MM energy computation."""

    def test_total_energy_h2_with_waters(self, h2_molecule_config):
        """Total energy should include both QM and MM contributions."""
        waters = initialize_solvent_ring(
            model=TIP3P_WATER,
            n_molecules=3,
            center=np.array([0.0, 0.0, 0.37]),
            radius=4.0,
            random_seed=42,
        )

        e_total = compute_total_energy(h2_molecule_config, waters)

        # Energy should be reasonable
        assert isinstance(e_total, float)
        # H2 + water environment should be around -1 to -2 Ha
        assert -3.0 < e_total < 0.0


class TestMMCorrection:
    """Tests for MM correction calculation."""

    def test_mm_correction_sign(self, h2_molecule_config):
        """MM correction should be non-zero with solvent."""
        waters = initialize_solvent_ring(
            model=TIP3P_WATER,
            n_molecules=2,
            center=np.array([0.0, 0.0, 0.37]),
            radius=4.0,
            random_seed=42,
        )

        e_vacuum = compute_hf_energy_vacuum(h2_molecule_config)
        correction = compute_mm_correction(h2_molecule_config, waters, e_vacuum)

        # Correction should be non-zero
        assert correction != 0.0

    def test_mm_correction_zero_without_solvent(self, h2_molecule_config):
        """MM correction should be zero without solvent."""
        e_vacuum = compute_hf_energy_vacuum(h2_molecule_config)
        correction = compute_mm_correction(h2_molecule_config, [], e_vacuum)
        assert abs(correction) < 1e-10

    def test_mm_correction_magnitude(self, h3op_molecule_config):
        """H3O+ should have larger MM correction due to ion-dipole."""
        waters = initialize_solvent_ring(
            model=TIP3P_WATER,
            n_molecules=2,
            center=h3op_molecule_config.center,
            radius=3.5,
            random_seed=42,
        )

        e_vacuum = compute_hf_energy_vacuum(h3op_molecule_config)
        correction = compute_mm_correction(h3op_molecule_config, waters, e_vacuum)

        # Correction should be meaningful
        correction_kcal = abs(correction) * HARTREE_TO_KCAL_MOL
        # Ion-dipole interaction typically several kcal/mol
        assert 0.01 < correction_kcal < 100


class TestCallbackImplementations:
    """Tests for pure_callback compatible implementations."""

    def test_vacuum_impl_matches_high_level(self, h2_molecule_config):
        """Callback impl should match high-level function."""
        e_high_level = compute_hf_energy_vacuum(h2_molecule_config)

        coords_flat = np.array(h2_molecule_config.coords).flatten()
        e_impl = _compute_hf_energy_vacuum_impl(
            h2_molecule_config.symbols,
            coords_flat,
            h2_molecule_config.charge,
            h2_molecule_config.basis,
        )

        assert abs(e_high_level - e_impl) < 1e-10

    def test_solvated_impl_matches_high_level(self, h2_molecule_config):
        """Callback impl should match high-level function with solvent."""
        waters = initialize_solvent_ring(
            model=TIP3P_WATER,
            n_molecules=2,
            center=np.array([0.0, 0.0, 0.37]),
            radius=4.0,
            random_seed=42,
        )

        e_high_level = compute_hf_energy_solvated(h2_molecule_config, waters)

        # Prepare flat arrays for impl
        coords_flat = np.array(h2_molecule_config.coords).flatten()
        from examples.mc_solvation.solvent import get_mm_embedding_data

        mm_coords, mm_charges = get_mm_embedding_data(waters)

        e_impl = _compute_hf_energy_solvated_impl(
            h2_molecule_config.symbols,
            coords_flat,
            h2_molecule_config.charge,
            h2_molecule_config.basis,
            mm_coords.flatten(),
            mm_charges,
        )

        assert abs(e_high_level - e_impl) < 1e-10


class TestEnergyCallback:
    """Tests for energy callback factory."""

    def test_create_energy_callback(self, solvation_config_minimal):
        """Should create a working energy callback."""
        callback = create_energy_callback(solvation_config_minimal)
        assert callable(callback)

    def test_callback_returns_energy(self, solvation_config_minimal):
        """Callback should return energy value."""
        callback = create_energy_callback(solvation_config_minimal)

        # Prepare inputs
        qm_coords_flat = np.array(solvation_config_minimal.molecule.coords).flatten()
        solvent_states = np.array(
            [
                [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        energy = callback(qm_coords_flat, solvent_states)

        assert isinstance(energy, (float, np.floating))
        assert not np.isnan(energy)
        assert not np.isinf(energy)
