# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Unit tests for mc_solvation.constants module.

Tests verify physical constant values and their relationships.
"""

import pytest

from examples.mc_solvation.constants import (  # Unit conversion; TIP3P parameters; Default parameters
    ANGSTROM_TO_BOHR,
    BOLTZMANN_CONSTANT,
    COULOMB_CONSTANT,
    DEFAULT_INITIAL_WATER_DISTANCE,
    DEFAULT_N_ESTIMATION_WIRES,
    DEFAULT_N_MC_STEPS,
    DEFAULT_N_QPE_SHOTS,
    DEFAULT_N_TROTTER_STEPS,
    DEFAULT_N_WATERS,
    DEFAULT_QPE_INTERVAL,
    DEFAULT_ROTATION_STEP,
    DEFAULT_TEMPERATURE,
    DEFAULT_TRANSLATION_STEP,
    HARTREE_TO_KCAL_MOL,
    KCAL_TO_HARTREE,
    TIP3P_EPSILON_OO,
    TIP3P_HOH_ANGLE,
    TIP3P_HYDROGEN_CHARGE,
    TIP3P_OH_BOND_LENGTH,
    TIP3P_OXYGEN_CHARGE,
    TIP3P_SIGMA_OO,
)


class TestUnitConversionConstants:
    """Tests for unit conversion constants."""

    def test_hartree_kcal_reciprocal(self):
        """Hartree-kcal/mol conversion should be reciprocal."""
        assert abs(HARTREE_TO_KCAL_MOL * KCAL_TO_HARTREE - 1.0) < 1e-10

    def test_hartree_to_kcal_reasonable_value(self):
        """Hartree to kcal/mol should be approximately 627.5."""
        # Well-known value: 1 Hartree ≈ 627.5 kcal/mol
        assert 627.0 < HARTREE_TO_KCAL_MOL < 628.0

    def test_angstrom_to_bohr_reasonable_value(self):
        """Angstrom to Bohr should be approximately 1.89."""
        # Well-known value: 1 Å ≈ 1.89 Bohr
        assert 1.88 < ANGSTROM_TO_BOHR < 1.90

    def test_boltzmann_constant_reasonable_value(self):
        """Boltzmann constant in Hartree/K should be approximately 3.17e-6."""
        # kB ≈ 3.166811 × 10^-6 Hartree/K
        assert 3.1e-6 < BOLTZMANN_CONSTANT < 3.2e-6

    def test_constants_are_floats(self):
        """All conversion constants should be Python floats."""
        assert isinstance(HARTREE_TO_KCAL_MOL, float)
        assert isinstance(KCAL_TO_HARTREE, float)
        assert isinstance(ANGSTROM_TO_BOHR, float)
        assert isinstance(BOLTZMANN_CONSTANT, float)


class TestTIP3PWaterParameters:
    """Tests for TIP3P water model parameters."""

    def test_oh_bond_length_reasonable(self):
        """O-H bond length should be approximately 0.96 Angstrom."""
        # Experimental: ~0.9572 Å
        assert 0.95 < TIP3P_OH_BOND_LENGTH < 0.97

    def test_hoh_angle_reasonable(self):
        """H-O-H angle should be approximately 104.5 degrees."""
        # Experimental: ~104.52°
        assert 104.0 < TIP3P_HOH_ANGLE < 105.0

    def test_charge_neutrality(self):
        """Water molecule should be charge neutral."""
        total_charge = TIP3P_OXYGEN_CHARGE + 2 * TIP3P_HYDROGEN_CHARGE
        assert abs(total_charge) < 1e-10

    def test_oxygen_negative(self):
        """Oxygen should have negative partial charge."""
        assert TIP3P_OXYGEN_CHARGE < 0

    def test_hydrogen_positive(self):
        """Hydrogen should have positive partial charge."""
        assert TIP3P_HYDROGEN_CHARGE > 0

    def test_sigma_oo_reasonable(self):
        """LJ sigma should be approximately 3.15 Angstrom."""
        assert 3.0 < TIP3P_SIGMA_OO < 3.3

    def test_epsilon_oo_reasonable(self):
        """LJ epsilon should be approximately 0.15 kcal/mol."""
        assert 0.1 < TIP3P_EPSILON_OO < 0.2

    def test_coulomb_constant_reasonable(self):
        """Coulomb constant should be approximately 332 kcal·Å/(mol·e²)."""
        assert 330.0 < COULOMB_CONSTANT < 334.0


class TestDefaultParameters:
    """Tests for default simulation parameters."""

    def test_default_n_waters_positive(self):
        """Default water count should be positive."""
        assert DEFAULT_N_WATERS > 0

    def test_default_n_mc_steps_positive(self):
        """Default MC steps should be positive."""
        assert DEFAULT_N_MC_STEPS > 0

    def test_default_temperature_room_temperature(self):
        """Default temperature should be room temperature (~300K)."""
        assert 295.0 < DEFAULT_TEMPERATURE < 305.0

    def test_default_translation_step_reasonable(self):
        """Translation step should be on the order of 0.1-1 Angstrom."""
        assert 0.1 < DEFAULT_TRANSLATION_STEP < 1.0

    def test_default_rotation_step_reasonable(self):
        """Rotation step should be between 0 and π/4 radians."""
        import math

        assert 0 < DEFAULT_ROTATION_STEP < math.pi / 4

    def test_default_water_distance_reasonable(self):
        """Initial water distance should be 2-6 Angstrom."""
        assert 2.0 < DEFAULT_INITIAL_WATER_DISTANCE < 6.0

    def test_default_qpe_interval_positive(self):
        """QPE interval should be positive."""
        assert DEFAULT_QPE_INTERVAL > 0

    def test_default_n_estimation_wires_positive(self):
        """Number of estimation wires should be positive."""
        assert DEFAULT_N_ESTIMATION_WIRES > 0

    def test_default_n_trotter_steps_positive(self):
        """Number of Trotter steps should be positive."""
        assert DEFAULT_N_TROTTER_STEPS > 0

    def test_default_n_qpe_shots_positive(self):
        """Number of QPE shots should be positive."""
        assert DEFAULT_N_QPE_SHOTS > 0
