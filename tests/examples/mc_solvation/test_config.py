# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Unit tests for mc_solvation.config module.

Tests verify configuration dataclass behavior, property calculations,
and validation logic.
"""

import numpy as np
import pytest

from examples.mc_solvation.config import MoleculeConfig, QPEConfig, SolvationConfig
from examples.mc_solvation.constants import (
    BOLTZMANN_CONSTANT,
    DEFAULT_N_ESTIMATION_WIRES,
    DEFAULT_N_MC_STEPS,
    DEFAULT_N_QPE_SHOTS,
    DEFAULT_N_TROTTER_STEPS,
    DEFAULT_N_WATERS,
    DEFAULT_QPE_INTERVAL,
    DEFAULT_TEMPERATURE,
)


class TestMoleculeConfig:
    """Tests for MoleculeConfig dataclass."""

    def test_create_h2_molecule(self, h2_molecule_config):
        """Should create H2 molecule configuration."""
        assert h2_molecule_config.name == "H2"
        assert h2_molecule_config.symbols == ["H", "H"]
        assert h2_molecule_config.charge == 0
        assert h2_molecule_config.active_electrons == 2
        assert h2_molecule_config.active_orbitals == 2

    def test_create_h3op_molecule(self, h3op_molecule_config):
        """Should create H3O+ molecule configuration."""
        assert h3op_molecule_config.name == "H3O+"
        assert h3op_molecule_config.symbols == ["O", "H", "H", "H"]
        assert h3op_molecule_config.charge == 1
        assert h3op_molecule_config.active_electrons == 4
        assert h3op_molecule_config.active_orbitals == 4

    def test_n_atoms_property(self, h2_molecule_config, h3op_molecule_config):
        """n_atoms property should return correct atom count."""
        assert h2_molecule_config.n_atoms == 2
        assert h3op_molecule_config.n_atoms == 4

    def test_coords_array_property(self, h2_molecule_config):
        """coords_array should return numpy array."""
        coords = h2_molecule_config.coords_array
        assert isinstance(coords, np.ndarray)
        assert coords.shape == (2, 3)
        np.testing.assert_array_almost_equal(coords, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

    def test_center_property(self, h2_molecule_config):
        """center property should return geometric center."""
        center = h2_molecule_config.center
        assert isinstance(center, np.ndarray)
        assert center.shape == (3,)
        # H2: center between (0,0,0) and (0,0,0.74) is (0,0,0.37)
        np.testing.assert_array_almost_equal(center, [0.0, 0.0, 0.37])

    def test_default_basis(self):
        """Default basis set should be sto-3g."""
        mol = MoleculeConfig(
            name="H2",
            symbols=["H", "H"],
            coords=[[0, 0, 0], [0, 0, 0.74]],
            charge=0,
            active_electrons=2,
            active_orbitals=2,
        )
        assert mol.basis == "sto-3g"

    def test_custom_basis(self):
        """Should accept custom basis set."""
        mol = MoleculeConfig(
            name="H2",
            symbols=["H", "H"],
            coords=[[0, 0, 0], [0, 0, 0.74]],
            charge=0,
            active_electrons=2,
            active_orbitals=2,
            basis="6-31g",
        )
        assert mol.basis == "6-31g"

    def test_validate_success(self, h2_molecule_config):
        """validate() should pass for valid configuration."""
        h2_molecule_config.validate()  # Should not raise

    def test_validate_mismatched_symbols_coords(self):
        """validate() should raise for mismatched symbols and coords count."""
        mol = MoleculeConfig(
            name="H2",
            symbols=["H", "H"],
            coords=[[0, 0, 0]],  # Only 1 coord for 2 symbols
            charge=0,
            active_electrons=2,
            active_orbitals=2,
        )
        with pytest.raises(ValueError, match="must match"):
            mol.validate()

    def test_validate_wrong_coord_dimension(self):
        """validate() should raise for non-3D coordinates."""
        mol = MoleculeConfig(
            name="H2",
            symbols=["H", "H"],
            coords=[[0, 0], [0, 0]],  # 2D instead of 3D
            charge=0,
            active_electrons=2,
            active_orbitals=2,
        )
        with pytest.raises(ValueError, match="3 components"):
            mol.validate()


class TestQPEConfig:
    """Tests for QPEConfig dataclass."""

    def test_default_values(self):
        """Should use default values when not specified."""
        config = QPEConfig()
        assert config.n_estimation_wires == DEFAULT_N_ESTIMATION_WIRES
        assert config.n_trotter_steps == DEFAULT_N_TROTTER_STEPS
        assert config.n_shots == DEFAULT_N_QPE_SHOTS
        assert config.qpe_interval == DEFAULT_QPE_INTERVAL
        assert config.use_catalyst is True

    def test_custom_values(self, qpe_config_minimal):
        """Should accept custom values."""
        assert qpe_config_minimal.n_estimation_wires == 3
        assert qpe_config_minimal.n_trotter_steps == 2
        assert qpe_config_minimal.n_shots == 5
        assert qpe_config_minimal.use_catalyst is False

    def test_target_resolution_default(self):
        """Default target resolution should be ~2 kcal/mol."""
        config = QPEConfig()
        assert config.target_resolution == 0.003

    def test_energy_range_default(self):
        """Default energy range should be ±0.1 Ha."""
        config = QPEConfig()
        assert config.energy_range == 0.2


class TestSolvationConfig:
    """Tests for SolvationConfig dataclass."""

    def test_create_minimal_config(self, solvation_config_minimal):
        """Should create minimal solvation configuration."""
        assert solvation_config_minimal.n_waters == 3
        assert solvation_config_minimal.n_mc_steps == 10
        assert solvation_config_minimal.qpe_mode == "vacuum_correction"
        assert solvation_config_minimal.verbose is False

    def test_default_values(self, h2_molecule_config):
        """Should use default values when not specified."""
        config = SolvationConfig(molecule=h2_molecule_config)
        assert config.n_waters == DEFAULT_N_WATERS
        assert config.n_mc_steps == DEFAULT_N_MC_STEPS
        assert config.temperature == DEFAULT_TEMPERATURE
        assert config.qpe_mode == "vacuum_correction"
        assert config.random_seed == 42
        assert config.verbose is True

    def test_n_qpe_evaluations_property(self, solvation_config_minimal):
        """n_qpe_evaluations should calculate correctly."""
        # n_mc_steps=10, qpe_interval=5 -> 2 evaluations
        assert solvation_config_minimal.n_qpe_evaluations == 2

    def test_kt_property(self, solvation_config_minimal):
        """kt property should calculate thermal energy correctly."""
        expected_kt = BOLTZMANN_CONSTANT * 300.0
        assert abs(solvation_config_minimal.kt - expected_kt) < 1e-15

    def test_validate_success(self, solvation_config_minimal):
        """validate() should pass for valid configuration."""
        solvation_config_minimal.validate()  # Should not raise

    def test_validate_zero_waters(self, h2_molecule_config, qpe_config_minimal):
        """validate() should raise for zero waters."""
        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=qpe_config_minimal,
            n_waters=0,
        )
        with pytest.raises(ValueError, match="n_waters"):
            config.validate()

    def test_validate_zero_mc_steps(self, h2_molecule_config, qpe_config_minimal):
        """validate() should raise for zero MC steps."""
        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=qpe_config_minimal,
            n_mc_steps=0,
        )
        with pytest.raises(ValueError, match="n_mc_steps"):
            config.validate()

    def test_validate_qpe_interval_exceeds_mc_steps(self, h2_molecule_config, qpe_config_minimal):
        """validate() should raise if qpe_interval > n_mc_steps."""
        qpe_config = QPEConfig(qpe_interval=100)
        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=qpe_config,
            n_mc_steps=10,
        )
        with pytest.raises(ValueError, match="qpe_interval"):
            config.validate()

    def test_validate_invalid_qpe_mode(self, h2_molecule_config, qpe_config_minimal):
        """validate() should raise for invalid qpe_mode."""
        # Need to bypass dataclass type checking by setting attribute directly
        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=qpe_config_minimal,
        )
        # Force invalid mode (bypassing Literal type hint)
        object.__setattr__(config, "qpe_mode", "invalid_mode")
        with pytest.raises(ValueError, match="Invalid qpe_mode"):
            config.validate()

    def test_vacuum_correction_mode(self, solvation_config_minimal):
        """Should accept vacuum_correction mode."""
        assert solvation_config_minimal.qpe_mode == "vacuum_correction"

    def test_mm_embedded_mode(self, h2_molecule_config, qpe_config_minimal):
        """Should accept mm_embedded mode."""
        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=qpe_config_minimal,
            qpe_mode="mm_embedded",
        )
        assert config.qpe_mode == "mm_embedded"

    def test_qpe_driven_mode(self, h2_molecule_config, qpe_config_minimal):
        """Should accept qpe_driven mode without raising."""
        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=qpe_config_minimal,
            qpe_mode="qpe_driven",
            n_mc_steps=50,
        )
        assert config.qpe_mode == "qpe_driven"
        config.validate()  # Should not raise

    def test_n_qpe_evaluations_qpe_driven(self, h2_molecule_config, qpe_config_minimal):
        """n_qpe_evaluations for qpe_driven should equal n_mc_steps."""
        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=qpe_config_minimal,
            qpe_mode="qpe_driven",
            n_mc_steps=50,
        )
        assert config.n_qpe_evaluations == 50
