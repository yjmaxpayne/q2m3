"""Tests for q2m3.solvation.config — configuration dataclasses."""

import pytest

from q2m3.solvation.config import MoleculeConfig, QPEConfig, SolvationConfig

# =============================================================================
# MoleculeConfig Tests
# =============================================================================


class TestMoleculeConfig:
    """Tests for MoleculeConfig frozen dataclass."""

    def test_creation_with_required_fields(self):
        mol = MoleculeConfig(
            name="H2",
            symbols=["H", "H"],
            coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            charge=0,
        )
        assert mol.name == "H2"
        assert mol.charge == 0
        assert mol.active_electrons == 2  # default
        assert mol.active_orbitals == 2  # default
        assert mol.basis == "sto-3g"  # default

    def test_charge_is_required(self):
        """charge has no default — must be explicitly provided."""
        with pytest.raises(TypeError):
            MoleculeConfig(
                name="H2",
                symbols=["H", "H"],
                coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            )

    def test_frozen(self):
        mol = MoleculeConfig(
            name="H2",
            symbols=["H", "H"],
            coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            charge=0,
        )
        with pytest.raises(AttributeError):
            mol.name = "H3"

    def test_coords_array_property(self):
        import numpy as np

        mol = MoleculeConfig(
            name="H2",
            symbols=["H", "H"],
            coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            charge=0,
        )
        arr = mol.coords_array
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 3)
        assert pytest.approx(arr[1, 2]) == 0.74

    def test_validate_passes(self):
        mol = MoleculeConfig(
            name="H2",
            symbols=["H", "H"],
            coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            charge=0,
        )
        mol.validate()  # should not raise

    def test_validate_symbol_coord_mismatch(self):
        mol = MoleculeConfig(
            name="bad",
            symbols=["H", "H", "H"],
            coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            charge=0,
        )
        with pytest.raises(ValueError, match="symbols.*must match.*coordinate"):
            mol.validate()

    def test_validate_coord_dimension(self):
        mol = MoleculeConfig(
            name="bad",
            symbols=["H"],
            coords=[[0.0, 0.0]],  # only 2 components
            charge=0,
        )
        with pytest.raises(ValueError, match="3 components"):
            mol.validate()

    def test_custom_active_space(self):
        mol = MoleculeConfig(
            name="H3O+",
            symbols=["O", "H", "H", "H"],
            coords=[[0.0, 0.0, 0.0]] * 4,
            charge=1,
            active_electrons=4,
            active_orbitals=4,
        )
        assert mol.active_electrons == 4
        assert mol.active_orbitals == 4
        assert mol.charge == 1


# =============================================================================
# QPEConfig Tests
# =============================================================================


class TestQPEConfig:
    """Tests for QPEConfig frozen dataclass."""

    def test_defaults(self):
        qpe = QPEConfig()
        assert qpe.n_estimation_wires == 4
        assert qpe.n_trotter_steps == 10
        assert qpe.n_shots == 0  # analytical mode
        assert qpe.target_resolution == 0.003
        assert qpe.energy_range == 0.2
        assert qpe.qpe_interval == 10

    def test_frozen(self):
        qpe = QPEConfig()
        with pytest.raises(AttributeError):
            qpe.n_shots = 100

    def test_custom_values(self):
        qpe = QPEConfig(n_estimation_wires=6, n_trotter_steps=20, n_shots=50)
        assert qpe.n_estimation_wires == 6
        assert qpe.n_trotter_steps == 20
        assert qpe.n_shots == 50

    def test_no_use_catalyst_field(self):
        """use_catalyst removed per ADR-004."""
        assert not hasattr(QPEConfig(), "use_catalyst")


# =============================================================================
# SolvationConfig Tests
# =============================================================================


class TestSolvationConfig:
    """Tests for SolvationConfig frozen dataclass."""

    @pytest.fixture
    def h2_mol(self):
        return MoleculeConfig(
            name="H2",
            symbols=["H", "H"],
            coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            charge=0,
        )

    def test_creation_defaults(self, h2_mol):
        cfg = SolvationConfig(molecule=h2_mol)
        assert cfg.hamiltonian_mode == "dynamic"
        assert cfg.n_waters == 10
        assert cfg.n_mc_steps == 500
        assert cfg.temperature == 300.0
        assert cfg.translation_step == 0.3
        assert cfg.rotation_step == 0.2618
        assert cfg.initial_water_distance == 4.0
        assert cfg.random_seed == 42
        assert cfg.verbose is True

    def test_frozen(self, h2_mol):
        cfg = SolvationConfig(molecule=h2_mol)
        with pytest.raises(AttributeError):
            cfg.n_waters = 20

    def test_hamiltonian_mode_hf_corrected(self, h2_mol):
        cfg = SolvationConfig(molecule=h2_mol, hamiltonian_mode="hf_corrected")
        assert cfg.hamiltonian_mode == "hf_corrected"

    def test_hamiltonian_mode_fixed(self, h2_mol):
        cfg = SolvationConfig(molecule=h2_mol, hamiltonian_mode="fixed")
        assert cfg.hamiltonian_mode == "fixed"

    def test_hamiltonian_mode_dynamic(self, h2_mol):
        cfg = SolvationConfig(molecule=h2_mol, hamiltonian_mode="dynamic")
        assert cfg.hamiltonian_mode == "dynamic"

    # --- n_qpe_evaluations property ---

    def test_n_qpe_evaluations_hf_corrected(self, h2_mol):
        cfg = SolvationConfig(
            molecule=h2_mol,
            hamiltonian_mode="hf_corrected",
            n_mc_steps=100,
            qpe_config=QPEConfig(qpe_interval=10),
        )
        assert cfg.n_qpe_evaluations == 10  # 100 // 10

    def test_n_qpe_evaluations_fixed(self, h2_mol):
        cfg = SolvationConfig(
            molecule=h2_mol,
            hamiltonian_mode="fixed",
            n_mc_steps=100,
        )
        assert cfg.n_qpe_evaluations == 100

    def test_n_qpe_evaluations_dynamic(self, h2_mol):
        cfg = SolvationConfig(
            molecule=h2_mol,
            hamiltonian_mode="dynamic",
            n_mc_steps=200,
        )
        assert cfg.n_qpe_evaluations == 200

    # --- kt property ---

    def test_kt_at_300k(self, h2_mol):
        from q2m3.constants import BOLTZMANN_CONSTANT

        cfg = SolvationConfig(molecule=h2_mol, temperature=300.0)
        assert pytest.approx(cfg.kt) == BOLTZMANN_CONSTANT * 300.0

    def test_kt_at_0k(self, h2_mol):
        cfg = SolvationConfig(molecule=h2_mol, temperature=0.0)
        assert cfg.kt == 0.0

    # --- validate() ---

    def test_validate_passes(self, h2_mol):
        cfg = SolvationConfig(molecule=h2_mol)
        result = cfg.validate()
        assert result is cfg  # supports chaining

    def test_validate_n_waters_zero(self, h2_mol):
        cfg = SolvationConfig(molecule=h2_mol, n_waters=0)
        with pytest.raises(ValueError, match="n_waters"):
            cfg.validate()

    def test_validate_n_mc_steps_zero(self, h2_mol):
        cfg = SolvationConfig(molecule=h2_mol, n_mc_steps=0)
        with pytest.raises(ValueError, match="n_mc_steps"):
            cfg.validate()

    def test_validate_invalid_mode(self, h2_mol):
        cfg = SolvationConfig.__new__(SolvationConfig)
        # Bypass frozen to inject invalid value for testing
        object.__setattr__(cfg, "molecule", h2_mol)
        object.__setattr__(cfg, "qpe_config", QPEConfig())
        object.__setattr__(cfg, "hamiltonian_mode", "invalid_mode")
        object.__setattr__(cfg, "n_waters", 10)
        object.__setattr__(cfg, "n_mc_steps", 500)
        object.__setattr__(cfg, "temperature", 300.0)
        object.__setattr__(cfg, "translation_step", 0.3)
        object.__setattr__(cfg, "rotation_step", 0.2618)
        object.__setattr__(cfg, "initial_water_distance", 4.0)
        object.__setattr__(cfg, "random_seed", 42)
        object.__setattr__(cfg, "verbose", True)
        with pytest.raises(ValueError, match="hamiltonian_mode"):
            cfg.validate()

    def test_validate_qpe_interval_exceeds_mc_steps(self, h2_mol):
        cfg = SolvationConfig(
            molecule=h2_mol,
            hamiltonian_mode="hf_corrected",
            n_mc_steps=5,
            qpe_config=QPEConfig(qpe_interval=10),
        )
        with pytest.raises(ValueError, match="qpe_interval"):
            cfg.validate()

    def test_validate_qpe_interval_ignored_for_non_hf_corrected(self, h2_mol):
        """qpe_interval > n_mc_steps is fine for fixed/dynamic modes."""
        cfg = SolvationConfig(
            molecule=h2_mol,
            hamiltonian_mode="dynamic",
            n_mc_steps=5,
            qpe_config=QPEConfig(qpe_interval=10),
        )
        cfg.validate()  # should not raise
