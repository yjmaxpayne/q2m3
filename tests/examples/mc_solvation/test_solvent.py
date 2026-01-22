# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Unit tests for mc_solvation.solvent module.

Tests verify solvent model definitions, coordinate transformations,
and MM energy calculations.
"""

import numpy as np
import pytest

from examples.mc_solvation.constants import ANGSTROM_TO_BOHR
from examples.mc_solvation.solvent import (
    SOLVENT_MODELS,
    SPC_E_WATER,
    TIP3P_WATER,
    SolventAtom,
    SolventModel,
    SolventMolecule,
    _euler_to_rotation_matrix,
    compute_mm_energy,
    get_mm_embedding_data,
    get_mm_embedding_data_bohr,
    initialize_solvent_ring,
    molecules_to_state_array,
    state_array_to_molecules,
)


class TestSolventAtom:
    """Tests for SolventAtom dataclass."""

    def test_create_atom(self):
        """Should create solvent atom with given parameters."""
        atom = SolventAtom("O", (0.0, 0.0, 0.0), -0.834, lj_sigma=3.15, lj_epsilon=0.15)
        assert atom.symbol == "O"
        assert atom.local_coord == (0.0, 0.0, 0.0)
        assert atom.charge == -0.834
        assert atom.lj_sigma == 3.15
        assert atom.lj_epsilon == 0.15

    def test_default_lj_params(self):
        """LJ parameters should default to zero."""
        atom = SolventAtom("H", (1.0, 0.0, 0.0), 0.417)
        assert atom.lj_sigma == 0.0
        assert atom.lj_epsilon == 0.0

    def test_local_coord_array_property(self):
        """local_coord_array should return numpy array."""
        atom = SolventAtom("O", (1.0, 2.0, 3.0), -0.8)
        coords = atom.local_coord_array
        assert isinstance(coords, np.ndarray)
        np.testing.assert_array_equal(coords, [1.0, 2.0, 3.0])

    def test_frozen_immutable(self):
        """SolventAtom should be immutable (frozen dataclass)."""
        atom = SolventAtom("O", (0.0, 0.0, 0.0), -0.8)
        with pytest.raises(Exception):  # FrozenInstanceError
            atom.symbol = "H"


class TestSolventModel:
    """Tests for SolventModel dataclass."""

    def test_tip3p_model_structure(self, tip3p_model):
        """TIP3P model should have correct structure."""
        assert tip3p_model.name == "TIP3P"
        assert tip3p_model.n_atoms == 3
        assert tip3p_model.symbols == ["O", "H", "H"]

    def test_tip3p_charge_neutral(self, tip3p_model):
        """TIP3P water should be charge neutral."""
        total_charge = tip3p_model.charges.sum()
        assert abs(total_charge) < 1e-10

    def test_spce_model_structure(self, spce_model):
        """SPC/E model should have correct structure."""
        assert spce_model.name == "SPC/E"
        assert spce_model.n_atoms == 3
        assert spce_model.symbols == ["O", "H", "H"]

    def test_spce_charge_neutral(self, spce_model):
        """SPC/E water should be charge neutral."""
        total_charge = spce_model.charges.sum()
        assert abs(total_charge) < 1e-10

    def test_local_coords_shape(self, tip3p_model):
        """local_coords should have shape (n_atoms, 3)."""
        coords = tip3p_model.local_coords
        assert coords.shape == (3, 3)

    def test_oxygen_at_origin(self, tip3p_model):
        """Oxygen should be at local origin."""
        np.testing.assert_array_almost_equal(tip3p_model.local_coords[0], [0.0, 0.0, 0.0])

    def test_get_lj_params_oxygen(self, tip3p_model):
        """Oxygen should have LJ parameters."""
        sigma, epsilon = tip3p_model.get_lj_params(0)
        assert sigma > 0
        assert epsilon > 0

    def test_get_lj_params_hydrogen(self, tip3p_model):
        """Hydrogen should have no LJ parameters in TIP3P."""
        sigma, epsilon = tip3p_model.get_lj_params(1)
        assert sigma == 0.0
        assert epsilon == 0.0

    def test_solvent_models_registry(self):
        """SOLVENT_MODELS should contain TIP3P and SPC/E."""
        assert "TIP3P" in SOLVENT_MODELS
        assert "SPC/E" in SOLVENT_MODELS
        assert SOLVENT_MODELS["TIP3P"] is TIP3P_WATER
        assert SOLVENT_MODELS["SPC/E"] is SPC_E_WATER


class TestEulerRotationMatrix:
    """Tests for Euler angle rotation matrix."""

    def test_identity_rotation(self):
        """Zero Euler angles should give identity matrix."""
        R = _euler_to_rotation_matrix(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_rotation_orthogonal(self):
        """Rotation matrix should be orthogonal (R @ R.T = I)."""
        angles = np.array([0.5, 0.3, 0.7])
        R = _euler_to_rotation_matrix(angles)
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3))

    def test_rotation_determinant_one(self):
        """Rotation matrix should have determinant +1."""
        angles = np.array([1.0, 0.5, -0.3])
        R = _euler_to_rotation_matrix(angles)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_90deg_yaw_rotation(self):
        """90° yaw rotation should rotate x to y."""
        yaw = np.pi / 2
        R = _euler_to_rotation_matrix(np.array([0.0, 0.0, yaw]))
        x_axis = np.array([1.0, 0.0, 0.0])
        rotated = R @ x_axis
        np.testing.assert_array_almost_equal(rotated, [0.0, 1.0, 0.0])


class TestSolventMolecule:
    """Tests for SolventMolecule dataclass."""

    def test_create_molecule_default(self, tip3p_model):
        """Should create molecule with default position and orientation."""
        mol = SolventMolecule(tip3p_model)
        np.testing.assert_array_equal(mol.position, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(mol.euler_angles, [0.0, 0.0, 0.0])

    def test_create_molecule_with_position(self, tip3p_model):
        """Should create molecule with specified position."""
        pos = np.array([1.0, 2.0, 3.0])
        mol = SolventMolecule(tip3p_model, position=pos)
        np.testing.assert_array_equal(mol.position, pos)

    def test_get_atom_coords_at_origin(self, tip3p_model):
        """Atom coords at origin should match local coords."""
        mol = SolventMolecule(tip3p_model)
        coords = mol.get_atom_coords()
        np.testing.assert_array_almost_equal(coords, tip3p_model.local_coords)

    def test_get_atom_coords_translated(self, tip3p_model):
        """Atom coords should be translated by position."""
        offset = np.array([5.0, 0.0, 0.0])
        mol = SolventMolecule(tip3p_model, position=offset)
        coords = mol.get_atom_coords()
        expected = tip3p_model.local_coords + offset
        np.testing.assert_array_almost_equal(coords, expected)

    def test_get_charges(self, tip3p_model):
        """get_charges should return model charges."""
        mol = SolventMolecule(tip3p_model)
        np.testing.assert_array_equal(mol.get_charges(), tip3p_model.charges)

    def test_to_state_vector(self, tip3p_model):
        """to_state_vector should concatenate position and angles."""
        pos = np.array([1.0, 2.0, 3.0])
        angles = np.array([0.1, 0.2, 0.3])
        mol = SolventMolecule(tip3p_model, position=pos, euler_angles=angles)
        state = mol.to_state_vector()
        np.testing.assert_array_equal(state, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

    def test_from_state_vector(self, tip3p_model):
        """from_state_vector should reconstruct molecule."""
        state = np.array([4.0, 5.0, 6.0, 0.5, 0.6, 0.7])
        mol = SolventMolecule.from_state_vector(tip3p_model, state)
        np.testing.assert_array_equal(mol.position, [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(mol.euler_angles, [0.5, 0.6, 0.7])

    def test_state_vector_roundtrip(self, tip3p_model):
        """State vector conversion should be lossless."""
        pos = np.array([1.5, 2.5, 3.5])
        angles = np.array([0.3, 0.4, 0.5])
        mol1 = SolventMolecule(tip3p_model, position=pos, euler_angles=angles)
        state = mol1.to_state_vector()
        mol2 = SolventMolecule.from_state_vector(tip3p_model, state)
        np.testing.assert_array_almost_equal(mol1.position, mol2.position)
        np.testing.assert_array_almost_equal(mol1.euler_angles, mol2.euler_angles)


class TestMMEmbeddingData:
    """Tests for MM embedding data extraction."""

    def test_single_molecule_coords(self, tip3p_model):
        """Should return correct coords for single molecule."""
        mol = SolventMolecule(tip3p_model)
        coords, charges = get_mm_embedding_data([mol])
        assert coords.shape == (3, 3)
        assert charges.shape == (3,)

    def test_multiple_molecules_coords(self, tip3p_model):
        """Should concatenate coords from multiple molecules."""
        mols = [
            SolventMolecule(tip3p_model, position=np.array([0.0, 0.0, 0.0])),
            SolventMolecule(tip3p_model, position=np.array([5.0, 0.0, 0.0])),
        ]
        coords, charges = get_mm_embedding_data(mols)
        assert coords.shape == (6, 3)
        assert charges.shape == (6,)

    def test_empty_molecules_list(self):
        """Should handle empty molecule list."""
        coords, charges = get_mm_embedding_data([])
        assert coords.shape == (0, 3)
        assert charges.shape == (0,)

    def test_bohr_conversion(self, tip3p_model):
        """Bohr coords should be Angstrom coords * conversion factor."""
        mol = SolventMolecule(tip3p_model, position=np.array([1.0, 0.0, 0.0]))
        coords_ang, _ = get_mm_embedding_data([mol])
        coords_bohr, _ = get_mm_embedding_data_bohr([mol])
        np.testing.assert_array_almost_equal(coords_bohr, coords_ang * ANGSTROM_TO_BOHR)


class TestMMEnergyCalculation:
    """Tests for MM energy calculation."""

    def test_single_molecule_zero_energy(self, tip3p_model):
        """Single molecule should have zero MM energy."""
        mol = SolventMolecule(tip3p_model)
        energy = compute_mm_energy([mol])
        assert energy == 0.0

    def test_empty_molecules_zero_energy(self):
        """Empty molecule list should have zero MM energy."""
        energy = compute_mm_energy([])
        assert energy == 0.0

    def test_two_molecules_positive_at_short_distance(self, tip3p_model):
        """Two molecules very close should have positive (repulsive) LJ energy."""
        mol1 = SolventMolecule(tip3p_model, position=np.array([0.0, 0.0, 0.0]))
        mol2 = SolventMolecule(tip3p_model, position=np.array([2.0, 0.0, 0.0]))
        energy = compute_mm_energy([mol1, mol2])
        # LJ repulsion dominates at short distance
        assert energy > 0

    def test_two_molecules_negative_at_equilibrium(self, tip3p_model):
        """Two molecules at equilibrium distance should have negative energy."""
        # TIP3P sigma ~3.15 Å, equilibrium at ~3.5 Å (2^(1/6) * sigma)
        mol1 = SolventMolecule(tip3p_model, position=np.array([0.0, 0.0, 0.0]))
        mol2 = SolventMolecule(tip3p_model, position=np.array([4.5, 0.0, 0.0]))
        energy = compute_mm_energy([mol1, mol2])
        # At moderate distance, Coulomb + LJ attractive
        assert energy < 0

    def test_energy_increases_with_closer_molecules(self, tip3p_model):
        """Energy should increase as molecules get closer (repulsion)."""
        mol1 = SolventMolecule(tip3p_model, position=np.array([0.0, 0.0, 0.0]))

        mol2_close = SolventMolecule(tip3p_model, position=np.array([2.5, 0.0, 0.0]))
        mol2_far = SolventMolecule(tip3p_model, position=np.array([4.0, 0.0, 0.0]))

        energy_close = compute_mm_energy([mol1, mol2_close])
        energy_far = compute_mm_energy([mol1, mol2_far])

        assert energy_close > energy_far


class TestSolventInitialization:
    """Tests for solvent initialization utilities."""

    def test_initialize_ring_count(self, tip3p_model):
        """Should create specified number of molecules."""
        molecules = initialize_solvent_ring(
            model=tip3p_model,
            n_molecules=5,
            center=np.array([0.0, 0.0, 0.0]),
            radius=4.0,
        )
        assert len(molecules) == 5

    def test_initialize_ring_positions(self, tip3p_model):
        """Molecules should be equidistant from center."""
        center = np.array([1.0, 2.0, 3.0])
        radius = 4.0
        molecules = initialize_solvent_ring(
            model=tip3p_model,
            n_molecules=4,
            center=center,
            radius=radius,
        )
        for mol in molecules:
            distance = np.linalg.norm(mol.position - center)
            assert abs(distance - radius) < 1e-10

    def test_initialize_ring_in_xy_plane(self, tip3p_model):
        """Molecule centers should be in the x-y plane (same z as center)."""
        center = np.array([0.0, 0.0, 5.0])
        molecules = initialize_solvent_ring(
            model=tip3p_model,
            n_molecules=3,
            center=center,
            radius=4.0,
        )
        for mol in molecules:
            assert abs(mol.position[2] - center[2]) < 1e-10

    def test_initialize_ring_reproducible(self, tip3p_model):
        """Same seed should produce same configuration."""
        mols1 = initialize_solvent_ring(
            model=tip3p_model,
            n_molecules=3,
            center=np.zeros(3),
            radius=4.0,
            random_seed=42,
        )
        mols2 = initialize_solvent_ring(
            model=tip3p_model,
            n_molecules=3,
            center=np.zeros(3),
            radius=4.0,
            random_seed=42,
        )
        for m1, m2 in zip(mols1, mols2):
            np.testing.assert_array_equal(m1.position, m2.position)
            np.testing.assert_array_equal(m1.euler_angles, m2.euler_angles)


class TestStateArrayConversion:
    """Tests for state array conversion utilities."""

    def test_molecules_to_state_array(self, water_molecules_3):
        """Should convert molecules to state array."""
        states = molecules_to_state_array(water_molecules_3)
        assert states.shape == (3, 6)

    def test_state_array_to_molecules(self, tip3p_model):
        """Should convert state array back to molecules."""
        states = np.array(
            [
                [1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
                [4.0, 5.0, 6.0, 0.4, 0.5, 0.6],
            ]
        )
        molecules = state_array_to_molecules(tip3p_model, states)
        assert len(molecules) == 2
        np.testing.assert_array_equal(molecules[0].position, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(molecules[1].euler_angles, [0.4, 0.5, 0.6])

    def test_state_array_roundtrip(self, water_molecules_3):
        """State array conversion should be lossless."""
        states = molecules_to_state_array(water_molecules_3)
        model = water_molecules_3[0].model
        restored = state_array_to_molecules(model, states)

        for orig, rest in zip(water_molecules_3, restored):
            np.testing.assert_array_almost_equal(orig.position, rest.position)
            np.testing.assert_array_almost_equal(orig.euler_angles, rest.euler_angles)
