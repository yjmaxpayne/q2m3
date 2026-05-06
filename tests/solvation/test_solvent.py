"""Tests for q2m3.solvation.solvent — solvent models and MM energy."""

import warnings

import numpy as np
import pytest

from q2m3.solvation.solvent import (
    SPC_E_WATER,
    TIP3P_WATER,
    SolventMolecule,
    compute_mm_energy,
    get_mm_embedding_data,
    get_mm_embedding_data_bohr,
    initialize_solvent_ring,
    molecules_to_state_array,
    state_array_to_molecules,
)

# =============================================================================
# TIP3P Model Parameter Tests
# =============================================================================


class TestTIP3PParameters:
    """Verify TIP3P water model parameters match Jorgensen et al. (1983)."""

    def test_tip3p_name(self):
        assert TIP3P_WATER.name == "TIP3P"

    def test_tip3p_n_atoms(self):
        assert TIP3P_WATER.n_atoms == 3

    def test_tip3p_symbols(self):
        assert TIP3P_WATER.symbols == ["O", "H", "H"]

    def test_tip3p_oxygen_charge(self):
        assert TIP3P_WATER.atoms[0].charge == -0.834

    def test_tip3p_hydrogen_charge(self):
        assert TIP3P_WATER.atoms[1].charge == 0.417
        assert TIP3P_WATER.atoms[2].charge == 0.417

    def test_tip3p_charge_neutrality(self):
        total = sum(a.charge for a in TIP3P_WATER.atoms)
        assert pytest.approx(total, abs=1e-10) == 0.0

    def test_tip3p_lj_sigma(self):
        assert TIP3P_WATER.atoms[0].lj_sigma == 3.15061

    def test_tip3p_lj_epsilon(self):
        assert TIP3P_WATER.atoms[0].lj_epsilon == 0.152

    def test_tip3p_hydrogen_no_lj(self):
        """TIP3P hydrogen atoms have no LJ parameters."""
        assert TIP3P_WATER.atoms[1].lj_sigma == 0.0
        assert TIP3P_WATER.atoms[1].lj_epsilon == 0.0

    def test_tip3p_oh_bond_length(self):
        """OH bond length ~0.9572 Angstrom."""
        o_pos = np.array(TIP3P_WATER.atoms[0].local_coord)
        h_pos = np.array(TIP3P_WATER.atoms[1].local_coord)
        bond = np.linalg.norm(h_pos - o_pos)
        assert pytest.approx(bond, abs=1e-4) == 0.9572


class TestSPCEParameters:
    """Verify SPC/E water model basic properties."""

    def test_spce_name(self):
        assert SPC_E_WATER.name == "SPC/E"

    def test_spce_charge_neutrality(self):
        total = sum(a.charge for a in SPC_E_WATER.atoms)
        assert pytest.approx(total, abs=1e-10) == 0.0


# =============================================================================
# SolventMolecule Tests
# =============================================================================


class TestSolventMolecule:
    """Tests for SolventMolecule instance behavior."""

    def test_default_position_at_origin(self):
        mol = SolventMolecule(model=TIP3P_WATER)
        np.testing.assert_array_equal(mol.position, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(mol.euler_angles, [0.0, 0.0, 0.0])

    def test_get_atom_coords_identity_rotation(self):
        """With zero rotation, global coords = local coords + position."""
        pos = np.array([1.0, 2.0, 3.0])
        mol = SolventMolecule(model=TIP3P_WATER, position=pos)
        coords = mol.get_atom_coords()
        assert coords.shape == (3, 3)
        # O atom should be at position (local coord is origin)
        np.testing.assert_allclose(coords[0], pos, atol=1e-10)

    def test_get_atom_coords_preserves_bond_lengths(self):
        """Rotation should preserve bond lengths."""
        mol = SolventMolecule(
            model=TIP3P_WATER,
            position=np.array([5.0, 0.0, 0.0]),
            euler_angles=np.array([0.5, 1.0, 1.5]),
        )
        coords = mol.get_atom_coords()
        oh1 = np.linalg.norm(coords[1] - coords[0])
        oh2 = np.linalg.norm(coords[2] - coords[0])
        assert pytest.approx(oh1, abs=1e-10) == pytest.approx(oh2, abs=1e-10)
        assert pytest.approx(oh1, abs=1e-4) == 0.9572

    def test_get_charges(self):
        mol = SolventMolecule(model=TIP3P_WATER)
        charges = mol.get_charges()
        assert len(charges) == 3
        assert charges[0] == -0.834

    def test_state_vector_roundtrip(self):
        """to_state_vector / from_state_vector roundtrip."""
        pos = np.array([1.0, 2.0, 3.0])
        angles = np.array([0.1, 0.2, 0.3])
        mol = SolventMolecule(model=TIP3P_WATER, position=pos, euler_angles=angles)
        state = mol.to_state_vector()
        assert state.shape == (6,)

        mol2 = SolventMolecule.from_state_vector(TIP3P_WATER, state)
        np.testing.assert_allclose(mol2.position, pos)
        np.testing.assert_allclose(mol2.euler_angles, angles)


# =============================================================================
# MM Embedding Tests
# =============================================================================


class TestMMEmbedding:
    """Tests for PySCF embedding data extraction."""

    def test_get_mm_embedding_data_single_molecule(self):
        mol = SolventMolecule(model=TIP3P_WATER, position=np.array([5.0, 0.0, 0.0]))
        coords, charges = get_mm_embedding_data([mol])
        assert coords.shape == (3, 3)
        assert charges.shape == (3,)

    def test_get_mm_embedding_data_empty(self):
        coords, charges = get_mm_embedding_data([])
        assert coords.shape == (0, 3)
        assert charges.shape == (0,)

    def test_get_mm_embedding_data_bohr_conversion(self):
        from q2m3.constants import ANGSTROM_TO_BOHR

        mol = SolventMolecule(model=TIP3P_WATER, position=np.array([1.0, 0.0, 0.0]))
        coords_bohr, charges = get_mm_embedding_data_bohr([mol])
        coords_ang, _ = get_mm_embedding_data([mol])
        np.testing.assert_allclose(coords_bohr, coords_ang * ANGSTROM_TO_BOHR)


# =============================================================================
# MM Energy Tests
# =============================================================================


class TestMMEnergy:
    """Tests for classical MM energy calculation."""

    def test_single_molecule_zero_energy(self):
        """Single molecule has no intermolecular energy."""
        mol = SolventMolecule(model=TIP3P_WATER, position=np.array([0.0, 0.0, 0.0]))
        energy = compute_mm_energy([mol])
        assert energy == 0.0

    def test_empty_zero_energy(self):
        assert compute_mm_energy([]) == 0.0

    def test_two_molecules_nonzero_energy(self):
        """Two water molecules should have nonzero interaction energy."""
        mol1 = SolventMolecule(model=TIP3P_WATER, position=np.array([0.0, 0.0, 0.0]))
        mol2 = SolventMolecule(model=TIP3P_WATER, position=np.array([3.0, 0.0, 0.0]))
        energy = compute_mm_energy([mol1, mol2])
        assert isinstance(energy, float)
        assert energy != 0.0

    def test_energy_decreases_with_distance(self):
        """Energy should approach zero as molecules move apart."""
        mol1 = SolventMolecule(model=TIP3P_WATER, position=np.array([0.0, 0.0, 0.0]))
        mol_near = SolventMolecule(model=TIP3P_WATER, position=np.array([4.0, 0.0, 0.0]))
        mol_far = SolventMolecule(model=TIP3P_WATER, position=np.array([20.0, 0.0, 0.0]))
        e_near = compute_mm_energy([mol1, mol_near])
        e_far = compute_mm_energy([mol1, mol_far])
        assert abs(e_far) < abs(e_near)

    def test_overlapping_molecules_return_inf_without_runtime_warning(self):
        """Exact intermolecular overlap is rejected without divide-by-zero warnings."""
        mol1 = SolventMolecule(model=TIP3P_WATER, position=np.array([0.0, 0.0, 0.0]))
        mol2 = SolventMolecule(model=TIP3P_WATER, position=np.array([0.0, 0.0, 0.0]))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            energy = compute_mm_energy([mol1, mol2])

        assert np.isposinf(energy)


# =============================================================================
# State Array Conversion Tests
# =============================================================================


class TestStateArrayConversion:
    """Tests for molecules_to_state_array / state_array_to_molecules."""

    def test_roundtrip(self):
        mols = [
            SolventMolecule(
                model=TIP3P_WATER,
                position=np.array([float(i), 0.0, 0.0]),
                euler_angles=np.array([0.1 * i, 0.0, 0.0]),
            )
            for i in range(3)
        ]
        states = molecules_to_state_array(mols)
        assert states.shape == (3, 6)

        mols2 = state_array_to_molecules(TIP3P_WATER, states)
        assert len(mols2) == 3

        for m1, m2 in zip(mols, mols2, strict=False):
            np.testing.assert_allclose(m1.position, m2.position)
            np.testing.assert_allclose(m1.euler_angles, m2.euler_angles)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitializeSolventRing:
    """Tests for initialize_solvent_ring."""

    def test_correct_count(self):
        mols = initialize_solvent_ring(
            model=TIP3P_WATER,
            n_molecules=5,
            center=np.array([0.0, 0.0, 0.0]),
            radius=4.0,
        )
        assert len(mols) == 5

    def test_ring_distance(self):
        """All molecules should be at specified radius from center."""
        center = np.array([1.0, 2.0, 3.0])
        mols = initialize_solvent_ring(
            model=TIP3P_WATER,
            n_molecules=6,
            center=center,
            radius=5.0,
        )
        for mol in mols:
            dist = np.linalg.norm(mol.position - center)
            assert pytest.approx(dist, abs=1e-10) == 5.0

    def test_reproducible_with_seed(self):
        """Same seed should produce same orientations."""
        mols_a = initialize_solvent_ring(
            model=TIP3P_WATER,
            n_molecules=3,
            center=np.array([0.0, 0.0, 0.0]),
            radius=4.0,
            random_seed=42,
        )
        mols_b = initialize_solvent_ring(
            model=TIP3P_WATER,
            n_molecules=3,
            center=np.array([0.0, 0.0, 0.0]),
            radius=4.0,
            random_seed=42,
        )
        for a, b in zip(mols_a, mols_b, strict=False):
            np.testing.assert_array_equal(a.euler_angles, b.euler_angles)

    def test_all_on_xy_plane(self):
        """Ring is in x-y plane: z coordinate matches center z."""
        center = np.array([0.0, 0.0, 5.0])
        mols = initialize_solvent_ring(
            model=TIP3P_WATER,
            n_molecules=4,
            center=center,
            radius=3.0,
        )
        for mol in mols:
            assert mol.position[2] == center[2]
