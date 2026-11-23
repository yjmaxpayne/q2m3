# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Tests for q2m3.solvation.solvent — solvent models and MM energy."""

import warnings

import numpy as np
import pytest

from q2m3.constants import KCAL_TO_HARTREE, TIP3P_EPSILON_OO, TIP3P_HOH_ANGLE, TIP3P_SIGMA_OO
from q2m3.solvation.solvent import (
    SPC_E_WATER,
    TIP3P_WATER,
    SolventAtom,
    SolventModel,
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


# =============================================================================
# Shared LJ Pair Kernel: compute_mm_energy Characterization
# =============================================================================


def test_compute_mm_energy_bitwise_unchanged():
    """Sharing the LJ pair kernel must not change a single bit of the MM energy.

    The expected values are the ``repr`` of ``compute_mm_energy`` recorded on the
    baseline source, before the pair kernel was extracted. The second
    configuration has a close intermolecular contact (1.04 A) with repulsive energy.
    """
    spread = [
        [0.0, 0.0, 0.0, 0.1, 0.2, 0.3],
        [3.1, 0.4, -0.2, -0.5, 1.1, 0.7],
        [-1.2, 2.9, 0.8, 2.0, -0.4, 1.3],
    ]
    close_pair = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.3, 0.1, 0.0, 0.0, 0.0, 3.0],
        [0.5, -3.4, 1.0, 1.0, 0.5, -2.0],
    ]
    for states, expected in (
        (spread, -0.001614170586215291),
        (close_pair, 0.05843977168933852),
    ):
        mols = [SolventMolecule.from_state_vector(TIP3P_WATER, np.array(s)) for s in states]
        assert compute_mm_energy(mols) == expected


# =============================================================================
# MM Geometry Gate (check_mm_geometry)
# =============================================================================

H3O_PLUS_COORDS = np.array(
    [
        [0.0000, 0.0000, 0.1173],
        [0.0000, 0.9572, -0.4692],
        [0.8286, -0.4786, -0.4692],
        [-0.8286, -0.4786, -0.4692],
    ]
)
# Water oxygen 4 A from the H3O+ centroid; the hydrogens point away from the solute.
_FAR_EAST = H3O_PLUS_COORDS.mean(axis=0) + np.array([4.0, 0.0, 0.0])
_FAR_WEST = H3O_PLUS_COORDS.mean(axis=0) + np.array([-4.0, 0.0, 0.0])
_YAW_PI = np.array([0.0, 0.0, np.pi])
# Water whose hydrogen is 0.87 A from an H3O+ hydrogen (oxygen 1.2 A away).
_CLOSE_POSITION = H3O_PLUS_COORDS[1] + np.array([0.0, 1.2, 0.0])
_CLOSE_EULER = np.array([0.3, 0.2, 0.1])


def _triatomic_model(
    symbols: tuple[str, str, str], bond_A: float, angle_deg: float, name: str = "custom"
) -> SolventModel:
    """Build a symmetric triatomic model with the given bond length and angle."""
    half = np.radians(angle_deg / 2)
    x, y = bond_A * np.cos(half), bond_A * np.sin(half)
    return SolventModel(
        name=name,
        atoms=(
            SolventAtom(symbols[0], (0.0, 0.0, 0.0), -0.834, lj_sigma=3.15061, lj_epsilon=0.152),
            SolventAtom(symbols[1], (x, y, 0.0), 0.417),
            SolventAtom(symbols[2], (x, -y, 0.0), 0.417),
        ),
    )


def _check(molecules, qm_coords=H3O_PLUS_COORDS, **kwargs):
    from q2m3.solvation.solvent import check_mm_geometry

    return check_mm_geometry(molecules, qm_coords, **kwargs)


def _qm_mm_distances(qm_coords: np.ndarray, mm_coords: np.ndarray) -> np.ndarray:
    """Distance matrix of shape (n_qm, n_mm), computed independently of the gate."""
    return np.linalg.norm(qm_coords[:, None, :] - mm_coords[None, :, :], axis=-1)


def test_mm_geometry_accepts_tip3p_water():
    """A TIP3P water 4 A from H3O+ satisfies all three rules."""
    report = _check([SolventMolecule(TIP3P_WATER, _FAR_EAST)])
    assert report.passed is True
    assert report.violations == ()


def test_mm_geometry_rejects_stretched_oh_only():
    """A stretched or a compressed O-H bond violates only the bond rule."""
    stretched = _triatomic_model(("O", "H", "H"), 0.98, 104.52)
    report = _check([SolventMolecule(stretched, _FAR_EAST)])
    assert report.violations == ("oh_bond",)
    assert report.passed is False
    # The bond tolerance comes from the keyword argument.
    assert _check([SolventMolecule(stretched, _FAR_EAST)], oh_tol_A=0.05).violations == ()

    # Only the second O-H bond is stretched.
    half = np.radians(TIP3P_HOH_ANGLE / 2)
    uneven = SolventModel(
        name="uneven",
        atoms=(
            SolventAtom("O", (0.0, 0.0, 0.0), -0.834, lj_sigma=3.15061, lj_epsilon=0.152),
            SolventAtom("H", (0.9572 * np.cos(half), 0.9572 * np.sin(half), 0.0), 0.417),
            SolventAtom("H", (0.98 * np.cos(half), -0.98 * np.sin(half), 0.0), 0.417),
        ),
    )
    assert _check([SolventMolecule(uneven, _FAR_EAST)]).violations == ("oh_bond",)

    # A compressed bond counts too (absolute deviation), also in a later molecule.
    compressed = _triatomic_model(("O", "H", "H"), 0.94, 104.52)
    report = _check(
        [SolventMolecule(TIP3P_WATER, _FAR_EAST), SolventMolecule(compressed, _FAR_WEST, _YAW_PI)]
    )
    assert report.violations == ("oh_bond",)


def test_mm_geometry_rejects_linear_water_only():
    """A linear or a too-narrow H-O-H angle violates only the angle rule."""
    linear = _triatomic_model(("O", "H", "H"), 0.9572, 180.0)
    report = _check([SolventMolecule(linear, _FAR_EAST)])
    assert report.violations == ("hoh_angle",)
    assert report.passed is False
    # The angle tolerance comes from the keyword argument.
    assert _check([SolventMolecule(linear, _FAR_EAST)], hoh_tol_deg=80.0).violations == ()

    # This rotation makes the computed cosine round to -1.0000000000000002;
    # the reported deviation must stay finite (180 - 104.52 degrees).
    rotated = SolventMolecule(linear, _FAR_EAST, np.array([-3.0, -2.5, -1.5]))
    report = _check([rotated])
    assert report.violations == ("hoh_angle",)
    assert np.isfinite(report.hoh_max_dev_deg)
    assert report.hoh_max_dev_deg == pytest.approx(180.0 - TIP3P_HOH_ANGLE, abs=1e-9)

    # A narrower angle counts too (absolute deviation), also in a later molecule.
    narrow = _triatomic_model(("O", "H", "H"), 0.9572, 90.0)
    report = _check(
        [SolventMolecule(TIP3P_WATER, _FAR_EAST), SolventMolecule(narrow, _FAR_WEST, _YAW_PI)]
    )
    assert report.violations == ("hoh_angle",)


def test_mm_geometry_rejects_close_water_only():
    """The QM-MM distance rule uses all atom pairs, a strict bound and the keyword threshold."""
    close = SolventMolecule(TIP3P_WATER, _CLOSE_POSITION, _CLOSE_EULER)
    oracle = _qm_mm_distances(H3O_PLUS_COORDS, close.get_atom_coords()).min()
    report = _check([close])
    assert report.violations == ("qm_mm_distance",)
    assert report.min_qm_mm_A == pytest.approx(oracle, abs=1e-12)
    # The close water is the second molecule.
    far = SolventMolecule(TIP3P_WATER, _FAR_EAST)
    assert _check([far, close]).violations == ("qm_mm_distance",)

    # Threshold equal to the measured distance: "greater than" is strict.
    accepted = [SolventMolecule(TIP3P_WATER, _FAR_EAST)]
    measured = _check(accepted).min_qm_mm_A
    assert _check(accepted, min_qm_mm_A=measured).violations == ("qm_mm_distance",)

    # The water hydrogen is the nearest MM atom; the water oxygen alone is far enough.
    h_nearest = SolventMolecule(
        TIP3P_WATER, H3O_PLUS_COORDS[0] + np.array([2.6, 0.0, 0.0]), _YAW_PI
    )
    pairs = _qm_mm_distances(H3O_PLUS_COORDS, h_nearest.get_atom_coords())
    assert pairs[:, 0].min() > 1.5 > pairs.min()
    report = _check([h_nearest])
    assert report.violations == ("qm_mm_distance",)
    assert report.min_qm_mm_A == pytest.approx(pairs.min(), abs=1e-12)


def test_mm_geometry_rejects_nonfinite_coordinates():
    """A NaN coordinate fails the gate instead of passing silently or raising."""
    qm_coords = H3O_PLUS_COORDS.copy()
    qm_coords[-1, 2] = np.nan
    report = _check([SolventMolecule(TIP3P_WATER, _FAR_EAST)], qm_coords)
    assert report.passed is False


def test_mm_geometry_registered_fixtures_pass():
    """All six registered fixtures pass the gate and their states rebuild their coordinates."""
    from examples.sqd._inputs import load_mm_tip3p_fixtures

    fixtures = load_mm_tip3p_fixtures()
    assert sorted(f["name"] for f in fixtures) == sorted(
        [
            "h2o_1tip3p_4e4o",
            "h2o_3tip3p_4e4o",
            "h2o_3tip3p_8e6o",
            "h3o_plus_3tip3p_4e4o",
            "h3o_plus_8tip3p_4e4o",
            "ac2_h3o_plus_3tip3p",
        ]
    )
    for fixture in fixtures:
        states = np.asarray(fixture["water_states"], dtype=float)
        molecules = [SolventMolecule.from_state_vector(TIP3P_WATER, s) for s in states]
        rebuilt = np.vstack([m.get_atom_coords() for m in molecules])
        assert np.max(np.abs(rebuilt - np.asarray(fixture["water_coords"]))) <= 1e-12
        report = _check(molecules, np.asarray(fixture["qm_coords"], dtype=float))
        assert report.passed is True, fixture["name"]
        assert report.min_qm_mm_A > 1.5, fixture["name"]


def test_mm_check_flags_non_water_model():
    """A molecule that is not O-H-H is flagged and left out of the bond and angle deviations."""
    methylene = _triatomic_model(("C", "H", "H"), 1.09, 120.0, name="CH2")
    other = SolventMolecule(methylene, _FAR_WEST, _YAW_PI)
    report = _check([other, SolventMolecule(TIP3P_WATER, _FAR_EAST)])
    assert report.violations == ("non_water_model",)
    assert report.passed is False
    assert report.oh_max_dev_A <= 1e-12
    assert report.hoh_max_dev_deg <= 1e-9

    only_other = _check([other])
    assert np.isnan(only_other.oh_max_dev_A)
    assert np.isnan(only_other.hoh_max_dev_deg)
    assert only_other.violations == ("non_water_model", "oh_bond", "hoh_angle")
    assert only_other.passed is False


def test_mm_check_orders_multiple_violations():
    """Violations keep the fixed rule order, not alphabetical order."""
    stretched = _triatomic_model(("O", "H", "H"), 0.98, 104.52)
    report = _check([SolventMolecule(stretched, _CLOSE_POSITION, _CLOSE_EULER)])
    assert report.violations == ("oh_bond", "qm_mm_distance")

    stretched_linear = _triatomic_model(("O", "H", "H"), 0.98, 180.0)
    report = _check([SolventMolecule(stretched_linear, _CLOSE_POSITION, _CLOSE_EULER)])
    assert report.violations == ("oh_bond", "hoh_angle", "qm_mm_distance")


# =============================================================================
# QM-MM Lennard-Jones Repulsion (qm_mm_lj_energy)
# =============================================================================


def _lj_energy(qm_symbols, qm_coords, molecules):
    from q2m3.solvation.solvent import qm_mm_lj_energy

    return qm_mm_lj_energy(qm_symbols, qm_coords, molecules)


def test_qm_mm_lj_is_zero_at_sigma():
    """A QM oxygen at distance sigma from a TIP3P oxygen has zero LJ energy."""
    from q2m3.solvation.solvent import QM_MM_LJ_PARAMETERS

    assert QM_MM_LJ_PARAMETERS["O"] == (TIP3P_SIGMA_OO, TIP3P_EPSILON_OO)
    assert QM_MM_LJ_PARAMETERS["H"] == (0.0, 0.0)
    water = SolventMolecule(TIP3P_WATER, np.array([TIP3P_SIGMA_OO, 0.0, 0.0]))
    assert abs(_lj_energy(["O"], np.zeros((1, 3)), [water])) <= 1e-15


def test_qm_mm_lj_minimum_equals_minus_epsilon():
    """At 2^(1/6) sigma the energy is -epsilon, converted once to Hartree."""
    water = SolventMolecule(TIP3P_WATER, np.array([2 ** (1 / 6) * TIP3P_SIGMA_OO, 0.0, 0.0]))
    energy = _lj_energy(["O"], np.zeros((1, 3)), [water])
    assert energy == pytest.approx(-TIP3P_EPSILON_OO * KCAL_TO_HARTREE, abs=1e-15)


def test_qm_mm_lj_ignores_zero_epsilon_sites():
    """Hydrogens on either side carry no LJ term, even when they overlap."""
    symbols = ["H", "O", "H", "H"]
    qm_coords = H3O_PLUS_COORDS[[1, 0, 2, 3]].copy()
    water = SolventMolecule(TIP3P_WATER, H3O_PLUS_COORDS[0] + np.array([3.0, 0.0, 0.0]))
    reference = _lj_energy(symbols, qm_coords, [water])
    assert reference != 0.0

    moved_qm = qm_coords.copy()
    moved_qm[0] += np.array([0.0, 0.5, 0.3])
    assert _lj_energy(symbols, moved_qm, [water]) == reference

    rotated = SolventMolecule(TIP3P_WATER, water.position, np.array([0.4, -0.7, 1.1]))
    assert _lj_energy(symbols, qm_coords, [rotated]) == reference

    # QM hydrogen on top of a water hydrogen (r = 0 for an epsilon-zero pair).
    overlap = qm_coords.copy()
    overlap[0] = water.get_atom_coords()[1]
    energy = _lj_energy(symbols, overlap, [water])
    assert np.isfinite(energy)
    assert energy == reference

    # QM hydrogen on top of the water oxygen; the QM oxygen stays 3 A away.
    h_on_o = qm_coords.copy()
    h_on_o[0] = water.get_atom_coords()[0]
    energy = _lj_energy(symbols, h_on_o, [water])
    assert np.isfinite(energy)
    assert energy == reference

    # QM oxygen on top of a water hydrogen.
    o_on_h = qm_coords.copy()
    o_on_h[1] = water.get_atom_coords()[1]
    assert np.isfinite(_lj_energy(symbols, o_on_h, [water]))


def test_qm_mm_lj_rejects_unknown_element():
    """An element outside the parameter table raises and names the element."""
    from q2m3.solvation.solvent import validate_qm_mm_lj_elements

    water = SolventMolecule(TIP3P_WATER, _FAR_EAST)
    with pytest.raises(ValueError, match="'C'"):
        _lj_energy(["O", "H", "C"], H3O_PLUS_COORDS[:3], [water])
    with pytest.raises(ValueError, match="'C'"):
        validate_qm_mm_lj_elements(["O", "H", "C"])


def test_qm_mm_lj_rejects_solute_without_lj_sites():
    """A solute without any epsilon > 0 atom (H2) raises."""
    from q2m3.solvation.solvent import validate_qm_mm_lj_elements

    water = SolventMolecule(TIP3P_WATER, _FAR_EAST)
    h2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    with pytest.raises(ValueError):
        _lj_energy(["H", "H"], h2, [water])
    with pytest.raises(ValueError):
        validate_qm_mm_lj_elements(["H", "H"])


def test_qm_mm_lj_rejects_nonfinite_coordinates():
    """An infinite QM or MM coordinate raises instead of returning a number."""
    qm_coords = H3O_PLUS_COORDS.copy()
    qm_coords[2, 1] = np.inf
    with pytest.raises(ValueError):
        _lj_energy(["O", "H", "H", "H"], qm_coords, [SolventMolecule(TIP3P_WATER, _FAR_EAST)])

    # Finite QM coordinates; the second water sits at an infinite position.
    waters = [
        SolventMolecule(TIP3P_WATER, _FAR_EAST),
        SolventMolecule(TIP3P_WATER, np.array([np.inf, 0.0, 0.0])),
    ]
    with pytest.raises(ValueError, match="finite"):
        _lj_energy(["O", "H", "H", "H"], H3O_PLUS_COORDS, waters)


def test_qm_mm_lj_rejects_mismatched_coordinate_shape():
    """QM coordinates must have one row of three values per QM symbol."""
    water = SolventMolecule(TIP3P_WATER, _FAR_EAST)
    symbols = ["O", "H", "H", "H"]
    # One row missing, and the right number of rows with only two columns.
    for qm_coords in (H3O_PLUS_COORDS[:3], H3O_PLUS_COORDS[:, :2]):
        with pytest.raises(ValueError, match=r"shape \(4, 3\)"):
            _lj_energy(symbols, qm_coords, [water])
