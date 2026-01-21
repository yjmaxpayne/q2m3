# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Tests for Monte Carlo sampling module.
"""

import numpy as np


class TestWaterMolecule:
    """Test WaterMolecule class for TIP3P water representation."""

    def test_create_water_at_origin(self):
        """Test creating water molecule at origin with default orientation."""
        from q2m3.sampling import WaterMolecule

        water = WaterMolecule(position=np.array([0.0, 0.0, 0.0]))

        # Check oxygen position (at center)
        assert water.oxygen_position is not None
        np.testing.assert_array_almost_equal(water.oxygen_position, [0.0, 0.0, 0.0])

        # Check we have 3 atoms
        coords = water.get_atom_coords()
        assert coords.shape == (3, 3)  # 3 atoms, 3 coordinates each

    def test_water_geometry_correct(self):
        """Test TIP3P water geometry: O-H bond = 0.9572 Å, H-O-H angle = 104.52°."""
        from q2m3.sampling import WaterMolecule

        water = WaterMolecule(position=np.array([0.0, 0.0, 0.0]))
        coords = water.get_atom_coords()

        # O at index 0, H at indices 1, 2
        o_pos = coords[0]
        h1_pos = coords[1]
        h2_pos = coords[2]

        # Check O-H bond lengths (~0.9572 Å)
        oh1_dist = np.linalg.norm(h1_pos - o_pos)
        oh2_dist = np.linalg.norm(h2_pos - o_pos)
        assert abs(oh1_dist - 0.9572) < 0.001
        assert abs(oh2_dist - 0.9572) < 0.001

        # Check H-O-H angle (~104.52°)
        v1 = h1_pos - o_pos
        v2 = h2_pos - o_pos
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle_deg = np.degrees(np.arccos(cos_angle))
        assert abs(angle_deg - 104.52) < 0.1

    def test_water_translation(self):
        """Test water molecule at different position."""
        from q2m3.sampling import WaterMolecule

        pos = np.array([3.0, 2.0, 1.0])
        water = WaterMolecule(position=pos)

        np.testing.assert_array_almost_equal(water.oxygen_position, pos)

    def test_water_rotation_euler(self):
        """Test water molecule with Euler angle rotation."""
        from q2m3.sampling import WaterMolecule

        # Create water with 90 degree rotation around z-axis
        water_default = WaterMolecule(position=np.array([0.0, 0.0, 0.0]))
        water_rotated = WaterMolecule(
            position=np.array([0.0, 0.0, 0.0]),
            euler_angles=np.array([0.0, 0.0, np.pi / 2]),  # 90° around z
        )

        # After 90° z-rotation, x-components become y-components
        coords_default = water_default.get_atom_coords()
        coords_rotated = water_rotated.get_atom_coords()

        # H atoms should have rotated positions
        assert not np.allclose(coords_default, coords_rotated)

    def test_get_tip3p_charges(self):
        """Test TIP3P partial charges: O=-0.834, H=+0.417."""
        from q2m3.sampling import WaterMolecule

        water = WaterMolecule(position=np.array([0.0, 0.0, 0.0]))
        charges = water.get_charges()

        assert len(charges) == 3
        assert abs(charges[0] - (-0.834)) < 0.001  # O charge
        assert abs(charges[1] - 0.417) < 0.001  # H1 charge
        assert abs(charges[2] - 0.417) < 0.001  # H2 charge
        assert abs(sum(charges)) < 0.001  # Total charge = 0

    def test_copy_creates_independent_molecule(self):
        """Test that copy() creates an independent water molecule."""
        from q2m3.sampling import WaterMolecule

        water1 = WaterMolecule(position=np.array([0.0, 0.0, 0.0]))
        water2 = water1.copy()

        # Modify water2 position
        water2.position = np.array([1.0, 1.0, 1.0])

        # water1 should be unchanged
        np.testing.assert_array_almost_equal(water1.oxygen_position, [0.0, 0.0, 0.0])


class TestTIP3PForceField:
    """Test TIP3P force field for MM energy calculations."""

    def test_coulomb_energy_two_waters(self):
        """Test Coulomb energy between two water molecules."""
        from q2m3.sampling import TIP3PForceField, WaterMolecule

        # Two waters at 3 Angstrom separation (O-O distance)
        water1 = WaterMolecule(position=np.array([0.0, 0.0, 0.0]))
        water2 = WaterMolecule(position=np.array([3.0, 0.0, 0.0]))

        ff = TIP3PForceField()
        e_coulomb = ff.compute_coulomb_energy([water1, water2])

        # Coulomb energy should be negative (attractive due to H-O interactions)
        # Rough estimate: dominant interaction is H(+0.417) with O(-0.834)
        # at ~2.5 Å gives ~ -0.01 Ha (order of magnitude)
        assert e_coulomb < 0  # Should be attractive
        assert abs(e_coulomb) < 1.0  # Should be reasonable magnitude in Hartree

    def test_lj_energy_two_waters(self):
        """Test Lennard-Jones energy between two water molecules (O-O only)."""
        from q2m3.sampling import TIP3PForceField, WaterMolecule

        # Two waters at 3.5 Angstrom (near LJ minimum for TIP3P)
        water1 = WaterMolecule(position=np.array([0.0, 0.0, 0.0]))
        water2 = WaterMolecule(position=np.array([3.5, 0.0, 0.0]))

        ff = TIP3PForceField()
        e_lj = ff.compute_lj_energy([water1, water2])

        # LJ should be small negative at equilibrium distance (~3.15 Å sigma)
        # At 3.5 Å, should be close to minimum
        assert isinstance(e_lj, float)

    def test_lj_energy_repulsive_at_close_distance(self):
        """Test LJ energy is strongly repulsive at close distances."""
        from q2m3.sampling import TIP3PForceField, WaterMolecule

        # Two waters at 2.0 Angstrom (very close, should be repulsive)
        water1 = WaterMolecule(position=np.array([0.0, 0.0, 0.0]))
        water2 = WaterMolecule(position=np.array([2.0, 0.0, 0.0]))

        ff = TIP3PForceField()
        e_lj = ff.compute_lj_energy([water1, water2])

        # Should be strongly repulsive (positive)
        assert e_lj > 0

    def test_total_mm_energy(self):
        """Test total MM energy = LJ + Coulomb."""
        from q2m3.sampling import TIP3PForceField, WaterMolecule

        water1 = WaterMolecule(position=np.array([0.0, 0.0, 0.0]))
        water2 = WaterMolecule(position=np.array([3.0, 0.0, 0.0]))

        ff = TIP3PForceField()
        e_total = ff.compute_mm_energy([water1, water2])
        e_lj = ff.compute_lj_energy([water1, water2])
        e_coulomb = ff.compute_coulomb_energy([water1, water2])

        # Total should be sum of components
        assert abs(e_total - (e_lj + e_coulomb)) < 1e-10

    def test_single_water_has_zero_mm_energy(self):
        """Test that single water has zero intermolecular energy."""
        from q2m3.sampling import TIP3PForceField, WaterMolecule

        water = WaterMolecule(position=np.array([0.0, 0.0, 0.0]))

        ff = TIP3PForceField()
        e_total = ff.compute_mm_energy([water])

        assert e_total == 0.0

    def test_three_waters_includes_all_pairs(self):
        """Test that three waters compute all pairwise interactions."""
        from q2m3.sampling import TIP3PForceField, WaterMolecule

        water1 = WaterMolecule(position=np.array([0.0, 0.0, 0.0]))
        water2 = WaterMolecule(position=np.array([3.0, 0.0, 0.0]))
        water3 = WaterMolecule(position=np.array([0.0, 3.0, 0.0]))

        ff = TIP3PForceField()

        # Energy of all three
        e_all = ff.compute_mm_energy([water1, water2, water3])

        # Sum of pairwise energies
        e_12 = ff.compute_mm_energy([water1, water2])
        e_13 = ff.compute_mm_energy([water1, water3])
        e_23 = ff.compute_mm_energy([water2, water3])

        # Should be equal (pairwise additive)
        assert abs(e_all - (e_12 + e_13 + e_23)) < 1e-10


class TestMCMoveGenerator:
    """Test Monte Carlo move generator for water molecules."""

    def test_propose_translation_changes_position(self):
        """Test that translation move changes water position."""
        from q2m3.sampling import MCMoveGenerator, WaterMolecule

        water = WaterMolecule(position=np.array([0.0, 0.0, 0.0]))
        generator = MCMoveGenerator(translation_step=0.3, rotation_step=np.radians(15))

        np.random.seed(42)  # Reproducibility
        new_water = generator.propose_translation(water)

        # Position should change
        assert not np.allclose(water.position, new_water.position)

        # Original should be unchanged
        np.testing.assert_array_almost_equal(water.position, [0.0, 0.0, 0.0])

        # Translation should be bounded by step size
        displacement = np.linalg.norm(new_water.position - water.position)
        assert displacement <= 0.3 * np.sqrt(3) + 1e-10  # Max is step * sqrt(3)

    def test_propose_rotation_changes_orientation(self):
        """Test that rotation move changes water orientation."""
        from q2m3.sampling import MCMoveGenerator, WaterMolecule

        water = WaterMolecule(position=np.array([0.0, 0.0, 0.0]))
        generator = MCMoveGenerator(translation_step=0.3, rotation_step=np.radians(15))

        np.random.seed(42)
        new_water = generator.propose_rotation(water)

        # Euler angles should change
        assert not np.allclose(water.euler_angles, new_water.euler_angles)

        # Position should remain same
        np.testing.assert_array_almost_equal(water.position, new_water.position)

    def test_propose_move_randomly_selects_translation_or_rotation(self):
        """Test that propose_move randomly selects move type."""
        from q2m3.sampling import MCMoveGenerator, WaterMolecule

        water = WaterMolecule(position=np.array([0.0, 0.0, 0.0]))
        generator = MCMoveGenerator(translation_step=0.3, rotation_step=np.radians(15))

        # Generate many moves and check both types occur
        np.random.seed(42)
        n_translations = 0
        n_rotations = 0

        for _ in range(100):
            new_water = generator.propose_move(water)
            if not np.allclose(water.position, new_water.position):
                n_translations += 1
            if not np.allclose(water.euler_angles, new_water.euler_angles):
                n_rotations += 1

        # Both types should occur (with high probability)
        assert n_translations > 0
        assert n_rotations > 0

    def test_propose_move_for_random_water_in_list(self):
        """Test proposing move for one water in a list."""
        from q2m3.sampling import MCMoveGenerator, WaterMolecule

        waters = [
            WaterMolecule(position=np.array([0.0, 0.0, 0.0])),
            WaterMolecule(position=np.array([3.0, 0.0, 0.0])),
            WaterMolecule(position=np.array([0.0, 3.0, 0.0])),
        ]
        generator = MCMoveGenerator(translation_step=0.3, rotation_step=np.radians(15))

        np.random.seed(42)
        new_waters, moved_idx = generator.propose_move_for_system(waters)

        # Should return new list and index of moved water
        assert len(new_waters) == 3
        assert 0 <= moved_idx < 3

        # Only one water should change
        n_changed = sum(
            1
            for i in range(3)
            if not (
                np.allclose(waters[i].position, new_waters[i].position)
                and np.allclose(waters[i].euler_angles, new_waters[i].euler_angles)
            )
        )
        assert n_changed == 1


class TestMetropolisSampler:
    """Test Metropolis-Hastings sampler for solvation optimization."""

    def test_sampler_initialization(self):
        """Test sampler initialization with energy function."""
        from q2m3.sampling import MetropolisSampler, TIP3PForceField, WaterMolecule

        waters = [
            WaterMolecule(position=np.array([3.0, 0.0, 0.0])),
            WaterMolecule(position=np.array([-3.0, 0.0, 0.0])),
        ]

        def energy_fn(ws):
            ff = TIP3PForceField()
            return ff.compute_mm_energy(ws)

        sampler = MetropolisSampler(
            waters=waters,
            energy_function=energy_fn,
            temperature=300.0,
        )

        assert sampler.temperature == 300.0
        assert len(sampler.waters) == 2

    def test_metropolis_accept_lower_energy(self):
        """Test that lower energy moves are always accepted."""
        from q2m3.sampling import MetropolisSampler

        # Mock energy function: new < old should always accept
        def energy_fn(ws):
            return sum(w.position[0] for w in ws)  # Energy = sum of x-coords

        from q2m3.sampling import WaterMolecule

        waters = [WaterMolecule(position=np.array([1.0, 0.0, 0.0]))]

        sampler = MetropolisSampler(waters=waters, energy_function=energy_fn, temperature=300.0)

        # Lower energy should always be accepted
        old_energy = 1.0
        new_energy = 0.5
        assert sampler._accept_move(old_energy, new_energy) is True

    def test_metropolis_probabilistic_accept_higher_energy(self):
        """Test that higher energy moves are accepted probabilistically."""
        from q2m3.sampling import MetropolisSampler, WaterMolecule

        def energy_fn(ws):
            return 0.0

        waters = [WaterMolecule(position=np.array([0.0, 0.0, 0.0]))]

        sampler = MetropolisSampler(waters=waters, energy_function=energy_fn, temperature=300.0)

        # With small energy difference and high temperature, should sometimes accept
        np.random.seed(42)
        n_accepted = sum(
            1 for _ in range(1000) if sampler._accept_move(old_energy=0.0, new_energy=0.001)
        )

        # Should accept some but not all
        assert 0 < n_accepted < 1000

    def test_run_returns_trajectory(self):
        """Test that run() returns sampling trajectory."""
        from q2m3.sampling import MetropolisSampler, TIP3PForceField, WaterMolecule

        waters = [
            WaterMolecule(position=np.array([3.0, 0.0, 0.0])),
            WaterMolecule(position=np.array([-3.0, 0.0, 0.0])),
        ]

        ff = TIP3PForceField()

        def energy_fn(ws):
            return ff.compute_mm_energy(ws)

        sampler = MetropolisSampler(waters=waters, energy_function=energy_fn, temperature=300.0)

        np.random.seed(42)
        result = sampler.run(n_steps=10)

        # Should return trajectory info
        assert "energies" in result
        assert "acceptance_rate" in result
        assert "best_energy" in result
        assert "best_config" in result

        # Should have correct number of energies
        assert len(result["energies"]) == 10

    def test_run_finds_lower_energy_config(self):
        """Test that sampling can find lower energy configurations."""
        from q2m3.sampling import MetropolisSampler, TIP3PForceField, WaterMolecule

        # Start with waters far apart
        waters = [
            WaterMolecule(position=np.array([5.0, 0.0, 0.0])),
            WaterMolecule(position=np.array([-5.0, 0.0, 0.0])),
        ]

        ff = TIP3PForceField()
        initial_energy = ff.compute_mm_energy(waters)

        def energy_fn(ws):
            return ff.compute_mm_energy(ws)

        sampler = MetropolisSampler(
            waters=waters,
            energy_function=energy_fn,
            temperature=300.0,
            translation_step=0.5,
        )

        np.random.seed(42)
        result = sampler.run(n_steps=50)

        # Best energy should be at least as good as initial
        # (might not improve much in 50 steps, but shouldn't get worse systematically)
        assert result["best_energy"] <= initial_energy + 0.01  # Allow small noise
