# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Tests for internal finite-shell solvation structure-analysis helpers."""

import numpy as np

from q2m3.solvation.structure_analysis import (
    coordination_by_cutoff,
    load_state_trajectory_csv,
    radial_density_profile,
    select_random_snapshot_indices,
    select_representative_snapshot_indices,
    write_state_trajectory_csv,
    write_xyz_snapshot,
    xyz_atom_count,
)


def _trajectory() -> np.ndarray:
    """Small deterministic two-water trajectory."""
    states = np.zeros((4, 2, 6), dtype=float)
    states[:, 0, 0] = [1.0, 1.2, 1.4, 1.6]
    states[:, 1, 0] = [3.0, 3.2, 3.4, 3.6]
    states[:, 1, 1] = [0.0, 0.2, 0.0, -0.2]
    return states


def test_state_trajectory_csv_roundtrip(tmp_path):
    """Long-form trajectory CSV preserves solvent state arrays."""
    path = tmp_path / "trajectory.csv"
    states = _trajectory()

    write_state_trajectory_csv(path, states)
    loaded = load_state_trajectory_csv(path)

    np.testing.assert_allclose(loaded, states)


def test_rdf_histogram_counts_are_conserved():
    """Histogram counts equal frames x waters when bins cover all distances."""
    states = _trajectory()
    solute = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    profile = radial_density_profile(states, solute, np.array([0.0, 2.0, 4.0, 6.0]))

    assert int(np.sum(profile.counts)) == states.shape[0] * states.shape[1]
    assert profile.n_frames == states.shape[0]
    assert profile.cumulative_coordination[-1] == states.shape[1]


def test_coordination_count_is_monotonic_with_cutoff():
    """Mean coordination cannot decrease as cutoff increases."""
    states = _trajectory()
    solute = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    counts = coordination_by_cutoff(states, solute, np.array([1.0, 2.0, 4.0, 6.0]))

    assert np.all(np.diff(counts) >= 0.0)


def test_xyz_atom_count_for_solute_plus_waters(tmp_path):
    """XYZ writer includes solute atoms plus three TIP3P atoms per water."""
    path = tmp_path / "snapshot.xyz"
    states = _trajectory()

    write_xyz_snapshot(
        path,
        ["H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        states[0],
        comment="test snapshot",
    )

    assert xyz_atom_count(path) == 2 + 2 * 3


def test_snapshot_selection_is_deterministic():
    """Representative snapshot selection is deterministic for fixed inputs."""
    states = _trajectory()
    solute = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

    first = select_representative_snapshot_indices(states, solute, n_snapshots=2, start=1)
    second = select_representative_snapshot_indices(states, solute, n_snapshots=2, start=1)

    assert first == second
    assert len(first) == 2
    assert all(index >= 1 for index in first)


def test_random_snapshot_selection_is_seeded_and_windowed():
    """Random snapshot selection is reproducible inside the requested tail window."""
    states = _trajectory()

    first = select_random_snapshot_indices(states, n_snapshots=2, start=2, stop=4, random_seed=7)
    second = select_random_snapshot_indices(states, n_snapshots=2, start=2, stop=4, random_seed=7)

    assert first == second
    assert first == [2, 3]
