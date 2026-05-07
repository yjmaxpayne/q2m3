# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Internal structure-analysis helpers for finite-shell solvation outputs.

The routines in this module are pure NumPy/CSV helpers around the solvent
state arrays already returned by :func:`q2m3.solvation.run_solvation`. They are
kept out of ``q2m3.solvation.__init__`` because they support optional
post-processing workflows rather than the primary solvation API.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .solvent import TIP3P_WATER, state_array_to_molecules

STATE_COLUMNS = ("x", "y", "z", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class RadialProfile:
    """Finite-shell oxygen radial-density profile around a solute center."""

    bin_edges: np.ndarray
    counts: np.ndarray
    shell_volumes: np.ndarray
    density_per_a3: np.ndarray
    mean_count_per_frame: np.ndarray
    cumulative_coordination: np.ndarray
    n_frames: int

    @property
    def bin_centers(self) -> np.ndarray:
        """Bin centers in Angstrom."""
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])


def validate_state_trajectory(states: np.ndarray) -> np.ndarray:
    """Return a float trajectory array with shape ``(n_steps, n_waters, 6)``."""
    arr = np.asarray(states, dtype=float)
    if arr.ndim != 3 or arr.shape[2] != len(STATE_COLUMNS):
        raise ValueError(
            "solvent state trajectory must have shape (n_steps, n_waters, 6); " f"got {arr.shape}"
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("solvent state trajectory must include at least one step and water")
    return arr


def write_state_trajectory_csv(path: Path, states: np.ndarray) -> None:
    """Write solvent states as a long-form CSV for deterministic round-trips."""
    arr = validate_state_trajectory(states)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["step", "water_index", *STATE_COLUMNS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(arr.shape[0]):
            for water_index in range(arr.shape[1]):
                row = {
                    "step": step,
                    "water_index": water_index,
                    **{
                        column: f"{arr[step, water_index, i]:.12g}"
                        for i, column in enumerate(STATE_COLUMNS)
                    },
                }
                writer.writerow(row)


def load_state_trajectory_csv(path: Path) -> np.ndarray:
    """Load a solvent-state trajectory CSV.

    The preferred schema is long-form ``step,water_index,x,y,z,roll,pitch,yaw``.
    A compact wide schema with columns like ``water_0_x`` or ``w0_x`` is also
    accepted for compatibility with ad hoc notebook exports.
    """
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError(f"{path} contains no trajectory rows")

    fields = set(fieldnames)
    if {"step", "water_index", *STATE_COLUMNS}.issubset(fields):
        return _load_long_state_rows(rows, path)
    return _load_wide_state_rows(rows, fieldnames, path)


def _load_long_state_rows(rows: list[dict[str, str]], path: Path) -> np.ndarray:
    """Load the canonical long-form state schema."""
    steps = sorted({int(row["step"]) for row in rows})
    waters = sorted({int(row["water_index"]) for row in rows})
    step_to_idx = {step: i for i, step in enumerate(steps)}
    water_to_idx = {water: i for i, water in enumerate(waters)}
    states = np.full((len(steps), len(waters), len(STATE_COLUMNS)), np.nan, dtype=float)
    for row in rows:
        step = step_to_idx[int(row["step"])]
        water = water_to_idx[int(row["water_index"])]
        states[step, water] = [float(row[column]) for column in STATE_COLUMNS]

    if np.isnan(states).any():
        raise ValueError(f"{path} has missing step/water state entries")
    return states


def _load_wide_state_rows(
    rows: list[dict[str, str]], fieldnames: list[str], path: Path
) -> np.ndarray:
    """Load a wide state schema with one row per step."""
    pattern = re.compile(r"^(?:water_|w)(\d+)_(" + "|".join(STATE_COLUMNS) + r")$")
    matches: dict[int, dict[str, str]] = {}
    for field in fieldnames:
        match = pattern.match(field)
        if match:
            water_index = int(match.group(1))
            column = match.group(2)
            matches.setdefault(water_index, {})[column] = field
    if not matches or any(set(cols) != set(STATE_COLUMNS) for cols in matches.values()):
        raise ValueError(
            f"{path} must contain either long-form state columns or complete water_N_* columns"
        )

    waters = sorted(matches)
    states = np.zeros((len(rows), len(waters), len(STATE_COLUMNS)), dtype=float)
    for step, row in enumerate(rows):
        for water_pos, water_index in enumerate(waters):
            columns = matches[water_index]
            states[step, water_pos] = [float(row[columns[column]]) for column in STATE_COLUMNS]
    return states


def solute_center(solute_coords: np.ndarray) -> np.ndarray:
    """Return the arithmetic center of solute coordinates."""
    coords = np.asarray(solute_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
        raise ValueError(f"solute coordinates must have shape (n_atoms, 3); got {coords.shape}")
    return coords.mean(axis=0)


def water_oxygen_distances(states: np.ndarray, solute_coords: np.ndarray) -> np.ndarray:
    """Return oxygen-center distances to the solute center for every frame/water."""
    arr = validate_state_trajectory(states)
    center = solute_center(solute_coords)
    oxygen_positions = arr[:, :, :3]
    return np.linalg.norm(oxygen_positions - center[None, None, :], axis=2)


def radial_density_profile(
    states: np.ndarray,
    solute_coords: np.ndarray,
    bin_edges: np.ndarray,
    *,
    start: int = 0,
) -> RadialProfile:
    """Compute finite-shell oxygen radial density and cumulative coordination."""
    arr = validate_state_trajectory(states)
    if start < 0 or start >= arr.shape[0]:
        raise ValueError(f"start must be in [0, {arr.shape[0] - 1}], got {start}")
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be a strictly increasing 1D array")

    distances = water_oxygen_distances(arr[start:], solute_coords).reshape(-1)
    counts, _ = np.histogram(distances, bins=edges)
    shell_volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    n_frames = arr.shape[0] - start
    mean_count = counts.astype(float) / n_frames
    density = mean_count / shell_volumes
    cumulative = np.cumsum(counts).astype(float) / n_frames
    return RadialProfile(
        bin_edges=edges,
        counts=counts.astype(int),
        shell_volumes=shell_volumes,
        density_per_a3=density,
        mean_count_per_frame=mean_count,
        cumulative_coordination=cumulative,
        n_frames=n_frames,
    )


def shell_counts_per_frame(
    states: np.ndarray,
    solute_coords: np.ndarray,
    cutoff: float,
    *,
    start: int = 0,
) -> np.ndarray:
    """Return per-frame water counts within ``cutoff`` Angstrom."""
    arr = validate_state_trajectory(states)
    if cutoff < 0:
        raise ValueError(f"cutoff must be non-negative, got {cutoff}")
    distances = water_oxygen_distances(arr[start:], solute_coords)
    return np.sum(distances <= cutoff, axis=1).astype(int)


def coordination_by_cutoff(
    states: np.ndarray,
    solute_coords: np.ndarray,
    cutoffs: np.ndarray,
    *,
    start: int = 0,
) -> np.ndarray:
    """Return mean coordination count for each cutoff."""
    arr = validate_state_trajectory(states)
    cutoff_arr = np.asarray(cutoffs, dtype=float)
    if cutoff_arr.ndim != 1 or len(cutoff_arr) == 0:
        raise ValueError("cutoffs must be a non-empty 1D array")
    distances = water_oxygen_distances(arr[start:], solute_coords)
    return np.array([np.mean(np.sum(distances <= cutoff, axis=1)) for cutoff in cutoff_arr])


def select_representative_snapshot_indices(
    states: np.ndarray,
    solute_coords: np.ndarray,
    *,
    n_snapshots: int = 3,
    cutoff: float = 3.5,
    start: int = 0,
) -> list[int]:
    """Select deterministic post-burn snapshots spanning shell occupancy."""
    arr = validate_state_trajectory(states)
    if n_snapshots < 1:
        raise ValueError(f"n_snapshots must be >= 1, got {n_snapshots}")
    start = max(0, min(start, arr.shape[0] - 1))
    candidates = np.arange(start, arr.shape[0], dtype=int)
    if len(candidates) <= n_snapshots:
        return candidates.tolist()

    scores = shell_counts_per_frame(arr, solute_coords, cutoff, start=start)
    quantiles = np.linspace(0.15, 0.85, n_snapshots)
    targets = np.quantile(scores, quantiles)
    selected: list[int] = []
    for target in targets:
        order = np.lexsort((candidates, np.abs(scores - target)))
        for idx in order:
            candidate = int(candidates[idx])
            if candidate not in selected:
                selected.append(candidate)
                break

    if len(selected) < n_snapshots:
        for candidate in candidates:
            if int(candidate) not in selected:
                selected.append(int(candidate))
            if len(selected) == n_snapshots:
                break
    return sorted(selected)


def select_random_snapshot_indices(
    states: np.ndarray,
    *,
    n_snapshots: int = 3,
    start: int = 0,
    stop: int | None = None,
    random_seed: int = 42,
) -> list[int]:
    """Select reproducible random snapshot indices from a trajectory window."""
    arr = validate_state_trajectory(states)
    if n_snapshots < 1:
        raise ValueError(f"n_snapshots must be >= 1, got {n_snapshots}")
    start = max(0, min(start, arr.shape[0] - 1))
    stop_idx = arr.shape[0] if stop is None else max(start + 1, min(stop, arr.shape[0]))
    candidates = np.arange(start, stop_idx, dtype=int)
    if len(candidates) <= n_snapshots:
        return candidates.tolist()
    rng = np.random.default_rng(random_seed)
    chosen = rng.choice(candidates, size=n_snapshots, replace=False)
    return sorted(int(index) for index in chosen)


def state_to_symbols_coords(
    solute_symbols: list[str],
    solute_coords: np.ndarray,
    solvent_state: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Convert one solvent-state frame to combined solute + TIP3P atom arrays."""
    coords = np.asarray(solute_coords, dtype=float)
    state = np.asarray(solvent_state, dtype=float)
    if state.ndim != 2 or state.shape[1] != len(STATE_COLUMNS):
        raise ValueError(f"one solvent frame must have shape (n_waters, 6); got {state.shape}")

    symbols = list(solute_symbols)
    all_coords = [coords]
    for molecule in state_array_to_molecules(TIP3P_WATER, state):
        symbols.extend(TIP3P_WATER.symbols)
        all_coords.append(molecule.get_atom_coords())
    return symbols, np.vstack(all_coords)


def write_xyz_snapshot(
    path: Path,
    solute_symbols: list[str],
    solute_coords: np.ndarray,
    solvent_state: np.ndarray,
    *,
    comment: str = "",
) -> None:
    """Write one solute + solvent-shell frame as an XYZ file."""
    symbols, coords = state_to_symbols_coords(solute_symbols, solute_coords, solvent_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(symbols)), comment]
    for symbol, coord in zip(symbols, coords, strict=True):
        lines.append(f"{symbol} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def xyz_atom_count(path: Path) -> int:
    """Return the atom count declared by an XYZ file."""
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    return int(first_line)


def radial_profile_to_rows(profile: RadialProfile) -> list[dict[str, Any]]:
    """Convert a radial profile to CSV-ready dictionaries."""
    rows: list[dict[str, Any]] = []
    for i in range(len(profile.counts)):
        rows.append(
            {
                "bin_start_a": float(profile.bin_edges[i]),
                "bin_end_a": float(profile.bin_edges[i + 1]),
                "bin_center_a": float(profile.bin_centers[i]),
                "oxygen_count": int(profile.counts[i]),
                "mean_count_per_frame": float(profile.mean_count_per_frame[i]),
                "shell_volume_a3": float(profile.shell_volumes[i]),
                "radial_density_per_a3": float(profile.density_per_a3[i]),
                "cumulative_coordination": float(profile.cumulative_coordination[i]),
                "n_frames": int(profile.n_frames),
            }
        )
    return rows
