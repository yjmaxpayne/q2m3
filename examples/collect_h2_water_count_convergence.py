#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Collect lightweight H2 HF/MM water-count convergence support data.

The output is explicitly classical QM/MM support data. It samples H2 in TIP3P
water using Hartree-Fock electrostatic embedding plus classical water-water MM
energy for Metropolis acceptance. It does not run dynamic QPE and should not be
interpreted as a full dynamic-QPE convergence study.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from q2m3.constants import BOLTZMANN_CONSTANT, HARTREE_TO_KCAL_MOL
from q2m3.solvation.config import MoleculeConfig
from q2m3.solvation.energy import compute_hf_energy_solvated, compute_hf_energy_vacuum
from q2m3.solvation.solvent import (
    TIP3P_WATER,
    compute_mm_energy,
    initialize_solvent_ring,
    molecules_to_state_array,
    state_array_to_molecules,
)
from q2m3.solvation.structure_analysis import (
    shell_counts_per_frame,
    validate_state_trajectory,
    water_oxygen_distances,
    write_state_trajectory_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "h2_water_count_convergence"

H2_MOLECULE = MoleculeConfig(
    name="H2",
    symbols=["H", "H"],
    coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    charge=0,
    active_electrons=2,
    active_orbitals=2,
    basis="sto-3g",
)

SamplingRunner = Callable[..., dict[str, Any]]


def _parse_int_csv(value: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _propose_move(
    state: np.ndarray,
    rng: np.random.Generator,
    translation_step: float,
    rotation_step: float,
) -> np.ndarray:
    """Propose a solvent rigid-body move."""
    trial = state.copy()
    trial[:3] += rng.uniform(-translation_step, translation_step, size=3)
    trial[3:] += rng.uniform(-rotation_step, rotation_step, size=3)
    return trial


def _hf_mm_energy(states: np.ndarray) -> tuple[float, float, float]:
    """Return solvated HF, solvent-solvent MM, and total acceptance energy."""
    molecules = state_array_to_molecules(TIP3P_WATER, np.asarray(states, dtype=float))
    e_hf = compute_hf_energy_solvated(H2_MOLECULE, molecules)
    e_mm = compute_mm_energy(molecules)
    return e_hf, e_mm, e_hf + e_mm


def run_classical_hf_mm_sampling(
    *,
    n_waters: int,
    n_mc_steps: int,
    random_seed: int,
    temperature: float,
    translation_step: float,
    rotation_step: float,
    initial_water_distance: float,
) -> dict[str, Any]:
    """Run a lightweight classical QM/MM MC trajectory for one water count."""
    rng = np.random.default_rng(random_seed)
    qm_center = H2_MOLECULE.coords_array.mean(axis=0)
    molecules = initialize_solvent_ring(
        TIP3P_WATER,
        n_waters,
        qm_center,
        initial_water_distance,
        random_seed=random_seed,
    )
    states = molecules_to_state_array(molecules)
    e_vacuum = compute_hf_energy_vacuum(H2_MOLECULE)
    current_hf, current_mm, current_total = _hf_mm_energy(states)
    kt = BOLTZMANN_CONSTANT * temperature

    hf_energies = np.zeros(n_mc_steps, dtype=float)
    mm_energies = np.zeros(n_mc_steps, dtype=float)
    total_energies = np.zeros(n_mc_steps, dtype=float)
    trajectory = np.zeros((n_mc_steps, n_waters, 6), dtype=float)
    n_accepted = 0

    for step in range(n_mc_steps):
        water_index = int(rng.integers(0, n_waters))
        trial_states = states.copy()
        trial_states[water_index] = _propose_move(
            states[water_index], rng, translation_step, rotation_step
        )
        trial_hf, trial_mm, trial_total = _hf_mm_energy(trial_states)

        delta_e = trial_total - current_total
        if delta_e <= 0.0 or rng.random() < np.exp(-delta_e / kt):
            states = trial_states
            current_hf = trial_hf
            current_mm = trial_mm
            current_total = trial_total
            n_accepted += 1

        hf_energies[step] = current_hf
        mm_energies[step] = current_mm
        total_energies[step] = current_total
        trajectory[step] = states

    return {
        "n_waters": n_waters,
        "n_mc_steps": n_mc_steps,
        "random_seed": random_seed,
        "e_vacuum_ha": e_vacuum,
        "hf_energies": hf_energies,
        "mm_energies": mm_energies,
        "total_energies": total_energies,
        "trajectory_solvent_states": trajectory,
        "acceptance_rate": n_accepted / n_mc_steps,
        "n_accepted": n_accepted,
    }


def summarize_sampling_result(
    result: dict[str, Any],
    *,
    late_fraction: float,
    shell_cutoff: float,
) -> dict[str, Any]:
    """Summarize one water-count MC result into a CSV row."""
    trajectory = validate_state_trajectory(np.asarray(result["trajectory_solvent_states"]))
    hf_energies = np.asarray(result["hf_energies"], dtype=float)
    total_energies = np.asarray(result["total_energies"], dtype=float)
    n_steps = len(hf_energies)
    late_start = max(0, min(n_steps - 1, int(np.floor(n_steps * (1.0 - late_fraction)))))
    e_vacuum = float(result["e_vacuum_ha"])
    late_hf = hf_energies[late_start:]
    late_total = total_energies[late_start:]
    shell_counts = shell_counts_per_frame(
        trajectory,
        H2_MOLECULE.coords_array,
        shell_cutoff,
        start=late_start,
    )
    distances = water_oxygen_distances(trajectory[late_start:], H2_MOLECULE.coords_array)
    nearest = np.min(distances, axis=1)
    late_shift_ha = float(np.mean(late_hf) - e_vacuum)

    return {
        "n_waters": int(result["n_waters"]),
        "n_mc_steps": int(result["n_mc_steps"]),
        "random_seed": int(result["random_seed"]),
        "late_start_step": int(late_start),
        "acceptance_rate": float(result["acceptance_rate"]),
        "n_accepted": int(result.get("n_accepted", round(result["acceptance_rate"] * n_steps))),
        "e_vacuum_ha": e_vacuum,
        "late_hf_energy_ha": float(np.mean(late_hf)),
        "late_hf_solvation_shift_ha": late_shift_ha,
        "late_hf_solvation_shift_mha": late_shift_ha * 1000.0,
        "late_hf_solvation_shift_kcal_mol": late_shift_ha * HARTREE_TO_KCAL_MOL,
        "late_total_energy_ha": float(np.mean(late_total)),
        "late_total_std_ha": float(np.std(late_total, ddof=1)) if len(late_total) > 1 else 0.0,
        f"mean_shell_count_{shell_cutoff:.2f}a": float(np.mean(shell_counts)),
        f"final_shell_count_{shell_cutoff:.2f}a": int(
            shell_counts_per_frame(trajectory, H2_MOLECULE.coords_array, shell_cutoff)[-1]
        ),
        "late_nearest_water_a": float(np.mean(nearest)),
        "data_class": "classical_qmmm_hf_mm",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write row dictionaries to CSV."""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_convergence(
    output_dir: Path,
    *,
    n_waters_list: list[int],
    n_mc_steps: int,
    random_seed: int,
    temperature: float,
    translation_step: float,
    rotation_step: float,
    initial_water_distance: float,
    late_fraction: float,
    shell_cutoff: float,
    write_trajectories: bool,
    runner: SamplingRunner | None = None,
) -> dict[str, Any]:
    """Collect convergence rows across water counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = runner or run_classical_hf_mm_sampling
    rows: list[dict[str, Any]] = []
    trajectory_paths: list[str] = []

    for offset, n_waters in enumerate(n_waters_list):
        result = runner(
            n_waters=n_waters,
            n_mc_steps=n_mc_steps,
            random_seed=random_seed + offset,
            temperature=temperature,
            translation_step=translation_step,
            rotation_step=rotation_step,
            initial_water_distance=initial_water_distance,
        )
        row = summarize_sampling_result(
            result,
            late_fraction=late_fraction,
            shell_cutoff=shell_cutoff,
        )
        rows.append(row)
        if write_trajectories:
            trajectory_path = output_dir / f"h2_hfmm_{n_waters:02d}waters_trajectory.csv"
            write_state_trajectory_csv(
                trajectory_path,
                np.asarray(result["trajectory_solvent_states"], dtype=float),
            )
            trajectory_paths.append(str(trajectory_path))

    csv_path = output_dir / "h2_water_count_convergence.csv"
    json_path = output_dir / "h2_water_count_convergence.json"
    _write_csv(csv_path, rows)
    payload = {
        "metadata": {
            "schema": "h2_water_count_convergence.v1",
            "data_class": "classical_qmmm_hf_mm",
            "note": (
                "Classical QM/MM HF/MM support data only; this does not run or "
                "claim full dynamic QPE convergence."
            ),
            "temperature_k": temperature,
            "shell_cutoff_a": shell_cutoff,
            "late_fraction": late_fraction,
        },
        "rows": rows,
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "trajectories": trajectory_paths,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None, runner: SamplingRunner | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-waters", default="2,4,6,8,10,12")
    parser.add_argument("--n-mc-steps", type=int, default=60)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--translation-step", type=float, default=0.3)
    parser.add_argument("--rotation-step", type=float, default=0.2618)
    parser.add_argument("--initial-water-distance", type=float, default=4.0)
    parser.add_argument("--late-fraction", type=float, default=0.5)
    parser.add_argument("--shell-cutoff", type=float, default=3.5)
    parser.add_argument("--write-trajectories", action="store_true")
    args = parser.parse_args(argv)

    payload = collect_convergence(
        args.output_dir,
        n_waters_list=_parse_int_csv(args.n_waters),
        n_mc_steps=args.n_mc_steps,
        random_seed=args.random_seed,
        temperature=args.temperature,
        translation_step=args.translation_step,
        rotation_step=args.rotation_step,
        initial_water_distance=args.initial_water_distance,
        late_fraction=args.late_fraction,
        shell_cutoff=args.shell_cutoff,
        write_trajectories=args.write_trajectories,
        runner=runner,
    )
    print(f"wrote {payload['outputs']['csv']}")
    print(f"wrote {payload['outputs']['json']}")
    for path in payload["outputs"]["trajectories"]:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
