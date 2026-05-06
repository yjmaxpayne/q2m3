#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Collect the canonical H2 depth-5 replay dataset for shell analysis.

The workflow is:
1. Run one dynamic MC trajectory to obtain the accepted solvent path.
2. Replay fixed and dynamic quantum evaluations on that same saved path.
3. Write the accepted-state trajectory CSV plus a paired fixed/dynamic delta CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from q2m3.solvation import (
    MoleculeConfig,
    QPEConfig,
    SolvationConfig,
    replay_quantum_trajectory,
    run_solvation,
)
from q2m3.solvation.structure_analysis import write_state_trajectory_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_TRAJECTORY_CSV = OUTPUT_DIR / "h2_trotter5_dynamic_mc_trajectory.csv"
DEFAULT_DELTA_CSV = OUTPUT_DIR / "h2_three_mode_delta_corr_pol.csv"
DEFAULT_SUMMARY_JSON = OUTPUT_DIR / "h2_depth5_replay_dataset.json"

H2_MOLECULE = MoleculeConfig(
    name="H2",
    symbols=["H", "H"],
    coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    charge=0,
    active_electrons=2,
    active_orbitals=2,
    basis="sto-3g",
)


def _base_config(
    *,
    mode: str,
    n_mc_steps: int,
    n_waters: int,
    random_seed: int,
    temperature: float,
    translation_step: float,
    rotation_step: float,
    initial_water_distance: float,
) -> SolvationConfig:
    """Build the canonical H2 depth-5 QPE/MM configuration."""
    return SolvationConfig(
        molecule=H2_MOLECULE,
        qpe_config=QPEConfig(
            n_estimation_wires=8,
            n_trotter_steps=5,
            n_shots=0,
            qpe_interval=1,
            target_resolution=0.0002,
            energy_range=0.2,
        ),
        hamiltonian_mode=mode,  # type: ignore[arg-type]
        n_waters=n_waters,
        n_mc_steps=n_mc_steps,
        temperature=temperature,
        translation_step=translation_step,
        rotation_step=rotation_step,
        initial_water_distance=initial_water_distance,
        random_seed=random_seed,
        verbose=False,
    )


def _write_delta_csv(
    path: Path,
    *,
    fixed: np.ndarray,
    dynamic: np.ndarray,
    hf_dynamic: np.ndarray,
) -> None:
    """Write paired fixed/dynamic replay energies and delta-corr-pol."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "fixed_energy_ha",
                "dynamic_energy_ha",
                "hf_dynamic_energy_ha",
                "delta_corr_pol_ha",
            ],
        )
        writer.writeheader()
        for step in range(len(fixed)):
            writer.writerow(
                {
                    "step": step,
                    "fixed_energy_ha": f"{fixed[step]:.12g}",
                    "dynamic_energy_ha": f"{dynamic[step]:.12g}",
                    "hf_dynamic_energy_ha": f"{hf_dynamic[step]:.12g}",
                    "delta_corr_pol_ha": f"{(dynamic[step] - fixed[step]):.12g}",
                }
            )


def collect_dataset(
    *,
    n_mc_steps: int,
    n_waters: int,
    random_seed: int,
    temperature: float,
    translation_step: float,
    rotation_step: float,
    initial_water_distance: float,
    trajectory_csv: Path,
    delta_csv: Path,
    summary_json: Path,
) -> dict[str, Any]:
    """Run the dynamic trajectory and matched-path fixed/dynamic replays."""
    dynamic_mc_config = _base_config(
        mode="dynamic",
        n_mc_steps=n_mc_steps,
        n_waters=n_waters,
        random_seed=random_seed,
        temperature=temperature,
        translation_step=translation_step,
        rotation_step=rotation_step,
        initial_water_distance=initial_water_distance,
    )
    dynamic_mc = run_solvation(dynamic_mc_config, show_plots=False)
    trajectory = np.asarray(dynamic_mc["trajectory_solvent_states"], dtype=float)
    write_state_trajectory_csv(trajectory_csv, trajectory)

    fixed_replay = replay_quantum_trajectory(
        _base_config(
            mode="fixed",
            n_mc_steps=n_mc_steps,
            n_waters=n_waters,
            random_seed=random_seed,
            temperature=temperature,
            translation_step=translation_step,
            rotation_step=rotation_step,
            initial_water_distance=initial_water_distance,
        ),
        trajectory,
    )
    dynamic_replay = replay_quantum_trajectory(dynamic_mc_config, trajectory)

    fixed_q = np.asarray(fixed_replay["quantum_energies"], dtype=float)
    dynamic_q = np.asarray(dynamic_replay["quantum_energies"], dtype=float)
    hf_dynamic = np.asarray(dynamic_replay["hf_energies"], dtype=float)
    _write_delta_csv(delta_csv, fixed=fixed_q, dynamic=dynamic_q, hf_dynamic=hf_dynamic)

    payload = {
        "metadata": {
            "schema": "h2_depth5_replay_dataset.v1",
            "n_estimation_wires": 8,
            "n_trotter_steps": 5,
            "n_shots": 0,
            "mode": "dynamic accepted-path + fixed/dynamic replay",
        },
        "n_mc_steps": int(n_mc_steps),
        "n_waters": int(n_waters),
        "acceptance_rate": float(dynamic_mc["acceptance_rate"]),
        "delta_mean_kcal_mol": float(np.mean(dynamic_q - fixed_q) * 627.509),
        "trajectory_csv": str(trajectory_csv),
        "delta_csv": str(delta_csv),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    payload["summary_json"] = str(summary_json)
    return payload


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-mc-steps", type=int, default=1000)
    parser.add_argument("--n-waters", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--translation-step", type=float, default=0.3)
    parser.add_argument("--rotation-step", type=float, default=0.2618)
    parser.add_argument("--initial-water-distance", type=float, default=4.0)
    parser.add_argument("--trajectory-csv", type=Path, default=DEFAULT_TRAJECTORY_CSV)
    parser.add_argument("--delta-csv", type=Path, default=DEFAULT_DELTA_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    args = parser.parse_args(argv)

    payload = collect_dataset(
        n_mc_steps=args.n_mc_steps,
        n_waters=args.n_waters,
        random_seed=args.random_seed,
        temperature=args.temperature,
        translation_step=args.translation_step,
        rotation_step=args.rotation_step,
        initial_water_distance=args.initial_water_distance,
        trajectory_csv=args.trajectory_csv,
        delta_csv=args.delta_csv,
        summary_json=args.summary_json,
    )
    print(f"wrote {payload['trajectory_csv']}")
    print(f"wrote {payload['delta_csv']}")
    print(f"wrote {payload['summary_json']}")


if __name__ == "__main__":
    main()
