#!/usr/bin/env python3
"""Run fixed-MO H2O/STO-3G SQD with asymmetric point charges.

Coordinates are Angstrom, charges are elementary charges, and total energies
are Hartree. The (4e, 4o) active space uses 8 system qubits and freezes three
occupied orbitals. This example saves the complete public result for each mode
and a lossless ``sqd`` namespace for downstream integral-based integrations.
It does not run an MC loop, RDM feedback, or self-consistent MM polarization.

Run with the SQD extra installed and OMP/MKL/OPENBLAS_NUM_THREADS=1::

    python -m examples.qmmm.h2o_sqd_validation --output data/output/examples/h2o-sqd

The sparse sampled space is not expected to reproduce the full-space reference.
Errors are signed SQD-minus-reference in mHa; positive means SQD is higher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def run_validation(output_dir: Path, *, shots: int = 4096, seed: int = 7) -> dict[str, Any]:
    """Execute both embedding modes and save complete, honest result records.

    Args:
        output_dir: Directory for public JSON results and downstream mapping examples.
        shots: Positive sample count within the installed calibrated resource domain.
        seed: Nonnegative random seed shared by both runs.

    Returns:
        Input metadata, signed errors, capability limits, and output filenames.

    Raises:
        ValueError: Invalid input or mismatched orbital frames.
        RuntimeError: An underlying numerical or resource check fails.
    """
    from q2m3.sqd.config import LUCJConfig, ReferenceConfig
    from q2m3.sqd.orchestrator import run_sqd
    from q2m3.utils.io import save_json_results

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates = np.array([[0.0, 0.0, 0.0], [0.13, 0.1, 0.96], [0.88, -0.12, -0.29]])
    mm_coords = np.array([[2.3, 0.4, 1.2], [-1.7, 2.1, 0.3]])
    mm_charges = np.array([0.31, -0.19])
    lucj = LUCJConfig(n_reps=1, shots=shots)
    reference = ReferenceConfig(wall_budget_s=120, rss_budget_mb=8192)
    report: dict[str, Any] = {
        "symbols": ["O", "H", "H"],
        "coordinates_angstrom": coordinates.tolist(),
        "basis": "sto-3g",
        "charge": 0,
        "active_space": [4, 4],
        "system_qubits": 8,
        "active_indices": [3, 4, 5, 6],
        "n_core_orbitals": 3,
        "mm_charges_e": mm_charges.tolist(),
        "mm_coords_angstrom": mm_coords.tolist(),
        "mm_source": "synthetic asymmetric point charges; not a sampled solvent trajectory",
        "seed": seed,
        "shots": shots,
        "n_reps": 1,
        "energy_unit": "Ha",
        "difference_unit": "mHa",
        "budget_per_entry": {"wall_s": reference.wall_budget_s, "rss_mb": reference.rss_budget_mb},
        "capabilities": {
            "fixed_mo_point_charges": True,
            "mc_loop": False,
            "rdm_feedback": False,
            "self_consistent_mm_polarization": False,
        },
        "resource_scope": "embedded measurements end at start of final result construction; "
        "use external process-tree monitoring for complete-run RSS",
        "runs": {},
    }
    frame_id = None
    for mode in ("diagonal", "full_oneelectron"):
        result = run_sqd(
            report["symbols"],
            coordinates,
            active_electrons=4,
            active_orbitals=4,
            charge=0,
            basis="sto-3g",
            mm_charges=mm_charges,
            mm_coords=mm_coords,
            embedding_mode=mode,
            lucj=lucj,
            reference=reference,
            seed=seed,
            verbose=False,
        )
        current_frame = result.provenance["context"]["frame_id"]
        if frame_id is not None and current_frame != frame_id:
            raise ValueError("Embedding modes did not reproduce the same vacuum MO frame")
        frame_id = current_frame
        public_path = output_dir / f"{mode}.json"
        save_json_results(result, public_path)
        payload = json.loads(public_path.read_text())
        # Preserve all 40 fields, including null reasons, attempts, warnings and versions.
        save_json_results({"sqd": payload}, output_dir / f"{mode}-downstream.json")
        report["runs"][mode] = {
            "sqd_energy_ha": payload["sqd_energy"],
            "baseline_energy_ha": payload["baseline_energy"],
            "baseline_tier": payload["baseline_tier"],
            "delta_mHa": payload["delta_mHa"],
            "subspace_dims": payload["subspace_dims"],
            "full_ci_dim": payload["full_ci_dim"],
            "frame_id": current_frame,
            "hamiltonian_id": payload["provenance"]["context"]["hamiltonian_id"],
            "result_file": public_path.name,
            "downstream_file": f"{mode}-downstream.json",
        }
    report["full_minus_diagonal_baseline_mHa"] = 1000 * (
        report["runs"]["full_oneelectron"]["baseline_energy_ha"]
        - report["runs"]["diagonal"]["baseline_energy_ha"]
    )
    save_json_results(report, output_dir / "report.json")
    return report


def main() -> int:
    """Parse arguments and run the two-mode water example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    from examples.sqd._presentation import (
        execute,
        print_result,
        save_summary,
        summary_row,
    )

    def calculation(args, directory):
        report = run_validation(directory, shots=args.shots, seed=args.seed)
        rows = []
        for mode in report["runs"]:
            payload = json.loads((directory / f"{mode}.json").read_text())
            print_result(payload, mode)
            rows.append(summary_row(payload, mode, f"{mode}.json"))
        save_summary(directory, rows)
        print(
            "Full-minus-diagonal reference shift: "
            f"{report['full_minus_diagonal_baseline_mHa']:.6f} mHa. "
            "This compares two fixed-frame Hamiltonians, to the quality of their references; "
            "SQD-minus-reference residuals above describe solver error. "
            "The asymmetric fixed charges are artificial, not a solvation free energy."
        )

    return execute(args, "h2o-embedding", calculation)


if __name__ == "__main__":
    raise SystemExit(main())
