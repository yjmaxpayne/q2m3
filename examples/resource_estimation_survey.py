#!/usr/bin/env python3
# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
EFTQC Resource Estimation Survey: 5-system matrix.

Quantifies fault-tolerant quantum computing resource requirements (Toffoli
gates, logical qubits, T-depth, Hamiltonian 1-norm, runtime estimate) across
a panel of molecular systems specified in the closing-plan §03b task 1
(W1-W2 deliverable).

System matrix:
    H2          (2e,2o)  STO-3G  — vacuum baseline
    HeH+        (2e,2o)  STO-3G  — smallest charged heteronuclear control
    H3+         (2e,3o)  STO-3G  — small cationic 6-qubit bridge system
    H4 linear   (4e,4o)  STO-3G  — hydrogen-chain 8-qubit bridge system
    LiH         (4e,4o)  STO-3G  — standard small-molecule benchmark
    H2O (4e,4o) STO-3G  — reduced water active-space compile benchmark
    H3O+        (4e,4o)  STO-3G  — current POC chemistry
    H3O+        (4e,4o)  6-31G   — basis-set scaling probe
    H2O         (8e,6o)  STO-3G  — NH3-scale closed-shell solvent molecule
    CH4         (8e,7o)  STO-3G  — NH3-scale saturated neutral molecule
    NH3         (8e,7o)  STO-3G  — neutral closed-shell, 14 system qubits
    NH4+        (8e,7o)  STO-3G  — NH3-scale cationic tetrahedral molecule
    Formamide   (8e,8o)  STO-3G  — smaller-than-glycine organic fragment
    Glycine     (6e,6o)  STO-3G  — biomolecular fragment, SQD comparison

Outputs (under data/output/):
    qre_survey.csv
    qre_survey.json

Geometry sources are typical equilibrium structures from CCCBDB / PubChem;
QRE results are dominated by the 2-electron integral structure, so few-pm
geometric uncertainty has negligible effect on the reported counts.

Usage:
    OMP_NUM_THREADS=4 uv run python examples/resource_estimation_survey.py
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from q2m3.core import (
    derive_t_resources,
    estimate_eftqc_runtime,
    estimate_resources,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "output"
DEFAULT_TARGET_ERROR = 0.0016  # Hartree, ~1 kcal/mol (chemical accuracy)
DEFAULT_TOFFOLI_CYCLE_US = 1.0  # us per Toffoli, common EFTQC assumption


@dataclass(frozen=True)
class SystemSpec:
    """Specification for one row in the survey matrix."""

    label: str
    symbols: list[str]
    coords: np.ndarray
    charge: int
    basis: str
    active_electrons: int
    active_orbitals: int
    geometry_source: str
    skip_reason: str = ""


def survey_systems() -> list[SystemSpec]:
    """Return the 5-system matrix from closing-plan §03b task 1."""
    h3op_coords = np.array(
        [
            [0.0000, 0.0000, 0.1173],
            [0.0000, 0.9572, -0.4692],
            [0.8286, -0.4786, -0.4692],
            [-0.8286, -0.4786, -0.4692],
        ]
    )
    return [
        SystemSpec(
            label="H2",
            symbols=["H", "H"],
            coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
            charge=0,
            basis="sto-3g",
            active_electrons=2,
            active_orbitals=2,
            geometry_source="HF/STO-3G equilibrium, R=0.74 A",
        ),
        SystemSpec(
            label="HeH+",
            symbols=["He", "H"],
            coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.774]]),
            charge=1,
            basis="sto-3g",
            active_electrons=2,
            active_orbitals=2,
            geometry_source="near-equilibrium HeH+ control, R=0.774 A",
        ),
        SystemSpec(
            label="H3+",
            symbols=["H", "H", "H"],
            coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.9], [0.779, 0.0, 0.45]]),
            charge=1,
            basis="sto-3g",
            active_electrons=2,
            active_orbitals=3,
            geometry_source="equilateral-like H3+ bridge geometry, H-H about 0.9 A",
        ),
        SystemSpec(
            label="H4 linear",
            symbols=["H", "H", "H", "H"],
            coords=np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74], [0.0, 0.0, 1.48], [0.0, 0.0, 2.22]]
            ),
            charge=0,
            basis="sto-3g",
            active_electrons=4,
            active_orbitals=4,
            geometry_source="linear H4 chain, adjacent H-H=0.74 A",
        ),
        SystemSpec(
            label="LiH",
            symbols=["Li", "H"],
            coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.60]]),
            charge=0,
            basis="sto-3g",
            active_electrons=4,
            active_orbitals=4,
            geometry_source="near-equilibrium LiH small-molecule benchmark, R=1.60 A",
        ),
        SystemSpec(
            label="H2O (4e,4o)",
            symbols=["O", "H", "H"],
            coords=np.array(
                [
                    [0.000, 0.000, 0.000],
                    [0.958, 0.000, 0.000],
                    [-0.240, 0.927, 0.000],
                ]
            ),
            charge=0,
            basis="sto-3g",
            active_electrons=4,
            active_orbitals=4,
            geometry_source="gas-phase water reduced active-space compile benchmark",
        ),
        SystemSpec(
            label="H3O+ STO-3G",
            symbols=["O", "H", "H", "H"],
            coords=h3op_coords,
            charge=1,
            basis="sto-3g",
            active_electrons=4,
            active_orbitals=4,
            geometry_source="pyramidal C3v H3O+ geometry used by h3o_mc_solvation.py",
        ),
        SystemSpec(
            label="H3O+ 6-31G",
            symbols=["O", "H", "H", "H"],
            coords=h3op_coords,
            charge=1,
            basis="6-31g",
            active_electrons=4,
            active_orbitals=4,
            geometry_source="pyramidal C3v H3O+ geometry used by h3o_mc_solvation.py",
        ),
        SystemSpec(
            label="H2O",
            symbols=["O", "H", "H"],
            coords=np.array(
                [
                    [0.000, 0.000, 0.000],
                    [0.958, 0.000, 0.000],
                    [-0.240, 0.927, 0.000],
                ]
            ),
            charge=0,
            basis="sto-3g",
            active_electrons=8,
            active_orbitals=6,
            geometry_source="gas-phase equilibrium, O-H=0.958 A, angle=104.5 deg",
        ),
        SystemSpec(
            label="CH4",
            symbols=["C", "H", "H", "H", "H"],
            coords=np.array(
                [
                    [0.000, 0.000, 0.000],
                    [0.629, 0.629, 0.629],
                    [-0.629, -0.629, 0.629],
                    [-0.629, 0.629, -0.629],
                    [0.629, -0.629, -0.629],
                ]
            ),
            charge=0,
            basis="sto-3g",
            active_electrons=8,
            active_orbitals=7,
            geometry_source="tetrahedral methane, C-H=1.089 A",
        ),
        SystemSpec(
            label="NH3",
            symbols=["N", "H", "H", "H"],
            coords=np.array(
                [
                    [0.000, 0.000, 0.149],
                    [0.939, 0.000, -0.347],
                    [-0.470, 0.813, -0.347],
                    [-0.470, -0.813, -0.347],
                ]
            ),
            charge=0,
            basis="sto-3g",
            active_electrons=8,
            active_orbitals=7,
            geometry_source="CCCBDB experimental, N-H=1.012 A, ang=106.7 deg",
        ),
        SystemSpec(
            label="NH4+",
            symbols=["N", "H", "H", "H", "H"],
            coords=np.array(
                [
                    [0.000, 0.000, 0.000],
                    [0.641, 0.641, 0.641],
                    [-0.641, -0.641, 0.641],
                    [-0.641, 0.641, -0.641],
                    [0.641, -0.641, -0.641],
                ]
            ),
            charge=1,
            basis="sto-3g",
            active_electrons=8,
            active_orbitals=7,
            geometry_source="tetrahedral ammonium, N-H=1.110 A",
        ),
        SystemSpec(
            label="Formamide",
            symbols=["C", "O", "N", "H", "H", "H"],
            coords=np.array(
                [
                    [0.0000, 0.0000, 0.0000],
                    [-1.2170, 0.5560, 0.0000],
                    [1.1190, 0.6580, 0.0000],
                    [0.0000, -1.0930, 0.0000],
                    [1.0430, 1.6420, 0.0000],
                    [2.0280, 0.1620, 0.0000],
                ]
            ),
            charge=0,
            basis="sto-3g",
            active_electrons=8,
            active_orbitals=8,
            geometry_source="planar formamide HCONH2 fragment; smaller-than-glycine organic control",
        ),
        SystemSpec(
            label="Glycine",
            symbols=["N", "H", "H", "C", "H", "H", "C", "O", "O", "H"],
            coords=np.array(
                [
                    [-1.870, 0.231, 0.000],
                    [-2.286, 0.764, 0.769],
                    [-2.286, 0.764, -0.769],
                    [-0.421, 0.137, 0.000],
                    [-0.060, -0.405, 0.880],
                    [-0.060, -0.405, -0.880],
                    [0.282, 1.519, 0.000],
                    [-0.350, 2.563, 0.000],
                    [1.620, 1.430, 0.000],
                    [1.978, 2.313, 0.000],
                ]
            ),
            charge=0,
            basis="sto-3g",
            active_electrons=6,
            active_orbitals=6,
            geometry_source="PubChem CID 750 (planar approximation)",
            # PennyLane's active-space integral path materializes the full 30^4
            # 2-electron tensor before truncation; observed wallclock > 23 min
            # on a 4-thread CPU. Re-enable once a frozen-core integral path
            # avoids the full tensor allocation.
            skip_reason="wallclock > 20 min (PennyLane qchem materializes full 30^4 2e tensor)",
        ),
    ]


def estimate_one(spec: SystemSpec) -> dict:
    """Run the full QRE pipeline for one system and return a flat record."""
    t0 = time.perf_counter()
    res = estimate_resources(
        symbols=spec.symbols,
        coords=spec.coords,
        charge=spec.charge,
        basis=spec.basis,
        active_electrons=spec.active_electrons,
        active_orbitals=spec.active_orbitals,
        target_error=DEFAULT_TARGET_ERROR,
    )
    elapsed = time.perf_counter() - t0

    # Derived T-resources and runtime estimate
    t_res = derive_t_resources(toffoli_gates=res.toffoli_gates)
    qpe_iters = int(np.ceil(res.hamiltonian_1norm / DEFAULT_TARGET_ERROR))
    runtime = estimate_eftqc_runtime(
        qpe_iterations=qpe_iters,
        toffoli_gates=res.toffoli_gates,
        toffoli_cycle_microseconds=DEFAULT_TOFFOLI_CYCLE_US,
    )

    return {
        "label": spec.label,
        "basis": res.basis,
        "charge": spec.charge,
        "active_electrons": spec.active_electrons,
        "active_orbitals": spec.active_orbitals,
        "n_system_qubits": res.n_system_qubits,
        "logical_qubits": res.logical_qubits,
        "toffoli_gates": res.toffoli_gates,
        "t_count": t_res["t_count"],
        "t_depth": t_res["t_depth"],
        "hamiltonian_1norm_Ha": res.hamiltonian_1norm,
        "target_error_Ha": res.target_error,
        "qpe_iterations": qpe_iters,
        "runtime_seconds": runtime["runtime_seconds"],
        "runtime_hours": runtime["runtime_hours"],
        "runtime_days": runtime["runtime_days"],
        "toffoli_cycle_us": runtime["toffoli_cycle_microseconds"],
        "geometry_source": spec.geometry_source,
        "estimation_wallclock_s": round(elapsed, 3),
    }


def write_outputs(records: list[dict], output_dir: Path) -> tuple[Path, Path]:
    """Persist records to CSV and JSON in output_dir; return both paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "qre_survey.csv"
    json_path = output_dir / "qre_survey.json"

    # Union all keys to keep CSV header consistent across success/skip rows
    fieldnames: list[str] = []
    for rec in records:
        for key in rec:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    payload = {
        "metadata": {
            "tool": "pennylane.estimator.DoubleFactorization",
            "target_error_Hartree": DEFAULT_TARGET_ERROR,
            "toffoli_cycle_us": DEFAULT_TOFFOLI_CYCLE_US,
            "t_count_formula": "7 * Toffoli (Nielsen-Chuang)",
            "t_depth_formula": "7 * Toffoli (sequential upper bound)",
            "active_space_path": "qml.qchem.electron_integrals(mol, core=, active=)",
        },
        "systems": records,
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2, default=float)
    return csv_path, json_path


def print_summary(records: list[dict]) -> None:
    """Render a fixed-width summary table to stdout."""
    print("=" * 100)
    print("  EFTQC Resource Estimation Survey")
    print("=" * 100)
    header = f"  {'System':<15} {'Basis':<8} {'AS':<8} {'Sys-q':<6} {'Logical-q':<10} {'Toffoli':<14} {'lambda(Ha)':<12} {'Runtime':<12}"
    print(header)
    print("  " + "-" * 96)
    for r in records:
        as_label = f"({r['active_electrons']}e,{r['active_orbitals']}o)"
        if r["toffoli_gates"] is None:
            print(
                f"  {r['label']:<15} {r['basis']:<8} {as_label:<8} "
                f"{'-':<6} {'-':<10} {'SKIPPED':<14} {'-':<12} "
                f"{r.get('error', '')[:40]:<40}"
            )
            continue
        runtime_str = (
            f"{r['runtime_seconds']:.2e}s"
            if r["runtime_seconds"] < 60
            else (
                f"{r['runtime_hours']:.2f}h"
                if r["runtime_hours"] < 48
                else f"{r['runtime_days']:.2f}d"
            )
        )
        print(
            f"  {r['label']:<15} {r['basis']:<8} {as_label:<8} "
            f"{r['n_system_qubits']:<6} {r['logical_qubits']:<10} "
            f"{r['toffoli_gates']:<14,} {r['hamiltonian_1norm_Ha']:<12.3f} "
            f"{runtime_str:<12}"
        )
    print("=" * 100)


def _failure_record(spec: SystemSpec, error: BaseException | str) -> dict:
    """Build a placeholder record for a system that failed estimation."""
    err = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    return {
        "label": spec.label,
        "basis": spec.basis,
        "charge": spec.charge,
        "active_electrons": spec.active_electrons,
        "active_orbitals": spec.active_orbitals,
        "n_system_qubits": None,
        "logical_qubits": None,
        "toffoli_gates": None,
        "t_count": None,
        "t_depth": None,
        "hamiltonian_1norm_Ha": None,
        "target_error_Ha": DEFAULT_TARGET_ERROR,
        "qpe_iterations": None,
        "runtime_seconds": None,
        "runtime_hours": None,
        "runtime_days": None,
        "toffoli_cycle_us": DEFAULT_TOFFOLI_CYCLE_US,
        "geometry_source": spec.geometry_source,
        "estimation_wallclock_s": None,
        "error": err,
    }


def main() -> None:
    specs = survey_systems()
    records: list[dict] = []
    for spec in specs:
        if spec.skip_reason:
            print(f"[*] Skipping: {spec.label} ({spec.skip_reason})", flush=True)
            records.append(_failure_record(spec, f"SKIPPED: {spec.skip_reason}"))
            write_outputs(records, OUTPUT_DIR)
            continue
        print(
            f"[*] Estimating: {spec.label} ({spec.basis}, "
            f"({spec.active_electrons}e,{spec.active_orbitals}o)) ...",
            flush=True,
        )
        try:
            rec = estimate_one(spec)
            records.append(rec)
            print(
                f"    OK in {rec['estimation_wallclock_s']}s "
                f"-> {rec['toffoli_gates']:,} Toffoli, "
                f"{rec['logical_qubits']} logical qubits",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 — record failure, continue survey
            records.append(_failure_record(spec, e))
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)

        # Incremental write so partial results survive interruption / timeout
        write_outputs(records, OUTPUT_DIR)

    csv_path, json_path = write_outputs(records, OUTPUT_DIR)
    print()
    print_summary(records)
    print()
    print(f"  Wrote: {csv_path}")
    print(f"  Wrote: {json_path}")


if __name__ == "__main__":
    main()
