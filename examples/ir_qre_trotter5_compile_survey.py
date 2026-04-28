#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Measure standardized IR-QRE compile rows with n_trotter=5.

The output is the single measured compile source consumed by
``examples/ir_qre_correlation_analysis.py``. Each successful row uses the same
4-bit estimation register and 5 Trotter steps so that compile-side data can be
joined to QRE rows without mixing incompatible circuit shapes.

Usage:
    OMP_NUM_THREADS=2 uv run python examples/ir_qre_trotter5_compile_survey.py
    OMP_NUM_THREADS=2 uv run python examples/ir_qre_trotter5_compile_survey.py --systems H2,HeH+
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import sys
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.resource_estimation_survey import survey_systems  # noqa: E402
from q2m3.molecule import MoleculeConfig  # noqa: E402
from q2m3.profiling import ParentSideMonitor, ProfileResult, run_single_profile  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_OUTPUT_JSON = OUTPUT_DIR / "ir_qre_trotter5_compile_survey.json"
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "ir_qre_trotter5_compile_survey.csv"
STANDARD_N_ESTIMATION_WIRES = 4
STANDARD_N_TROTTER = 5
DEFAULT_SYSTEM_LABELS = [
    "H2",
    "HeH+",
    "H3+",
    "H4 linear",
    "LiH",
    "H2O (4e,4o)",
    "H3O+ STO-3G",
]
DEFAULT_MODES = ["fixed", "dynamic"]


def systems_by_label() -> dict[str, MoleculeConfig]:
    """Return profiling molecule configs keyed by survey label."""
    systems: dict[str, MoleculeConfig] = {}
    for spec in survey_systems():
        systems[spec.label] = MoleculeConfig(
            name=spec.label,
            symbols=spec.symbols,
            coords=spec.coords.tolist(),
            charge=spec.charge,
            active_electrons=spec.active_electrons,
            active_orbitals=spec.active_orbitals,
            basis=spec.basis,
        )
    return systems


def _split_csv_arg(value: str) -> list[str]:
    """Split a comma-separated CLI value while preserving labels with spaces."""
    return [part.strip() for part in value.split(",") if part.strip()]


def _snapshot_to_dict(snapshot: Any) -> dict[str, float] | None:
    """Convert a MemorySnapshot-like object to a JSON-serializable dict."""
    if snapshot is None:
        return None
    if hasattr(snapshot, "__dataclass_fields__"):
        return asdict(snapshot)
    keys = [
        "label",
        "rss_mb",
        "vm_peak_mb",
        "maxrss_mb",
        "maxrss_children_mb",
        "tracemalloc_peak_mb",
        "tracemalloc_current_mb",
        "elapsed_s",
    ]
    return {key: getattr(snapshot, key) for key in keys if hasattr(snapshot, key)}


def _phase_b_total_mb(result: ProfileResult, parent_data: dict[str, Any]) -> float:
    """Return the best available compile-phase peak RSS estimate in MB."""
    phase_b = result.phase_b
    phase_b_total = 0.0
    if phase_b is not None:
        phase_b_total = phase_b.maxrss_mb + phase_b.maxrss_children_mb
    parent_peak = max(
        float(parent_data.get("peak_hwm_mb", 0.0) or 0.0),
        float(parent_data.get("peak_rss_mb", 0.0) or 0.0),
    )
    return max(phase_b_total, parent_peak, float(result.timeline_peak_mb or 0.0))


def _stage_size(ir_analysis: list[tuple[str, float, int]], stage_name: str) -> float | None:
    """Return IR size in KB for a named Catalyst stage."""
    for stage, size_kb, _lines in ir_analysis:
        if stage == stage_name:
            return float(size_kb)
    return None


def bufferization_amplification(ir_analysis: list[tuple[str, float, int]]) -> float | None:
    """Return BufferizationStage size divided by its previous stage size."""
    buffer_size = _stage_size(ir_analysis, "BufferizationStage")
    if buffer_size is None:
        return None
    previous_size = _stage_size(ir_analysis, "HLOLoweringStage")
    if previous_size is None or previous_size <= 0.0:
        return None
    return round(buffer_size / previous_size, 3)


def _error_record(label: str, result: ProfileResult, parent_data: dict[str, Any]) -> dict[str, Any]:
    """Convert a failed profiling result into a provenance-preserving row."""
    return {
        "label": label,
        "classical_case": f"{label} {result.mode} 4-bit 5-trotter",
        "classical_mode": result.mode,
        "classical_status": "failed",
        "error": result.error,
        "n_system_qubits": result.n_system_qubits,
        "n_estimation_wires": result.n_estimation_wires,
        "total_qubits": result.n_system_qubits + result.n_estimation_wires,
        "n_trotter": result.n_trotter,
        "n_terms": result.n_terms,
        "ir_ops_lower_bound": result.ir_scale,
        "compile_rss_gb": None,
        "compile_time_s": None,
        "bufferization_amp": None,
        "prob_sum": result.prob_sum,
        "parent_peak_rss_mb": parent_data.get("peak_rss_mb"),
        "parent_peak_hwm_mb": parent_data.get("peak_hwm_mb"),
    }


def profile_result_to_record(
    label: str,
    result: ProfileResult,
    parent_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize one ProfileResult into the trotter-5 compile schema."""
    parent_data = parent_data or {}
    if result.error:
        return _error_record(label, result, parent_data)
    if result.n_trotter != STANDARD_N_TROTTER:
        raise ValueError(f"{label} produced n_trotter={result.n_trotter}; expected 5")
    if result.n_estimation_wires != STANDARD_N_ESTIMATION_WIRES:
        raise ValueError(
            f"{label} produced n_estimation_wires={result.n_estimation_wires}; expected 4"
        )
    if result.phase_b is None:
        raise ValueError(f"{label} has no compile-phase measurement")

    compile_rss_mb = _phase_b_total_mb(result, parent_data)
    return {
        "label": label,
        "classical_case": f"{label} {result.mode} 4-bit 5-trotter",
        "classical_mode": result.mode,
        "classical_status": "measured",
        "n_system_qubits": result.n_system_qubits,
        "n_estimation_wires": result.n_estimation_wires,
        "total_qubits": result.n_system_qubits + result.n_estimation_wires,
        "n_trotter": result.n_trotter,
        "n_terms": result.n_terms,
        "ir_ops_lower_bound": result.ir_scale,
        "compile_rss_gb": round(compile_rss_mb / 1024.0, 4),
        "compile_time_s": round(result.phase_b.elapsed_s, 3),
        "bufferization_amp": bufferization_amplification(result.ir_analysis),
        "prob_sum": float(result.prob_sum),
        "phase_a": _snapshot_to_dict(result.phase_a),
        "phase_b": _snapshot_to_dict(result.phase_b),
        "phase_c": _snapshot_to_dict(result.phase_c),
        "timeline_peak_mb": result.timeline_peak_mb,
        "parent_peak_rss_mb": parent_data.get("peak_rss_mb"),
        "parent_peak_hwm_mb": parent_data.get("peak_hwm_mb"),
        "ir_analysis": [list(item) for item in result.ir_analysis],
    }


def _profile_worker(
    mol: MoleculeConfig,
    n_est: int,
    n_trotter: int,
    mode: str,
    queue: multiprocessing.Queue,
) -> None:
    """Run one profiling pass in a subprocess and send the result to a queue."""
    try:
        tracemalloc.start()
        result = run_single_profile(mol, n_est=n_est, n_trotter=n_trotter, mode=mode)
        tracemalloc.stop()
        queue.put(result)
    except Exception as exc:  # noqa: BLE001 - preserve failure in output JSON.
        queue.put(
            ProfileResult(
                molecule=mol.name,
                n_system_qubits=mol.active_orbitals * 2,
                n_estimation_wires=n_est,
                n_trotter=n_trotter,
                n_terms=0,
                ir_scale=0,
                mode=mode,
                error=f"{type(exc).__name__}: {exc}",
            )
        )


def run_profile_in_subprocess(
    mol: MoleculeConfig,
    n_est: int,
    n_trotter: int,
    mode: str,
    timeout_s: int,
) -> tuple[ProfileResult, dict[str, Any]]:
    """Run one compile profile in a monitored subprocess."""
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_profile_worker, args=(mol, n_est, n_trotter, mode, queue))
    proc.start()

    monitor = ParentSideMonitor(proc.pid, interval_s=0.1)
    monitor.start()
    proc.join(timeout=timeout_s)
    monitor.stop()

    parent_data = {
        "peak_rss_mb": monitor.peak_rss_mb,
        "peak_hwm_mb": monitor.peak_hwm_mb,
        "peak_smaps": monitor.peak_smaps,
        "n_samples": len(monitor.samples),
    }
    if proc.is_alive():
        proc.kill()
        proc.join()
        return (
            ProfileResult(
                molecule=mol.name,
                n_system_qubits=mol.active_orbitals * 2,
                n_estimation_wires=n_est,
                n_trotter=n_trotter,
                n_terms=0,
                ir_scale=0,
                mode=mode,
                error=f"timeout after {timeout_s}s",
            ),
            parent_data,
        )
    if not queue.empty():
        return queue.get(), parent_data
    return (
        ProfileResult(
            molecule=mol.name,
            n_system_qubits=mol.active_orbitals * 2,
            n_estimation_wires=n_est,
            n_trotter=n_trotter,
            n_terms=0,
            ir_scale=0,
            mode=mode,
            error="no result from subprocess",
        ),
        parent_data,
    )


def write_outputs(records: list[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    """Write compile records to JSON and CSV."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "schema": "ir_qre_trotter5_compile_survey.v1",
            "n_estimation_wires": STANDARD_N_ESTIMATION_WIRES,
            "n_trotter": STANDARD_N_TROTTER,
            "n_trotter_steps": STANDARD_N_TROTTER,
            "status": "measured rows only; failed rows are retained for provenance",
        },
        "systems": records,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")

    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames and key not in {
                "phase_a",
                "phase_b",
                "phase_c",
                "ir_analysis",
            }:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["label", "classical_case", "classical_status"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def load_existing_records(json_path: Path) -> list[dict[str, Any]]:
    """Load existing standardized output so subset reruns append safely."""
    if not json_path.exists():
        return []
    with json_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("systems", [])
    for record in records:
        if record.get("n_trotter") != STANDARD_N_TROTTER:
            label = record.get("label", "<unknown>")
            raise ValueError(f"{json_path}:{label} must use n_trotter=5")
    return records


def upsert_record(
    records: list[dict[str, Any]], new_record: dict[str, Any]
) -> list[dict[str, Any]]:
    """Replace the same system/mode/case or append a new measured row."""
    key = (new_record.get("label"), new_record.get("classical_case"))
    updated: list[dict[str, Any]] = []
    replaced = False
    for record in records:
        if (record.get("label"), record.get("classical_case")) == key:
            updated.append(new_record)
            replaced = True
        else:
            updated.append(record)
    if not replaced:
        updated.append(new_record)
    return updated


Runner = Callable[[MoleculeConfig, int, int, str, int], tuple[ProfileResult, dict[str, Any]]]


def main(argv: list[str] | None = None, runner: Runner | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        default=",".join(DEFAULT_SYSTEM_LABELS),
        help="Comma-separated survey labels to measure.",
    )
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated modes: fixed,dynamic.",
    )
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args(argv)

    systems = systems_by_label()
    labels = _split_csv_arg(args.systems)
    modes = _split_csv_arg(args.modes)
    unknown_labels = [label for label in labels if label not in systems]
    if unknown_labels:
        raise ValueError(f"unknown systems: {', '.join(unknown_labels)}")
    invalid_modes = [mode for mode in modes if mode not in {"fixed", "dynamic"}]
    if invalid_modes:
        raise ValueError(f"unknown modes: {', '.join(invalid_modes)}")

    runner = runner or run_profile_in_subprocess
    records = load_existing_records(args.output_json)
    write_outputs(records, args.output_json, args.output_csv)

    for label in labels:
        mol = systems[label]
        for mode in modes:
            print(f"[*] Profiling {label} {mode} 4-bit 5-trotter", flush=True)
            result, parent_data = runner(
                mol, STANDARD_N_ESTIMATION_WIRES, STANDARD_N_TROTTER, mode, args.timeout_s
            )
            record = profile_result_to_record(label, result, parent_data)
            records = upsert_record(records, record)
            write_outputs(records, args.output_json, args.output_csv)
            if record["classical_status"] == "measured":
                print(
                    f"    OK: compile={record['compile_time_s']}s, "
                    f"rss={record['compile_rss_gb']}GB, terms={record['n_terms']}",
                    flush=True,
                )
            else:
                print(f"    FAILED: {record.get('error')}", flush=True)

    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
