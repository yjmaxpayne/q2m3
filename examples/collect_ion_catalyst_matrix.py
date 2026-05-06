#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Collect an ion/neutral QRE and PennyLane Catalyst compile matrix.

The collector joins existing QRE rows with fixed/dynamic Catalyst @qjit compile
records. It does not run Catalyst by default. Missing or memory-boundary systems
are preserved as explicit status rows so the figure can show the tested edge of
the compile-once/runtime-coefficient path.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_QRE_JSON = OUTPUT_DIR / "qre_survey.json"
DEFAULT_COMPILE_JSON = OUTPUT_DIR / "ir_qre_trotter5_compile_survey.json"
DEFAULT_H3O_BENCHMARK_JSON = OUTPUT_DIR / "h3o_8bit_qpe_benchmark.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "ion_catalyst_matrix"

TARGET_SYSTEMS = [
    {"label": "H2", "qre_label": "H2", "compile_label": "H2", "system_class": "neutral"},
    {"label": "HeH+", "qre_label": "HeH+", "compile_label": "HeH+", "system_class": "cation"},
    {"label": "H3+", "qre_label": "H3+", "compile_label": "H3+", "system_class": "cation"},
    {
        "label": "H3O+",
        "qre_label": "H3O+ STO-3G",
        "compile_label": "H3O+ STO-3G",
        "system_class": "cation",
    },
    {"label": "NH4+", "qre_label": "NH4+", "compile_label": "NH4+", "system_class": "cation"},
]
TIMING_MODES = ("fixed", "dynamic")
TIMING_LABELS_DEFAULT = "H2,HeH+,H3+"

TimingRunner = Callable[[str, int], dict[str, Any]]


def load_json_records(path: Path, key: str = "systems") -> list[dict[str, Any]]:
    """Load list records from a JSON payload, returning an empty list when absent."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get(key, [])
    if not isinstance(records, list):
        raise ValueError(f"{path}:{key} must be a list")
    return records


def _records_by_label(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the first record for each label."""
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        label = record.get("label")
        if label is not None and label not in out:
            out[str(label)] = record
    return out


def _compile_sort_key(record: dict[str, Any]) -> tuple[int, int, int]:
    """Prefer measured standardized 4-bit trotter-5 rows."""
    measured = 0 if record.get("classical_status") == "measured" else 1
    trotter5 = 0 if record.get("n_trotter") == 5 else 1
    est4 = 0 if record.get("n_estimation_wires") == 4 else 1
    return measured, trotter5, est4


def best_compile_row(
    compile_records: list[dict[str, Any]],
    *,
    label: str,
    mode: str,
) -> dict[str, Any] | None:
    """Return the best compile row for a system/mode pair."""
    matches = [
        record
        for record in compile_records
        if record.get("label") == label and record.get("classical_mode") == mode
    ]
    if not matches:
        return None
    return sorted(matches, key=_compile_sort_key)[0]


def _phase_c_per_eval_s(record: dict[str, Any] | None) -> float | None:
    """Extract per-evaluation execution time from phase C when available."""
    if not record:
        return None
    phase_c = record.get("phase_c")
    if not isinstance(phase_c, dict):
        return None
    elapsed = phase_c.get("elapsed_s")
    if elapsed is None:
        return None
    label = str(phase_c.get("label", ""))
    n_exec = 5
    if "(" in label and "x" in label:
        try:
            n_exec = int(label.split("(")[1].split("x")[0])
        except (IndexError, ValueError):
            n_exec = 5
    return float(elapsed) / max(1, n_exec)


def _mode_columns(
    record: dict[str, Any] | None,
    mode: str,
    *,
    reference_mc_steps: int,
) -> dict[str, Any]:
    """Flatten one fixed/dynamic compile row into mode-prefixed columns."""
    prefix = f"{mode}_"
    if record is None:
        return {
            f"{prefix}status": "not_measured",
            f"{prefix}compile_rss_gb": None,
            f"{prefix}compile_time_s": None,
            f"{prefix}n_estimation_wires": None,
            f"{prefix}n_trotter": None,
            f"{prefix}n_terms": None,
            f"{prefix}per_eval_s": None,
            f"{prefix}compile_once_catalyst_reference_mc_s": None,
            f"{prefix}no_catalyst_reference_mc_s": None,
            f"{prefix}compile_once_speedup_reference_mc": None,
            f"{prefix}provenance": None,
        }

    per_eval = _phase_c_per_eval_s(record)
    compile_time = record.get("compile_time_s")
    compile_once_reference = None
    if compile_time is not None and per_eval is not None:
        compile_once_reference = float(compile_time) + float(reference_mc_steps) * per_eval

    return {
        f"{prefix}status": record.get("classical_status", "unknown"),
        f"{prefix}compile_rss_gb": record.get("compile_rss_gb"),
        f"{prefix}compile_time_s": compile_time,
        f"{prefix}n_estimation_wires": record.get("n_estimation_wires"),
        f"{prefix}n_trotter": record.get("n_trotter"),
        f"{prefix}n_terms": record.get("n_terms"),
        f"{prefix}per_eval_s": per_eval,
        f"{prefix}compile_once_catalyst_reference_mc_s": compile_once_reference,
        f"{prefix}no_catalyst_reference_mc_s": None,
        f"{prefix}compile_once_speedup_reference_mc": None,
        f"{prefix}provenance": record.get("classical_provenance") or record.get("classical_case"),
    }


def load_h3o_boundary(path: Path) -> dict[str, Any]:
    """Load H3O+ fixed success and OOM boundary evidence, if present."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("results", [])
    out: dict[str, Any] = {}
    for record in records:
        status = str(record.get("status", "")).lower()
        label = str(record.get("label", "")).lower()
        if record.get("mode") == "fixed" and status == "success" and record.get("n_est") == 4:
            out["fixed_4bit_success"] = record
        if "oom" in status or "oom" in label:
            out["oom_boundary"] = record
    return out


def _apply_h3o_evidence(
    row: dict[str, Any],
    h3o_evidence: dict[str, Any],
    *,
    reference_mc_steps: int,
) -> None:
    """Fill H3O+ fixed/boundary fields from benchmark evidence when needed."""
    fixed = h3o_evidence.get("fixed_4bit_success")
    if fixed and row.get("fixed_status") == "not_measured":
        row.update(
            {
                "fixed_status": "measured_h3o_mc",
                "fixed_compile_rss_gb": (
                    float(fixed["rss_total_peak_mb"]) / 1024.0
                    if fixed.get("rss_total_peak_mb") is not None
                    else None
                ),
                "fixed_compile_time_s": fixed.get("compile_s"),
                "fixed_n_estimation_wires": fixed.get("n_est"),
                "fixed_n_trotter": fixed.get("n_trotter"),
                "fixed_per_eval_s": (
                    float(fixed["per_qpe_ms"]) / 1000.0
                    if fixed.get("per_qpe_ms") is not None
                    else None
                ),
                "fixed_provenance": "data/output/h3o_8bit_qpe_benchmark.json:4bit_fixed_analytical",
            }
        )
        if row["fixed_compile_time_s"] is not None and row["fixed_per_eval_s"] is not None:
            row["fixed_compile_once_catalyst_reference_mc_s"] = float(
                row["fixed_compile_time_s"]
            ) + float(reference_mc_steps) * float(row["fixed_per_eval_s"])

    boundary = h3o_evidence.get("oom_boundary")
    if boundary and row.get("dynamic_status") == "not_measured":
        row["dynamic_status"] = "boundary_unmeasured"
        row["boundary_note"] = (
            "H3O+ dynamic trotter-5 was not measured in the local matrix; "
            "existing H3O+ high-resolution Catalyst benchmark records an OOM "
            "compile boundary on the same workstation class."
        )
        row["boundary_provenance"] = "data/output/h3o_8bit_qpe_benchmark.json:OOM"


def _build_standard_qpe_callable(
    ops: list[Any],
    coeffs: list[float],
    hf_state,
    circuit_params: dict[str, Any],
    *,
    mode: str,
):
    """Build the standardized non-Catalyst QPE callable used for timing controls."""
    import pennylane as qml

    from q2m3.core.device_utils import select_device as _select_device

    n_system = circuit_params["n_system_qubits"]
    n_est = circuit_params["n_estimation_wires"]
    n_trotter = circuit_params["n_trotter"]
    base_time = circuit_params["base_time"]
    system_wires = list(range(n_system))
    est_wires = list(range(n_system, n_system + n_est))
    dev = _select_device("lightning.qubit", n_system + n_est, use_catalyst=False)

    def _prepare_hf_state() -> None:
        for wire, occ in zip(system_wires, hf_state, strict=True):
            if occ == 1:
                qml.PauliX(wires=wire)

    if mode == "fixed":
        h_fixed = qml.dot(list(coeffs), ops)

        @qml.qnode(dev)
        def qpe_fixed():
            _prepare_hf_state()
            for wire in est_wires:
                qml.Hadamard(wires=wire)
            for k, ew in enumerate(est_wires):
                t = (2 ** (n_est - 1 - k)) * base_time
                qml.ctrl(
                    qml.adjoint(qml.TrotterProduct(h_fixed, time=t, n=n_trotter, order=2)),
                    control=ew,
                )
            qml.adjoint(qml.QFT)(wires=est_wires)
            return qml.probs(wires=est_wires)

        return qpe_fixed

    @qml.qnode(dev)
    def qpe_dynamic(coeffs_arr):
        h_runtime = qml.dot(coeffs_arr, ops)
        _prepare_hf_state()
        for wire in est_wires:
            qml.Hadamard(wires=wire)
        for k, ew in enumerate(est_wires):
            t = (2 ** (n_est - 1 - k)) * base_time
            qml.ctrl(
                qml.adjoint(
                    qml.TrotterProduct(
                        h_runtime,
                        time=t,
                        n=n_trotter,
                        order=2,
                        check_hermitian=False,
                    )
                ),
                control=ew,
            )
        qml.adjoint(qml.QFT)(wires=est_wires)
        return qml.probs(wires=est_wires)

    return qpe_dynamic


def _time_standard_execution(
    callable_fn,
    coeffs: list[float],
    *,
    n_iterations: int,
    is_fixed: bool,
) -> dict[str, float]:
    """Measure repeated non-Catalyst execution time on the standardized QPE circuit."""
    import numpy as np

    coeffs_arr = np.array(coeffs, dtype=np.float64)
    start = time.perf_counter()
    result = None
    for _ in range(n_iterations):
        result = callable_fn() if is_fixed else callable_fn(coeffs_arr)
    elapsed = time.perf_counter() - start
    prob_sum = float(np.sum(result)) if result is not None else 0.0
    return {
        "elapsed_s": elapsed,
        "per_eval_s": elapsed / max(1, n_iterations),
        "prob_sum": prob_sum,
    }


def run_repeated_qpe_timing(label: str, n_iterations: int) -> dict[str, Any]:
    """Run optional mode-resolved repeated-QPE timing on the standardized 4-bit/trotter-5 circuit."""
    from examples.ir_qre_trotter5_compile_survey import systems_by_label
    from q2m3.profiling import (
        profile_execution,
        profile_hamiltonian_build,
        profile_qjit_compilation,
        profile_qjit_compilation_fixed,
    )

    systems = systems_by_label()
    if label not in systems:
        raise ValueError(f"no standardized timing preset for {label}")
    mol = systems[label]
    snap_a, ops, coeffs, hf_state, circuit_params = profile_hamiltonian_build(
        mol, n_est=4, n_trotter=5
    )
    _ = snap_a

    out: dict[str, Any] = {
        "label": label,
        "n_iterations": int(n_iterations),
        "timing_basis": mol.basis,
        "timing_n_estimation_wires": 4,
        "timing_n_trotter": 5,
        "timing_provenance": "standardized_qpe_profile",
    }

    for mode in TIMING_MODES:
        prefix = f"{mode}_"
        is_fixed = mode == "fixed"
        standard_fn = _build_standard_qpe_callable(
            ops,
            coeffs,
            hf_state,
            circuit_params,
            mode=mode,
        )
        standard_timing = _time_standard_execution(
            standard_fn,
            coeffs,
            n_iterations=n_iterations,
            is_fixed=is_fixed,
        )
        if is_fixed:
            snap_b, _timeline, _ir_analysis, compiled_fn = profile_qjit_compilation_fixed(
                ops,
                coeffs,
                hf_state,
                circuit_params,
                keep_intermediate=False,
            )
        else:
            snap_b, _timeline, _ir_analysis, compiled_fn = profile_qjit_compilation(
                ops,
                coeffs,
                hf_state,
                circuit_params,
                keep_intermediate=False,
            )
        catalyst_snap, catalyst_prob_sum = profile_execution(
            compiled_fn,
            coeffs,
            n_calls=n_iterations,
            is_fixed=is_fixed,
        )
        catalyst_per_eval = float(catalyst_snap.elapsed_s) / max(1, n_iterations)
        out.update(
            {
                f"{prefix}no_catalyst_repeated_qpe_s": standard_timing["elapsed_s"],
                f"{prefix}catalyst_repeated_qpe_s": float(catalyst_snap.elapsed_s),
                f"{prefix}no_catalyst_per_eval_s": standard_timing["per_eval_s"],
                f"{prefix}catalyst_repeated_qpe_per_eval_s": catalyst_per_eval,
                f"{prefix}catalyst_speedup": (
                    standard_timing["elapsed_s"] / float(catalyst_snap.elapsed_s)
                    if catalyst_snap.elapsed_s > 0.0
                    else None
                ),
                f"{prefix}compile_time_s_control": float(snap_b.elapsed_s),
                f"{prefix}standard_prob_sum": standard_timing["prob_sum"],
                f"{prefix}catalyst_prob_sum": catalyst_prob_sum,
            }
        )
    return out


def _timing_rows_by_label(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Key optional timing rows by label."""
    return {str(row["label"]): row for row in rows if row.get("label") is not None}


def load_timing_rows(path: Path | None) -> list[dict[str, Any]]:
    """Load optional repeated-QPE timing rows from JSON or CSV."""
    if path is None or not path.exists():
        return []
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("rows", payload.get("timings", []))
        if isinstance(rows, list):
            return rows
        raise ValueError(f"{path} must contain a rows or timings list")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _apply_timing_columns(
    row: dict[str, Any],
    timing: dict[str, Any] | None,
    *,
    reference_mc_steps: int,
) -> None:
    """Attach mode-resolved repeated-QPE timing and 1000-step MC model columns."""
    if timing is None:
        row.update(
            {
                "timing_n_iterations": None,
                "timing_basis": None,
                "timing_n_estimation_wires": None,
                "timing_n_trotter": None,
                "timing_provenance": None,
            }
        )
        for mode in TIMING_MODES:
            prefix = f"{mode}_"
            row.update(
                {
                    f"{prefix}no_catalyst_repeated_qpe_s": None,
                    f"{prefix}catalyst_repeated_qpe_s": None,
                    f"{prefix}no_catalyst_per_eval_s": None,
                    f"{prefix}catalyst_repeated_qpe_per_eval_s": None,
                    f"{prefix}catalyst_speedup": None,
                    f"{prefix}compile_time_s_control": None,
                    f"{prefix}standard_prob_sum": None,
                    f"{prefix}catalyst_prob_sum": None,
                }
            )
        return

    row.update(
        {
            "timing_n_iterations": timing.get("n_iterations"),
            "timing_basis": timing.get("timing_basis"),
            "timing_n_estimation_wires": timing.get("timing_n_estimation_wires"),
            "timing_n_trotter": timing.get("timing_n_trotter"),
            "timing_provenance": timing.get("timing_provenance", "standardized_qpe_profile"),
        }
    )
    for mode in TIMING_MODES:
        prefix = f"{mode}_"
        no_catalyst_per_eval = timing.get(f"{prefix}no_catalyst_per_eval_s")
        if no_catalyst_per_eval is None:
            repeated = timing.get(f"{prefix}no_catalyst_repeated_qpe_s")
            n_iterations = timing.get("n_iterations")
            if repeated is not None and n_iterations:
                no_catalyst_per_eval = float(repeated) / max(1, int(n_iterations))
        row.update(
            {
                f"{prefix}no_catalyst_repeated_qpe_s": timing.get(
                    f"{prefix}no_catalyst_repeated_qpe_s"
                ),
                f"{prefix}catalyst_repeated_qpe_s": timing.get(f"{prefix}catalyst_repeated_qpe_s"),
                f"{prefix}no_catalyst_per_eval_s": no_catalyst_per_eval,
                f"{prefix}catalyst_repeated_qpe_per_eval_s": timing.get(
                    f"{prefix}catalyst_repeated_qpe_per_eval_s"
                ),
                f"{prefix}catalyst_speedup": timing.get(f"{prefix}catalyst_speedup"),
                f"{prefix}compile_time_s_control": timing.get(f"{prefix}compile_time_s_control"),
                f"{prefix}standard_prob_sum": timing.get(f"{prefix}standard_prob_sum"),
                f"{prefix}catalyst_prob_sum": timing.get(f"{prefix}catalyst_prob_sum"),
            }
        )
        if no_catalyst_per_eval is not None:
            row[f"{prefix}no_catalyst_reference_mc_s"] = float(reference_mc_steps) * float(
                no_catalyst_per_eval
            )
            compile_once = row.get(f"{prefix}compile_once_catalyst_reference_mc_s")
            if compile_once is not None:
                compile_once_value = float(compile_once)
                if compile_once_value > 0.0:
                    row[f"{prefix}compile_once_speedup_reference_mc"] = (
                        row[f"{prefix}no_catalyst_reference_mc_s"] / compile_once_value
                    )


def build_matrix_rows(
    *,
    qre_json: Path,
    compile_jsons: list[Path],
    h3o_benchmark_json: Path,
    reference_mc_steps: int,
    timing_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Join QRE, Catalyst compile, boundary, and optional timing rows."""
    qre_by_label = _records_by_label(load_json_records(qre_json))
    compile_records: list[dict[str, Any]] = []
    for path in compile_jsons:
        compile_records.extend(load_json_records(path))
    timings = _timing_rows_by_label(timing_rows or [])
    h3o_evidence = load_h3o_boundary(h3o_benchmark_json)

    rows: list[dict[str, Any]] = []
    for target in TARGET_SYSTEMS:
        qre = qre_by_label.get(target["qre_label"], {})
        fixed = best_compile_row(compile_records, label=target["compile_label"], mode="fixed")
        dynamic = best_compile_row(compile_records, label=target["compile_label"], mode="dynamic")
        row: dict[str, Any] = {
            "label": target["label"],
            "qre_label": target["qre_label"],
            "reference_mc_steps": int(reference_mc_steps),
            "system_class": target["system_class"],
            "charge": qre.get("charge"),
            "basis": qre.get("basis"),
            "active_electrons": qre.get("active_electrons"),
            "active_orbitals": qre.get("active_orbitals"),
            "n_system_qubits": qre.get("n_system_qubits"),
            "logical_qubits": qre.get("logical_qubits"),
            "toffoli_gates": qre.get("toffoli_gates"),
            "t_depth": qre.get("t_depth"),
            "hamiltonian_1norm_Ha": qre.get("hamiltonian_1norm_Ha"),
            "qre_provenance": str(qre_json) if qre else None,
            "boundary_note": "",
            "boundary_provenance": "",
        }
        row.update(_mode_columns(fixed, "fixed", reference_mc_steps=reference_mc_steps))
        row.update(_mode_columns(dynamic, "dynamic", reference_mc_steps=reference_mc_steps))
        if target["label"] == "H3O+":
            _apply_h3o_evidence(row, h3o_evidence, reference_mc_steps=reference_mc_steps)
        if target["label"] == "NH4+" and row.get("dynamic_status") == "not_measured":
            row["boundary_note"] = (
                "NH4+ QRE is present, but local Catalyst fixed/dynamic compile rows "
                "were not measured in the trotter-5 matrix."
            )

        _apply_timing_columns(
            row, timings.get(target["label"]), reference_mc_steps=reference_mc_steps
        )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV with stable union field order."""
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


def collect_matrix(
    output_dir: Path,
    *,
    qre_json: Path,
    compile_jsons: list[Path],
    h3o_benchmark_json: Path,
    reference_mc_steps: int,
    timing_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write ion Catalyst matrix CSV/JSON outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_matrix_rows(
        qre_json=qre_json,
        compile_jsons=compile_jsons,
        h3o_benchmark_json=h3o_benchmark_json,
        reference_mc_steps=reference_mc_steps,
        timing_rows=timing_rows,
    )
    csv_path = output_dir / "ion_catalyst_matrix.csv"
    json_path = output_dir / "ion_catalyst_matrix.json"
    _write_csv(csv_path, rows)
    payload = {
        "metadata": {
            "schema": "ion_catalyst_matrix.v2",
            "catalyst": "PennyLane Catalyst @qjit",
            "reference_mc_steps": int(reference_mc_steps),
            "note": (
                "Measured rows are joined from existing artifacts; missing rows are explicit "
                "statuses. Repeated-QPE controls use the same standardized 4-bit, five-Trotter-step "
                "circuit as the compile survey, while 1000-step MC totals compare no-Catalyst "
                "execution against compile-once Catalyst execution."
            ),
        },
        "rows": rows,
        "outputs": {"csv": str(csv_path), "json": str(json_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    return payload


def main(
    argv: list[str] | None = None,
    timing_runner: TimingRunner | None = None,
) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qre-json", type=Path, default=DEFAULT_QRE_JSON)
    parser.add_argument(
        "--compile-json",
        type=Path,
        action="append",
        default=None,
        help="Compile JSON to join. May be passed multiple times.",
    )
    parser.add_argument("--h3o-benchmark-json", type=Path, default=DEFAULT_H3O_BENCHMARK_JSON)
    parser.add_argument("--timing-json", type=Path, default=None)
    parser.add_argument("--run-timing", action="store_true")
    parser.add_argument("--timing-labels", default=TIMING_LABELS_DEFAULT)
    parser.add_argument("--timing-iterations", type=int, default=5)
    parser.add_argument("--reference-mc-steps", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    timing_rows = load_timing_rows(args.timing_json)
    if args.run_timing:
        runner = timing_runner or run_repeated_qpe_timing
        for label in [part.strip() for part in args.timing_labels.split(",") if part.strip()]:
            timing_rows.append(runner(label, args.timing_iterations))

    payload = collect_matrix(
        args.output_dir,
        qre_json=args.qre_json,
        compile_jsons=args.compile_json or [DEFAULT_COMPILE_JSON],
        h3o_benchmark_json=args.h3o_benchmark_json,
        reference_mc_steps=args.reference_mc_steps,
        timing_rows=timing_rows,
    )
    print(f"wrote {payload['outputs']['csv']}")
    print(f"wrote {payload['outputs']['json']}")


if __name__ == "__main__":
    main()
