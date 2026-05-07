#!/usr/bin/env python3
# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Safely scan H3O+ dynamic QPE compile memory versus Trotter steps.

Each Trotter count runs in an isolated subprocess. The parent process polls
system memory and kills the whole subprocess tree when used RAM reaches the
configured fraction of total RAM, preventing Catalyst compilation from driving
the machine into OOM.

Usage:
    OMP_NUM_THREADS=8 uv run python examples/h3o_dynamic_trotter_oom_scan.py
    OMP_NUM_THREADS=8 uv run python examples/h3o_dynamic_trotter_oom_scan.py --max-trotter 12
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import signal
import sys
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Any

from q2m3.molecule import MoleculeConfig
from q2m3.profiling import ProfileResult, run_single_profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_OUTPUT_JSON = OUTPUT_DIR / "h3o_dynamic_trotter_oom_scan.json"
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "h3o_dynamic_trotter_oom_scan.csv"

H3OP = MoleculeConfig(
    name="H3O+",
    symbols=["O", "H", "H", "H"],
    coords=[
        [0.0000, 0.0000, 0.1173],
        [0.0000, 0.9572, -0.4692],
        [0.8286, -0.4786, -0.4692],
        [-0.8286, -0.4786, -0.4692],
    ],
    charge=1,
    active_electrons=4,
    active_orbitals=4,
    basis="sto-3g",
)

STOP_STATUSES = {"memory_guard_tree", "timeout"}


def read_system_memory_mb() -> dict[str, float]:
    """Return Linux system memory totals in MB using /proc/meminfo."""
    values: dict[str, float] = {}
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            key, raw_value = line.split(":", maxsplit=1)
            parts = raw_value.split()
            if parts:
                values[key] = float(parts[0]) / 1024.0

    total = values["MemTotal"]
    available = values.get("MemAvailable", values.get("MemFree", 0.0))
    used = total - available
    return {
        "total_mb": total,
        "available_mb": available,
        "used_mb": used,
        "used_fraction": used / total if total else 0.0,
    }


def _read_proc_status_mb(pid: int) -> dict[str, float]:
    """Read VmRSS and VmHWM for one process in MB."""
    result = {"VmRSS": 0.0, "VmHWM": 0.0}
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            for line in f:
                for key in result:
                    if line.startswith(f"{key}:"):
                        result[key] = float(line.split()[1]) / 1024.0
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return result


def _child_pids(pid: int) -> list[int]:
    """Return direct Linux child PIDs for a process."""
    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        text = children_path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []
    return [int(part) for part in text.split()] if text else []


def _process_tree(pid: int) -> list[int]:
    """Return pid plus all known descendants."""
    pending = [pid]
    seen: list[int] = []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.append(current)
        pending.extend(_child_pids(current))
    return seen


def _process_tree_memory_mb(pid: int) -> dict[str, float]:
    """Return current RSS and high-water RSS across a process tree."""
    rss = 0.0
    hwm = 0.0
    pids = _process_tree(pid)
    for tree_pid in pids:
        status = _read_proc_status_mb(tree_pid)
        rss += status["VmRSS"]
        hwm += status["VmHWM"]
    return {"tree_rss_mb": rss, "tree_hwm_mb": hwm, "tree_n_processes": float(len(pids))}


def _kill_process_tree(pid: int) -> None:
    """Terminate a process tree, escalating to SIGKILL if needed."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for tree_pid in reversed(_process_tree(pid)):
            try:
                os.kill(tree_pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(0.5)


def _snapshot_to_dict(snapshot: Any) -> dict[str, Any] | None:
    """Convert a dataclass memory snapshot to JSON-friendly data."""
    if snapshot is None:
        return None
    return asdict(snapshot) if hasattr(snapshot, "__dataclass_fields__") else dict(snapshot)


def _compile_rss_mb(result: ProfileResult, monitor_data: dict[str, Any]) -> float | None:
    """Return the best available compile RSS estimate in MB."""
    if result.error or result.phase_b is None:
        return None
    phase_b_total = result.phase_b.maxrss_mb + result.phase_b.maxrss_children_mb
    parent_peak = max(
        float(monitor_data.get("peak_tree_rss_mb", 0.0) or 0.0),
        float(monitor_data.get("peak_tree_hwm_mb", 0.0) or 0.0),
    )
    return max(phase_b_total, parent_peak, float(result.timeline_peak_mb or 0.0))


def profile_result_to_record(result: ProfileResult, monitor_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a completed profile result into one scan record."""
    if result.error:
        return {
            "status": "failed",
            "error": result.error,
            "n_estimation_wires": result.n_estimation_wires,
            "n_trotter": result.n_trotter,
            "mode": result.mode,
            **monitor_data,
        }

    compile_rss_mb = _compile_rss_mb(result, monitor_data)
    compile_time_s = result.phase_b.elapsed_s if result.phase_b else None
    return {
        "status": "measured",
        "molecule": result.molecule,
        "mode": result.mode,
        "n_system_qubits": result.n_system_qubits,
        "n_estimation_wires": result.n_estimation_wires,
        "total_qubits": result.n_system_qubits + result.n_estimation_wires,
        "n_trotter": result.n_trotter,
        "n_terms": result.n_terms,
        "ir_ops_lower_bound": result.ir_scale,
        "compile_rss_mb": round(compile_rss_mb, 1) if compile_rss_mb is not None else None,
        "compile_rss_gb": round(compile_rss_mb / 1024.0, 4) if compile_rss_mb else None,
        "compile_time_s": round(compile_time_s, 3) if compile_time_s is not None else None,
        "prob_sum": float(result.prob_sum),
        "phase_a": _snapshot_to_dict(result.phase_a),
        "phase_b": _snapshot_to_dict(result.phase_b),
        "phase_c": _snapshot_to_dict(result.phase_c),
        "timeline_peak_mb": result.timeline_peak_mb,
        "timeline_samples": result.timeline_samples,
        "ir_analysis": [list(item) for item in result.ir_analysis],
        **monitor_data,
    }


def _profile_worker(
    n_estimation_wires: int,
    n_trotter: int,
    mode: str,
    queue: multiprocessing.Queue,
    ir_dir: str | None,
) -> None:
    """Run one compile profile in a subprocess."""
    try:
        tracemalloc.start()
        result = run_single_profile(
            H3OP,
            n_est=n_estimation_wires,
            n_trotter=n_trotter,
            mode=mode,
            ir_dir=ir_dir,
        )
        tracemalloc.stop()
        queue.put(result)
    except Exception as exc:  # noqa: BLE001 - preserve failures in scan output.
        queue.put(
            ProfileResult(
                molecule=H3OP.name,
                n_system_qubits=H3OP.active_orbitals * 2,
                n_estimation_wires=n_estimation_wires,
                n_trotter=n_trotter,
                n_terms=0,
                ir_scale=0,
                mode=mode,
                error=f"{type(exc).__name__}: {exc}",
            )
        )


def run_profile_with_memory_guard(
    *,
    n_trotter: int,
    n_estimation_wires: int,
    mode: str,
    timeout_s: int,
    threshold_fraction: float,
    poll_s: float,
    total_memory_mb: float | None = None,
    ir_dir: str | None = None,
) -> dict[str, Any]:
    """Run one Trotter profile with parent-side system-memory guard."""
    total_memory_mb = total_memory_mb or read_system_memory_mb()["total_mb"]
    threshold_mb = total_memory_mb * threshold_fraction
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_profile_worker,
        args=(n_estimation_wires, n_trotter, mode, queue, ir_dir),
    )

    start = time.monotonic()
    proc.start()
    peak_system_used_mb = 0.0
    peak_system_fraction = 0.0
    peak_tree_rss_mb = 0.0
    peak_tree_hwm_mb = 0.0
    peak_tree_n_processes = 0.0
    n_samples = 0
    status = "running"

    while proc.is_alive():
        proc.join(timeout=poll_s)
        elapsed_s = time.monotonic() - start
        system_memory = read_system_memory_mb()
        tree_memory = _process_tree_memory_mb(proc.pid or 0)
        n_samples += 1

        peak_system_used_mb = max(peak_system_used_mb, system_memory["used_mb"])
        peak_system_fraction = max(peak_system_fraction, system_memory["used_fraction"])
        peak_tree_rss_mb = max(peak_tree_rss_mb, tree_memory["tree_rss_mb"])
        peak_tree_hwm_mb = max(peak_tree_hwm_mb, tree_memory["tree_hwm_mb"])
        peak_tree_n_processes = max(peak_tree_n_processes, tree_memory["tree_n_processes"])

        if system_memory["used_mb"] >= threshold_mb:
            status = "memory_guard_tree"
            _kill_process_tree(proc.pid or 0)
            proc.join(timeout=5)
            break
        if elapsed_s >= timeout_s:
            status = "timeout"
            _kill_process_tree(proc.pid or 0)
            proc.join(timeout=5)
            break

    elapsed_s = time.monotonic() - start
    monitor_data = {
        "elapsed_s": round(elapsed_s, 3),
        "memory_threshold_fraction": threshold_fraction,
        "memory_threshold_mb": round(threshold_mb, 1),
        "peak_system_used_mb": round(peak_system_used_mb, 1),
        "peak_system_used_fraction": round(peak_system_fraction, 4),
        "peak_tree_rss_mb": round(peak_tree_rss_mb, 1),
        "peak_tree_hwm_mb": round(peak_tree_hwm_mb, 1),
        "peak_tree_n_processes": int(peak_tree_n_processes),
        "monitor_samples": n_samples,
        "exitcode": proc.exitcode,
    }

    if status in STOP_STATUSES:
        return {
            "status": status,
            "molecule": H3OP.name,
            "mode": mode,
            "n_estimation_wires": n_estimation_wires,
            "n_trotter": n_trotter,
            **monitor_data,
        }
    if not queue.empty():
        result = queue.get()
        return profile_result_to_record(result, monitor_data)
    return {
        "status": "failed",
        "error": "no result from subprocess",
        "molecule": H3OP.name,
        "mode": mode,
        "n_estimation_wires": n_estimation_wires,
        "n_trotter": n_trotter,
        **monitor_data,
    }


def scan_trotter_values(
    *,
    start_trotter: int,
    max_trotter: int,
    n_estimation_wires: int,
    mode: str,
    timeout_s: int,
    threshold_fraction: float,
    poll_s: float,
    total_memory_mb: float | None = None,
    runner=run_profile_with_memory_guard,
) -> list[dict[str, Any]]:
    """Scan Trotter values and stop at the first memory guard or timeout."""
    records: list[dict[str, Any]] = []
    for n_trotter in range(start_trotter, max_trotter + 1):
        record = runner(
            n_trotter=n_trotter,
            n_estimation_wires=n_estimation_wires,
            mode=mode,
            timeout_s=timeout_s,
            threshold_fraction=threshold_fraction,
            poll_s=poll_s,
            total_memory_mb=total_memory_mb,
        )
        records.append(record)
        if record.get("status") in STOP_STATUSES:
            break
    return records


def _csv_fieldnames(records: list[dict[str, Any]]) -> list[str]:
    """Return stable CSV fields, skipping nested diagnostic payloads."""
    fieldnames: list[str] = []
    for record in records:
        for key, value in record.items():
            if isinstance(value, dict | list | tuple):
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames or ["status", "n_trotter"]


def write_outputs(
    records: list[dict[str, Any]],
    json_path: Path,
    csv_path: Path,
    *,
    threshold_fraction: float,
    total_memory_mb: float,
) -> None:
    """Persist scan records to JSON and CSV."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "schema": "h3o_dynamic_trotter_oom_scan.v1",
            "molecule": "H3O+",
            "mode": "dynamic",
            "active_space": "(4e, 4o)",
            "threshold_fraction": threshold_fraction,
            "total_memory_mb": round(total_memory_mb, 1),
            "threshold_mb": round(total_memory_mb * threshold_fraction, 1),
        },
        "records": records,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")

    fieldnames = _csv_fieldnames(records)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-trotter", type=int, default=1)
    parser.add_argument("--max-trotter", type=int, default=20)
    parser.add_argument("--n-estimation-wires", type=int, default=4)
    parser.add_argument("--mode", choices=["dynamic", "fixed"], default="dynamic")
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--threshold-fraction", type=float, default=0.80)
    parser.add_argument("--poll-s", type=float, default=0.25)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args(argv)

    if not 0.0 < args.threshold_fraction < 1.0:
        raise ValueError("--threshold-fraction must be between 0 and 1")
    if args.start_trotter < 1:
        raise ValueError("--start-trotter must be >= 1")
    if args.max_trotter < args.start_trotter:
        raise ValueError("--max-trotter must be >= --start-trotter")

    total_memory_mb = read_system_memory_mb()["total_mb"]
    print(
        f"H3O+ {args.mode} scan: n_est={args.n_estimation_wires}, "
        f"trotter={args.start_trotter}..{args.max_trotter}, "
        f"stop at {args.threshold_fraction:.0%} RAM "
        f"({total_memory_mb * args.threshold_fraction / 1024:.2f} GB)",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    for n_trotter in range(args.start_trotter, args.max_trotter + 1):
        print(f"[*] Profiling H3O+ {args.mode} n_trotter={n_trotter}", flush=True)
        record = run_profile_with_memory_guard(
            n_trotter=n_trotter,
            n_estimation_wires=args.n_estimation_wires,
            mode=args.mode,
            timeout_s=args.timeout_s,
            threshold_fraction=args.threshold_fraction,
            poll_s=args.poll_s,
            total_memory_mb=total_memory_mb,
        )
        records.append(record)
        write_outputs(
            records,
            args.output_json,
            args.output_csv,
            threshold_fraction=args.threshold_fraction,
            total_memory_mb=total_memory_mb,
        )

        status = record.get("status")
        rss = record.get("compile_rss_gb") or record.get("peak_tree_rss_mb")
        print(f"    {status}: rss={rss}, elapsed={record.get('elapsed_s')}s", flush=True)
        if status in STOP_STATUSES:
            print(f"[stop] {status} at n_trotter={n_trotter}", flush=True)
            break

    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main(sys.argv[1:])
