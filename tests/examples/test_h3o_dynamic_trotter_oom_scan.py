# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Tests for examples/h3o_dynamic_trotter_oom_scan.py."""

import csv
import json


def test_scan_trotter_values_stops_on_memory_guard():
    """The scan stops before trying larger Trotter counts after guard trip."""
    from examples.h3o_dynamic_trotter_oom_scan import scan_trotter_values

    calls = []

    def fake_runner(**kwargs):
        n_trotter = kwargs["n_trotter"]
        calls.append(n_trotter)
        return {
            "status": "memory_guard_tree" if n_trotter == 2 else "measured",
            "n_trotter": n_trotter,
        }

    records = scan_trotter_values(
        start_trotter=1,
        max_trotter=5,
        n_estimation_wires=4,
        mode="dynamic",
        timeout_s=60,
        threshold_fraction=0.8,
        poll_s=0.5,
        total_memory_mb=64_000.0,
        runner=fake_runner,
    )

    assert calls == [1, 2]
    assert [record["n_trotter"] for record in records] == [1, 2]
    assert records[-1]["status"] == "memory_guard_tree"


def test_write_outputs_writes_json_and_csv(tmp_path):
    """Output files retain scan metadata and flat per-Trotter records."""
    from examples.h3o_dynamic_trotter_oom_scan import write_outputs

    json_path = tmp_path / "scan.json"
    csv_path = tmp_path / "scan.csv"
    records = [
        {
            "status": "measured",
            "n_trotter": 1,
            "compile_rss_mb": 1024.0,
            "phase_b": {"elapsed_s": 1.2},
        }
    ]

    write_outputs(records, json_path, csv_path, threshold_fraction=0.8, total_memory_mb=64_000.0)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["schema"] == "h3o_dynamic_trotter_oom_scan.v1"
    assert payload["metadata"]["threshold_fraction"] == 0.8
    assert payload["records"] == records

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"status": "measured", "n_trotter": "1", "compile_rss_mb": "1024.0"}]
