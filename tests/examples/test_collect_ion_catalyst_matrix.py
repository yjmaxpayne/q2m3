# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Tests for examples/collect_ion_catalyst_matrix.py."""

import csv
import json


def _write_qre(tmp_path):
    """Write compact QRE fixture for target matrix systems."""
    path = tmp_path / "qre_survey.json"
    systems = []
    for i, label in enumerate(["H2", "HeH+", "H3+", "H3O+ STO-3G", "NH4+"], start=1):
        systems.append(
            {
                "label": label,
                "basis": "sto-3g",
                "charge": 0 if label == "H2" else 1,
                "active_electrons": 2 if i <= 3 else 8,
                "active_orbitals": 2 + i,
                "n_system_qubits": 2 * (2 + i),
                "logical_qubits": 100 + i,
                "toffoli_gates": 1000 * i,
                "t_depth": 7000 * i,
                "hamiltonian_1norm_Ha": 1.5 * i,
            }
        )
    path.write_text(json.dumps({"systems": systems}), encoding="utf-8")
    return path


def _write_compile(tmp_path):
    """Write compact compile fixture with fixed/dynamic measured rows."""
    path = tmp_path / "compile.json"
    systems = [
        {
            "label": "H2",
            "classical_case": "H2 fixed 4-bit 5-trotter",
            "classical_mode": "fixed",
            "classical_status": "measured",
            "n_estimation_wires": 4,
            "n_trotter": 5,
            "n_terms": 15,
            "compile_rss_gb": 1.2,
            "compile_time_s": 10.0,
            "phase_c": {"label": "Phase C: Execution (5x)", "elapsed_s": 0.05},
        },
        {
            "label": "H2",
            "classical_case": "H2 dynamic 4-bit 5-trotter",
            "classical_mode": "dynamic",
            "classical_status": "measured",
            "n_estimation_wires": 4,
            "n_trotter": 5,
            "n_terms": 15,
            "compile_rss_gb": 3.4,
            "compile_time_s": 30.0,
            "phase_c": {"label": "Phase C: Execution (5x)", "elapsed_s": 0.1},
        },
        {
            "label": "H3+",
            "classical_case": "H3+ fixed 4-bit 5-trotter",
            "classical_mode": "fixed",
            "classical_status": "measured",
            "n_estimation_wires": 4,
            "n_trotter": 5,
            "n_terms": 62,
            "compile_rss_gb": 2.0,
            "compile_time_s": 20.0,
        },
    ]
    path.write_text(json.dumps({"systems": systems}), encoding="utf-8")
    return path


def _write_h3o(tmp_path):
    """Write H3O benchmark fixture with fixed success and OOM boundary."""
    path = tmp_path / "h3o.json"
    payload = {
        "results": [
            {
                "label": "4bit_fixed_analytical",
                "mode": "fixed",
                "n_est": 4,
                "n_trotter": 10,
                "status": "success",
                "compile_s": 0.5,
                "per_qpe_ms": 100.0,
                "rss_total_peak_mb": 2048.0,
            },
            {
                "label": "8bit_fixed_analytical",
                "mode": "fixed",
                "n_est": 8,
                "n_trotter": 10,
                "status": "OOM_KILLED",
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fake_timing_runner(label, n_iterations):
    """Return optional repeated-QPE timing without running Catalyst."""
    return {
        "label": label,
        "n_iterations": n_iterations,
        "timing_basis": "sto-3g",
        "timing_n_estimation_wires": 4,
        "timing_n_trotter": 5,
        "timing_provenance": "test",
        "fixed_no_catalyst_repeated_qpe_s": 12.0,
        "fixed_catalyst_repeated_qpe_s": 3.0,
        "fixed_no_catalyst_per_eval_s": 3.0,
        "fixed_catalyst_repeated_qpe_per_eval_s": 0.75,
        "fixed_catalyst_speedup": 4.0,
        "dynamic_no_catalyst_repeated_qpe_s": 16.0,
        "dynamic_catalyst_repeated_qpe_s": 4.0,
        "dynamic_no_catalyst_per_eval_s": 4.0,
        "dynamic_catalyst_repeated_qpe_per_eval_s": 1.0,
        "dynamic_catalyst_speedup": 4.0,
    }


def test_main_writes_ion_catalyst_matrix_with_status_rows(tmp_path):
    """Matrix collector preserves measured, missing, timing, and boundary states."""
    from examples.collect_ion_catalyst_matrix import main

    out_dir = tmp_path / "matrix"
    main(
        [
            "--qre-json",
            str(_write_qre(tmp_path)),
            "--compile-json",
            str(_write_compile(tmp_path)),
            "--h3o-benchmark-json",
            str(_write_h3o(tmp_path)),
            "--run-timing",
            "--timing-labels",
            "H2",
            "--timing-iterations",
            "4",
            "--output-dir",
            str(out_dir),
        ],
        timing_runner=_fake_timing_runner,
    )

    csv_path = out_dir / "ion_catalyst_matrix.csv"
    json_path = out_dir / "ion_catalyst_matrix.json"
    assert csv_path.exists()
    assert json_path.exists()

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_label = {row["label"]: row for row in rows}

    assert by_label["H2"]["fixed_status"] == "measured"
    assert by_label["H2"]["dynamic_status"] == "measured"
    assert by_label["H2"]["fixed_no_catalyst_repeated_qpe_s"] == "12.0"
    assert by_label["H2"]["dynamic_no_catalyst_repeated_qpe_s"] == "16.0"
    assert by_label["H2"]["fixed_compile_once_catalyst_reference_mc_s"] == "20.0"
    assert by_label["H2"]["fixed_no_catalyst_reference_mc_s"] == "3000.0"
    assert by_label["H2"]["fixed_compile_once_speedup_reference_mc"] == "150.0"
    assert by_label["H2"]["dynamic_compile_once_catalyst_reference_mc_s"] == "50.0"
    assert by_label["H2"]["dynamic_no_catalyst_reference_mc_s"] == "4000.0"
    assert by_label["H2"]["dynamic_compile_once_speedup_reference_mc"] == "80.0"
    assert by_label["H3O+"]["fixed_status"] == "measured_h3o_mc"
    assert by_label["H3O+"]["dynamic_status"] == "boundary_unmeasured"
    assert by_label["NH4+"]["dynamic_status"] == "not_measured"
    assert "PennyLane Catalyst" in json_path.read_text(encoding="utf-8")
