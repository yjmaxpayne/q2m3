# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Tests for examples/ir_qre_correlation_analysis.py."""

import csv
import json


def _write_qre_fixture(tmp_path):
    """Create a compact QRE JSON fixture with three successful systems."""
    path = tmp_path / "qre_survey.json"
    payload = {
        "metadata": {"target_error_Hartree": 0.0016},
        "systems": [
            {
                "label": "H2",
                "basis": "sto-3g",
                "active_electrons": 2,
                "active_orbitals": 2,
                "n_system_qubits": 4,
                "logical_qubits": 115,
                "toffoli_gates": 1_224_608,
                "t_depth": 8_572_256,
                "hamiltonian_1norm_Ha": 2.3136733880391414,
            },
            {
                "label": "LiH",
                "basis": "sto-3g",
                "active_electrons": 4,
                "active_orbitals": 4,
                "n_system_qubits": 8,
                "logical_qubits": 131,
                "toffoli_gates": 6_517_595,
                "t_depth": 45_623_165,
                "hamiltonian_1norm_Ha": 6.118,
            },
            {
                "label": "HeH+",
                "basis": "sto-3g",
                "active_electrons": 2,
                "active_orbitals": 2,
                "n_system_qubits": 4,
                "logical_qubits": 115,
                "toffoli_gates": 1_517_824,
                "t_depth": 10_624_768,
                "hamiltonian_1norm_Ha": 2.868,
            },
            {
                "label": "H3+",
                "basis": "sto-3g",
                "active_electrons": 2,
                "active_orbitals": 3,
                "n_system_qubits": 6,
                "logical_qubits": 124,
                "toffoli_gates": 3_186_752,
                "t_depth": 22_307_264,
                "hamiltonian_1norm_Ha": 4.017,
            },
            {
                "label": "H3O+ STO-3G",
                "basis": "sto-3g",
                "active_electrons": 4,
                "active_orbitals": 4,
                "n_system_qubits": 8,
                "logical_qubits": 131,
                "toffoli_gates": 6_789_930,
                "t_depth": 47_529_510,
                "hamiltonian_1norm_Ha": 6.373999390184238,
            },
            {
                "label": "NH3",
                "basis": "sto-3g",
                "active_electrons": 8,
                "active_orbitals": 7,
                "n_system_qubits": 14,
                "logical_qubits": 288,
                "toffoli_gates": 36_504_804,
                "t_depth": 255_533_628,
                "hamiltonian_1norm_Ha": 18.06724442295652,
            },
            {
                "label": "H2O",
                "basis": "sto-3g",
                "active_electrons": 8,
                "active_orbitals": 6,
                "n_system_qubits": 12,
                "logical_qubits": 210,
                "toffoli_gates": 20_000_000,
                "t_depth": 140_000_000,
                "hamiltonian_1norm_Ha": 12.0,
            },
            {
                "label": "CH4",
                "basis": "sto-3g",
                "active_electrons": 8,
                "active_orbitals": 7,
                "n_system_qubits": 14,
                "logical_qubits": 260,
                "toffoli_gates": 30_000_000,
                "t_depth": 210_000_000,
                "hamiltonian_1norm_Ha": 16.0,
            },
            {
                "label": "NH4+",
                "basis": "sto-3g",
                "active_electrons": 8,
                "active_orbitals": 7,
                "n_system_qubits": 14,
                "logical_qubits": 270,
                "toffoli_gates": 32_000_000,
                "t_depth": 224_000_000,
                "hamiltonian_1norm_Ha": 17.0,
            },
            {
                "label": "Formamide",
                "basis": "sto-3g",
                "active_electrons": 8,
                "active_orbitals": 8,
                "n_system_qubits": 16,
                "logical_qubits": 350,
                "toffoli_gates": 52_000_000,
                "t_depth": 364_000_000,
                "hamiltonian_1norm_Ha": 23.0,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_trotter5_compile_fixture(tmp_path):
    """Create a standardized n_trotter=5 compile JSON fixture."""
    path = tmp_path / "ir_qre_trotter5_compile_survey.json"
    payload = {
        "metadata": {"n_estimation_wires": 4, "n_trotter": 5, "n_trotter_steps": 5},
        "systems": [
            {
                "label": "LiH",
                "classical_case": "LiH fixed 4-bit 5-trotter",
                "classical_mode": "fixed",
                "classical_status": "measured",
                "n_terms": 193,
                "n_estimation_wires": 4,
                "n_trotter": 5,
                "ir_ops_lower_bound": 3860,
                "compile_rss_gb": 4.2,
                "compile_time_s": 31.5,
                "bufferization_amp": 1.0,
            },
            {
                "label": "LiH",
                "classical_case": "LiH dynamic 4-bit 5-trotter",
                "classical_mode": "dynamic",
                "classical_status": "measured",
                "n_terms": 193,
                "n_estimation_wires": 4,
                "n_trotter": 5,
                "ir_ops_lower_bound": 3860,
                "compile_rss_gb": 8.4,
                "compile_time_s": 63.0,
                "bufferization_amp": 4.4,
            },
            {
                "label": "HeH+",
                "classical_case": "HeH+ fixed 4-bit 5-trotter",
                "classical_mode": "fixed",
                "classical_status": "measured",
                "n_terms": 27,
                "n_estimation_wires": 4,
                "n_trotter": 5,
                "ir_ops_lower_bound": 540,
                "compile_rss_gb": 2.1,
                "compile_time_s": 42.8,
                "bufferization_amp": 1.0,
            },
            {
                "label": "H3+",
                "classical_case": "H3+ fixed 4-bit 5-trotter",
                "classical_mode": "fixed",
                "classical_status": "measured",
                "n_terms": 62,
                "n_estimation_wires": 4,
                "n_trotter": 5,
                "ir_ops_lower_bound": 1240,
                "compile_rss_gb": 2.7,
                "compile_time_s": 101.8,
                "bufferization_amp": 1.0,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_correlation_rows_cover_three_systems_and_four_dimensions(tmp_path):
    """Summary rows only use standardized trotter-5 measured compile data."""
    from examples.ir_qre_correlation_analysis import (
        DIMENSIONS,
        build_correlation_rows,
        compute_dimension_stats,
    )

    rows = build_correlation_rows(
        _write_qre_fixture(tmp_path),
        _write_trotter5_compile_fixture(tmp_path),
    )

    assert len({row["label"] for row in rows}) >= 3
    assert {row["n_trotter"] for row in rows} == {5}
    assert all(row["classical_status"] == "measured" for row in rows)
    assert any(row["classical_case"] == "LiH fixed 4-bit 5-trotter" for row in rows)
    assert any(row["classical_case"] == "LiH dynamic 4-bit 5-trotter" for row in rows)
    assert not any("2-bit 1-trotter" in row["classical_case"] for row in rows)
    assert len(DIMENSIONS) == 4
    for dimension in DIMENSIONS:
        assert any(row[dimension.x_key] is not None for row in rows)
        assert any(row[dimension.y_key] is not None for row in rows)

    stats = compute_dimension_stats(rows)

    assert {stat.dimension_id for stat in stats} == {
        dimension.dimension_id for dimension in DIMENSIONS
    }
    assert all(stat.n_points >= 3 for stat in stats)
    assert all(stat.conclusion for stat in stats)


def test_trotter5_loader_rejects_nonstandard_compile_rows(tmp_path):
    """Nonstandard measured rows are rejected instead of silently mixed in."""
    from examples.ir_qre_correlation_analysis import build_correlation_rows

    compile_json = _write_trotter5_compile_fixture(tmp_path)
    payload = json.loads(compile_json.read_text(encoding="utf-8"))
    payload["systems"][0]["n_trotter"] = 10
    compile_json.write_text(json.dumps(payload), encoding="utf-8")

    try:
        build_correlation_rows(_write_qre_fixture(tmp_path), compile_json)
    except ValueError as exc:
        assert "n_trotter=5" in str(exc)
    else:
        raise AssertionError("expected nonstandard trotter row to fail")


def test_main_writes_csv_report_and_four_publication_figures(tmp_path):
    """The CLI writes the summary table, report, stats table, and 300-dpi figures."""
    from examples.ir_qre_correlation_analysis import main

    out_dir = tmp_path / "output"
    main(
        [
            "--qre-json",
            str(_write_qre_fixture(tmp_path)),
            "--compile-json",
            str(_write_trotter5_compile_fixture(tmp_path)),
            "--output-dir",
            str(out_dir),
        ]
    )

    summary_csv = out_dir / "ir_qre_correlation.csv"
    cases_csv = out_dir / "ir_qre_correlation_cases.csv"
    stats_csv = out_dir / "ir_qre_correlation_stats.csv"
    report = out_dir / "ir_qre_correlation_report.md"
    figures = sorted(out_dir.glob("ir_qre_correlation_D*.png"))

    assert summary_csv.exists()
    assert cases_csv.exists()
    assert stats_csv.exists()
    assert report.exists()
    assert len(figures) == 4

    with summary_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 3
    assert "ir_ops_lower_bound" in rows[0]
    assert "toffoli_gates" in rows[0]
    assert {int(row["n_trotter"]) for row in rows} == {5}
    assert any(row["classical_case"] == "LiH fixed 4-bit 5-trotter" for row in rows)
    assert any(row["classical_case"] == "LiH dynamic 4-bit 5-trotter" for row in rows)

    report_text = report.read_text(encoding="utf-8")
    assert "IR-QRE Correlation Analysis" in report_text
    assert "trotter-5 standard" in report_text
    assert "case-level" in report_text
    assert "reduced-shape dynamic" not in report_text
    assert "Power fit" in report_text
    assert "y = a * x^b" in report_text
    assert "Linear R2" not in report_text
    assert "Measured compile-data status" in report_text
    assert "Null-result handling" in report_text

    with stats_csv.open(newline="") as f:
        stats_rows = list(csv.DictReader(f))
    assert "power_law_formula" in stats_rows[0]
    assert "linear_r_squared" not in stats_rows[0]
