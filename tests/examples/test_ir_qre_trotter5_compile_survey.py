# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Tests for examples/ir_qre_trotter5_compile_survey.py."""

import csv
import json
from dataclasses import replace


def test_target_systems_cover_standard_total_qubit_ladder():
    """The default compile survey spans 8-, 10-, and 12-qubit targets."""
    from examples.ir_qre_trotter5_compile_survey import DEFAULT_SYSTEM_LABELS, systems_by_label

    systems = systems_by_label()
    total_qubits = {
        label: systems[label].active_orbitals * 2 + 4 for label in DEFAULT_SYSTEM_LABELS
    }

    assert total_qubits["H2"] == 8
    assert total_qubits["HeH+"] == 8
    assert total_qubits["H3+"] == 10
    assert total_qubits["LiH"] == 12
    assert total_qubits["H3O+ STO-3G"] == 12


def test_profile_result_to_record_uses_trotter5_metadata(mock_result, parent_data):
    """Profile results are normalized into the trotter-5 compile schema."""
    from examples.ir_qre_trotter5_compile_survey import profile_result_to_record

    result = replace(
        mock_result,
        molecule="LiH",
        n_system_qubits=8,
        n_estimation_wires=4,
        n_trotter=5,
        n_terms=193,
        ir_scale=3860,
        mode="dynamic",
        ir_analysis=[
            ("mlir", 10.0, 100),
            ("QuantumCompilationStage", 20.0, 200),
            ("HLOLoweringStage", 40.0, 300),
            ("BufferizationStage", 100.0, 400),
        ],
    )

    record = profile_result_to_record("LiH", result, parent_data)

    assert record["classical_case"] == "LiH dynamic 4-bit 5-trotter"
    assert record["classical_status"] == "measured"
    assert record["n_trotter"] == 5
    assert record["n_estimation_wires"] == 4
    assert record["ir_ops_lower_bound"] == 3860
    assert record["compile_rss_gb"] > 0
    assert record["bufferization_amp"] == 2.5


def test_main_writes_trotter5_json_and_csv(tmp_path, mock_result, parent_data):
    """The CLI writes incremental standardized JSON and CSV outputs."""
    from examples import ir_qre_trotter5_compile_survey as survey

    output_json = tmp_path / "compile.json"
    output_csv = tmp_path / "compile.csv"
    calls = []

    def fake_runner(mol, n_est, n_trotter, mode, timeout_s):
        calls.append((mol.name, n_est, n_trotter, mode, timeout_s))
        result = replace(
            mock_result,
            molecule=mol.name,
            n_system_qubits=mol.active_orbitals * 2,
            n_estimation_wires=n_est,
            n_trotter=n_trotter,
            n_terms=15,
            ir_scale=n_est * n_trotter * 15,
            mode=mode,
        )
        return result, parent_data

    survey.main(
        [
            "--systems",
            "H2",
            "--modes",
            "fixed,dynamic",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        runner=fake_runner,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    rows = payload["systems"]

    assert len(calls) == 2
    assert payload["metadata"]["n_trotter"] == 5
    assert {row["n_trotter"] for row in rows} == {5}
    assert {row["classical_mode"] for row in rows} == {"fixed", "dynamic"}

    with output_csv.open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 2
    assert csv_rows[0]["n_trotter"] == "5"


def test_main_preserves_existing_rows_when_running_subset(tmp_path, mock_result, parent_data):
    """Running a second system appends/replaces cases instead of clearing prior data."""
    from examples import ir_qre_trotter5_compile_survey as survey

    output_json = tmp_path / "compile.json"
    output_csv = tmp_path / "compile.csv"

    def fake_runner(mol, n_est, n_trotter, mode, timeout_s):
        result = replace(
            mock_result,
            molecule=mol.name,
            n_system_qubits=mol.active_orbitals * 2,
            n_estimation_wires=n_est,
            n_trotter=n_trotter,
            n_terms=15,
            ir_scale=n_est * n_trotter * 15,
            mode=mode,
        )
        return result, parent_data

    base_args = [
        "--modes",
        "fixed",
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
    ]
    survey.main(["--systems", "H2", *base_args], runner=fake_runner)
    survey.main(["--systems", "HeH+", *base_args], runner=fake_runner)

    rows = json.loads(output_json.read_text(encoding="utf-8"))["systems"]

    assert [row["label"] for row in rows] == ["H2", "HeH+"]
    assert {row["n_trotter"] for row in rows} == {5}
