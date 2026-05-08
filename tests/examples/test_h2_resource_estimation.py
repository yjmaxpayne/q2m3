# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Smoke tests for examples/h2_resource_estimation.py."""

from types import SimpleNamespace


def test_h2_resource_estimation_main_smoke(monkeypatch, capsys):
    """The example main path runs and prints vacuum/solvated resource rows."""
    from examples import h2_resource_estimation as example

    vacuum = SimpleNamespace(
        logical_qubits=115,
        toffoli_gates=1_200_000,
        hamiltonian_1norm=1.25,
        n_system_qubits=4,
        target_error=0.0016,
        n_mm_charges=0,
    )
    solvated = SimpleNamespace(
        logical_qubits=115,
        toffoli_gates=1_210_000,
        hamiltonian_1norm=1.27,
        n_system_qubits=4,
        target_error=0.0016,
        n_mm_charges=6,
    )
    comparison = SimpleNamespace(
        vacuum=vacuum,
        solvated=solvated,
        delta_gates_percent=0.8,
        delta_lambda_percent=1.6,
    )

    monkeypatch.setattr(example, "estimate_resources", lambda **_kwargs: vacuum)
    monkeypatch.setattr(example, "compare_vacuum_solvated", lambda **_kwargs: comparison)

    example.main()

    output = capsys.readouterr().out
    assert "H2 EFTQC Resource Estimation" in output
    assert "Vacuum vs Solvated Comparison" in output
    assert "MM charges: 6 point charges" in output
