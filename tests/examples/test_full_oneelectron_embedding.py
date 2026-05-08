# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Smoke tests for examples/full_oneelectron_embedding.py."""

from types import SimpleNamespace


def _resource(mode: str, lambda_ha: float, toffoli_gates: int, offdiag: float):
    diagnostics = SimpleNamespace(
        delta_h_diag_fro=0.02,
        delta_h_offdiag_fro=offdiag,
        delta_h_offdiag_to_diag=0.25 if offdiag else 0.0,
        delta_h_hermitian_max_abs=0.0,
        delta_h_trace_ha=-0.01,
        delta_nuclear_mm_ha=0.001,
        delta_core_constant_ha=0.0,
        fixed_mo=mode != "none",
        two_electron_tensor_fixed=True,
    )
    return SimpleNamespace(
        embedding_mode=mode,
        hamiltonian_1norm=lambda_ha,
        logical_qubits=115,
        toffoli_gates=toffoli_gates,
        n_system_qubits=4,
        target_error=0.0016,
        n_mm_charges=0 if mode == "none" else 3,
        embedding_diagnostics=diagnostics,
    )


def test_full_oneelectron_embedding_main_smoke(monkeypatch, capsys):
    """The tutorial script prints all embedding resource rows and boundaries."""
    from examples import full_oneelectron_embedding as example

    rows = {
        "none": _resource("none", 1.20, 1_200_000, 0.0),
        "diagonal": _resource("diagonal", 1.21, 1_205_000, 0.0),
        "full_oneelectron": _resource("full_oneelectron", 1.24, 1_220_000, 0.006),
    }

    def fake_estimate_resources(**kwargs):
        if kwargs.get("mm_charges") is None:
            return rows["none"]
        return rows[kwargs["embedding_mode"]]

    monkeypatch.setattr(example, "estimate_resources", fake_estimate_resources)

    example.main()

    output = capsys.readouterr().out
    assert "Full One-Electron Fixed-MO Embedding" in output
    assert "vacuum" in output
    assert "diagonal" in output
    assert "full_oneelectron" in output
    assert "delta_h_offdiag_fro" in output
    assert "not a relaxed solvation energy" in output
