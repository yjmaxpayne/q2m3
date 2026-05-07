# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Tests for qpe_profiler module."""

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

LINUX = sys.platform.startswith("linux")


@pytest.fixture
def h2_mol():
    from q2m3.molecule import MoleculeConfig

    return MoleculeConfig(
        name="H2",
        symbols=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        charge=0,
        active_electrons=2,
        active_orbitals=2,
    )


@pytest.mark.skipif(not LINUX, reason="take_snapshot requires Linux /proc")
def test_profile_hamiltonian_build_returns_correct_types(h2_mol):
    """profile_hamiltonian_build returns (snapshot, ops, coeffs, hf_state, circuit_params)."""
    from q2m3.profiling.qpe_profiler import profile_hamiltonian_build

    snapshot, ops, coeffs, hf_state, circuit_params = profile_hamiltonian_build(
        h2_mol, n_est=2, n_trotter=1
    )

    assert hasattr(snapshot, "rss_mb")
    assert isinstance(ops, list)
    assert isinstance(coeffs, list)
    assert isinstance(hf_state, np.ndarray)
    assert isinstance(circuit_params, dict)
    # 精确断言 circuit_params 必须包含的所有键（与原型 L442-452 对齐）
    assert "n_system_qubits" in circuit_params
    assert "n_estimation_wires" in circuit_params
    assert "base_time" in circuit_params
    assert "n_trotter" in circuit_params
    assert "n_terms" in circuit_params


def test_profile_qpe_profiler_importable():
    """Sanity check: module can be imported without Catalyst installed."""
    from q2m3.profiling import qpe_profiler  # noqa: F401

    assert hasattr(qpe_profiler, "profile_hamiltonian_build")
    assert hasattr(qpe_profiler, "profile_qjit_compilation")
    assert hasattr(qpe_profiler, "profile_qjit_compilation_fixed")
    assert hasattr(qpe_profiler, "profile_execution")
