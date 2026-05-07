# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Tests for orchestrator module. Subprocess calls are mocked."""

import importlib.util
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from q2m3.profiling.memory import ProfileResult
from q2m3.profiling.orchestrator import (
    H2_SWEEP_GRID,
    MOLECULES,
    run_both_modes,
    run_single_profile,
    run_sweep,
)

LINUX = sys.platform.startswith("linux")
# orchestrator.py imports qpe_profiler which conditionally imports catalyst;
# guard tests that patch internal functions behind CATALYST_AVAILABLE
CATALYST_AVAILABLE = importlib.util.find_spec("catalyst") is not None


def test_molecules_dict_has_h2_and_h3o():
    assert "h2" in MOLECULES
    assert "h3o" in MOLECULES
    from q2m3.molecule import MoleculeConfig

    assert isinstance(MOLECULES["h2"], MoleculeConfig)
    assert isinstance(MOLECULES["h3o"], MoleculeConfig)


def test_h2_sweep_grid_is_list_of_tuples():
    assert isinstance(H2_SWEEP_GRID, list)
    assert len(H2_SWEEP_GRID) > 0
    for item in H2_SWEEP_GRID:
        assert isinstance(item, tuple)
        assert len(item) == 2
        n_est, n_trotter = item
        assert isinstance(n_est, int)
        assert isinstance(n_trotter, int)


@pytest.mark.skipif(not LINUX, reason="take_snapshot requires Linux /proc")
@pytest.mark.skipif(
    not CATALYST_AVAILABLE, reason="requires catalyst (orchestrator imports qpe_profiler)"
)
def test_run_single_profile_calls_on_progress():
    """on_progress callback should be called during profile run."""
    progress_messages = []

    def capture_progress(msg: str):
        progress_messages.append(msg)

    mock_snap = MagicMock()
    mock_snap.rss_mb = 100.0
    mock_snap.vm_peak_mb = 200.0
    mock_snap.maxrss_mb = 200.0
    mock_snap.maxrss_children_mb = 0.0
    mock_snap.tracemalloc_peak_mb = 10.0
    mock_snap.tracemalloc_current_mb = 5.0
    mock_snap.elapsed_s = 1.0
    mock_snap.label = "mock"
    mock_timeline = MagicMock()
    mock_timeline.peak_rss_mb = 200.0
    mock_timeline.samples = []
    mock_compiled_fn = MagicMock()
    mock_compiled_fn.return_value = np.array([0.25, 0.25, 0.25, 0.25])

    with (
        patch("q2m3.profiling.orchestrator.profile_hamiltonian_build") as mock_a,
        patch("q2m3.profiling.orchestrator.profile_qjit_compilation") as mock_b,
        patch("q2m3.profiling.orchestrator.profile_execution") as mock_c,
    ):
        mock_a.return_value = (
            mock_snap,
            [MagicMock()],  # ops
            [0.5],  # coeffs
            np.array([1, 1, 0, 0]),  # hf_state
            {
                "n_system_qubits": 4,
                "n_terms": 2,
                "base_time": 0.1,
                "n_estimation_wires": 2,
                "n_trotter": 1,
            },
        )
        mock_b.return_value = (mock_snap, mock_timeline, [], mock_compiled_fn)
        mock_c.return_value = (mock_snap, 1.0)

        result = run_single_profile(
            MOLECULES["h2"],
            n_est=2,
            n_trotter=1,
            mode="dynamic",
            on_progress=capture_progress,
        )

    assert isinstance(result, ProfileResult)
    assert len(progress_messages) >= 1


@pytest.mark.skipif(not LINUX, reason="take_snapshot requires Linux /proc")
@pytest.mark.skipif(
    not CATALYST_AVAILABLE, reason="requires catalyst (orchestrator imports qpe_profiler)"
)
def test_run_single_profile_without_progress_callback():
    """run_single_profile works without on_progress callback."""
    mock_snap = MagicMock()
    mock_snap.rss_mb = 100.0
    mock_snap.vm_peak_mb = 200.0
    mock_snap.maxrss_mb = 200.0
    mock_snap.maxrss_children_mb = 0.0
    mock_snap.tracemalloc_peak_mb = 10.0
    mock_snap.tracemalloc_current_mb = 5.0
    mock_snap.elapsed_s = 1.0
    mock_snap.label = "mock"
    mock_timeline = MagicMock()
    mock_timeline.peak_rss_mb = 200.0
    mock_timeline.samples = []
    mock_compiled_fn = MagicMock()
    mock_compiled_fn.return_value = np.array([0.25, 0.25, 0.25, 0.25])

    with (
        patch("q2m3.profiling.orchestrator.profile_hamiltonian_build") as mock_a,
        patch("q2m3.profiling.orchestrator.profile_qjit_compilation") as mock_b,
        patch("q2m3.profiling.orchestrator.profile_execution") as mock_c,
    ):
        mock_a.return_value = (
            mock_snap,
            [MagicMock()],
            [0.5],
            np.array([1, 1, 0, 0]),
            {
                "n_system_qubits": 4,
                "n_terms": 2,
                "base_time": 0.1,
                "n_estimation_wires": 2,
                "n_trotter": 1,
            },
        )
        mock_b.return_value = (mock_snap, mock_timeline, [], mock_compiled_fn)
        mock_c.return_value = (mock_snap, 1.0)

        # Should not raise even without callback
        result = run_single_profile(
            MOLECULES["h2"],
            n_est=2,
            n_trotter=1,
            mode="dynamic",
        )

    assert isinstance(result, ProfileResult)


def test_run_both_modes_returns_four_tuple():
    """run_both_modes returns (fixed_result, dynamic_result, parent_fixed, parent_dynamic)."""
    mock_fixed = ProfileResult(
        molecule="h2",
        n_system_qubits=4,
        n_estimation_wires=2,
        n_trotter=1,
        n_terms=10,
        ir_scale=20,
        mode="fixed",
    )
    mock_dynamic = ProfileResult(
        molecule="h2",
        n_system_qubits=4,
        n_estimation_wires=2,
        n_trotter=1,
        n_terms=10,
        ir_scale=20,
        mode="dynamic",
    )

    with patch("q2m3.profiling.orchestrator._run_mode_in_subprocess") as mock_spawn:
        # Two subprocess calls: one for fixed, one for dynamic
        mock_spawn.side_effect = [
            (
                mock_fixed,
                {"peak_rss_mb": 500.0, "peak_hwm_mb": 510.0, "peak_smaps": {}, "n_samples": 10},
            ),
            (
                mock_dynamic,
                {"peak_rss_mb": 800.0, "peak_hwm_mb": 820.0, "peak_smaps": {}, "n_samples": 10},
            ),
        ]
        result_fixed, result_dynamic, parent_fixed, parent_dynamic = run_both_modes(
            "h2", n_est=2, n_trotter=1
        )

    assert result_fixed.mode == "fixed"
    assert result_dynamic.mode == "dynamic"
    assert "peak_rss_mb" in parent_fixed
    assert "peak_rss_mb" in parent_dynamic


def test_run_both_modes_with_progress_callback():
    """on_progress callback is called during run_both_modes execution."""
    progress_messages = []

    def capture_progress(msg: str):
        progress_messages.append(msg)

    mock_fixed = ProfileResult(
        molecule="h2",
        n_system_qubits=4,
        n_estimation_wires=2,
        n_trotter=1,
        n_terms=10,
        ir_scale=20,
        mode="fixed",
    )
    mock_dynamic = ProfileResult(
        molecule="h2",
        n_system_qubits=4,
        n_estimation_wires=2,
        n_trotter=1,
        n_terms=10,
        ir_scale=20,
        mode="dynamic",
    )

    with patch("q2m3.profiling.orchestrator._run_mode_in_subprocess") as mock_spawn:
        mock_spawn.side_effect = [
            (
                mock_fixed,
                {"peak_rss_mb": 500.0, "peak_hwm_mb": 510.0, "peak_smaps": {}, "n_samples": 10},
            ),
            (
                mock_dynamic,
                {"peak_rss_mb": 800.0, "peak_hwm_mb": 820.0, "peak_smaps": {}, "n_samples": 10},
            ),
        ]
        run_both_modes("h2", n_est=2, n_trotter=1, on_progress=capture_progress)

    assert len(progress_messages) >= 1


def test_run_both_modes_without_progress_callback():
    """run_both_modes works without on_progress callback."""
    mock_fixed = ProfileResult(
        molecule="h2",
        n_system_qubits=4,
        n_estimation_wires=2,
        n_trotter=1,
        n_terms=10,
        ir_scale=20,
        mode="fixed",
    )
    mock_dynamic = ProfileResult(
        molecule="h2",
        n_system_qubits=4,
        n_estimation_wires=2,
        n_trotter=1,
        n_terms=10,
        ir_scale=20,
        mode="dynamic",
    )

    with patch("q2m3.profiling.orchestrator._run_mode_in_subprocess") as mock_spawn:
        mock_spawn.side_effect = [
            (
                mock_fixed,
                {"peak_rss_mb": 500.0, "peak_hwm_mb": 510.0, "peak_smaps": {}, "n_samples": 10},
            ),
            (
                mock_dynamic,
                {"peak_rss_mb": 800.0, "peak_hwm_mb": 820.0, "peak_smaps": {}, "n_samples": 10},
            ),
        ]
        # Should not raise even without callback
        result = run_both_modes("h2", n_est=2, n_trotter=1)

    assert len(result) == 4


def test_run_sweep_returns_results_for_all_grid_points():
    """run_sweep returns dict keyed by (n_est, n_trotter) for all grid points (P2-F)."""
    test_grid = [(2, 1), (4, 2)]
    mock_result = ProfileResult(
        molecule="h2",
        n_system_qubits=4,
        n_estimation_wires=2,
        n_trotter=1,
        n_terms=10,
        ir_scale=20,
        mode="dynamic",
    )

    with patch("q2m3.profiling.orchestrator._run_mode_in_subprocess") as mock_spawn:
        mock_spawn.return_value = (
            mock_result,
            {"peak_rss_mb": 500.0, "peak_hwm_mb": 510.0, "peak_smaps": {}, "n_samples": 10},
        )
        results = run_sweep(mol_key="h2", mode="dynamic", grid=test_grid)

    assert isinstance(results, dict), "run_sweep must return dict[tuple[int,int], ProfileResult]"
    assert len(results) == len(test_grid), "dict must have one entry per grid point"
    for n_est, n_trotter in test_grid:
        assert (n_est, n_trotter) in results, f"key ({n_est}, {n_trotter}) missing from results"


def test_run_sweep_skips_failed_grid_points():
    """run_sweep continues when a grid point raises an exception (skip-on-error behavior)."""
    test_grid = [(2, 1), (4, 2)]
    mock_result = ProfileResult(
        molecule="h2",
        n_system_qubits=4,
        n_estimation_wires=2,
        n_trotter=1,
        n_terms=10,
        ir_scale=20,
        mode="dynamic",
    )

    with patch("q2m3.profiling.orchestrator._run_mode_in_subprocess") as mock_spawn:
        # First point succeeds, second raises
        mock_spawn.side_effect = [
            (
                mock_result,
                {"peak_rss_mb": 500.0, "peak_hwm_mb": 510.0, "peak_smaps": {}, "n_samples": 10},
            ),
            RuntimeError("subprocess failed"),
        ]
        # Should not raise; failed point is skipped
        results = run_sweep(mol_key="h2", mode="dynamic", grid=test_grid)

    assert isinstance(results, dict)
    assert (2, 1) in results, "successful grid point must be present"
    assert (4, 2) not in results, "failed grid point must be absent (skipped)"


def test_orchestrator_no_console_print():
    """Verify orchestrator module has no console.print calls at import time."""
    import inspect

    import q2m3.profiling.orchestrator as mod

    source = inspect.getsource(mod)
    assert "console.print" not in source, "orchestrator must not use console.print"
