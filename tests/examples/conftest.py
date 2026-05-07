# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Shared fixtures for examples/qpe_memory_profile.py tests."""

import os
from io import StringIO

import pytest
from rich.console import Console

from q2m3.profiling.memory import MemorySnapshot, ProfileResult


@pytest.fixture(autouse=True)
def restore_cwd():
    """Restore working directory after each test (safety net for ir_output_dir)."""
    original = os.getcwd()
    yield
    os.chdir(original)


@pytest.fixture
def mock_snapshot():
    """Minimal valid MemorySnapshot for testing output functions."""
    return MemorySnapshot(
        label="test",
        rss_mb=100.0,
        vm_peak_mb=150.0,
        maxrss_mb=200.0,
        maxrss_children_mb=300.0,
        tracemalloc_peak_mb=50.0,
        tracemalloc_current_mb=30.0,
        elapsed_s=5.0,
    )


@pytest.fixture
def mock_result(mock_snapshot):
    """Minimal valid ProfileResult with all three phases populated."""
    return ProfileResult(
        molecule="H2",
        n_system_qubits=4,
        n_estimation_wires=2,
        n_trotter=1,
        n_terms=15,
        ir_scale=30,
        mode="dynamic",
        phase_a=mock_snapshot,
        phase_b=mock_snapshot,
        phase_c=mock_snapshot,
        timeline_peak_mb=250.0,
        timeline_samples=[(0.0, 100.0), (1.0, 200.0), (2.0, 250.0), (3.0, 180.0)],
        ir_analysis=[("mlir", 45.2, 1234), ("LLVMIRTranslation", 251.8, 6789)],
        prob_sum=0.999999,
    )


@pytest.fixture
def capture_console(monkeypatch):
    """Redirect examples.qpe_memory_profile.console output to StringIO.

    Returns (console, buf) tuple. All Rich output goes to buf.
    """
    buf = StringIO()
    console = Console(file=buf, width=120)
    monkeypatch.setattr("examples.qpe_memory_profile.console", console)
    return console, buf


@pytest.fixture
def parent_data():
    """Mock parent-side monitoring data dict for run_both_modes tests."""
    return {
        "peak_rss_mb": 500.0,
        "peak_hwm_mb": 510.0,
        "peak_smaps": {"Rss": 500.0, "Anonymous": 450.0},
        "n_samples": 50,
    }
