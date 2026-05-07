# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Tests for q2m3.profiling.memory module."""

import sys
import time

import pytest

from q2m3.profiling.memory import (
    MemorySnapshot,
    MemoryTimeline,
    ParentSideMonitor,
    ProfileResult,
    take_snapshot,
)

# Platform guard: /proc filesystem required
LINUX = sys.platform.startswith("linux")


def test_memory_snapshot_dataclass():
    snap = MemorySnapshot(
        label="test",
        rss_mb=100.0,
        vm_peak_mb=200.0,
        maxrss_mb=150.0,
        maxrss_children_mb=0.0,
        tracemalloc_peak_mb=10.0,
        tracemalloc_current_mb=8.0,
    )
    assert snap.label == "test"
    assert snap.rss_mb == 100.0
    assert snap.elapsed_s == 0.0  # default


def test_profile_result_dataclass():
    result = ProfileResult(
        molecule="H2",
        n_system_qubits=4,
        n_estimation_wires=2,
        n_trotter=1,
        n_terms=10,
        ir_scale=20,
    )
    assert result.molecule == "H2"
    assert result.mode == "dynamic"  # default
    assert result.error is None  # default


@pytest.mark.skipif(not LINUX, reason="requires Linux /proc filesystem")
def test_take_snapshot_returns_snapshot():
    snap = take_snapshot("test_snap")
    assert isinstance(snap, MemorySnapshot)
    assert snap.label == "test_snap"
    assert snap.rss_mb > 0
    assert snap.maxrss_mb > 0


@pytest.mark.skipif(not LINUX, reason="requires Linux /proc filesystem")
def test_memory_timeline_context_manager():
    with MemoryTimeline() as timeline:
        time.sleep(0.2)  # Allow some samples to be collected
    assert isinstance(timeline.peak_rss_mb, float)
    assert timeline.peak_rss_mb > 0
    assert len(timeline.samples) > 0


@pytest.mark.skipif(not LINUX, reason="requires Linux /proc filesystem")
def test_parent_side_monitor():
    import subprocess

    # Start a short-lived subprocess
    proc = subprocess.Popen(["python3", "-c", "import time; time.sleep(0.3)"])
    monitor = ParentSideMonitor(proc.pid, interval_s=0.05)
    monitor.start()
    proc.wait()
    monitor.stop()
    assert isinstance(monitor.peak_rss_mb, float)
