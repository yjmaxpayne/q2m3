# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Tests for q2m3.profiling.memory module."""

import json
import os
import signal
import sys
import textwrap
import time
from pathlib import Path

import pytest

from q2m3.profiling.memory import (
    MemorySnapshot,
    MemoryTimeline,
    ParentSideMonitor,
    ProfileResult,
    measure_command,
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


@pytest.mark.skipif(not LINUX, reason="requires Linux /proc filesystem")
def test_measure_command_success_and_nonzero_exit(tmp_path):
    command = [sys.executable, "-c", "print('hello')"]
    record = measure_command(command, tmp_path / "ok")

    assert record["exit_code"] == 0
    assert record["reason"] is None
    assert record["command"] == command
    assert record["command"] is not command
    assert set(record) == {
        "command",
        "exit_code",
        "reason",
        "wall_s",
        "peak_tree_rss_mb",
        "poll_s",
        "rss_budget_mb",
        "wall_budget_s",
        "scope",
        "limitation",
    }
    assert record["poll_s"] == 0.01
    assert (tmp_path / "ok" / "stdout.log").read_text().strip() == "hello"
    assert json.loads((tmp_path / "ok" / "resources.json").read_text()) == record

    failed_command = [sys.executable, "-c", "raise SystemExit(3)"]
    failed = measure_command(failed_command, tmp_path / "exit")

    assert failed["exit_code"] == 3
    assert failed["reason"] is None
    assert json.loads((tmp_path / "exit" / "resources.json").read_text()) == failed


@pytest.mark.skipif(not LINUX, reason="requires Linux /proc filesystem")
def test_measure_command_timeout_and_descendant_rss(tmp_path):
    timed = measure_command(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "timeout",
        wall_s=0.3,
    )
    assert timed["reason"] == "wall_budget_exceeded"
    assert timed["exit_code"] != 0
    assert timed["wall_s"] < 5.0

    script = textwrap.dedent("""
        import subprocess
        import sys
        import time

        child = subprocess.Popen(
            [sys.executable, "-c", "import time; data = b'x' * 80000000; time.sleep(30)"]
        )
        print(child.pid, flush=True)
        time.sleep(30)
        """)
    killed = measure_command(
        [sys.executable, "-c", script],
        tmp_path / "rss",
        wall_s=10.0,
        rss_mb=60.0,
    )

    assert killed["reason"] == "rss_budget_exceeded"
    assert killed["peak_tree_rss_mb"] >= 60.0
    assert killed["exit_code"] != 0


@pytest.mark.skipif(not LINUX, reason="requires Linux /proc filesystem")
def test_measure_command_cleans_group_after_leader_exit(tmp_path):
    script = textwrap.dedent("""
        import subprocess
        import sys

        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        print(child.pid, flush=True)
        """)
    directory = tmp_path / "group"
    record = measure_command([sys.executable, "-c", script], directory)

    assert record["exit_code"] == 0
    assert record["reason"] is None
    child_pid = int((directory / "stdout.log").read_text().split()[0])
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            status_path = Path(f"/proc/{child_pid}/status")
            if not status_path.exists():
                break
            state = status_path.read_text()
            if "State:\tZ" in state or "State:\tX" in state:
                break
            time.sleep(0.01)
        else:
            pytest.fail(f"descendant {child_pid} is still alive")
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.skipif(not LINUX, reason="requires Linux /proc filesystem")
def test_measure_command_cwd_env_and_rejection(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    before = Path.cwd()
    record = measure_command(
        [
            sys.executable,
            "-c",
            (
                "import os; print(os.getcwd()); print(os.environ['Q2M3_TEST_ENV']); "
                "print(os.environ['OMP_NUM_THREADS']); print(os.environ['JAX_PLATFORMS'])"
            ),
        ],
        tmp_path / "cwd",
        cwd=work,
        env={"Q2M3_TEST_ENV": "custom", "OMP_NUM_THREADS": "7"},
    )

    assert record["exit_code"] == 0
    lines = (tmp_path / "cwd" / "stdout.log").read_text().splitlines()
    assert Path(lines[0]) == work.resolve()
    assert lines[1] == "custom"
    assert lines[2] == "7"
    assert lines[3] == "cpu"
    assert Path.cwd() == before

    command = [sys.executable, "-c", "pass"]
    cases = [
        ("empty", lambda directory: measure_command([], directory)),
        ("string", lambda directory: measure_command("echo hi", directory)),
        ("non-string", lambda directory: measure_command([sys.executable, 3], directory)),
        ("empty-executable", lambda directory: measure_command([""], directory)),
        ("nul", lambda directory: measure_command([sys.executable, "bad\0arg"], directory)),
        ("zero-wall", lambda directory: measure_command(command, directory, wall_s=0)),
        ("negative-wall", lambda directory: measure_command(command, directory, wall_s=-1)),
        ("nan-wall", lambda directory: measure_command(command, directory, wall_s=float("nan"))),
        ("inf-wall", lambda directory: measure_command(command, directory, wall_s=float("inf"))),
        ("bool-wall", lambda directory: measure_command(command, directory, wall_s=True)),
        ("zero-rss", lambda directory: measure_command(command, directory, rss_mb=0)),
        ("nan-rss", lambda directory: measure_command(command, directory, rss_mb=float("nan"))),
        ("bool-rss", lambda directory: measure_command(command, directory, rss_mb=False)),
        (
            "env-empty-key",
            lambda directory: measure_command(command, directory, env={"": "x"}),
        ),
        (
            "env-equals-key",
            lambda directory: measure_command(command, directory, env={"A=B": "x"}),
        ),
        (
            "env-nul-value",
            lambda directory: measure_command(command, directory, env={"A": "x\0y"}),
        ),
        (
            "env-non-string",
            lambda directory: measure_command(command, directory, env={"A": 1}),
        ),
        (
            "env-non-mapping",
            lambda directory: measure_command(command, directory, env=[("A", "x")]),
        ),
    ]
    for name, call in cases:
        directory = tmp_path / f"reject-{name}"
        with pytest.raises(ValueError):
            call(directory)
        assert not directory.exists(), name

    with pytest.raises(OSError):
        measure_command(command, tmp_path / "bad-cwd", cwd=tmp_path / "missing")
    assert not (tmp_path / "bad-cwd").exists()

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        measure_command(command, existing)

    missing_output = tmp_path / "missing-command"
    with pytest.raises(FileNotFoundError):
        measure_command([str(tmp_path / "no-such-binary")], missing_output)
    assert (missing_output / "stdout.log").exists()
    assert not (missing_output / "resources.json").exists()


def test_measure_command_exports():
    import q2m3.profiling as profiling
    from q2m3.profiling import measure_command as exported
    from q2m3.profiling.memory import measure_command as implemented

    assert exported is implemented
    assert "measure_command" in profiling.__all__
    assert {"take_snapshot", "MemoryTimeline"} <= set(profiling.__all__)
