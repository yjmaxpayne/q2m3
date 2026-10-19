# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Memory measurement infrastructure for QPE circuit compilation profiling.

Provides multi-layer memory measurement:
  - resource.getrusage(RUSAGE_SELF/RUSAGE_CHILDREN) → process lifetime peak RSS
  - /proc/self/status VmRSS/VmPeak → current/peak RSS (Linux kernel level)
  - tracemalloc → Python heap only

This module has zero project dependencies (stdlib only).
"""

import json
import math
import os
import resource
import signal
import subprocess
import sys
import threading
import time
import tracemalloc
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MemorySnapshot:
    """Multi-layer memory snapshot at a single point in time."""

    label: str
    rss_mb: float  # /proc/self/status VmRSS
    vm_peak_mb: float  # /proc/self/status VmPeak
    maxrss_mb: float  # resource.getrusage RUSAGE_SELF (cumulative peak)
    maxrss_children_mb: float  # resource.getrusage RUSAGE_CHILDREN peak
    tracemalloc_peak_mb: float  # Python heap peak since last reset
    tracemalloc_current_mb: float  # Python heap current
    elapsed_s: float = 0.0


@dataclass
class ProfileResult:
    """Complete profiling result for one parameter combination."""

    molecule: str
    n_system_qubits: int
    n_estimation_wires: int
    n_trotter: int
    n_terms: int
    ir_scale: int  # n_est × n_trotter × n_terms
    mode: str = "dynamic"  # "fixed" or "dynamic"
    phase_a: MemorySnapshot | None = None
    phase_b: MemorySnapshot | None = None
    phase_c: MemorySnapshot | None = None
    timeline_peak_mb: float = 0.0
    timeline_samples: list = field(default_factory=list)
    ir_analysis: list = field(default_factory=list)  # list of (stage, size_kb, lines)
    prob_sum: float = 0.0
    error: str | None = None


def read_proc_status(pid: int | str = "self") -> dict[str, float]:
    """Parse /proc/[pid]/status for VmRSS and VmPeak (Linux only).

    Args:
        pid: Process ID to read, or "self" for current process.
    """
    result = {"VmRSS": 0.0, "VmPeak": 0.0, "VmSize": 0.0, "VmHWM": 0.0}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                for key in result:
                    if line.startswith(key + ":"):
                        # Format: "VmRSS:    123456 kB"
                        result[key] = float(line.split()[1]) / 1024.0  # kB → MB
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return result


def read_smaps_rollup(pid: int | str = "self") -> dict[str, float]:
    """Parse /proc/[pid]/smaps_rollup to categorize RSS into memory types.

    Returns dict with MB values for:
      - Rss: total resident
      - Pss: proportional share
      - Anonymous: heap + anonymous mmap (C++ allocations, LLVM JIT code)
      - LazyFree, AnonHugePages, etc.
    """
    result = {}
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    try:
                        result[key] = float(parts[1]) / 1024.0  # kB → MB
                    except ValueError:
                        pass
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return result


def take_snapshot(label: str) -> MemorySnapshot:
    """Take a comprehensive memory snapshot from all measurement layers.

    Captures both RUSAGE_SELF (Python process) and RUSAGE_CHILDREN (catalyst
    subprocess) to detect the measurement blind spot where MLIR→LLVM
    compilation happens in a child process.
    """
    proc = read_proc_status()
    ru_self = resource.getrusage(resource.RUSAGE_SELF)
    ru_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    # ru_maxrss is in KB on Linux
    maxrss_mb = ru_self.ru_maxrss / 1024.0
    maxrss_children_mb = ru_children.ru_maxrss / 1024.0

    tm_current, tm_peak = tracemalloc.get_traced_memory()

    return MemorySnapshot(
        label=label,
        rss_mb=proc["VmRSS"],
        vm_peak_mb=proc["VmPeak"],
        maxrss_mb=maxrss_mb,
        maxrss_children_mb=maxrss_children_mb,
        tracemalloc_peak_mb=tm_peak / (1024 * 1024),
        tracemalloc_current_mb=tm_current / (1024 * 1024),
    )


class MemoryTimeline:
    """Background daemon thread sampling /proc/self/status VmRSS at 100ms intervals.

    Also captures /proc/self/smaps_rollup at peak RSS to categorize memory.
    """

    def __init__(self, interval_s: float = 0.1):
        self._interval = interval_s
        self._samples: list[tuple[float, float]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._peak_rss = 0.0
        self._peak_smaps: dict[str, float] = {}

    def __enter__(self):
        self._start_time = time.monotonic()
        self._samples.clear()
        self._stop_event.clear()
        self._peak_rss = 0.0
        self._peak_smaps = {}
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _sample_loop(self):
        while not self._stop_event.is_set():
            elapsed = time.monotonic() - self._start_time
            proc = read_proc_status()
            rss = proc["VmRSS"]
            self._samples.append((elapsed, rss))
            # Capture smaps_rollup at new peak
            if rss > self._peak_rss:
                self._peak_rss = rss
                self._peak_smaps = read_smaps_rollup()
            self._stop_event.wait(self._interval)

    @property
    def peak_rss_mb(self) -> float:
        if not self._samples:
            return 0.0
        return max(s[1] for s in self._samples)

    @property
    def peak_smaps(self) -> dict[str, float]:
        """Memory categorization captured at peak RSS moment."""
        return dict(self._peak_smaps)

    @property
    def samples(self) -> list[tuple[float, float]]:
        return list(self._samples)


class ParentSideMonitor:
    """Monitor a child process's RSS from the parent side via /proc/PID/status.

    This provides an independent measurement that validates the child's
    self-reported ru_maxrss. Critical for detecting measurement blind spots
    where getrusage misses certain allocation patterns (mmap, LLVM JIT, etc.).
    """

    def __init__(self, pid: int, interval_s: float = 0.1):
        self._pid = pid
        self._interval = interval_s
        self._samples: list[tuple[float, float]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._peak_rss = 0.0
        self._peak_smaps: dict[str, float] = {}
        self._peak_hwm = 0.0

    def start(self):
        self._start_time = time.monotonic()
        self._samples.clear()
        self._stop_event.clear()
        self._peak_rss = 0.0
        self._peak_smaps = {}
        self._peak_hwm = 0.0
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _sample_loop(self):
        while not self._stop_event.is_set():
            elapsed = time.monotonic() - self._start_time
            proc = read_proc_status(self._pid)
            rss = proc["VmRSS"]
            hwm = proc["VmHWM"]
            if rss > 0:
                self._samples.append((elapsed, rss))
                if rss > self._peak_rss:
                    self._peak_rss = rss
                    self._peak_smaps = read_smaps_rollup(self._pid)
                if hwm > self._peak_hwm:
                    self._peak_hwm = hwm
            self._stop_event.wait(self._interval)

    @property
    def peak_rss_mb(self) -> float:
        return self._peak_rss

    @property
    def peak_hwm_mb(self) -> float:
        """VmHWM: kernel-tracked high water mark for RSS (most reliable)."""
        return self._peak_hwm

    @property
    def peak_smaps(self) -> dict[str, float]:
        return dict(self._peak_smaps)

    @property
    def samples(self) -> list[tuple[float, float]]:
        return list(self._samples)


def _proc_rss_bytes(pid: int) -> int:
    """Read VmRSS for one process and return bytes, or zero when unavailable."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key == "VmRSS":
                return int(value.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def _process_tree_pids(pid: int) -> set[int]:
    """Return the process and all visible descendants."""
    found = {pid}
    pending = [pid]
    while pending:
        current = pending.pop()
        for path in Path(f"/proc/{current}/task").glob("*/children"):
            try:
                children = {int(child) for child in path.read_text().split()} - found
            except (OSError, ValueError):
                continue
            found.update(children)
            pending.extend(children)
    return found


def _validate_command(command: Any) -> None:
    if not isinstance(command, list) or not command:
        raise ValueError("command must be a non-empty list of strings")
    for item in command:
        if not isinstance(item, str) or "\0" in item:
            raise ValueError("command items must be strings without NUL")
    if not command[0]:
        raise ValueError("command executable must not be empty")


def _validate_budget(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _validate_env(env: Any) -> dict[str, str]:
    if env is None:
        return {}
    if not isinstance(env, Mapping):
        raise ValueError("env must be a string mapping")
    result = {}
    for key, value in env.items():
        if not isinstance(key, str) or not key or "=" in key or "\0" in key:
            raise ValueError("env keys must be non-empty strings without '=' or NUL")
        if not isinstance(value, str) or "\0" in value:
            raise ValueError("env values must be strings without NUL")
        result[key] = value
    return result


def _resolve_cwd(cwd: Any) -> Path | None:
    if cwd is None:
        return None
    path = Path(cwd).resolve()
    if not path.exists():
        raise FileNotFoundError(f"cwd does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"cwd is not a directory: {path}")
    return path


def _require_linux_proc() -> None:
    if not sys.platform.startswith("linux") or not Path("/proc").is_dir():
        raise RuntimeError("measure_command requires Linux with /proc support")


def measure_command(
    command: list[str],
    directory: Path,
    *,
    wall_s: float = 900.0,
    rss_mb: float = 8192.0,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run a command and record wall time, exit state, and process-tree RSS.

    The monitor supports Linux systems with /proc. It polls the command
    process and its current descendants every 0.01 seconds. The observer
    process is excluded. RSS values are decimal megabytes. Polled samples
    do not bound arbitrary transient allocations.

    Args:
        command: Non-empty argv list. Each item is a string without NUL.
        directory: New evidence directory. A relative path is relative to
            the caller's current directory.
        wall_s: Positive finite wall-time budget in seconds.
        rss_mb: Positive finite process-tree RSS budget in decimal megabytes.
        cwd: Working directory for the command. None uses the caller's
            current directory. The parent process does not change directory.
        env: String mapping merged over the default child environment.

    Returns:
        A record with command, exit_code, reason, wall_s, peak_tree_rss_mb,
        poll_s, rss_budget_mb, wall_budget_s, scope, and limitation.

    Raises:
        ValueError: An input value is invalid. The function rejects it before
            it creates the output directory or starts a process.
        FileExistsError: The output directory already exists.
        RuntimeError: The platform lacks Linux /proc support.
        OSError: The working directory is missing or the command cannot start.
    """
    _validate_command(command)
    wall_budget_s = _validate_budget(wall_s, "wall_s")
    rss_budget_mb = _validate_budget(rss_mb, "rss_mb")
    env_overrides = _validate_env(env)
    resolved_cwd = _resolve_cwd(cwd)
    _require_linux_proc()

    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=False)

    child_env = dict(os.environ)
    child_env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        JAX_PLATFORMS="cpu",
    )
    child_env.update(env_overrides)

    started = time.monotonic()
    peak = 0
    reason = None
    process = None
    with (
        (output_directory / "stdout.log").open("w") as log,
        (output_directory / "rss.jsonl").open("w") as trace,
    ):
        try:
            process = subprocess.Popen(
                command,
                shell=False,
                cwd=resolved_cwd,
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while process.poll() is None:
                rss = sum(_proc_rss_bytes(pid) for pid in _process_tree_pids(process.pid))
                peak = max(peak, rss)
                elapsed = time.monotonic() - started
                trace.write(json.dumps([elapsed, rss]) + "\n")
                if elapsed >= wall_budget_s or rss >= rss_budget_mb * 1_000_000:
                    reason = (
                        "wall_budget_exceeded"
                        if elapsed >= wall_budget_s
                        else "rss_budget_exceeded"
                    )
                    break
                time.sleep(0.01)
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

    record = {
        "command": list(command),
        "exit_code": process.returncode,
        "reason": reason,
        "wall_s": time.monotonic() - started,
        "peak_tree_rss_mb": peak / 1_000_000,
        "poll_s": 0.01,
        "rss_budget_mb": rss_budget_mb,
        "wall_budget_s": wall_budget_s,
        "scope": "interpreter launch through exit; simultaneous descendants; observer excluded",
        "limitation": "polled observations do not bound arbitrary transient allocations",
    }
    (output_directory / "resources.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
