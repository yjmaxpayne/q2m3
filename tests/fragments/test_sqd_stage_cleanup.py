"""SQD stage exit paths leave no process of the stage behind.

On timeout, KeyboardInterrupt, SIGTERM and normal exit the parent kills the child, the
child's descendants and every process of the user that carries the stage marker. The
child turns SIGTERM into SystemExit and gets SIGTERM when its parent dies.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from q2m3.fragments import sqd_stage
from q2m3.fragments.sqd_stage import SubprocessSQDRunner, parent_timeout_s
from q2m3.sqd.config import ReferenceConfig
from q2m3.sqd.exceptions import ReferenceTimeoutError

REPO = Path(__file__).resolve().parents[2]
SLEEPER = "tests.fragments._child_sleeper"
DRIVER = "tests.fragments._stage_driver"
WATCH = "tests.fragments._child_watch"
STAGE_ID = "Q2M3_SQD_STAGE_ID"
SURVIVORS = "SQD stage processes survived cleanup: [{}]"
# Short enough for a test, and longer than the start-up and SQD run of one small job.
WALL_BUDGET_S = 60.0


def _alive(pid: int) -> bool:
    """A zombie has already exited, so it does not count as alive."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _environ(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    return dict(entry.decode().split("=", 1) for entry in raw if b"=" in entry)


def _carrying(name: str, value: str) -> list[int]:
    """Pids whose exec-time environment has ``name=value``."""
    pids = []
    for item in os.listdir("/proc"):
        if item.isdigit():
            with contextlib.suppress(OSError):
                if _environ(int(item)).get(name) == value:
                    pids.append(int(item))
    return pids


def _wait_for(condition, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while not (value := condition()) and time.monotonic() < deadline:
        time.sleep(0.05)
    return value


def _kill(*pids: int | None) -> None:
    for pid in pids:
        if pid is not None and _alive(pid):
            os.kill(pid, signal.SIGKILL)


def _assert_stage_gone(probe: dict) -> None:
    assert _carrying(STAGE_ID, probe["stage_id"]) == []
    assert not _alive(probe["orphan"])
    assert not any(_alive(probe[key]) for key in ("plain", "plain_child") if key in probe)


@pytest.fixture
def child_path(monkeypatch):
    # The child imports tests.fragments, and its cwd is the job directory.
    monkeypatch.setenv("PYTHONPATH", f"{REPO}:{REPO / 'src'}")


@pytest.fixture
def sleeper(child_path, monkeypatch, tmp_path):
    """Probe path of the sleeper child; kills what a failing run leaves behind."""
    probe = tmp_path / "probe.json"
    monkeypatch.setenv("Q2M3_TEST_CHILD_PROBE", str(probe))
    yield probe
    if probe.exists():
        data = json.loads(probe.read_text())
        _kill(
            *_carrying(STAGE_ID, data["stage_id"]),
            data["orphan"],
            data.get("plain"),
            data.get("plain_child"),
        )


def _run_sleeper(monkeypatch, mode: str, **runner_kwargs):
    monkeypatch.setenv("Q2M3_TEST_CHILD_MODE", mode)
    return SubprocessSQDRunner(child_module=SLEEPER, **runner_kwargs).run([])


def test_timeout_kills_the_child_its_descendants_and_marked_orphans(sleeper, monkeypatch):
    with pytest.raises(ReferenceTimeoutError, match="2 s"):
        _run_sleeper(monkeypatch, "sleep", timeout_s=2)

    probe = json.loads(sleeper.read_text())
    # Only the marker scan reaches the orphan, and only the descendant snapshot reaches
    # the plain child and its own child, which have no marker.
    assert probe["orphan_parent"] != probe["child"]
    _assert_stage_gone(probe)


def test_without_the_marker_scan_the_orphan_survives(sleeper, monkeypatch):
    monkeypatch.setattr(sqd_stage, "_marked", lambda stage_id: [])
    with pytest.raises(ReferenceTimeoutError):
        _run_sleeper(monkeypatch, "sleep", timeout_s=2)

    probe = json.loads(sleeper.read_text())
    with pytest.raises(AssertionError):
        _assert_stage_gone(probe)
    assert _alive(probe["orphan"])
    assert not _alive(probe["plain"]) and not _alive(probe["plain_child"])


@pytest.mark.parametrize(
    ("signum", "returncode"),
    [(signal.SIGINT, -signal.SIGINT), (signal.SIGTERM, 128 + signal.SIGTERM)],
    ids=["keyboard-interrupt", "sigterm"],
)
def test_signal_to_the_caller_cleans_up_the_stage(sleeper, monkeypatch, signum, returncode):
    monkeypatch.setenv("Q2M3_TEST_CHILD_MODE", "sleep")
    with open(sleeper.with_suffix(".stderr"), "wb") as stderr:
        driver = subprocess.Popen([sys.executable, "-m", DRIVER, "sleeper"], stderr=stderr)
    try:
        assert _wait_for(sleeper.exists, 60.0)
        driver.send_signal(signum)
        assert driver.wait(timeout=30) == returncode
    finally:
        _kill(driver.pid)
        driver.wait()

    _assert_stage_gone(json.loads(sleeper.read_text()))


def test_normal_exit_also_reaps_marked_orphans(sleeper, monkeypatch):
    outcome = _run_sleeper(monkeypatch, "linger")

    assert (outcome.results, outcome.warnings) == ((), ())
    _assert_stage_gone(json.loads(sleeper.read_text()))


def test_marked_survivors_become_outcome_warnings(sleeper, monkeypatch):
    monkeypatch.setattr(sqd_stage, "_sigkill", lambda pid: None)
    outcome = _run_sleeper(monkeypatch, "linger")

    probe = json.loads(sleeper.read_text())
    assert outcome.warnings == (SURVIVORS.format(probe["orphan"]),)


def test_survivors_of_a_failed_stage_are_noted_on_the_error(sleeper, monkeypatch):
    monkeypatch.setattr(sqd_stage, "_sigkill", lambda pid: None)
    with pytest.raises(ReferenceTimeoutError) as info:
        _run_sleeper(monkeypatch, "sleep", timeout_s=2)

    probe = json.loads(sleeper.read_text())
    assert info.value.__notes__ == [SURVIVORS.format(probe["orphan"])]


def test_marker_scan_repeats_five_times_with_pauses(monkeypatch):
    tag = str(uuid.uuid4())
    code = "import time; time.sleep(60)"
    marked = subprocess.Popen([sys.executable, "-c", code], env={**os.environ, STAGE_ID: tag})
    try:
        monkeypatch.setattr(sqd_stage, "_sigkill", lambda pid: None)
        start = time.monotonic()
        survivors = sqd_stage._kill_marked(tag)
        elapsed = time.monotonic() - start
    finally:
        marked.kill()
        marked.wait()

    assert survivors == [marked.pid]
    assert 0.5 <= elapsed < 2.0


def test_marker_scan_skips_processes_of_other_users(monkeypatch):
    tag = str(uuid.uuid4())
    code = "import time; time.sleep(60)"
    marked = subprocess.Popen([sys.executable, "-c", code], env={**os.environ, STAGE_ID: tag})
    try:
        assert _wait_for(lambda: sqd_stage._marked(tag) == [marked.pid], 5.0)
        uid = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: uid + 1)
        assert sqd_stage._marked(tag) == []
    finally:
        marked.kill()
        marked.wait()


def test_default_timeout_is_the_job_budgets_plus_a_start_up_margin(sleeper, monkeypatch):
    jobs = [SimpleNamespace(reference=ReferenceConfig(wall_budget_s=s)) for s in (600.0, 900.0)]
    assert parent_timeout_s(jobs) == 1800.0
    assert parent_timeout_s([]) == 300.0

    # Without timeout_s the runner waits for the default limit.
    monkeypatch.setattr(sqd_stage, "parent_timeout_s", lambda jobs: 2.0)
    with pytest.raises(ReferenceTimeoutError, match="2.0 s"):
        _run_sleeper(monkeypatch, "sleep")


def _terminate(signum, frame):
    raise AssertionError("the caller's handler must not run")


def test_caller_sigterm_handler_is_restored_after_the_stage(sleeper, monkeypatch):
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        _run_sleeper(monkeypatch, "linger")
        assert signal.getsignal(signal.SIGTERM) is _terminate
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_handler_not_set_from_python_is_restored_as_the_default(sleeper, monkeypatch):
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        # getsignal returns None for a handler that Python did not install.
        monkeypatch.setattr(signal, "getsignal", lambda signum: None)
        _run_sleeper(monkeypatch, "linger")
    finally:
        assert signal.signal(signal.SIGTERM, previous) is signal.SIG_DFL


def test_stage_outside_the_main_thread_warns_and_keeps_the_handler(sleeper, monkeypatch):
    monkeypatch.setenv("Q2M3_TEST_CHILD_MODE", "linger")
    runner = SubprocessSQDRunner(child_module=SLEEPER)
    outcomes = []
    thread = threading.Thread(target=lambda: outcomes.append(runner.run([])))
    thread.start()
    thread.join(timeout=60)

    (outcome,) = outcomes
    assert outcome.warnings == (
        "SQD stage ran outside the main thread; SIGTERM does not clean up the stage",
    )


@pytest.fixture
def watch(child_path, tmp_path):
    """Starts the watch child; kills the children that a failing test leaves behind."""
    probe = tmp_path / "watch"
    started = []

    def start(parent_pid: str, command=(sys.executable, "-m", WATCH)):
        env = {**os.environ, "Q2M3_TEST_CHILD_PROBE": str(probe), STAGE_ID: str(uuid.uuid4())}
        env["Q2M3_SQD_PARENT_PID"] = parent_pid
        with open(probe.with_suffix(".stderr"), "wb") as stderr:
            started.append(subprocess.Popen(list(command), env=env, stderr=stderr))
        return started[-1]

    yield probe, start
    ready = probe.with_suffix(".ready")
    leftovers = json.loads(ready.read_text()).values() if ready.exists() else ()
    _kill(*(process.pid for process in started), *leftovers)
    for process in started:
        process.wait()


def test_child_sigterm_kills_its_stage_and_runs_finally_blocks(watch):
    probe, start = watch
    child = start(str(os.getpid()))
    assert _wait_for(probe.with_suffix(".ready").exists, 60.0)
    worker = json.loads(probe.with_suffix(".ready").read_text())["worker"]
    assert _alive(worker)

    child.send_signal(signal.SIGTERM)
    assert child.wait(timeout=30) == 128 + signal.SIGTERM
    assert probe.with_suffix(".finally").exists()
    # No finally block of the child reaches the workers; the signal handler does.
    sibling = json.loads(probe.with_suffix(".ready").read_text())["sibling"]
    assert _wait_for(lambda: not _alive(worker) and not _alive(sibling), 5.0)


def test_sigterm_to_a_forked_worker_leaves_the_rest_of_the_stage_alone(watch):
    probe, start = watch
    child = start(str(os.getpid()))
    assert _wait_for(probe.with_suffix(".ready").exists, 60.0)
    ready = json.loads(probe.with_suffix(".ready").read_text())

    # The workers inherit the child's handler, and only the child may clean up the stage.
    # The signal goes to the later worker: a scan in pid order reaches its sibling first.
    os.kill(ready["sibling"], signal.SIGTERM)
    assert not _wait_for(lambda: child.poll() is not None or not _alive(ready["worker"]), 2.0)


def test_child_exits_at_once_when_its_parent_is_already_gone(watch):
    probe, start = watch
    # A recorded parent pid that differs from the real parent pid.
    child = start("1")

    assert child.wait(timeout=60) == 1
    assert "SQD parent exited before the child started" in probe.with_suffix(".stderr").read_text()
    assert not probe.with_suffix(".ready").exists()


def test_child_gets_sigterm_when_its_parent_dies(watch):
    probe, start = watch
    # The launcher records its own pid for the child it starts, then sleeps.
    code = (
        "import os, subprocess, sys, time\n"
        "env = {**os.environ, 'Q2M3_SQD_PARENT_PID': str(os.getpid())}\n"
        f"subprocess.Popen([sys.executable, '-m', '{WATCH}'], env=env)\n"
        "time.sleep(600)\n"
    )
    launcher = start("unused", (sys.executable, "-c", code))
    assert _wait_for(probe.with_suffix(".ready").exists, 60.0)

    launcher.kill()
    launcher.wait()
    assert _wait_for(probe.with_suffix(".finally").exists, 30.0)
    worker = json.loads(probe.with_suffix(".ready").read_text())["worker"]
    assert _wait_for(lambda: not _alive(worker), 5.0)


@pytest.mark.sqd
@pytest.mark.slow
def test_killed_caller_leaves_no_stage_process_behind(child_path, tmp_path):
    with open(tmp_path / "driver.stderr", "wb") as stderr:
        command = [sys.executable, "-m", DRIVER, "water", str(WALL_BUDGET_S)]
        driver = subprocess.Popen(command, stderr=stderr)
    parent = str(driver.pid)
    job_dir = None
    try:
        # The child and a supervised SQD worker both carry the parent marker.
        stage = _wait_for(lambda: len(_carrying("Q2M3_SQD_PARENT_PID", parent)) >= 2, 120.0)
        assert stage, (tmp_path / "driver.stderr").read_text()[-2000:]
        (child,) = (
            pid for pid in _carrying("Q2M3_SQD_PARENT_PID", parent) if _ppid(pid) == driver.pid
        )
        stage_id = _environ(child)[STAGE_ID]
        job_dir = Path(os.readlink(f"/proc/{child}/cwd"))

        driver.kill()
        driver.wait()
        assert _wait_for(lambda: not _carrying(STAGE_ID, stage_id), WALL_BUDGET_S)
        # The child stopped the job instead of finishing it.
        assert not (job_dir / "result_0.pkl").exists()
        # A parent killed by SIGKILL cannot remove its private directory.
        assert job_dir.name.startswith("q2m3-sqd-") and job_dir.is_dir()
    finally:
        _kill(driver.pid, *_carrying("Q2M3_SQD_PARENT_PID", parent))
        driver.wait()
        if job_dir is not None:
            shutil.rmtree(job_dir, ignore_errors=True)


def _ppid(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
