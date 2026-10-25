"""SQD stage of the fragment workflow, run in a child interpreter.

The child gets a copy of the caller environment with the OpenMP, MKL, OpenBLAS and rayon
pools set to one thread each. The caller process does not run SQD and keeps its own
thread settings. Jobs, results and failures cross the process boundary as pickle files
in a private directory.

On every exit path the parent kills the child, the child's descendants and every
process of the user that carries the stage marker in its environment. SIGTERM to the
caller becomes ``SystemExit`` during the stage, so this cleanup runs before the caller
exits.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from q2m3.fragments.errors import FragmentWorkflowError
from q2m3.sqd.config import IntegralContext, LUCJConfig, ReferenceConfig
from q2m3.sqd.exceptions import ReferenceTimeoutError
from q2m3.sqd.transport import decode_result, encode_result

if TYPE_CHECKING:
    from q2m3.sqd.result import SQDResult

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
)
STDERR_TAIL_LINES = 20
# Wait-limit share of seed construction and child start-up, which the budgets exclude.
STARTUP_MARGIN_S = 300.0
CLEANUP_SCANS = 5
# Time for processes that are exiting, or in uninterruptible sleep, to disappear.
CLEANUP_PAUSE_S = 0.1


@dataclasses.dataclass(frozen=True)
class SQDJob:
    """Integral-level SQD input of one subsystem.

    The fields are the arguments of :func:`q2m3.sqd.run_sqd_from_integrals`. The child
    builds the CCSD seed from the same integrals.
    """

    h1: np.ndarray
    h2: np.ndarray
    e_core: float
    norb: int
    nelec: tuple[int, int]
    context: IntegralContext
    lucj: LUCJConfig = dataclasses.field(default_factory=LUCJConfig)
    reference: ReferenceConfig = dataclasses.field(default_factory=ReferenceConfig)
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class ThreadRecord:
    """Thread settings that the child observed before one SQD job."""

    environment: Mapping[str, str | None]
    pyscf_threads: int


@dataclasses.dataclass(frozen=True)
class SQDStageOutcome:
    """Results and thread records in job order, the stage wall time and its warnings."""

    results: tuple[SQDResult, ...]
    thread_records: tuple[ThreadRecord, ...]
    wall_s: float
    warnings: tuple[str, ...] = ()


class SQDStageRunner(Protocol):
    """Runs SQD jobs outside the caller process."""

    def run(self, jobs: Sequence[SQDJob]) -> SQDStageOutcome: ...


class SubprocessSQDRunner:
    """Runs all jobs in one child interpreter, one job after another.

    Args:
        child_module: Module run as ``python -m <child_module> <job_dir>``.
        timeout_s: Wait limit for the child in seconds. None uses :func:`parent_timeout_s`.
    """

    def __init__(
        self, *, child_module: str = "q2m3.fragments._sqd_child", timeout_s: float | None = None
    ) -> None:
        self.child_module = child_module
        self.timeout_s = timeout_s

    def run(self, jobs: Sequence[SQDJob]) -> SQDStageOutcome:
        """Run ``jobs`` in a child interpreter and return its results in job order.

        Raises:
            ReferenceTimeoutError: The child did not exit within the wait limit.
            FragmentWorkflowError: The child exited without results or a typed failure, or
                its failure could not be pickled.
            Exception: The failure of the first failing job, with the type the child saw.
        """
        start = monotonic()
        warnings: list[str] = []
        timeout_s = parent_timeout_s(jobs) if self.timeout_s is None else self.timeout_s
        # mkdtemp creates the directory with mode 0700.
        with _sigterm_as_exit(warnings), tempfile.TemporaryDirectory(prefix="q2m3-sqd-") as name:
            job_dir = Path(name)
            for index, job in enumerate(jobs):
                (job_dir / f"job_{index}.pkl").write_bytes(pickle.dumps(_job_payload(job)))
            returncode = self._launch(job_dir, timeout_s, warnings)
            results, records = _collect(job_dir, len(jobs), returncode)
        return SQDStageOutcome(results, records, monotonic() - start, tuple(warnings))

    def _launch(self, job_dir: Path, timeout_s: float, warnings: list[str]) -> int:
        stage_id = str(uuid.uuid4())
        env = dict(os.environ)
        env.update(dict.fromkeys(THREAD_VARIABLES, "1"))
        env["Q2M3_SQD_STAGE_ID"] = stage_id
        env["Q2M3_SQD_PARENT_PID"] = str(os.getpid())
        # Output goes to files: a pipe that is read only after exit blocks the child once full.
        with (
            open(job_dir / "stdout.txt", "wb") as stdout,
            open(job_dir / "stderr.txt", "wb") as stderr,
        ):
            # The calling thread starts and waits for the child, because the child's
            # parent-death signal follows the thread that created it.
            child = subprocess.Popen(
                [sys.executable, "-m", self.child_module, str(job_dir)],
                cwd=job_dir,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                try:
                    returncode = child.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    message = f"SQD stage exceeded its wait limit of {timeout_s} s"
                    raise ReferenceTimeoutError(message) from None
            except BaseException as exc:
                for note in _clean_up(child, stage_id):
                    exc.add_note(note)
                raise
            warnings.extend(_clean_up(child, stage_id))
            return returncode


def parent_timeout_s(jobs: Sequence[SQDJob]) -> float:
    """Default wait limit in seconds: the wall budgets of the jobs plus a start-up margin."""
    return sum(job.reference.wall_budget_s for job in jobs) + STARTUP_MARGIN_S


def _exit_on_sigterm(signum: int, frame: object) -> None:
    """Turn SIGTERM into ``SystemExit``, so that cleanup and ``finally`` blocks run."""
    raise SystemExit(128 + signum)


@contextlib.contextmanager
def _sigterm_as_exit(warnings: list[str]) -> Iterator[None]:
    """Install :func:`_exit_on_sigterm` for the stage; only the main thread can do so."""
    if threading.current_thread() is not threading.main_thread():
        warnings.append(
            "SQD stage ran outside the main thread; SIGTERM does not clean up the stage"
        )
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        yield
    finally:
        # None means that the handler was not installed from Python.
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)


def _clean_up(child: subprocess.Popen, stage_id: str) -> list[str]:
    """Kill the child, its descendants and the marked processes; return survivor warnings."""
    # Snapshot first: when the child dies, its descendants move to another parent.
    for pid in sorted(_descendants(child.pid), reverse=True):
        _sigkill(pid)
    child.kill()
    child.wait()
    survivors = _kill_marked(stage_id)
    return [f"SQD stage processes survived cleanup: {survivors}"] if survivors else []


def _descendants(root: int) -> set[int]:
    """Pids below ``root`` in the process tree, read from /proc."""
    found: set[int] = set()
    frontier = [root]
    while frontier:
        pid = frontier.pop()
        try:
            tasks = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        for task in tasks:
            try:
                text = Path(f"/proc/{pid}/task/{task}/children").read_text()
            except OSError:
                continue
            children = {int(child) for child in text.split()} - found
            found |= children
            frontier.extend(children)
    return found


def _kill_marked(stage_id: str, exclude: int | None = None) -> list[int]:
    """Kill marked processes other than ``exclude`` until a scan finds none.

    Returns the pids of the last scan.
    """
    marked: list[int] = []
    for _ in range(CLEANUP_SCANS):
        marked = [pid for pid in _marked(stage_id) if pid != exclude]
        if not marked:
            break
        for pid in marked:
            _sigkill(pid)
        sleep(CLEANUP_PAUSE_S)
    return marked


def _marked(stage_id: str) -> list[int]:
    """Pids of this user whose exec-time environment carries the stage marker."""
    entry = f"Q2M3_SQD_STAGE_ID={stage_id}".encode()
    uid = os.getuid()
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            if os.stat(f"/proc/{name}").st_uid != uid:
                continue
            environ = Path(f"/proc/{name}/environ").read_bytes()
        except OSError:  # the process exited, or its environment is not readable
            continue
        if entry in environ.split(b"\0"):
            pids.append(int(name))
    return pids


def _sigkill(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def _job_payload(job: SQDJob) -> dict[str, Any]:
    """Job file content: plain values, and records in the SQD transport encoding."""
    payload = {name: getattr(job, name) for name in ("h1", "h2", "e_core", "norb", "nelec")}
    payload["seed"] = job.seed
    for name in ("context", "lucj", "reference"):
        payload[name] = encode_result(getattr(job, name))
    return payload


def _collect(
    job_dir: Path, count: int, returncode: int
) -> tuple[tuple[SQDResult, ...], tuple[ThreadRecord, ...]]:
    error = job_dir / "error.pkl"
    if error.exists():
        raise pickle.loads(error.read_bytes())
    paths = [job_dir / f"result_{index}.pkl" for index in range(count)]
    if not all(path.exists() for path in paths):
        stderr = (job_dir / "stderr.txt").read_text(errors="replace").splitlines()
        tail = "\n".join(stderr[-STDERR_TAIL_LINES:])
        raise FragmentWorkflowError(
            f"SQD child exited with exit code {returncode} without results; stderr tail:\n{tail}"
        )
    outputs = [pickle.loads(path.read_bytes()) for path in paths]
    results = tuple(decode_result(output["result"]) for output in outputs)
    return results, tuple(ThreadRecord(**output["threads"]) for output in outputs)
