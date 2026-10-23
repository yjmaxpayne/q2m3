"""SQD stage of the fragment workflow, run in a child interpreter.

The child gets a copy of the caller environment with the OpenMP, MKL, OpenBLAS and rayon
pools set to one thread each. The caller process does not run SQD and keeps its own
thread settings. Jobs, results and failures cross the process boundary as pickle files
in a private directory.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from q2m3.fragments.errors import FragmentWorkflowError
from q2m3.sqd.config import IntegralContext, LUCJConfig, ReferenceConfig
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
    """Results and thread records in job order, and the wall time of the stage."""

    results: tuple[SQDResult, ...]
    thread_records: tuple[ThreadRecord, ...]
    wall_s: float


class SQDStageRunner(Protocol):
    """Runs SQD jobs outside the caller process."""

    def run(self, jobs: Sequence[SQDJob]) -> SQDStageOutcome: ...


class SubprocessSQDRunner:
    """Runs all jobs in one child interpreter, one job after another.

    Args:
        child_module: Module run as ``python -m <child_module> <job_dir>``.
        timeout_s: Wait limit for the child in seconds. None waits without a limit.
    """

    def __init__(
        self, *, child_module: str = "q2m3.fragments._sqd_child", timeout_s: float | None = None
    ) -> None:
        self.child_module = child_module
        self.timeout_s = timeout_s

    def run(self, jobs: Sequence[SQDJob]) -> SQDStageOutcome:
        """Run ``jobs`` in a child interpreter and return its results in job order.

        Raises:
            FragmentWorkflowError: The child exited without results or a typed failure, or
                its failure could not be pickled.
            Exception: The failure of the first failing job, with the type the child saw.
        """
        start = monotonic()
        # mkdtemp creates the directory with mode 0700.
        with tempfile.TemporaryDirectory(prefix="q2m3-sqd-") as name:
            job_dir = Path(name)
            for index, job in enumerate(jobs):
                (job_dir / f"job_{index}.pkl").write_bytes(pickle.dumps(_job_payload(job)))
            returncode = self._launch(job_dir)
            results, records = _collect(job_dir, len(jobs), returncode)
        return SQDStageOutcome(results, records, monotonic() - start)

    def _launch(self, job_dir: Path) -> int:
        env = dict(os.environ)
        env.update(dict.fromkeys(THREAD_VARIABLES, "1"))
        env["Q2M3_SQD_STAGE_ID"] = str(uuid.uuid4())
        env["Q2M3_SQD_PARENT_PID"] = str(os.getpid())
        # Output goes to files: a pipe that is read only after exit blocks the child once full.
        with (
            open(job_dir / "stdout.txt", "wb") as stdout,
            open(job_dir / "stderr.txt", "wb") as stderr,
        ):
            return subprocess.run(
                [sys.executable, "-m", self.child_module, str(job_dir)],
                cwd=job_dir,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                timeout=self.timeout_s,
                check=False,
            ).returncode


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
