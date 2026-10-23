"""Child interpreter of the SQD stage: ``python -m q2m3.fragments._sqd_child <job_dir>``.

The parent sets the thread variables before this interpreter starts. The child solves
the job files in index order and writes one result file per job. At the first failure it
writes ``error.pkl`` and does not run the remaining jobs.
"""

from __future__ import annotations

import os
import pickle
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pyscf import lib

from q2m3.fragments.errors import FragmentWorkflowError
from q2m3.fragments.sqd_stage import THREAD_VARIABLES
from q2m3.sqd import build_ccsd_seed_from_integrals, run_sqd_from_integrals
from q2m3.sqd.exceptions import ResourceLimitError
from q2m3.sqd.result import SQDResult
from q2m3.sqd.transport import decode_result, encode_result

# Upper memory limit of every SQD component, in decimal MB.
COMPONENT_CAP_MB = 8192.0


def run_jobs(job_dir: Path, solve: Callable[[dict[str, Any]], SQDResult]) -> int:
    """Solve the job files of ``job_dir`` in index order and return the exit code."""
    try:
        for index in range(len(list(job_dir.glob("job_*.pkl")))):
            threads = {
                "environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
                "pyscf_threads": lib.num_threads(),
            }
            result = solve(pickle.loads((job_dir / f"job_{index}.pkl").read_bytes()))
            output = {"result": encode_result(result), "threads": threads}
            (job_dir / f"result_{index}.pkl").write_bytes(pickle.dumps(output))
    except Exception as exc:  # the parent re-raises every job failure
        (job_dir / "error.pkl").write_bytes(_error_bytes(exc))
        return 1
    return 0


def solve(job: dict[str, Any]) -> SQDResult:
    """Build the CCSD seed of one job and run full SQD on the same integrals.

    Raises:
        ResourceLimitError: No memory is left for the job after the current RSS.
    """
    context, lucj, reference = (decode_result(job[key]) for key in ("context", "lucj", "reference"))
    available = host_available_mb(reference.rss_budget_mb)
    if available <= 0:
        raise ResourceLimitError(f"No memory left for the SQD job: host_available_mb={available}")
    arrays = (job["h1"], job["h2"], job["e_core"])
    shape = {"norb": job["norb"], "nelec": job["nelec"], "context": context}
    seed_data = build_ccsd_seed_from_integrals(
        *arrays, **shape, host_available_mb=available, rss_budget_mb=reference.rss_budget_mb
    )
    return run_sqd_from_integrals(
        *arrays,
        **shape,
        seed_data=seed_data,
        lucj=lucj,
        reference=reference,
        seed=job["seed"],
        mode="full",
    )


def host_available_mb(rss_budget_mb: float) -> float:
    """Memory for one job in decimal MB: the smallest cap minus the current RSS."""
    return min(rss_budget_mb, _mem_available_mb(), COMPONENT_CAP_MB) - _rss_mb()


def _rss_mb() -> float:
    return _proc_kib(Path("/proc/self/status"), "VmRSS:") * 1024 / 1e6


def _mem_available_mb() -> float:
    return _proc_kib(Path("/proc/meminfo"), "MemAvailable:") * 1024 / 1e6


def _proc_kib(path: Path, key: str) -> int:
    return next(
        int(line.split()[1]) for line in path.read_text().splitlines() if line.startswith(key)
    )


def _error_bytes(exc: Exception) -> bytes:
    """Pickle ``exc``. A failure that does not survive a round trip keeps its type name."""
    try:
        blob = pickle.dumps(exc)
        pickle.loads(blob)
    except Exception:
        blob = pickle.dumps(FragmentWorkflowError(f"{type(exc).__name__}: {exc}"))
    return blob


if __name__ == "__main__":
    sys.exit(run_jobs(Path(sys.argv[1]), solve))
