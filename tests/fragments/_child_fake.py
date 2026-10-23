"""Test child interpreter: the real job-file protocol around a scripted solver.

``Q2M3_TEST_CHILD_MODE`` selects the script. ``Q2M3_TEST_CHILD_PROBE`` names a JSON file
that receives what the child observed about its own process.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pyscf import lib

from q2m3.fragments import _sqd_child
from q2m3.sqd.exceptions import ProvenanceMismatchError, ReferenceTimeoutError
from q2m3.sqd.result import SQDResult
from tests.sqd.test_result import result_data


class UnpicklableError(RuntimeError):
    """A failure that cannot cross a pickle boundary."""

    def __reduce__(self):
        raise TypeError("holds an OS handle")


class TwoArgumentError(RuntimeError):
    """A failure that pickles but cannot be rebuilt from its pickled arguments."""

    def __init__(self, name: str, size: int) -> None:
        super().__init__(f"{name} {size}")


FAILURES = {
    "timeout": ReferenceTimeoutError("deadline expired in child"),
    "provenance": ProvenanceMismatchError("frame mismatch in child"),
    "unpicklable": UnpicklableError("socket handle"),
    "unloadable": TwoArgumentError("grid", 7),
}
# The second job fails, so the probe shows whether the third job still ran.
FAILING_JOB = 2


def main() -> int:
    mode = os.environ["Q2M3_TEST_CHILD_MODE"]
    if mode == "crash":
        # More than a pipe buffer of output before the tail the parent must report.
        sys.stderr.write("x" * 70_000 + "\n")
        sys.stderr.write("".join(f"line {i}\n" for i in range(30)))
        return 3
    # A pool size other than 1 shows that the record reads the child's own value.
    lib.num_threads(2)
    solved: list[int] = []

    def solve(job: dict) -> SQDResult:
        solved.append(job["seed"])
        if mode in FAILURES and len(solved) == FAILING_JOB:
            raise FAILURES[mode]
        return SQDResult(**(result_data() | {"seed": job["seed"]}))

    code = _sqd_child.run_jobs(Path(sys.argv[1]), solve)
    probe = {
        "argv": sys.argv[1:],
        "cwd": os.getcwd(),
        "cwd_mode": os.stat(".").st_mode & 0o777,
        "executable": sys.executable,
        "session_leader": os.getsid(0) == os.getpid(),
        "environ": dict(os.environ),
        "solved": solved,
    }
    Path(os.environ["Q2M3_TEST_CHILD_PROBE"]).write_text(json.dumps(probe))
    return code


if __name__ == "__main__":
    sys.exit(main())
