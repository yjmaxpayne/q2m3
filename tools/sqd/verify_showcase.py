"""Independently monitor a showcase command from interpreter launch through exit.

Linux /proc process-tree RSS, polled every 10 ms, is an observation rather than
a bound on arbitrary transients. This does not extend SQD resource certification.
Run from the checkout, for example::

    python -m tools.sqd.verify_showcase --output data/output/examples/check \
        -- python -m examples.sqd.glycine_ground_state
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from tools.sqd.calibrate_resources import ROOT, proc_bytes, tree_pids


def measure(command: list[str], directory: Path, *, wall_s=900.0, rss_mb=8192.0) -> dict:
    """Supervise a complete command and retain failure, stdout and RSS evidence."""
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    peak = 0
    reason = None
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        JAX_PLATFORMS="cpu",
    )
    with (directory / "stdout.log").open("w") as log, (directory / "rss.jsonl").open("w") as trace:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                rss = sum(proc_bytes(pid).get("VmRSS", 0) for pid in tree_pids(process.pid))
                peak = max(peak, rss)
                elapsed = time.monotonic() - started
                trace.write(json.dumps([elapsed, rss]) + "\n")
                if elapsed >= wall_s or rss >= rss_mb * 1e6:
                    reason = "wall_budget_exceeded" if elapsed >= wall_s else "rss_budget_exceeded"
                    break
                time.sleep(0.01)
        finally:
            # The new session isolates the command and its descendants from this observer.
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    record = dict(
        command=command,
        exit_code=process.returncode,
        reason=reason,
        wall_s=time.monotonic() - started,
        peak_tree_rss_mb=peak / 1e6,
        poll_s=0.01,
        rss_budget_mb=rss_mb,
        wall_budget_s=wall_s,
        scope="interpreter launch through exit; simultaneous descendants; observer excluded",
        limitation="polled observations do not bound arbitrary transient allocations",
    )
    (directory / "resources.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> int:
    """Run a supplied argv without a shell and return its success or budget failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wall-budget", type=float, default=900.0)
    parser.add_argument("--rss-budget", type=float, default=8192.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.wall_budget <= 0 or args.rss_budget <= 0:
        parser.error("a command and positive budgets are required")
    result = measure(command, args.output, wall_s=args.wall_budget, rss_mb=args.rss_budget)
    print(json.dumps(result, indent=2))
    return 0 if result["exit_code"] == 0 and result["reason"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
