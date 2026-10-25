"""Test child: the SQD child's parent watch around a long sleep with a finally block.

``Q2M3_TEST_CHILD_PROBE`` names a path. After the watch, the child forks two workers in
their own sessions that no finally block of the child kills, as SQD workers that started
just before a signal. The child writes its own pid and the worker pids as JSON to
``<probe>.ready`` before it sleeps, and writes ``<probe>.finally`` when its finally block
runs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from q2m3.fragments._sqd_child import watch_parent

SLEEP_S = 600


def _worker() -> int:
    pid = os.fork()
    if pid == 0:
        os.setsid()
        time.sleep(SLEEP_S)
        os._exit(0)
    return pid


def main() -> int:
    probe = Path(os.environ["Q2M3_TEST_CHILD_PROBE"])
    watch_parent()
    workers = _worker(), _worker()
    try:
        ready = {"child": os.getpid(), "worker": workers[0], "sibling": workers[1]}
        probe.with_suffix(".ready").write_text(json.dumps(ready))
        time.sleep(SLEEP_S)
    finally:
        probe.with_suffix(".finally").write_text("ran")
    return 0


if __name__ == "__main__":
    sys.exit(main())
