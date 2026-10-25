"""Test child that leaves processes outside the normal reach of the stage cleanup.

``Q2M3_TEST_CHILD_MODE=sleep`` creates an orphan in its own session that keeps the stage
marker, and a plain child with a child of its own, both without the marker, then sleeps. ``linger``
creates only the orphan and exits. The module imports only the standard library, so the
child reaches this state well within a two-second timeout. ``Q2M3_TEST_CHILD_PROBE``
names a JSON file that receives the pids.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

SLEEP_S = 600
MARKER = b"Q2M3_SQD_STAGE_ID="


def _parent(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])


def _orphan() -> int:
    """A sleeper whose parent exits at once, so it is no descendant of this child."""
    read, write = os.pipe()
    middle = os.fork()
    if middle == 0:
        os.setsid()
        sleeper = os.fork()
        if sleeper == 0:
            time.sleep(SLEEP_S)
            os._exit(0)
        os.write(write, str(sleeper).encode())
        os._exit(0)
    os.waitpid(middle, 0)
    return int(os.read(read, 32))


def _plain() -> tuple[int, int]:
    """A direct child and its own child, both with an empty environment."""
    pid = os.fork()
    if pid == 0:
        code = f"import os, time; os.fork(); time.sleep({SLEEP_S})"
        os.execve(sys.executable, [sys.executable, "-c", code], {})
    # Until exec, the child still shows the inherited environment.
    while any(
        entry.startswith(MARKER) for entry in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    ):
        time.sleep(0.01)
    children = Path(f"/proc/{pid}/task/{pid}/children")
    while not children.read_text().split():
        time.sleep(0.01)
    return pid, int(children.read_text().split()[0])


def main() -> int:
    mode = os.environ["Q2M3_TEST_CHILD_MODE"]
    orphan = _orphan()
    probe = {
        "stage_id": os.environ["Q2M3_SQD_STAGE_ID"],
        "child": os.getpid(),
        "orphan": orphan,
        "orphan_parent": _parent(orphan),
    }
    if mode == "sleep":
        probe["plain"], probe["plain_child"] = _plain()
    path = Path(os.environ["Q2M3_TEST_CHILD_PROBE"])
    path.with_suffix(".tmp").write_text(json.dumps(probe))
    os.replace(path.with_suffix(".tmp"), path)
    if mode == "sleep":
        time.sleep(SLEEP_S)
    return 0


if __name__ == "__main__":
    sys.exit(main())
