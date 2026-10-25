"""Adapters for quantum-chemistry programs that run in their own environments."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from typing import Protocol


class ExternalProgramRunner(Protocol):
    """Runs one command of an external program and returns its exit code and output."""

    def run(
        self, argv: Sequence[str], *, cwd: str, env: Mapping[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]: ...


class SubprocessRunner:
    """Default runner: ``subprocess.run`` with text output captured in memory."""

    def run(
        self, argv: Sequence[str], *, cwd: str, env: Mapping[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(env),
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
