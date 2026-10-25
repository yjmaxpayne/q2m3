"""Test caller of the SQD stage, for signals that reach the caller process.

``python -m tests.fragments._stage_driver sleeper`` runs the sleeper test child with no
jobs. ``python -m tests.fragments._stage_driver water <wall_budget_s>`` runs the real
child with one water (4e,4o) job.
"""

from __future__ import annotations

import signal
import sys

from q2m3.fragments.sqd_stage import SQDJob, SubprocessSQDRunner
from q2m3.molecule import MoleculeConfig
from q2m3.sqd.config import ReferenceConfig
from q2m3.sqd.integrals import build_integrals

WATER = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]


def main() -> int:
    # A process started in the background by a non-interactive shell can inherit an
    # ignored SIGINT, and then gets no KeyboardInterrupt.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    if sys.argv[1] == "sleeper":
        SubprocessSQDRunner(child_module="tests.fragments._child_sleeper").run([])
        return 0
    config = MoleculeConfig("water", ["O", "H", "H"], WATER, 0, 4, 4, "sto-3g")
    data = build_integrals(config, host_available_mb=4096)
    reference = ReferenceConfig(wall_budget_s=float(sys.argv[2]))
    job = SQDJob(
        data.h1, data.h2, data.e_core, data.norb, data.nelec, data.context, reference=reference
    )
    SubprocessSQDRunner().run([job])
    return 0


if __name__ == "__main__":
    sys.exit(main())
