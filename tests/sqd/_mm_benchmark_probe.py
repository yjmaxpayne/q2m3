"""Run the MM TIP3P benchmark in a fresh interpreter and capture SQD subspaces.

``run_sqd`` solves in a forked child, so the capture travels through a file.
Before each point the parent names the capture file in ``Q2M3_SUBSPACE_CAPTURE``.
The wrapped ``kernel_fixed_space`` writes the first call of the point only:
that call re-checks the SQD energy on the SQD strings.

Usage, from the checkout root with the three thread variables set to 1::

    python -m tests.sqd._mm_benchmark_probe --output <new dir> --fixtures <name> ...
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

CAPTURE_ENV = "Q2M3_SUBSPACE_CAPTURE"


def _capturing(original):
    def wrapper(h1, h2, e_core, ci_strings, **kwargs):
        path = Path(os.environ[CAPTURE_ENV])
        if not path.exists():
            alpha, beta = (np.asarray(s, dtype=np.int64) for s in ci_strings)
            np.savez(path, h1=h1, h2=h2, e_core=e_core, alpha=alpha, beta=beta)
        return original(h1, h2, e_core, ci_strings, **kwargs)

    return wrapper


def install_capture() -> None:
    """Wrap ``kernel_fixed_space`` where it is defined and where observables bound it."""
    from q2m3.sqd import diagonalize, observables

    for module in (diagonalize, observables):
        module.kernel_fixed_space = _capturing(module.kernel_fixed_space)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested fixtures in both modes and write report and captures."""
    from examples.sqd._inputs import load_mm_tip3p_fixtures
    from examples.sqd.mm_tip3p_benchmark import (
        MODES,
        benchmark_inputs,
        benchmark_point,
        write_outputs,
    )

    fixtures = {fixture["name"]: fixture for fixture in load_mm_tip3p_fixtures()}
    parser = argparse.ArgumentParser(description="MM TIP3P benchmark capture probe")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", nargs="+", choices=list(fixtures), required=True)
    args = parser.parse_args(argv)

    captures = args.output / "captures"
    captures.mkdir(parents=True)
    install_capture()
    records = []
    for name in args.fixtures:
        inp = benchmark_inputs(fixtures[name])
        for mode in MODES:
            os.environ[CAPTURE_ENV] = str(captures / f"{name}__{mode}.npz")
            records.append(benchmark_point(inp, mode, 0))
    write_outputs(records, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
