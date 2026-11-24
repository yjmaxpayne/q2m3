#!/usr/bin/env python3
# Copied from the maintainer process-tooling repository, scripts/blas_kernel_matrix.py
# at 790f5d7. Change the source there, then copy it here.
"""Run a test command under two OpenBLAS kernels and fail when any run fails (BFX.0003).

OpenBLAS picks a kernel for the CPU. A numerical baseline recorded with one kernel can
differ in the last bits on a CI runner with another CPU. The script runs the command once
per kernel with OPENBLAS_CORETYPE set, so the result does not depend on the runner CPU.

OpenBLAS ignores an unknown OPENBLAS_CORETYPE value without a message and uses the native
kernel. Some names also load another kernel: Prescott and Core2 load Katmai, Zen loads
Haswell. Before any run, the script imports numpy in its own interpreter with
OPENBLAS_VERBOSE=2 and reads the "Core: <name>" line. The name must equal the requested
kernel. Sandybridge (AVX, no FMA) and Haswell (AVX2, FMA) report their own names. Run the
script with the project interpreter, for example:

    uv run python blas_kernel_matrix.py -- python -m pytest -n 0 tests/

Exit codes: 0 when every run passes. 1 when a run fails. 2 when OpenBLAS does not load a
requested kernel (no command runs), or when the arguments are wrong.

Known gaps: OpenBLAS x86-64 kernels only. MKL, Accelerate and ARM kernels are not covered.
"""

import argparse
import os
import re
import subprocess
import sys

DEFAULT_KERNELS = "Sandybridge,Haswell"


def loaded_kernel(kernel):
    """Return the kernel that OpenBLAS reports for OPENBLAS_CORETYPE=kernel, or None."""
    env = {**os.environ, "OPENBLAS_CORETYPE": kernel, "OPENBLAS_VERBOSE": "2"}
    probe = subprocess.run(
        [sys.executable, "-c", "import numpy"], env=env, capture_output=True, text=True
    )
    found = re.findall(r"^Core: (\S+)", probe.stdout + probe.stderr, re.MULTILINE)
    return found[-1] if found else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--kernels", default=DEFAULT_KERNELS, help="default: %(default)s")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="test command after --")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the test command after --")
    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    for kernel in kernels:
        got = loaded_kernel(kernel)
        if got is None or got.lower() != kernel.lower():
            print(
                f"blas_kernel_matrix: OpenBLAS did not load kernel {kernel} "
                f"(numpy reports {got or 'no OpenBLAS core'}). No command ran.",
                file=sys.stderr,
            )
            return 2
    codes = {}
    for kernel in kernels:
        print(f"blas_kernel_matrix: OPENBLAS_CORETYPE={kernel}", flush=True)
        env = {**os.environ, "OPENBLAS_CORETYPE": kernel}
        codes[kernel] = subprocess.run(command, env=env).returncode
    for kernel, code in codes.items():
        print(f"{kernel}\t{code}")
    return 1 if any(codes.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
