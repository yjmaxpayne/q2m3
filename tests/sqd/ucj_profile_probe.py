"""Measure the 36-factor UCJ workload in supervised, isolated processes.

Run from the checkout with ``python tests/sqd/ucj_profile_probe.py OUTPUT.json``.
This probe measures a proposed resource domain before enabling that domain.
It calls ffsim directly. It does not bypass guards in a production calculation.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

POINTS = ((2, 1), (4, 2), (6, 3), (8, 4), (9, 6), (10, 5), (5, 2), (7, 3), (10, 4))
REPEATS = 3
REPS = 36
SHOTS = 100_000
RSS_CAP_MB = 2048
WALL_CAP_S = 90


def worker(norb, occupied, seed, phase_file):
    """Exercise dense amplitudes and preserve the production live-array schedule."""
    import ffsim
    import numpy as np

    # Include the same optional imports as the production executor.
    from q2m3.sqd import orchestrator  # noqa: F401
    from q2m3.sqd.resources import estimate_rss_mb, load_resource_model

    nelec = (occupied, occupied)
    virtual = norb - occupied
    rng = np.random.default_rng(seed)
    t1 = rng.normal(scale=0.03, size=(occupied, virtual))
    t2 = rng.normal(scale=0.03, size=(occupied, occupied, virtual, virtual))
    t2 = (t2 + t2.transpose(1, 0, 3, 2)) / 2
    phase_file.write_text("prepare")
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1=t1, n_reps=REPS, optimize=False)
    state = ffsim.apply_unitary(ffsim.hartree_fock_state(norb, nelec), op, norb=norb, nelec=nelec)
    assert np.isclose(np.linalg.norm(state), 1)
    phase_file.write_text("sample")
    samples = ffsim.sample_state_vector(
        state,
        norb=norb,
        nelec=nelec,
        shots=SHOTS,
        seed=seed,
        concatenate=True,
        bitstring_type=ffsim.BitstringType.BIT_ARRAY,
    )
    assert samples.shape == (SHOTS, 2 * norb)
    assert np.all(samples[:, :norb].sum(axis=1) == occupied)
    assert np.all(samples[:, norb:].sum(axis=1) == occupied)
    predicted = {
        stage: estimate_rss_mb(
            norb,
            nelec,
            stage=stage,
            model=load_resource_model(path=os.environ.get("Q2M3_UCJ_PROFILE")),
        )
        for stage in ("prepare", "sample")
    }
    print(
        json.dumps(
            {
                "predicted_mb": predicted,
                "operator_nbytes": op.diag_coulomb_mats.nbytes
                + op.orbital_rotations.nbytes
                + op.final_orbital_rotation.nbytes,
            }
        )
    )


def measure(norb, occupied, seed):
    """Enforce process-tree memory and wall limits during one measurement."""
    with tempfile.TemporaryDirectory(prefix="ucj-profile-") as tmp:
        phase_file = Path(tmp) / "phase"
        phase_file.write_text("imports")
        env = {
            **os.environ,
            **dict.fromkeys(
                (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ),
                "1",
            ),
            "JAX_PLATFORMS": "cpu",
        }
        command = [
            sys.executable,
            __file__,
            "--worker",
            str(norb),
            str(occupied),
            str(seed),
            str(phase_file),
        ]
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        ) as process:
            started = time.monotonic()
            peaks = {"imports": 0.0, "prepare": 0.0, "sample": 0.0}
            try:
                while process.poll() is None:
                    tree = psutil.Process(process.pid)
                    rss = 0
                    for child in [tree, *tree.children(recursive=True)]:
                        try:
                            rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    phase = phase_file.read_text().strip() or "imports"
                    peaks[phase] = max(peaks[phase], rss / 1e6)
                    if rss / 1e6 >= RSS_CAP_MB or time.monotonic() - started >= WALL_CAP_S:
                        raise RuntimeError("UCJ profile probe exceeded its process budget")
                    time.sleep(0.005)
                stdout, stderr = process.communicate()
                if process.returncode:
                    raise RuntimeError(stderr)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        row = json.loads(stdout)
        row.update(
            norb=norb,
            occupied=occupied,
            seed=seed,
            peak_mb=peaks,
            elapsed_s=time.monotonic() - started,
        )
        # Include import peak in both checks, even if imports exceed stage peaks.
        row["margins_mb"] = {
            stage: row["predicted_mb"][stage] - max(peaks["imports"], peaks[stage])
            for stage in ("prepare", "sample")
        }
        return row


def main(output, seed_offset=0):
    """Write measurements without fitting or changing the resource model."""
    rows = []
    for point in POINTS:
        for seed in range(seed_offset, seed_offset + REPEATS):
            row = measure(*point, seed)
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "n_reps": REPS,
        "shots": SHOTS,
        "repeats": REPEATS,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "process_rss_cap_mb": RSS_CAP_MB,
        "process_wall_cap_s": WALL_CAP_S,
        "minimum_margin_mb": min(min(row["margins_mb"].values()) for row in rows),
        "measurements": rows,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    if report["minimum_margin_mb"] < 64:
        raise RuntimeError("The proposed profile lacks 64 MB of measured RSS margin")


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        worker(*map(int, sys.argv[2:5]), Path(sys.argv[5]))
    else:
        main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0)
