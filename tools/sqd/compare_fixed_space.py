"""Compare the released recovery loop with a single fixed sampled space.

Run with the SQD extra and single-thread BLAS::

    python tools/sqd/compare_fixed_space.py --output tmp/sqd/fixed-space

Each molecule/seed runs in a fresh, RSS/time-supervised process. Shared-space
agreement only validates adaptation: both paths use PySCF selected CI. The noisy
nitrogen case is an adversarial recovery diagnostic, not a hardware noise model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Support the documented direct-script command as well as python -m.
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.sqd.calibrate_resources import (  # noqa: E402
    ROOT,
    digest,
    installed_versions,
    proc_bytes,
    run_monitored,
    write_json,
)


def molecule_fixture(name):
    """Return explicit Angstrom geometries and active spaces used in the comparison."""
    from q2m3.molecule import MoleculeConfig

    fixtures = {
        "h2": (["H", "H"], [[0, 0, 0], [0, 0, 0.74]], 0, 2, 2, "sto-3g"),
        "h3o": (
            ["O", "H", "H", "H"],
            [
                [0, 0, 0.1173],
                [0, 0.9572, -0.4692],
                [0.8286, -0.4786, -0.4692],
                [-0.8286, -0.4786, -0.4692],
            ],
            1,
            4,
            4,
            "sto-3g",
        ),
        "glycine": (
            ["N", "H", "H", "C", "H", "H", "C", "O", "O", "H"],
            [
                [-1.870, 0.231, 0],
                [-2.286, 0.764, 0.769],
                [-2.286, 0.764, -0.769],
                [-0.421, 0.137, 0],
                [-0.060, -0.405, 0.880],
                [-0.060, -0.405, -0.880],
                [0.282, 1.519, 0],
                [-0.350, 2.563, 0],
                [1.620, 1.430, 0],
                [1.978, 2.313, 0],
            ],
            0,
            6,
            6,
            "sto-3g",
        ),
        "n2": (["N", "N"], [[0, 0, 0], [0, 0, 1.098]], 0, 10, 10, "cc-pvdz"),
    }
    return MoleculeConfig(name, *fixtures[name])


def run_worker(name: str, seed: int, prefix: Path) -> None:

    import numpy as np

    from q2m3.sqd.ansatz import build_ccsd_seed, build_lucj
    from q2m3.sqd.config import LUCJConfig
    from q2m3.sqd.diagonalize import (
        _diagonalize_recoverable_samples,
        diagonalize_samples,
        kernel_fixed_space,
    )
    from q2m3.sqd.integrals import build_integrals
    from q2m3.sqd.sampling import FfsimSampler

    timings = {}

    def stage(label, operation):
        def emit(event):
            stamp = time.monotonic_ns()
            with prefix.with_suffix(".events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(stage=label, event=event, time_ns=stamp, **proc_bytes(os.getpid()))
                    )
                    + "\n"
                )
            ack = prefix.with_suffix(".ack")
            deadline = time.monotonic() + 10
            while not ack.exists() or ack.read_text() != str(stamp):
                if time.monotonic() > deadline:
                    raise TimeoutError("RSS supervisor did not acknowledge snapshot")
                time.sleep(0.001)

        emit("start")
        start = time.monotonic()
        result = operation()
        timings[label] = time.monotonic() - start
        emit("end")
        return result

    molecule = molecule_fixture(name)
    budget = dict(host_available_mb=4096.0, rss_budget_mb=4096.0)
    data = stage("integrals", lambda: build_integrals(molecule, **budget))
    coupled = stage("ccsd", lambda: build_ccsd_seed(data, **budget))
    lucj = LUCJConfig(n_reps=2, shots=100_000)
    op = stage("lucj", lambda: build_lucj(data, coupled, lucj=lucj, **budget))
    sampler = FfsimSampler(**budget, retained_mb=data.mo_coeff.nbytes / 1e6)
    state = stage("prepare", lambda op=op: sampler.prepare(op, data.norb, data.nelec))
    samples = stage(
        "sample",
        lambda state=state: sampler.sample(
            state, data.norb, data.nelec, shots=lucj.shots, seed=seed
        ),
    )
    del state, op
    retained = (data.mo_coeff.nbytes + coupled.t1.nbytes + coupled.t2.nbytes) / 1e6
    args = (data.h1, data.h2, data.e_core)
    common = dict(norb=data.norb, nelec=data.nelec, **budget)
    controls = dict(
        samples_per_batch=300, num_batches=2, max_iterations=2, symmetrize_spin=False, seed=seed
    )

    def solve_addon(bits, *, noisy=False, extra=0.0):
        kwargs = dict(common, **controls, retained_mb=retained + extra)
        if noisy:
            return _diagonalize_recoverable_samples(*args, bits, **kwargs)
        return diagonalize_samples(*args, bits, **kwargs)

    def solve_direct(strings, *, extra=0.0):
        return kernel_fixed_space(
            *args, strings, **common, retained_mb=retained + samples.nbytes / 1e6 + extra
        )

    def sampled_strings(bits):
        # Explicit independent big-endian decode, beta left / alpha right.
        n = data.norb
        legal = (bits[:, :n].sum(axis=1) == data.nelec[1]) & (
            bits[:, n:].sum(axis=1) == data.nelec[0]
        )
        weights = 1 << np.arange(n - 1, -1, -1)
        hf = (1 << data.nelec[0]) - 1
        return tuple(
            np.unique(np.append(bits[legal, block] @ weights, hf)).astype(np.int64)
            for block in (slice(n, 2 * n), slice(0, n))
        )

    def output_mb(result):
        return (result.amplitudes.nbytes + sum(s.nbytes for s in result.ci_strings)) / 1e6

    addon = stage("addon", lambda: solve_addon(samples))
    fixed = stage("shared_space", lambda: solve_direct(addon.ci_strings, extra=output_mb(addon)))
    adaptation_error = abs(fixed.energy - addon.energy)
    if adaptation_error > 1e-10:
        raise AssertionError(f"Shared-space adapter disagreement: {adaptation_error}")
    direct = stage(
        "sampled_space",
        lambda: solve_direct(
            sampled_strings(samples),
            extra=output_mb(addon) + output_mb(fixed),
        ),
    )
    diagnostic = None
    if name == "n2":
        noisy = samples.copy()
        hf_bits = np.zeros(2 * data.norb, dtype=bool)
        hf_bits[data.norb - data.nelec[1] : data.norb] = True
        hf_bits[2 * data.norb - data.nelec[0] :] = True
        rows = np.flatnonzero(np.any(noisy != hf_bits, axis=1))
        # Remove one occupied beta bit from every non-HF observed determinant.
        # Postselection loses all excited pairs; recovery sees the SAME input.
        noisy[rows, np.argmax(noisy[rows, : data.norb], axis=1)] = False
        corrupted_rows = int(len(rows))
        del rows, hf_bits
        live = sum(output_mb(result) for result in (addon, fixed, direct))
        recovered = stage(
            "noisy_addon", lambda: solve_addon(noisy, noisy=True, extra=samples.nbytes / 1e6 + live)
        )
        postselected = stage(
            "noisy_direct",
            lambda: solve_direct(
                sampled_strings(noisy),
                extra=noisy.nbytes / 1e6 + live + output_mb(recovered),
            ),
        )
        diagnostic = {
            "noise_model": "adversarial beta-electron removal from every non-HF row",
            "corrupted_rows": corrupted_rows,
            "initial_occupancies": "None; learned only from noisy-input postselection",
            "input_sha256": hashlib.sha256(noisy.tobytes()).hexdigest(),
            "recovered_energy_ha": recovered.energy,
            "postselected_direct_energy_ha": postselected.energy,
            "direct_minus_recovered_mHa": 1000 * (postselected.energy - recovered.energy),
            "recovered_dims": recovered.subspace_dims,
            "postselected_dims": postselected.subspace_dims,
            "allocation_audit": [asdict(row) for row in recovered.allocation_audit],
        }
    result = {
        "system": name,
        "seed": seed,
        "geometry_unit": "Angstrom",
        "molecule": asdict(molecule),
        "frame_id": data.context.frame_id,
        "hamiltonian_id": data.context.hamiltonian_id,
        "versions": installed_versions(),
        "controls": controls,
        "fixed_controls": dict(
            carryover_threshold=0, max_space=12, max_cycle=100, tol=1e-12, include_hf=True
        ),
        "n_reps": 2,
        "shots": lucj.shots,
        "sample_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
        "unique_sample_pairs": int(len(np.unique(samples, axis=0))),
        "full_ci_dim": math.prod(math.comb(data.norb, n) for n in data.nelec),
        "hf_energy_ha": data.hf_energy,
        "ccsd_energy_ha": coupled.ccsd_energy,
        "e_core_ha": data.e_core,
        "addon_energy_ha": addon.energy,
        "direct_energy_ha": direct.energy,
        "shared_space_error_ha": adaptation_error,
        "direct_minus_addon_mHa": 1000 * (direct.energy - addon.energy),
        "within_small_system_threshold": abs(direct.energy - addon.energy) <= 0.0001,
        "addon_dims": addon.subspace_dims,
        "direct_dims": direct.subspace_dims,
        "allocation_audit": [asdict(row) for row in addon.allocation_audit],
        "noisy_recovery": diagnostic,
        "timings_s": timings,
        "rss_budget_mb": 4096,
        "default": "released_addon",
        "replacement": False,
        "replacement_reason": "single fixed space has no recovery/batch/self-consistency loop",
    }
    write_json(prefix.with_suffix(".result.json"), result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=("h2", "h3o", "glycine", "n2"),
        default=["h2", "h3o", "glycine", "n2"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args.systems[0], args.seeds[0], args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "source-manifest.json",
        {
            str(path.relative_to(ROOT)): digest(path)
            for path in [
                Path(__file__).resolve(),
                ROOT / "tools/sqd/calibrate_resources.py",
                ROOT / "src/q2m3/sqd/resource_calibration.json",
                *sorted((ROOT / "src/q2m3/sqd").glob("*.py")),
            ]
        },
    )
    results = []
    for name in args.systems:
        for seed in args.seeds:
            prefix = args.output / f"{name}-{seed}"
            run = run_monitored(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--systems",
                    name,
                    "--seeds",
                    str(seed),
                    "--output",
                    str(prefix.resolve()),
                ],
                prefix,
                timeout_s=600,
            )
            if run["status"] != "ok":
                raise RuntimeError(f"Failed/censored run {name}/{seed}: {run}")
            results.append(json.loads(prefix.with_suffix(".result.json").read_text()))
    write_json(args.output / "summary.json", results)
    small = [r for r in results if r["system"] != "n2"]
    write_json(
        args.output / "decision.json",
        {
            "execution_completed": True,
            "small_system_0_1_mHa_gate": (
                all(r["within_small_system_threshold"] for r in small) if small else None
            ),
            "small_systems_complete": {r["system"] for r in small} == {"h2", "h3o", "glycine"},
            "replace_default": False,
            "reason": "Fixed kernel omits recovery/batching/self-consistency, irrespective of energy gaps",
        },
    )


if __name__ == "__main__":
    main()
