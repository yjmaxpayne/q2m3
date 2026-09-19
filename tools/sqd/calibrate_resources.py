"""Serial RSS calibration with held-out cases and external process-tree fuses.

Run with the SQD extra in an isolated environment::

    python tools/sqd/calibrate_resources.py --output tmp/sqd/calibration

The output directory is immutable per run. The preregistration is written before
any worker starts. Coefficients use training observations only and are frozen
before held-out workers start. No scientific imports occur in the supervisor.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

POINTS = (
    ("fit", 4, 1, 2, 4),
    ("fit", 6, 2, 6, 12),
    ("fit", 8, 2, 8, 24),
    ("fit", 8, 4, 40, 40),
    ("fit", 10, 3, 60, 100),
    ("fit", 10, 5, 252, 252),
    ("holdout", 6, 3, 12, 6),
    ("holdout", 8, 3, 20, 50),
    ("holdout", 10, 4, 100, 60),
)
PROFILE = {
    "shots": 100_000,
    "n_reps": 2,
    "num_batches": 2,
    "max_iterations": 2,
    "max_space": 12,
    "threads": 1,
    "schedule": "serial-retained-inputs-v1",
    "integrals": "active-orthonormal-synthetic",
    "carryover_threshold": "0",
    "sci_cutoffs": "1e-4,1e-5",
}
PACKAGES = (
    "numpy",
    "scipy",
    "pyscf",
    "ffsim",
    "qiskit",
    "qiskit-addon-sqd",
    "pennylane",
    "pennylane-catalyst",
    "jax",
    "jaxlib",
)
POLL_S = 0.01
ROOT = Path(__file__).resolve().parents[2]


def installed_versions() -> dict[str, str]:
    """Record actual versions, explicitly recording the optional Catalyst absence."""
    result = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            if package != "pennylane-catalyst":
                raise
            result[package] = "not-installed"
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def proc_bytes(pid: int) -> dict[str, int]:
    """Read Linux RSS/HWM with explicit KiB-to-bytes conversion."""
    values = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmHWM"):
                values[key] = int(value.split()[0]) * 1024
    except (OSError, ProcessLookupError):
        pass
    return values


def tree_pids(pid: int) -> set[int]:
    """Find descendants, including children started from non-main threads."""
    found = {pid}
    todo = [pid]
    while todo:
        current = todo.pop()
        for path in Path(f"/proc/{current}/task").glob("*/children"):
            try:
                children = {int(p) for p in path.read_text().split()} - found
                found.update(children)
                todo.extend(children)
            except OSError:
                pass
    return found


def host_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is required for guarded execution")


def run_monitored(
    command: list[str],
    prefix: Path,
    *,
    rss_cap_bytes: int = 4_096_000_000,
    timeout_s: float = 300,
    poll_s: float = POLL_S,
) -> dict:
    """Execute one process group, observing supervisor plus all live descendants.

    RSS is a polling measurement; lifetime child HWM is reported separately.
    Censored/failed executions are never accepted as calibration observations.
    """
    if not 0 < rss_cap_bytes <= 12_288_000_000 or not 0 < timeout_s < math.inf or poll_s <= 0:
        raise ValueError("Positive finite budgets and a hard-bounded RSS cap are required")
    cap = min(rss_cap_bytes, host_available_bytes() // 2)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        JAX_PLATFORMS="cpu",
        PYTHONHASHSEED="0",
        PYTHONPATH=str(ROOT / "src"),
        SQD_CALIBRATION_MONITORED="1",
    )
    start = time.monotonic()
    peak = child_hwm = max_processes = 0
    stage = "imports"
    peaks: dict[str, int] = {}
    reason = None
    events_path = prefix.with_suffix(".events.jsonl")
    events_path.touch()
    ack_path = prefix.with_suffix(".ack")
    with (
        prefix.with_suffix(".log").open("w") as log,
        events_path.open() as events,
        prefix.with_suffix(".rss.jsonl").open("w") as trace,
    ):
        proc = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while True:
                pending_ack = None
                for line in events:
                    event = json.loads(line)
                    stage = event["stage"]
                    child_hwm = max(child_hwm, event.get("VmHWM", 0))
                    pending_ack = event["time_ns"]
                pids = tree_pids(proc.pid) | {os.getpid()}
                rss = sum(proc_bytes(pid).get("VmRSS", 0) for pid in pids)
                elapsed = time.monotonic() - start
                peak = max(peak, rss)
                peaks[stage] = max(peaks.get(stage, 0), rss)
                max_processes = max(max_processes, len(pids))
                trace.write(json.dumps([elapsed, stage, rss, len(pids)]) + "\n")
                if pending_ack is not None:
                    ack_path.write_text(str(pending_ack))
                if proc.poll() is not None:
                    break
                if rss >= cap or host_available_bytes() < 512_000_000 or elapsed >= timeout_s:
                    reason = (
                        "rss"
                        if rss >= cap
                        else "host" if host_available_bytes() < 512_000_000 else "timeout"
                    )
                    break
                time.sleep(poll_s)
        finally:
            # Kill the group even if its leader exited, so a descendant cannot escape
            # by outliving that leader. Workers in this workload are strictly serial.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    result = {
        "command": command,
        "cwd": str(ROOT),
        "exit_code": proc.returncode,
        "status": "censored" if reason else "ok" if proc.returncode == 0 else "failed",
        "reason": reason,
        "wall_s": time.monotonic() - start,
        "peak_tree_bytes": peak,
        "child_lifetime_hwm_bytes": child_hwm,
        "stage_peaks_bytes": peaks,
        "max_processes": max_processes,
        "rss_cap_bytes": cap,
        "timeout_s": timeout_s,
        "poll_s": poll_s,
        "threads": 1,
        "log_sha256": digest(prefix.with_suffix(".log")),
    }
    write_json(prefix.with_suffix(".run.json"), result)
    return result


def guarded_call(operation: Callable[[], Any], norb: int, nelec: tuple[int, int], **kwargs) -> Any:
    """Check candidate memory before invoking an allocation-bearing operation."""
    from q2m3.sqd.resources import guard_allocation

    guard_allocation(norb, nelec, **kwargs)
    return operation()


class ScanPrior:
    """Uncalibrated scan prior; usable only inside the externally fused probe."""

    model_id = "uncalibrated-scan-prior"

    def upper_bound_mb(self, norb, nelec, *, subspace_dims, retained_mb, **kwargs):
        from q2m3.sqd.resources import workspace_features

        return 1024 + 2 * sum(workspace_features(norb, nelec, subspace_dims)[1:]) + retained_mb


def worker(point: tuple, prefix: Path, model_path: Path | None) -> None:
    """Measure active-space prototype stages with known retained input arrays."""
    events = prefix.with_suffix(".events.jsonl")

    def emit(stage, **values):
        stamp = time.monotonic_ns()
        with events.open("a") as stream:
            stream.write(
                json.dumps(dict(stage=stage, time_ns=stamp, **proc_bytes(os.getpid()), **values))
                + "\n"
            )
        if os.environ.get("SQD_CALIBRATION_MONITORED") == "1":
            ack = prefix.with_suffix(".ack")
            deadline = time.monotonic() + 10
            while not ack.exists() or ack.read_text() != str(stamp):
                if time.monotonic() > deadline:
                    raise TimeoutError("Supervisor did not acknowledge stage snapshot")
                time.sleep(0.001)

    emit("imports", event="start")
    import ffsim
    import numpy as np
    from pyscf import ao2mo, cc, fci, gto, scf
    from qiskit.primitives import BitArray
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci

    from q2m3.sqd.resources import addon_subspace_bound, load_resource_model

    model = ScanPrior() if model_path is None else load_resource_model(model_path, profile=PROFILE)
    _, norb, population, da, db = point
    nelec = (population, population)
    dims = (da, db)
    seed = 20260917
    inventory: dict[str, Any] = {}
    metrics = {}
    emit("imports", event="end")

    def stage(name, operation, **extra):
        label = extra.get("solver_method", name) if name == "reference" else name
        emit(label, event="start", inventory_bytes={k: int(v.nbytes) for k, v in inventory.items()})
        start = time.monotonic()
        result = guarded_call(
            operation,
            norb,
            nelec,
            stage=name,
            model=model,
            host_available_mb=host_available_bytes() / 2e6,
            rss_budget_mb=4096,
            **extra,
        )
        emit(
            label,
            event="end",
            wall_s=time.monotonic() - start,
            inventory_bytes={k: int(v.nbytes) for k, v in inventory.items()},
        )
        return result

    def integrals():
        # A stable, interacting active-space Hamiltonian in an orthonormal basis.
        # No arbitrary molecular AO basis / geometry scaling is being calibrated.
        rng = np.random.default_rng(seed)
        h1 = np.diag(np.arange(norb, dtype=float))
        h1 += 0.01 * (np.ones((norb, norb)) - np.eye(norb))
        factors = rng.normal(scale=0.015, size=(norb, norb, norb))
        factors += factors.transpose(0, 2, 1)
        h2 = np.einsum("Lpq,Lrs->pqrs", factors, factors)
        mol = gto.M(verbose=0)
        mol.nelectron = 2 * population
        mol.incore_anyway = True
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args: h1
        mf.get_ovlp = lambda *args: np.eye(norb)
        mf._eri = ao2mo.restore(8, h2, norb)
        mf.kernel(dm0=np.diag([2.0] * population + [0.0] * (norb - population)))
        assert mf.converged
        one = mf.mo_coeff.T @ h1 @ mf.mo_coeff
        two = ao2mo.incore.full(h2, mf.mo_coeff, compact=False).reshape((norb,) * 4)
        inventory.update(h1=one, h2=two, mo_coeff=mf.mo_coeff, ao_eri=mf._eri, ao_h1=h1, ao_h2=h2)
        metrics["hf_energy_ha"] = float(mf.e_tot)
        return mf, one, two

    mf, h1, h2 = stage("integrals", integrals)

    def ccsd():
        solver = cc.CCSD(mf)
        solver.conv_tol = 1e-10
        solver.kernel()
        assert solver.converged
        inventory.update(t1=solver.t1, t2=solver.t2)
        metrics["ccsd_energy_ha"] = float(solver.e_tot)
        return solver.t1, solver.t2

    t1, t2 = stage("ccsd", ccsd)

    def prepare():
        # Exercise real amplitude decomposition and a dense UCJ state as a
        # conservative occupancy profile, independently of chemical accuracy.
        seeded = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1=t1, n_reps=2)
        dense = ffsim.random.random_ucj_op_spin_balanced(norb, n_reps=2, seed=seed)
        vec = ffsim.hartree_fock_state(norb, nelec)
        vec = ffsim.apply_unitary(vec, seeded, norb=norb, nelec=nelec)
        vec = ffsim.apply_unitary(vec, dense, norb=norb, nelec=nelec)
        assert abs(np.vdot(vec, vec).real - 1) < 1e-10
        inventory["state"] = vec
        return vec

    stage("prepare", prepare)

    def sample():
        bits = np.asarray(
            ffsim.sample_state_vector(
                inventory["state"],
                norb=norb,
                nelec=nelec,
                shots=PROFILE["shots"],
                bitstring_type=ffsim.BitstringType.BIT_ARRAY,
                seed=seed,
            )
        )
        assert bits.shape == (PROFILE["shots"], 2 * norb)
        assert np.all(bits[:, :norb].sum(axis=1) == population)
        assert np.all(bits[:, norb:].sum(axis=1) == population)
        inventory["bits"] = bits
        return bits

    bits = stage("sample", sample)
    del inventory["state"]

    def diagonalize():
        strings = np.asarray(fci.cistring.make_strings(range(norb), population))
        include = (strings[:da], strings[:db])
        seen = []

        def fixed_solver(batches, one, two, n, electrons):
            outputs = []
            for a, b in batches:
                actual = (len(a), len(b))
                seen.append(actual)
                outputs.append(
                    guarded_call(
                        lambda a=a, b=b: solve_sci(
                            (a, b), one, two, n, electrons, max_space=12, max_cycle=100
                        ),
                        n,
                        electrons,
                        stage="diagonalize",
                        subspace_dims=actual,
                        model=model,
                        host_available_mb=host_available_bytes() / 2e6,
                        rss_budget_mb=4096,
                    )
                )
            return outputs

        # Invalid occupations force recovery; full sector/max_dim prebound is
        # independent of recovery results, carryover and spin symmetrization.
        noisy = bits.copy()
        noisy[::3, 0] ^= True
        bit_array = BitArray.from_bool_array(noisy, order="big")
        result = diagonalize_fermionic_hamiltonian(
            h1,
            h2,
            bit_array,
            300,
            norb,
            nelec,
            num_batches=2,
            max_iterations=2,
            max_dim=dims,
            include_configurations=include,
            carryover_threshold=0,
            symmetrize_spin=False,
            sci_solver=fixed_solver,
            seed=seed,
        )
        inventory["sqd_amplitudes"] = result.sci_state.amplitudes
        metrics["sqd_energy_ha"] = float(result.energy)
        metrics["fixed_candidates"] = seen
        assert all(a <= da and b <= db for a, b in seen)
        assert result.sci_state.amplitudes.shape == dims
        return result

    result = stage(
        "diagonalize", diagonalize, subspace_dims=addon_subspace_bound(norb, nelec, max_dim=dims)
    )

    def exact():
        energy, ci = fci.direct_spin1.kernel(h1, h2, norb, nelec, max_space=12)
        assert np.isfinite(energy)
        metrics["exact_energy_ha"] = float(energy)
        return ci

    exact_ci = stage("reference", exact, solver_method="exact_casci")
    del exact_ci

    def selected():
        for cutoff in (1e-4, 1e-5):
            solver = fci.selected_ci.SelectedCI()
            solver.select_cutoff = cutoff
            energy, ci = solver.kernel(h1, h2, norb, nelec, max_space=12)
            assert solver.converged and np.isfinite(energy)
            metrics[f"selected_energy_{cutoff}_ha"] = float(energy)
        return ci

    selected_ci = stage("reference", selected, solver_method="selected_ci_pyscf")
    del selected_ci

    def comparison():
        strings = np.asarray(fci.cistring.make_strings(range(norb), population))
        rng = np.random.default_rng(seed)
        # Simultaneously retained ranked and random candidate pools; include HF.
        pools = []
        for count in dims:
            rank = strings[:count].copy()
            random = np.sort(np.r_[strings[0], rng.choice(strings[1:], count - 1, replace=False)])
            pools.extend((rank, random))
        for i, array in enumerate(pools):
            inventory[f"comparison_pool_{i}"] = array
        return pools

    pools = stage("comparison", comparison, subspace_dims=dims)
    assert pools and result.sci_state.amplitudes.size == da * db
    metrics.update(
        point=point,
        seed=seed,
        active_electrons=2 * population,
        system_qubits=2 * norb,
        estimation_qubits=0,
        ci_dim=math.comb(norb, population) ** 2,
        model_id=model.model_id,
        inventory_bytes={k: int(v.nbytes) for k, v in inventory.items()},
    )
    write_json(prefix.with_suffix(".metrics.json"), metrics)


def fit(output: Path) -> None:
    """Freeze nonnegative stage fits, floors and margin using only fit records."""
    import numpy as np
    from scipy.optimize import nnls

    from q2m3.sqd.resources import STAGES, workspace_features

    coefficients = {}
    for stage in STAGES:
        rows, values = [], []
        for index, (kind, norb, population, da, db) in enumerate(POINTS):
            if kind != "fit":
                continue
            dims = (da, db) if stage in ("diagonalize", "comparison") else None
            if stage == "reference":
                dims = (math.comb(norb, population),) * 2
            observed = []
            for repeat in range(3):
                record = json.loads((output / f"p{index}-r{repeat}.run.json").read_text())
                if record["status"] != "ok":
                    raise RuntimeError("Censored or failed point cannot enter fitting")
                names = ("exact_casci", "selected_ci_pyscf") if stage == "reference" else (stage,)
                observed.append(max(record["stage_peaks_bytes"][name] for name in names) / 1e6)
            rows.append(workspace_features(norb, (population, population), dims))
            values.append(max(observed))
        x, y = np.asarray(rows), np.asarray(values)
        floors = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        correction, _ = nnls(x, np.maximum(0, y - x @ floors))
        coef = floors + correction
        coef[0] += max(0.0, float(np.max(y - x @ coef))) + 64
        coefficients[stage] = (1.25 * coef).tolist()
    data = {
        "schema": "sqd.rss.v1",
        "status": "training_only",
        "model_id": "serial-active-space-20260917-v1",
        "versions": installed_versions(),
        "profile": PROFILE,
        "domain": {"max_norb": 10, "max_ci_dim": 63504, "max_sector_dim": 252},
        "coefficients": coefficients,
        "method": "NNLS over training maxima; allocation floors; max positive residual + 64 MB; times 1.25",
        "preregistration_sha256": digest(output / "preregistration.json"),
    }
    write_json(output / "model-training.json", data)


def validate_holdouts(output: Path) -> None:
    from q2m3.sqd.resources import STAGES, CalibratedResourceModel

    data = json.loads((output / "model-training.json").read_text())
    model = CalibratedResourceModel(
        data["model_id"],
        tuple((s, tuple(data["coefficients"][s])) for s in STAGES),
        **data["domain"],
    )
    comparisons = []
    for index, point in enumerate(POINTS):
        kind, norb, population, da, db = point
        for repeat in range(3):
            record = json.loads((output / f"p{index}-r{repeat}.run.json").read_text())
            if record["status"] != "ok":
                raise RuntimeError("Every registered point must finish uncensored")
            for stage in STAGES:
                methods = ("exact_casci", "selected_ci_pyscf") if stage == "reference" else (None,)
                for method in methods:
                    dims = (da, db) if stage in ("diagonalize", "comparison") else None
                    predicted = model.upper_bound_mb(
                        norb,
                        (population, population),
                        stage=stage,
                        subspace_dims=dims,
                        retained_mb=0,
                        solver_method=method,
                    )
                    measured = record["stage_peaks_bytes"][method or stage] / 1e6
                    comparisons.append(
                        dict(
                            point=index,
                            repeat=repeat,
                            split=kind,
                            stage=stage,
                            solver_method=method,
                            predicted_mb=predicted,
                            measured_mb=measured,
                            margin_mb=predicted - measured,
                            passed=predicted >= measured,
                        )
                    )
    write_json(output / "validation.json", comparisons)
    if not all(row["passed"] for row in comparisons):
        raise RuntimeError(
            "Model failed holdouts; preserve failure and preregister a new calibration"
        )
    data["status"] = "validated_holdout"
    data["validation"] = {
        "fit_points": 6,
        "holdout_points": 3,
        "repeats": 3,
        "all_passed": True,
        "minimum_margin_mb": min(r["margin_mb"] for r in comparisons),
        "observations_sha256": digest(output / "validation.json"),
    }
    write_json(output / "resource_calibration.json", data)


def scan(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    lock_path = ROOT / "tmp/sqd/heavy-experiment.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as slot:
        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prereg = {
            "points": POINTS,
            "repeats": 3,
            "seed": 20260917,
            "profile": PROFILE,
            "fit": "NNLS floors [0,1,1,1,1], positive residual +64MB, multiply 1.25; training maxima",
            "gate": "every held-out stage sample <= frozen prediction; no tuning on holdout",
            "poll_s": POLL_S,
            "rss_cap_bytes": 4096000000,
            "timeout_s": 300,
            "python": sys.version,
            "platform": platform.platform(),
            "versions": installed_versions(),
            "host_available_bytes": host_available_bytes(),
            "source_sha256": digest(Path(__file__)),
            "lock_sha256": digest(ROOT / "uv.lock"),
            "scope": "active synthetic integrals, bounded serial prototypes, not production-chain acceptance",
        }
        write_json(output / "preregistration.json", prereg)
        for index, point in enumerate(POINTS):
            if index == 6:
                subprocess.run([sys.executable, __file__, "--fit", str(output)], check=True)
            for repeat in range(3):
                prefix = output / f"p{index}-r{repeat}"
                record = run_monitored(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        json.dumps(point),
                        "--prefix",
                        str(prefix),
                    ],
                    prefix,
                )
                print(
                    f"{prefix.name}: {record['status']} {record['peak_tree_bytes']/1e6:.1f} MB {record['wall_s']:.1f}s",
                    flush=True,
                )
                if record["status"] != "ok":
                    raise RuntimeError(f"Stop at {prefix}: {record['status']}")
        subprocess.run([sys.executable, __file__, "--validate", str(output)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker")
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(tuple(json.loads(args.worker)), args.prefix, args.model)
    elif args.fit:
        fit(args.fit)
    elif args.validate:
        validate_holdouts(args.validate)
    elif args.output:
        scan(args.output.resolve())
    else:
        parser.error("--output is required")


if __name__ == "__main__":
    main()
