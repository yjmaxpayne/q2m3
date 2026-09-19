"""Reproducible molecular SQD benchmarks with complete results and RSS evidence.

Run in the installed SQD environment with single-thread BLAS. Each invocation
creates a new output directory. Defaults are the public LUCJ defaults; historical
profiles are attempted without reducing their requested layers or shots. RSS is
sampled every 10 ms with stage-boundary handshakes, not an arbitrary-transient
upper bound. Four-arm comparisons include the SCI pool construction cost.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("integrals", "ccsd", "prepare", "sample", "diagonalize", "reference", "comparison")
LIMIT_BYTES = 2_000_000_000


def _load(filename):
    spec = importlib.util.spec_from_file_location(
        Path(filename).stem, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory_verdict(sample_bytes: int | None, complete_bytes: int | None) -> str:
    """Judge both observed windows against the strict decimal two-GB limit."""
    if any(v is not None and v >= LIMIT_BYTES for v in (sample_bytes, complete_bytes)):
        return "fail"
    if sample_bytes is None or complete_bytes is None:
        return "inconclusive"
    return "pass" if max(sample_bytes, complete_bytes) < LIMIT_BYTES else "fail"


def glycine_statistics(rows: list[dict]) -> dict:
    """Summarize five distinct sparse-space seeds; the range is not a CI."""
    if len(rows) != 5 or {r["seed"] for r in rows} != set(range(5)):
        raise ValueError("Glycine requires seeds 0..4")
    if any(r["subspace_dim"] >= r["full_ci_dim"] for r in rows):
        raise ValueError("Glycine requires non-full spaces")
    values = [r["delta_mHa"] for r in rows]
    if any(v is None or not math.isfinite(v) for v in values):
        raise ValueError("Missing or nonfinite seed result")
    return dict(
        mean_mHa=statistics.mean(values),
        sample_std_mHa=statistics.stdev(values),
        max_abs_error_mHa=max(map(abs, values)),
        min_mHa=min(values),
        max_mHa=max(values),
        range_is_confidence_interval=False,
    )


def audit_record(row: dict) -> dict:
    """Independently check units, result completeness, fairness and RSS semantics.

    Args:
        row: A completed benchmark record containing the complete public result.

    Returns:
        Independent scalar checks, with no requirement that SQD beat another arm.

    Raises:
        ValueError: A scientific claim or measurement contradicts its evidence.
    """
    r = row["sqd"]
    if len(r) != 40 or r["iso_active_space_ccsd_energy"] is None:
        raise ValueError("Incomplete four-arm result")
    for field, arm in [
        ("delta_mHa", "baseline_energy"),
        ("delta_vs_sci_mHa", "iso_ndet_sci_energy"),
    ]:
        expected = (r["sqd_energy"] - r[arm]) * 1000
        if not math.isclose(expected, r[field], abs_tol=1e-9, rel_tol=0):
            raise ValueError("Incorrect energy difference or mHa units")
    if r["baseline_tier"] == "T0" and r["sqd_energy"] < r["baseline_energy"] - 1e-8:
        raise ValueError("Variational bound violated")
    if r["baseline_tier"] != "T0" and (r["ratio_sqd_over_sci"] is not None or not r["warnings"]):
        raise ValueError("Approximate reference presented as exact")
    dims = r["subspace_dims"]
    if math.prod(dims) != r["subspace_dim"] or r["subspace_dim"] > r["full_ci_dim"]:
        raise ValueError("Incorrect subspace dimension")
    if row["interpretation"] != (
        "convention_regression_not_accuracy_advantage"
        if row["system"] in ("h2", "h3o")
        else "finite_sample_comparison"
    ):
        raise ValueError("Unsupported accuracy claim")
    if r["subspace_dim"] == r["full_ci_dim"] and row["full_space_discriminating"]:
        raise ValueError("Full space is nondiscriminating")
    electrons, norb = r["active_space"]
    for strings in r["diagnostics"]["comparison_ci_strings"].values():
        for spin, seq in enumerate(strings):
            if (
                len(seq) != dims[spin]
                or len(set(seq)) != len(seq)
                or (1 << (electrons // 2)) - 1 not in seq
                or any(n < 0 or n >= 1 << norb or n.bit_count() != electrons // 2 for n in seq)
            ):
                raise ValueError("Unfair or illegal spin-sector comparison")
    curve = r["unique_dets_vs_shots"]
    if (
        curve[-1] != [r["shots"], row["sampling"]["unique_pairs"]]
        or row["sampling"]["shape"] != [r["shots"], 2 * norb]
        or row["sampling"]["spin_counts"] != [electrons // 2, electrons // 2]
        or not row["sampling"]["valid"]
    ):
        raise ValueError("Sampling integrity or unique curve mismatch")
    if any(
        s2 <= s1 or u2 < u1 or u2 > s2 for (s1, u1), (s2, u2) in zip(curve, curve[1:], strict=False)
    ):
        raise ValueError("Invalid unique curve")
    resources = row["resources"]
    if not all(resources["stage_complete"].values()):
        raise ValueError("Incomplete stage measurement")
    if resources["peak_tree_bytes"] < max(resources["stage_peaks_bytes"].values()):
        raise ValueError("Sampling-only RSS cannot represent the complete run")
    verdict = memory_verdict(
        resources["stage_peaks_bytes"].get("sample"), resources["peak_tree_bytes"]
    )
    if row["memory_verdict"] != verdict:
        raise ValueError("Incorrect memory verdict")
    return dict(
        sampling_integrity=True,
        four_arm_spin_fairness=True,
        variational_check="passed" if r["baseline_tier"] == "T0" else "not_exact_reference",
        delta_vs_ccsd_mHa=1000 * (r["sqd_energy"] - r["iso_active_space_ccsd_energy"]),
        subspace_fraction=r["subspace_dim"] / r["full_ci_dim"],
    )


def _instrument(directory, monitor):
    """Observe production calls without changing their inputs, guards or solvers."""
    import functools

    import numpy as np

    import q2m3.sqd.orchestrator as workflow
    from q2m3.sqd.sampling import FfsimSampler

    current = ["imports"]
    events = directory / "events.jsonl"
    ack = directory / "ack"
    root_pid = os.getpid()

    def emit(stage, boundary):
        stamp = time.monotonic_ns()
        event = dict(
            stage=stage,
            boundary=boundary,
            time_ns=stamp,
            pid=os.getpid(),
            **monitor.proc_bytes(os.getpid()),
        )
        with events.open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        end = time.monotonic() + 10
        while not ack.exists() or ack.read_text() != str(stamp):
            if time.monotonic() >= end:
                raise TimeoutError("Stage monitor handshake missing")
            time.sleep(0.001)

    def wrap(function, stage, capture=None):
        @functools.wraps(function)
        def call(*args, **kwargs):
            current[0] = stage
            emit(stage, "start")
            value = function(*args, **kwargs)
            emit(stage, "end")
            if capture is not None:
                capture(value)
            current[0] = "between"
            return value

        return call

    def capture_integrals(data):
        np.savez(
            directory / "integrals.npz",
            h1=data.h1,
            h2=data.h2,
            e_core=data.e_core,
            norb=data.norb,
            nelec=data.nelec,
            mo_coeff=data.mo_coeff,
        )

    def capture_diagonalized(value):
        np.savez(
            directory / "subspace.npz",
            alpha=value.ci_strings[0],
            beta=value.ci_strings[1],
            amplitudes=value.amplitudes,
        )

    def capture_samples(samples):
        np.save(directory / "samples.npy", samples)
        n = samples.shape[1] // 2
        _write(
            directory / "sampling.json",
            dict(
                shape=list(samples.shape),
                dtype=str(samples.dtype),
                valid=bool(
                    samples.dtype == bool
                    and np.all(samples[:, :n].sum(1) == samples[0, :n].sum())
                    and np.all(samples[:, n:].sum(1) == samples[0, n:].sum())
                ),
                spin_counts=[int(samples[0, n:].sum()), int(samples[0, :n].sum())],
                unique_pairs=int(len(np.unique(samples, axis=0))),
            ),
        )

    # Every imported guard alias is wrapped so its actual inventory is retained.
    from q2m3.sqd import resources

    original_guard = resources.guard_allocation

    def guard(*args, **kwargs):
        predicted = original_guard(*args, **kwargs)
        outside = monitor.tree_pids(root_pid) - monitor.tree_pids(os.getpid())
        extra = sum(monitor.proc_bytes(pid).get("VmRSS", 0) for pid in outside)
        with (directory / "guards.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        stage=current[0],
                        guard_stage=kwargs["stage"],
                        predicted_bytes=predicted * 1e6,
                        outside_caller_bytes=extra,
                        tree_bound_bytes=predicted * 1e6 + extra,
                        pid=os.getpid(),
                    )
                )
                + "\n"
            )
        return predicted

    for name, module in tuple(sys.modules.items()):
        if (
            name.startswith("q2m3.sqd")
            and getattr(module, "guard_allocation", None) is original_guard
        ):
            module.guard_allocation = guard
    for name, stage, capture in [
        ("build_integrals", "integrals", capture_integrals),
        ("build_ccsd_seed", "ccsd", None),
        ("build_lucj_from_integrals", "prepare", None),
        ("diagonalize_samples", "diagonalize", capture_diagonalized),
        ("run_reference", "reference", None),
        ("run_comparisons", "comparison", None),
    ]:
        setattr(workflow, name, wrap(getattr(workflow, name), stage, capture))
    FfsimSampler.prepare = wrap(FfsimSampler.prepare, "prepare")
    FfsimSampler.sample = wrap(FfsimSampler.sample, "sample", capture_samples)
    return workflow


def _worker(directory):
    import numpy as np
    from pyscf import gto

    from q2m3.sqd.config import LUCJConfig, ReferenceConfig
    from q2m3.sqd.exceptions import ResourceModelDomainError
    from q2m3.utils.io import save_json_results

    request = json.loads((directory / "request.json").read_text())
    spec = request["system"]
    workflow = _instrument(directory, _load("calibrate_resources.py"))
    mol = gto.M(atom=spec["atom"], basis=spec["basis"], charge=spec["charge"], verbose=0)
    try:
        result = workflow.run_sqd(
            [mol.atom_symbol(i) for i in range(mol.natm)],
            np.array(mol.atom_coords(unit="Angstrom")),
            active_electrons=spec["active_space"][0],
            active_orbitals=spec["active_space"][1],
            charge=spec["charge"],
            basis=spec["basis"],
            lucj=LUCJConfig(**request["profile"]),
            reference=ReferenceConfig(),
            seed=request["seed"],
            verbose=False,
        )
    except ResourceModelDomainError as exc:
        _write(
            directory / "rejection.json",
            dict(
                type=type(exc).__name__,
                message=str(exc),
                reason="outside_calibration_domain",
                rejected_dimensions=_load("connectivity_comparison.py").profile_rejections(
                    request["profile"] | {"num_batches": 2, "max_iterations": 2}
                ),
            ),
        )
        return
    save_json_results(result, directory / "result.json")


def _measure(directory):
    monitor = _load("calibrate_resources.py")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=str(ROOT / "src"),
        JAX_PLATFORMS="cpu",
        PYTHONHASHSEED="0",
    )
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(directory)]
    start = time.monotonic()
    stage = "imports"
    peak = 0
    peaks, boundaries, hwm = {}, {}, {}
    seen = set()
    reason = None
    events_path = directory / "events.jsonl"
    events_path.touch()
    with (
        (directory / "worker.log").open("w") as log,
        events_path.open() as events,
        (directory / "rss.jsonl").open("w") as trace,
    ):
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while True:
                pending = []
                for line in events:
                    event = json.loads(line)
                    stage = event["stage"]
                    boundaries.setdefault(stage, []).append(event["boundary"])
                    hwm[str(event["pid"])] = max(
                        hwm.get(str(event["pid"]), 0), event.get("VmHWM", 0)
                    )
                    pending.append(event)
                pids = monitor.tree_pids(process.pid)
                seen.update(pids)
                rss = sum(monitor.proc_bytes(pid).get("VmRSS", 0) for pid in pids)
                peak = max(peak, rss)
                peaks[stage] = max(peaks.get(stage, 0), rss)
                elapsed = time.monotonic() - start
                trace.write(json.dumps([elapsed, stage, rss, len(pids)]) + "\n")
                for event in pending:
                    (directory / "ack").write_text(str(event["time_ns"]))
                    if event["boundary"] == "end":
                        stage = "between"
                if process.poll() is not None:
                    break
                if (
                    rss >= 8_192_000_000
                    or elapsed >= 900
                    or monitor.host_available_bytes() < 512_000_000
                ):
                    reason = "external_budget_exceeded"
                    break
                time.sleep(0.01)
        finally:
            seen.update(monitor.tree_pids(process.pid))
            for pid in seen:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
    guards_path = directory / "guards.jsonl"
    guards = (
        [json.loads(x) for x in guards_path.read_text().splitlines()]
        if guards_path.exists()
        else []
    )
    checks = {}
    for name in STAGES:
        predictions = [g["tree_bound_bytes"] for g in guards if g["stage"] == name]
        bound = max(predictions) if predictions else None
        actual = peaks.get(name)
        checks[name] = dict(
            predicted_tree_bytes=bound,
            observed_tree_bytes=actual,
            passed=None if bound is None or actual is None else actual <= bound,
        )
    return dict(
        command=command,
        exit_code=process.returncode,
        reason=reason,
        peak_tree_bytes=peak,
        stage_peaks_bytes=peaks,
        wall_s=time.monotonic() - start,
        poll_s=0.01,
        stage_complete={
            s: boundaries.get(s, []).count("start") == boundaries.get(s, []).count("end")
            and bool(boundaries.get(s))
            for s in STAGES
        },
        process_lifetime_hwm_bytes=hwm,
        g2b_stage_checks=checks,
        rss_cap_bytes=8_192_000_000,
        wall_budget_s=900,
        scope="fresh interpreter through process exit; root plus simultaneous descendants; observer excluded",
        g2b_scope="max actual allocation bound per stage plus simultaneously resident outside-caller RSS; no refit",
        measurement_limit="10ms polling and boundary snapshots do not bound arbitrary transients",
    )


def run_benchmark(system: str, output_dir: Path) -> dict:
    """Run default and applicable historical profiles in clean serial processes.

    Args:
        system: h2, h3o, glycine or n2.
        output_dir: New exclusive output directory.

    Returns:
        Complete machine-readable report with measurements and missing-data reasons.
    """
    output_dir = Path(output_dir).resolve()
    spec = next(s for s in _load("connectivity_comparison.py").SYSTEMS if s["name"] == system)
    output_dir.mkdir(parents=True, exist_ok=False)
    profiles = {"default": dict(n_reps=2, shots=100000)}
    if system in ("glycine", "n2"):
        profiles["historical"] = dict(n_reps=4, shots=200000 if system == "glycine" else 500000)
    sources = [
        Path(__file__),
        Path(__file__).with_name("connectivity_comparison.py"),
        Path(__file__).with_name("calibrate_resources.py"),
        ROOT / "uv.lock",
        *sorted((ROOT / "src/q2m3").rglob("*.py")),
        ROOT / "src/q2m3/sqd/resource_calibration.json",
        ROOT / "pyproject.toml",
        ROOT / "tests/examples/test_sqd_benchmarks.py",
    ]
    manifest = dict(
        schema="sqd.benchmark.manifest.v1",
        system=spec,
        profiles=profiles,
        seeds=list(range(5)) if system == "glycine" else [0],
        coordinate_unit="Angstrom",
        energy_unit="Ha",
        difference_unit="mHa",
        system_qubits=2 * spec["active_space"][1],
        estimation_qubits=0,
        versions=_load("calibrate_resources.py").installed_versions(),
        python=sys.version,
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={str(p.relative_to(ROOT)): _hash(p) for p in sources},
        memory_threshold_bytes=LIMIT_BYTES,
        diagonalizer="current public defaults: samples_per_batch=min(300, shots), num_batches=2, max_iterations=2; historical label only preserves reps/shots",
        command=[
            sys.executable,
            str(Path(__file__).resolve()),
            "--system",
            system,
            "--output",
            str(output_dir),
        ],
    )
    _write(output_dir / "manifest.json", manifest)
    (output_dir / "manifest.json").chmod(0o444)
    records = []
    for profile_name, profile in profiles.items():
        for seed in manifest["seeds"]:
            directory = output_dir / f"{profile_name}-seed{seed}"
            directory.mkdir()
            _write(directory / "request.json", dict(system=spec, profile=profile, seed=seed))
            resources = _measure(directory)
            _write(directory / "resources.json", resources)
            row = dict(system=system, profile=profile_name, seed=seed, resources=resources)
            if (
                (directory / "result.json").exists()
                and resources["exit_code"] == 0
                and resources["reason"] is None
            ):
                result = json.loads((directory / "result.json").read_text())
                row.update(
                    status="completed",
                    sqd=result,
                    sampling=json.loads((directory / "sampling.json").read_text()),
                    interpretation=(
                        "convention_regression_not_accuracy_advantage"
                        if system in ("h2", "h3o")
                        else "finite_sample_comparison"
                    ),
                    full_space_discriminating=False,
                    memory_verdict=memory_verdict(
                        resources["stage_peaks_bytes"].get("sample"), resources["peak_tree_bytes"]
                    ),
                )
                row["audit"] = audit_record(row)
            else:
                rejection = (
                    json.loads((directory / "rejection.json").read_text())
                    if (directory / "rejection.json").exists()
                    else dict(
                        reason="execution_failed",
                        message=(directory / "worker.log").read_text()[-4000:],
                    )
                )
                row.update(
                    status="unavailable",
                    sqd=None,
                    null_reason=rejection,
                    memory_verdict="inconclusive",
                    numerical_solve_completed=False,
                )
            records.append(row)
    report = dict(
        schema="sqd.benchmark.v1",
        manifest=manifest,
        records=records,
        g2b_registered_default_cases=all(
            r["status"] == "completed"
            and all(c["passed"] is True for c in r["resources"]["g2b_stage_checks"].values())
            for r in records
            if r["profile"] == "default"
        ),
        historical_scope_status=(
            "outside_certified_domain" if "historical" in profiles else "not_requested"
        ),
        scientific_limits=[
            "No chemical-accuracy or quantum-advantage promise.",
            "Historical unreachable profiles have no energy or memory pass.",
            "Finite-seed range is not a confidence interval.",
            "Logical full-connectivity simulation; no hardware readiness claim.",
        ],
    )
    default = [
        r["sqd"] for r in records if r["profile"] == "default" and r["status"] == "completed"
    ]
    report["statistics"] = (
        glycine_statistics(default) if system == "glycine" and len(default) == 5 else None
    )
    report["statistics_reason"] = (
        None if report["statistics"] is not None else "not_five_valid_glycine_seeds"
    )
    _write(output_dir / "report.json", report)
    hashes = {
        str(p.relative_to(output_dir)): _hash(p) for p in output_dir.rglob("*") if p.is_file()
    }
    _write(output_dir / "hashes.json", hashes)
    for p in output_dir.rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    verify_artifacts(output_dir)
    return report


def verify_artifacts(output_dir: Path) -> dict:
    """Read and verify every archived artifact and completed scientific record."""
    output_dir = Path(output_dir)
    for name, digest in json.loads((output_dir / "hashes.json").read_text()).items():
        if _hash(output_dir / name) != digest:
            raise ValueError(f"artifact hash mismatch: {name}")
    report = json.loads((output_dir / "report.json").read_text())
    if report["manifest"] != json.loads((output_dir / "manifest.json").read_text()):
        raise ValueError("manifest/report mismatch")
    for row in report["records"]:
        if row["status"] == "completed":
            directory = output_dir / f"{row['profile']}-seed{row['seed']}"
            if row["sqd"] != json.loads((directory / "result.json").read_text()):
                raise ValueError("Complete 40-field result was not preserved")
            if row["audit"] != audit_record(row):
                raise ValueError("report audit mismatch")
    return report


def main(system: str = "n2") -> None:
    """Run an example or its private measurement worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=["h2", "h3o", "glycine", "n2"], default=system)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
    else:
        report = run_benchmark(args.system, args.output or Path(f"tmp/{args.system}-sqd"))
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
