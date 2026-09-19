"""Replay the registered orbital scan's shared resource preflight.

The requested workload is outside the shipped calibration. This driver records
that limitation before constructing any orbitals. It does not implement the four
orbital solvers or claim their labels were consumed. If the resource gate opens,
it stops: a numerical scan and its frame-consistency oracles are then required.

Run in the frozen SQD environment with single-thread BLAS::

    python tools/sqd/orbital_basis_scan.py --output tmp/orbital-scan

An output directory is created exclusively. Read-only files and SHA256 checks
make the registration tamper-evident; they are not a security boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
BASES = ("canonical_rhf", "natural", "casscf", "localized")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o444)


def expect_domain_rejection(call, expected):
    """Capture only the specified workload rejection; propagate other failures."""
    from q2m3.sqd.exceptions import ResourceModelDomainError

    start = monotonic()
    try:
        call()
    except ResourceModelDomainError as exc:
        if f"uncalibrated workload {expected}; limit=" not in str(exc):
            raise
        return {
            "exception": type(exc).__name__,
            "message": str(exc),
            "wall_s": monotonic() - start,
            "predicted_rss_mb": None,
            "prediction_reason": "outside_calibration_domain",
        }
    raise RuntimeError("Workload accepted: implement and validate numerical orbital scan first")


def run_scan(output: Path) -> dict:
    """Register unchanged inputs, probe production guards, and write missing-data records.

    Args:
        output: New evidence directory; existing directories are never overwritten.

    Returns:
        Inconclusive report, explicitly limited to shared preflight evidence.

    Raises:
        FileExistsError: Output already exists.
        RuntimeError: The formerly unsupported workload is now accepted.
    """
    output.mkdir(parents=True, exist_ok=False)
    experiment = {
        "symbols": ["N", "N"],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]],
        "coordinate_unit": "Angstrom",
        "geometry_choice": "equilibrium probe, fixed before execution",
        "basis_set": "cc-pvdz",
        "charge": 0,
        "active_space": [10, 10],
        "system_qubits": 20,
        "nelec": [5, 5],
        "n_reps": 4,
        "shots": 500000,
        "seeds": [0, 1, 2],
        "orbital_bases": list(BASES),
        "wall_budget_s_per_entry": 60.0,
        "rss_budget_mb": 8192.0,
        "allow_large": False,
        "mode": "full",
        "threshold_mHa": 2.0,
        "error_definition": "1000 * (E_sqd,b - E_exact,b)",
        "reference_rule": "own exact for changed span; verify rotated exact before sharing scalar",
        "seed_rule": "rebuild CCSD in each integral frame; never relabel RHF amplitudes",
        "classification": "missing or censored points => inconclusive; no capability claim",
    }
    sources = sorted((ROOT / "src/q2m3").rglob("*.py")) + [
        Path(__file__).resolve(),
        ROOT / "src/q2m3/sqd/resource_calibration.json",
        ROOT / "uv.lock",
    ]
    manifest = {
        "schema": "orbital-scan.preflight.v1",
        "experiment": experiment,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes": {str(p.relative_to(ROOT)): _hash(p) for p in sources},
        "versions": {
            package: version(package)
            for package in ("numpy", "scipy", "pyscf", "ffsim", "qiskit", "qiskit-addon-sqd")
        },
        "python": sys.version,
        "platform": platform.platform(),
        "threads": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "command": sys.argv,
    }
    _write(output / "manifest.json", manifest)  # Registration precedes all solver calls.
    from q2m3.sqd.config import LUCJConfig, ReferenceConfig
    from q2m3.sqd.orchestrator import run_sqd
    from q2m3.sqd.resources import load_resource_model

    # Check environment independently: a version mismatch is not workload evidence.
    model = load_resource_model()
    profile_probes = {
        key: expect_domain_rejection(
            lambda key=key: load_resource_model(profile={key: experiment[key]}),
            f"{key}={experiment[key]}",
        )
        for key in ("n_reps", "shots")
    }
    probes = []
    for seed in experiment["seeds"]:
        probe = expect_domain_rejection(
            lambda seed=seed: run_sqd(
                experiment["symbols"],
                experiment["coordinates"],
                active_electrons=experiment["active_space"][0],
                active_orbitals=experiment["active_space"][1],
                basis=experiment["basis_set"],
                charge=experiment["charge"],
                lucj=LUCJConfig(n_reps=experiment["n_reps"], shots=experiment["shots"]),
                reference=ReferenceConfig(
                    wall_budget_s=experiment["wall_budget_s_per_entry"],
                    rss_budget_mb=experiment["rss_budget_mb"],
                ),
                seed=seed,
                allow_large=experiment["allow_large"],
                mode=experiment["mode"],
                verbose=False,
            ),
            f"n_reps={experiment['n_reps']}",
        )
        probes.append(probe | {"seed": seed, "entry": "run_sqd", "scope": "shared_preflight"})
    records = [
        {
            "basis": basis,
            "seed": seed,
            "status": "blocked_before_orbital_construction",
            "orbital_method_consumed": False,
            "evidence_seed": seed,
            "evidence_scope": "shared_preflight_not_four_independent_orbital_runs",
            "frame_id": None,
            "hamiltonian_id": None,
            "active_span_relation": None,
            "ccsd_converged": None,
            "subspace_dims": None,
            "sqd_result": None,
            "energies_ha": dict.fromkeys(("sqd", "ccsd", "sci", "random", "exact")),
            "delta_mHa": None,
            "null_reason": "uncalibrated_workload",
        }
        for basis in experiment["orbital_bases"]
        for seed in experiment["seeds"]
    ]
    report = {
        "schema": "orbital-scan.preflight.v1",
        "manifest_sha256": _hash(output / "manifest.json"),
        "conclusion": "inconclusive",
        "scope": "shared_preflight_only",
        "reason": "uncalibrated_workload_not_measured_memory_exhaustion",
        "model_id": model.model_id,
        "profile_probes": profile_probes,
        "entry_probes": probes,
        "records": records,
        "statistics": {
            "paired_ranges_mHa": [None] * len(experiment["seeds"]),
            "mean_range_mHa": None,
            "per_basis": {
                b: dict.fromkeys(("mean_mHa", "sample_std_mHa", "min_mHa", "max_mHa"))
                for b in experiment["orbital_bases"]
            },
            "null_reason": "no_numerical_orbital_results",
            "range_is_confidence_interval": False,
        },
        "oracle_status": {
            "nontrivial_orbital_bypass": "not_run_no_orbital_path_executed",
            "wrong_frame_amplitudes": "not_run_no_amplitudes_constructed",
        },
        "parent_peak_rss_bytes_at_report": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "rss_scope": "parent-only Linux HWM; complete tree monitoring is external",
        "next_step": "ARCH-SQD-002 input: resource-domain extension and real orbital/frame oracles; "
        "no change to scientific positioning justified",
    }
    _write(output / "report.json", report)
    _write(
        output / "hashes.json",
        {p.name: _hash(p) for p in (output / "manifest.json", output / "report.json")},
    )
    return verify_artifacts(output)


def verify_artifacts(output: Path) -> dict:
    """Verify frozen evidence bytes before returning a report to its consumer."""
    hashes = json.loads((output / "hashes.json").read_text())
    if set(hashes) != {"manifest.json", "report.json"}:
        raise ValueError("Incomplete hash manifest")
    for name, digest in hashes.items():
        if _hash(output / name) != digest:
            raise ValueError(f"Evidence hash mismatch: {name}")
    report = json.loads((output / "report.json").read_text())
    if report["manifest_sha256"] != hashes["manifest.json"]:
        raise ValueError("Registration hash mismatch")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_scan(args.output)
    print(f"{report['conclusion']}: {report['reason']}; {len(report['records'])} unmeasured points")


if __name__ == "__main__":
    main()
