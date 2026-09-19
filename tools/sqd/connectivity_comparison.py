"""Audit heavy-hex UCJ restrictions and retain unreachable original workloads.

This is a resource preflight report plus a separate real molecular operator
witness. It does not report SQD energy comparisons for rejected workloads.
Run with single-thread BLAS in the frozen SQD environment and a new --output.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_SOURCE_SHA256 = "872536739ce9855de640e3842a3e9e8a6db7b46ac40567587601c970006c9712"
SYSTEMS = [
    {
        "label": "H2/STO-3G R=0.74 (2e,2o)",
        "atom": "H 0 0 0; H 0 0 0.74",
        "basis": "sto-3g",
        "charge": 0,
        "n_reps": 2,
        "shots": 50000,
        "samples_per_batch": 100,
        "num_batches": 3,
        "max_iterations": 5,
        "target_mHa": 1.0,
        "name": "h2",
        "geometry_format": "cartesian",
        "active_space": [2, 2],
        "coordinate_unit": "Angstrom",
        "input_origin": "recovered_original_poc",
    },
    {
        "label": "H3O+/STO-3G (4e,4o)",
        "atom": "O 0.0 0.0 0.0; H 0.0 0.94 0.26; H 0.814 -0.47 0.26; H -0.814 -0.47 0.26",
        "basis": "sto-3g",
        "charge": 1,
        "n_reps": 2,
        "shots": 50000,
        "samples_per_batch": 300,
        "num_batches": 3,
        "max_iterations": 5,
        "target_mHa": 2.0,
        "name": "h3o",
        "geometry_format": "cartesian",
        "active_space": [4, 4],
        "coordinate_unit": "Angstrom",
        "input_origin": "recovered_original_poc",
    },
    {
        "label": "Glycine(NH2-CH2-COOH)/STO-3G (6e,6o)",
        "atom": "\nC\nC 1 1.52\nN 1 1.47 2 110.0\nO 2 1.21 1 125.0 3 180.0\nO 2 1.35 1 111.0 3 0.0\nH 1 1.09 2 108.0 3 120.0\nH 1 1.09 2 108.0 3 -120.0\nH 3 1.01 1 110.0 2 60.0\nH 3 1.01 1 110.0 2 -60.0\nH 5 0.97 2 107.0 1 0.0\n",
        "basis": "sto-3g",
        "charge": 0,
        "n_reps": 4,
        "shots": 200000,
        "samples_per_batch": 2000,
        "num_batches": 5,
        "max_iterations": 6,
        "target_mHa": 1.0,
        "name": "glycine",
        "geometry_format": "zmatrix",
        "active_space": [6, 6],
        "coordinate_unit": "Angstrom",
        "input_origin": "recovered_original_poc",
    },
    {
        "name": "n2",
        "atom": "N 0 0 0; N 0 0 1.1",
        "basis": "cc-pvdz",
        "charge": 0,
        "active_space": [10, 10],
        "n_reps": 4,
        "shots": 500000,
        "samples_per_batch": 10000,
        "num_batches": 2,
        "max_iterations": 2,
        "geometry_format": "cartesian",
        "coordinate_unit": "Angstrom",
        "input_origin": "registered_extension_1.1A_historical_reps_shots_current_diagonalizer_defaults",
    },
]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o444)


def connectivity_capability() -> dict:
    """Probe the installed generator and preserve its actual source and signature."""
    from ffsim.variational import util

    generator = getattr(util, "interaction_pairs_spin_balanced", None)
    result = {
        "version": version("ffsim"),
        "generator_available": callable(generator),
        "api": "ffsim.variational.util.interaction_pairs_spin_balanced",
        "module_sha256": _hash(Path(util.__file__)),
    }
    if generator is None:
        return result | {"reason": "generator_missing_in_installed_version", "pairs": None}
    return result | {
        "signature": str(inspect.signature(generator)),
        "source": inspect.getsource(generator),
        "pairs": {s["name"]: generator("heavy-hex", s["active_space"][1]) for s in SYSTEMS},
        "reason": None,
    }


def summarize_pairs(rows: list[dict]) -> dict:
    """Classify the registered three seeds; missing points never imply success.

    Args:
        rows: Exactly one row for each seed 0, 1, 2 with paired D in mHa.

    Returns:
        Mean, sample standard deviation, observed range and bounded conclusion.

    Raises:
        ValueError: Seed inventory or finite-value contract is violated.
    """
    if len(rows) != 3 or {r["seed"] for r in rows} != {0, 1, 2}:
        raise ValueError("Expected exactly seeds 0, 1, 2")
    values = [r["D_mHa"] for r in rows]
    if any(v is not None and (isinstance(v, bool) or not np.isfinite(v)) for v in values):
        raise ValueError("Nonfinite paired difference")
    if any(v is None for v in values):
        return dict(
            conclusion="inconclusive",
            mean_mHa=None,
            sample_std_mHa=None,
            min_mHa=None,
            max_mHa=None,
            null_reason="missing_paired_energy",
            range_is_confidence_interval=False,
        )
    low, high = min(values), max(values)
    conclusion = (
        "negative" if low > 5.0 else ("not_refuted_in_sample" if high <= 5.0 else "inconclusive")
    )
    return dict(
        conclusion=conclusion,
        mean_mHa=float(np.mean(values)),
        sample_std_mHa=float(np.std(values, ddof=1)),
        min_mHa=low,
        max_mHa=high,
        null_reason=None,
        range_is_confidence_interval=False,
    )


def profile_rejections(spec: dict) -> dict:
    """Exercise each frozen profile dimension; unrelated failures propagate."""
    from q2m3.sqd.exceptions import ResourceModelDomainError
    from q2m3.sqd.resources import load_resource_model

    load_resource_model()  # A dependency mismatch is not workload evidence.
    failures = {}
    for key in ("n_reps", "shots", "num_batches", "max_iterations"):
        try:
            load_resource_model(profile={key: spec[key]})
        except ResourceModelDomainError as exc:
            if f"uncalibrated workload {key}={spec[key]}; limit=" not in str(exc):
                raise
            failures[key] = {
                "exception": type(exc).__name__,
                "message": str(exc),
                "predicted_rss_mb": None,
                "prediction_reason": "outside_calibration_domain",
            }
    return failures


def operator_witness() -> tuple[dict, dict]:
    """Use real same-frame CCSD and production builders to audit both UCJ operators.

    Returns:
        Operator-only evidence and arrays for independent matrix/state inspection.

    Raises:
        ValueError: Restrictions are bypassed or the fixture is nondiscriminating.
    """
    import ffsim
    from ffsim.variational.util import interaction_pairs_spin_balanced
    from pyscf import gto

    from q2m3.molecule import MoleculeConfig
    from q2m3.sqd.ansatz import build_ccsd_seed, build_lucj
    from q2m3.sqd.config import LUCJConfig
    from q2m3.sqd.integrals import build_integrals

    spec = SYSTEMS[1]
    mol = gto.M(atom=spec["atom"], basis=spec["basis"], charge=spec["charge"], verbose=0)
    molecule = MoleculeConfig(
        "h3o",
        [mol.atom_symbol(i) for i in range(mol.natm)],
        mol.atom_coords(unit="Angstrom").tolist(),
        1,
        4,
        4,
    )
    budget = dict(host_available_mb=8192.0, rss_budget_mb=8192.0)
    data = build_integrals(molecule, **budget)
    seed = build_ccsd_seed(data, **budget)
    pairs = interaction_pairs_spin_balanced("heavy-hex", 4)
    if pairs is None or any(channel is None for channel in pairs):
        raise ValueError("Restricted connectivity must have concrete pairs")
    full = build_lucj(data, seed, lucj=LUCJConfig(n_reps=2, shots=50000), **budget)
    restricted = build_lucj(
        data, seed, lucj=LUCJConfig(n_reps=2, shots=50000, interaction_pairs=pairs), **budget
    )
    forbidden = []
    for channel, allowed in enumerate(pairs):
        mask = np.zeros((4, 4), dtype=bool)
        for i, j in allowed:
            mask[i, j] = mask[j, i] = True
        maximum = float(np.max(np.abs(restricted.diag_coulomb_mats[:, channel, ~mask])))
        if maximum != 0.0:
            raise ValueError("Restricted operator contains forbidden interactions")
        forbidden.append(maximum)
    hf = ffsim.hartree_fock_state(4, (2, 2))
    states = [ffsim.apply_unitary(hf.copy(), op, norb=4, nelec=(2, 2)) for op in (full, restricted)]
    if any(not np.all(np.isfinite(v)) or abs(np.vdot(v, v).real - 1) > 1e-10 for v in states):
        raise ValueError("Invalid prepared state")
    difference = float(np.max(np.abs(full.diag_coulomb_mats - restricted.diag_coulomb_mats)))
    infidelity = float(1 - abs(np.vdot(*states)) ** 2)
    if difference <= 1e-8 or infidelity <= 1e-8:
        raise ValueError("Nondiscriminating operator fixture")
    witness = dict(
        scope="h3o_operator_only_no_sqd_energy_comparison",
        full_pairs=None,
        restricted_pairs=pairs,
        n_reps=2,
        shots_config=50000,
        shots_executed=0,
        forbidden_max_abs=forbidden,
        matrix_max_difference=difference,
        state_infidelity=infidelity,
        frame_id=data.context.frame_id,
        hamiltonian_id=data.context.hamiltonian_id,
        ccsd_energy_ha=seed.ccsd_energy,
        hardware_layout_verified=False,
        topology_scope="logical_diagonal_coulomb_mask_not_backend_transpilation",
    )
    arrays = dict(
        full_mats=full.diag_coulomb_mats,
        restricted_mats=restricted.diag_coulomb_mats,
        full_state=states[0],
        restricted_state=states[1],
        t1=seed.t1,
        t2=seed.t2,
        full_rotations=full.orbital_rotations,
        restricted_rotations=restricted.orbital_rotations,
    )
    return witness, arrays


def run_comparison(output: Path) -> dict:
    """Freeze original workloads, report domain rejections and audit real restrictions.

    Args:
        output: New directory; existing evidence is never overwritten.

    Returns:
        Explicitly inconclusive molecular report plus a separate operator witness.

    Raises:
        RuntimeError: A workload becomes reachable and requires numerical comparison.
    """
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted((ROOT / "src/q2m3").rglob("*.py")) + [
        Path(__file__).resolve(),
        ROOT / "src/q2m3/sqd/resource_calibration.json",
        ROOT / "uv.lock",
    ]
    manifest = dict(
        schema="connectivity.preflight.v1",
        systems=SYSTEMS,
        seeds=[0, 1, 2],
        original_source_sha256=ORIGINAL_SOURCE_SHA256,
        original_seed=20260721,
        source_hashes={str(p.relative_to(ROOT)): _hash(p) for p in sources},
        versions={
            p: version(p)
            for p in ("numpy", "scipy", "pyscf", "ffsim", "qiskit", "qiskit-addon-sqd")
        },
        python=sys.version,
        platform=platform.platform(),
        command=sys.argv,
        threads={
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        budget=dict(external_wall_s=900, external_tree_rss_bytes=8192000000),
        threshold_mHa=5.0,
        difference_definition="D=delta_restricted-delta_full",
        delta_definition="1000*(E_sqd-E_same_H_reference)",
        witness_scope="h3o real CCSD operator construction only; no sampling or SQD",
    )
    _write(output / "manifest.json", manifest)
    capability = connectivity_capability()
    rejections = {s["name"]: profile_rejections(s) for s in SYSTEMS}
    if any(not rejection for rejection in rejections.values()):
        raise RuntimeError("Workload accepted: numerical paired experiment required")
    witness, arrays = (
        operator_witness()
        if capability["generator_available"]
        else ({"scope": "not_run_generator_missing", "reason": capability["reason"]}, {})
    )
    np.savez(output / "operator-witness.npz", **arrays)
    (output / "operator-witness.npz").chmod(0o444)
    records = [
        dict(
            system=s["name"],
            seed=seed,
            full_energy_ha=None,
            restricted_energy_ha=None,
            full_delta_mHa=None,
            restricted_delta_mHa=None,
            D_mHa=None,
            status="inconclusive",
            molecular_pair_executed=False,
            null_reason="original_workload_outside_calibration",
            full_requested_pairs=None,
            restricted_requested_pairs=(
                capability["pairs"][s["name"]] if capability["pairs"] else None
            ),
            rejection_evidence_system=s["name"],
            evidence_scope="shared_profile_probe_not_independent_seed_runs",
        )
        for s in SYSTEMS
        for seed in (0, 1, 2)
    ]
    report = dict(
        schema="connectivity.preflight.v1",
        manifest_sha256=_hash(output / "manifest.json"),
        capability=capability,
        profile_rejections=rejections,
        operator_witness=witness,
        records=records,
        statistics={
            s["name"]: summarize_pairs([r for r in records if r["system"] == s["name"]])
            for s in SYSTEMS
        },
        conclusion="inconclusive",
        hardware_ready=False,
        reason="original_workload_outside_calibration",
    )
    _write(output / "report.json", report)
    _write(
        output / "hashes.json",
        {p: _hash(output / p) for p in ("manifest.json", "report.json", "operator-witness.npz")},
    )
    return verify_artifacts(output)


def verify_artifacts(output: Path) -> dict:
    """Check every frozen artifact before consuming its report."""
    hashes = json.loads((output / "hashes.json").read_text())
    if set(hashes) != {"manifest.json", "report.json", "operator-witness.npz"}:
        raise ValueError("Incomplete hash manifest")
    for name, digest in hashes.items():
        if _hash(output / name) != digest:
            raise ValueError(f"Evidence hash mismatch: {name}")
    report = json.loads((output / "report.json").read_text())
    if report["manifest_sha256"] != hashes["manifest.json"]:
        raise ValueError("Registration hash mismatch")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_comparison(args.output)
    print(f"{report['conclusion']}: {report['reason']}; no hardware-readiness claim")


if __name__ == "__main__":
    main()
