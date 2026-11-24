"""Static SQD benchmark in both MM embedding modes on registered TIP3P fixtures.

Each fixture runs ``run_sqd`` once with ``embedding_mode="diagonal"`` and once
with ``embedding_mode="full_oneelectron"``. The run writes ``report.json`` and
``summary.md`` to a new exclusive directory.

Run from the checkout root::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        uv run python -m examples.sqd.mm_tip3p_benchmark --output tmp/mm_tip3p_benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from examples.sqd._inputs import load_mm_tip3p_fixtures
from q2m3.constants import BANNED_TERMS
from q2m3.solvation.solvent import (
    TIP3P_WATER,
    check_mm_geometry,
    get_mm_embedding_data,
    state_array_to_molecules,
)

MODES = ("diagonal", "full_oneelectron")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
CURVE_POINTS = 256
SHIFT_CAVEAT = (
    "The full-vs-diagonal energy shift reaches 1 mHa or more only for near-contact "
    "configurations. A separate set of typical solvation-shell configurations gave "
    "median shifts of 0.31-0.37 mHa."
)
FULL_SPACE_NOTE = (
    "Points marked `full` sample the full CI space. "
    "In this domain SQD is equivalent to exact diagonalization."
)


@dataclass(frozen=True)
class BenchmarkInput:
    """QM inputs and MM point charges for one fixture (Angstrom, elementary charge)."""

    name: str
    symbols: list[str]
    coordinates: np.ndarray
    charge: int
    basis: str
    active_electrons: int
    active_orbitals: int
    mm_coords: np.ndarray
    mm_charges: np.ndarray


def benchmark_inputs(fixture: dict) -> BenchmarkInput:
    """Build TIP3P waters from the fixture state vectors and check their geometry.

    Raises:
        ValueError: The MM geometry check fails. The message names each violation.
    """
    molecules = state_array_to_molecules(TIP3P_WATER, np.asarray(fixture["water_states"]))
    qm_coords = np.asarray(fixture["qm_coords"], dtype=float)
    report = check_mm_geometry(molecules, qm_coords)
    if not report.passed:
        raise ValueError(
            f"fixture {fixture['name']} fails the MM geometry check: "
            f"{', '.join(report.violations)}"
        )
    mm_coords, mm_charges = get_mm_embedding_data(molecules)
    return BenchmarkInput(
        name=fixture["name"],
        symbols=list(fixture["qm_symbols"]),
        coordinates=qm_coords,
        charge=int(fixture["charge"]),
        basis=fixture["basis"],
        active_electrons=int(fixture["active_electrons"]),
        active_orbitals=int(fixture["active_orbitals"]),
        mm_coords=mm_coords,
        mm_charges=mm_charges,
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _log_spaced(curve: Sequence[Sequence[int]] | None) -> list[list[int]]:
    """Keep at most CURVE_POINTS log-spaced points, including the first and last."""
    points = [[int(shots), int(unique)] for shots, unique in curve or ()]
    if len(points) <= CURVE_POINTS:
        return points
    picks = np.unique(np.rint(np.geomspace(1, len(points), CURVE_POINTS)).astype(int) - 1)
    return [points[i] for i in picks]


def benchmark_point(inp: BenchmarkInput, mode: str, seed: int) -> dict:
    """Run SQD for one fixture in one embedding mode and return the point record."""
    from q2m3.sqd import run_sqd

    result = run_sqd(
        inp.symbols,
        inp.coordinates,
        active_electrons=inp.active_electrons,
        active_orbitals=inp.active_orbitals,
        charge=inp.charge,
        basis=inp.basis,
        mm_charges=inp.mm_charges,
        mm_coords=inp.mm_coords,
        embedding_mode=mode,
        seed=seed,
        verbose=False,
    )
    return {
        "fixture": inp.name,
        "mode": mode,
        "status": result.status,
        "sqd_energy": _optional_float(result.sqd_energy),
        "baseline_energy": float(result.baseline_energy),
        "baseline_tier": result.baseline_tier,
        "baseline_method": result.baseline_method,
        "delta_mHa": _optional_float(result.delta_mHa),
        "subspace_dim": result.subspace_dim,
        "full_ci_dim": int(result.full_ci_dim),
        "unique_dets_vs_shots": _log_spaced(result.unique_dets_vs_shots),
        "seed": int(result.seed),
    }


def _number(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _shift_rows(records: Sequence[dict]) -> list[str]:
    """Return one row per fixture with the exact (T0) energy shift, full minus diagonal.

    SQD energies are not used: a partial subspace adds sampling error to the shift.
    """
    exact = {
        (r["fixture"], r["mode"]): r["baseline_energy"]
        for r in records
        if r["baseline_tier"] == "T0"
    }
    rows = []
    for fixture in dict.fromkeys(r["fixture"] for r in records):
        diag = exact.get((fixture, "diagonal"))
        full = exact.get((fixture, "full_oneelectron"))
        if diag is not None and full is not None:
            rows.append(f"| {fixture} | {1000 * (full - diag):+.4f} |")
    return rows


def render_benchmark_summary(records: Sequence[dict]) -> str:
    """Render a Markdown summary of the point records.

    Raises:
        ValueError: The summary contains a banned term, in any letter case.
    """
    lines = [
        "# SQD static benchmark on TIP3P water fixtures",
        "",
        "Energies are in Hartree. `delta_mHa` is SQD minus the baseline, in mHa.",
        "",
        "| fixture | mode | status | SQD energy | baseline energy | tier | baseline method "
        "| delta_mHa | subspace / full CI |",
        "|---|---|---|---:|---:|---|---|---:|---|",
    ]
    for r in records:
        dims = f"{r['subspace_dim']} / {r['full_ci_dim']}"
        if r["subspace_dim"] == r["full_ci_dim"]:
            dims += " (full)"
        lines.append(
            f"| {r['fixture']} | {r['mode']} | {r['status']} | {_number(r['sqd_energy'], '.10f')} "
            f"| {r['baseline_energy']:.10f} | {r['baseline_tier']} | {r['baseline_method']} "
            f"| {_number(r['delta_mHa'], '+.2f')} | {dims} |"
        )
    shift_rows = _shift_rows(records)
    if shift_rows:
        lines += [
            "",
            "| fixture | exact (T0) energy shift, full minus diagonal / mHa |",
            "|---|---:|",
        ]
        lines += shift_rows
    lines += ["", SHIFT_CAVEAT]
    if any(r["subspace_dim"] == r["full_ci_dim"] for r in records):
        lines += ["", FULL_SPACE_NOTE]
    text = "\n".join(lines) + "\n"
    folded = text.casefold()
    found = [term for term in BANNED_TERMS if term.casefold() in folded]
    if found:
        raise ValueError(f"summary contains banned terms: {found}")
    return text


def write_outputs(records: Sequence[dict], out_dir: Path) -> None:
    """Write ``report.json`` and ``summary.md`` to ``out_dir``."""
    out_dir = Path(out_dir)
    summary = render_benchmark_summary(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = json.dumps({"records": list(records)}, indent=1, allow_nan=False)
    (out_dir / "report.json").write_text(report + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")


def _environment_problems() -> list[str]:
    problems = []
    if not (sys.platform.startswith("linux") and platform.machine() == "x86_64"):
        problems.append(f"SQD runs only on Linux x86_64, not {sys.platform}/{platform.machine()}")
    problems += [f"{name} must be 1" for name in THREAD_VARS if os.environ.get(name) != "1"]
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and return the process exit code."""
    fixtures = {fixture["name"]: fixture for fixture in load_mm_tip3p_fixtures()}
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--output", type=Path, help="New exclusive run directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixtures", nargs="+", choices=list(fixtures), default=list(fixtures))
    args = parser.parse_args(argv)

    problems = _environment_problems()
    if problems:
        print("Cannot run the benchmark:", *problems, sep="\n  ", file=sys.stderr)
        print(
            "Set the thread variables before Python starts:\n"
            "  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
            file=sys.stderr,
        )
        return 2

    from examples.sqd._presentation import output_directory

    try:
        out_dir = output_directory(args.output, "mm_tip3p_benchmark")
    except FileExistsError:
        parser.error(f"--output {args.output} already exists. Name a new directory.")

    records = []
    for name in args.fixtures:
        inp = benchmark_inputs(fixtures[name])
        for mode in MODES:
            record = benchmark_point(inp, mode, args.seed)
            records.append(record)
            print(f"{name} {mode}: {record['status']}, delta {record['delta_mHa']} mHa")
    write_outputs(records, out_dir)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"Wrote {out_dir / 'report.json'} and {out_dir / 'summary.md'}")
    return 0 if all(r["status"] == "completed" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
