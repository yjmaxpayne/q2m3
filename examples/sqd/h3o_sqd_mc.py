"""Monte Carlo solvation of H3O+ in 8 TIP3P waters with SQD energies of the QM region.

H3O+ uses a (4e, 4o) active space in STO-3G at 300 K. Every Monte Carlo step
calls ``run_sqd`` with the MM point charges in ``full_oneelectron`` mode. The
acceptance energy is ``E_sqd + E_mm(sol-sol) + E_LJ(QM-MM)``. The run writes
``summary.md`` and ``result.json`` to a new exclusive directory. The default
500 steps take about 20 min on one core.

Run from the checkout root::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        uv run python -m examples.sqd.h3o_sqd_mc --output tmp/h3o_sqd_mc
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from q2m3.molecule import MoleculeConfig

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
H3O_COORDS_A = [
    [0.0000, 0.0000, 0.1173],
    [0.0000, 0.9572, -0.4692],
    [0.8286, -0.4786, -0.4692],
    [-0.8286, -0.4786, -0.4692],
]


def h3o_molecule() -> MoleculeConfig:
    """Return H3O+ with a (4e, 4o) active space in STO-3G."""
    return MoleculeConfig(
        name="H3O+",
        symbols=["O", "H", "H", "H"],
        coords=H3O_COORDS_A,
        charge=1,
        active_electrons=4,
        active_orbitals=4,
        basis="sto-3g",
    )


def _initial_min_qm_mm_A(config) -> float:
    """Nearest QM-MM atom distance of the initial water ring, in Angstrom."""
    from q2m3.solvation.solvent import TIP3P_WATER, get_mm_embedding_data, initialize_solvent_ring

    qm = np.asarray(config.molecule.coords_array, dtype=np.float64)
    ring = initialize_solvent_ring(
        TIP3P_WATER,
        config.n_waters,
        qm.mean(axis=0),
        config.initial_water_distance,
        config.random_seed,
    )
    mm, _ = get_mm_embedding_data(ring)
    return float(np.linalg.norm(qm[:, None, :] - mm[None, :, :], axis=-1).min())


def result_payload(result, config, initial_min_qm_mm_A: float, wall_s: float) -> dict:
    """Collect the JSON record of one run."""
    steps = result.mc.step_results
    sqd = [result.initial_step.sqd_result, *result.sqd_results]
    hf_ref = config.acceptance_uses_hf_reference
    return {
        "n_mc_steps": config.n_mc_steps,
        "random_seed": config.random_seed,
        "temperature_K": config.temperature,
        "acceptance_rate": float(result.mc.acceptance_rate),
        "accepted": [bool(a) for a in result.accepted],
        "initial_min_qm_mm_A": initial_min_qm_mm_A,
        "min_qm_mm_A": [float(d) for d in result.min_qm_mm_A],
        "domain": result.provenance["domain"],
        "provenance": dict(result.provenance),
        "initial_acceptance_energy_ha": float(result.initial_step.acceptance_energy(hf_ref)),
        "trial_acceptance_energy_ha": [float(s.acceptance_energy(hf_ref)) for s in steps],
        "trial_sqd_energy_ha": [float(s.e_qpe) for s in steps],
        "trial_mm_sol_sol_ha": [float(s.e_mm_sol_sol) for s in steps],
        "trial_qm_mm_lj_ha": [float(s.e_qmmm_vdw) for s in steps],
        "status": [r.status for r in sqd],
        "subspace_dim": [int(r.subspace_dim) for r in sqd],
        "full_ci_dim": [int(r.full_ci_dim) for r in sqd],
        "peak_rss_mb": [float(r.diagnostics["resources"]["peak_rss_mb"]) for r in sqd],
        "timings_s": {
            "wall": wall_s,
            "step_callback": [float(s.callback_time) for s in steps],
            "sqd_solver": [float(s.qpe_time) for s in steps],
        },
    }


def _environment_problems() -> list[str]:
    problems = []
    if not (sys.platform.startswith("linux") and platform.machine() == "x86_64"):
        problems.append(f"SQD runs only on Linux x86_64, not {sys.platform}/{platform.machine()}")
    problems += [f"{name} must be 1" for name in THREAD_VARS if os.environ.get(name) != "1"]
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Monte Carlo solvation and return the process exit code."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--steps", type=int, default=500, help="Monte Carlo steps")
    parser.add_argument("--seed", type=int, default=42, help="Monte Carlo random seed")
    parser.add_argument("--output", type=Path, help="New exclusive run directory")
    args = parser.parse_args(argv)

    problems = _environment_problems()
    if problems:
        print("Cannot run the Monte Carlo solvation:", *problems, sep="\n  ", file=sys.stderr)
        print(
            "Set the thread variables before Python starts:\n"
            "  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
            file=sys.stderr,
        )
        return 2

    from examples.sqd._presentation import output_directory, write_json
    from q2m3.solvation.sqd_mc import (
        SQDSolvationConfig,
        render_sqd_mc_summary,
        run_sqd_solvation,
    )

    try:
        out_dir = output_directory(args.output, "h3o_sqd_mc")
    except FileExistsError:
        parser.error(f"--output {args.output} already exists. Name a new directory.")

    config = SQDSolvationConfig(
        molecule=h3o_molecule(),
        n_waters=8,
        n_mc_steps=args.steps,
        temperature=300.0,
        random_seed=args.seed,
        initial_water_distance=3.5,
        embedding_mode="full_oneelectron",
        sqd_seed=0,
    )
    initial_min = _initial_min_qm_mm_A(config)
    start = time.perf_counter()
    result = run_sqd_solvation(config)
    wall_s = time.perf_counter() - start

    summary = render_sqd_mc_summary(result)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    write_json(out_dir / "result.json", result_payload(result, config, initial_min, wall_s))
    print(summary)
    print(f"Wall time {wall_s:.1f} s. Wrote {out_dir / 'summary.md'} and {out_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
