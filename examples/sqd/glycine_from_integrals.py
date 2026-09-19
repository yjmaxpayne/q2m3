"""Build authenticated glycine integrals/CCSD seed and compare both public SQD entries.

This is an adapter consistency check: both entries share the SQD solver. The
engine budget starts at its API call; preprocessing is timed separately. Use
``tools.sqd.verify_showcase`` to supervise imports, integral/seed assembly, both
calls, plotting and serialization with an external whole-process budget.
"""

from __future__ import annotations

import math
from pathlib import Path
from time import monotonic

from examples.sqd._inputs import glycine_geometry
from examples.sqd._presentation import (
    describe_input,
    execute,
    parser,
    print_result,
    save_result,
    save_summary,
    summary_row,
    write_json,
)


def calculate(args, directory) -> None:
    """Assemble chemist integrals once, then check the geometry adapter numerically."""
    from q2m3.molecule import MoleculeConfig
    from q2m3.sqd import LUCJConfig, ReferenceConfig, run_sqd, run_sqd_from_integrals
    from q2m3.sqd.ansatz import build_ccsd_seed
    from q2m3.sqd.integrals import build_integrals

    symbols, coordinates = glycine_geometry()
    n = args.active_space
    write_json(directory / "input.json", describe_input(symbols, coordinates, n, args))
    available_mb = next(
        int(line.split()[1]) * 1024 / 1e6
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    cap = min(available_mb, args.rss_budget, 8192.0)
    started = monotonic()
    molecule = MoleculeConfig("glycine", symbols, coordinates, 0, n, n, "sto-3g")
    data = build_integrals(molecule, host_available_mb=cap, rss_budget_mb=args.rss_budget)
    integral_s = monotonic() - started
    started = monotonic()
    seed_data = build_ccsd_seed(data, host_available_mb=cap, rss_budget_mb=args.rss_budget)
    write_json(
        directory / "assembly.json",
        dict(
            integrals_s=integral_s,
            ccsd_s=monotonic() - started,
            h1_shape=list(data.h1.shape),
            h2_shape=list(data.h2.shape),
            convention="real chemist (pq|rs), no transpose; e_core added once",
            e_core_ha=data.e_core,
            active_indices=list(data.context.active_indices),
            frame_id=data.context.frame_id,
            hamiltonian_id=data.context.hamiltonian_id,
            resource_scope="preprocessing outside engine measurement; use external supervisor",
        ),
    )
    lucj = LUCJConfig(n_reps=2, shots=args.shots)
    reference = ReferenceConfig(wall_budget_s=args.wall_budget, rss_budget_mb=args.rss_budget)
    integral_result = run_sqd_from_integrals(
        data.h1,
        data.h2,
        data.e_core,
        norb=data.norb,
        nelec=data.nelec,
        context=data.context,
        seed_data=seed_data,
        lucj=lucj,
        reference=reference,
        seed=args.seed,
    )
    low = save_result(directory, "integrals", integral_result)
    print_result(low, "Integral entry")
    high_result = run_sqd(
        symbols,
        coordinates,
        active_electrons=n,
        active_orbitals=n,
        lucj=lucj,
        reference=reference,
        seed=args.seed,
        verbose=False,
    )
    high = save_result(directory, "geometry", high_result)
    rows = [
        summary_row(low, "integrals", "integrals.json"),
        summary_row(high, "geometry", "geometry.json"),
    ]
    save_summary(directory, rows, plots=not args.no_plots)
    energies = (
        "sqd_energy",
        "baseline_energy",
        "hf_energy",
        "iso_active_space_ccsd_energy",
        "iso_ndet_sci_energy",
        "iso_ndet_random_energy",
    )
    matched = all(
        (
            low[key] == high[key]
            if low[key] is None or high[key] is None
            else math.isclose(low[key], high[key], rel_tol=0, abs_tol=1e-8)
        )
        for key in energies
    ) and all(
        low["provenance"]["context"][key] == high["provenance"]["context"][key]
        for key in ("frame_id", "hamiltonian_id", "active_indices")
    )
    write_json(
        directory / "consistency.json",
        dict(
            matched=matched,
            tolerance_ha=1e-8,
            differences_ha={
                key: None if low[key] is None or high[key] is None else low[key] - high[key]
                for key in energies
            },
            scope="adapter consistency; shared solver is not an independent oracle",
        ),
    )
    if not matched:
        raise ValueError("Integral and geometry adapters disagree; inspect preserved results")
    print(
        "Integral/geometry adapters agree within 1e-8 Ha in the same frame. "
        "This checks assembly consistency; the solver is shared."
    )


def main() -> int:
    """Run the explicit-integrals tutorial."""
    return execute(parser(__doc__).parse_args(), "glycine-integrals", calculate)


if __name__ == "__main__":
    raise SystemExit(main())
