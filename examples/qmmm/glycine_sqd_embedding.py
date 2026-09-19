"""Compare vacuum, diagonal and full-one-electron SQD in a fixed glycine MO frame."""

from __future__ import annotations

from examples.sqd._inputs import fixed_waters, glycine_geometry
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
    """Separate environment-induced reference shifts from SQD approximation errors."""
    from q2m3.sqd import LUCJConfig, ReferenceConfig, run_sqd

    symbols, coordinates = glycine_geometry()
    charges, positions = fixed_waters()
    n = args.active_space
    metadata = describe_input(symbols, coordinates, n, args)
    metadata.update(
        mm_charges_e=charges.tolist(),
        mm_coordinates_angstrom=positions.tolist(),
        environment="two artificially placed fixed neutral TIP3P waters",
        limitations="no MM polarization, orbital relaxation, sampling or free energy",
    )
    write_json(directory / "input.json", metadata)
    rows = []
    frame_id = None
    results = {}
    for mode in ("vacuum", "diagonal", "full_oneelectron"):
        environment = (
            {}
            if mode == "vacuum"
            else dict(mm_charges=charges, mm_coords=positions, embedding_mode=mode)
        )
        result = run_sqd(
            symbols,
            coordinates,
            active_electrons=n,
            active_orbitals=n,
            lucj=LUCJConfig(n_reps=2, shots=args.shots),
            reference=ReferenceConfig(
                wall_budget_s=args.wall_budget, rss_budget_mb=args.rss_budget
            ),
            seed=args.seed,
            verbose=False,
            **environment,
        )
        payload = save_result(directory, mode, result)
        results[mode] = payload
        current = payload["provenance"]["context"]["frame_id"]
        if frame_id is not None and current != frame_id:
            raise ValueError("Embedding modes must share the same vacuum MO frame")
        frame_id = current
        print_result(payload, mode)
        row = summary_row(payload, mode, f"{mode}.json")
        vacuum = results["vacuum"]
        row["reference_shift_vs_vacuum_mHa"] = 1000 * (
            payload["baseline_energy"] - vacuum["baseline_energy"]
        )
        row["sqd_shift_vs_vacuum_mHa"] = 1000 * (payload["sqd_energy"] - vacuum["sqd_energy"])
        row["solver_error_change_mHa"] = payload["delta_mHa"] - vacuum["delta_mHa"]
        row["reference_shift_exact"] = payload["baseline_tier"] == vacuum["baseline_tier"] == "T0"
        rows.append(row)
        save_summary(directory, rows, plots=False)
    save_summary(directory, rows, plots=not args.no_plots)
    print("Fixed-environment energy shifts (mHa):")
    for row in rows:
        print(
            f"  {row['label']}: reference shift={row['reference_shift_vs_vacuum_mHa']:.6f}; "
            f"SQD shift={row['sqd_shift_vs_vacuum_mHa']:.6f}; "
            f"change in solver residual={row['solver_error_change_mHa']:.6f}"
        )
    print(
        "These are fixed point-charge energy differences, not solvation free energies. "
        "MO orbitals and the two-electron tensor stay fixed; the environment is not polarized. "
        "Reference shifts are physical Hamiltonian changes only to the quality of each reference."
    )


def main() -> int:
    """Run the fixed-environment glycine tutorial."""
    return execute(parser(__doc__).parse_args(), "glycine-embedding", calculate)


if __name__ == "__main__":
    raise SystemExit(main())
