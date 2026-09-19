"""Glycine/STO-3G: geometry, CCSD-seeded LUCJ sampling, SQD and classical controls."""

from __future__ import annotations

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
    """Run the selected active space through the public production engine."""
    from q2m3.sqd import LUCJConfig, ReferenceConfig, run_sqd

    symbols, coordinates = glycine_geometry()
    active = args.active_space
    write_json(directory / "input.json", describe_input(symbols, coordinates, active, args))
    result = run_sqd(
        symbols,
        coordinates,
        active_electrons=active,
        active_orbitals=active,
        lucj=LUCJConfig(n_reps=2, shots=args.shots),
        reference=ReferenceConfig(wall_budget_s=args.wall_budget, rss_budget_mb=args.rss_budget),
        seed=args.seed,
        verbose=False,
    )
    payload = save_result(directory, "result", result)
    print_result(payload, "Glycine")
    print(
        "Changing the active space changes the Hamiltonian and recovered correlation. "
        "Its total-energy change is separate from SQD's within-space solver error."
    )
    save_summary(
        directory, [summary_row(payload, "Glycine", "result.json")], plots=not args.no_plots
    )


def main() -> int:
    """Run from the checkout: python -m examples.sqd.glycine_ground_state."""
    return execute(parser(__doc__).parse_args(), "glycine", calculate)


if __name__ == "__main__":
    raise SystemExit(main())
