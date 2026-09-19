"""H2/STO-3G CAS(2e,2o): a minimal complete public SQD calculation."""

from __future__ import annotations

import numpy as np

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
    """Call the public geometry API and explain its full-space regression."""
    from q2m3.sqd import LUCJConfig, ReferenceConfig, run_sqd

    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    write_json(directory / "input.json", describe_input(symbols, coordinates, 2, args))
    result = run_sqd(
        symbols,
        coordinates,
        active_electrons=2,
        active_orbitals=2,
        lucj=LUCJConfig(n_reps=2, shots=args.shots),
        reference=ReferenceConfig(wall_budget_s=args.wall_budget, rss_budget_mb=args.rss_budget),
        seed=args.seed,
        verbose=False,
    )
    payload = save_result(directory, "result", result)
    print_result(payload, "H2")
    save_summary(directory, [summary_row(payload, "H2", "result.json")], plots=not args.no_plots)


def main() -> int:
    """Run from the checkout: python -m examples.sqd.h2_ground_state."""
    return execute(parser(__doc__, h2=True).parse_args(), "h2", calculate)


if __name__ == "__main__":
    raise SystemExit(main())
