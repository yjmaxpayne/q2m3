"""Compare water-dimer energies, reference tiers, and estimated quantum resources."""

from __future__ import annotations

from examples.fragments._common import calculation_options, execute, parser, save_input, save_result
from examples.fragments._inputs import WATER_SETTINGS, water_input


def calculate(args, directory, *, references=()):
    """Run the complete tutorial through the public fragment API."""
    from q2m3.fragments import run

    molecule = water_input(directory)
    options = calculation_options(args, *WATER_SETTINGS)
    options["references"] = references
    save_input(directory, molecule, options)
    result = run(molecule, **options)
    return save_result(directory, result)


def main(argv=None) -> int:
    """Run the full water example from the checkout root."""
    args = parser(__doc__, mode="full").parse_args(argv)
    return execute(args, "water-dimer-full", calculate)


if __name__ == "__main__":
    raise SystemExit(main())
