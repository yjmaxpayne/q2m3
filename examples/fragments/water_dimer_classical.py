"""Define water fragments and calculate HF/MP2 counterpoise interaction energies."""

from __future__ import annotations

from examples.fragments._common import calculation_options, execute, parser, save_input, save_result
from examples.fragments._inputs import WATER_SETTINGS, water_input


def calculate(args, directory, *, references=()):
    """Calculate the tutorial geometry with the public fragment API."""
    from q2m3.fragments import run

    molecule = water_input(directory)
    options = calculation_options(args, *WATER_SETTINGS)
    options["references"] = references
    save_input(directory, molecule, options)
    result = run(molecule, **options)
    return save_result(directory, result)


def main(argv=None) -> int:
    """Run the classical water example from the checkout root."""
    args = parser(__doc__, mode="classical_only").parse_args(argv)
    return execute(args, "water-dimer-classical", calculate)


if __name__ == "__main__":
    raise SystemExit(main())
