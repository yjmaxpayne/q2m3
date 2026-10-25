"""Calculate the normalized crystal dimer with the teaching STO-3G active space."""

from __future__ import annotations

from examples.fragments._common import calculation_options, execute, parser, save_input, save_result
from examples.fragments._inputs import CRYSTAL_SETTINGS
from examples.fragments.crystal_dimer_prepare import calculate as prepare


def calculate(args, directory, *, references=()):
    """Connect the normalized geometry to the public interaction-energy API."""
    from q2m3.fragments import run

    molecule = prepare(args, directory)
    options = calculation_options(args, *CRYSTAL_SETTINGS)
    options["references"] = references
    save_input(directory, molecule, options)
    result = run(molecule, **options)
    return save_result(directory, result)


def main(argv=None) -> int:
    """Run the crystal calculation, with optional full mode."""
    args = parser(__doc__, mode="classical_only", choose_mode=True).parse_args(argv)
    return execute(args, "crystal-dimer-interaction", calculate)


if __name__ == "__main__":
    raise SystemExit(main())
