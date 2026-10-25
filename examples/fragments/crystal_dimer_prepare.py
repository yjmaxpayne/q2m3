"""Read COD 2010678 and normalize X-H lengths in its asymmetric unit."""

from __future__ import annotations

import shutil

from examples.fragments._common import execute, parser, save_input
from examples.fragments._inputs import CRYSTAL_ID, DATA, save_structure


def calculate(args, directory):
    """Read and normalize the structure with the public structure API."""
    from q2m3.fragments.structure import normalize_xh, read_dimer_cif

    source = directory / "cod_2010678.cif"
    shutil.copyfile(DATA / source.name, source)
    original = read_dimer_cif(source, name=CRYSTAL_ID)
    save_input(directory, original, {"normalize_xh": True, "scope": "asymmetric unit"})
    normalized = normalize_xh(original)
    save_structure(directory, original, normalized)
    print("Saved original.xyz, normalized.xyz, fragments.json, and xh_bonds.csv", flush=True)
    return normalized


def main(argv=None) -> int:
    """Prepare the crystal dimer without an energy calculation."""
    args = parser(__doc__).parse_args(argv)
    return execute(args, "crystal-dimer-prepare", calculate)


if __name__ == "__main__":
    raise SystemExit(main())
