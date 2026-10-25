"""Fixed tutorial inputs and structure output helpers."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np

from examples.fragments._common import write_json

DATA = Path(__file__).parent / "data"
WATER_ID = "water-dimer"
CRYSTAL_ID = "nitrophenol-methylpyridine"
WATER_SETTINGS = ("cc-pvdz", ("O 2p", "H 1s"), 0.7)
CRYSTAL_SETTINGS = ("sto-3g", ("0 O 2p", "4 N 2p"), 0.1)


def water_input(directory: Path):
    """Copy the S22 geometry and declare the two water fragments."""
    from q2m3.fragments import Molecule

    source = directory / "water-dimer.xyz"
    shutil.copyfile(DATA / source.name, source)
    return Molecule.from_xyz(source, fragments=[(0, 1, 2), (3, 4, 5)], name=WATER_ID)


def write_xyz(path: Path, molecule) -> None:
    """Save atom order and Angstrom coordinates at round-trip precision."""
    lines = [str(len(molecule.symbols)), f"{molecule.name}, Angstrom"]
    lines.extend(
        f"{symbol} " + " ".join(f"{value:.17g}" for value in xyz)
        for symbol, xyz in zip(molecule.symbols, molecule.coordinates_angstrom, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_structure(directory: Path, original, normalized) -> None:
    """Save fragment indices and measure each changed X-H bond."""
    from q2m3.fragments.structure import XH_BOND_TARGETS_ANGSTROM

    write_xyz(directory / "original.xyz", original)
    write_xyz(directory / "normalized.xyz", normalized)
    write_json(
        directory / "fragments.json",
        {
            "fragment_id": original.name,
            "index_base": 0,
            "fragments": dict(zip(("A", "B"), original.fragments, strict=True)),
            "charges": original.charges,
            "multiplicities": original.multiplicities,
            "original_geometry_sha256": original.geometry_sha256(),
            "normalized_geometry_sha256": normalized.geometry_sha256(),
            "xh_targets_angstrom": XH_BOND_TARGETS_ANGSTROM,
            "scope": "asymmetric unit only; no symmetry expansion or periodic images",
        },
    )
    before = np.asarray(original.coordinates_angstrom)
    after = np.asarray(normalized.coordinates_angstrom)
    with (directory / "xh_bonds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "h_index",
                "parent_index",
                "parent_element",
                "before_angstrom",
                "after_angstrom",
                "target_angstrom",
                "change_angstrom",
            ),
        )
        writer.writeheader()
        for h, symbol in enumerate(original.symbols):
            if symbol != "H":
                continue
            distances = np.linalg.norm(before - before[h], axis=1)
            distances[h] = np.inf
            parent = int(np.argmin(distances))
            element = original.symbols[parent]
            length = float(np.linalg.norm(after[h] - after[parent]))
            writer.writerow(
                dict(
                    h_index=h,
                    parent_index=parent,
                    parent_element=element,
                    before_angstrom=distances[parent],
                    after_angstrom=length,
                    target_angstrom=XH_BOND_TARGETS_ANGSTROM[element],
                    change_angstrom=length - distances[parent],
                )
            )
