"""Experimental dimer input from a CIF, and X-H bond-length normalization."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Final

import numpy as np
from scipy.sparse.csgraph import connected_components

from q2m3.fragments.molecule import Molecule

# Standard neutron-diffraction X-H bond lengths (Angstrom), keyed by the parent element.
# Source: Allen, Kennard, Watson, Brammer, Orpen & Taylor, J. Chem. Soc., Perkin Trans. 2,
# 1987, S1-S19, doi:10.1039/p298700000s1. The values are transcribed from Bergedahl et al.,
# Int. J. Mol. Sci. 2025, 26, 1671, doi:10.3390/ijms26041671 (CC BY 4.0), Methods; the
# 1987 tables were not read.
XH_BOND_TARGETS_ANGSTROM: Final[Mapping[str, float]] = MappingProxyType(
    {"C": 1.083, "N": 1.009, "O": 0.983}
)


def read_dimer_cif(source: str | os.PathLike[str], *, name: str | None = None) -> Molecule:
    """Read the asymmetric unit of a CIF as a neutral singlet dimer.

    The atoms of the first data block with a structure are taken as listed: no symmetry
    expansion and no periodic images. The two fragments are the two covalently bonded
    molecules (ASE covalent-radius cutoffs). Fragment A holds the first listed atom.

    Raises:
        ImportError: ASE (the optional ``viz`` extra) is not installed.
        ValueError: The CIF has no structure, or its asymmetric unit is not exactly two
            covalently bonded molecules. For example, a molecule is split across the cell
            boundary, or the dimer needs a symmetry operation.
    """
    try:
        from ase.io.cif import parse_cif
        from ase.neighborlist import build_neighbor_list, natural_cutoffs
    except ImportError:
        raise ImportError(
            "read_dimer_cif needs ASE, which is not installed. "
            "Install it with: uv sync --extra viz"
        ) from None
    block = next((b for b in parse_cif(os.fspath(source)) if b.has_structure()), None)
    if block is None:
        raise ValueError(f"{os.fspath(source)} has no data block with a structure")
    atoms = block.get_unsymmetrized_structure()
    # ASE 3.26 already returns pbc=False here. Set it so that no version bonds periodic images.
    atoms.pbc = False
    bonds = build_neighbor_list(atoms, cutoffs=natural_cutoffs(atoms))
    count, labels = connected_components(bonds.get_connectivity_matrix(), directed=False)
    if count != 2:
        raise ValueError(
            f"the asymmetric unit holds {count} covalently bonded molecules, expected 2"
        )
    in_a = labels == labels[0]
    fragments = (
        tuple(int(i) for i in np.flatnonzero(in_a)),
        tuple(int(i) for i in np.flatnonzero(~in_a)),
    )
    coordinates = tuple(tuple(float(c) for c in xyz) for xyz in atoms.positions)
    return Molecule(
        tuple(atoms.get_chemical_symbols()), coordinates, fragments, (0, 0), (1, 1), name
    )


def normalize_xh(molecule: Molecule) -> Molecule:
    """Move each H along its X-H bond to the target length in ``XH_BOND_TARGETS_ANGSTROM``.

    The parent X of an H is the nearest other atom. Only H atoms move. The result keeps the
    fragments, charges, multiplicities and name.

    Known limit: normalization corrects the short X-H bonds of X-ray H positions. It does
    not correct the polarization of X-H bonds in hydrogen bonds, so a normalized H is not
    the true H position in a hydrogen bond.

    Raises:
        ValueError: The parent of an H has no target length.
    """
    xyz = np.array(molecule.coordinates_angstrom, dtype=float)
    normalized = xyz.copy()
    for h, symbol in enumerate(molecule.symbols):
        if symbol != "H":
            continue
        distances = np.linalg.norm(xyz - xyz[h], axis=1)
        distances[h] = np.inf
        parent = int(np.argmin(distances))
        element = molecule.symbols[parent]
        if element not in XH_BOND_TARGETS_ANGSTROM:
            raise ValueError(f"no X-H target for {element} (atom {parent}, parent of H atom {h})")
        bond = (xyz[h] - xyz[parent]) / distances[parent]
        normalized[h] = xyz[parent] + XH_BOND_TARGETS_ANGSTROM[element] * bond
    coordinates = tuple(tuple(float(c) for c in row) for row in normalized)
    return replace(molecule, coordinates_angstrom=coordinates)
