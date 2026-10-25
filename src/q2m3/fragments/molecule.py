"""Frozen molecular input of the fragment interaction-energy workflow."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ELEMENTS = frozenset("""
    H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn
    Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce
    Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
    Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl
    Mc Lv Ts Og
    """.split())


@dataclass(frozen=True)
class Molecule:
    """Geometry in Angstrom, with an optional split into two fragments.

    ``charges`` and ``multiplicities`` hold one entry per fragment, or one entry for a
    single molecule (``fragments is None``).
    """

    symbols: tuple[str, ...]
    coordinates_angstrom: tuple[tuple[float, float, float], ...]
    fragments: tuple[tuple[int, ...], tuple[int, ...]] | None
    charges: tuple[int, ...]
    multiplicities: tuple[int, ...]
    name: str | None = None

    @classmethod
    def from_xyz(
        cls,
        source: str | os.PathLike[str],
        *,
        fragments: Sequence[Sequence[int]] | None = None,
        charges: Sequence[int] | None = None,
        multiplicities: Sequence[int] | None = None,
        name: str | None = None,
    ) -> Molecule:
        """Build a molecule from an XYZ file path or XYZ text in Angstrom.

        A string that contains no newline is read as a file path.

        Raises:
            ValueError: The XYZ text is malformed, an element is unknown, the fragments
                are not exactly two non-empty sets that partition the atoms, the number
                of charges or multiplicities does not match the fragments, or a
                multiplicity is not 1.
        """
        text = source if isinstance(source, str) and "\n" in source else Path(source).read_text()
        symbols, coordinates = _parse_xyz(text)
        parts = None if fragments is None else _partition(fragments, len(symbols))
        count = 1 if parts is None else 2
        charges = tuple(int(c) for c in charges) if charges is not None else (0,) * count
        multiplicities = (
            tuple(int(m) for m in multiplicities) if multiplicities is not None else (1,) * count
        )
        if len(charges) != count or len(multiplicities) != count:
            raise ValueError(f"expected {count} charges and multiplicities, one per fragment")
        if any(m != 1 for m in multiplicities):
            raise ValueError(f"only singlet multiplicities are supported, got {multiplicities}")
        return cls(symbols, coordinates, parts, charges, multiplicities, name)

    @property
    def charge(self) -> int:
        """Total charge of the molecule."""
        return sum(self.charges)

    def geometry_sha256(self) -> str:
        """Lowercase hex SHA-256 of the geometry and fragment definition.

        The digested bytes are compact ASCII JSON with sorted keys
        (``json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)``).
        The payload keys are ``symbols``, ``coordinates_angstrom`` (lists of Python
        ``float``), ``fragments`` (lists of atom indices, or ``null``), ``charges`` and
        ``multiplicities``. ``name`` is not digested.
        """
        payload = {
            "symbols": list(self.symbols),
            "coordinates_angstrom": [[float(c) for c in xyz] for xyz in self.coordinates_angstrom],
            "fragments": None if self.fragments is None else [list(f) for f in self.fragments],
            "charges": list(self.charges),
            "multiplicities": list(self.multiplicities),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _parse_xyz(text: str) -> tuple[tuple[str, ...], tuple[tuple[float, float, float], ...]]:
    lines = text.strip("\n").splitlines()
    try:
        count = int(lines[0])
    except (IndexError, ValueError):
        raise ValueError("first XYZ line must be the number of atoms") from None
    atom_lines = [line for line in lines[2:] if line.strip()]
    if len(atom_lines) != count:
        raise ValueError(f"XYZ declares {count} atoms but lists {len(atom_lines)} atom lines")
    symbols, coordinates = [], []
    for line in atom_lines:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"XYZ atom line needs a symbol and 3 coordinates: {line!r}")
        if fields[0] not in ELEMENTS:
            raise ValueError(f"unknown element symbol {fields[0]!r}")
        symbols.append(fields[0])
        x, y, z = (float(value) for value in fields[1:])
        coordinates.append((x, y, z))
    return tuple(symbols), tuple(coordinates)


def _partition(
    fragments: Sequence[Sequence[int]], n_atoms: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(fragments) != 2:
        raise ValueError(f"a dimer needs two fragments, got {len(fragments)}")
    parts = tuple(tuple(sorted(int(i) for i in fragment)) for fragment in fragments)
    flat = [i for part in parts for i in part]
    if not all(parts) or sorted(flat) != list(range(n_atoms)):
        raise ValueError(f"fragments {parts} do not partition atoms 0..{n_atoms - 1}")
    return parts[0], parts[1]
