"""Input validation and geometry digest of the fragment-workflow ``Molecule``."""

from __future__ import annotations

import hashlib
import json

import pytest

from q2m3.fragments.molecule import Molecule

DIMER_FRAGMENTS = ((0, 1, 2), (3, 4, 5))


def _xyz(atoms, comment: str = "water dimer") -> str:
    lines = [str(len(atoms)), comment]
    lines += [f"{symbol} {x!r} {y!r} {z!r}" for symbol, (x, y, z) in atoms]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    "fragments",
    [
        ((0, 1, 2), (2, 3, 4, 5)),  # atom 2 in both fragments
        ((0, 1), (3, 4, 5)),  # atom 2 in neither fragment
        ((0, 1, 2), (3, 4, 5, 6)),  # atom 6 does not exist
        ((0, 1, 2, 3, 4, 5), ()),  # empty fragment
    ],
)
def test_fragments_must_partition_the_atoms(water_dimer, fragments):
    with pytest.raises(ValueError, match="partition"):
        Molecule.from_xyz(_xyz(water_dimer), fragments=fragments)


def test_dimer_needs_exactly_two_fragments(water_dimer):
    with pytest.raises(ValueError, match="two fragments"):
        Molecule.from_xyz(_xyz(water_dimer), fragments=((0, 1), (2, 3), (4, 5)))


def test_only_singlet_fragments_are_supported(water_dimer):
    with pytest.raises(ValueError, match="multiplicit"):
        Molecule.from_xyz(_xyz(water_dimer), fragments=DIMER_FRAGMENTS, multiplicities=(1, 3))


def test_unknown_element_is_rejected(water_dimer):
    atoms = (("Xx", water_dimer[0][1]), *water_dimer[1:])
    with pytest.raises(ValueError, match="Xx"):
        Molecule.from_xyz(_xyz(atoms), fragments=DIMER_FRAGMENTS)


def test_atom_count_line_must_match_the_atom_lines(water_dimer):
    text = _xyz(water_dimer).replace("6\n", "5\n", 1)
    with pytest.raises(ValueError, match="atom"):
        Molecule.from_xyz(text, fragments=DIMER_FRAGMENTS)


def test_file_and_string_inputs_give_equal_molecules(water_dimer, tmp_path):
    path = tmp_path / "dimer.xyz"
    path.write_text(_xyz(water_dimer))
    kwargs = {"fragments": DIMER_FRAGMENTS, "charges": (0, 0), "name": "water dimer"}

    from_path = Molecule.from_xyz(path, **kwargs)
    from_text = Molecule.from_xyz(_xyz(water_dimer), **kwargs)

    assert from_path == from_text
    assert Molecule.from_xyz(str(path), **kwargs) == from_text
    assert from_text.symbols == tuple(symbol for symbol, _ in water_dimer)
    assert from_text.coordinates_angstrom == tuple(xyz for _, xyz in water_dimer)


def test_fragment_index_order_does_not_change_the_molecule(water_dimer):
    shuffled = Molecule.from_xyz(_xyz(water_dimer), fragments=((2, 0, 1), (5, 3, 4)))
    ordered = Molecule.from_xyz(_xyz(water_dimer), fragments=DIMER_FRAGMENTS)

    assert shuffled == ordered
    assert shuffled.geometry_sha256() == ordered.geometry_sha256()


def test_geometry_digest_matches_the_documented_byte_format(water_dimer):
    molecule = Molecule.from_xyz(_xyz(water_dimer), fragments=DIMER_FRAGMENTS, charges=(1, -1))
    payload = {
        "symbols": [symbol for symbol, _ in water_dimer],
        "coordinates_angstrom": [[float(c) for c in xyz] for _, xyz in water_dimer],
        "fragments": [list(fragment) for fragment in DIMER_FRAGMENTS],
        "charges": [1, -1],
        "multiplicities": [1, 1],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    assert molecule.geometry_sha256() == hashlib.sha256(encoded.encode("ascii")).hexdigest()


def test_geometry_digest_changes_with_the_last_coordinate_digit(water_dimer):
    symbol, (x, y, z) = water_dimer[0]
    moved = ((symbol, (x, y, z + 1e-6)), *water_dimer[1:])

    original = Molecule.from_xyz(_xyz(water_dimer), fragments=DIMER_FRAGMENTS)
    shifted = Molecule.from_xyz(_xyz(moved), fragments=DIMER_FRAGMENTS)

    assert original.geometry_sha256() != shifted.geometry_sha256()


def test_single_molecule_has_no_fragments(water_dimer):
    monomer = water_dimer[:3]
    molecule = Molecule.from_xyz(_xyz(monomer), charges=(0,))

    assert molecule.fragments is None
    assert molecule.charges == (0,)
    assert molecule.multiplicities == (1,)
