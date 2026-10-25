"""Dimer input from a CIF and X-H bond-length normalization.

The CIF is COD entry 2010678, the 1:1 complex of 4-nitrophenol and 4-methylpyridine
(Jin, Pan, Xu & Xu, Acta Cryst. C, 2000, doi:10.1107/S0108270100001074). Its asymmetric
unit holds one molecule of each component, linked by an O-H...N hydrogen bond. The H
positions come from X-ray data, so every X-H bond is shorter than the neutron target.
Trailing spaces on two COD comment lines are removed; no data value is changed.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from q2m3.fragments.molecule import Molecule
from q2m3.fragments.structure import XH_BOND_TARGETS_ANGSTROM, normalize_xh, read_dimer_cif

CIF_PATH = Path(__file__).parent / "data" / "cod_2010678.cif"
TOLERANCE_ANGSTROM = 1e-3

requires_ase = pytest.mark.skipif(
    importlib.util.find_spec("ase") is None, reason="ASE is not installed (uv sync --extra viz)"
)


@pytest.fixture(scope="module")
def dimer() -> Molecule:
    return read_dimer_cif(CIF_PATH, name="nitrophenol-methylpyridine")


def _xh_bonds(molecule: Molecule) -> list[tuple[int, int, float]]:
    """(H index, parent index, length) for each H, the parent being the nearest other atom."""
    xyz = np.array(molecule.coordinates_angstrom)
    bonds = []
    for h, symbol in enumerate(molecule.symbols):
        if symbol == "H":
            distances = np.linalg.norm(xyz - xyz[h], axis=1)
            distances[h] = np.inf
            parent = int(np.argmin(distances))
            bonds.append((h, parent, float(distances[parent])))
    return bonds


def _formula(molecule: Molecule, fragment: tuple[int, ...]) -> Counter[str]:
    return Counter(molecule.symbols[i] for i in fragment)


def test_targets_are_the_cited_neutron_values():
    assert XH_BOND_TARGETS_ANGSTROM == {"C": 1.083, "N": 1.009, "O": 0.983}


@requires_ase
def test_asymmetric_unit_is_split_into_the_two_molecules(dimer):
    assert len(dimer.symbols) == 29
    a, b = dimer.fragments
    # The hydroxyl H stays with its phenol; the hydrogen bond does not join the fragments.
    assert _formula(dimer, a) == Counter(C=6, H=5, N=1, O=3)
    assert _formula(dimer, b) == Counter(C=6, H=7, N=1)
    assert dimer.charges == (0, 0)
    assert dimer.multiplicities == (1, 1)
    assert dimer.name == "nitrophenol-methylpyridine"


@requires_ase
def test_xray_hydrogen_positions_miss_the_targets(dimer):
    bonds = _xh_bonds(dimer)
    assert len(bonds) == 12
    for _, parent, length in bonds:
        assert abs(length - XH_BOND_TARGETS_ANGSTROM[dimer.symbols[parent]]) > TOLERANCE_ANGSTROM


@requires_ase
def test_normalized_xh_bonds_match_the_targets(dimer):
    normalized = normalize_xh(dimer)
    bonds = _xh_bonds(normalized)
    assert len(bonds) == 12
    for _, parent, length in bonds:
        target = XH_BOND_TARGETS_ANGSTROM[normalized.symbols[parent]]
        assert abs(length - target) <= TOLERANCE_ANGSTROM


@requires_ase
def test_normalization_moves_hydrogens_only_along_their_bonds(dimer):
    normalized = normalize_xh(dimer)
    before = np.array(dimer.coordinates_angstrom)
    after = np.array(normalized.coordinates_angstrom)
    heavy = [i for i, symbol in enumerate(dimer.symbols) if symbol != "H"]
    np.testing.assert_array_equal(after[heavy], before[heavy])
    for h, parent, length in _xh_bonds(dimer):
        np.testing.assert_allclose(
            (after[h] - after[parent]) / np.linalg.norm(after[h] - after[parent]),
            (before[h] - before[parent]) / length,
            atol=1e-12,
        )
    assert normalized.symbols == dimer.symbols
    assert normalized.fragments == dimer.fragments
    assert (normalized.charges, normalized.multiplicities, normalized.name) == (
        dimer.charges,
        dimer.multiplicities,
        dimer.name,
    )


@requires_ase
def test_molecule_split_across_the_cell_boundary_is_rejected(tmp_path):
    # Move the hydroxyl H by one lattice vector: it is bonded only through a periodic image.
    text = CIF_PATH.read_text()
    assert "\nH .020(4) " in text
    shifted = tmp_path / "shifted.cif"
    shifted.write_text(text.replace("\nH .020(4) ", "\nH 1.020(4) "))
    with pytest.raises(ValueError, match="3 covalently bonded molecules"):
        read_dimer_cif(shifted)


@requires_ase
def test_cif_without_a_structure_is_rejected(tmp_path):
    empty = tmp_path / "empty.cif"
    empty.write_text("data_empty\n_chemical_name_common 'no atoms'\n")
    with pytest.raises(ValueError, match="no data block with a structure"):
        read_dimer_cif(empty)


def test_hydrogen_on_an_atom_without_a_target_is_rejected():
    hydrogen_sulfide = Molecule.from_xyz("3\n\nS 0.0 0.0 0.0\nH 1.34 0.0 0.0\nH 0.0 1.34 0.0\n")
    with pytest.raises(ValueError, match="no X-H target for S"):
        normalize_xh(hydrogen_sulfide)


def test_missing_ase_raises_with_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "ase.io.cif", None)
    with pytest.raises(ImportError, match="uv sync --extra viz"):
        read_dimer_cif(CIF_PATH)
