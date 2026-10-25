"""Active-space selection: AVAS labels, the ghost-atom guard and the preflight.

AVAS does not search labels on the computational basis. It copies the molecule with the
MINAO basis and matches each label as a regular expression against the AO labels of that
copy, where ghost atoms appear as ``GHOST-<element>``.
"""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import gto
from pyscf.mcscf import avas

from q2m3.fragments.active_space import (
    MAX_NORB,
    minao_indices,
    preflight,
    resolve_labels,
    select_active_spaces,
)
from q2m3.fragments.counterpoise import Subsystem, counterpoise
from q2m3.fragments.errors import FragmentWorkflowError
from q2m3.fragments.molecule import Molecule
from q2m3.fragments.result import ActiveSpaceRecord
from tests.fragments.conftest import AVAS_SETTINGS, WATER_DIMER

ELEMENT_LABELS = ("O 2p", "H 1s")
# AVAS reference values of the same dimer from the prototype runs: (nelecas, ncas, ncore).
S6_ACTIVE_SPACES = {"AB": (12, 8, 4), "A": (6, 4, 2), "B": (6, 4, 2)}


def _minao_copy(mol: gto.Mole) -> gto.Mole:
    """The copy that pyscf.mcscf.avas builds before it searches the labels."""
    pmol = mol.copy()
    pmol.atom, pmol.unit, pmol.symmetry, pmol.basis = mol._atom, "B", False, "minao"
    pmol.build(False, False)
    return pmol


def _real_aos(subsystem: Subsystem, shells: set[tuple[str, str]]) -> list[int]:
    """MINAO AOs of real atoms whose (element, shell) is in ``shells``."""
    aos = _minao_copy(subsystem.mol).ao_labels(fmt=False)
    return [
        i
        for i, (atom, symbol, shell, _m) in enumerate(aos)
        if atom in subsystem.real_atoms and (symbol, shell) in shells
    ]


def _record(nelecas: int, ncas: int) -> ActiveSpaceRecord:
    return ActiveSpaceRecord(nelecas=nelecas, ncas=ncas, ncore=0, ao_labels=(), minao_indices=())


@pytest.fixture(scope="module")
def subsystems() -> dict[str, Subsystem]:
    xyz = "\n".join(["6", ""] + [f"{s} {x!r} {y!r} {z!r}" for s, (x, y, z) in WATER_DIMER])
    molecule = Molecule.from_xyz(xyz + "\n", fragments=((0, 1, 2), (3, 4, 5)))
    return counterpoise(molecule, basis="cc-pvdz")


def test_element_labels_select_only_real_atom_aos(subsystems):
    patterns = resolve_labels(subsystems, ELEMENT_LABELS)
    by_index = resolve_labels(subsystems, AVAS_SETTINGS["S6"][0])

    for name in ("A", "B"):
        sub = subsystems[name]
        labels = _minao_copy(sub.mol).ao_labels()
        selected = minao_indices(sub, patterns[name])

        assert not any("GHOST" in labels[i] for i in selected)
        assert not any("GHOST" in p for p in patterns[name])
        assert list(selected) == _real_aos(sub, {("O", "2p"), ("H", "1s")})
        # The element form and the atom-indexed form name the same shells here.
        assert selected == minao_indices(sub, by_index[name])


def test_raw_element_labels_given_to_avas_reach_ghost_aos(subsystems, monkeypatch):
    # Negative control: the element labels as plain AVAS substrings also match ghost AOs.
    searched: list[str] = []
    search = gto.Mole.search_ao_label

    def recording_search(mol, label):
        hits = search(mol, label)
        searched.extend(mol.ao_labels()[i].strip() for i in hits)
        return hits

    monkeypatch.setattr(gto.Mole, "search_ao_label", recording_search)
    avas.avas(subsystems["A"].scf, list(ELEMENT_LABELS), threshold=0.7, verbose=0)

    assert {"3 GHOST-O 2px", "3 GHOST-O 2py", "3 GHOST-O 2pz"} <= set(searched)


@pytest.mark.parametrize(
    "label",
    ["9 O 2p", "1 O 2p", "O 3d", "x O 2p", "O", "0 O 2p extra"],
    ids=["no-atom-9", "atom-1-is-hydrogen", "no-minao-3d", "bad-index", "one-field", "four"],
)
def test_label_without_a_matching_ao_is_rejected(subsystems, label):
    with pytest.raises(ValueError, match="AVAS label"):
        resolve_labels(subsystems, ["0 O 2p", label])


def test_label_only_on_fragment_a_is_rejected_before_avas(subsystems, monkeypatch):
    calls: list[object] = []
    monkeypatch.setattr(avas, "avas", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(FragmentWorkflowError, match="subsystem B"):
        select_active_spaces(subsystems, ["2 H 1s"], 0.2)
    assert calls == []


def test_monomer_without_active_virtual_orbital_fails_preflight(subsystems):
    labels, threshold = AVAS_SETTINGS["S1"]

    with pytest.raises(FragmentWorkflowError, match=r"subsystem B: \(6e,3o\).*virtual"):
        select_active_spaces(subsystems, labels, threshold)


@pytest.mark.parametrize(
    ("nelecas", "ncas", "rule"),
    [
        (0, 2, "no active occupied orbital"),
        (4, 2, "no active virtual orbital"),
        (3, 4, "odd number of active electrons"),
        (2, MAX_NORB + 1, f"more than {MAX_NORB} active orbitals"),
    ],
    ids=["no-occupied", "no-virtual", "odd-electrons", "norb-11"],
)
def test_preflight_rejects_each_rule_alone(nelecas, ncas, rule):
    records = {"AB": _record(4, 4), "A": _record(2, 2), "B": _record(nelecas, ncas)}

    with pytest.raises(FragmentWorkflowError, match=rf"subsystem B: \({nelecas}e,{ncas}o\)"):
        preflight(records)
    with pytest.raises(FragmentWorkflowError, match=rule):
        preflight(records)


def test_preflight_accepts_the_largest_allowed_active_space():
    assert MAX_NORB == 10
    preflight({"M": _record(2, MAX_NORB)})


def test_s6_active_spaces_match_the_reference_run(subsystems):
    spaces = select_active_spaces(subsystems, *AVAS_SETTINGS["S6"])

    counts = {name: (r.nelecas, r.ncas, r.ncore) for name, (r, _mo) in spaces.items()}
    assert counts == S6_ACTIVE_SPACES
    for name, (record, mo) in spaces.items():
        sub = subsystems[name]
        pmol = _minao_copy(sub.mol)
        assert all(type(v) is int for v in (record.nelecas, record.ncas, record.ncore))
        assert list(record.minao_indices) == _real_aos(sub, {("O", "2p"), ("H", "1s")})
        assert [list(pmol.search_ao_label([p])) for p in record.ao_labels] == [
            [i] for i in record.minao_indices
        ]
        # AVAS ran with canonicalize=True: the Fock matrix is diagonal in the active occupied
        # block up to the SCF convergence (off-diagonal <= 1.2e-7 Ha here, >= 3.5e-3 Ha
        # without canonicalization).
        occupied = slice(record.ncore, record.ncore + record.nelecas // 2)
        block = (mo.T @ sub.scf.get_fock() @ mo)[occupied, occupied]
        assert np.max(np.abs(block - np.diag(np.diag(block)))) < 1e-5


def test_anchored_pattern_does_not_match_a_longer_atom_index():
    chain = gto.M(atom=[("H", (0.0, 0.0, 0.74 * i)) for i in range(12)], basis="sto-3g")
    sub = Subsystem("M", chain, None, tuple(range(12)), ())

    patterns = resolve_labels({"M": sub}, ["1 H 1s"])["M"]

    assert [list(_minao_copy(chain).search_ao_label([p])) for p in patterns] == [[1]]


def test_ghost_guard_rejects_a_pattern_on_a_ghost_atom(subsystems):
    with pytest.raises(FragmentWorkflowError, match="ghost"):
        minao_indices(subsystems["A"], ("^0 O 2px\\s*$", "^3 GHOST-O 2px\\s*$"))
