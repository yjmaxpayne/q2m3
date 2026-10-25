"""Correspondence of the dimer and monomer active spaces.

The active orbitals of monomers A and B, joined and orthonormalised in the shared AO metric,
must span the dimer active space block by block: occupied with occupied, virtual with
virtual. The cosines of the principal angles between the two spaces measure the match.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from pyscf.mcscf import avas

from q2m3.fragments import workflow
from q2m3.fragments.active_space import (
    COS_THRESHOLD,
    check_correspondence,
    resolve_labels,
    select_active_spaces,
)
from q2m3.fragments.counterpoise import Subsystem, counterpoise
from q2m3.fragments.errors import ActiveSpaceMismatchError
from q2m3.fragments.molecule import Molecule
from q2m3.fragments.result import ActiveSpaceRecord
from tests.fragments.conftest import AVAS_SETTINGS, WATER_DIMER

# S1 with the lone pairs of the donor oxygen added.
S3_LABELS = ("0 O 2p", "2 H 1s", "3 O 2p")
# Minimum cosines (occupied, virtual) of the S6 spaces from the prototype detector.
S6_MIN_COS = (0.999, 0.971)

# S66 system 19, methanol (atoms 0-5, hydroxyl H = 1) with pyridine (atoms 6-16, N = 6), in
# Angstrom. Copied verbatim from the S66 geometries used by the prototype runs.
METHANOL_PYRIDINE = (
    ("O", (-0.62765177, 0.08746727, 0.00147128)),
    ("H", (0.34360203, 0.12230333, -0.00060045)),
    ("C", (-0.97793123, -1.27855601, 0.00123841)),
    ("H", (-2.06339209, -1.34204332, 0.00500898)),
    ("H", (-0.61488369, -1.80637584, -0.88538395)),
    ("H", (-0.60864033, -1.80823682, 0.88417273)),
    ("N", (2.27233665, 0.0164323, -0.00162684)),
    ("C", (2.96870504, -0.00800303, -1.14634644)),
    ("H", (2.38422645, 0.01522051, -2.05732188)),
    ("C", (4.35834211, -0.05774589, -1.19503169)),
    ("H", (4.86569445, -0.07503793, -2.14881442)),
    ("C", (5.06871533, -0.08345851, 0.00058133)),
    ("H", (6.14905134, -0.12122326, 0.00143063)),
    ("C", (4.35646788, -0.0584374, 1.19512119)),
    ("H", (4.86226662, -0.07626173, 2.14960688)),
    ("C", (2.96691424, -0.00868772, 1.1441671)),
    ("H", (2.38090845, 0.01398671, 2.05428579)),
)
# Same labels and threshold for AB, A and B: the hydrogen bond O-H...N.
CORRESPONDING_SETTINGS = (
    (("0 O 2p", "1 H 1s", "6 N 2p"), 0.2),
    (("1 H 1s", "6 N 2p"), 0.2),
)
# Monomer A on a C-H bond instead of the hydroxyl, at a threshold that keeps the counts.
MISMATCHED_SETTINGS = (
    (CORRESPONDING_SETTINGS[0], (("0 O 2p", "3 H 1s"), 0.15)),
    (CORRESPONDING_SETTINGS[1], (("3 H 1s",), 0.1)),
)


def _xyz(atoms) -> str:
    lines = [str(len(atoms)), ""] + [f"{s} {x!r} {y!r} {z!r}" for s, (x, y, z) in atoms]
    return "\n".join(lines) + "\n"


def _dimer_subsystems(atoms, n_a: int) -> dict[str, Subsystem]:
    fragments = (tuple(range(n_a)), tuple(range(n_a, len(atoms))))
    return counterpoise(Molecule.from_xyz(_xyz(atoms), fragments=fragments), basis="cc-pvdz")


@pytest.fixture(scope="module")
def subsystems() -> dict[str, Subsystem]:
    return _dimer_subsystems(WATER_DIMER, 3)


def _overlap(subsystems) -> np.ndarray:
    return subsystems["AB"].mol.intor("int1e_ovlp")


def _avas(subsystems, name: str, labels, threshold: float):
    """AVAS space of one subsystem with its own labels and threshold, without the preflight."""
    patterns = resolve_labels({"AB": subsystems["AB"], name: subsystems[name]}, labels)[name]
    sub = subsystems[name]
    ncas, nelecas, mo = avas.avas(
        sub.scf, list(patterns), threshold=threshold, canonicalize=True, verbose=0
    )
    nelecas = int(nelecas)
    record = ActiveSpaceRecord(nelecas, ncas, (sub.mol.nelectron - nelecas) // 2, patterns, ())
    return record, mo


def _record(nelecas: int, ncas: int) -> ActiveSpaceRecord:
    return ActiveSpaceRecord(nelecas=nelecas, ncas=ncas, ncore=0, ao_labels=(), minao_indices=())


def _space(nelecas: int, *orbitals: np.ndarray):
    return _record(nelecas, len(orbitals)), np.column_stack(orbitals)


E = np.eye(5)  # orthonormal AO basis: the shared metric is the identity


def _tilted(i: int, cos: float) -> np.ndarray:
    """Basis function ``i`` rotated towards function 4 by an angle with cosine ``cos``."""
    return cos * E[:, i] + np.sqrt(1.0 - cos**2) * E[:, 4]


def test_monomers_selected_at_another_threshold_break_the_counts(subsystems):
    ab = _avas(subsystems, "AB", S3_LABELS, 0.2)
    a, b = (_avas(subsystems, name, S3_LABELS, 0.5) for name in ("A", "B"))

    with pytest.raises(
        ActiveSpaceMismatchError, match=r"counts AB \(14e,8o\) != A \(6e,4o\) \+ B \(6e,3o\)"
    ):
        check_correspondence(ab, a, b, _overlap(subsystems))


def test_counts_alone_reject_monomers_that_share_an_orbital():
    # Both monomers hold the same occupied orbital, so every block matches in dimension
    # and angle; only the electron and orbital counts reveal the double counting.
    a = _space(2, E[:, 0], E[:, 1])
    b = _space(2, E[:, 0], E[:, 2])
    ab = _space(2, E[:, 0], E[:, 1], E[:, 2])

    with pytest.raises(ActiveSpaceMismatchError, match="counts") as failure:
        check_correspondence(ab, a, b, np.eye(5))
    assert "block" not in str(failure.value)


def test_linearly_dependent_monomer_orbitals_lower_the_rank():
    # A and B share their occupied orbital: the union has two columns but rank one, so it
    # cannot span the two occupied dimer orbitals although the counts agree.
    a = _space(2, E[:, 0], E[:, 1])
    b = _space(2, E[:, 0], E[:, 2])
    ab = _space(4, E[:, 0], E[:, 3], E[:, 1], E[:, 2])

    with pytest.raises(
        ActiveSpaceMismatchError, match=r"occupied block: AB dimension 2 != A\+B rank 1"
    ):
        check_correspondence(ab, a, b, np.eye(5))


def test_empty_virtual_block_does_not_correspond():
    # No subsystem has an active virtual orbital: there is no angle to measure.
    a = _space(2, E[:, 0])
    b = _space(2, E[:, 1])
    ab = _space(4, E[:, 0], E[:, 1])

    with pytest.raises(ActiveSpaceMismatchError, match=r"virtual block: min cosine 0\.000 < 0\.9"):
        check_correspondence(ab, a, b, np.eye(5))


def test_monomer_on_the_wrong_oh_bond_fails_both_blocks(subsystems):
    labels, threshold = AVAS_SETTINGS["S1"]
    ab, b = (_avas(subsystems, name, labels, threshold) for name in ("AB", "B"))
    a = _avas(subsystems, "A", ("1 H 1s",), threshold)  # free O-H bond, not the donor bond

    report = check_correspondence(ab, a, b, _overlap(subsystems), cos_threshold=0.0)
    assert report.counts_conserved
    assert (report.min_cos_occupied, report.min_cos_virtual) == pytest.approx(
        (0.385, 0.368), abs=5e-3
    )
    with pytest.raises(
        ActiveSpaceMismatchError, match=r"occupied block: min cosine 0\.38\d < 0\.9.*virtual"
    ):
        check_correspondence(ab, a, b, _overlap(subsystems))


@pytest.mark.parametrize("block", ["occupied", "virtual"])
def test_threshold_rejects_cosine_089_and_accepts_091(block):
    a = _space(2, E[:, 0], E[:, 1])
    b = _space(2, E[:, 2], E[:, 3])

    def dimer(cos):
        occupied = _tilted(0, cos) if block == "occupied" else E[:, 0]
        virtual = _tilted(1, cos) if block == "virtual" else E[:, 1]
        return _space(4, occupied, E[:, 2], virtual, E[:, 3])

    with pytest.raises(ActiveSpaceMismatchError, match=rf"{block} block: min cosine 0\.890 < 0\.9"):
        check_correspondence(dimer(0.89), a, b, np.eye(5))
    report = check_correspondence(dimer(0.91), a, b, np.eye(5))
    assert report.corresponds
    assert getattr(report, f"min_cos_{block}") == pytest.approx(0.91, abs=1e-12)


def test_s6_spaces_correspond(subsystems):
    spaces = select_active_spaces(subsystems, *AVAS_SETTINGS["S6"])

    report = check_correspondence(spaces["AB"], spaces["A"], spaces["B"], _overlap(subsystems))
    assert report.corresponds and report.counts_conserved
    assert (report.min_cos_occupied, report.min_cos_virtual) == pytest.approx(S6_MIN_COS, abs=5e-3)
    assert min(report.min_cos_occupied, report.min_cos_virtual) >= 0.950
    assert (report.rank_occupied, report.rank_virtual) == (6, 2)
    assert report.cos_threshold == COS_THRESHOLD == 0.90


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("Q2M3_FRAGMENT_CALIBRATION") != "1",
    reason="correspondence threshold calibration runs only with Q2M3_FRAGMENT_CALIBRATION=1",
)
def test_phenol_pyridine_calibration(record_property):
    """The threshold separates corresponding from mismatched spaces on an O-H...N dimer.

    The phenol...pyridine geometry is not available yet, so the S66 methanol...pyridine
    dimer stands in for it. Corresponding samples go through the public workflow. Mismatched
    samples put monomer A on a C-H bond with the same active-space counts.
    """
    molecule = Molecule.from_xyz(
        _xyz(METHANOL_PYRIDINE), fragments=(tuple(range(6)), tuple(range(6, 17)))
    )
    corresponding = []
    for labels, threshold in CORRESPONDING_SETTINGS:
        result = workflow.run(
            molecule, avas_labels=labels, avas_threshold=threshold, mode="classical_only"
        )
        report = result.correspondence
        corresponding.append(min(report.min_cos_occupied, report.min_cos_virtual))

    subsystems = _dimer_subsystems(METHANOL_PYRIDINE, 6)
    mismatched = []
    for (labels, threshold), (labels_a, threshold_a) in MISMATCHED_SETTINGS:
        ab, b = (_avas(subsystems, name, labels, threshold) for name in ("AB", "B"))
        a = _avas(subsystems, "A", labels_a, threshold_a)
        report = check_correspondence(ab, a, b, _overlap(subsystems), cos_threshold=0.0)
        assert report.counts_conserved
        mismatched.append(min(report.min_cos_occupied, report.min_cos_virtual))

    summary = {"min_cos_corresponding": min(corresponding), "max_cos_mismatched": max(mismatched)}
    for key, value in summary.items():
        record_property(key, float(value))
    print(json.dumps({"corresponding": corresponding, "mismatched": mismatched} | summary))
    assert max(mismatched) < COS_THRESHOLD <= min(corresponding)
