"""Import of external interaction-energy references from CSV and QCSchema JSON files."""

from __future__ import annotations

import json

import pytest

from q2m3.constants import HARTREE_TO_KCAL_MOL, HARTREE_TO_KJ_MOL
from q2m3.fragments.errors import ReferenceImportError
from q2m3.fragments.importers import import_references
from q2m3.fragments.result import decompose_errors

FRAGMENT = "water-dimer"
BASIS = "cc-pvdz"
HEADER = "fragment_id,method,basis,delta_e,unit"


def _csv(tmp_path, *rows: str, header: str = HEADER):
    path = tmp_path / "references.csv"
    path.write_text("\n".join((header, *rows)) + "\n")
    return path


def _qcschema(tmp_path, **changes):
    record = {
        "return_result": -0.0079,
        "model": {"method": "CCSD(T)", "basis": "CBS"},
        "extras": {"q2m3_fragment_id": FRAGMENT},
    } | changes
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(record))
    return path


def _import(path):
    return import_references(path, fragment_id=FRAGMENT, basis=BASIS)


def test_csv_without_a_unit_column_is_rejected(tmp_path):
    path = _csv(
        tmp_path, f"{FRAGMENT},CCSD(T),CBS,-0.0079", header="fragment_id,method,basis,delta_e"
    )
    with pytest.raises(ReferenceImportError, match="unit"):
        _import(path)


def test_csv_unit_must_be_one_of_the_allowed_spellings(tmp_path):
    path = _csv(tmp_path, f"{FRAGMENT},CCSD(T),CBS,-4.96,kcal")
    with pytest.raises(ReferenceImportError, match="'kcal'"):
        _import(path)


def test_csv_without_a_row_for_the_fragment_is_rejected(tmp_path):
    path = _csv(tmp_path, "phenol-pyridine,CCSD(T),CBS,-0.0110,Ha")
    with pytest.raises(ReferenceImportError, match=FRAGMENT):
        _import(path)


def test_csv_keeps_only_the_rows_of_the_fragment(tmp_path):
    path = _csv(
        tmp_path,
        "phenol-pyridine,CCSD(T),CBS,-0.0110,Ha",
        f"{FRAGMENT},CCSD(T),CBS,-0.0079,Ha",
    )
    (reference,) = _import(path)
    assert (reference.fragment_id, reference.method, reference.delta_e) == (
        FRAGMENT,
        "CCSD(T)",
        -0.0079,
    )


def test_same_number_in_ha_and_in_kcal_per_mol_gives_different_model_errors(tmp_path):
    # Ignoring the unit column would read both files as the same reference.
    delta_e = {"casci": -0.0050}
    errors = []
    for unit in ("Ha", "kcal/mol"):
        references = _import(_csv(tmp_path, f"{FRAGMENT},CCSD(T),CBS,-4.96,{unit}"))
        errors.append(decompose_errors(delta_e, references)[0].model)
    assert errors[0] == pytest.approx(-0.0050 + 4.96)
    assert errors[1] == pytest.approx(-0.0050 + 4.96 / HARTREE_TO_KCAL_MOL)


@pytest.mark.parametrize(
    ("value", "unit", "hartree"),
    [
        ("-0.0079", "Ha", -0.0079),
        ("-4.96", "kcal/mol", -4.96 / 627.5094),
        ("-20.75", "kJ/mol", -20.75 / (627.5094 * 4.184)),
    ],
)
def test_csv_values_are_converted_to_hartree(tmp_path, value, unit, hartree):
    (reference,) = _import(_csv(tmp_path, f"{FRAGMENT},CCSD(T),CBS,{value},{unit}"))
    assert reference.delta_e == pytest.approx(hartree, rel=1e-15)


def test_kj_per_mol_constant_uses_the_thermochemical_calorie():
    assert HARTREE_TO_KJ_MOL == HARTREE_TO_KCAL_MOL * 4.184


@pytest.mark.parametrize(
    ("basis", "category"),
    [("cc-pVDZ", "same_basis_full_space"), ("CBS", "external"), ("aug-cc-pvdz", "external")],
)
def test_category_follows_the_basis_of_the_reference(tmp_path, basis, category):
    (reference,) = _import(_csv(tmp_path, f"{FRAGMENT},CCSD(T),{basis},-0.0079,Ha"))
    assert (reference.basis, reference.category) == (basis, category)


def test_qcschema_record_with_the_trial_field_set_is_imported(tmp_path):
    (reference,) = _import(_qcschema(tmp_path))
    assert reference.fragment_id == FRAGMENT
    assert (reference.method, reference.basis) == ("CCSD(T)", "CBS")
    assert reference.delta_e == -0.0079
    assert reference.category == "external"


def test_qcschema_basis_sets_the_category(tmp_path):
    path = _qcschema(tmp_path, model={"method": "CCSD(T)", "basis": "cc-pVDZ"})
    (reference,) = _import(path)
    assert reference.category == "same_basis_full_space"


@pytest.mark.parametrize(
    "extras",
    [{}, {"q2m3_fragment_id": "phenol-pyridine"}],
    ids=["missing-fragment-id", "other-fragment"],
)
def test_qcschema_record_must_name_this_fragment(tmp_path, extras):
    with pytest.raises(ReferenceImportError, match="q2m3_fragment_id"):
        _import(_qcschema(tmp_path, extras=extras))


def test_unknown_file_type_is_rejected(tmp_path):
    path = tmp_path / "references.txt"
    path.write_text(HEADER + "\n")
    with pytest.raises(ReferenceImportError, match=".txt"):
        _import(path)
