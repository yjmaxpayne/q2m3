"""Import of external interaction-energy references.

Two formats are read:

- CSV with the columns ``fragment_id``, ``method``, ``basis``, ``delta_e`` and ``unit``.
  The unit is ``Ha``, ``kcal/mol`` or ``kJ/mol``. Only rows of the requested fragment are
  kept.
- QCSchema JSON, one record. The importer reads ``return_result`` (Ha), ``model.method``,
  ``model.basis`` and ``extras["q2m3_fragment_id"]``. This field set is a trial: it becomes
  stable once a real QCSchema output from a quantum-chemistry program passes the same
  import checks.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from q2m3.constants import HARTREE_TO_KCAL_MOL, HARTREE_TO_KJ_MOL
from q2m3.fragments.errors import ReferenceImportError
from q2m3.fragments.result import ExternalReference

CSV_COLUMNS = ("fragment_id", "method", "basis", "delta_e", "unit")
HARTREE_PER_UNIT = {
    "Ha": 1.0,
    "kcal/mol": 1.0 / HARTREE_TO_KCAL_MOL,
    "kJ/mol": 1.0 / HARTREE_TO_KJ_MOL,
}


def import_references(
    path: str | PathLike[str], *, fragment_id: str, basis: str
) -> tuple[ExternalReference, ...]:
    """Read the references of ``fragment_id`` from a CSV or QCSchema JSON file.

    A reference whose basis equals ``basis`` (ignoring case) is a same-basis full-space
    reference; any other is an external reference. Values are returned in Ha.

    Raises:
        ReferenceImportError: The file is neither ``.csv`` nor ``.json``; a CSV column is
            missing; a unit is not ``Ha``, ``kcal/mol`` or ``kJ/mol``; a value is not a
            number; no CSV row has ``fragment_id``; or the QCSchema record lacks a field
            or names another fragment.
    """
    path = Path(path)
    if path.suffix == ".csv":
        rows = _csv_rows(path, fragment_id)
    elif path.suffix == ".json":
        rows = [_qcschema_row(path, fragment_id)]
    else:
        raise ReferenceImportError(f"{path}: reference files must be .csv or .json")
    return tuple(_reference(row, basis) for row in rows)


def _csv_rows(path: Path, fragment_id: str) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in CSV_COLUMNS if column not in (reader.fieldnames or ())]
        if missing:
            raise ReferenceImportError(f"{path}: missing columns: {', '.join(missing)}")
        rows = [row for row in reader if row["fragment_id"].strip() == fragment_id]
    if not rows:
        raise ReferenceImportError(f"{path}: no row for fragment {fragment_id!r}")
    for row in rows:
        unit = row["unit"].strip()
        if unit not in HARTREE_PER_UNIT:
            raise ReferenceImportError(
                f"{path}: unit must be one of {tuple(HARTREE_PER_UNIT)}, got {unit!r}"
            )
        row["delta_e"] = _number(path, row["delta_e"]) * HARTREE_PER_UNIT[unit]
    return rows


def _qcschema_row(path: Path, fragment_id: str) -> dict[str, Any]:
    record = json.loads(path.read_text())
    try:
        model, extras = record["model"], record["extras"]
        row = {
            "fragment_id": extras["q2m3_fragment_id"],
            "method": model["method"],
            "basis": model["basis"],
            "delta_e": _number(path, record["return_result"]),
        }
    except (KeyError, TypeError) as exc:
        raise ReferenceImportError(f"{path}: QCSchema record lacks the field {exc}") from exc
    if row["fragment_id"] != fragment_id:
        raise ReferenceImportError(
            f"{path}: extras['q2m3_fragment_id'] is {row['fragment_id']!r}, "
            f"expected {fragment_id!r}"
        )
    return row


def _number(path: Path, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ReferenceImportError(f"{path}: interaction energy {value!r} is not a number") from exc


def _reference(row: Mapping[str, Any], basis: str) -> ExternalReference:
    reference_basis = row["basis"].strip()
    same = reference_basis.lower() == basis.strip().lower()
    return ExternalReference(
        fragment_id=row["fragment_id"].strip(),
        method=row["method"].strip(),
        basis=reference_basis,
        delta_e=row["delta_e"],
        category="same_basis_full_space" if same else "external",
    )
