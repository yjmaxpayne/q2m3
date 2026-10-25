"""Keep the published SQD contract and research status reviewable in a checkout."""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from q2m3.sqd.result import SQDResult

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "doc/source/api-reference/sqd-results.rst"
EVIDENCE = ROOT / "doc/source/user-guide/sqd-validation.rst"
FIELD_ROW = r"^   \* - ``([a-z][a-zA-Z0-9_]*)``\n(.*?)(?=^   \* -|\Z)"


def _field_rows(text):
    return dict(re.findall(FIELD_ROW, text, re.MULTILINE | re.DOTALL))


def _validate_contract(text):
    rows = _field_rows(text)
    assert set(rows) == {field.name for field in fields(SQDResult)}
    for name in ("baseline_method", "baseline_tier", "baseline_downgrade_reason", "warnings"):
        assert rows[name].strip(), name


def _validate_research(text):
    for experiment, status in (("E1", "inconclusive"), ("E3", "not_run"), ("E6", "inconclusive")):
        assert re.search(rf"^   \* - {experiment}\n     - {status}\n", text, re.MULTILINE)


def test_documented_fields_match_public_result_contract():
    _validate_contract(CONTRACT.read_text())


def test_documented_research_status_preserves_missing_experiments():
    _validate_research(EVIDENCE.read_text())


@pytest.mark.parametrize("name", ["sqd_energy", "baseline_method", "baseline_tier", "warnings"])
def test_documentation_guard_rejects_missing_fields(name):
    text = CONTRACT.read_text()
    broken = re.sub(
        rf"^   \* - ``{name}``\n.*?(?=^   \* -|\Z)",
        "",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert broken != text
    with pytest.raises(AssertionError):
        _validate_contract(broken)


def test_documentation_guard_rejects_shci_alias_without_method():
    text = CONTRACT.read_text().replace("``baseline_method``", "``shci``")
    with pytest.raises(AssertionError):
        _validate_contract(text)


def test_documentation_guard_rejects_unrun_gpu_experiment_claim():
    text = EVIDENCE.read_text().replace("* - E3\n     - not_run", "* - E3\n     - completed")
    assert text != EVIDENCE.read_text()
    with pytest.raises(AssertionError):
        _validate_research(text)
