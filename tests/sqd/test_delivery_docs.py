"""Keep the published SQD contract and research status reviewable in a checkout."""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from q2m3.sqd.result import SQDResult

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "doc/source/sqd.md"


def _field_rows(text):
    return dict(re.findall(r"^\| `([a-z][a-zA-Z0-9_]*)` \| (.+)\|$", text, re.MULTILINE))


def _validate_contract(text):
    rows = _field_rows(text)
    assert set(rows) == {field.name for field in fields(SQDResult)}
    for name in ("baseline_method", "baseline_tier", "baseline_downgrade_reason", "warnings"):
        assert rows[name].strip(), name


def _validate_research(text):
    for experiment, status in (("E1", "inconclusive"), ("E3", "not_run"), ("E6", "inconclusive")):
        assert re.search(rf"^\| {experiment} \| {status} \|", text, re.MULTILINE)


def test_documented_fields_match_public_result_contract():
    _validate_contract(GUIDE.read_text())


def test_documented_research_status_preserves_missing_experiments():
    _validate_research(GUIDE.read_text())


@pytest.mark.parametrize("name", ["sqd_energy", "baseline_method", "baseline_tier", "warnings"])
def test_documentation_guard_rejects_missing_fields(name):
    text = GUIDE.read_text()
    broken = re.sub(rf"^\| `{name}` \|.*\n", "", text, flags=re.MULTILINE)
    assert broken != text
    with pytest.raises(AssertionError):
        _validate_contract(broken)


def test_documentation_guard_rejects_shci_alias_without_method():
    text = GUIDE.read_text().replace("| `baseline_method` |", "| `shci` |")
    with pytest.raises(AssertionError):
        _validate_contract(text)


def test_documentation_guard_rejects_unrun_gpu_experiment_claim():
    text = GUIDE.read_text().replace("| E3 | not_run |", "| E3 | completed |")
    assert text != GUIDE.read_text()
    with pytest.raises(AssertionError):
        _validate_research(text)
