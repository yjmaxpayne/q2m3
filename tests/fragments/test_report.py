"""Markdown report of fragment results: content, banned terms, Ks display, and null values."""

from __future__ import annotations

from dataclasses import replace

import pytest

from q2m3.constants import HARTREE_TO_KCAL_MOL
from q2m3.fragments import report
from q2m3.fragments.report import BANNED_TERMS, FIXED_STATEMENT, render_report
from q2m3.fragments.result import (
    ERROR_TERMS,
    QPE_ESTIMATOR,
    SQD_ESTIMATOR,
    FragmentInteractionResult,
    ResourceEstimate,
)
from tests.fragments.test_result_validate import _both_imported, full_result

# Toffoli counts differ per subsystem so that a table that repeats one subsystem fails.
TOFFOLI = {"AB": 61166562, "A": 4410040, "B": 4518082}
KS_TABLE = {"cocrystal-1": 0.85, "cocrystal-2": 12.5}
STATISTICS_TERMS = ("Spearman", "Pearson", "p-value", "p 值", "相关系数")


def _resources(name: str) -> dict[str, ResourceEstimate]:
    qpe = {"hamiltonian_1norm": 26.057031, "logical_qubits": 310, "toffoli_gates": TOFFOLI[name]}
    sqd = {"qubits": 16, "full_ci_dim": 784, "rss_mb": {"prepare": 690.54592, "sample": 714.99328}}
    return {
        "qpe": ResourceEstimate(QPE_ESTIMATOR, {}, qpe, "cost estimate"),
        "sqd": ResourceEstimate(SQD_ESTIMATOR, {}, sqd, "workload"),
    }


def report_result(**changes) -> FragmentInteractionResult:
    """Full result with both references, all four error terms, and both resource estimates."""
    result = _both_imported()
    subsystems = {
        name: replace(sub, resources=_resources(name)) for name, sub in result.subsystems.items()
    }
    return replace(result, subsystems=subsystems, **changes)


def _ha(value: float) -> str:
    return f"{value:.6f}"


def _kcal(value: float) -> str:
    return f"{value * HARTREE_TO_KCAL_MOL:.3f}"


ERROR_ROWS = {
    "algorithm": "| 算法误差",
    "active_space": "| 活性空间误差",
    "basis": "| 基组误差",
    "model": "| 模型误差",
}


def _cells(text: str, row_prefix: str) -> list[str]:
    """Return the stripped cells of the first table row that starts with ``row_prefix``."""
    row = next(line for line in text.splitlines() if line.startswith(row_prefix))
    return [cell.strip() for cell in row.strip("|").split("|")]


def assert_report_contents(text: str, result: FragmentInteractionResult) -> None:
    """Check that the report holds every layer, error term, resource estimate, and the statement."""
    assert FIXED_STATEMENT in text
    for value in result.delta_e.values():
        if value is not None:
            assert _ha(value) in text and _kcal(value) in text
    for term in ERROR_TERMS:
        # Compare whole cells: "12.550" is a substring of "-12.550".
        value = getattr(result.errors, term)
        expected = [_ha(value), _ha(abs(value)), _kcal(value), _kcal(abs(value))]
        assert _cells(text, ERROR_ROWS[term])[1:5] == expected
    assert "活性空间 + 基组" in text
    assert "双因子分解成本估计" in text
    assert all(str(count) in text for count in TOFFOLI.values())
    assert "690.5" in text and "715.0" in text
    assert "同基组全空间参考来源：imported" in text


def test_report_holds_energies_errors_resources_and_the_statement():
    result = report_result()
    # A negative term is needed: on non-negative values a signed column equals its absolute one.
    assert any(getattr(result.errors, term) < 0 for term in ERROR_TERMS)

    assert_report_contents(render_report([result]), result)


def test_report_covers_every_result_it_is_given():
    first, second = report_result(), report_result(fragment_id="phenol-pyridine")

    text = render_report([first, second])

    assert "water-dimer" in text and "phenol-pyridine" in text


@pytest.mark.parametrize("term", [*BANNED_TERMS, "Quantum Advantage"])
def test_banned_term_in_a_template_is_rejected(monkeypatch, term):
    monkeypatch.setattr(report, "TITLE", f"# {term}")

    with pytest.raises(ValueError, match="banned term"):
        render_report([report_result()])


def test_banned_term_in_a_ks_label_is_rejected():
    with pytest.raises(ValueError, match="banned term"):
        render_report([report_result()], ks_table={"量子优势 sample": 1.0})


def test_banned_terms_hold_the_two_required_phrases():
    assert {"量子优势", "quantum advantage"} <= set(BANNED_TERMS)


def test_ks_table_is_shown_next_to_the_results_without_statistics():
    text = render_report([report_result()], ks_table=KS_TABLE)

    assert all(f"| {label} | {ks:g} |" in text for label, ks in KS_TABLE.items())
    assert not [term for term in STATISTICS_TERMS if term.casefold() in text.casefold()]


def test_report_without_ks_table_has_no_ks_section():
    assert "Ks 值" not in render_report([report_result()])


def test_contents_check_fails_when_the_statement_is_missing(monkeypatch):
    result = report_result()
    monkeypatch.setattr(report, "FIXED_STATEMENT", "")

    with pytest.raises(AssertionError):
        assert_report_contents(render_report([result]), result)


def test_null_values_show_their_reasons_and_no_number():
    base = full_result()
    sub = base.subsystems["A"]
    sub = replace(
        sub, resources={}, null_reasons={**sub.null_reasons, "resources.qpe": "qpe skipped"}
    )
    result = replace(
        base,
        subsystems={**base.subsystems, "A": sub},
        null_reasons={
            **base.null_reasons,
            "delta_e.ccsd_t": "ccsd_t over budget",
            "errors.basis": "missing external reference",
            "errors.model": "model needs both references",
        },
    )

    text = render_report([result])

    ccsd_t_row = next(line for line in text.splitlines() if line.startswith("| CCSD(T) |"))
    assert "ccsd_t over budget" in ccsd_t_row and "0.000" not in ccsd_t_row
    basis_row = next(line for line in text.splitlines() if line.startswith(ERROR_ROWS["basis"]))
    assert "missing external reference" in basis_row and "0.000" not in basis_row
    assert "model needs both references" in text
    qpe_a_row = next(
        line
        for line in text.split("双因子分解成本估计")[1].splitlines()
        if line.startswith("| A |")
    )
    assert "qpe skipped" in qpe_a_row
