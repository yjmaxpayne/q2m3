"""Regression checks for truncation bias and explicit example acceptance."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from examples.fragments import _common, _inputs
from q2m3.constants import HARTREE_TO_KCAL_MOL


@pytest.mark.parametrize(
    "settings,reps", [(_inputs.WATER_SETTINGS, 24), (_inputs.CRYSTAL_SETTINGS, 36)]
)
def test_full_examples_keep_all_ucj_factors(settings, reps):
    args = _common.parser("test", mode="full").parse_args([])
    options = _common.calculation_options(args, *settings)
    assert options["lucj"].n_reps == reps
    assert options["lucj"].shots == 100_000


def example_result(errors=(0.0002, 0.0, 0.00004), tier="T0"):
    subsystems = {}
    for name, reference, error, active in zip(
        ("AB", "A", "B"), (-784.7, -502.4, -282.29), errors, ((12, 9), (6, 4), (6, 5)), strict=True
    ):
        subsystems[name] = SimpleNamespace(
            e_casci=reference if tier == "T0" else None,
            e_sqd=reference + error,
            sqd=SimpleNamespace(
                baseline_tier=tier, baseline_untrustworthy=False, n_reps=36, active_space=active
            ),
        )
    delta = {
        key: getattr(subsystems["AB"], f"e_{key}")
        - getattr(subsystems["A"], f"e_{key}")
        - getattr(subsystems["B"], f"e_{key}")
        for key in ("sqd", "casci")
        if tier == "T0" or key == "sqd"
    }
    delta.setdefault("casci", None)
    return SimpleNamespace(mode="full", subsystems=subsystems, delta_e=delta)


def test_accuracy_evidence_includes_subsystems_and_interaction():
    from examples.fragments._accuracy import assess_accuracy

    evidence = assess_accuracy(example_result())
    assert evidence["passed"]
    assert evidence["subsystems"]["AB"]["error_mha"] == pytest.approx(0.2)
    assert evidence["interaction_error_kcal_mol"] == pytest.approx(0.00016 * HARTREE_TO_KCAL_MOL)


@pytest.mark.parametrize(
    "errors,tier",
    [
        ((0.01347956, 0.00030437, 0.00200762), "T0"),  # original crystal failure
        ((0.004, 0.002, 0.002), "T0"),  # exact cancellation must not hide subsystem errors
        ((-0.0001, 0.0, 0.0), "T0"),  # nonvariational result
        ((0.0004, 0.0, 0.0), "T0"),  # interaction > 0.25 kcal/mol
        ((float("nan"), 0.0, 0.0), "T0"),
        ((0.0, 0.0, 0.0), "T1"),
    ],
)
def test_accuracy_rejects_inadequate_results(errors, tier):
    from examples.fragments._accuracy import assess_accuracy

    assert not assess_accuracy(example_result(errors, tier))["passed"]


def test_accuracy_rejects_truncated_preparation_even_with_small_error():
    from examples.fragments._accuracy import assess_accuracy

    result = example_result()
    result.subsystems["AB"].sqd.n_reps = 2
    assert not assess_accuracy(result)["passed"]


def test_failure_retains_results_and_accuracy_evidence(tmp_path, monkeypatch):
    result = example_result((0.004, 0.002, 0.002))
    result.fragment_id = "test"
    result.basis = "sto-3g"
    result.null_reasons = {}
    for method in ("hf", "mp2", "ccsd_t"):
        result.delta_e[method] = None
        result.null_reasons[f"delta_e.{method}"] = "not used by this acceptance test"
    result.validate = lambda: None
    actual_writer = _common.write_json
    monkeypatch.setattr(
        _common,
        "write_json",
        lambda path, value: actual_writer(path, {"retained": True} if value is result else value),
    )
    monkeypatch.setattr(_common, "result_warnings", lambda _: [])
    monkeypatch.setattr("q2m3.fragments.render_report", lambda *_, **__: "Report\n")
    output = tmp_path / "failed"
    status = _common.execute(
        SimpleNamespace(output=output), "test", lambda args, path: _common.save_result(path, result)
    )
    assert status == 1
    assert (output / "result.json").exists()
    assert not json.loads((output / "accuracy.json").read_text())["passed"]
    assert "FAIL" in (output / "report.md").read_text()
    assert "accuracy" in (output / "failure.json").read_text().lower()


def test_extended_resource_profile_remains_bounded():
    from q2m3.sqd.exceptions import ResourceModelDomainError
    from q2m3.sqd.resources import load_resource_model

    model = load_resource_model(profile={"n_reps": 36, "shots": 100_000})
    assert model.model_id != "serial-active-space-20260917-v1"
    with pytest.raises(ResourceModelDomainError, match="n_reps"):
        load_resource_model(profile={"n_reps": 37})
    with pytest.raises(ResourceModelDomainError, match="shots"):
        load_resource_model(profile={"n_reps": 36, "shots": 100_001})
