"""Result contracts remain useful without installing the sampling dependencies."""

from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

from q2m3.sqd.result import ReferenceResult, SQDResult, T1Residual, T2Diagnostics

REFERENCE_ONLY_FIELDS = (
    "sqd_energy",
    "delta_mHa",
    "delta_vs_sci_mHa",
    "ratio_sqd_over_sci",
    "iso_ndet_sci_energy",
    "iso_ndet_random_energy",
    "unique_dets_vs_shots",
    "subspace_dim",
    "subspace_dims",
    "n_reps",
    "shots",
    "backend",
)


def result_data():
    """A small, algebraically consistent completed result."""
    return dict(
        schema_version="sqd.result.v1",
        status="completed",
        sqd_energy=-1.12,
        hf_energy=-1.0,
        hf_reference_kind="canonical_rhf",
        baseline_energy=-1.14,
        iso_active_space_ccsd_energy=-1.13,
        iso_ndet_sci_energy=-1.11,
        iso_ndet_random_energy=-1.05,
        baseline_tier="T0",
        baseline_method="exact_casci",
        baseline_uncertainty_mHa=0.0,
        baseline_uncertainty_kind="exact_active_space",
        baseline_downgrade_reason=None,
        baseline_untrustworthy=False,
        baseline_t1_residual=None,
        t1_diagnostic=None,
        t2_diagnostics=None,
        reference_attempts=(),
        delta_mHa=20.0,
        delta_vs_sci_mHa=-10.0,
        ratio_sqd_over_sci=2 / 3,
        unique_dets_vs_shots=((5, 2), (10, 3)),
        active_space=(2, 2),
        n_reps=2,
        shots=10,
        seed=0,
        subspace_dim=4,
        subspace_dims=(2, 2),
        full_ci_dim=4,
        embedding_mode="vacuum",
        two_electron_tensor_fixed=True,
        fixed_mo=True,
        backend="ffsim",
        versions={"numpy": "2"},
        diagnostics={},
        provenance={},
        timings_s={
            name: (1.0 if name == "total" else 0.1)
            for name in (
                "geometry",
                "integrals",
                "ccsd",
                "prepare",
                "sample",
                "diagonalize",
                "reference",
                "comparison",
                "total",
            )
        },
        null_reasons={
            "baseline_downgrade_reason": "exact_reference",
            "baseline_t1_residual": "not_t1",
            "t1_diagnostic": "not_t2",
            "t2_diagnostics": "not_t2",
        },
        warnings=(),
    )


def reference_only_data():
    data = result_data()
    data["status"] = "reference_only"
    for name in REFERENCE_ONLY_FIELDS:
        data[name] = None
        data["null_reasons"][name] = "reference_only"
    for name in ("prepare", "sample", "diagonalize", "comparison"):
        data["timings_s"][name] = None
        data["null_reasons"][f"timings_s.{name}"] = "reference_only"
    return data


def test_result_has_exactly_forty_fields_and_returns_self():
    result = SQDResult(**result_data())
    assert len(fields(result)) == 40
    assert result.validate() is result
    with pytest.raises(FrozenInstanceError):
        result.seed = 1


def test_result_deep_snapshot_survives_input_mutation():
    data = result_data()
    coords = np.array([[1.0, 2.0, 3.0]])
    data["provenance"] = {"molecule": {"coordinates": coords}}
    result = SQDResult(**data)
    coords[0, 0] = 99
    data["versions"]["numpy"] = "changed"
    assert result.provenance["molecule"]["coordinates"][0][0] == 1.0
    assert result.versions["numpy"] == "2"
    with pytest.raises(TypeError):
        result.provenance["molecule"]["coordinates"] = ()


def test_reference_only_accepts_real_absence_with_reasons():
    assert SQDResult(**reference_only_data()).validate().sqd_energy is None


@pytest.mark.parametrize(
    "name,value",
    [
        ("delta_mHa", 0),
        ("unique_dets_vs_shots", ()),
        ("shots", 0),
        ("backend", "ffsim"),
    ],
)
def test_reference_only_rejects_fabricated_execution(name, value):
    data = reference_only_data()
    data[name] = value
    with pytest.raises(ValueError, match=name):
        SQDResult(**data).validate()


def test_reference_only_requires_explicit_reason():
    data = reference_only_data()
    del data["null_reasons"]["sqd_energy"]
    with pytest.raises(ValueError, match="sqd_energy"):
        SQDResult(**data).validate()


@pytest.mark.parametrize(
    "path", ["hf_energy", "delta_mHa", "diagnostics", "provenance", "timings_s"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_rejected_at_every_depth(path, value):
    data = result_data()
    data[path] = {"nested": [value]} if path in ("diagnostics", "provenance") else value
    if path == "timings_s":
        data[path] = {**result_data()[path], "total": value}
    with pytest.raises(ValueError):
        SQDResult(**data).validate()


@pytest.mark.parametrize(
    "changes",
    [
        {"delta_mHa": 0.0},
        {"delta_vs_sci_mHa": 10.0},
        {"ratio_sqd_over_sci": 1.0},
        {"subspace_dim": 3},
        {"unique_dets_vs_shots": ((5, 2),)},
        {"unique_dets_vs_shots": ((5, 6), (10, 7))},
        {"seed": True},
        {"full_ci_dim": 5},
        {"active_space": (3, 2)},
        {"schema_version": "old"},
    ],
)
def test_inconsistent_result_rejected(changes):
    data = result_data()
    data.update(changes)
    with pytest.raises(ValueError):
        SQDResult(**data).validate()


def test_unknown_uncertainty_cannot_be_zero():
    data = result_data()
    data.update(
        baseline_tier="T1+",
        baseline_method="shci_dice",
        baseline_uncertainty_kind="unknown",
        baseline_downgrade_reason="budget",
        ratio_sqd_over_sci=None,
        warnings=("baseline downgraded",),
    )
    data["null_reasons"]["ratio_sqd_over_sci"] = "non_exact_reference"
    data["null_reasons"]["baseline_uncertainty_mHa"] = "solver_error_unknown"
    with pytest.raises(ValueError, match="uncertainty"):
        SQDResult(**data).validate()
    data["baseline_uncertainty_mHa"] = None
    data["null_reasons"]["baseline_uncertainty_mHa"] = "solver_error_unknown"
    assert SQDResult(**data).validate().baseline_uncertainty_mHa is None


def test_reference_result_preserves_unknown_error():
    result = ReferenceResult(
        -1.1,
        "T1+",
        "shci_dice",
        None,
        "unknown",
        "unknown_error",
        "budget",
        None,
        None,
        False,
        "ham",
        "frame",
        (),
        ("downgraded",),
    )
    assert result.validate() is result
    with pytest.raises(ValueError, match="uncertainty"):
        replace(result, uncertainty_mHa=0).validate()


def test_t1_residual_belongs_to_loose_cutoff():
    residual = T1Residual(1e-4, 1e-5, -1.1, -1.11, 10.0, None)
    assert residual.validate() is residual
    with pytest.raises(ValueError):
        replace(residual, loose_residual_mHa=0).validate()
    assert T1Residual(None, 1e-4, None, -1.1, None, "paired_cutoff_not_run").validate()


def test_t2_flags_remain_independent_and_small_denominator_is_unknown():
    result = T2Diagnostics(0.03, True, -0.001, -0.1, 0.01, True, False, False, None)
    assert result.validate() is result
    with pytest.raises(ValueError):
        replace(result, high_t1=False).validate()
    assert T2Diagnostics(
        0.0, False, 0.0, 0.0, None, False, True, True, "correlation_denominator_too_small"
    ).validate()
    with pytest.raises(ValueError):
        T2Diagnostics(0.0, True, 0.0, 0.0, 0.0, False, False, False, None).validate()


def test_total_timing_covers_disjoint_executed_stages():
    data = result_data()
    data["timings_s"]["total"] = 0.1
    with pytest.raises(ValueError, match="total"):
        SQDResult(**data).validate()


def test_reference_only_without_ccsd_has_no_fabricated_energy_or_timing():
    data = reference_only_data()
    data["iso_active_space_ccsd_energy"] = None
    data["timings_s"]["ccsd"] = None
    data["null_reasons"].update(iso_active_space_ccsd_energy="not_run")
    data["null_reasons"]["timings_s.ccsd"] = "not_run"
    assert SQDResult(**data).validate().iso_active_space_ccsd_energy is None


@pytest.mark.parametrize("reason", [float("nan"), [], 3])
def test_nested_reference_records_reject_nonstring_reasons(reason):
    from q2m3.sqd.result import ReferenceAttempt

    with pytest.raises(ValueError):
        ReferenceAttempt("T0", "exact_casci", "success", reason, 0.1, 2.0).validate()
    with pytest.raises(ValueError):
        T1Residual(1e-4, 1e-5, -1.1, -1.11, 10, reason).validate()
    with pytest.raises(ValueError):
        ReferenceResult(
            -1.1,
            "T0",
            "exact_casci",
            0,
            "exact_active_space",
            reason,
            None,
            None,
            None,
            False,
            "ham",
            "frame",
            (),
            (),
        ).validate()


def test_completed_result_requires_executed_sampling_timing():
    data = result_data()
    data["timings_s"]["sample"] = None
    data["null_reasons"]["timings_s.sample"] = "not_run"
    with pytest.raises(ValueError, match="sample"):
        SQDResult(**data).validate()


def test_tiny_negative_sqd_ratio_records_tolerance_but_keeps_raw_gap():
    data = result_data()
    data["sqd_energy"] = data["baseline_energy"] - 5e-11
    data["delta_mHa"] = 1000 * (data["sqd_energy"] - data["baseline_energy"])
    data["delta_vs_sci_mHa"] = 1000 * (data["sqd_energy"] - data["iso_ndet_sci_energy"])
    data["ratio_sqd_over_sci"] = 0
    with pytest.raises(ValueError, match="tolerance"):
        SQDResult(**data).validate()
    data["diagnostics"]["ratio_tolerance_ha"] = 1e-10
    result = SQDResult(**data).validate()
    assert result.delta_mHa < 0
    assert result.ratio_sqd_over_sci == 0


def test_nonnegative_residuals_and_ratios_cannot_hide_in_numeric_tolerance():
    with pytest.raises(ValueError):
        T1Residual(1e-4, 1e-5, -1.0, -1.0, -1e-10, None).validate()
    with pytest.raises(ValueError):
        T2Diagnostics(0, True, 0, -1, -1e-10, False, False, False, None).validate()
    data = result_data()
    data.update(sqd_energy=-1.14, delta_mHa=0, delta_vs_sci_mHa=-30, ratio_sqd_over_sci=-1e-10)
    with pytest.raises(ValueError):
        SQDResult(**data).validate()


@pytest.mark.parametrize("factory", [result_data, reference_only_data])
def test_sqd_result_strict_json_roundtrip_preserves_fields_and_absence(tmp_path, factory):
    import json

    from q2m3.utils.io import save_json_results

    data = factory()
    data["diagnostics"]["nested_record"] = T1Residual(
        None, 1e-4, None, -1.1, None, "paired_cutoff_not_run"
    )
    result = SQDResult(**data).validate()
    output = tmp_path / "result.json"
    save_json_results(result, output)

    def reject_constant(value):
        raise AssertionError(f"nonstandard JSON constant: {value}")

    loaded = json.loads(output.read_text(), parse_constant=reject_constant)
    expected = {
        "schema_version",
        "status",
        "sqd_energy",
        "hf_energy",
        "hf_reference_kind",
        "baseline_energy",
        "iso_active_space_ccsd_energy",
        "iso_ndet_sci_energy",
        "iso_ndet_random_energy",
        "baseline_tier",
        "baseline_method",
        "baseline_uncertainty_mHa",
        "baseline_uncertainty_kind",
        "baseline_downgrade_reason",
        "baseline_untrustworthy",
        "baseline_t1_residual",
        "t1_diagnostic",
        "t2_diagnostics",
        "reference_attempts",
        "delta_mHa",
        "delta_vs_sci_mHa",
        "ratio_sqd_over_sci",
        "unique_dets_vs_shots",
        "active_space",
        "n_reps",
        "shots",
        "seed",
        "subspace_dim",
        "subspace_dims",
        "full_ci_dim",
        "embedding_mode",
        "two_electron_tensor_fixed",
        "fixed_mo",
        "backend",
        "versions",
        "diagnostics",
        "provenance",
        "timings_s",
        "null_reasons",
        "warnings",
    }
    assert {field.name for field in fields(result)} == expected == set(loaded)
    assert loaded["diagnostics"]["nested_record"]["reason"] == "paired_cutoff_not_run"
    assert loaded["active_space"] == [2, 2]
    if result.status == "reference_only":
        for name in REFERENCE_ONLY_FIELDS:
            assert loaded[name] is None
            assert loaded["null_reasons"][name] == "reference_only"
    else:
        assert loaded["sqd_energy"] == -1.12
        assert loaded["unique_dets_vs_shots"] == [[5, 2], [10, 3]]


@pytest.mark.parametrize("stage", ["reference"])
def test_reference_only_retained_energy_requires_execution_timing(stage):
    data = reference_only_data()
    data["timings_s"][stage] = None
    data["null_reasons"][f"timings_s.{stage}"] = "not_run"
    with pytest.raises(ValueError, match=stage):
        SQDResult(**data).validate()


@pytest.mark.parametrize("factory", [result_data, reference_only_data])
def test_supplied_integrals_and_seed_do_not_fabricate_local_timings(factory):
    data = factory()
    for stage in ("geometry", "integrals", "ccsd"):
        data["timings_s"][stage] = None
        data["null_reasons"][f"timings_s.{stage}"] = "supplied_integrals_and_seed"
    data["provenance"]["seed_source"] = "supplied_seed"
    assert SQDResult(**data).validate().iso_active_space_ccsd_energy == -1.13
