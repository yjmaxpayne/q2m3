# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Lightweight tests for examples/resources/resource_estimation_survey.py."""

import pytest


def test_survey_systems_include_nh3_scale_expansion():
    """Survey includes the added near-NH3 systems requested for task 3."""
    from examples.resources.resource_estimation_survey import survey_systems

    specs = survey_systems()
    labels = {spec.label for spec in specs}
    runnable = [spec for spec in specs if not spec.skip_reason]

    assert {"H2O", "CH4", "NH4+", "Formamide"}.issubset(labels)
    assert len(runnable) >= 8

    nh4 = next(spec for spec in specs if spec.label == "NH4+")
    assert nh4.charge == 1
    assert nh4.active_electrons == 8
    assert nh4.active_orbitals == 7

    formamide = next(spec for spec in specs if spec.label == "Formamide")
    assert len(formamide.symbols) < 10  # smaller than the 10-atom Glycine row
    assert formamide.active_electrons + formamide.active_orbitals <= 20
    assert formamide.active_orbitals * 2 < 20


def test_survey_systems_include_measured_compile_bridge_points():
    """Survey includes small systems used to add measured compile data."""
    from examples.resources.resource_estimation_survey import survey_systems

    specs = survey_systems()
    labels = {spec.label for spec in specs}

    assert {"HeH+", "H3+", "H4 linear", "LiH", "H2O (4e,4o)"}.issubset(labels)

    lih = next(spec for spec in specs if spec.label == "LiH")
    assert lih.active_electrons == 4
    assert lih.active_orbitals == 4

    reduced_water = next(spec for spec in specs if spec.label == "H2O (4e,4o)")
    assert reduced_water.active_orbitals * 2 == 8


def test_estimate_one_runtime_uses_total_toffoli_once():
    """BUG-QRE-RUNTIME-001: survey row must not re-multiply DF total by an iteration count.

    ``EFTQCResources.toffoli_gates`` is already the whole-QPE Toffoli count, so
    runtime_seconds == toffoli_gates * tick_us * 1e-6 with no extra factor, and the
    informational walk_operator_calls column is PennyLane's ceil(pi*lambda/(2*eps)).
    """
    import numpy as np

    from examples.resources.resource_estimation_survey import (
        DEFAULT_TARGET_ERROR,
        estimate_one,
        survey_systems,
    )

    h2 = next(spec for spec in survey_systems() if spec.label == "H2")
    rec = estimate_one(h2)

    assert rec["runtime_seconds"] == pytest.approx(rec["toffoli_gates"] * rec["tick_us"] * 1e-6)
    assert rec["walk_operator_calls"] == int(
        np.ceil(np.pi * rec["hamiltonian_1norm_Ha"] / (2 * DEFAULT_TARGET_ERROR))
    )
    assert "qpe_iterations" not in rec
