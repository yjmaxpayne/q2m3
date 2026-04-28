# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Lightweight tests for examples/resource_estimation_survey.py."""


def test_survey_systems_include_nh3_scale_expansion():
    """Survey includes the added near-NH3 systems requested for task 3."""
    from examples.resource_estimation_survey import survey_systems

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
    from examples.resource_estimation_survey import survey_systems

    specs = survey_systems()
    labels = {spec.label for spec in specs}

    assert {"HeH+", "H3+", "H4 linear", "LiH", "H2O (4e,4o)"}.issubset(labels)

    lih = next(spec for spec in specs if spec.label == "LiH")
    assert lih.active_electrons == 4
    assert lih.active_orbitals == 4

    reduced_water = next(spec for spec in specs if spec.label == "H2O (4e,4o)")
    assert reduced_water.active_orbitals * 2 == 8
