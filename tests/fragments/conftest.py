"""Shared inputs for fragment-workflow tests."""

from __future__ import annotations

import os

import pytest

# S22 water dimer (system 2) in Angstrom; monomer A = atoms 0-2, monomer B = atoms 3-5.
# Copied verbatim from the geometry used by the fragment-workflow prototype runs.
WATER_DIMER = (
    ("O", (-1.551007, -0.114520, 0.000000)),
    ("H", (-1.934259, 0.762503, 0.000000)),
    ("H", (-0.599677, 0.040712, 0.000000)),
    ("O", (1.350625, 0.111469, 0.000000)),
    ("H", (1.680398, -0.373741, -0.758561)),
    ("H", (1.680398, -0.373741, 0.758561)),
)

# AVAS labels and threshold per active-space setting, shared by the dimer and both monomers.
# Copied verbatim from the active-space settings of the same prototype runs.
AVAS_SETTINGS = {
    "S1": (("2 H 1s", "3 O 2p"), 0.2),
    "S2": (("2 H 1s", "3 O 2p"), 0.1),
    "S6": (("0 O 2p", "1 H 1s", "2 H 1s", "3 O 2p", "4 H 1s", "5 H 1s"), 0.7),
}


def pytest_collection_modifyitems(config, items):
    """Skip external tests unless Q2M3_PSI4_PYTHON names the Psi4 interpreter."""
    if os.environ.get("Q2M3_PSI4_PYTHON"):
        return
    skip = pytest.mark.skip(reason="Q2M3_PSI4_PYTHON is not set to the Psi4 interpreter path")
    for item in items:
        if "external" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def water_dimer():
    return WATER_DIMER


@pytest.fixture
def avas_settings():
    return AVAS_SETTINGS
