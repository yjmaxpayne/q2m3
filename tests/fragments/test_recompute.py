"""A full-mode result can be recomputed from its provenance and the same geometry."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from pyscf.mcscf import avas

from q2m3.fragments import FragmentWorkflowError, Molecule, run
from tests.fragments.conftest import AVAS_SETTINGS, WATER_DIMER

pytestmark = [pytest.mark.sqd, pytest.mark.slow]

# Settings that differ from the AVAS default threshold and from the run() defaults of
# basis and seed, so a recompute that drops one of them runs a different calculation.
BASIS = "sto-3g"
THRESHOLD = 0.8
SEED = 11
DETERMINISTIC_LAYERS = ("hf", "mp2", "casci", "ccsd_t")
TOLERANCE_HA = 1e-10
LABELS = AVAS_SETTINGS["S6"][0]
REPO = Path(__file__).resolve().parents[2]
DIMER = Molecule(
    symbols=tuple(symbol for symbol, _ in WATER_DIMER),
    coordinates_angstrom=tuple(xyz for _, xyz in WATER_DIMER),
    fragments=((0, 1, 2), (3, 4, 5)),
    charges=(0, 0),
    multiplicities=(1, 1),
)


@pytest.fixture(scope="module")
def child_path():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PYTHONPATH", f"{REPO}:{REPO / 'src'}")
        yield


@pytest.fixture(scope="module")
def original(child_path):
    return run(DIMER, avas_labels=LABELS, avas_threshold=THRESHOLD, basis=BASIS, seed=SEED)


def recompute(molecule, result):
    """Run ``molecule`` again with the settings recorded in ``result.provenance``."""
    provenance = result.provenance
    assert molecule.geometry_sha256() == provenance["geometry_sha256"]
    return run(
        molecule,
        avas_labels=provenance["avas_labels"],
        avas_threshold=provenance["avas_threshold"],
        basis=provenance["basis"],
        mode=result.mode,
        seed=provenance["seed"],
    )


def test_avas_default_threshold_selects_a_rejected_active_space():
    default = inspect.signature(avas.avas).parameters["threshold"].default
    assert THRESHOLD != default
    with pytest.raises(FragmentWorkflowError, match=r"\(16e,12o\) has more than 10"):
        run(DIMER, avas_labels=LABELS, avas_threshold=default, basis=BASIS, mode="classical_only")


def test_recompute_from_provenance_reproduces_the_result(original):
    again = recompute(DIMER, original)

    for layer in DETERMINISTIC_LAYERS:
        assert abs(again.delta_e[layer] - original.delta_e[layer]) <= TOLERANCE_HA, layer
    for name, subsystem in original.subsystems.items():
        repeat = again.subsystems[name]
        assert repeat.active_space == subsystem.active_space, name
        assert repeat.sqd.seed == subsystem.sqd.seed == SEED, name
        assert abs(repeat.e_sqd - subsystem.e_sqd) <= TOLERANCE_HA, name
