"""QPE and SQD resource estimates of fragment subsystems, and their checks in results.

The estimates come from the same active-space integrals that the SQD stage solves.
"""

from __future__ import annotations

import dataclasses
import math
from importlib.metadata import version
from types import SimpleNamespace

import pytest

from q2m3.core.resource_estimation import estimate_qpe_resources_from_integrals
from q2m3.fragments.active_space import select_active_spaces
from q2m3.fragments.counterpoise import counterpoise
from q2m3.fragments.integrals import build_integrals
from q2m3.fragments.molecule import Molecule
from q2m3.fragments.resources import qpe_estimate, sqd_estimate
from q2m3.fragments.result import (
    QPE_ESTIMATOR,
    SQD_ESTIMATOR,
    ResourceEstimate,
    SubsystemResult,
)
from q2m3.sqd.config import LUCJConfig
from q2m3.sqd.exceptions import ResourceModelDomainError
from q2m3.sqd.resources import estimate_rss_mb, load_resource_model
from tests.fragments.conftest import AVAS_SETTINGS, WATER_DIMER

DIMER_FRAGMENTS = ((0, 1, 2), (3, 4, 5))
LABELS, THRESHOLD = AVAS_SETTINGS["S6"]
SUBSYSTEMS = ("AB", "A", "B")
VERSIONS = {"q2m3": version("q2m3"), "pennylane": version("pennylane")}


def _xyz(atoms) -> str:
    lines = [str(len(atoms)), ""] + [f"{s} {x!r} {y!r} {z!r}" for s, (x, y, z) in atoms]
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def integrals():
    molecule = Molecule.from_xyz(_xyz(WATER_DIMER), fragments=DIMER_FRAGMENTS)
    subsystems = counterpoise(molecule, basis="cc-pvdz")
    spaces = select_active_spaces(subsystems, LABELS, THRESHOLD)
    return {
        name: build_integrals(
            molecule, subsystems[name], record, mo, avas_labels=LABELS, avas_threshold=THRESHOLD
        )
        for name, (record, mo) in spaces.items()
    }


@pytest.mark.parametrize("name", SUBSYSTEMS)
def test_qpe_estimate_comes_from_the_subsystem_integrals(integrals, name):
    built = integrals[name]

    estimate = qpe_estimate(built, basis="cc-pvdz")

    expected = estimate_qpe_resources_from_integrals(built.h1, built.h2, basis="cc-pvdz")
    assert estimate.values == dataclasses.asdict(expected)
    assert estimate.estimator == QPE_ESTIMATOR
    assert estimate.versions == VERSIONS
    # The numbers cost a QPE on this Hamiltonian; no circuit was executed.
    assert "cost estimate" in estimate.scope
    assert "not a QPEEngine" in estimate.scope


# (lambda in Ha, logical qubits, Toffoli gates) from the prototype double-factorization run
# on the same S6 active spaces. The dimer values move when either truncation tolerance moves.
S6_QPE_REFERENCE = {
    "AB": (26.057031281298833, 310, 61166562),
    "A": (8.775194047834402, 133, 9348360),
    "B": (8.769497919796239, 213, 9582930),
}


@pytest.mark.parametrize("name", SUBSYSTEMS)
def test_qpe_estimate_of_s6_subsystems_matches_reference_values(integrals, name):
    values = qpe_estimate(integrals[name], basis="cc-pvdz").values

    lam, qubits, toffoli = S6_QPE_REFERENCE[name]
    assert values["hamiltonian_1norm"] == pytest.approx(lam, abs=1e-6)
    assert values["logical_qubits"] == qubits
    assert values["toffoli_gates"] == toffoli


@pytest.mark.sqd
@pytest.mark.parametrize("name", SUBSYSTEMS)
def test_sqd_estimate_records_workload_and_stage_memory(integrals, name):
    built = integrals[name]
    lucj = LUCJConfig(n_reps=1, shots=50_000)

    estimate = sqd_estimate(built, lucj=lucj)

    model = load_resource_model()
    norb, nelec = built.record.ncas, built.nelec
    assert estimate.values == {
        "model_id": model.model_id,
        "qubits": 2 * norb,
        "shots": 50_000,
        "n_reps": 1,
        "full_ci_dim": math.comb(norb, nelec[0]) * math.comb(norb, nelec[1]),
        "rss_mb": {
            stage: estimate_rss_mb(norb, nelec, stage=stage, model=model)
            for stage in ("prepare", "sample")
        },
    }
    assert estimate.estimator == SQD_ESTIMATOR
    assert estimate.versions == VERSIONS


@pytest.mark.sqd
def test_sqd_estimate_rejects_a_workload_outside_the_calibrated_profile(integrals):
    with pytest.raises(ResourceModelDomainError, match="shots"):
        sqd_estimate(integrals["A"], lucj=LUCJConfig(shots=200_000))


QPE = ResourceEstimate(QPE_ESTIMATOR, VERSIONS, {"hamiltonian_1norm": 8.8}, "qpe scope")
SQD = ResourceEstimate(SQD_ESTIMATOR, VERSIONS, {"qubits": 8}, "sqd scope")


def _subsystem(resources, **extra_reasons) -> SubsystemResult:
    reasons = {key: "not computed" for key in ("e_mp2", "e_ccsd_t", "e_casci", "e_sqd", "sqd")}
    return SubsystemResult(
        name="A",
        real_atoms=(0, 1, 2),
        ghost_atoms=(3, 4, 5),
        n_ao=48,
        e_hf=-76.0,
        e_mp2=None,
        e_ccsd_t=None,
        e_casci=None,
        e_sqd=None,
        active_space=None,
        sqd=None,
        resources=resources,
        null_reasons=reasons | {"active_space": "not resolved"} | extra_reasons,
    )


def test_estimates_in_their_own_slots_validate():
    subsystem = _subsystem({"qpe": QPE, "sqd": SQD})
    assert subsystem.validate() is subsystem


def test_null_slot_with_a_reason_validates():
    _subsystem({"qpe": None, "sqd": SQD}, **{"resources.qpe": "not computed"}).validate()


def test_qpe_estimate_in_the_sqd_slot_is_rejected():
    with pytest.raises(ValueError, match=r"resources\['sqd'\].*" + QPE_ESTIMATOR):
        _subsystem({"qpe": QPE, "sqd": QPE}).validate()


@pytest.mark.parametrize(
    "estimate",
    [
        dataclasses.replace(QPE, estimator=""),
        {"estimator": QPE_ESTIMATOR, "hamiltonian_1norm": 8.8},
        object(),
        SimpleNamespace(estimator=QPE_ESTIMATOR, values={}),
    ],
    ids=["empty", "plain-mapping", "other-type", "not-a-record"],
)
def test_estimate_without_an_estimator_is_rejected(estimate):
    with pytest.raises(ValueError, match=r"resources\['qpe'\]"):
        _subsystem({"qpe": estimate, "sqd": SQD}).validate()


def test_unknown_resource_key_is_rejected():
    with pytest.raises(ValueError, match="rss"):
        _subsystem({"qpe": QPE, "sqd": SQD, "rss": SQD}).validate()
