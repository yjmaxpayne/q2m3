"""Cost estimate and budget gate of the full-space CCSD(T) reference.

The gate predicts the peak RSS and the wall time of each counterpoise subsystem before any
CCSD(T) runs. The RSS criterion uses the largest of the predictions, and the wall-time
criterion uses their sum, because the subsystems run one after the other. A prediction
equal to the budget passes.
"""

from __future__ import annotations

import json
import os

import pytest
from pyscf import cc, lib
from pyscf.data.elements import chemcore

from q2m3.constants import HARTREE_TO_KCAL_MOL
from q2m3.fragments.counterpoise import counterpoise
from q2m3.fragments.errors import FragmentWorkflowError
from q2m3.fragments.molecule import Molecule
from q2m3.fragments.references import (
    BASE_MB,
    MODEL_ID,
    CCSDTBudget,
    CostEstimate,
    estimate_ccsd_t_cost,
    frozen_core_ccsd_t,
    run_ccsd_t_stage,
)
from tests.fragments.conftest import WATER_DIMER
from tests.fragments.test_correspondence import METHANOL_PYRIDINE, _xyz

BUDGET = CCSDTBudget()
# Free memory above the default RSS budget, so the budget itself is the effective limit.
PLENTY_MB = 64000.0
# Water dimer in cc-pVDZ: (nocc, nvir) after the frozen core, per subsystem.
WATER_ACTIVE = {"AB": (8, 38), "A": (4, 43), "B": (4, 43)}
WATER_FROZEN = {"AB": 2, "A": 1, "B": 1}
ENERGY = -1.0


@pytest.fixture(scope="module")
def water():
    molecule = Molecule.from_xyz(_xyz(WATER_DIMER), fragments=((0, 1, 2), (3, 4, 5)))
    subsystems = counterpoise(molecule, basis="cc-pvdz")
    return subsystems, {name: chemcore(sub.mol) for name, sub in subsystems.items()}


class RecordingModel:
    """Cost model whose predictions depend only on nocc: 8 for AB, 4 for A and B."""

    def __init__(self, rss_mb: dict[int, float], wall_s: dict[int, float]):
        self.rss_mb, self.wall_s, self.calls = rss_mb, wall_s, []

    def __call__(self, nao, nocc, nvir, nfrozen, *, max_memory_mb):
        self.calls.append((nao, nocc, nvir, nfrozen, max_memory_mb))
        return CostEstimate(self.rss_mb[nocc], self.wall_s[nocc], True, "recording", False)


class RecordingSolver:
    def __init__(self):
        self.calls = []

    def __call__(self, subsystem, nfrozen, *, max_memory_mb):
        self.calls.append((subsystem.name, nfrozen, max_memory_mb))
        return ENERGY


def _stage(water, model, *, available=PLENTY_MB, budget=BUDGET):
    subsystems, nfrozen = water
    solver = RecordingSolver()
    stage = run_ccsd_t_stage(
        subsystems,
        nfrozen,
        budget,
        cost_model=model,
        mem_available_mb=lambda: available,
        solver=solver,
    )
    return stage, solver


def test_rss_prediction_equal_to_the_budget_runs_every_subsystem(water):
    # The largest prediction comes from A and B, not from the dimer.
    model = RecordingModel({8: 1000.0, 4: BUDGET.rss_mb}, {8: 1.0, 4: 1.0})
    stage, solver = _stage(water, model)
    assert stage.ran
    assert stage.energies == {"AB": ENERGY, "A": ENERGY, "B": ENERGY}
    max_memory = BUDGET.rss_mb - BASE_MB
    assert solver.calls == [(name, WATER_FROZEN[name], max_memory) for name in ("AB", "A", "B")]
    assert [call[1:4] for call in model.calls] == [
        (*WATER_ACTIVE[name], WATER_FROZEN[name]) for name in ("AB", "A", "B")
    ]


def test_rss_prediction_one_mb_above_the_budget_runs_nothing(water):
    model = RecordingModel({8: 1000.0, 4: BUDGET.rss_mb + 1.0}, {8: 1.0, 4: 1.0})
    stage, solver = _stage(water, model)
    assert not stage.ran
    assert solver.calls == []
    assert stage.energies == {"AB": None, "A": None, "B": None}
    assert stage.reason == "predicted rss_mb 20001.0 > budget 20000.0"
    assert stage.cost["predicted"] == {"rss_mb": BUDGET.rss_mb + 1.0, "wall_s": 3.0}
    assert stage.cost["measured"] == {}


def test_wall_criterion_uses_the_sum_of_the_three_predictions(water):
    half, quarter = BUDGET.wall_s / 2, BUDGET.wall_s / 4
    stage, solver = _stage(water, RecordingModel({8: 1.0, 4: 1.0}, {8: half, 4: quarter}))
    assert stage.ran and len(solver.calls) == 3

    stage, solver = _stage(water, RecordingModel({8: 1.0, 4: 1.0}, {8: half + 1.0, 4: quarter}))
    assert not stage.ran and solver.calls == []
    assert stage.reason == "predicted wall_s 7201.0 > budget 7200.0"


def test_both_exceeded_quantities_are_reported(water):
    model = RecordingModel({8: 1.0, 4: BUDGET.rss_mb + 1.0}, {8: BUDGET.wall_s, 4: 1.0})
    stage, _ = _stage(water, model)
    assert stage.reason == (
        "predicted rss_mb 20001.0 > budget 20000.0; predicted wall_s 7202.0 > budget 7200.0"
    )


def test_available_memory_below_the_budget_becomes_the_effective_limit(water):
    available = 15000.0
    model = RecordingModel({8: 1.0, 4: available}, {8: 1.0, 4: 1.0})
    stage, solver = _stage(water, model, available=available)
    assert stage.ran
    assert {call[4] for call in model.calls} == {available - BASE_MB}
    assert {call[2] for call in solver.calls} == {available - BASE_MB}
    assert stage.cost["effective_rss_mb"] == available

    stage, solver = _stage(
        water, RecordingModel({8: 1.0, 4: available + 1.0}, {8: 1.0, 4: 1.0}), available=available
    )
    assert not stage.ran
    assert stage.reason == "predicted rss_mb 15001.0 > budget 15000.0"


def test_run_records_the_estimates_and_the_measured_cost(water):
    model = RecordingModel({8: 2.0, 4: 3.0}, {8: 4.0, 4: 5.0})
    stage, _ = _stage(water, model)
    cost = stage.cost
    assert cost["model_id"] == "recording"
    assert cost["budget"] == {"rss_mb": BUDGET.rss_mb, "wall_s": BUDGET.wall_s}
    assert cost["max_memory_mb"] == BUDGET.rss_mb - BASE_MB
    assert cost["predicted"] == {"rss_mb": 3.0, "wall_s": 14.0}
    assert cost["estimates"]["AB"]["rss_mb"] == 2.0
    assert set(cost["measured"]) == {"AB", "A", "B"}
    assert cost["rss_before_mb"] > 0
    for measured in cost["measured"].values():
        assert measured["peak_rss_mb"] > 0
        assert measured["wall_s"] >= 0


@pytest.mark.parametrize("threads", [8, 1])
def test_threads_other_than_eight_add_a_warning(water, threads):
    model = RecordingModel({8: 1.0, 4: 1.0}, {8: 1.0, 4: 1.0})
    previous = lib.num_threads()
    try:
        lib.num_threads(threads)
        stage, _ = _stage(water, model)
    finally:
        lib.num_threads(previous)
    assert stage.cost["threads"] == threads
    assert bool(stage.warnings) is (threads != 8)
    if threads != 8:
        assert stage.warnings == (
            f"CCSD(T) wall-time model is calibrated at 8 threads; this run uses {threads}",
        )


@pytest.fixture(scope="module")
def water_monomer():
    return counterpoise(Molecule.from_xyz(_xyz(WATER_DIMER[:3])), basis="cc-pvdz")["M"]


def test_solver_gives_pyscf_the_memory_limit(water_monomer, monkeypatch):
    # The water monomer fits any limit, so only the value PySCF receives shows the limit.
    seen = []
    ao2mo = cc.ccsd.CCSD.ao2mo

    def recording_ao2mo(self, *args, **kwargs):
        seen.append(self.max_memory)
        return ao2mo(self, *args, **kwargs)

    monkeypatch.setattr(cc.ccsd.CCSD, "ao2mo", recording_ao2mo)
    frozen_core_ccsd_t(water_monomer, 1, max_memory_mb=1234.5)
    assert seen == [1234.5]


def test_unconverged_ccsd_is_an_error(water_monomer, monkeypatch):
    monkeypatch.setattr(cc.ccsd.CCSD, "max_cycle", 1)
    with pytest.raises(FragmentWorkflowError, match="CCSD of subsystem M did not converge"):
        frozen_core_ccsd_t(water_monomer, 1, max_memory_mb=4000.0)


# POC-C calibration points and the phenol...pyridine extrapolation, with the predictions
# printed in fit_output.txt: (nao, nocc, nvir, nfrozen, max_memory, RSS MB, wall s, in-core).
# The file prints RSS as integers and wall times with one decimal, the extrapolation as
# integers; the comparison allows half a unit of the last printed digit.
FIT_POINTS = {
    "S66-19 AB mm4000": (157, 22, 127, 8, 4000, 4000, 109.8, False),
    "S66-19 AB mm32000": (157, 22, 127, 8, 32000, 4033, 109.8, True),
    "S66-19 B mm32000": (157, 15, 136, 6, 32000, 3299, 56.9, True),
    "W A mm4000": (48, 4, 43, 1, 4000, 172, 1.1, True),
    "phenol-pyridine AB mm32000": (237, 33, 191, 13, 32000, 15862, 1660, True),
}


@pytest.mark.parametrize("point", FIT_POINTS.values(), ids=FIT_POINTS)
def test_cost_model_reproduces_the_fitted_predictions(point):
    nao, nocc, nvir, nfrozen, max_memory, rss, wall, incore = point
    estimate = estimate_ccsd_t_cost(nao, nocc, nvir, nfrozen, max_memory_mb=max_memory)
    assert round(estimate.rss_mb) == rss
    assert abs(estimate.wall_s - wall) <= (0.5 if isinstance(wall, int) else 0.05)
    assert estimate.cc_incore is incore
    assert estimate.model_id == MODEL_ID == "pyscf-2.14.0-rccsd_t-ccpvdz-v1"
    # The calibration covers nao <= 157; larger systems are extrapolations.
    assert estimate.extrapolated is (nao > 157)


# fit_output.txt: S66-19 counterpoise-corrected CCSD(T) interaction energy (kcal/mol).
S66_19_DELTA_E_CCSD_T_KCAL = -5.1150


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("Q2M3_FRAGMENT_CALIBRATION") != "1",
    reason="CCSD(T) cost calibration runs only with Q2M3_FRAGMENT_CALIBRATION=1",
)
def test_phenol_pyridine_calibration(record_property):
    """Predicted and measured CCSD(T) cost of the three counterpoise subsystems.

    The phenol...pyridine geometry is not available yet, so the S66 methanol...pyridine
    dimer stands in for it. The stage runs in this process after the counterpoise SCF, as in
    the workflow. ``rss_before_mb`` is the resident set before the stage, which the
    standalone-process model does not include.
    """
    molecule = Molecule.from_xyz(
        _xyz(METHANOL_PYRIDINE), fragments=(tuple(range(6)), tuple(range(6, 17)))
    )
    subsystems = counterpoise(molecule, basis="cc-pvdz")
    nfrozen = {name: chemcore(sub.mol) for name, sub in subsystems.items()}
    stage = run_ccsd_t_stage(subsystems, nfrozen, BUDGET)
    assert stage.ran
    energies = stage.energies
    delta_e = (energies["AB"] - energies["A"] - energies["B"]) * HARTREE_TO_KCAL_MOL
    summary = {"delta_e_ccsd_t_kcal": delta_e}
    for name, measured in stage.cost["measured"].items():
        predicted = stage.cost["estimates"][name]
        summary |= {
            f"{name}_rss_predicted_mb": predicted["rss_mb"],
            f"{name}_rss_measured_mb": measured["peak_rss_mb"],
            f"{name}_wall_predicted_s": predicted["wall_s"],
            f"{name}_wall_measured_s": measured["wall_s"],
        }
    summary["rss_before_mb"] = stage.cost["rss_before_mb"]
    summary["threads"] = stage.cost["threads"]
    for key, value in summary.items():
        record_property(key, float(value))
    print(json.dumps(stage.cost | {"delta_e_ccsd_t_kcal": delta_e}))
    assert abs(delta_e - S66_19_DELTA_E_CCSD_T_KCAL) <= 5e-5
