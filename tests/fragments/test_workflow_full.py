"""``run(mode="full")`` on the water dimer, with the SQD stage in a real child interpreter.

The reference energies come from PySCF CASCI on the same AVAS orbitals, computed here
without the fragment integrals.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest
from pyscf import cc, mcscf

from q2m3.constants import HARTREE_TO_KCAL_MOL
from q2m3.fragments import workflow
from q2m3.fragments.active_space import select_active_spaces
from q2m3.fragments.counterpoise import counterpoise
from q2m3.fragments.molecule import Molecule
from q2m3.fragments.references import CCSDTBudget
from q2m3.fragments.result import (
    QPE_ESTIMATOR,
    SQD_ESTIMATOR,
    FragmentInteractionResult,
    promote_sqd_energies,
)
from q2m3.fragments.sqd_stage import SQDJob, SubprocessSQDRunner
from q2m3.molecule import MoleculeConfig
from q2m3.sqd.config import LUCJConfig, ReferenceConfig
from q2m3.sqd.integrals import build_integrals
from tests.fragments.conftest import AVAS_SETTINGS, WATER_DIMER
from tests.fragments.test_result_validate import full_result, full_subsystem

pytestmark = [pytest.mark.sqd, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[2]
DIMER_FRAGMENTS = ((0, 1, 2), (3, 4, 5))
# Counterpoise-corrected CASCI interaction energy of the water dimer for the S6 labels,
# from the fragment-workflow prototype run (kcal/mol).
PROTOTYPE_DELTA_E_CASCI_KCAL = -3.5737713888900826
# Labels whose three active spaces are small enough for SQD to sample every determinant.
FULL_SUBSPACE_LABELS = (("1 H 1s", "4 H 1s"), 0.2)
# Seed and wall budget that differ from the defaults, so the run must pass them on.
SEED = 3
WALL_BUDGET_S = 901.0
STAGE_WARNING = "warning added by the wrapping runner"
# Fewer shots than the default still sample every determinant of these small spaces.
SHOTS = 50_000
WATER = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]


def _xyz(atoms) -> str:
    return f"{len(atoms)}\n\n" + "\n".join(f"{s} {x} {y} {z}" for s, (x, y, z) in atoms)


def _dimer() -> Molecule:
    return Molecule.from_xyz(_xyz(WATER_DIMER), fragments=DIMER_FRAGMENTS)


def _independent_casci(molecule: Molecule, labels, threshold) -> dict[str, float]:
    subsystems = counterpoise(molecule, basis="cc-pvdz")
    energies = {}
    for name, (record, mo) in select_active_spaces(subsystems, labels, threshold).items():
        casci = mcscf.CASCI(subsystems[name].scf, record.ncas, record.nelecas)
        casci.verbose = 0
        energies[name] = casci.kernel(mo_coeff=mo)[0]
    return energies


@pytest.fixture(scope="module")
def child_path():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PYTHONPATH", f"{REPO}:{REPO / 'src'}")
        yield


@pytest.fixture(scope="module")
def s6(child_path) -> FragmentInteractionResult:
    labels, threshold = AVAS_SETTINGS["S6"]
    reference = ReferenceConfig(wall_budget_s=WALL_BUDGET_S)
    return workflow.run(
        _dimer(), avas_labels=labels, avas_threshold=threshold, seed=SEED, reference=reference
    )


def test_s6_baselines_are_t0_casci_that_match_an_independent_casci(s6):
    labels, threshold = AVAS_SETTINGS["S6"]
    reference = _independent_casci(_dimer(), labels, threshold)
    for name, subsystem in s6.subsystems.items():
        assert (subsystem.sqd.baseline_tier, subsystem.sqd.baseline_method) == (
            "T0",
            "exact_casci",
        )
        assert abs(subsystem.e_casci - reference[name]) <= 1e-8
    delta_kcal = s6.delta_e["casci"] * HARTREE_TO_KCAL_MOL
    assert abs(delta_kcal - PROTOTYPE_DELTA_E_CASCI_KCAL) <= 1e-6
    assert s6.errors.algorithm == pytest.approx(s6.delta_e["sqd"] - s6.delta_e["casci"], abs=1e-12)
    assert "errors.algorithm" not in s6.null_reasons


def test_full_run_passes_its_settings_to_the_sqd_stage(s6):
    for subsystem in s6.subsystems.values():
        assert subsystem.sqd.seed == SEED
        assert subsystem.sqd.provenance["reference_config"]["wall_budget_s"] == WALL_BUDGET_S
    records = s6.provenance["threads"]["sqd_stage"]
    assert len(records) == 3
    for record in records:
        assert set(record.environment.values()) == {"1"}
        assert record.pyscf_threads == 1


def test_full_run_records_both_resource_estimates(s6):
    for subsystem in s6.subsystems.values():
        assert subsystem.resources["qpe"].estimator == QPE_ESTIMATOR
        assert subsystem.resources["qpe"].values["basis"] == "cc-pvdz"
        assert subsystem.resources["sqd"].estimator == SQD_ESTIMATOR
        assert subsystem.resources["sqd"].values["qubits"] == 2 * subsystem.active_space.ncas
        assert not {"resources.qpe", "resources.sqd"} & set(subsystem.null_reasons)


class _WarningRunner:
    def run(self, jobs):
        outcome = SubprocessSQDRunner().run(jobs)
        return dataclasses.replace(outcome, warnings=(*outcome.warnings, STAGE_WARNING))


def test_algorithm_error_vanishes_when_sqd_samples_the_whole_space(child_path, monkeypatch):
    monkeypatch.setattr(workflow, "_make_runner", _WarningRunner)
    labels, threshold = FULL_SUBSPACE_LABELS
    lucj = LUCJConfig(shots=SHOTS)
    result = workflow.run(_dimer(), avas_labels=labels, avas_threshold=threshold, lucj=lucj)
    for subsystem in result.subsystems.values():
        # Premise: the sampled subspace is the whole CI space of the active space.
        assert math.prod(subsystem.sqd.subspace_dims) == subsystem.sqd.full_ci_dim
        assert subsystem.sqd.provenance["requested_lucj"]["shots"] == SHOTS
        assert subsystem.resources["sqd"].values["shots"] == SHOTS
    assert abs(result.errors.algorithm) <= 1e-8
    assert STAGE_WARNING in result.warnings


def test_full_run_computes_the_frozen_core_ccsd_t_reference(s6):
    subsystems = counterpoise(_dimer(), basis="cc-pvdz")
    # One frozen 1s orbital per oxygen nucleus; ghost oxygens add none.
    for name, frozen in {"AB": 2, "A": 1, "B": 1}.items():
        solver = cc.CCSD(subsystems[name].scf, frozen=frozen)
        solver.kernel()
        expected = solver.e_tot + solver.ccsd_t()
        assert abs(s6.subsystems[name].e_ccsd_t - expected) <= 1e-7
        assert "e_ccsd_t" not in s6.subsystems[name].null_reasons
    assert s6.delta_e["ccsd_t"] is not None
    cost = s6.provenance["ccsd_t_cost"]
    assert cost["ran"] and set(cost["measured"]) == {"AB", "A", "B"}
    warned = any(w.startswith("CCSD(T) wall-time model is calibrated") for w in s6.warnings)
    assert warned is (cost["threads"] != 8)


def test_single_molecule_full_run_returns_its_subsystem_with_sqd(child_path):
    monomer = Molecule.from_xyz(_xyz(WATER_DIMER[:3]))
    # A 1 MB budget rejects CCSD(T); the run still completes the SQD stage.
    subsystem = workflow.run(
        monomer, avas_labels=("1 H 1s",), avas_threshold=0.2, ccsd_t_budget=CCSDTBudget(rss_mb=1.0)
    )
    assert subsystem.name == "M"
    assert subsystem.e_ccsd_t is None
    assert subsystem.null_reasons["e_ccsd_t"].startswith("predicted rss_mb ")
    assert subsystem.null_reasons["e_ccsd_t"].endswith(" > budget 1.0")
    assert subsystem.e_sqd == subsystem.sqd.sqd_energy
    assert subsystem.e_casci == subsystem.sqd.baseline_energy
    assert set(subsystem.resources) == {"qpe", "sqd"}


@pytest.fixture(scope="module")
def ten_orbital_job() -> SQDJob:
    config = MoleculeConfig("h2o", ["O", "H", "H"], WATER, 0, 10, 10, "6-31g")
    data = build_integrals(config, host_available_mb=4096)
    assert (data.norb, data.nelec) == (10, (5, 5))
    return SQDJob(data.h1, data.h2, data.e_core, data.norb, data.nelec, data.context)


@pytest.mark.parametrize(
    ("wall_budget_s", "tier"),
    # 600 s is below the T0 and T1 cost predictions of (10e,10o), so both are skipped.
    [(900.0, "T0"), (600.0, "T2")],
    ids=["default-budget", "small-budget"],
)
def test_ten_orbital_result_records_the_tier_that_ran(
    child_path, ten_orbital_job, wall_budget_s, tier
):
    job = dataclasses.replace(
        ten_orbital_job, reference=ReferenceConfig(wall_budget_s=wall_budget_s)
    )
    sqd = SubprocessSQDRunner().run([job]).results[0]
    assert sqd.baseline_tier == tier
    e_sqd, e_casci, reasons = promote_sqd_energies(sqd)
    subsystems = {}
    for name in ("AB", "A", "B"):
        subsystem = full_subsystem(name, sqd, e_sqd=e_sqd, e_casci=e_casci)
        reasons_all = dict(subsystem.null_reasons) | reasons
        subsystems[name] = dataclasses.replace(subsystem, null_reasons=reasons_all)
    result = full_result(subsystems).validate()
    if tier == "T0":
        assert (e_casci, reasons) == (sqd.baseline_energy, {})
        assert result.errors.algorithm is not None
        return
    assert (e_casci, reasons) == (None, {"e_casci": "baseline_tier=T2"})
    assert result.errors.algorithm is None
    assert "errors.algorithm" in result.null_reasons
    filled = dataclasses.replace(result, errors=dataclasses.replace(result.errors, algorithm=0.0))
    with pytest.raises(ValueError, match="errors.algorithm"):
        filled.validate()
