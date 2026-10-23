"""The one-call workflow in classical-only mode: HF and MP2 counterpoise energies, no SQD.

MP2 references are explicit PySCF runs with ghost atoms and a fixed frozen-core count:
one O 1s orbital per oxygen with a nucleus.
"""

from __future__ import annotations

import json
import math

import pyscf
import pytest
from pyscf import gto, lib, mp, scf

from q2m3.fragments import workflow
from q2m3.fragments.molecule import Molecule
from q2m3.fragments.result import FragmentInteractionResult, SubsystemResult
from q2m3.utils.io import save_json_results
from tests.fragments.conftest import WATER_DIMER

DIMER_FRAGMENTS = ((0, 1, 2), (3, 4, 5))
SUBSYSTEM_ATOMS = {"AB": range(6), "A": range(3), "B": range(3, 6)}
FROZEN_CORE = {"AB": 2, "A": 1, "B": 1}
LABELS = ("2 H 1s", "3 O 2p")
THRESHOLD = 0.2
# MP2 interaction energy of the same dimer from the prototype reference run, in Ha.
PROTOTYPE_DELTA_E_MP2 = -0.006271604103
CLASSICAL_THREADS = 2


def _xyz(atoms) -> str:
    lines = [str(len(atoms)), ""] + [f"{s} {x!r} {y!r} {z!r}" for s, (x, y, z) in atoms]
    return "\n".join(lines) + "\n"


def _explicit_energies(atoms, real, frozen) -> tuple[float, float]:
    """RHF and frozen-core MP2 energies; atoms outside ``real`` keep only their basis."""
    spec = [(s if i in real else f"ghost-{s}", xyz) for i, (s, xyz) in enumerate(atoms)]
    mf = scf.RHF(gto.M(atom=spec, basis="cc-pvdz", unit="Angstrom", verbose=0))
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf.e_tot, mp.MP2(mf, frozen=frozen).run().e_tot


class RecordingRunner:
    """SQD stage runner that records calls and fails if a job reaches it."""

    def __init__(self, calls: list) -> None:
        self.calls = calls

    def run(self, jobs):
        self.calls.append(("run", tuple(jobs)))
        raise AssertionError("classical-only mode must not run the SQD stage")


def _classical_run(water_dimer, **kwargs):
    calls: list = []

    def make_runner():
        calls.append(("make_runner",))
        return RecordingRunner(calls)

    molecule = Molecule.from_xyz(_xyz(water_dimer), fragments=DIMER_FRAGMENTS, **kwargs)
    # Two threads differ from the single-thread test environment, so the record is observable.
    with pytest.MonkeyPatch.context() as patch, lib.with_omp_threads(CLASSICAL_THREADS):
        patch.setattr(workflow, "_make_runner", make_runner)
        result = workflow.run(
            molecule, avas_labels=LABELS, avas_threshold=THRESHOLD, mode="classical_only", seed=7
        )
    return molecule, result, calls


@pytest.fixture(scope="module")
def classical():
    return _classical_run(WATER_DIMER)


def test_classical_only_never_builds_or_calls_the_sqd_runner(classical):
    _, result, calls = classical
    assert calls == []
    assert isinstance(result, FragmentInteractionResult)
    assert result.mode == "classical_only"
    for sub in result.subsystems.values():
        assert sub.sqd is None and "sqd" in sub.null_reasons
    assert result.delta_e["hf"] is not None and result.delta_e["mp2"] is not None
    for layer in ("sqd", "casci", "ccsd_t"):
        assert result.delta_e[layer] is None
        # The interaction-energy reason cites the subsystem field and its own reason.
        subsystem_reason = result.subsystems["AB"].null_reasons[f"e_{layer}"]
        assert f"e_{layer} is null for AB, A, B" in result.null_reasons[f"delta_e.{layer}"]
        assert subsystem_reason in result.null_reasons[f"delta_e.{layer}"]


def test_classical_energies_match_explicit_pyscf(classical, water_dimer):
    _, result, _ = classical
    explicit = {
        name: _explicit_energies(water_dimer, set(atoms), FROZEN_CORE[name])
        for name, atoms in SUBSYSTEM_ATOMS.items()
    }
    for name, (e_hf, e_mp2) in explicit.items():
        sub = result.subsystems[name]
        assert sub.real_atoms == tuple(SUBSYSTEM_ATOMS[name])
        assert sub.ghost_atoms == tuple(i for i in range(6) if i not in SUBSYSTEM_ATOMS[name])
        assert sub.n_ao == 48
        assert result.subsystems[name].e_hf == pytest.approx(e_hf, abs=1e-9)
        assert result.subsystems[name].e_mp2 == pytest.approx(e_mp2, abs=1e-9)
    hf = explicit["AB"][0] - explicit["A"][0] - explicit["B"][0]
    mp2 = explicit["AB"][1] - explicit["A"][1] - explicit["B"][1]
    assert result.delta_e["hf"] == pytest.approx(hf, abs=1e-9)
    assert result.delta_e["mp2"] == pytest.approx(mp2, abs=1e-9)
    assert result.delta_e["mp2"] == pytest.approx(PROTOTYPE_DELTA_E_MP2, abs=1e-9)
    assert result.provenance["frozen_core"] == FROZEN_CORE


def test_provenance_records_the_classical_settings(classical):
    molecule, result, _ = classical
    provenance = result.provenance
    assert set(provenance) == {
        "geometry_sha256",
        "basis",
        "cart",
        "avas_labels",
        "avas_threshold",
        "seed",
        "scf_conv_tol",
        "frozen_core",
        "versions",
        "threads",
    }
    assert provenance["geometry_sha256"] == molecule.geometry_sha256()
    assert provenance["basis"] == result.basis == "cc-pvdz"
    assert provenance["cart"] is False
    assert provenance["avas_labels"] == LABELS
    assert provenance["avas_threshold"] == THRESHOLD
    assert provenance["seed"] == 7
    assert provenance["scf_conv_tol"] == 1e-10
    versions = provenance["versions"]
    assert set(versions) == {"q2m3", "pyscf", "numpy", "ffsim", "qiskit-addon-sqd", "pennylane"}
    assert versions["pyscf"] == pyscf.__version__
    assert provenance["threads"] == {"classical_stage": CLASSICAL_THREADS}


def test_fragment_id_is_the_molecule_name_or_the_geometry_digest(classical, water_dimer):
    molecule, result, _ = classical
    assert result.fragment_id == molecule.geometry_sha256()
    _, named, _ = _classical_run(water_dimer, name="water-dimer")
    assert named.fragment_id == "water-dimer"


def test_result_serializes_to_strict_json(classical, tmp_path):
    _, result, _ = classical
    path = tmp_path / "result.json"
    save_json_results(result, path)
    text = path.read_text()
    assert "NaN" not in text
    loaded = json.loads(text)
    assert loaded["schema_version"] == "fragments.result.v1"
    assert loaded["subsystems"]["AB"]["sqd"] is None
    assert loaded["delta_e"]["casci"] is None
    assert math.isclose(loaded["delta_e"]["mp2"], result.delta_e["mp2"], abs_tol=0.0)


def test_single_molecule_returns_its_subsystem_result(water_dimer):
    monomer = water_dimer[:3]
    result = workflow.run(
        Molecule.from_xyz(_xyz(monomer)),
        avas_labels=("O 2p",),
        avas_threshold=THRESHOLD,
        mode="classical_only",
    )
    e_hf, e_mp2 = _explicit_energies(monomer, set(range(3)), 1)
    assert isinstance(result, SubsystemResult)
    assert result.name == "M" and result.ghost_atoms == ()
    assert result.e_hf == pytest.approx(e_hf, abs=1e-9)
    assert result.e_mp2 == pytest.approx(e_mp2, abs=1e-9)


def _no_scf(*args, **kwargs):
    raise AssertionError("the SCF stage must not start")


def _dimer(water_dimer) -> Molecule:
    return Molecule.from_xyz(_xyz(water_dimer), fragments=DIMER_FRAGMENTS)


def test_full_mode_without_the_sqd_extra_fails_before_any_scf(monkeypatch, water_dimer):
    monkeypatch.setattr(workflow, "extra_available", lambda extra: extra != "sqd")
    monkeypatch.setattr(workflow, "counterpoise", _no_scf)
    with pytest.raises(ImportError, match="uv sync --extra sqd"):
        workflow.run(_dimer(water_dimer), avas_labels=LABELS, avas_threshold=THRESHOLD)


def test_full_mode_with_the_sqd_extra_is_not_available_yet(monkeypatch, water_dimer):
    monkeypatch.setattr(workflow, "extra_available", lambda extra: True)
    monkeypatch.setattr(workflow, "counterpoise", _no_scf)
    with pytest.raises(NotImplementedError):
        workflow.run(_dimer(water_dimer), avas_labels=LABELS, avas_threshold=THRESHOLD)


def test_unknown_mode_is_rejected(monkeypatch, water_dimer):
    monkeypatch.setattr(workflow, "counterpoise", _no_scf)
    with pytest.raises(ValueError, match="mode"):
        workflow.run(_dimer(water_dimer), avas_labels=LABELS, avas_threshold=THRESHOLD, mode="fast")


def test_imported_references_are_not_supported_yet(monkeypatch, water_dimer):
    monkeypatch.setattr(workflow, "counterpoise", _no_scf)
    with pytest.raises(NotImplementedError):
        workflow.run(
            _dimer(water_dimer),
            avas_labels=LABELS,
            avas_threshold=THRESHOLD,
            mode="classical_only",
            references=("reference.csv",),
        )


@pytest.mark.parametrize(
    ("labels", "threshold"),
    [("2 H 1s", THRESHOLD), ((2,), THRESHOLD), (LABELS, "0.2"), (LABELS, True)],
    ids=["bare-string", "non-string-label", "string-threshold", "bool-threshold"],
)
def test_avas_arguments_are_type_checked(monkeypatch, water_dimer, labels, threshold):
    monkeypatch.setattr(workflow, "counterpoise", _no_scf)
    with pytest.raises(TypeError, match="avas_"):
        workflow.run(
            _dimer(water_dimer),
            avas_labels=labels,
            avas_threshold=threshold,
            mode="classical_only",
        )
