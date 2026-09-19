"""Fixed vacuum-frame water embedding through both public SQD entry points."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import fields, replace
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.sqd.test_diagonalize import projected_matrix, samples_for

pytestmark = pytest.mark.sqd
SYMBOLS = ["O", "H", "H"]
COORDS = np.array([[0.0, 0.0, 0.0], [0.13, 0.1, 0.96], [0.88, -0.12, -0.29]])
CHARGES = np.array([0.31, -0.19])
MM_COORDS = np.array([[2.3, 0.4, 1.2], [-1.7, 2.1, 0.3]])
MODES = ("diagonal", "full_oneelectron")
ENERGIES = (
    "sqd_energy",
    "hf_energy",
    "baseline_energy",
    "iso_active_space_ccsd_energy",
    "iso_ndet_sci_energy",
    "iso_ndet_random_energy",
)


def close(actual, expected, tolerance=1e-10):
    assert actual == pytest.approx(expected, abs=tolerance, rel=0)


def example_module():
    path = Path(__file__).resolve().parents[2] / "examples/qmmm/h2o_sqd_validation.py"
    spec = importlib.util.spec_from_file_location("water_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options():
    from q2m3.sqd.config import LUCJConfig, ReferenceConfig

    return dict(
        lucj=LUCJConfig(n_reps=1, shots=4096),
        reference=ReferenceConfig(wall_budget_s=120, rss_budget_mb=8192),
        seed=7,
    )


def low(data, seed_data, **kwargs):
    from q2m3.sqd.orchestrator import run_sqd_from_integrals

    return run_sqd_from_integrals(
        data.h1,
        data.h2,
        data.e_core,
        norb=4,
        nelec=(2, 2),
        context=data.context,
        seed_data=seed_data,
        **(options() | kwargs),
    )


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    """Capture actual worker inputs; reuse that one vacuum MO for both modes."""
    from q2m3.interfaces.fixed_mo_embedding import (
        FixedMOEmbeddingDiagnostics,
        FixedMOEmbeddingResult,
    )
    from q2m3.sqd import integrals, orchestrator
    from q2m3.sqd.config import CCSDSeed, IntegralContext
    from q2m3.sqd.integrals import IntegralData
    from q2m3.utils.io import save_json_results

    out = tmp_path_factory.mktemp("water-embedding")
    original_helper = integrals.build_fixed_mo_embedding_integrals
    original_build = orchestrator.build_integrals
    original_seed = orchestrator.build_ccsd_seed
    fixed_helper = None
    rows = {}
    # Full is unpatched numerically. Diagonal consumes exactly its helper snapshot.
    for mode in ("full_oneelectron", "diagonal"):

        def capture_helper(*args, _fixed=fixed_helper, _mode=mode, **kwargs):
            value = original_helper(*args, **kwargs) if _fixed is None else _fixed
            np.testing.assert_array_equal(kwargs["mm_charges"], CHARGES)
            np.testing.assert_array_equal(kwargs["mm_coords"], MM_COORDS)
            save_json_results(value, out / f"{_mode}-helper.json")
            return value

        def capture_build(*args, _mode=mode, **kwargs):
            value = original_build(*args, **kwargs)
            save_json_results(value, out / f"{_mode}-integrals.json")
            return value

        def capture_seed(*args, _mode=mode, **kwargs):
            value = original_seed(*args, **kwargs)
            save_json_results(value, out / f"{_mode}-seed.json")
            return value

        with (
            patch.object(integrals, "build_fixed_mo_embedding_integrals", capture_helper),
            patch.object(orchestrator, "build_integrals", capture_build),
            patch.object(orchestrator, "build_ccsd_seed", capture_seed),
        ):
            high = orchestrator.run_sqd(
                SYMBOLS,
                COORDS,
                active_electrons=4,
                active_orbitals=4,
                mm_charges=CHARGES,
                mm_coords=MM_COORDS,
                embedding_mode=mode,
                verbose=False,
                **options(),
            )
        raw = json.loads((out / f"{mode}-integrals.json").read_text())
        raw["context"] = IntegralContext(**raw["context"])
        data = IntegralData(**raw)
        raw_seed = json.loads((out / f"{mode}-seed.json").read_text())
        seed_data = CCSDSeed(**raw_seed)
        raw_helper = json.loads((out / f"{mode}-helper.json").read_text())
        raw_helper["diagnostics"] = FixedMOEmbeddingDiagnostics(**raw_helper["diagnostics"])
        for name in (
            "one_electron_vacuum",
            "two_electron",
            "delta_h_active",
            "delta_h_diag",
            "delta_h_offdiag",
            "mo_coeff",
        ):
            raw_helper[name] = np.array(raw_helper[name])
        raw_helper["active_indices"] = tuple(raw_helper["active_indices"])
        fixed_helper = FixedMOEmbeddingResult(**raw_helper)
        delta = fixed_helper.delta_h_diag if mode == "diagonal" else fixed_helper.delta_h_active
        # Explicit helper -> chemist three-tuple -> public low-level entry.
        replay = replace(
            data,
            h1=fixed_helper.one_electron_vacuum + delta,
            h2=fixed_helper.two_electron.transpose(0, 3, 1, 2),
            e_core=fixed_helper.vacuum_core_constant + fixed_helper.delta_core_constant,
        )
        result = low(replay, seed_data)
        save_json_results(high, out / f"{mode}-high.json")
        save_json_results(result, out / f"{mode}-low.json")
        rows[mode] = dict(
            data=data, seed=seed_data, helper=fixed_helper, high=high, low=result, out=out
        )
    return rows


@pytest.mark.parametrize("mode", MODES)
def test_helper_geometry_low_and_complete_public_json(chains, mode, tmp_path):
    from q2m3.sqd.result import SQDResult
    from q2m3.utils.io import save_json_results

    row = chains[mode]
    high, result, data = row["high"], row["low"], row["data"]
    for key in ENERGIES:
        close(getattr(high, key), getattr(result, key))
    assert high.subspace_dims == result.subspace_dims
    assert high.unique_dets_vs_shots == result.unique_dets_vs_shots
    assert data.context.active_indices == (3, 4, 5, 6)
    assert data.context.n_core_orbitals == 3
    assert abs(data.context.vacuum_core_constant) > 10
    assert high.hf_reference_kind == "fixed_frame_determinant"
    assert high.embedding_mode == mode
    assert high.baseline_tier == "T0"
    assert high.fixed_mo and high.two_electron_tensor_fixed
    path = tmp_path / "result.json"
    save_json_results(high, path)
    payload = json.loads(path.read_text())
    assert set(payload) == {f.name for f in fields(SQDResult)}
    assert len(payload) == 40
    source = payload["provenance"]["context"]["source"]
    assert source["mm_charges"] == CHARGES.tolist()
    assert source["mm_coords_angstrom"] == MM_COORDS.tolist()
    assert source["frame_source"] == "vacuum_rhf"
    assert payload["provenance"]["ccsd_seed"]["frame_id"] == data.context.frame_id
    assert payload["seed"] == 7
    assert payload["null_reasons"] == dict(high.null_reasons)
    assert payload["warnings"] == list(high.warnings)


@pytest.mark.parametrize("mode", MODES)
def test_actual_sparse_subspaces_and_full_fixture_against_independent_ci(chains, mode):
    from q2m3.sqd.diagonalize import diagonalize_samples

    data, result = chains[mode]["data"], chains[mode]["high"]
    assert 1 < result.subspace_dim < result.full_ci_dim == 36
    for arm, key in (
        ("sqd", "sqd_energy"),
        ("sci", "iso_ndet_sci_energy"),
        ("random", "iso_ndet_random_energy"),
    ):
        strings = result.diagnostics["comparison_ci_strings"][arm]
        matrix = projected_matrix(data.h1, data.h2, data.e_core, strings)
        close(getattr(result, key), np.linalg.eigvalsh(matrix)[0])
    strings = np.array(sorted(sum(1 << i for i in c) for c in combinations(range(4), 2)))
    oracle = np.linalg.eigvalsh(
        projected_matrix(data.h1, data.h2, data.e_core, (strings, strings))
    )[0]
    close(result.baseline_energy, oracle)
    full = diagonalize_samples(
        data.h1,
        data.h2,
        data.e_core,
        samples_for((strings, strings)),
        norb=4,
        nelec=(2, 2),
        host_available_mb=8192,
        seed=7,
    )
    assert full.subspace_dims == (6, 6)
    close(full.energy, result.baseline_energy)
    close(result.delta_mHa, 1000 * (result.sqd_energy - oracle), tolerance=1e-7)
    assert result.delta_mHa > 0  # Honest sparse-space error, no artificial exact fixture.
    from q2m3.utils.io import save_json_results

    save_json_results(
        {
            "full_fixture_energy_ha": full.energy,
            "independent_full_energy_ha": oracle,
            "full_fixture_t0_error_ha": abs(full.energy - result.baseline_energy),
            "t0_oracle_error_ha": abs(result.baseline_energy - oracle),
            "sparse_delta_mHa": result.delta_mHa,
        },
        chains[mode]["out"] / f"{mode}-oracle.json",
    )


def test_modes_use_one_vacuum_frame_and_nonzero_offdiagonal(chains):
    diagonal, full = (chains[m]["data"] for m in MODES)
    helper = chains["full_oneelectron"]["helper"]
    assert diagonal.context.frame_id == full.context.frame_id
    np.testing.assert_array_equal(diagonal.mo_coeff, full.mo_coeff)
    np.testing.assert_array_equal(diagonal.h2, full.h2)
    np.testing.assert_allclose(full.h1 - diagonal.h1, helper.delta_h_offdiag, atol=1e-12, rtol=0)
    assert np.linalg.norm(helper.delta_h_offdiag) > 1e-5
    assert (
        abs(chains[MODES[0]]["high"].baseline_energy - chains[MODES[1]]["high"].baseline_energy)
        > 1e-6
    )


@pytest.mark.parametrize("mode", MODES)
def test_nonempty_zero_charges_recover_same_frame_vacuum(chains, mode):
    from q2m3.molecule import MoleculeConfig
    from q2m3.sqd.ansatz import build_ccsd_seed
    from q2m3.sqd.integrals import build_integrals

    helper = chains[mode]["helper"]
    zero = build_integrals(
        MoleculeConfig("water", SYMBOLS, COORDS.tolist(), 0, 4, 4),
        mm_charges=np.zeros(2),
        mm_coords=MM_COORDS,
        embedding_mode=mode,
        host_available_mb=8192,
    )
    np.testing.assert_array_equal(zero.mo_coeff, helper.mo_coeff)
    assert zero.context.frame_id == chains[mode]["data"].context.frame_id
    np.testing.assert_allclose(zero.h1, helper.one_electron_vacuum, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        zero.h2, helper.two_electron.transpose(0, 3, 1, 2), atol=1e-12, rtol=0
    )
    close(zero.e_core, helper.vacuum_core_constant)
    close(zero.context.delta_core_constant, 0)
    vacuum = build_integrals(
        MoleculeConfig("water", SYMBOLS, COORDS.tolist(), 0, 4, 4), host_available_mb=8192
    )
    np.testing.assert_array_equal(vacuum.mo_coeff, zero.mo_coeff)
    np.testing.assert_allclose(vacuum.h1, zero.h1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(vacuum.h2, zero.h2, atol=1e-12, rtol=0)
    close(vacuum.e_core, zero.e_core)
    seed_data = build_ccsd_seed(zero, host_available_mb=8192)
    vacuum_seed = build_ccsd_seed(vacuum, host_available_mb=8192)
    zero_result, vacuum_result = low(zero, seed_data), low(vacuum, vacuum_seed)
    for key in ENERGIES:
        close(getattr(zero_result, key), getattr(vacuum_result, key))
    from q2m3.utils.io import save_json_results

    save_json_results(
        {"zero": zero_result, "vacuum": vacuum_result, "zero_integrals": zero},
        chains[mode]["out"] / f"{mode}-zero.json",
    )


@pytest.mark.parametrize("mode", MODES)
def test_core_shift_moves_all_total_energies_not_deltas(chains, mode):
    from q2m3.sqd.integrals import hamiltonian_id

    data, seed_data, result = (chains[mode][k] for k in ("data", "seed", "low"))
    shift = 0.375
    identifier = hamiltonian_id(data.h1, data.h2, data.e_core + shift, norb=4, nelec=(2, 2))
    context = replace(
        data.context,
        vacuum_core_constant=data.context.vacuum_core_constant + shift,
        hamiltonian_id=identifier,
    )
    shifted_data = replace(
        data, e_core=data.e_core + shift, hf_energy=data.hf_energy + shift, context=context
    )
    shifted_seed = replace(
        seed_data,
        hf_energy=seed_data.hf_energy + shift,
        ccsd_energy=seed_data.ccsd_energy + shift,
        hamiltonian_id=identifier,
    )
    shifted = low(shifted_data, shifted_seed)
    for key in ENERGIES:
        close(getattr(shifted, key), getattr(result, key) + shift)
    for key in ("delta_mHa", "delta_vs_sci_mHa"):
        close(getattr(shifted, key), getattr(result, key), tolerance=1e-7)
    assert shifted.subspace_dims == result.subspace_dims
    assert shifted.unique_dets_vs_shots == result.unique_dets_vs_shots
    assert shifted.provenance["context"]["hamiltonian_id"] == identifier
    assert identifier != data.context.hamiltonian_id
    from q2m3.utils.io import save_json_results

    save_json_results(
        {"shift_ha": shift, "shifted": shifted, "integrals": shifted_data, "seed": shifted_seed},
        chains[mode]["out"] / f"{mode}-shift.json",
    )


def test_relaxed_mm_reference_is_not_the_fixed_vacuum_frame(chains):
    from pyscf import ao2mo, gto, mcscf, qmmm, scf

    data, result = chains["full_oneelectron"]["data"], chains["full_oneelectron"]["high"]
    mol = gto.M(
        atom=list(zip(SYMBOLS, COORDS.tolist(), strict=True)),
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = qmmm.mm_charge(scf.RHF(mol), MM_COORDS, CHARGES, unit="Angstrom").run()
    assert mf.converged
    cas = mcscf.CASCI(mf, 4, (2, 2))
    one, core = cas.get_h1eff(mo_coeff=data.mo_coeff)
    two = ao2mo.restore(1, cas.get_h2eff(mo_coeff=data.mo_coeff), 4)
    np.testing.assert_allclose(data.h1, one, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data.h2, two, atol=1e-12, rtol=0)
    close(data.e_core, core)
    fixed = float(cas.kernel(mo_coeff=data.mo_coeff)[0])
    relaxed = float(cas.kernel(mo_coeff=mf.mo_coeff)[0])
    close(result.baseline_energy, fixed)
    assert abs(relaxed - fixed) > 1e-6
    with pytest.raises(AssertionError):
        close(result.baseline_energy, relaxed)
    from q2m3.utils.io import save_json_results

    save_json_results(
        {
            "fixed_casci_ha": fixed,
            "relaxed_casci_ha": relaxed,
            "relaxed_minus_fixed_ha": relaxed - fixed,
        },
        chains["full_oneelectron"]["out"] / "relaxed-counterexample.json",
    )


def test_same_shape_wrong_frame_seed_rejected(chains):
    from q2m3.sqd.exceptions import ProvenanceMismatchError

    data, seed_data = (chains["full_oneelectron"][k] for k in ("data", "seed"))
    with pytest.raises(ProvenanceMismatchError):
        low(data, replace(seed_data, frame_id="another-orbital-frame"))
    # Relabelled amplitudes in another virtual gauge must fail physical reception too.
    signs = np.array([-1.0, 1.0])
    bad_seed = replace(
        seed_data,
        t1=seed_data.t1 * signs,
        t2=seed_data.t2 * signs[None, None, :, None] * signs[None, None, None, :],
    )
    with pytest.raises(ProvenanceMismatchError):
        low(data, bad_seed)


def test_constant_and_mode_faults_are_detectable(chains):
    row = chains["full_oneelectron"]
    data, helper, result = (row[k] for k in ("data", "helper", "high"))
    strings = result.diagnostics["comparison_ci_strings"]["sqd"]
    oracle = float(np.linalg.eigvalsh(projected_matrix(data.h1, data.h2, data.e_core, strings))[0])
    for wrong in (oracle - helper.vacuum_core_constant, oracle + helper.delta_nuclear_mm):
        assert abs(wrong - result.sqd_energy) > 1e-3
        with pytest.raises(AssertionError):
            close(wrong, result.sqd_energy)
    wrong_h1 = data.h1 - helper.delta_h_offdiag
    strings = (np.array([3, 5, 6, 9, 10, 12]),) * 2
    wrong = np.linalg.eigvalsh(projected_matrix(wrong_h1, data.h2, data.e_core, strings))[0]
    with pytest.raises(AssertionError):
        close(wrong, result.baseline_energy)


def test_example_complete_downstream_mapping(tmp_path):
    from q2m3.sqd.result import SQDResult

    report = example_module().run_validation(tmp_path, shots=4096, seed=7)
    assert report["active_space"] == [4, 4]
    assert report["system_qubits"] == 8
    for mode in MODES:
        payload = json.loads((tmp_path / f"{mode}.json").read_text())
        consumer = json.loads((tmp_path / f"{mode}-downstream.json").read_text())
        assert set(payload) == {f.name for f in fields(SQDResult)}
        for key, value in payload.items():
            assert consumer["sqd"][key] == value
        assert payload["embedding_mode"] == mode
        assert payload["delta_mHa"] > 0
        assert payload["provenance"]["context"]["source"]["mm_charges"] == CHARGES.tolist()
        assert report["runs"][mode]["delta_mHa"] == payload["delta_mHa"]
    assert report["capabilities"]["mc_loop"] is False
    assert report["capabilities"]["rdm_feedback"] is False
    assert report["capabilities"]["self_consistent_mm_polarization"] is False


@pytest.mark.parametrize("mode", MODES)
def test_geometry_assembly_against_vacuum_mo_ao_oracle(mode):
    """Independent AO/CASCI reference detects altered constants, modes and MOs."""
    from pyscf import ao2mo, gto, mcscf, qmmm, scf

    from q2m3.molecule import MoleculeConfig
    from q2m3.sqd.integrals import build_integrals

    mol = gto.M(
        atom=list(zip(SYMBOLS, COORDS.tolist(), strict=True)),
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    vacuum = scf.RHF(mol).run()
    assert vacuum.converged
    mo = vacuum.mo_coeff.copy()
    for column in mo.T:
        if column[np.argmax(np.abs(column))] < 0:
            column *= -1
    embedded = qmmm.mm_charge(scf.RHF(mol), MM_COORDS, CHARGES, unit="Angstrom")
    cas = mcscf.CASCI(vacuum, 4, (2, 2))
    one, core = cas.get_h1eff(mo_coeff=mo)
    two = ao2mo.restore(1, cas.get_h2eff(mo_coeff=mo), 4)
    delta = mo.T @ (embedded.get_hcore() - vacuum.get_hcore()) @ mo
    delta_core = 2 * np.trace(delta[:3, :3]) + embedded.energy_nuc() - vacuum.energy_nuc()
    active_delta = delta[3:, 3:]
    expected = one + (np.diag(np.diag(active_delta)) if mode == "diagonal" else active_delta)
    data = build_integrals(
        MoleculeConfig("water", SYMBOLS, COORDS.tolist(), 0, 4, 4),
        mm_charges=CHARGES,
        mm_coords=MM_COORDS,
        embedding_mode=mode,
        host_available_mb=8192,
    )
    np.testing.assert_allclose(data.mo_coeff, mo, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data.h1, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data.h2, two, atol=1e-12, rtol=0)
    close(data.e_core, core + delta_core)
