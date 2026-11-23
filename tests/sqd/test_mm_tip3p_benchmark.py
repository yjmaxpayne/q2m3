"""Static benchmark of both SQD embedding modes on the registered TIP3P fixtures.

Every energy is checked against independent PySCF CASCI or a same-subspace
projection. The real SQD runs share one subprocess per module.
"""

from __future__ import annotations

import copy
import functools
import importlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from examples.sqd._inputs import load_mm_tip3p_fixtures
from q2m3.constants import BANNED_TERMS
from tests.sqd._mm_oracles import (
    independent_casci,
    independent_offdiag_delta_h1,
    mm_embedding,
)
from tests.sqd.test_diagonalize import projected_matrix

ROOT = Path(__file__).resolve().parents[2]
MODES = ("diagonal", "full_oneelectron")
# Short ids keep fixture names such as "ac2_..." out of the -k selection.
SHORT_IDS = {
    "h2o_1tip3p_4e4o": "h2o_w1_4e4o",
    "h2o_3tip3p_4e4o": "h2o_w3_4e4o",
    "h2o_3tip3p_8e6o": "h2o_w3_8e6o",
    "h3o_plus_3tip3p_4e4o": "h3o_w3_4e4o",
    "h3o_plus_8tip3p_4e4o": "h3o_w8_4e4o",
    "ac2_h3o_plus_3tip3p": "h3o_w3_contact",
}
CONTACT = "ac2_h3o_plus_3tip3p"
TOL_HA = 1e-10
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

FIXTURES = {fixture["name"]: fixture for fixture in load_mm_tip3p_fixtures()}
POINTS = [
    pytest.param(name, mode, id=f"{SHORT_IDS[name]}-{mode}") for name in SHORT_IDS for mode in MODES
]
real_sqd = [
    pytest.mark.slow,
    pytest.mark.sqd,
    pytest.mark.skipif(
        not (sys.platform.startswith("linux") and platform.machine() == "x86_64"),
        reason="real SQD runs only on Linux x86_64",
    ),
]


def _bench():
    return importlib.import_module("examples.sqd.mm_tip3p_benchmark")


def _apply(marks):
    def decorate(func):
        for mark in reversed(marks):
            func = mark(func)
        return func

    return decorate


def _casci_args(name, mm_coords=None, mm_charges=None):
    fixture = FIXTURES[name]
    if mm_coords is None:
        mm_coords, mm_charges = mm_embedding(fixture)
    return (
        fixture["qm_symbols"],
        fixture["qm_coords"],
        fixture["charge"],
        mm_coords,
        mm_charges,
        fixture["active_orbitals"],
        fixture["active_electrons"],
    )


@functools.cache
def _independent(name, mode):
    return independent_casci(*_casci_args(name), mode=mode)


def _independent_shift(name, mm_coords=None, mm_charges=None):
    args = _casci_args(name, mm_coords, mm_charges)
    return independent_casci(*args, mode="full_oneelectron") - independent_casci(
        *args, mode="diagonal"
    )


@pytest.fixture(scope="module")
def benchmark_run(tmp_path_factory):
    """Run every fixture in both modes once, in a fresh interpreter."""
    out = tmp_path_factory.mktemp("mm_tip3p_benchmark") / "out"
    env = {**os.environ, **{name: "1" for name in THREAD_VARS}}
    command = [sys.executable, "-m", "tests.sqd._mm_benchmark_probe", "--output", str(out)]
    proc = subprocess.run(
        [*command, "--fixtures", *SHORT_IDS],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        pytest.fail(f"benchmark probe failed (rc={proc.returncode}):\n{proc.stderr[-4000:]}")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    records = {(record["fixture"], record["mode"]): record for record in report["records"]}
    return out, report["records"], records


def _capture(out, name, mode):
    with np.load(out / "captures" / f"{name}__{mode}.npz") as data:
        return {key: data[key] for key in ("h1", "h2", "e_core", "alpha", "beta")}


@_apply(real_sqd)
@pytest.mark.parametrize(("name", "mode"), POINTS)
def test_ac1_t0_matches_independent_casci(benchmark_run, name, mode):
    record = benchmark_run[2][(name, mode)]
    assert record["baseline_tier"] == "T0"
    assert abs(record["baseline_energy"] - _independent(name, mode)) <= TOL_HA


@_apply(real_sqd)
@pytest.mark.parametrize(("name", "mode"), POINTS)
def test_ac1_sqd_matches_same_subspace_oracle(benchmark_run, name, mode):
    out, _, records = benchmark_run
    record = records[(name, mode)]
    assert record["status"] == "completed"
    captured = _capture(out, name, mode)
    strings = (captured["alpha"], captured["beta"])
    assert len(strings[0]) * len(strings[1]) == record["subspace_dim"]
    matrix = projected_matrix(captured["h1"], captured["h2"], float(captured["e_core"]), strings)
    assert abs(record["sqd_energy"] - np.linalg.eigvalsh(matrix)[0]) <= TOL_HA
    # The capture comes from the code under test; full subspaces also meet independent CASCI.
    if record["subspace_dim"] == record["full_ci_dim"]:
        assert abs(record["sqd_energy"] - _independent(name, mode)) <= TOL_HA


@_apply(real_sqd)
def test_ac1_outputs_gap_and_unique_determinant_curves(benchmark_run):
    _, records, by_point = benchmark_run
    assert len(records) == len(POINTS)
    assert set(by_point) == {(name, mode) for name in SHORT_IDS for mode in MODES}
    for record in records:
        assert isinstance(record["delta_mHa"], float) and math.isfinite(record["delta_mHa"])
        assert record["unique_dets_vs_shots"]


def _contact_integrals():
    from q2m3.molecule import MoleculeConfig
    from q2m3.sqd.integrals import build_integrals

    fixture = FIXTURES[CONTACT]
    molecule = MoleculeConfig(
        name=fixture["name"],
        symbols=fixture["qm_symbols"],
        coords=fixture["qm_coords"],
        charge=fixture["charge"],
        active_electrons=fixture["active_electrons"],
        active_orbitals=fixture["active_orbitals"],
        basis=fixture["basis"],
    )
    mm_coords, mm_charges = mm_embedding(fixture)
    return {
        mode: build_integrals(
            molecule,
            mm_charges=mm_charges,
            mm_coords=mm_coords,
            embedding_mode=mode,
            host_available_mb=4096.0,
        )
        for mode in MODES
    }


def _assert_offdiag_matches(data, independent):
    delta = data["full_oneelectron"].h1 - data["diagonal"].h1
    offdiag = delta - np.diag(np.diag(delta))
    assert np.max(np.abs(offdiag - independent)) <= TOL_HA
    assert np.linalg.norm(independent) >= 1e-6
    contexts = {mode: data[mode].context for mode in MODES}
    assert contexts["diagonal"].hamiltonian_id != contexts["full_oneelectron"].hamiltonian_id
    assert contexts["diagonal"].frame_id == contexts["full_oneelectron"].frame_id


@pytest.fixture(scope="module")
def contact_offdiag():
    return independent_offdiag_delta_h1(*_casci_args(CONTACT))


@pytest.mark.sqd
def test_ac2_a_offdiagonal_block_matches_independent_h1eff(contact_offdiag):
    _assert_offdiag_matches(_contact_integrals(), contact_offdiag)


@pytest.mark.sqd
def test_ac2_a_comparator_rejects_swapped_modes(contact_offdiag):
    data = _contact_integrals()
    swapped = {"diagonal": data["full_oneelectron"], "full_oneelectron": data["diagonal"]}
    with pytest.raises(AssertionError):
        _assert_offdiag_matches(swapped, contact_offdiag)


@pytest.mark.slow
def test_ac2_b_t0_energy_shift_is_significant():
    shift = _independent_shift(CONTACT)
    assert abs(shift) >= 1e-3
    assert abs(shift * 1e3 - FIXTURES[CONTACT]["measured_t0_dE_mHa"]) <= 1e-6


@pytest.mark.slow
def test_ac2_b_negative_controls_stay_below_threshold():
    mm_coords, mm_charges = mm_embedding(FIXTURES[CONTACT])
    assert abs(_independent_shift(CONTACT, mm_coords, np.zeros_like(mm_charges))) <= 1e-12
    for axis in np.eye(3):
        assert abs(_independent_shift(CONTACT, mm_coords + 10.0 * axis, mm_charges)) < 1e-3


@_apply(real_sqd)
def test_ac2_c_sqd_end_to_end_energy_shift(benchmark_run):
    records = benchmark_run[2]
    points = {mode: records[(CONTACT, mode)] for mode in MODES}
    for record in points.values():
        assert record["subspace_dim"] == record["full_ci_dim"]
    shift_sqd = points["full_oneelectron"]["sqd_energy"] - points["diagonal"]["sqd_energy"]
    shift_t0 = _independent(CONTACT, "full_oneelectron") - _independent(CONTACT, "diagonal")
    assert abs(shift_sqd) >= 1e-3
    assert abs(shift_sqd - shift_t0) <= TOL_HA


def _synthetic_records():
    base = {
        "status": "completed",
        "unique_dets_vs_shots": [[10, 4], [100, 9]],
        "seed": 0,
    }
    return [
        {
            **base,
            "fixture": "h2o_1tip3p_4e4o",
            "mode": "diagonal",
            "sqd_energy": -75.01,
            "baseline_energy": -75.02,
            "baseline_tier": "T0",
            "baseline_method": "fci_direct_spin1",
            "delta_mHa": 2.75,
            "subspace_dim": 4,
            "full_ci_dim": 36,
        },
        {
            **base,
            "fixture": "h3o_plus_3tip3p_4e4o",
            "mode": "full_oneelectron",
            "sqd_energy": -75.31,
            "baseline_energy": -75.30,
            "baseline_tier": "T2",
            "baseline_method": "ccsd_active_space",
            "delta_mHa": -4.25,
            "subspace_dim": 36,
            "full_ci_dim": 36,
        },
    ]


def test_honesty_benchmark_summary_has_no_banned_terms():
    records = _synthetic_records()
    text = _bench().render_benchmark_summary(records)
    folded = text.casefold()
    assert [term for term in BANNED_TERMS if term.casefold() in folded] == []
    for record in records:
        assert f"{abs(record['delta_mHa']):.2f}" in text
        assert record["baseline_tier"] in text
        assert record["baseline_method"] in text


def _case_forms(term):
    alternating = "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(term))
    return sorted({term.lower(), term.upper(), alternating})


def _placements(value, word):
    return (word + value, value[:2] + word + value[2:], value + word)


def test_honesty_benchmark_summary_rejects_banned_term():
    render = _bench().render_benchmark_summary
    records = _synthetic_records()
    records[1]["baseline_method"] = "Quantum Advantage"
    with pytest.raises(ValueError):
        render(records)
    for field in ("fixture", "mode", "status", "baseline_method"):
        for term in BANNED_TERMS:
            for word in _case_forms(term):
                for injected in _placements(_synthetic_records()[1][field], word):
                    records = _synthetic_records()
                    records[1][field] = injected
                    with pytest.raises(ValueError):
                        render(records)
        for word in ("classical advantage", "量子模拟"):
            for injected in _placements(_synthetic_records()[1][field], word):
                records = _synthetic_records()
                records[1][field] = injected
                assert injected in render(records)


def test_benchmark_inputs_rejects_failing_geometry():
    bench = _bench()
    fixture = FIXTURES["h3o_plus_3tip3p_4e4o"]
    bench.benchmark_inputs(copy.deepcopy(fixture))
    moved = copy.deepcopy(fixture)
    moved["water_states"][-1][:3] = (np.asarray(fixture["qm_coords"][0]) + [0, 0, 0.8]).tolist()
    with pytest.raises(ValueError, match="qm_mm_distance"):
        bench.benchmark_inputs(moved)


@pytest.mark.sqd
def test_benchmark_point_forwards_mode_and_seed(monkeypatch):
    import q2m3.sqd
    import q2m3.sqd.orchestrator

    bench = _bench()
    calls = []

    def fake_run_sqd(*args, **kwargs):
        calls.append(kwargs)
        return _fake_result(kwargs["seed"])

    monkeypatch.setattr(q2m3.sqd, "run_sqd", fake_run_sqd, raising=False)
    monkeypatch.setattr(q2m3.sqd.orchestrator, "run_sqd", fake_run_sqd)
    inputs = bench.benchmark_inputs(copy.deepcopy(FIXTURES["h2o_1tip3p_4e4o"]))
    record = bench.benchmark_point(inputs, "full_oneelectron", 7)
    assert len(calls) == 1
    assert calls[0]["embedding_mode"] == "full_oneelectron"
    assert calls[0]["seed"] == 7
    assert calls[0]["verbose"] is False
    assert record["seed"] == 7


def _fake_result(seed):
    from types import SimpleNamespace

    return SimpleNamespace(
        status="completed",
        sqd_energy=-75.0,
        baseline_energy=-75.001,
        baseline_tier="T0",
        baseline_method="fci_direct_spin1",
        delta_mHa=1.0,
        subspace_dim=9,
        subspace_dims=(3, 3),
        full_ci_dim=36,
        unique_dets_vs_shots=((1, 1), (10, 5), (100, 9)),
        seed=seed,
        embedding_mode="full_oneelectron",
    )
