# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Fast tests for the SQD-driven MC solvation loop with an injected fake solver.

Expected values come from independent calls to the public MM helpers or from
independent NumPy geometry. No test calls a private helper of the module under test.
"""

from __future__ import annotations

import importlib
import platform
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from q2m3.constants import BANNED_TERMS, BOLTZMANN_CONSTANT
from q2m3.molecule import MoleculeConfig
from q2m3.solvation.energy import StepResult
from q2m3.solvation.solvent import (
    TIP3P_WATER,
    compute_mm_energy,
    qm_mm_lj_energy,
    state_array_to_molecules,
)

pytestmark = pytest.mark.skipif(
    not (sys.platform.startswith("linux") and platform.machine() == "x86_64"),
    reason="SQD-MC runs only on Linux x86_64",
)

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
FULL_SUBSPACE_TEXT = (
    "In this domain SQD equals exact diagonalization; the sampler adds no information "
    "beyond CASCI."
)
TIP3P_OH_A = 0.9572
TIP3P_CHARGES = np.array([-0.834, 0.417, 0.417])
# Three waters about 3 A from the solute, each with its own orientation.
STATES = np.array(
    [
        [2.9, 0.3, 0.2, 0.1, 0.4, -0.3],
        [-1.6, 2.6, -0.4, 1.2, -0.7, 0.5],
        [-1.2, -2.7, 0.6, -0.9, 0.2, 2.1],
    ]
)


@pytest.fixture(autouse=True)
def _single_thread_env(monkeypatch):
    for name in THREAD_VARS:
        monkeypatch.setenv(name, "1")


@pytest.fixture(autouse=True)
def _restore_global_rng():
    # run_sqd_solvation reseeds the global np.random state; keep other tests order-independent.
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.fixture
def sqd_mc():
    return importlib.import_module("q2m3.solvation.sqd_mc")


def _h3o(**overrides) -> MoleculeConfig:
    fields = dict(
        name="H3O+",
        symbols=["O", "H", "H", "H"],
        coords=[
            [0.0, 0.0, 0.1],
            [0.95, 0.0, -0.25],
            [-0.475, 0.823, -0.25],
            [-0.475, -0.823, -0.25],
        ],
        charge=1,
        active_electrons=4,
        active_orbitals=3,
        basis="6-31g",
    )
    fields.update(overrides)
    return MoleculeConfig(**fields)


def _config(sqd_mc, **overrides):
    fields = dict(molecule=_h3o(), n_waters=3, n_mc_steps=3)
    fields.update(overrides)
    return sqd_mc.SQDSolvationConfig(**fields)


def _fake_result(**overrides) -> SimpleNamespace:
    fields = dict(
        status="completed",
        sqd_energy=-76.0,
        hf_energy=-75.9,
        subspace_dim=36,
        full_ci_dim=36,
        delta_mHa=0.0,
        baseline_tier="exact",
        baseline_method="casci",
        baseline_energy=-76.0,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _gap_result(sqd_energy: float, gap_ha: float, **overrides) -> SimpleNamespace:
    """Fake result whose delta_mHa and sqd_energy - baseline_energy give the same gap."""
    return _fake_result(
        sqd_energy=sqd_energy,
        baseline_energy=sqd_energy - gap_ha,
        delta_mHa=1000.0 * gap_ha,
        **overrides,
    )


class FakeSolver:
    """Record every call and return (or raise) the queued items in order."""

    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def __call__(self, symbols, coordinates, **kwargs):
        self.calls.append((list(symbols), np.array(coordinates, copy=True), kwargs))
        item = self.items[min(len(self.calls), len(self.items)) - 1]
        if isinstance(item, BaseException):
            raise item
        return item


def _qm_flat(molecule: MoleculeConfig) -> np.ndarray:
    return np.asarray(molecule.coords, dtype=np.float64).flatten()


def _mm_atoms(states: np.ndarray) -> np.ndarray:
    molecules = state_array_to_molecules(TIP3P_WATER, states)
    return np.vstack([mol.get_atom_coords() for mol in molecules])


def _min_distance(qm_coords: np.ndarray, mm_coords: np.ndarray) -> float:
    diff = qm_coords[:, None, :] - mm_coords[None, :, :]
    return float(np.linalg.norm(diff, axis=-1).min())


# =============================================================================
# Configuration
# =============================================================================


def test_config_validate_rejects_unknown_element(sqd_mc):
    molecule = _h3o(symbols=["C", "O"], coords=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    with pytest.raises(ValueError, match="'C'"):
        _config(sqd_mc, molecule=molecule).validate()


def test_config_validate_rejects_solute_without_lj_sites(sqd_mc):
    molecule = _h3o(
        name="H2",
        symbols=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        charge=0,
        active_electrons=2,
        active_orbitals=2,
    )
    with pytest.raises(ValueError, match="epsilon"):
        _config(sqd_mc, molecule=molecule).validate()


@pytest.mark.parametrize("name", THREAD_VARS)
def test_config_validate_rejects_thread_env(sqd_mc, monkeypatch, name):
    config = _config(sqd_mc)
    config.validate()
    monkeypatch.setenv(name, "4")
    with pytest.raises(ValueError, match=re.escape(f"SQD requires {name}=1")):
        config.validate()


def test_config_validate_rejects_nonpositive_steps(sqd_mc):
    _config(sqd_mc, n_mc_steps=1).validate()
    with pytest.raises(ValueError):
        _config(sqd_mc, n_mc_steps=0).validate()


# =============================================================================
# Step callback
# =============================================================================


def test_step_callback_labels_sqd_and_assembles_energy(sqd_mc):
    config = _config(sqd_mc)
    fake = _fake_result(sqd_energy=-76.25, hf_energy=-75.5)
    step = sqd_mc.create_sqd_step_callback(config, solver=FakeSolver([fake]))

    result = step(STATES.copy(), _qm_flat(config.molecule))

    molecules = state_array_to_molecules(TIP3P_WATER, STATES)
    e_mm = compute_mm_energy(molecules)
    e_lj = qm_mm_lj_energy(config.molecule.symbols, config.molecule.coords_array, molecules)
    # Precondition: swapped or zeroed MM terms must change the result.
    assert e_mm != 0.0 and e_lj != 0.0 and abs(e_mm - e_lj) > 1e-6
    assert isinstance(result, StepResult)
    assert result.quantum_method == "sqd"
    assert result.e_qpe == -76.25
    assert result.e_hf_ref == -75.5
    assert result.e_mm_sol_sol == e_mm
    assert result.e_qmmm_vdw == e_lj
    assert result.sqd_result is fake
    assert result.callback_time >= result.qpe_time >= 0.0


def test_step_callback_passes_fixed_seed_and_mode(sqd_mc):
    lucj, reference = object(), object()
    config = _config(
        sqd_mc,
        sqd_seed=7,
        embedding_mode="full_oneelectron",
        lucj=lucj,
        reference=reference,
    )
    solver = FakeSolver([_fake_result()])
    step = sqd_mc.create_sqd_step_callback(config, solver=solver)
    all_states = [STATES + 0.1 * k for k in range(3)]
    for states in all_states:
        step(states.copy(), _qm_flat(config.molecule))

    assert len(solver.calls) == 3
    for (symbols, coords, kwargs), states in zip(solver.calls, all_states, strict=True):
        assert symbols == ["O", "H", "H", "H"]
        assert coords.shape == (4, 3)
        np.testing.assert_array_equal(coords, config.molecule.coords_array)
        assert kwargs["active_electrons"] == 4
        assert kwargs["active_orbitals"] == 3
        assert kwargs["charge"] == 1
        assert kwargs["basis"] == "6-31g"
        assert kwargs["lucj"] is lucj
        assert kwargs["reference"] is reference
        assert "embedding_mode" in kwargs
        assert kwargs["embedding_mode"] == "full_oneelectron"
        assert kwargs["seed"] == 7
        assert kwargs["verbose"] is False
        mm_coords = np.asarray(kwargs["mm_coords"])
        assert mm_coords.shape == (9, 3)
        np.testing.assert_allclose(mm_coords[0::3], states[:, :3], rtol=0, atol=1e-12)
        oh = np.linalg.norm(mm_coords[1::3] - mm_coords[0::3], axis=1)
        np.testing.assert_allclose(oh, TIP3P_OH_A, rtol=0, atol=1e-9)
        np.testing.assert_allclose(kwargs["mm_charges"], np.tile(TIP3P_CHARGES, 3))


def test_step_callback_rejects_partial_subspace(sqd_mc):
    partial = _fake_result(subspace_dim=9, full_ci_dim=36)
    assert issubclass(sqd_mc.SubspaceDomainError, ValueError)
    closed = sqd_mc.create_sqd_step_callback(_config(sqd_mc), solver=FakeSolver([partial]))
    with pytest.raises(sqd_mc.SubspaceDomainError):
        closed(STATES.copy(), _qm_flat(_h3o()))

    opened = sqd_mc.create_sqd_step_callback(
        _config(sqd_mc, allow_partial_subspace=True), solver=FakeSolver([partial])
    )
    assert opened(STATES.copy(), _qm_flat(_h3o())).sqd_result is partial


def test_step_callback_rejects_incomplete_status(sqd_mc):
    for fake in (
        _fake_result(status="reference_only", sqd_energy=-76.0),
        _fake_result(status="completed", sqd_energy=float("nan")),
    ):
        step = sqd_mc.create_sqd_step_callback(_config(sqd_mc), solver=FakeSolver([fake]))
        with pytest.raises(RuntimeError):
            step(STATES.copy(), _qm_flat(_h3o()))


def test_step_callback_propagates_solver_errors(sqd_mc):
    error = RuntimeError("boom")
    step = sqd_mc.create_sqd_step_callback(_config(sqd_mc), solver=FakeSolver([error]))
    with pytest.raises(RuntimeError) as excinfo:
        step(STATES.copy(), _qm_flat(_h3o()))
    assert excinfo.value is error


# =============================================================================
# Driver
# =============================================================================


def test_run_sqd_solvation_records_every_step(sqd_mc):
    n_steps = 6
    config = _config(sqd_mc, n_waters=2, n_mc_steps=n_steps)
    # Energy swings of 1-10 Ha dominate the MM terms: drops accept, rises reject.
    energies = [-1.0, -2.0, 10.0, -3.0, 10.0, -4.0, 10.0]
    solver = FakeSolver([_fake_result(sqd_energy=e) for e in energies])

    result = sqd_mc.run_sqd_solvation(config, solver=solver)

    assert len(solver.calls) == n_steps + 1
    accepted = np.asarray(result.accepted)
    assert accepted.shape == (n_steps,)
    assert 0 < accepted.sum() < n_steps
    np.testing.assert_array_equal(accepted, result.mc.accepted)
    assert len(result.sqd_results) == n_steps
    assert all(got is want for got, want in zip(result.sqd_results, solver.items[1:], strict=True))

    qm = config.molecule.coords_array
    trajectory = result.mc.trajectory_solvent_states
    expected = [_min_distance(qm, _mm_atoms(states)) for states in trajectory]
    proposed = [_min_distance(qm, np.asarray(call[2]["mm_coords"])) for call in solver.calls[1:]]
    # Precondition: a rejected proposal has a different nearest QM-MM distance.
    assert any(abs(proposed[i] - expected[i]) > 1e-9 for i in range(n_steps) if not accepted[i])
    assert len(result.min_qm_mm_A) == n_steps
    np.testing.assert_allclose(result.min_qm_mm_A, expected, rtol=1e-12)


def test_run_sqd_solvation_reports_full_subspace_domain(sqd_mc):
    config = _config(sqd_mc, n_mc_steps=3)
    kt = BOLTZMANN_CONSTANT * config.temperature
    gaps = [0.010, 0.020, 0.030, 0.040]
    assert min(np.diff(gaps)) > kt
    solver = FakeSolver([_gap_result(-1.0 - i, gap) for i, gap in enumerate(gaps)])

    result = sqd_mc.run_sqd_solvation(config, solver=solver)

    assert result.provenance["domain"] == "full_subspace_exact_diagonalization"
    assert len(result.gap_warnings) == 0
    assert sqd_mc.FULL_SUBSPACE_STATEMENT == FULL_SUBSPACE_TEXT
    assert FULL_SUBSPACE_TEXT in sqd_mc.render_sqd_mc_summary(result)


def test_run_sqd_solvation_reports_partial_subspace_domain(sqd_mc):
    config = _config(sqd_mc, n_waters=2, n_mc_steps=5, allow_partial_subspace=True)
    assert config.acceptance_uses_hf_reference is False
    kt = BOLTZMANN_CONSTANT * config.temperature
    partial = dict(subspace_dim=9, full_ci_dim=36)
    # Call 0 is the initial state. MC steps: accept, reject, accept, accept, accept.
    # Gap changes between adjacent accepted states: 0.0, 0.3, 2.7, 0.2 mHa.
    # The rejected step jumps by 40 mHa between two accepted steps.
    plan = [(-1.0, 0.0100), (-2.0, 0.0100), (10.0, 0.0500), (-3.0, 0.0103)]
    plan += [(-4.0, 0.0130), (-5.0, 0.0132)]
    assert 0.0003 < kt < 0.0027 and 1000 * 0.0002 > kt
    solver = FakeSolver([_gap_result(e, gap, **partial) for e, gap in plan])

    result = sqd_mc.run_sqd_solvation(config, solver=solver)

    np.testing.assert_array_equal(result.accepted, [True, False, True, True, True])
    assert result.provenance["domain"] == "partial_subspace"
    assert len(result.gap_warnings) == 1
    assert FULL_SUBSPACE_TEXT not in sqd_mc.render_sqd_mc_summary(result)


def test_run_sqd_solvation_stops_at_step_zero_on_partial_subspace(sqd_mc):
    solver = FakeSolver([_fake_result(subspace_dim=9, full_ci_dim=36)])
    with pytest.raises(sqd_mc.SubspaceDomainError):
        sqd_mc.run_sqd_solvation(_config(sqd_mc), solver=solver)
    assert len(solver.calls) == 1


# =============================================================================
# Honesty of the summary
# =============================================================================

DELTAS = (9.75, 1.25, 2.75, 4.25)
TIERS = ("tier_init_z", "tier_exact_a", "tier_sci_b", "tier_dmrg_c")
METHODS = ("init-ref-z", "casci-ref-a", "shci-ref-b", "dmrg-ref-c")


def _summary_run(sqd_mc):
    items = [
        _gap_result(-1.0 - i, delta / 1000.0, baseline_tier=tier, baseline_method=method)
        for i, (delta, tier, method) in enumerate(zip(DELTAS, TIERS, METHODS, strict=True))
    ]
    config = _config(sqd_mc, n_mc_steps=3)
    return sqd_mc.run_sqd_solvation(config, solver=FakeSolver(items))


def test_honesty_sqd_mc_summary_has_no_banned_terms(sqd_mc):
    result = _summary_run(sqd_mc)
    text = sqd_mc.render_sqd_mc_summary(result)
    folded = text.casefold()
    assert [term for term in BANNED_TERMS if term.casefold() in folded] == []
    for delta, tier, method in zip(DELTAS[1:], TIERS[1:], METHODS[1:], strict=True):
        assert f"{delta:.2f}" in text
        assert tier in text
        assert method in text


def _case_forms(term: str) -> list[str]:
    alternating = "".join(c.upper() if i % 2 else c.lower() for i, c in enumerate(term))
    return sorted({term, term.lower(), term.upper(), term.title(), alternating})


def test_honesty_sqd_mc_summary_rejects_banned_term(sqd_mc):
    result = _summary_run(sqd_mc)
    render = sqd_mc.render_sqd_mc_summary
    target = result.sqd_results[1]
    original = target.baseline_method
    words = ["quantum ADVANTAGE", "Quantum Advantage", "量子优势"]
    words += [form for term in BANNED_TERMS for form in _case_forms(term)]
    for word in words:
        target.baseline_method = f"shci {word} ref"
        with pytest.raises(ValueError):
            render(result)
    for word in ("classical advantage", "量子模拟"):
        target.baseline_method = f"shci {word} ref"
        assert target.baseline_method in render(result)
    target.baseline_method = original
