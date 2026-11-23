# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Tests for MC loop module."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from q2m3.solvation.config import MoleculeConfig, QPEConfig, SolvationConfig
from q2m3.solvation.energy import StepResult
from q2m3.solvation.mc_loop import MCResult, _propose_move, create_mc_loop

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def h2_mol_config():
    """Minimal H2 molecule config."""
    return MoleculeConfig(
        name="H2",
        symbols=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        charge=0,
    )


@pytest.fixture
def base_config(h2_mol_config):
    """Minimal SolvationConfig for MC loop tests (fixed mode)."""
    return SolvationConfig(
        molecule=h2_mol_config,
        qpe_config=QPEConfig(),
        hamiltonian_mode="fixed",
        n_waters=3,
        n_mc_steps=20,
        temperature=300.0,
        translation_step=0.3,
        rotation_step=0.2618,
        random_seed=42,
        verbose=False,
    )


@pytest.fixture
def hf_corrected_config(h2_mol_config):
    """SolvationConfig in hf_corrected mode with qpe_interval=5."""
    return SolvationConfig(
        molecule=h2_mol_config,
        qpe_config=QPEConfig(qpe_interval=5),
        hamiltonian_mode="hf_corrected",
        n_waters=3,
        n_mc_steps=20,
        temperature=300.0,
        random_seed=42,
        verbose=False,
    )


@pytest.fixture
def initial_solvents():
    """3 solvent molecules as (n_waters, 6) array."""
    return np.array(
        [
            [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-2.0, 3.46, 0.0, 0.0, 0.0, 0.0],
            [-2.0, -3.46, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def qm_coords():
    """Flat QM coordinates."""
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.74])


def _make_fixed_callback(e_qpe=-1.1, e_mm=-0.05, e_hf=-1.12):
    """Create a deterministic step callback for testing."""
    call_count = 0

    def callback(solvents, coords):
        nonlocal call_count
        call_count += 1
        return StepResult(
            e_qpe=e_qpe + 0.001 * call_count,
            e_mm_sol_sol=e_mm,
            e_hf_ref=e_hf + 0.0005 * call_count,
            callback_time=0.001,
            qpe_time=0.002,
        )

    callback.call_count = lambda: call_count
    return callback


def _make_hf_corrected_callback(qpe_interval=5):
    """Create callback that returns NaN for e_qpe on non-QPE steps."""
    call_count = 0

    def callback(solvents, coords):
        nonlocal call_count
        call_count += 1
        is_qpe_step = (call_count - 1) % qpe_interval == 0
        return StepResult(
            e_qpe=-1.1 + 0.001 * call_count if is_qpe_step else float("nan"),
            e_mm_sol_sol=-0.05,
            e_hf_ref=-1.12 + 0.0005 * call_count,
            callback_time=0.001,
            qpe_time=0.002 if is_qpe_step else 0.0,
        )

    return callback


# ============================================================================
# MCResult dataclass tests
# ============================================================================


class TestMCResult:
    """Tests for MCResult dataclass."""

    def test_creation(self):
        """MCResult can be created with all required fields."""
        n = 10
        result = MCResult(
            initial_energy=-1.15,
            final_energy=-1.16,
            best_energy=-1.17,
            best_qpe_energy=-1.10,
            acceptance_rate=0.5,
            avg_energy=-1.155,
            quantum_energies=np.zeros(n),
            hf_energies=np.zeros(n),
            callback_times=np.zeros(n),
            quantum_times=np.zeros(n),
            final_solvent_states=np.zeros((3, 6)),
            best_solvent_states=np.zeros((3, 6)),
            best_qpe_solvent_states=np.zeros((3, 6)),
            trajectory_solvent_states=np.zeros((n, 3, 6)),
            n_quantum_evaluations=n,
            n_accepted=5,
        )
        assert result.initial_energy == -1.15
        assert result.acceptance_rate == 0.5
        assert result.n_accepted == 5

    def test_array_lengths_consistent(self):
        """All trajectory arrays have the same length."""
        n = 5
        arrs = {
            "quantum_energies": np.ones(n),
            "hf_energies": np.ones(n),
            "callback_times": np.ones(n),
            "quantum_times": np.ones(n),
        }
        result = MCResult(
            initial_energy=0.0,
            final_energy=0.0,
            best_energy=0.0,
            best_qpe_energy=0.0,
            acceptance_rate=0.0,
            avg_energy=0.0,
            final_solvent_states=np.zeros((2, 6)),
            best_solvent_states=np.zeros((2, 6)),
            best_qpe_solvent_states=np.zeros((2, 6)),
            trajectory_solvent_states=np.zeros((n, 2, 6)),
            n_quantum_evaluations=n,
            n_accepted=0,
            **arrs,
        )
        assert len(result.quantum_energies) == n
        assert len(result.hf_energies) == n
        assert len(result.callback_times) == n
        assert len(result.quantum_times) == n


# ============================================================================
# _propose_move tests
# ============================================================================


class TestProposeMove:
    """Tests for _propose_move helper."""

    def test_returns_same_shape(self):
        """Proposed state has same shape as input."""
        state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        rng = np.random.default_rng(0)
        new_state = _propose_move(state, rng, 0.3, 0.2618)
        assert new_state.shape == state.shape

    def test_does_not_mutate_input(self):
        """Original state is unchanged after proposal."""
        state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        original = state.copy()
        rng = np.random.default_rng(0)
        _propose_move(state, rng, 0.3, 0.2618)
        np.testing.assert_array_equal(state, original)

    def test_displacement_bounded(self):
        """Translation displacement is bounded by translation_step."""
        state = np.zeros(6)
        rng = np.random.default_rng(42)
        trans_step = 0.3
        for _ in range(100):
            new = _propose_move(state, rng, trans_step, 0.0)
            assert np.all(np.abs(new[:3]) <= trans_step)

    def test_rotation_bounded(self):
        """Rotation displacement is bounded by rotation_step."""
        state = np.zeros(6)
        rng = np.random.default_rng(42)
        rot_step = 0.2618
        for _ in range(100):
            new = _propose_move(state, rng, 0.0, rot_step)
            assert np.all(np.abs(new[3:]) <= rot_step)


# ============================================================================
# create_mc_loop basic tests
# ============================================================================


class TestCreateMCLoop:
    """Tests for create_mc_loop factory and MC loop execution."""

    def test_runs_correct_number_of_steps(self, base_config, initial_solvents, qm_coords):
        """MC loop executes n_mc_steps iterations."""
        callback = _make_fixed_callback()
        mc_loop = create_mc_loop(base_config, callback)
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=-1.15)
        assert len(result.quantum_energies) == base_config.n_mc_steps
        assert len(result.hf_energies) == base_config.n_mc_steps

    def test_acceptance_rate_range(self, base_config, initial_solvents, qm_coords):
        """Acceptance rate is between 0 and 1."""
        callback = _make_fixed_callback()
        mc_loop = create_mc_loop(base_config, callback)
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=-1.15)
        assert 0.0 <= result.acceptance_rate <= 1.0
        assert result.n_accepted == int(result.acceptance_rate * base_config.n_mc_steps)

    def test_seed_determinism(self, base_config, initial_solvents, qm_coords):
        """Same seed produces identical results."""
        cb1 = _make_fixed_callback()
        cb2 = _make_fixed_callback()
        loop1 = create_mc_loop(base_config, cb1)
        loop2 = create_mc_loop(base_config, cb2)
        r1 = loop1(initial_solvents, qm_coords, seed=123, initial_energy=-1.15)
        r2 = loop2(initial_solvents, qm_coords, seed=123, initial_energy=-1.15)
        np.testing.assert_array_equal(r1.quantum_energies, r2.quantum_energies)
        assert r1.acceptance_rate == r2.acceptance_rate
        np.testing.assert_array_equal(r1.final_solvent_states, r2.final_solvent_states)

    def test_different_seeds_differ(self, base_config, initial_solvents, qm_coords):
        """Different seeds produce different trajectories."""
        cb1 = _make_fixed_callback()
        cb2 = _make_fixed_callback()
        loop1 = create_mc_loop(base_config, cb1)
        loop2 = create_mc_loop(base_config, cb2)
        # Use high initial_energy so moves get accepted (enables divergence)
        r1 = loop1(initial_solvents, qm_coords, seed=1, initial_energy=0.0)
        r2 = loop2(initial_solvents, qm_coords, seed=999, initial_energy=0.0)
        # Final solvent states should differ (overwhelmingly likely)
        assert not np.allclose(r1.final_solvent_states, r2.final_solvent_states)

    def test_best_energy_tracked(self, base_config, initial_solvents, qm_coords):
        """best_energy <= initial_energy (energy should decrease or stay)."""
        callback = _make_fixed_callback()
        mc_loop = create_mc_loop(base_config, callback)
        init_e = -1.0  # Start at relatively high energy
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=init_e)
        assert result.best_energy <= init_e

    def test_avg_energy_computed(self, base_config, initial_solvents, qm_coords):
        """avg_energy is the mean of accepted energies across trajectory."""
        callback = _make_fixed_callback()
        mc_loop = create_mc_loop(base_config, callback)
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=-1.15)
        assert isinstance(result.avg_energy, float)
        assert not math.isnan(result.avg_energy)

    def test_mcresult_fields_complete(self, base_config, initial_solvents, qm_coords):
        """All MCResult fields are populated correctly."""
        callback = _make_fixed_callback()
        mc_loop = create_mc_loop(base_config, callback)
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=-1.15)
        # All arrays have correct length
        n = base_config.n_mc_steps
        assert result.quantum_energies.shape == (n,)
        assert result.hf_energies.shape == (n,)
        assert result.callback_times.shape == (n,)
        assert result.quantum_times.shape == (n,)
        # Solvent state shapes
        assert result.final_solvent_states.shape == (3, 6)
        assert result.best_solvent_states.shape == (3, 6)
        assert result.best_qpe_solvent_states.shape == (3, 6)
        assert result.trajectory_solvent_states.shape == (n, 3, 6)
        np.testing.assert_array_equal(
            result.trajectory_solvent_states[-1], result.final_solvent_states
        )
        # Scalar fields
        assert isinstance(result.initial_energy, float)
        assert isinstance(result.final_energy, float)
        assert isinstance(result.n_quantum_evaluations, int)


# ============================================================================
# Temperature limit behavior
# ============================================================================


class TestTemperatureLimits:
    """Tests for temperature-dependent Metropolis behavior."""

    def test_high_temperature_accepts_all(self, h2_mol_config, initial_solvents, qm_coords):
        """At very high temperature (kT >> delta_E), acceptance rate -> 1."""
        config = SolvationConfig(
            molecule=h2_mol_config,
            hamiltonian_mode="fixed",
            n_waters=3,
            n_mc_steps=100,
            temperature=1e8,  # Very high T
            random_seed=42,
            verbose=False,
        )
        # Callback with energy that increases (should still be accepted at high T)
        call_count = 0

        def high_energy_callback(solvents, coords):
            nonlocal call_count
            call_count += 1
            return StepResult(
                e_qpe=-1.0 + 0.01 * call_count,  # Increasing energy
                e_mm_sol_sol=0.0,
                e_hf_ref=-1.0,
                callback_time=0.0,
                qpe_time=0.0,
            )

        mc_loop = create_mc_loop(config, high_energy_callback)
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=-1.0)
        assert result.acceptance_rate > 0.95

    def test_low_temperature_rejects_uphill(self, h2_mol_config, initial_solvents, qm_coords):
        """At very low temperature, only downhill moves accepted."""
        config = SolvationConfig(
            molecule=h2_mol_config,
            hamiltonian_mode="fixed",
            n_waters=3,
            n_mc_steps=50,
            temperature=0.001,  # Very low T
            random_seed=42,
            verbose=False,
        )

        # Callback always producing higher energy than initial
        def uphill_callback(solvents, coords):
            return StepResult(
                e_qpe=0.0,  # Much higher than initial -10.0
                e_mm_sol_sol=0.0,
                e_hf_ref=0.0,
                callback_time=0.0,
                qpe_time=0.0,
            )

        mc_loop = create_mc_loop(config, uphill_callback)
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=-10.0)
        # All moves are uphill by ~10 Ha at T=0.001K -> exp(-10/kT) ≈ 0
        assert result.acceptance_rate == 0.0


# ============================================================================
# hf_corrected mode tests
# ============================================================================


class TestHFCorrectedMode:
    """Tests for hf_corrected mode specifics."""

    def test_acceptance_uses_hf_energy(self, hf_corrected_config, initial_solvents, qm_coords):
        """In hf_corrected mode, MC acceptance uses e_hf_ref, not e_qpe."""
        callback = _make_hf_corrected_callback(qpe_interval=5)
        mc_loop = create_mc_loop(hf_corrected_config, callback)
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=-1.17)
        # Should still run and produce valid results even with NaN QPE
        assert not math.isnan(result.final_energy)
        assert result.acceptance_rate >= 0.0

    def test_n_quantum_evaluations_less_than_steps(
        self, hf_corrected_config, initial_solvents, qm_coords
    ):
        """n_quantum_evaluations < n_mc_steps in hf_corrected mode."""
        callback = _make_hf_corrected_callback(qpe_interval=5)
        mc_loop = create_mc_loop(hf_corrected_config, callback)
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=-1.17)
        assert result.n_quantum_evaluations < hf_corrected_config.n_mc_steps
        # Should be approximately n_mc_steps / qpe_interval
        expected = sum(
            1
            for i in range(hf_corrected_config.n_mc_steps)
            if i % hf_corrected_config.qpe_config.qpe_interval == 0
        )
        assert result.n_quantum_evaluations == expected

    def test_nan_qpe_steps_recorded(self, hf_corrected_config, initial_solvents, qm_coords):
        """Non-QPE steps record NaN in quantum_energies."""
        callback = _make_hf_corrected_callback(qpe_interval=5)
        mc_loop = create_mc_loop(hf_corrected_config, callback)
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=-1.17)
        nan_count = np.sum(np.isnan(result.quantum_energies))
        valid_count = np.sum(~np.isnan(result.quantum_energies))
        assert nan_count > 0
        assert valid_count == result.n_quantum_evaluations


# ============================================================================
# best_qpe tracking tests
# ============================================================================


class TestBestQPETracking:
    """Tests for best QPE energy and solvent state tracking."""

    def test_best_qpe_only_updated_on_valid_qpe(
        self, hf_corrected_config, initial_solvents, qm_coords
    ):
        """best_qpe_solvent_states only updates when e_qpe is not NaN."""
        callback = _make_hf_corrected_callback(qpe_interval=5)
        mc_loop = create_mc_loop(hf_corrected_config, callback)
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=-1.17)
        # best_qpe_energy should be a valid (non-NaN) number
        assert not math.isnan(result.best_qpe_energy)

    def test_best_qpe_energy_is_minimum(self, base_config, initial_solvents, qm_coords):
        """best_qpe_energy is the minimum QPE energy observed."""
        callback = _make_fixed_callback()
        mc_loop = create_mc_loop(base_config, callback)
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=-1.15)
        # In fixed mode, all steps have QPE -> best should be min of quantum_energies
        assert result.best_qpe_energy <= np.nanmin(result.quantum_energies) + 1e-10


# ============================================================================
# fixed vs dynamic mode tests
# ============================================================================


class TestFixedDynamicMode:
    """Tests for fixed/dynamic mode (QPE every step)."""

    def test_n_quantum_evaluations_equals_steps(self, base_config, initial_solvents, qm_coords):
        """In fixed mode, all steps run QPE."""
        callback = _make_fixed_callback()
        mc_loop = create_mc_loop(base_config, callback)
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=-1.15)
        assert result.n_quantum_evaluations == base_config.n_mc_steps

    def test_no_nan_in_quantum_energies(self, base_config, initial_solvents, qm_coords):
        """In fixed mode, no NaN values in quantum_energies."""
        callback = _make_fixed_callback()
        mc_loop = create_mc_loop(base_config, callback)
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=-1.15)
        assert not np.any(np.isnan(result.quantum_energies))


# ============================================================================
# MCLoopSettings protocol and per-step record tests
# ============================================================================

_SETTINGS_ATTRIBUTES = (
    "n_waters",
    "n_mc_steps",
    "translation_step",
    "rotation_step",
    "kt",
    "acceptance_uses_hf_reference",
)


def _protocol_settings(**overrides) -> SimpleNamespace:
    """Build a duck-typed settings object with all six MCLoopSettings attributes."""
    values = {
        "n_waters": 3,
        "n_mc_steps": 20,
        "translation_step": 0.3,
        "rotation_step": 0.2618,
        "kt": 9.5e-4,
        "acceptance_uses_hf_reference": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _constant_callback(e_qpe=-1.1, e_mm=-0.05, e_vdw=0.0):
    """Create a callback whose acceptance energy does not depend on the configuration."""

    def callback(solvents, coords):
        return StepResult(
            e_qpe=e_qpe,
            e_mm_sol_sol=e_mm,
            e_hf_ref=e_qpe,
            callback_time=0.001,
            qpe_time=0.002,
            e_qmmm_vdw=e_vdw,
        )

    return callback


def _changed_steps(initial_solvents: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    """Return a bool mask: True where the solvent state differs from the previous step."""
    previous = np.concatenate([initial_solvents[None], trajectory[:-1]])
    return np.any(trajectory != previous, axis=(1, 2))


class TestMCLoopSettingsContract:
    """The MC loop depends only on MCLoopSettings and StepResult."""

    @pytest.mark.parametrize(
        ("config_fixture", "callback_factory", "initial_energy", "expected"),
        [
            pytest.param(
                "base_config",
                _make_fixed_callback,
                0.0,
                (-1.1490000000000005, -1.1490000000000002, -1.1490000000000002, 1),
                id="fixed",
            ),
            pytest.param(
                "hf_corrected_config",
                _make_hf_corrected_callback,
                -1.15,
                (-1.1648500000000002, -1.1600000000000001, -1.1695000000000002, 17),
                id="hf_corrected",
            ),
        ],
    )
    def test_qpe_trajectory_bitwise_unchanged(
        self,
        request,
        config_fixture,
        callback_factory,
        initial_energy,
        expected,
        initial_solvents,
        qm_coords,
    ):
        """QPE acceptance energies reproduce the recorded seed-42 trajectory bit for bit.

        The expected values are repr literals recorded on the pre-refactor loop. Each
        parameter set has both accepted and rejected steps, so swapping the acceptance
        branch changes n_accepted or avg_energy.
        """
        config = request.getfixturevalue(config_fixture)
        mc_loop = create_mc_loop(config, callback_factory())
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=initial_energy)
        observed = (result.avg_energy, result.final_energy, result.best_energy, result.n_accepted)
        assert observed == expected

    @pytest.mark.parametrize("missing", _SETTINGS_ATTRIBUTES)
    def test_create_mc_loop_rejects_settings_missing_attribute(self, missing):
        """create_mc_loop reads every settings attribute at creation time, without defaults."""
        settings = _protocol_settings()
        delattr(settings, missing)
        with pytest.raises(AttributeError, match=missing):
            create_mc_loop(settings, _constant_callback())

    def test_create_mc_loop_accepts_protocol_settings(self, initial_solvents, qm_coords):
        """Any object with the six attributes drives the loop, and step sizes go to the right axes.

        Constant energy accepts every step, so every step moves one molecule. Distinct
        translation and rotation bounds catch a swap of the two step sizes or a fallback
        to the SolvationConfig defaults.
        """
        settings = _protocol_settings(translation_step=1e-3, rotation_step=2e-3)
        mc_loop = create_mc_loop(settings, _constant_callback())
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=-1.15)

        assert result.trajectory_solvent_states.shape == (20, 3, 6)
        assert result.n_accepted == 20
        previous = np.concatenate([initial_solvents[None], result.trajectory_solvent_states[:-1]])
        step = np.abs(result.trajectory_solvent_states - previous)
        assert np.all(step[..., :3] <= 1e-3)
        assert np.all(step[..., 3:] <= 2e-3)

    def test_vdw_term_enters_acceptance(self, base_config, initial_solvents, qm_coords):
        """The Metropolis test uses e_qpe + e_mm_sol_sol + e_qmmm_vdw, counting vdw once."""
        e_qpe, e_mm = -1.1, -0.05
        cases = [
            (1.0, e_qpe + e_mm, 0.0),  # vdw raises every trial by 1 Ha at 300 K
            (0.0, e_qpe + e_mm, 1.0),  # no vdw: delta is zero
            (1.0, (e_qpe + e_mm) + 1.0, 1.0),  # initial energy includes vdw: delta is zero
        ]
        for e_vdw, initial_energy, expected_rate in cases:
            mc_loop = create_mc_loop(base_config, _constant_callback(e_qpe, e_mm, e_vdw))
            result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=initial_energy)
            assert result.acceptance_rate == expected_rate, (e_vdw, initial_energy)

        # The loop must add the terms in acceptance_energy order: (quantum + e_mm) + vdw.
        # With these values quantum + (e_mm + vdw) differs by one ULP.
        e_qpe, e_hf, e_mm, e_vdw = -1.0, -1.1, -0.05, 0.003

        def callback(solvents, coords):
            return StepResult(
                e_qpe=e_qpe,
                e_mm_sol_sol=e_mm,
                e_hf_ref=e_hf,
                callback_time=0.001,
                qpe_time=0.002,
                e_qmmm_vdw=e_vdw,
            )

        for uses_hf_reference, quantum in ((False, e_qpe), (True, e_hf)):
            settings = _protocol_settings(
                n_mc_steps=5, kt=1.0, acceptance_uses_hf_reference=uses_hf_reference
            )
            result = create_mc_loop(settings, callback)(
                initial_solvents, qm_coords, seed=42, initial_energy=0.0
            )
            assert result.final_energy == (quantum + e_mm) + e_vdw, uses_hf_reference

    def test_accepted_mask_matches_trajectory(
        self, hf_corrected_config, initial_solvents, qm_coords
    ):
        """accepted[i] is True exactly when step i changed the solvent state."""
        mc_loop = create_mc_loop(hf_corrected_config, _make_hf_corrected_callback())
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=-1.15)

        changed = _changed_steps(initial_solvents, result.trajectory_solvent_states)
        # Precondition: the mask mixes accepts and rejects, and a one-step shift changes it.
        assert changed.any() and not changed.all()
        assert not np.array_equal(changed, np.roll(changed, 1))

        assert result.accepted.dtype == np.bool_
        assert result.accepted.sum() == result.n_accepted
        np.testing.assert_array_equal(result.accepted, changed)

    def test_step_results_record_every_trial(self, base_config, initial_solvents, qm_coords):
        """step_results holds the trial StepResult of every step, accepted or rejected."""
        mc_loop = create_mc_loop(base_config, _make_fixed_callback())
        result = mc_loop(initial_solvents, qm_coords, seed=42, initial_energy=0.0)

        # Precondition: rejected steps exist, so recording only accepted trials is visible.
        assert result.n_accepted < base_config.n_mc_steps

        assert len(result.step_results) == base_config.n_mc_steps
        for i, step_result in enumerate(result.step_results):
            assert step_result.e_qpe == result.quantum_energies[i]
