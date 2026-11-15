"""Public SQD entry points execute and preserve the scientific result contract."""

import importlib
import json
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import pytest

pytestmark = pytest.mark.sqd


def mod():
    return importlib.import_module("q2m3.sqd.orchestrator")


def assert_same_energy(actual, expected):
    assert actual == pytest.approx(expected, abs=1e-10, rel=0)


def test_energy_oracle_rejects_ten_nanohartree_error():
    with pytest.raises(AssertionError):
        assert_same_energy(-75.0 + 1e-8, -75.0)


def test_geometry_reference_only_roundtrip(tmp_path):
    from q2m3.utils.io import save_json_results

    result = mod().run_sqd(
        ["H", "H"],
        np.array([[0, 0, 0], [0, 0, 0.74]]),
        active_electrons=2,
        active_orbitals=2,
        mode="reference_only",
        verbose=False,
    )
    assert result.validate() is result
    assert result.baseline_tier == "T0"
    assert result.baseline_energy < result.hf_energy
    assert result.sqd_energy is None
    assert result.backend is None
    assert result.null_reasons["shots"] == "reference_only"
    assert result.null_reasons["samples_per_batch"] == "reference_only"
    assert result.null_reasons["ext_sqd"] == "reference_only"
    assert result.diagnostics["resources"]["peak_rss_mb"] > 0
    path = tmp_path / "reference.json"
    save_json_results(result, path)
    assert json.loads(path.read_text())["status"] == "reference_only"
    with pytest.raises(TypeError):
        result.provenance["seed"] = 4


@pytest.fixture(scope="module")
def h2_data():
    from q2m3.molecule import MoleculeConfig
    from q2m3.sqd.ansatz import build_ccsd_seed
    from q2m3.sqd.integrals import build_integrals

    data = build_integrals(
        MoleculeConfig("h2", ["H", "H"], [[0, 0, 0], [0, 0, 0.9]], 0, 2, 4, "6-31g"),
        host_available_mb=4096,
    )
    return data, build_ccsd_seed(data, host_available_mb=4096)


def low(data, seed_data, **kwargs):
    return mod().run_sqd_from_integrals(
        data.h1,
        data.h2,
        data.e_core,
        norb=data.norb,
        nelec=data.nelec,
        context=data.context,
        seed_data=seed_data,
        **kwargs,
    )


def test_full_high_low_same_frame_seed(h2_data, tmp_path):
    from q2m3.sqd.config import LUCJConfig
    from q2m3.utils.io import save_json_results

    data, seed_data = h2_data
    lucj = LUCJConfig(n_reps=2, shots=512)
    high = mod().run_sqd(
        ["H", "H"],
        np.array([[0, 0, 0], [0, 0, 0.9]]),
        active_electrons=2,
        active_orbitals=4,
        basis="6-31g",
        lucj=lucj,
        seed=31,
        verbose=False,
    )
    result = low(data, seed_data, lucj=lucj, seed=31)
    for key in (
        "sqd_energy",
        "baseline_energy",
        "hf_energy",
        "iso_active_space_ccsd_energy",
        "iso_ndet_sci_energy",
        "iso_ndet_random_energy",
    ):
        assert_same_energy(getattr(high, key), getattr(result, key))
    assert high.subspace_dims == result.subspace_dims
    assert high.unique_dets_vs_shots == result.unique_dets_vs_shots
    assert result.seed == 31
    assert result.backend == "ffsim"
    assert result.provenance["context"]["hamiltonian_id"] == data.context.hamiltonian_id
    assert result.diagnostics["diagonalization_allocations"]
    path = tmp_path / "full.json"
    save_json_results(result, path)
    assert len(json.loads(path.read_text())) == 42


# SQD energy of the H2 fixture with LUCJConfig(n_reps=2, shots=512) and seed 31, before
# samples_per_batch could be requested. The recovery batch size was then min(300, shots).
H2_SQD_ENERGY_DEFAULT_BATCH = -1.1269862504414256


# Diagonalization runs in a supervised worker process, so a spy in this process cannot
# see the library call. The recorded value is the one the library received
# (test_recovery_records_the_batch_size_the_library_received).
def test_default_samples_per_batch_keeps_head_energy_and_records_actual_value(h2_data):
    from q2m3.sqd.config import LUCJConfig

    result = low(*h2_data, lucj=LUCJConfig(n_reps=2, shots=512), seed=31)
    assert_same_energy(result.sqd_energy, H2_SQD_ENERGY_DEFAULT_BATCH)
    assert result.samples_per_batch == 300 and type(result.samples_per_batch) is int
    assert result.provenance["requested_lucj"]["samples_per_batch"] is None
    assert result.ext_sqd is None
    assert result.null_reasons["ext_sqd"] == "ext_sqd_disabled"


def test_requested_samples_per_batch_reaches_the_recovery_library(h2_data):
    from q2m3.sqd.config import LUCJConfig

    result = low(*h2_data, lucj=LUCJConfig(n_reps=2, shots=512, samples_per_batch=30), seed=31)
    assert result.samples_per_batch == 30 and type(result.samples_per_batch) is int


def test_integral_entry_validates_ext_sqd_without_running_it(h2_data, monkeypatch):
    from q2m3.sqd.config import ExtSQDSettings, LUCJConfig

    def no_run(*args, **kwargs):
        raise AssertionError("an invalid ext_sqd must be rejected before execution")

    with monkeypatch.context() as patch:
        patch.setattr(mod(), "_run", no_run)
        for value in ({"AB": ExtSQDSettings()}, LUCJConfig(), ExtSQDSettings(order=0)):
            with pytest.raises(ValueError, match="ext_sqd|order"):
                low(*h2_data, ext_sqd=value)
    # A valid request passes validation and runs the ext stage.
    result = low(*h2_data, lucj=LUCJConfig(n_reps=2, shots=512), seed=31, ext_sqd=ExtSQDSettings())
    assert result.ext_sqd is not None
    assert result.ext_sqd.settings == ExtSQDSettings()
    assert "ext_sqd" not in result.null_reasons


EXT_PRIOR_STAGES = ("prepare", "sample", "diagonalize", "reference", "comparison")


def run_with_ext(h2_data):
    from q2m3.sqd.config import ExtSQDSettings, LUCJConfig

    return low(*h2_data, lucj=LUCJConfig(n_reps=2, shots=512), seed=31, ext_sqd=ExtSQDSettings())


def record_ext_calls(monkeypatch, path, *, expire_deadline=False):
    """Replace extend.run_ext_sqd with a recorder that writes one JSON line per call.

    The orchestrator runs _execute in a forked worker. A patch made before the fork is
    active in the worker, but a list filled in the worker does not return to this process.
    """
    from q2m3.sqd import extend

    real = extend.run_ext_sqd

    def recorder(*args, **kwargs):
        call = {"seed_source": kwargs.get("seed_source"), "deadline": kwargs.get("deadline")}
        call["called_at"] = monotonic()
        with path.open("a") as handle:
            handle.write(json.dumps(call) + "\n")
        if expire_deadline:
            kwargs["deadline"] = monotonic() - 1.0
        return real(*args, **kwargs)

    monkeypatch.setattr(extend, "run_ext_sqd", recorder)


def recorded_calls(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_ext_sqd_switch_adds_only_a_quantum_arm_and_keeps_the_sqd_result(h2_data):
    result = run_with_ext(h2_data)
    assert result.status == "completed"
    # The extension is a separate record; it must not replace the SQD energy.
    assert_same_energy(result.sqd_energy, H2_SQD_ENERGY_DEFAULT_BATCH)
    assert result.ext_sqd is not None
    assert tuple(arm.seed_source for arm in result.ext_sqd.arms) == ("quantum",)
    (arm,) = result.ext_sqd.arms
    assert arm.null_reason is None
    # A fixed-subspace eigenvalue cannot be lower than the exact baseline.
    assert arm.energy >= result.baseline_energy - 1e-8


def test_ext_wall_budget_null_arm_reaches_the_result(h2_data, monkeypatch, tmp_path):
    from q2m3.sqd import extend

    def solver_reached(*args, **kwargs):
        raise AssertionError("the ext solver ran after the time precheck failed")

    calls = tmp_path / "ext_calls.jsonl"
    record_ext_calls(monkeypatch, calls, expire_deadline=True)
    monkeypatch.setattr(extend, "kernel_fixed_space", solver_reached)
    result = run_with_ext(h2_data)
    assert [call["seed_source"] for call in recorded_calls(calls)] == ["quantum"]
    assert result.status == "completed"
    assert_same_energy(result.sqd_energy, H2_SQD_ENERGY_DEFAULT_BATCH)
    assert result.ext_sqd is not None
    (arm,) = result.ext_sqd.arms
    assert arm.null_reason == "ext_wall_budget"
    assert arm.energy is None
    assert "ext_sqd" not in result.null_reasons


def test_ext_stage_receives_the_run_deadline(h2_data, monkeypatch, tmp_path):
    from q2m3.sqd.config import ReferenceConfig

    calls = tmp_path / "ext_calls.jsonl"
    record_ext_calls(monkeypatch, calls)
    t_before = monotonic()
    result = run_with_ext(h2_data)
    assert len(recorded_calls(calls)) == 1
    (call,) = recorded_calls(calls)
    # The precheck needs the absolute deadline of this run. None disables it, a deadline
    # recomputed at the ext stage ignores the time already spent, and a remaining time
    # in seconds is earlier than the call.
    assert call["deadline"] is not None
    assert call["called_at"] < call["deadline"]
    spent = sum(result.timings_s[name] for name in EXT_PRIOR_STAGES)
    run_start = call["deadline"] - ReferenceConfig().wall_budget_s
    assert run_start <= call["called_at"] - spent
    # A remaining-seconds deadline gives run_start near -spent; that check alone passes
    # only on a host with a monotonic clock above the wall budget (G2b T2).
    assert t_before <= run_start
    (arm,) = result.ext_sqd.arms
    assert arm.null_reason is None


@pytest.mark.parametrize("tier", ["T1", "T2"])
def test_actual_downgraded_solver_result_json(h2_data, monkeypatch, tmp_path, tier):
    from q2m3.sqd import reference as reference_module
    from q2m3.utils.io import save_json_results

    original = reference_module.build_reference_candidates

    def candidates(*args, **kwargs):
        return tuple(
            replace(c, estimated_wall_s=1e8) if c.tier != tier else c
            for c in original(*args, **kwargs)
        )

    monkeypatch.setattr(reference_module, "build_reference_candidates", candidates)
    data, _ = h2_data
    with pytest.warns(RuntimeWarning, match="downgrad"):
        result = low(data, None, mode="reference_only")
    assert result.baseline_tier == tier
    assert result.baseline_downgrade_reason
    assert result.warnings
    assert result.baseline_uncertainty_mHa is None
    assert result.reference_attempts[-1].outcome == "success"
    path = tmp_path / f"{tier}.json"
    save_json_results(result, path)
    assert json.loads(path.read_text())["baseline_tier"] == tier


def test_seed_propagation_and_reference_only_omits_work(h2_data, monkeypatch):
    from q2m3.sqd.config import LUCJConfig

    engine = mod()
    for name in ("diagonalize_samples", "run_reference", "run_comparisons"):
        original = getattr(engine, name)

        def checked(*args, _original=original, **kwargs):
            assert kwargs["seed"] == 71
            return _original(*args, **kwargs)

        monkeypatch.setattr(engine, name, checked)
    sample = engine.FfsimSampler.sample

    def checked_sample(*args, **kwargs):
        assert kwargs["seed"] == 71
        return sample(*args, **kwargs)

    monkeypatch.setattr(engine.FfsimSampler, "sample", checked_sample)
    data, seed_data = h2_data
    low(data, seed_data, lucj=LUCJConfig(shots=128), seed=71)

    def forbidden(*args, **kwargs):
        raise AssertionError("reference-only executed SQD work")

    for name in ("build_lucj_from_integrals", "diagonalize_samples", "run_comparisons"):
        monkeypatch.setattr(engine, name, forbidden)
    monkeypatch.setattr(engine.FfsimSampler, "sample", forbidden)
    assert low(data, None, mode="reference_only", seed=71).status == "reference_only"


@pytest.mark.parametrize("failure", ["sampling", "ccsd", "budget", "baseline", "validate"])
def test_failures_never_return_success(h2_data, monkeypatch, failure):
    from q2m3.sqd.config import LUCJConfig, ReferenceConfig
    from q2m3.sqd.exceptions import (
        BaselineUnavailableError,
        CCSDConvergenceError,
        ResourceLimitError,
        SamplingIntegrityError,
    )

    data, seed_data = h2_data
    options = {"lucj": LUCJConfig(shots=128)}
    if failure == "sampling":

        def fail(*args, **kwargs):
            raise SamplingIntegrityError("injected bad sampling")

        monkeypatch.setattr(mod().FfsimSampler, "sample", fail)
        expected = SamplingIntegrityError
    elif failure == "ccsd":
        seed_data = replace(seed_data, converged=False)
        expected = CCSDConvergenceError
    elif failure == "budget":
        options["reference"] = ReferenceConfig(rss_budget_mb=1)
        expected = ResourceLimitError
    elif failure == "baseline":
        from q2m3.sqd import reference as reference_module

        original = reference_module.build_reference_candidates
        monkeypatch.setattr(
            reference_module,
            "build_reference_candidates",
            lambda *a, **k: tuple(replace(c, estimated_wall_s=1e8) for c in original(*a, **k)),
        )
        expected = BaselineUnavailableError
    else:
        options["seed"] = -1
        expected = ValueError
    with pytest.raises(expected):
        low(data, seed_data, **options)


def test_budget_guard_is_consumed_before_prepare(h2_data, monkeypatch):
    from q2m3.sqd.exceptions import ResourceLimitError

    def denied(*args, **kwargs):
        raise ResourceLimitError("preallocation guard consumed")

    monkeypatch.setattr(mod(), "guard_allocation", denied)
    with pytest.raises(ResourceLimitError, match="preallocation guard consumed"):
        low(*h2_data)


def test_low_seed_actual_energy_is_authenticated(h2_data):
    from q2m3.sqd.exceptions import ProvenanceMismatchError

    data, seed_data = h2_data
    with pytest.raises(ProvenanceMismatchError):
        low(
            data, replace(seed_data, ccsd_energy=seed_data.ccsd_energy + 0.1), mode="reference_only"
        )


def _detached_sleeper(path):
    os.setsid()
    Path(path).write_text(str(os.getpid()))
    sleep(30)


def _nested_sleep(path):
    child = multiprocessing.get_context("fork").Process(target=_detached_sleeper, args=(path,))
    child.start()
    sleep(30)


def test_timeout_terminates_descendant_in_another_session(tmp_path):
    from q2m3.sqd.exceptions import ReferenceTimeoutError

    path = tmp_path / "descendant.pid"
    with pytest.raises(ReferenceTimeoutError):
        mod()._supervise(_nested_sleep, (str(path),), {}, deadline=monotonic() + 0.5, cap=8192)
    pid = int(path.read_text())
    for _ in range(100):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            break
        if state == "Z":
            break
        sleep(0.01)
    else:
        pytest.fail("detached grandchild survived deadline cancellation")


def _allocate_until_killed():
    arrays = []
    while True:
        arrays.append(np.ones(4_000_000, dtype=np.float64))
        sleep(0.02)


def test_runtime_rss_guard_stops_actual_allocation():
    from q2m3.sqd.exceptions import ResourceLimitError

    engine = mod()
    cap = 2 * engine._rss_mb(os.getpid()) + 180
    with pytest.raises(ResourceLimitError, match="process tree RSS"):
        engine._supervise(_allocate_until_killed, (), {}, deadline=monotonic() + 10, cap=cap)


@pytest.mark.parametrize("entry", ["geometry", "integrals"])
@pytest.mark.parametrize("gate", ["validate", "guard"])
def test_public_entry_consumes_gate_before_numerical_work(h2_data, monkeypatch, entry, gate):
    from q2m3.sqd.exceptions import ResourceLimitError

    engine = mod()
    expected = ValueError if gate == "validate" else ResourceLimitError

    def reject(*args, **kwargs):
        raise expected("entry gate deliberately denied")

    def forbidden(*args, **kwargs):
        raise AssertionError("numerical work started before entry gate")

    monkeypatch.setattr(engine, "build_integrals", forbidden)
    monkeypatch.setattr(engine, "build_lucj_from_integrals", forbidden)
    if gate == "guard":
        monkeypatch.setattr(engine, "guard_allocation", reject)
    elif entry == "geometry":
        monkeypatch.setattr(engine.SQDConfig, "validate", reject)
    else:
        monkeypatch.setattr(engine, "validate_integral_inputs", reject)
    with pytest.raises(expected, match="entry gate deliberately denied"):
        if entry == "geometry":
            engine.run_sqd(
                ["H", "H"],
                np.array([[0, 0, 0], [0, 0, 0.74]]),
                active_electrons=2,
                active_orbitals=2,
                verbose=False,
            )
        else:
            low(*h2_data)


def test_geometry_ccsd_failure_precedes_reference(monkeypatch):
    from q2m3.sqd.exceptions import CCSDConvergenceError

    def failed_seed(*args, **kwargs):
        raise CCSDConvergenceError("geometry seed failed")

    def forbidden(*args, **kwargs):
        raise AssertionError("reference ran after failed CCSD")

    engine = mod()
    monkeypatch.setattr(engine, "build_ccsd_seed", failed_seed)
    monkeypatch.setattr(engine, "run_reference", forbidden)
    with pytest.raises(CCSDConvergenceError, match="geometry seed failed"):
        engine.run_sqd(
            ["H", "H"],
            np.array([[0, 0, 0], [0, 0, 0.74]]),
            active_electrons=2,
            active_orbitals=2,
            verbose=False,
        )


def test_partial_result_payload_remains_deadline_supervised(monkeypatch):
    import struct

    from q2m3.sqd.exceptions import ReferenceTimeoutError

    def partial_writer(sender, function, args, kwargs):
        os.setsid()
        os.write(sender.fileno(), struct.pack("!i", 10000) + b"partial")
        sleep(2)
        sender.close()

    monkeypatch.setattr(mod(), "_worker", partial_writer)
    started = monotonic()
    with pytest.raises(ReferenceTimeoutError):
        mod()._supervise(lambda: None, (), {}, deadline=started + 0.2, cap=8192)
    assert monotonic() - started < 1


def test_result_reconstruction_remains_deadline_supervised(monkeypatch):
    from q2m3.sqd.exceptions import ReferenceTimeoutError

    def payload_writer(sender, function, args, kwargs):
        os.setsid()
        sender.send(("result", ("value", np.ones(200_000))))
        sender.close()

    original = mod()._unwire

    def slow_reconstruction(value):
        sleep(1.2)
        return original(value)

    monkeypatch.setattr(mod(), "_worker", payload_writer)
    monkeypatch.setattr(mod(), "_unwire", slow_reconstruction)
    started = monotonic()
    with pytest.raises(ReferenceTimeoutError):
        mod()._supervise(lambda: None, (), {}, deadline=started + 0.3, cap=8192)
    assert monotonic() - started < 1


def test_float32_reference_hf_uses_float64_physical_arithmetic(h2_data):
    from q2m3.sqd.integrals import hamiltonian_id

    data, _ = h2_data
    h1, h2 = data.h1.astype(np.float32), data.h2.astype(np.float32)
    context = replace(
        data.context,
        hamiltonian_id=hamiltonian_id(h1, h2, data.e_core, norb=data.norb, nelec=data.nelec),
    )
    result = mod().run_sqd_from_integrals(
        h1,
        h2,
        data.e_core,
        norb=data.norb,
        nelec=data.nelec,
        context=context,
        seed_data=None,
        mode="reference_only",
    )
    one, two = h1.astype(np.float64), h2.astype(np.float64)
    o = data.nelec[0]
    expected = data.e_core + sum(2 * one[i, i] for i in range(o))
    expected += sum(2 * two[i, i, j, j] - two[i, j, j, i] for i in range(o) for j in range(o))
    assert result.hf_energy == pytest.approx(expected, abs=1e-12, rel=0)


class _AllocationAttemptedError(Exception):
    """A test stub was reached after the entry should already have rejected the input."""


def _forbid_allocation(monkeypatch):
    from q2m3.sqd import diagonalize, observables, sampling

    def forbidden(*args, **kwargs):
        raise _AllocationAttemptedError("allocation path reached")

    # Each module imports guard_allocation by name, so patch every namespace.
    for module in (mod(), sampling, diagonalize, observables):
        monkeypatch.setattr(module, "guard_allocation", forbidden)
    monkeypatch.setattr(mod(), "_run", forbidden)


@pytest.mark.parametrize("mode", ["full", "reference_only"])
def test_integral_entry_rejects_fifteen_orbitals_before_allocation(h2_data, monkeypatch, mode):
    from q2m3.sqd.exceptions import ResourceModelDomainError

    data, seed_data = h2_data
    _forbid_allocation(monkeypatch)
    with pytest.raises(ResourceModelDomainError, match="orbital limit 14"):
        mod().run_sqd_from_integrals(
            data.h1,
            data.h2,
            data.e_core,
            norb=15,
            nelec=data.nelec,
            context=data.context,
            seed_data=seed_data if mode == "full" else None,
            mode=mode,
        )


def test_geometry_entry_rejects_fifteen_orbitals_before_allocation(monkeypatch):
    from q2m3.sqd.exceptions import ResourceModelDomainError

    _forbid_allocation(monkeypatch)
    with pytest.raises(ResourceModelDomainError, match="orbital limit 14"):
        mod().run_sqd(
            ["H", "H"],
            np.array([[0, 0, 0], [0, 0, 0.74]]),
            active_electrons=2,
            active_orbitals=15,
            verbose=False,
        )


@pytest.mark.parametrize("entry", ["run", "sqd_supervise", "supervise", "reference"])
def test_parent_rss_cap_rejections_are_guard_rejections(monkeypatch, entry):
    from q2m3.sqd import reference
    from q2m3.sqd.config import ReferenceConfig
    from q2m3.sqd.exceptions import AllocationGuardError

    for module in (mod(), reference):
        monkeypatch.setattr(module, "_rss_mb", lambda pid: 1e9)
    calls = {
        "run": lambda: mod()._run(abs, (), {}, ReferenceConfig()),
        "sqd_supervise": lambda: mod()._supervise(
            abs, (), {}, deadline=monotonic() + 60, cap=100.0
        ),
        "supervise": lambda: reference._supervise(abs, (), {}, monotonic() + 60, 100.0),
        "reference": lambda: reference.run_reference(
            None, None, 0.0, norb=2, nelec=(1, 1), context=None, host_available_mb=100.0
        ),
    }
    with pytest.raises(AllocationGuardError, match="cap"):
        calls[entry]()


def test_full_run_survives_comparison_guard_rejection(h2_data, monkeypatch):
    from q2m3.sqd import observables
    from q2m3.sqd.exceptions import AllocationGuardError

    original = observables.guard_allocation

    def reject_comparison(*args, **kwargs):
        if kwargs["stage"] == "comparison":
            raise AllocationGuardError("predicted=9000.0 MB, budget=8192.0 MB, stage=comparison")
        return original(*args, **kwargs)

    monkeypatch.setattr(observables, "guard_allocation", reject_comparison)
    data, seed_data = h2_data
    result = low(data, seed_data)
    assert result.validate() is result
    assert result.status == "completed"
    reason = "comparison_rss_budget: predicted=9000.0 MB, budget=8192.0 MB, stage=comparison"
    for name in (
        "iso_ndet_sci_energy",
        "iso_ndet_random_energy",
        "delta_vs_sci_mHa",
        "ratio_sqd_over_sci",
        "unique_dets_vs_shots",
    ):
        assert getattr(result, name) is None
        assert result.null_reasons[name] == reason
    assert result.delta_mHa == pytest.approx(
        1000 * (result.sqd_energy - result.baseline_energy), abs=1e-9
    )
    assert result.subspace_dim == result.subspace_dims[0] * result.subspace_dims[1]
    assert result.timings_s["comparison"] is not None


GUARD_MESSAGE = "stage=ext, subspace_dims=(6, 6): predicted 9000 MB exceeds 1 MB"
NUMERICAL_MESSAGE = "Fixed-space Davidson solver did not converge"


def raise_from_ext_stage(monkeypatch, error):
    from q2m3.sqd import extend

    def failing(*args, **kwargs):
        raise error

    monkeypatch.setattr(extend, "run_ext_sqd", failing)


def assert_single_null_arm(result, reason):
    # A failed arm is recorded on the arm; the SQD result of the run is kept.
    assert result.status == "completed"
    assert_same_energy(result.sqd_energy, H2_SQD_ENERGY_DEFAULT_BATCH)
    assert result.ext_sqd is not None
    (arm,) = result.ext_sqd.arms
    assert arm.seed_source == "quantum"
    assert arm.energy is None
    assert arm.null_reason == reason
    assert arm.validate() is arm
    assert "ext_sqd" not in result.null_reasons


def test_ext_guard_rejection_becomes_an_ext_rss_budget_null_arm(h2_data, monkeypatch):
    from q2m3.sqd.exceptions import AllocationGuardError

    raise_from_ext_stage(monkeypatch, AllocationGuardError(GUARD_MESSAGE))
    assert_single_null_arm(run_with_ext(h2_data), f"ext_rss_budget: {GUARD_MESSAGE}")


def test_ext_numerical_error_becomes_an_ext_numerical_null_arm(h2_data, monkeypatch):
    from q2m3.sqd.exceptions import ReferenceNumericalError

    raise_from_ext_stage(monkeypatch, ReferenceNumericalError(NUMERICAL_MESSAGE))
    assert_single_null_arm(run_with_ext(h2_data), f"ext_numerical: {NUMERICAL_MESSAGE}")


def test_ext_exception_null_arm_writes_predicted_rss_as_json_null(h2_data, monkeypatch, tmp_path):
    # save_json_results uses allow_nan=False; a NaN sentinel would make the whole result unsavable.
    from q2m3.sqd.exceptions import AllocationGuardError
    from q2m3.utils.io import save_json_results

    raise_from_ext_stage(monkeypatch, AllocationGuardError(GUARD_MESSAGE))
    result = run_with_ext(h2_data)
    path = tmp_path / "ext_null_arm.json"
    save_json_results(result, path)
    (arm,) = json.loads(path.read_text())["ext_sqd"]["arms"]
    assert arm["null_reason"] == f"ext_rss_budget: {GUARD_MESSAGE}"
    assert arm["predicted_rss_mb"] is None


def test_ext_unexpected_error_still_fails_the_run(h2_data, monkeypatch):
    # ReferenceNumericalError is a RuntimeError, so catching RuntimeError would hide bugs.
    raise_from_ext_stage(monkeypatch, RuntimeError("injected ext failure"))
    with pytest.raises(RuntimeError, match="injected ext failure"):
        run_with_ext(h2_data)


def test_reference_only_mode_does_not_run_the_ext_stage(h2_data, monkeypatch, tmp_path):
    from q2m3.sqd.config import ExtSQDSettings

    calls = tmp_path / "ext_calls.jsonl"
    record_ext_calls(monkeypatch, calls)
    result = low(*h2_data, mode="reference_only", ext_sqd=ExtSQDSettings())
    assert recorded_calls(calls) == []
    assert result.status == "reference_only"
    assert result.ext_sqd is None
    assert result.null_reasons["ext_sqd"] == "reference_only"


def test_ext_stage_forwards_the_run_settings_seeds_and_resource_limits(
    h2_data, monkeypatch, tmp_path
):
    from dataclasses import asdict

    from q2m3.sqd import extend
    from q2m3.sqd.config import ExtSQDSettings, LUCJConfig

    orchestrator = mod()
    calls = tmp_path / "ext_calls.jsonl"
    real_ext, real_retained, real_supervise = (
        extend.run_ext_sqd,
        orchestrator._retained_mb,
        orchestrator._supervise,
    )

    def write(row):
        with calls.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def ext_recorder(h1, h2, e_core, seed_space, settings, **kwargs):
        (alpha, beta), amplitudes = seed_space
        names = ("seed_source", "host_available_mb", "rss_budget_mb", "retained_mb")
        row = {name: kwargs[name] for name in names + ("allow_large",)}
        write(row | {"kind": "ext", "settings": asdict(settings)})
        strings = [alpha.tolist(), beta.tolist()]
        write({"kind": "ext_seed", "strings": strings, "amplitudes": amplitudes.tolist()})
        return real_ext(h1, h2, e_core, seed_space, settings, **kwargs)

    def retained_recorder(samples, diagonalized):
        value = real_retained(samples, diagonalized)
        strings = [s.tolist() for s in diagonalized.ci_strings]
        amplitudes = diagonalized.amplitudes.tolist()
        write({"kind": "retained", "value": value, "strings": strings, "amplitudes": amplitudes})
        return value

    worker_caps = []

    def supervise_recorder(function, args, kwargs, **options):
        # Runs in this process; kwargs["cap"] is the RSS limit handed to the worker.
        worker_caps.append(kwargs["cap"])
        return real_supervise(function, args, kwargs, **options)

    monkeypatch.setattr(extend, "run_ext_sqd", ext_recorder)
    monkeypatch.setattr(orchestrator, "_retained_mb", retained_recorder)
    monkeypatch.setattr(orchestrator, "_supervise", supervise_recorder)
    requested = ExtSQDSettings(order=3, threshold=0.0)
    result = low(
        *h2_data,
        lucj=LUCJConfig(n_reps=2, shots=512),
        seed=31,
        ext_sqd=requested,
        allow_large=False,
    )

    def rows(kind):
        return [row for row in recorded_calls(calls) if row["kind"] == kind]

    # _retained_mb also sizes the reference stage, so the last row belongs to ext.
    (call,), (seed,), retained = rows("ext"), rows("ext_seed"), rows("retained")[-1]
    assert len(worker_caps) == 1
    # A default ExtSQDSettings would change the excitation and truncation of the arm.
    assert call["settings"] == json.loads(json.dumps(asdict(requested)))
    assert call["seed_source"] == "quantum"
    # The seeds are the SQD subspace of this run, not the HF string alone.
    assert seed["strings"] == retained["strings"]
    assert len(seed["strings"][0]) * len(seed["strings"][1]) > 1
    # XO26 (S7d, G2b T1): the amplitudes select the seed strings in run_ext_sqd; with
    # threshold=0.0 any non-negative replacement keeps all strings, so compare them here.
    assert seed["amplitudes"] == retained["amplitudes"]
    # A larger limit or a zero retained size would let the ext guard pass an
    # allocation that the worker RSS limit of this run cannot hold.
    assert call["host_available_mb"] == call["rss_budget_mb"] == worker_caps[0]
    assert call["retained_mb"] == retained["value"] > 0
    assert call["allow_large"] is False
    (arm,) = result.ext_sqd.arms
    assert arm.null_reason is None
