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
    assert len(json.loads(path.read_text())) == 40


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
