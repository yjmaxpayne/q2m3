"""The SQD stage runs in a child interpreter with single-thread pools.

The parent hands each job to the child through private pickle files and gets back
decoded results, the child's thread-pool records and the child's typed failures.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import fields
from pathlib import Path

import pytest

from q2m3.fragments import _sqd_child
from q2m3.fragments.errors import FragmentWorkflowError
from q2m3.fragments.sqd_stage import SQDJob, SubprocessSQDRunner, ThreadRecord, _job_payload
from q2m3.molecule import MoleculeConfig
from q2m3.sqd.config import LUCJConfig, ReferenceConfig
from q2m3.sqd.exceptions import ProvenanceMismatchError, ReferenceTimeoutError, ResourceLimitError
from q2m3.sqd.integrals import build_integrals
from q2m3.sqd.result import SQDResult
from q2m3.utils.io import save_json_results

REPO = Path(__file__).resolve().parents[2]
FAKE_CHILD = "tests.fragments._child_fake"
# The OpenMP, MKL, OpenBLAS and rayon pools that SQD needs at one thread each.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
)
SEEDS = (11, 12, 13)
WATER = [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]


def _jobs(config: MoleculeConfig, seeds=SEEDS, **settings) -> tuple[SQDJob, ...]:
    data = build_integrals(config, host_available_mb=4096)
    return tuple(
        SQDJob(
            data.h1, data.h2, data.e_core, data.norb, data.nelec, data.context, seed=s, **settings
        )
        for s in seeds
    )


def _h2(bond: float = 0.74) -> MoleculeConfig:
    return MoleculeConfig("h2", ["H", "H"], [[0, 0, 0], [0, 0, bond]], 0, 2, 2, "sto-3g")


@pytest.fixture(scope="module")
def h2_jobs():
    return _jobs(_h2())


@pytest.fixture
def child_path(monkeypatch):
    # The child imports tests.fragments, and its cwd is the job directory.
    monkeypatch.setenv("PYTHONPATH", f"{REPO}:{REPO / 'src'}")


@pytest.fixture
def caller(child_path, monkeypatch, tmp_path):
    """Caller environment with eight-thread pools; returns the child's probe path."""
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "8")
    probe = tmp_path / "probe.json"
    monkeypatch.setenv("Q2M3_TEST_CHILD_PROBE", str(probe))
    return probe


def _run_fake(monkeypatch, jobs, mode, **runner_kwargs):
    monkeypatch.setenv("Q2M3_TEST_CHILD_MODE", mode)
    return SubprocessSQDRunner(child_module=FAKE_CHILD, **runner_kwargs).run(jobs)


def test_results_and_thread_records_come_back_in_job_order(caller, monkeypatch, h2_jobs):
    outcome = _run_fake(monkeypatch, h2_jobs, "complete")

    assert all(isinstance(result, SQDResult) for result in outcome.results)
    assert [result.seed for result in outcome.results] == list(SEEDS)
    # The fake child sets two pyscf threads, so the record carries the child's own value.
    expected = ThreadRecord(dict.fromkeys(THREAD_VARIABLES, "1"), pyscf_threads=2)
    assert outcome.thread_records == (expected,) * len(SEEDS)
    assert outcome.wall_s > 0.0
    assert {name: os.environ[name] for name in THREAD_VARIABLES} == dict.fromkeys(
        THREAD_VARIABLES, "8"
    )


def test_child_runs_in_a_private_session_with_its_own_markers(caller, monkeypatch, h2_jobs):
    _run_fake(monkeypatch, h2_jobs, "complete")
    probe = json.loads(caller.read_text())

    assert probe["argv"] == [probe["cwd"]]
    assert Path(probe["cwd"]).name.startswith("q2m3-sqd-")
    assert probe["cwd_mode"] == 0o700
    assert probe["executable"] == sys.executable
    assert probe["session_leader"]
    environ = probe["environ"]
    assert {name: environ[name] for name in THREAD_VARIABLES} == dict.fromkeys(
        THREAD_VARIABLES, "1"
    )
    assert uuid.UUID(environ["Q2M3_SQD_STAGE_ID"]).version == 4
    assert environ["Q2M3_SQD_PARENT_PID"] == str(os.getpid())
    assert environ["Q2M3_TEST_CHILD_MODE"] == "complete"
    assert not Path(probe["cwd"]).exists()


@pytest.mark.parametrize(
    ("mode", "error", "message"),
    [
        ("timeout", ReferenceTimeoutError, "deadline expired in child"),
        ("provenance", ProvenanceMismatchError, "frame mismatch in child"),
    ],
)
def test_child_failure_is_raised_with_its_type(caller, monkeypatch, h2_jobs, mode, error, message):
    with pytest.raises(error, match=message) as info:
        _run_fake(monkeypatch, h2_jobs, mode)

    assert type(info.value) is error
    # The child stops at the failing second job.
    assert json.loads(caller.read_text())["solved"] == list(SEEDS[:2])


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("unpicklable", "UnpicklableError: socket handle"),
        ("unloadable", "TwoArgumentError: grid 7"),
    ],
)
def test_failure_that_cannot_cross_pickle_keeps_its_type_name_and_message(
    caller, monkeypatch, h2_jobs, mode, message
):
    with pytest.raises(FragmentWorkflowError) as info:
        _run_fake(monkeypatch, h2_jobs, mode)

    assert str(info.value) == message


def test_child_crash_reports_the_last_twenty_stderr_lines(caller, monkeypatch, h2_jobs):
    # A finite timeout turns a pipe-buffer deadlock into a failure instead of a hang.
    with pytest.raises(FragmentWorkflowError) as info:
        _run_fake(monkeypatch, h2_jobs, "crash", timeout_s=120.0)

    message = str(info.value)
    assert "exit code 3" in message
    assert "line 10\n" in message and "line 29" in message
    assert "line 9\n" not in message
    assert "xxx" not in message


def test_memory_headroom_is_the_smallest_cap_minus_current_rss(monkeypatch):
    monkeypatch.setattr(_sqd_child, "_rss_mb", lambda: 100.0)
    monkeypatch.setattr(_sqd_child, "_mem_available_mb", lambda: 1e6)
    assert _sqd_child.host_available_mb(1e9) == 8092.0
    assert _sqd_child.host_available_mb(500.0) == 400.0
    monkeypatch.setattr(_sqd_child, "_mem_available_mb", lambda: 300.0)
    assert _sqd_child.host_available_mb(1e9) == 200.0


def test_child_refuses_a_job_without_memory_headroom(monkeypatch, h2_jobs):
    monkeypatch.setattr(_sqd_child, "_rss_mb", lambda: 1e9)
    with pytest.raises(ResourceLimitError, match="host_available_mb"):
        _sqd_child.solve(_job_payload(h2_jobs[0]))


@pytest.mark.sqd
def test_real_child_completes_three_small_jobs(child_path):
    # Non-default settings show that each job's own settings reach the solver.
    settings = dict(
        lucj=LUCJConfig(n_reps=1, shots=4096), reference=ReferenceConfig(wall_budget_s=600.0)
    )
    jobs = _jobs(_h2(0.70), (1,), **settings) + _jobs(_h2(), (2, 3), **settings)
    outcome = SubprocessSQDRunner().run(jobs)

    assert [result.status for result in outcome.results] == ["completed"] * 3
    assert [result.seed for result in outcome.results] == [1, 2, 3]
    for result in outcome.results:
        assert (result.n_reps, result.shots) == (1, 4096)
        assert result.diagnostics["resources"]["wall_budget_s"] == 600.0


@pytest.mark.sqd
@pytest.mark.slow
def test_child_result_matches_an_in_process_run(child_path, tmp_path):
    from q2m3.sqd import build_ccsd_seed_from_integrals, run_sqd_from_integrals

    (job,) = _jobs(MoleculeConfig("water", ["O", "H", "H"], WATER, 0, 4, 4, "sto-3g"), (0,))
    (child,) = SubprocessSQDRunner().run([job]).results
    shape = dict(norb=job.norb, nelec=job.nelec, context=job.context)
    seed = build_ccsd_seed_from_integrals(
        job.h1, job.h2, job.e_core, **shape, host_available_mb=4096.0
    )
    parent = run_sqd_from_integrals(job.h1, job.h2, job.e_core, **shape, seed_data=seed, seed=0)

    child_fields, child_energies = _comparable(child, tmp_path / "child.json")
    parent_fields, parent_energies = _comparable(parent, tmp_path / "parent.json")
    assert len(child_fields) + len(child_energies) == len(fields(SQDResult)) == 40
    assert child_fields == parent_fields
    assert child_energies.keys() == parent_energies.keys()
    for name, value in child_energies.items():
        other = parent_energies[name]
        assert (value is None and other is None) or abs(value - other) <= 1e-10, name


@pytest.mark.sqd
@pytest.mark.slow
def test_child_pools_are_single_threaded_when_the_caller_uses_eight(caller, h2_jobs):
    outcome = SubprocessSQDRunner().run(h2_jobs)

    assert [result.status for result in outcome.results] == ["completed"] * 3
    expected = ThreadRecord(dict.fromkeys(THREAD_VARIABLES, "1"), pyscf_threads=1)
    assert outcome.thread_records == (expected,) * 3


def _comparable(result: SQDResult, path: Path) -> tuple[dict, dict]:
    """JSON fields with per-run measurements blanked, and the energies split off."""
    save_json_results(result, path)
    data = json.loads(path.read_text())
    # Warnings follow process state: the RSS boundary and the JAX fork warning.
    data["timings_s"] = data["warnings"] = None
    for key in ("resources", "comparison_allocations"):
        data["diagnostics"].pop(key)
    for attempt in data["reference_attempts"]:
        del attempt["wall_s"], attempt["peak_rss_mb"]
    energies = {name: data.pop(name) for name in list(data) if name.endswith("_energy")}
    return data, energies
