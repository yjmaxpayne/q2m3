"""Reference priority, scientific diagnostics, and process budget contracts."""

import importlib
from dataclasses import replace

import pytest

from q2m3.sqd.config import ReferenceConfig
from q2m3.sqd.exceptions import BaselineUnavailableError

pytestmark = pytest.mark.sqd


def mod():
    return importlib.import_module("q2m3.sqd.reference")


def candidates():
    m = mod()
    return tuple(
        m.ReferenceCandidate(t, name, True, None, wall, 1200.0, "test-domain", True)
        for t, name, wall in [
            ("T0", "exact_casci", 100.0),
            ("T1", "selected_ci_pyscf", 20.0),
            ("T1+", "shci_dice", 10.0),
            ("T2", "ccsd_t", 5.0),
        ]
    )


def test_resolver_priority_budget_boundaries():
    m = mod()
    cs = candidates()
    config = ReferenceConfig(plugins=("shci_dice",))

    def resolve(wall=200.0, rss=4096.0, cs=cs, config=config):
        return m.resolve_reference(
            4, (2, 2), config=config, candidates=cs, remaining_wall_s=wall, rss_budget_mb=rss
        )

    assert resolve().tier == "T0"
    assert resolve(100.0).tier == "T0"
    assert resolve(99.0).tier == "T1"
    assert resolve(19.0).tier == "T1+"
    assert resolve(9.0).tier == "T2"
    with pytest.raises(BaselineUnavailableError):
        resolve(4.0)
    with pytest.raises(BaselineUnavailableError):
        resolve(rss=1200.0)
    with pytest.raises(ValueError):
        resolve(cs=cs + (cs[0],))
    out = resolve(
        cs=tuple(
            replace(c, in_domain=False, estimated_wall_s=None, estimated_rss_mb=None)
            for c in cs[:1]
        )
        + cs[1:]
    )
    assert out.tier == "T1"
    assert out.downgrade_reason
    assert out.attempts[0].outcome == "out_of_domain"


def test_resolver_t1_ndet_domain_and_missing_plugins():
    m = mod()
    cs = tuple(replace(c, estimated_wall_s=1.0) for c in candidates()[1:])
    plan = m.resolve_reference(
        20,
        (10, 10),
        config=ReferenceConfig(),
        candidates=cs,
        remaining_wall_s=100.0,
        rss_budget_mb=4096.0,
    )
    assert plan.tier == "T2"
    assert any(
        a.method == "selected_ci_pyscf" and a.outcome == "out_of_domain" for a in plan.attempts
    )
    assert all(a.method != "shci_dice" for a in plan.attempts)


def test_t2_independent_flags_and_small_denominator():
    m = mod()
    for t1, converged, triples, corr, expected in [
        (0.021, True, 0.0, -1.0, (True, False, False)),
        (0.0, False, 0.0, -1.0, (False, True, False)),
        (0.0, True, -0.11, -1.0, (False, False, True)),
        (0.0, True, 0.0, 0.0, (False, False, True)),
    ]:
        d = m.t2_diagnostics(t1, converged, triples, corr)
        assert (d.high_t1, d.ccsd_not_converged, d.high_triples) == expected
        d.validate()
    assert m.t2_diagnostics(0.0, True, 0.0, 0.0).triples_ratio is None


@pytest.fixture(scope="module")
def hydrogen():
    from q2m3.molecule import MoleculeConfig
    from q2m3.sqd.integrals import build_integrals

    return build_integrals(
        MoleculeConfig("h2", ["H", "H"], [[0, 0, 0], [0, 0, 0.74]], 0, 2, 2),
        host_available_mb=4096.0,
    )


def run(data, **kwargs):
    return mod().run_reference(
        data.h1,
        data.h2,
        data.e_core,
        norb=data.norb,
        nelec=data.nelec,
        context=data.context,
        host_available_mb=4096.0,
        **kwargs,
    )


@pytest.mark.parametrize(
    "tier,method", [("T0", "exact_casci"), ("T1", "selected_ci_pyscf"), ("T2", "ccsd_t")]
)
def test_real_small_reference_energies(hydrogen, tier, method):
    from pyscf import fci

    m = mod()
    cs = tuple(
        replace(c, estimated_wall_s=1000.0) if c.method != method else c
        for c in m.build_reference_candidates(2, (1, 1), config=ReferenceConfig())
    )
    expected, _ = fci.direct_spin1.kernel(
        hydrogen.h1, hydrogen.h2, 2, (1, 1), ecore=hydrogen.e_core, tol=1e-12
    )
    if tier == "T0":
        result = run(hydrogen, candidates=cs)
        assert result.uncertainty_mHa == 0
    else:
        with pytest.warns(RuntimeWarning, match="reference"):
            result = run(hydrogen, candidates=cs)
        assert result.uncertainty_mHa is None
        assert result.downgrade_reason
    result.validate()
    assert result.energy == pytest.approx(expected, abs=1e-10)
    assert result.tier == tier
    assert result.attempts[-1].outcome == "success"
    assert result.attempts[-1].peak_rss_mb > 0
    if tier == "T1":
        assert result.t1_residual.loose_cutoff == 1e-4
        assert result.t1_residual.tight_cutoff == 1e-5


def test_production_candidates_domain_and_t2_budget():
    m = mod()
    config = ReferenceConfig(wall_budget_s=60.0)
    cs = m.build_reference_candidates(10, (5, 5), config=config)
    plan = m.resolve_reference(
        10, (5, 5), config=config, candidates=cs, remaining_wall_s=60.0, rss_budget_mb=4096.0
    )
    assert plan.tier == "T2"
    assert "engineering" in plan.model_id
    for c in m.build_reference_candidates(15, (7, 7), config=config):
        assert not c.in_domain
        assert c.estimated_rss_mb is None


class Plugin:
    method = "shci_dice"

    def __init__(self, behavior):
        self.behavior = behavior

    def solve(self, h1, h2, e_core, **kwargs):
        import time

        from q2m3.sqd.exceptions import ReferenceNumericalError, ReferenceUnavailableError
        from q2m3.sqd.result import ReferenceResult

        if self.behavior == "sleep":
            time.sleep(2.0)
        if self.behavior == "internal_import":
            raise ModuleNotFoundError("internal backend import defect", name="internal_defect")
        if self.behavior == "unavailable":
            raise ReferenceUnavailableError("missing optional top-level backend")
        if self.behavior == "numerical":
            raise ReferenceNumericalError("nonconverged plugin")
        if self.behavior == "memory":
            import numpy as np

            self.allocation = np.ones(1_000_000_000, dtype=np.uint8)
            time.sleep(2.0)
        c, p = kwargs["context"], kwargs["plan"]
        return ReferenceResult(
            -1.0,
            "T1+",
            self.method,
            None,
            "unknown",
            "plugin_error_unknown",
            p.downgrade_reason,
            None,
            None,
            False,
            c.hamiltonian_id,
            "wrong" if self.behavior == "wrong_frame" else c.frame_id,
            p.attempts,
            ("WARNING: plugin reference",),
        )


def plugin_run(data, behavior, *, wall=5.0, rss=4096.0):
    m = mod()
    plugin = Plugin(behavior)
    candidate = m.ReferenceCandidate("T1+", "shci_dice", True, None, 0.001, 1.0, "fixture", True)
    return run(
        data,
        config=ReferenceConfig(
            wall_budget_s=wall,
            rss_budget_mb=rss,
            plugins=("shci_dice",),
            allowed_tiers=("T0", "T1+"),
        ),
        plugins={"shci_dice": plugin},
        candidates=(candidate,),
    )


def test_plugin_supervisor_timeout_and_error_separation(hydrogen):
    from q2m3.sqd.exceptions import ReferenceNumericalError

    with pytest.raises(BaselineUnavailableError) as caught:
        plugin_run(hydrogen, "sleep", wall=0.15)
    assert any(a.outcome == "timeout" and a.wall_s >= 0.1 for a in caught.value.attempts)
    with pytest.raises(ModuleNotFoundError, match="internal"):
        plugin_run(hydrogen, "internal_import")
    with pytest.raises(ReferenceNumericalError, match="nonconverged"):
        plugin_run(hydrogen, "numerical")
    with pytest.raises(ReferenceNumericalError, match="identity"):
        plugin_run(hydrogen, "wrong_frame")
    with pytest.raises(BaselineUnavailableError) as caught:
        plugin_run(hydrogen, "unavailable")
    assert any(a.reason == "missing optional top-level backend" for a in caught.value.attempts)
    with pytest.warns(RuntimeWarning, match="reference"):
        result = plugin_run(hydrogen, "success")
    assert result.tier == "T1+"
    assert result.uncertainty_mHa is None


def test_supervisor_counts_parent_and_child_rss(hydrogen):
    import os

    from q2m3.sqd.exceptions import ResourceLimitError

    m = mod()
    # The cap permits forked baseline residency but not its touched extra 1 GB.
    cap = min(4000.0, 2 * m._rss_mb(os.getpid()) + 400.0)
    with pytest.raises(ResourceLimitError, match="process tree RSS"):
        plugin_run(hydrogen, "memory", rss=cap)


def test_tree_pids_ignores_process_exit_during_task_enumeration(monkeypatch):
    from pathlib import Path

    m = mod()
    original_glob = Path.glob

    def glob_with_vanishing_process(path, pattern):
        if path == Path("/proc/424242/task"):

            def vanished():
                raise FileNotFoundError(path)
                yield

            return vanished()
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", glob_with_vanishing_process)
    assert m._tree_pids(424242) == {424242}


def test_resolver_is_io_free(monkeypatch):
    import builtins

    m, cs, config = mod(), candidates(), ReferenceConfig()

    def forbidden(*args, **kwargs):
        raise AssertionError("resolver performed I/O/import")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "open", forbidden)
        patch.setattr(builtins, "__import__", forbidden)
        plan = m.resolve_reference(
            4, (2, 2), config=config, candidates=cs, remaining_wall_s=200.0, rss_budget_mb=4096.0
        )
    assert plan.tier == "T0"


def test_seed_failure_precedes_reference_downgrade(hydrogen):
    from q2m3.sqd.ansatz import build_ccsd_seed
    from q2m3.sqd.exceptions import CCSDConvergenceError

    seed = build_ccsd_seed(hydrogen, host_available_mb=4096.0)
    with pytest.raises(CCSDConvergenceError):
        run(
            hydrogen,
            seed_data=replace(seed, converged=False),
            config=ReferenceConfig(wall_budget_s=0.001),
        )


def test_hash_mismatch_never_falls_back(hydrogen):
    import numpy as np

    from q2m3.sqd.exceptions import ReferenceNumericalError

    with pytest.raises(ReferenceNumericalError, match="Hamiltonian"):
        run(replace(hydrogen, h1=hydrogen.h1 + np.eye(2) * 0.01))


def test_nonfinite_builtin_is_numerical_error(hydrogen, monkeypatch):
    import numpy as np
    from pyscf.fci import direct_spin1

    from q2m3.sqd.exceptions import ReferenceNumericalError

    def invalid(self, *args, **kwargs):
        self.converged = True
        return np.nan, np.ones((2, 2)) / 2

    monkeypatch.setattr(direct_spin1.FCISolver, "kernel", invalid)
    with pytest.raises(ReferenceNumericalError, match="finite"):
        run(hydrogen)


def test_t1_single_executed_cutoff_has_honest_label(hydrogen, monkeypatch):
    import time

    m = mod()
    config = ReferenceConfig()
    cs = tuple(
        replace(c, estimated_wall_s=1000.0) if c.tier != "T1" else c
        for c in m.build_reference_candidates(2, (1, 1), config=config)
    )
    plan = m.resolve_reference(
        2, (1, 1), config=config, candidates=cs, remaining_wall_s=100.0, rss_budget_mb=4096.0
    )
    clock = iter([0.0, 0.0, 99.9])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    result = m._PySCFReferenceSolver("selected_ci_pyscf").solve(
        hydrogen.h1,
        hydrogen.h2,
        hydrogen.e_core,
        norb=2,
        nelec=(1, 1),
        context=hydrogen.context,
        plan=plan,
        config=config,
        seed_data=None,
        seed=0,
    )
    assert result.t1_residual.tight_cutoff == 1e-4
    assert result.t1_residual.loose_cutoff is None
    assert result.t1_residual.reason == "paired_cutoff_not_run"
    assert result.uncertainty_mHa is None


@pytest.fixture(scope="module")
def water():
    from q2m3.molecule import MoleculeConfig
    from q2m3.sqd.integrals import build_integrals

    return build_integrals(
        MoleculeConfig(
            "water",
            ["O", "H", "H"],
            [[0.0, 0.0, 0.0], [0.13, 0.1, 0.96], [0.88, -0.12, -0.29]],
            0,
            6,
            6,
            basis="6-31g",
        ),
        host_available_mb=4096.0,
    )


def selected_candidates(data, tier):
    return tuple(
        replace(c, estimated_wall_s=1000.0) if c.tier != tier else c
        for c in mod().build_reference_candidates(data.norb, data.nelec, config=ReferenceConfig())
    )


def test_real_sci_residual_is_loose_error_not_tight_uncertainty(water):
    with pytest.warns(RuntimeWarning, match="reference"):
        result = run(water, candidates=selected_candidates(water, "T1"))
    residual = result.t1_residual
    assert residual.loose_residual_mHa > 1e-12
    assert residual.loose_residual_mHa == pytest.approx(
        1000 * abs(residual.loose_energy - residual.tight_energy)
    )
    assert result.energy == residual.tight_energy
    assert result.uncertainty_mHa is None


def test_rotated_ccsd_seed_preserves_triples_energy(water):
    import numpy as np

    from q2m3.sqd.ansatz import build_ccsd_seed
    from q2m3.sqd.integrals import hamiltonian_id

    seed = build_ccsd_seed(water, host_available_mb=4096.0)
    rng = np.random.default_rng(743)
    u_occ, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    u_virt, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    u = np.zeros((6, 6))
    u[:3, :3], u[3:, 3:] = u_occ, u_virt
    h1 = u.T @ water.h1 @ u
    h2 = np.einsum("pi,qj,rk,sl,pqrs->ijkl", u, u, u, u, water.h2, optimize=True)
    digest = hamiltonian_id(h1, h2, water.e_core, norb=6, nelec=(3, 3))
    context = replace(water.context, hamiltonian_id=digest, frame_id="rotated")
    rotated = replace(water, h1=h1, h2=h2, context=context)
    rotated_seed = replace(
        seed,
        t1=u_occ.T @ seed.t1 @ u_virt,
        t2=np.einsum(
            "pi,qj,ra,sb,pqrs->ijab", u_occ, u_occ, u_virt, u_virt, seed.t2, optimize=True
        ),
        hamiltonian_id=digest,
        frame_id="rotated",
    )
    with pytest.warns(RuntimeWarning, match="reference"):
        canonical = run(water, seed_data=seed, candidates=selected_candidates(water, "T2"))
    with pytest.warns(RuntimeWarning, match="reference"):
        actual = run(rotated, seed_data=rotated_seed, candidates=selected_candidates(rotated, "T2"))
    assert actual.energy == pytest.approx(canonical.energy, abs=1e-10)
    assert actual.t2.t1_diagnostic == pytest.approx(np.linalg.norm(seed.t1) / np.sqrt(6), abs=1e-12)
    assert abs(actual.t2.triples_correction_ha) > 1e-8
    with pytest.warns(RuntimeWarning, match="reference"):
        no_seed = run(rotated, candidates=selected_candidates(rotated, "T2"))
    assert no_seed.energy == pytest.approx(canonical.energy, abs=1e-10)


def test_forged_candidate_cannot_bypass_retained_allocation_guard(hydrogen):
    from q2m3.sqd.exceptions import ResourceLimitError

    m = mod()
    candidate = m.ReferenceCandidate("T0", "exact_casci", True, None, 0.001, 1.0, "fake", True)
    import os

    from q2m3.sqd.resources import estimate_rss_mb, load_resource_model

    inventory = estimate_rss_mb(
        2, (1, 1), stage="reference", solver_method="exact_casci", model=load_resource_model()
    )
    retained = max(0.0, 4096.0 - 2 * m._rss_mb(os.getpid()) - inventory + 100.0)
    with pytest.raises(ResourceLimitError):
        run(hydrogen, retained_mb=retained, candidates=(candidate,))


@pytest.mark.parametrize("reason", [None, "", "   ", 1, object(), []])
def test_unavailable_candidate_requires_nonempty_string_reason(reason):
    with pytest.raises(ValueError, match="reason"):
        mod().ReferenceCandidate("T0", "exact_casci", False, reason, None, None, "fixture", False)


def test_overflowing_t1_norm_is_numerical_error(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from q2m3.sqd import ansatz
    from q2m3.sqd.exceptions import ReferenceNumericalError

    class FiniteAmplitudeSolver:
        converged = True

        def kernel(self, **kwargs):
            return -0.1, np.array([[1e308]]), np.zeros((1, 1, 1, 1))

        def ccsd_t(self, **kwargs):
            return 0.0

    monkeypatch.setattr(ansatz, "_solver", lambda data: (FiniteAmplitudeSolver(), None, None, None))
    context = SimpleNamespace(hamiltonian_id="fixture", frame_id="fixture")
    plan = SimpleNamespace(tier="T2", method="ccsd_t", downgrade_reason="fixture", attempts=())
    with pytest.raises(ReferenceNumericalError, match="diagnostic"):
        mod()._PySCFReferenceSolver._triples(
            np.diag([-1.0, 1.0]), np.zeros((2, 2, 2, 2)), 0.0, 2, (1, 1), context, plan, None
        )
