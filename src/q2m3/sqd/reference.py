"""Budgeted reference ladder with pure planning and supervised Linux execution.

Wall predictions are versioned engineering work-count estimates, not calibrated
upper bounds. RSS models have explicitly bounded domains; a process-tree monitor
checks actual decimal MB every 10 ms and terminates overruns. Polling cannot
observe arbitrary instantaneous peaks. Numerical failures never trigger fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import comb, isfinite
from typing import Protocol

import numpy as np

from q2m3.sqd.config import CCSDSeed, IntegralContext, ReferenceConfig, Tier, validate_active_space
from q2m3.sqd.exceptions import BaselineUnavailableError
from q2m3.sqd.result import ReferenceAttempt, ReferenceResult

_METHODS = {
    "T0": ("exact_casci",),
    "T1": ("selected_ci_pyscf",),
    "T1+": ("shci_dice", "dmrg_block2"),
    "T2": ("ccsd_t",),
}
T1_FULL_NDET_MAX = 2_000_000_000


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ReferenceCandidate:
    """Externally established solver availability and resource prediction."""

    tier: Tier
    method: str
    available: bool
    unavailable_reason: str | None
    estimated_wall_s: float | None
    estimated_rss_mb: float | None
    model_id: str
    in_domain: bool

    def __post_init__(self):
        if self.tier not in _METHODS or self.method not in _METHODS[self.tier]:
            raise ValueError("Candidate tier/method mismatch")
        if not isinstance(self.available, bool) or not isinstance(self.in_domain, bool):
            raise ValueError("Candidate flags must be boolean")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("Candidate model_id required")
        if not self.available and (
            not isinstance(self.unavailable_reason, str) or not self.unavailable_reason.strip()
        ):
            raise ValueError("Unavailable candidate requires reason")
        for name in ("estimated_wall_s", "estimated_rss_mb"):
            value = getattr(self, name)
            if value is not None:
                _positive(value, name, zero=True)
            elif self.in_domain:
                raise ValueError("In-domain candidates require finite costs")


@dataclass(frozen=True)
class ReferencePlan:
    """Immutable selection with its effective budgets and ordered audit trail."""

    tier: Tier
    method: str
    remaining_wall_s: float
    rss_budget_mb: float
    estimated_wall_s: float
    estimated_rss_mb: float
    model_id: str
    downgrade_reason: str | None
    candidates: tuple[ReferenceCandidate, ...]
    attempts: tuple[ReferenceAttempt, ...]


class ReferenceSolver(Protocol):
    """Optional solver capability; execution occurs inside the budget supervisor."""

    method: str

    def solve(
        self,
        h1: np.ndarray,
        h2: np.ndarray,
        e_core: float,
        *,
        norb: int,
        nelec: tuple[int, int],
        context: IntegralContext,
        plan: ReferencePlan,
        config: ReferenceConfig,
        seed_data: CCSDSeed | None,
        seed: int,
    ) -> ReferenceResult: ...


def resolve_reference(
    norb: int,
    nelec: tuple[int, int],
    *,
    config: ReferenceConfig,
    candidates: tuple[ReferenceCandidate, ...],
    remaining_wall_s: float,
    rss_budget_mb: float,
    attempts: tuple[ReferenceAttempt, ...] = (),
) -> ReferencePlan:
    """Choose the first eligible solver without imports, probing, or other I/O.

    Wall equality is permitted; RSS equality is refused. Supplied attempts retain
    execution failures, preventing a timed-out or unavailable method being retried.
    """
    validate_active_space(norb, nelec)
    config.validate()
    _positive(remaining_wall_s, "remaining_wall_s", zero=True)
    _positive(rss_budget_mb, "rss_budget_mb")
    if any(not isinstance(c, ReferenceCandidate) for c in candidates):
        raise ValueError("candidates must contain ReferenceCandidate")
    by_method = {c.method: c for c in candidates}
    if len(by_method) != len(candidates):
        raise ValueError("Duplicate candidate method")
    for attempt in attempts:
        attempt.validate()
    wall = min(config.wall_budget_s, remaining_wall_s)
    rss = min(config.rss_budget_mb, rss_budget_mb)
    failed = {a.method for a in attempts if a.outcome in ("timeout", "unavailable")}
    trail = list(attempts)
    order = [("T0", "exact_casci"), ("T1", "selected_ci_pyscf")]
    order += [("T1+", name) for name in config.plugins]
    order += [("T2", "ccsd_t")]
    reasons = [a.reason for a in attempts if a.reason]
    for tier, method in order:
        if method in failed:
            continue
        candidate = by_method.get(method)
        if tier not in config.allowed_tiers:
            outcome, reason = "disabled", f"{method}: tier disabled"
        elif candidate is None or not candidate.available:
            outcome = "unavailable"
            reason = f"{method}: " + (candidate.unavailable_reason if candidate else "no candidate")
        elif not candidate.in_domain or (
            tier == "T1" and comb(norb, nelec[0]) ** 2 > T1_FULL_NDET_MAX
        ):
            outcome, reason = "out_of_domain", f"{method}: outside cost model domain"
        elif candidate.estimated_wall_s > wall or candidate.estimated_rss_mb >= rss:
            outcome, reason = "over_budget", f"{method}: predicted costs exceed remaining budget"
        else:
            trail.append(ReferenceAttempt(tier, method, "selected", None, 0.0, None))
            return ReferencePlan(
                tier,
                method,
                wall,
                rss,
                candidate.estimated_wall_s,
                candidate.estimated_rss_mb,
                candidate.model_id,
                None if tier == "T0" else "; ".join(reasons),
                tuple(candidates),
                tuple(trail),
            )
        reasons.append(reason)
        trail.append(ReferenceAttempt(tier, method, outcome, reason, 0.0, None))
    error = BaselineUnavailableError("; ".join(reasons) or "No eligible baseline")
    error.attempts = tuple(trail)
    raise error


@dataclass(frozen=True)
class _TriplesInventory:
    """Serial PySCF 2.14 RCCSD(T), <=10 orbitals, 200 cycles, async_io=False.

    RCCSD follows the fixed-frame 128 n^4 inventory. Triples adds 16 n^4,
    16 n^3, 8 n^2 doubles, plus 56-byte CacheJob and 24-byte permutation
    inventories bounded by n^3. Sources: pyscf/cc/ccsd_t.py and lib/cc/ccsd_t.c.
    The 1024 MB native reserve is an engineering allowance, not certification.
    """

    model_id: str = "serial-rccsd-t-inventory-v1"

    def upper_bound_mb(self, norb, nelec, *, stage, subspace_dims, retained_mb, solver_method):
        import platform
        from importlib.metadata import version

        from q2m3.sqd.exceptions import ResourceModelDomainError

        if (
            stage != "reference"
            or solver_method != "ccsd_t"
            or not 2 <= norb <= 10
            or version("pyscf") != "2.14.0"
            or platform.system() != "Linux"
            or platform.machine() != "x86_64"
        ):
            raise ResourceModelDomainError("RCCSD(T) inventory outside PySCF 2.14/norb<=10")
        n = norb
        inventory = 8 * (144 * n**4 + 16 * n**3 + 73 * n**2) + 80 * n**3
        return 1024.0 + retained_mb + inventory / 1e6


def _wall_estimate(norb, nelec, method):
    # Fixed 1e8 scalar-work/s proxy; 100 Davidson or 200 RCCSD iterations.
    # Hardware/conditioning can invalidate this estimate: enforce real deadline.
    dim = comb(norb, nelec[0]) ** 2
    work = {
        "exact_casci": 100 * dim * norb**4,
        "selected_ci_pyscf": 2 * 100 * dim * norb**4,
        "ccsd_t": 20 * 200 * norb**6 + norb**7,
    }[method]
    return 2.0 + work / 1e8


def _rss_mb(pid):
    from pathlib import Path

    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024 / 1e6
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0.0


def build_reference_candidates(
    norb: int,
    nelec: tuple[int, int],
    *,
    config: ReferenceConfig,
    retained_mb: float = 0.0,
    plugins: Mapping[str, ReferenceSolver] | None = None,
) -> tuple[ReferenceCandidate, ...]:
    """Probe bounded RSS models and explicitly registered optional capabilities.

    Builtin wall costs use a fixed engineering operation-count proxy, not a
    calibrated timing bound. Optional capabilities supply a ``candidate`` record;
    absent records are outside the cost domain. No binary is implicitly launched.
    Parent RSS is added to the solver inventory for simultaneous process residency.
    """
    import os

    from q2m3.sqd.exceptions import ResourceModelDomainError
    from q2m3.sqd.resources import estimate_rss_mb, load_resource_model

    validate_active_space(norb, nelec)
    config.validate()
    _positive(retained_mb, "retained_mb", zero=True)
    plugins = plugins or {}
    if any(name not in _METHODS["T1+"] for name in plugins):
        raise ValueError("Unknown plugin registration")
    candidates = []
    for tier, method in [("T0", "exact_casci"), ("T1", "selected_ci_pyscf"), ("T2", "ccsd_t")]:
        model_id = "engineering-wall-1e8-v1"
        try:
            model = (
                _TriplesInventory()
                if tier == "T2"
                else load_resource_model(profile={"max_space": 12})
            )
            if tier == "T1" and config.sci_cutoffs != (1e-4, 1e-5):
                raise ResourceModelDomainError("Selected-CI cutoffs outside calibrated profile")
            rss = estimate_rss_mb(
                norb,
                nelec,
                stage="reference",
                solver_method=method,
                model=model,
                retained_mb=retained_mb + 2 * _rss_mb(os.getpid()),
            )
            wall = _wall_estimate(norb, nelec, method)
            domain = True
            model_id += "/" + model.model_id
        except ResourceModelDomainError:
            rss = wall = None
            domain = False
        candidates.append(ReferenceCandidate(tier, method, True, None, wall, rss, model_id, domain))
    for name in config.plugins:
        plugin = plugins.get(name)
        if plugin is None:
            candidates.append(
                ReferenceCandidate(
                    "T1+",
                    name,
                    False,
                    "optional capability not registered",
                    None,
                    None,
                    "unregistered-plugin",
                    False,
                )
            )
        else:
            candidate = getattr(plugin, "candidate", None)
            if candidate is None:
                candidate = ReferenceCandidate(
                    "T1+", name, True, None, None, None, "plugin-cost-unavailable", False
                )
            if not isinstance(candidate, ReferenceCandidate) or candidate.method != name:
                raise ValueError("Plugin candidate identity mismatch")
            candidates.append(candidate)
    return tuple(candidates)


def t2_diagnostics(
    t1_diagnostic: float,
    ccsd_converged: bool,
    triples_correction_ha: float,
    ccsd_correlation_energy_ha: float,
):
    """Compute three independent reliability flags without NaN/zero substitutes."""
    from q2m3.sqd.result import T2Diagnostics

    small = abs(ccsd_correlation_energy_ha) <= 1e-12
    ratio = None if small else abs(triples_correction_ha / ccsd_correlation_energy_ha)
    return T2Diagnostics(
        t1_diagnostic,
        ccsd_converged,
        triples_correction_ha,
        ccsd_correlation_energy_ha,
        ratio,
        t1_diagnostic > 0.02,
        not ccsd_converged,
        small or ratio > 0.10,
        "correlation_denominator_too_small" if small else None,
    ).validate()


def _reference_result(energy, context, plan, *, residual=None, t2=None):
    exact = plan.tier == "T0"
    return ReferenceResult(
        float(energy),
        plan.tier,
        plan.method,
        0.0 if exact else None,
        "exact_active_space" if exact else "unknown",
        (
            None
            if exact
            else ("tight_cutoff_error_unknown" if plan.tier == "T1" else "ccsd_t_error_unknown")
        ),
        plan.downgrade_reason,
        residual,
        t2,
        bool(t2 and (t2.high_t1 or t2.ccsd_not_converged or t2.high_triples)),
        context.hamiltonian_id,
        context.frame_id,
        plan.attempts,
        () if exact else (f"WARNING: downgraded reference: {plan.downgrade_reason}",),
    ).validate()


class _PySCFReferenceSolver:
    """Builtin exact, adaptive selected-CI, and same-frame RCCSD(T) capability.

    Use ``run_reference`` for supervised execution, including hard deadline and
    total process RSS enforcement. This capability performs allocation guards but
    is deliberately the worker interface, not an independent execution supervisor.
    """

    def __init__(self, method: str):
        if method not in ("exact_casci", "selected_ci_pyscf", "ccsd_t"):
            raise ValueError("Unknown builtin method")
        self.method = method

    def solve(self, h1, h2, e_core, *, norb, nelec, context, plan, config, seed_data, seed):
        from time import monotonic

        from pyscf import fci, lib

        from q2m3.sqd.exceptions import ReferenceNumericalError, ReferenceTimeoutError
        from q2m3.sqd.resources import guard_allocation, load_resource_model
        from q2m3.sqd.result import T1Residual

        if lib.num_threads() != 1:
            raise ValueError("Reference inventory requires one numerical thread")
        if self.method != plan.method:
            raise ValueError("Solver/plan mismatch")
        model = (
            _TriplesInventory()
            if self.method == "ccsd_t"
            else load_resource_model(profile={"max_space": 12})
        )

        def guard():
            guard_allocation(
                norb,
                nelec,
                stage="reference",
                model=model,
                solver_method=self.method,
                host_available_mb=plan.rss_budget_mb,
                rss_budget_mb=plan.rss_budget_mb,
            )

        if self.method == "selected_ci_pyscf" and config.sci_cutoffs != (1e-4, 1e-5):
            from q2m3.sqd.exceptions import ResourceModelDomainError

            raise ResourceModelDomainError("Selected-CI cutoffs outside calibrated profile")
        guard()
        h1, h2 = (np.ascontiguousarray(a, dtype=np.float64) for a in (h1, h2))
        if self.method == "ccsd_t":
            return self._triples(h1, h2, e_core, norb, nelec, context, plan, seed_data)
        if self.method == "exact_casci":
            solver = fci.direct_spin1.FCI()
            energy, _ = solver.kernel(
                h1, h2, norb, nelec, ecore=e_core, max_space=12, max_cycle=100, tol=1e-12
            )
            if not solver.converged or not np.isfinite(energy):
                raise ReferenceNumericalError("Exact FCI did not converge to finite energy")
            return _reference_result(energy, context, plan)
        started = monotonic()
        energies = []
        for cutoff in config.sci_cutoffs:
            # Each cutoff has its own remaining-time preflight. The real outer
            # deadline still protects against an underestimated execution cost.
            if (
                monotonic() - started + _wall_estimate(norb, nelec, self.method) / 2
                > plan.remaining_wall_s
            ):
                if energies:
                    break
                raise ReferenceTimeoutError("Insufficient time for first selected-CI cutoff")
            guard()
            solver = fci.selected_ci.SelectedCI()
            solver.select_cutoff = cutoff
            solver.ci_coeff_cutoff = cutoff
            energy, _ = solver.kernel(
                h1, h2, norb, nelec, ecore=e_core, max_space=12, max_cycle=100, tol=1e-12
            )
            if not solver.converged or not np.isfinite(energy):
                raise ReferenceNumericalError("Selected-CI did not converge to finite energy")
            energies.append(float(energy))
        if len(energies) == 2:
            residual = T1Residual(
                *config.sci_cutoffs, *energies, 1000 * abs(energies[0] - energies[1]), None
            )
        else:
            residual = T1Residual(
                None, config.sci_cutoffs[0], None, energies[0], None, "paired_cutoff_not_run"
            )
        return _reference_result(energies[-1], context, plan, residual=residual)

    @staticmethod
    def _triples(h1, h2, e_core, norb, nelec, context, plan, seed_data):
        from q2m3.sqd.ansatz import _ActiveHamiltonian, _hf_fock, _solver
        from q2m3.sqd.exceptions import CCSDConvergenceError, ReferenceNumericalError

        data = _ActiveHamiltonian(h1, h2, e_core, norb, nelec, context)
        hf, fock = _hf_fock(data)
        if norb == nelec[0]:
            return _reference_result(hf, context, plan, t2=t2_diagnostics(0.0, True, 0.0, 0.0))
        # Semicanonicalize only within occupied/virtual blocks: no SCF, no
        # changed determinant/span. (T) requires diagonal block denominators.
        o = nelec[0]
        _, u_occ = np.linalg.eigh(fock[:o, :o])
        _, u_virt = np.linalg.eigh(fock[o:, o:])
        rotation = np.zeros((norb, norb))
        rotation[:o, :o], rotation[o:, o:] = u_occ, u_virt
        rotated_h1 = rotation.T @ h1 @ rotation
        rotated_h2 = np.einsum(
            "pi,qj,rk,sl,pqrs->ijkl", rotation, rotation, rotation, rotation, h2, optimize=True
        )
        data = _ActiveHamiltonian(rotated_h1, rotated_h2, e_core, norb, nelec, context)
        try:
            solver, eris, _, _ = _solver(data)
        except CCSDConvergenceError as exc:
            raise ReferenceNumericalError(str(exc)) from exc
        solver.async_io = False
        solver.max_cycle = 200
        if seed_data is None:
            correlation, t1, t2 = solver.kernel(eris=eris)
            converged = bool(solver.converged)
        else:
            t1 = u_occ.T @ seed_data.t1 @ u_virt
            t2 = np.einsum(
                "pi,qj,ra,sb,pqrs->ijab", u_occ, u_occ, u_virt, u_virt, seed_data.t2, optimize=True
            )
            correlation = seed_data.ccsd_energy - seed_data.hf_energy
            converged = seed_data.converged
        if not (np.isfinite(correlation) and np.all(np.isfinite(t1)) and np.all(np.isfinite(t2))):
            raise ReferenceNumericalError("Nonfinite RCCSD result")
        t1 = np.ascontiguousarray(t1, dtype=np.float64)
        t2 = np.ascontiguousarray(t2, dtype=np.float64)
        triples = float(solver.ccsd_t(t1=t1, t2=t2, eris=eris))
        if not np.isfinite(triples):
            raise ReferenceNumericalError("Nonfinite triples correction")
        diagnostic = float(np.linalg.norm(t1) / np.sqrt(2 * nelec[0]))
        if not np.isfinite(diagnostic):
            raise ReferenceNumericalError("Nonfinite T1 diagnostic from RCCSD amplitudes")
        diagnostics = t2_diagnostics(
            diagnostic,
            converged,
            triples,
            float(correlation),
        )
        return _reference_result(hf + correlation + triples, context, plan, t2=diagnostics)


def _tree_pids(pid):
    from pathlib import Path

    found, todo = {pid}, [pid]
    while todo:
        current = todo.pop()
        try:
            for path in Path(f"/proc/{current}/task").glob("*/children"):
                try:
                    children = {int(p) for p in path.read_text().split()} - found
                except (FileNotFoundError, ProcessLookupError):
                    continue
                found.update(children)
                todo.extend(children)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def _worker(connection, solver, args, kwargs):
    import os

    os.setsid()
    try:
        result = solver.solve(*args, **kwargs)
        connection.send(("result", result))
    except Exception as exc:
        # Transport errors with their original type; only the parent decides
        # whether that explicit type is a permitted fallback. Never suppress it.
        connection.send(("error", exc))
    finally:
        connection.close()


def _supervise(solver, args, kwargs, deadline, rss_cap):
    import multiprocessing
    import os
    import signal
    from time import monotonic, sleep

    from q2m3.sqd.exceptions import (
        ReferenceNumericalError,
        ReferenceTimeoutError,
        ResourceLimitError,
    )

    ctx = multiprocessing.get_context("fork")
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_worker, args=(sender, solver, args, kwargs))
    start, peak = monotonic(), _rss_mb(os.getpid())
    if peak >= rss_cap:
        raise ResourceLimitError("Reference parent already reaches RSS cap")
    process.start()
    sender.close()
    try:
        while True:
            peak = max(peak, sum(_rss_mb(pid) for pid in _tree_pids(os.getpid())))
            if peak >= rss_cap:
                raise ResourceLimitError(f"Reference process tree RSS={peak:.6f} MB >= {rss_cap}")
            if monotonic() >= deadline:
                raise ReferenceTimeoutError("Reference shared wall deadline exhausted")
            if receiver.poll():
                try:
                    kind, value = receiver.recv()
                except EOFError as exc:
                    raise ReferenceNumericalError(
                        "Reference worker closed without a result"
                    ) from exc
                if kind == "error":
                    raise value
                return value, monotonic() - start, peak
            if not process.is_alive():
                raise ReferenceNumericalError(f"Reference worker exited {process.exitcode}")
            sleep(0.01)
    except Exception as exc:
        exc.wall_s = monotonic() - start
        exc.peak_rss_mb = peak
        raise
    finally:
        # Terminate descendants too; a plugin cannot leave a solver child behind.
        if process.pid:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                if process.is_alive():
                    process.kill()
        process.join(timeout=1.0)
        receiver.close()
        process.close()


def run_reference(
    h1: np.ndarray,
    h2: np.ndarray,
    e_core: float,
    *,
    norb: int,
    nelec: tuple[int, int],
    context: IntegralContext,
    config: ReferenceConfig | None = None,
    seed_data: CCSDSeed | None = None,
    seed: int = 0,
    host_available_mb: float,
    rss_budget_mb: float | None = None,
    remaining_wall_s: float | None = None,
    retained_mb: float = 0.0,
    plugins: Mapping[str, ReferenceSolver] | None = None,
    candidates: tuple[ReferenceCandidate, ...] | None = None,
) -> ReferenceResult:
    """Execute the reference ladder with cumulative wall and process-tree RSS caps.

    Args:
        h1: Real same-frame one-electron Hamiltonian.
        h2: Chemist-ordered real two-electron tensor.
        e_core: Core constant, included exactly once, in Hartree.
        norb: Active spatial orbitals.
        nelec: Balanced alpha/beta electron counts.
        context: Immutable Hamiltonian and frame provenance.
        config: Reference budgets, cutoffs, and allowed capabilities.
        seed_data: Optional converged seed, verified before any reference attempt.
        seed: Nonnegative random seed forwarded to optional solvers.
        host_available_mb: Approved total run cap, decimal MB.
        rss_budget_mb: Additional total RSS cap.
        remaining_wall_s: Remaining whole-run wall budget, including planning.
        retained_mb: Additional live arrays outside builtin inventories.
        plugins: Explicit optional solver registrations, with candidate metadata.
        candidates: Caller-provided planning facts; runtime guards remain active.

    Returns:
        Validated reference energy, independent diagnostics, and audited attempts.

    Raises:
        BaselineUnavailableError: All allowed candidates are unreachable.
        ReferenceNumericalError: A solver returns invalid/nonfinite/mismatched data.
        ResourceLimitError: Actual process tree or allocation exceeds its cap.
        ValueError: Input structure/configuration or execution platform is invalid.
    """
    import os
    import platform
    import warnings
    from pathlib import Path
    from time import monotonic

    from q2m3.sqd.config import validate_integral_inputs
    from q2m3.sqd.exceptions import (
        ReferenceNumericalError,
        ReferenceTimeoutError,
        ReferenceUnavailableError,
        ResourceLimitError,
    )
    from q2m3.sqd.integrals import hamiltonian_id

    start = monotonic()
    config = (config or ReferenceConfig()).validate()
    _positive(host_available_mb, "host_available_mb")
    wall = config.wall_budget_s if remaining_wall_s is None else remaining_wall_s
    _positive(wall, "remaining_wall_s")
    deadline = start + min(wall, config.wall_budget_s)
    caps = [config.rss_budget_mb, host_available_mb, 8192.0]
    if rss_budget_mb is not None:
        _positive(rss_budget_mb, "rss_budget_mb")
        caps.append(rss_budget_mb)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Reference supervisor requires Linux x86_64 /proc and fork")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise ValueError(f"Reference inventory requires {variable}=1 before interpreter start")
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            caps.append(int(line.split()[1]) * 1024 / 1e6)
            break
    cap = min(caps)
    _positive(retained_mb, "retained_mb", zero=True)
    if _rss_mb(os.getpid()) + retained_mb >= cap:
        raise ResourceLimitError("Reference retained parent memory exceeds cap")
    validate_integral_inputs(
        h1,
        h2,
        e_core,
        norb=norb,
        nelec=nelec,
        context=context,
        seed_data=seed_data,
        reference=config,
        seed=seed,
        mode="reference_only" if seed_data is None else "full",
    )
    if hamiltonian_id(h1, h2, e_core, norb=norb, nelec=nelec) != context.hamiltonian_id:
        raise ReferenceNumericalError("Actual Hamiltonian differs from declared context")
    if seed_data is not None:
        # Seed verification is numerical work too: run it in the same supervisor.
        from q2m3.sqd.ansatz import _SeedInventory
        from q2m3.sqd.resources import guard_allocation

        guard_allocation(
            norb,
            nelec,
            stage="ccsd",
            model=_SeedInventory(norb),
            host_available_mb=cap,
            rss_budget_mb=cap,
            retained_mb=retained_mb + 2 * _rss_mb(os.getpid()),
        )
        _supervise(
            _SeedValidator(),
            (h1, h2, e_core),
            dict(
                norb=norb,
                nelec=nelec,
                context=context,
                seed_data=seed_data,
                host_available_mb=cap,
                rss_budget_mb=cap,
            ),
            deadline,
            cap,
        )
    plugins = plugins or {}
    available = (
        build_reference_candidates(
            norb, nelec, config=config, retained_mb=retained_mb, plugins=plugins
        )
        if candidates is None
        else candidates
    )
    attempts = ()
    while True:
        plan = resolve_reference(
            norb,
            nelec,
            config=config,
            candidates=available,
            remaining_wall_s=max(0.0, deadline - monotonic()),
            rss_budget_mb=cap,
            attempts=attempts,
        )
        solver = (
            plugins.get(plan.method) if plan.tier == "T1+" else _PySCFReferenceSolver(plan.method)
        )
        if solver is None:
            attempts = plan.attempts + (
                ReferenceAttempt(
                    plan.tier,
                    plan.method,
                    "unavailable",
                    "Optional capability is not registered",
                    0.0,
                    None,
                ),
            )
            continue
        if getattr(solver, "method", None) != plan.method:
            raise ValueError("Registered solver identity differs from selected method")
        if plan.tier != "T1+":
            from q2m3.sqd.resources import guard_allocation, load_resource_model

            model = (
                _TriplesInventory()
                if plan.tier == "T2"
                else load_resource_model(profile={"max_space": 12})
            )
            guard_allocation(
                norb,
                nelec,
                stage="reference",
                model=model,
                solver_method=plan.method,
                host_available_mb=cap,
                rss_budget_mb=cap,
                retained_mb=retained_mb + 2 * _rss_mb(os.getpid()),
            )
        elif plan.estimated_rss_mb + retained_mb + 2 * _rss_mb(os.getpid()) >= cap:
            raise ResourceLimitError("Plugin fork inventory exceeds process-tree budget")
        try:
            result, elapsed, peak = _supervise(
                solver,
                (h1, h2, e_core),
                dict(
                    norb=norb,
                    nelec=nelec,
                    context=context,
                    plan=plan,
                    config=config,
                    seed_data=seed_data,
                    seed=seed,
                ),
                deadline,
                cap,
            )
        except (ReferenceUnavailableError, ReferenceTimeoutError) as exc:
            outcome = "timeout" if isinstance(exc, ReferenceTimeoutError) else "unavailable"
            attempts = plan.attempts + (
                ReferenceAttempt(
                    plan.tier,
                    plan.method,
                    outcome,
                    str(exc),
                    getattr(exc, "wall_s", 0.0),
                    getattr(exc, "peak_rss_mb", None),
                ),
            )
            continue
        if not isinstance(result, ReferenceResult):
            raise ReferenceNumericalError("Solver did not return ReferenceResult")
        try:
            result.validate()
        except ValueError as exc:
            raise ReferenceNumericalError(f"Invalid reference result: {exc}") from exc
        if (
            result.hamiltonian_id != context.hamiltonian_id
            or result.frame_id != context.frame_id
            or result.tier != plan.tier
            or result.method != plan.method
        ):
            raise ReferenceNumericalError(
                "Returned reference identity differs from selected inputs"
            )
        result = replace(
            result,
            attempts=plan.attempts
            + (ReferenceAttempt(plan.tier, plan.method, "success", None, elapsed, peak),),
        )
        if result.tier != "T0":
            warnings.warn(
                f"WARNING: downgraded reference: {result.downgrade_reason}",
                RuntimeWarning,
                stacklevel=2,
            )
        return result.validate()


class _SeedValidator:
    """Run mandatory seed physics reception under the same execution supervisor."""

    def solve(self, *args, **kwargs):
        from q2m3.sqd.ansatz import validate_ccsd_seed_from_integrals

        validate_ccsd_seed_from_integrals(*args, **kwargs)
        return None
