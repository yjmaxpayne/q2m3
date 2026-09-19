"""Same-Hamiltonian, spin-matched four-arm observables.

Comparison selection, seed reception, and fixed-space solves share one supervised
wall deadline and process-tree RSS cap. SCI selection uses PySCF selected CI;
missing strings are filled by an explicitly recorded determinant-energy heuristic.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from q2m3.sqd.config import (
    CCSDSeed,
    IntegralContext,
    _finite,
    _freeze_snapshot,
    _integer,
    validate_integral_inputs,
)
from q2m3.sqd.diagonalize import DiagonalizationResult, _strings, kernel_fixed_space
from q2m3.sqd.exceptions import (
    ComparisonUnavailableError,
    ProvenanceMismatchError,
    ReferenceNumericalError,
    ReferenceTimeoutError,
    ResourceLimitError,
)
from q2m3.sqd.reference import _positive, _rss_mb, _supervise
from q2m3.sqd.resources import guard_allocation, load_resource_model
from q2m3.sqd.result import ReferenceResult
from q2m3.sqd.sampling import validate_samples


@dataclass(frozen=True)
class ComparisonResult:
    """Immutable observables and complete comparison selection provenance.

    Energies include the supplied core constant once, in Hartree. Timings include
    pool construction and physical seed/returned SQD verification. Saturation
    points count sampled pairs, whereas subspace_dim is a Cartesian product.
    """

    sqd_energy: float
    iso_active_space_ccsd_energy: float
    iso_ndet_sci_energy: float
    iso_ndet_random_energy: float | None
    delta_mHa: float
    delta_vs_sci_mHa: float
    ratio_sqd_over_sci: float | None
    subspace_dims: tuple[int, int]
    subspace_dim: int
    unique_dets_vs_shots: tuple[tuple[int, int], ...]
    hamiltonian_id: str
    frame_id: str
    seed: int
    ci_strings: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    null_reasons: Mapping[str, str]
    allocation_audit: tuple[Mapping[str, Any], ...]
    comparison_wall_s: float
    peak_rss_mb: float

    def __post_init__(self):
        for name in ("ci_strings", "diagnostics", "null_reasons", "allocation_audit"):
            object.__setattr__(self, name, _freeze_snapshot(getattr(self, name)))


def summarize_observables(
    *,
    sqd_energy: float | None,
    sci_energy: float | None,
    random_energy: float | None,
    hf_energy: float,
    baseline_energy: float,
    baseline_tier: str,
    mode: str = "full",
    random_reason: str | None = None,
) -> dict[str, Any]:
    """Compute scalar observables after the executor authenticates all frames.

    Args:
        sqd_energy: SQD energy in Ha, absent only in reference-only mode.
        sci_energy: Matched SCI energy in Ha.
        random_energy: Matched random energy, or explicitly unavailable.
        hf_energy: Same-Hamiltonian HF determinant expectation in Ha.
        baseline_energy: Reference energy in Ha.
        baseline_tier: T0, T1, T1+, or T2.
        mode: Full or reference_only.
        random_reason: Required when a full-mode random arm is unavailable.

    Returns:
        Gaps in mHa, dimensionless ratio, diagnostics, and null reasons.

    Raises:
        ValueError: Missing/nonfinite energy or invalid mode/tier.
        ReferenceNumericalError: Variational energy is below the T0 lower bound.
    """
    for name, energy in (("hf_energy", hf_energy), ("baseline_energy", baseline_energy)):
        _finite(energy, name)
    if baseline_tier not in ("T0", "T1", "T1+", "T2"):
        raise ValueError("Unknown baseline tier")
    if mode not in ("full", "reference_only"):
        raise ValueError("Unknown observable mode")
    if mode == "reference_only":
        if any(e is not None for e in (sqd_energy, sci_energy, random_energy)):
            raise ValueError("reference_only cannot contain comparison energies")
        names = (
            "sqd_energy",
            "delta_mHa",
            "delta_vs_sci_mHa",
            "ratio_sqd_over_sci",
            "iso_ndet_sci_energy",
            "iso_ndet_random_energy",
            "unique_dets_vs_shots",
            "subspace_dim",
            "subspace_dims",
            "n_reps",
            "shots",
            "backend",
        )
        return {
            **dict.fromkeys(names),
            "null_reasons": dict.fromkeys(names, "reference_only"),
            "diagnostics": {},
        }
    _finite(sqd_energy, "sqd_energy")
    _finite(sci_energy, "sci_energy")
    diagnostics, reasons = {}, {}
    sqd_gap, sci_gap = sqd_energy - baseline_energy, sci_energy - baseline_energy
    ratio = None
    if baseline_tier != "T0":
        reasons["ratio_sqd_over_sci"] = "non_exact_reference"
    else:
        if sqd_gap < -1e-10 or sci_gap < -1e-10:
            raise ReferenceNumericalError("Variational comparison below exact baseline")
        if sqd_gap < 0:
            diagnostics["ratio_tolerance_ha"] = 1e-10
        if sci_gap > 1e-10:
            ratio = max(0.0, sqd_gap) / sci_gap
        else:
            reasons["ratio_sqd_over_sci"] = "sci_denominator_too_small"
    quality = None
    if random_energy is None:
        if not isinstance(random_reason, str) or not random_reason.strip():
            raise ValueError("Unavailable random arm requires an explicit reason")
        reasons["iso_ndet_random_energy"] = random_reason
        diagnostics.update(random_degenerate=None, sampling_quality_reason=random_reason)
    else:
        _finite(random_energy, "random_energy")
        if baseline_tier == "T0" and random_energy - baseline_energy < -1e-10:
            raise ReferenceNumericalError("Random energy below exact baseline")
        degenerate = abs(random_energy - hf_energy) <= 1e-8
        diagnostics["random_degenerate"] = degenerate
        if degenerate:
            diagnostics["sampling_quality_reason"] = "random_degenerate"
        elif abs(random_energy - sci_energy) <= 1e-10:
            diagnostics["sampling_quality_reason"] = "random_sci_denominator_too_small"
        else:
            quality = (random_energy - sqd_energy) / (random_energy - sci_energy)
    diagnostics["sampling_quality"] = quality
    for value in (sqd_gap * 1000, (sqd_energy - sci_energy) * 1000, ratio, quality):
        if value is not None and not np.isfinite(value):
            raise ReferenceNumericalError("Nonfinite derived observable")
    return dict(
        delta_mHa=1000 * sqd_gap,
        delta_vs_sci_mHa=1000 * (sqd_energy - sci_energy),
        ratio_sqd_over_sci=ratio,
        diagnostics=diagnostics,
        null_reasons=reasons,
    )


def unique_dets_vs_shots(
    samples: np.ndarray, *, checkpoints: tuple[int, ...] | None = None
) -> tuple[tuple[int, int], ...]:
    """Count distinct sampled alpha/beta pairs at actual shot prefixes.

    Args:
        samples: Nonempty boolean matrix in beta-left/alpha-right big-endian order.
        checkpoints: Increasing prefixes; total shots is always appended.

    Returns:
        Cumulative (shots, unique pairs) points, including the final shot.

    Raises:
        ValueError: Invalid sample structure or prefix order.
    """
    if (
        not isinstance(samples, np.ndarray)
        or samples.dtype != np.dtype(bool)
        or samples.ndim != 2
        or not samples.shape[0]
        or not samples.shape[1]
        or samples.shape[1] % 2
    ):
        raise ValueError("samples must be a nonempty even-width boolean matrix")
    shots = len(samples)
    points = tuple(range(1, shots + 1)) if checkpoints is None else tuple(checkpoints)
    last = 0
    for point in points:
        _integer(point, "checkpoint", 1)
        if point <= last or point > shots:
            raise ValueError("checkpoints must increase within the shot count")
        last = point
    if not points or points[-1] != shots:
        points += (shots,)
    wanted, seen, curve = set(points), set(), []
    for index, sample in enumerate(samples, 1):
        seen.add(np.packbits(sample).tobytes())
        if index in wanted:
            curve.append((index, len(seen)))
    return tuple(curve)


def _sector(norb, electrons):
    from pyscf.fci import cistring

    return cistring.make_strings(range(norb), electrons)


def _random_strings(norb, nelec, dims, seed):
    rng = np.random.default_rng(seed)
    result = []
    for electrons, size in zip(nelec, dims, strict=True):
        hf = (1 << electrons) - 1
        pool = _sector(norb, electrons)
        others = pool[pool != hf]
        if size < 1 or size > len(pool):
            raise ComparisonUnavailableError("Random target exceeds legal spin sector")
        result.append(np.sort(np.append(rng.choice(others, size - 1, replace=False), hf)))
    return tuple(result)


def _select_sci(h1, h2, norb, nelec, dims):
    from pyscf import fci

    h1, h2 = (np.ascontiguousarray(a, dtype=np.float64) for a in (h1, h2))
    solver = fci.selected_ci.SelectedCI()
    solver.select_cutoff = solver.ci_coeff_cutoff = 1e-4
    energy, ci = solver.kernel(h1, h2, norb, nelec, max_space=12, max_cycle=100, tol=1e-12)
    if not solver.converged or not np.isfinite(energy) or not np.all(np.isfinite(ci)):
        raise ReferenceNumericalError("Comparison selected-CI did not converge")
    pools = tuple(_sector(norb, n) for n in nelec)
    # Ranking fallback: diagonal determinant energies against the other-spin HF.
    # The complete diagonal is bounded before entering this routine, not an FCI solve.
    diagonal = fci.direct_spin1.make_hdiag(h1, h2, norb, nelec).reshape(tuple(map(len, pools)))
    selected, expanded, rankings = [], [], []
    for spin, (electrons, size, pool) in enumerate(zip(nelec, dims, pools, strict=True)):
        hf = (1 << electrons) - 1
        if not 1 <= size <= len(pool):
            raise ComparisonUnavailableError("SCI target exceeds legal spin sector")
        weights = np.sum(np.abs(np.asarray(ci, dtype=np.float64)) ** 2, axis=1 - spin)
        if not np.all(np.isfinite(weights)):
            raise ReferenceNumericalError("Nonfinite SCI marginal weights")
        ranked = sorted(
            zip(map(int, ci._strs[spin]), map(float, weights), strict=False),
            key=lambda p: (-p[1], p[0]),
        )
        chosen = [hf] + [s for s, w in ranked if s != hf][: size - 1]
        hf_other = int(np.flatnonzero(pools[1 - spin] == (1 << nelec[1 - spin]) - 1)[0])
        diag = diagonal[:, hf_other] if spin == 0 else diagonal[hf_other, :]
        fallback = sorted(
            zip(map(int, pool), map(float, diag), strict=False), key=lambda p: (p[1], p[0])
        )
        additions = []
        for string, _ in fallback:
            if len(chosen) == size:
                break
            if string not in chosen:
                chosen.append(string)
                additions.append(string)
        if len(chosen) != size:
            raise ComparisonUnavailableError("SCI expansion failed to reach spin target")
        selected.append(np.asarray(sorted(chosen), dtype=np.int64))
        expanded.append(tuple(additions))
        rankings.append(tuple(ranked))
    return tuple(selected), dict(
        method="pyscf_selected_ci_marginal_weight",
        cutoff=1e-4,
        pool_dims=tuple(map(len, ci._strs)),
        pool_electronic_energy=float(energy),
        rankings=tuple(rankings),
        expanded_strings=tuple(expanded),
        expansion_heuristic="ascending_diagonal_energy_with_opposite_spin_HF_then_integer",
    )


def _random_wall_estimate(norb, dims):
    """Engineering planning proxy, not a certified runtime upper bound."""
    return 0.01 + 100 * math.prod(dims) * norb**4 / 1e8


class _ComparisonWorker:
    def solve(
        self,
        h1,
        h2,
        e_core,
        *,
        norb,
        nelec,
        context,
        seed_data,
        sqd,
        samples,
        seed,
        cap,
        retained_mb,
        deadline,
    ):
        from pyscf import lib

        from q2m3.sqd.ansatz import validate_ccsd_seed_from_integrals

        if lib.num_threads() != 1:
            raise ValueError("Comparison inventory requires one numerical thread")
        h1, h2 = (np.ascontiguousarray(a, dtype=np.float64) for a in (h1, h2))
        validate_ccsd_seed_from_integrals(
            h1,
            h2,
            e_core,
            norb=norb,
            nelec=nelec,
            context=context,
            seed_data=seed_data,
            host_available_mb=cap,
            rss_budget_mb=cap,
        )
        model = load_resource_model(profile={"max_space": 12, "shots": len(samples)})
        audit = []
        full_dims = tuple(math.comb(norb, n) for n in nelec)
        # Includes curve/set Python objects, selection CI, rankings and full hdiag,
        # returned/recomputed SQD and both comparison states retained concurrently.
        live = (
            retained_mb
            + (
                512 * len(samples)
                + 128 * math.prod(full_dims)
                + 64 * sum(full_dims)
                + 4 * sqd.amplitudes.nbytes
            )
            / 1e6
        )

        def guard(dims, phase):
            estimate = guard_allocation(
                norb,
                nelec,
                stage="comparison",
                model=model,
                subspace_dims=dims,
                solver_method="selected_ci_pyscf",
                retained_mb=live,
                host_available_mb=cap,
                rss_budget_mb=cap,
            )
            audit.append(
                dict(
                    stage="comparison",
                    phase=phase,
                    subspace_dims=dims,
                    retained_mb=live,
                    predicted_rss_mb=estimate,
                    model_id=model.model_id,
                )
            )

        guard(full_dims, "pool_construction")
        checked = kernel_fixed_space(
            h1,
            h2,
            e_core,
            sqd.ci_strings,
            norb=norb,
            nelec=nelec,
            host_available_mb=cap,
            rss_budget_mb=cap,
            retained_mb=live,
        )
        if (
            abs(checked.energy - sqd.energy) > 1e-10
            or abs(sqd.electronic_energy + e_core - sqd.energy) > 1e-10
        ):
            raise ReferenceNumericalError("SQD energy does not belong to supplied Hamiltonian")
        # Authenticate the actual returned vector, not only its scalar minimum.
        from pyscf.fci import direct_spin1, selected_ci

        ci = np.asarray(sqd.amplitudes, dtype=np.float64).view(selected_ci.SCIvector)
        ci._strs = sqd.ci_strings
        effective = direct_spin1.absorb_h1e(h1, h2, norb, nelec, 0.5)
        residual = float(
            np.linalg.norm(
                selected_ci.contract_2e(effective, ci, norb, nelec) - sqd.electronic_energy * ci
            )
        )
        if not np.isfinite(residual) or residual > 1e-6:
            raise ReferenceNumericalError("SQD projected residual exceeds 1e-6 Ha")
        sci_strings, trace = _select_sci(h1, h2, norb, nelec, sqd.subspace_dims)
        guard(_strings(sci_strings, norb, nelec), "sci_fixed_space")
        sci = kernel_fixed_space(
            h1,
            h2,
            e_core,
            sci_strings,
            norb=norb,
            nelec=nelec,
            host_available_mb=cap,
            rss_budget_mb=cap,
            retained_mb=live,
        )
        random_strings, random, random_reason = None, None, None
        if monotonic() + _random_wall_estimate(norb, sqd.subspace_dims) >= deadline:
            random_reason = "random_wall_budget"
        else:
            try:
                guard(sqd.subspace_dims, "random_fixed_space")
            except ResourceLimitError:
                random_reason = "random_rss_budget"
            else:
                random_strings = _random_strings(norb, nelec, sqd.subspace_dims, seed)
                random = kernel_fixed_space(
                    h1,
                    h2,
                    e_core,
                    random_strings,
                    norb=norb,
                    nelec=nelec,
                    host_available_mb=cap,
                    rss_budget_mb=cap,
                    retained_mb=live,
                )
        for result, requested in ((sci, sci_strings), (random, random_strings)):
            if result is not None and (
                result.subspace_dims != sqd.subspace_dims
                or any(
                    not np.array_equal(got, want)
                    for got, want in zip(result.ci_strings, requested, strict=True)
                )
            ):
                raise ComparisonUnavailableError("Comparison spin strings changed")
        return dict(
            sci_energy=sci.energy,
            random_energy=None if random is None else random.energy,
            random_reason=random_reason,
            ci_strings={
                "sqd": tuple(tuple(map(int, s)) for s in sqd.ci_strings),
                "sci": tuple(tuple(map(int, s)) for s in sci_strings),
                "random": (
                    None
                    if random_strings is None
                    else tuple(tuple(map(int, s)) for s in random_strings)
                ),
            },
            curve=unique_dets_vs_shots(samples),
            audit=tuple(audit),
            trace=trace,
            sqd_residual_ha=residual,
        )


def run_comparisons(
    h1: np.ndarray,
    h2: np.ndarray,
    e_core: float,
    *,
    norb: int,
    nelec: tuple[int, int],
    context: IntegralContext,
    seed_data: CCSDSeed,
    reference: ReferenceResult,
    sqd: DiagonalizationResult,
    samples: np.ndarray,
    seed: int,
    host_available_mb: float,
    rss_budget_mb: float = 8192.0,
    remaining_wall_s: float = 900.0,
    retained_mb: float = 0.0,
) -> ComparisonResult:
    """Execute four-arm comparisons in one bounded, same-frame worker.

    Args:
        h1: Real same-frame one-electron integrals in Ha.
        h2: Real chemist ERIs in Ha.
        e_core: Same-frame constant in Ha, added exactly once.
        norb: Number of active spatial orbitals.
        nelec: Balanced spin populations.
        context: Hamiltonian and frame provenance.
        seed_data: Converged seed; physically revalidated against supplied integrals.
        reference: Comparable baseline with matching frame/Hamiltonian IDs.
        sqd: Final returned SQD state, with its actual spin strings and amplitudes.
        samples: Valid boolean samples for the complete saturation trajectory.
        seed: Nonnegative random comparison seed, retained in the result.
        host_available_mb: Whole-process-tree RSS cap in decimal MB.
        rss_budget_mb: User RSS cap, limited by the default 8192 MB soft policy.
        remaining_wall_s: Shared remaining time including pool construction.
        retained_mb: Additional live caller arrays outside the operation inventory.

    Returns:
        Immutable four-arm values, selection trace, seeds and allocation evidence.

    Raises:
        ComparisonUnavailableError: Exact per-spin matching is impossible.
        ProvenanceMismatchError: Reference and inputs have different provenance.
        ReferenceNumericalError: A numerical or returned SQD consistency check fails.
        ReferenceTimeoutError: The shared wall deadline is exhausted.
        ResourceLimitError: A preflight or runtime RSS guard is exceeded.
    """
    started = monotonic()
    for name, value in (
        ("remaining_wall_s", remaining_wall_s),
        ("host_available_mb", host_available_mb),
        ("rss_budget_mb", rss_budget_mb),
    ):
        _positive(value, name)
    _positive(retained_mb, "retained_mb", zero=True)
    deadline = started + remaining_wall_s
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise ValueError(f"Comparison inventory requires {variable}=1 before interpreter start")
    caps = [host_available_mb, rss_budget_mb, 8192.0]
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            caps.append(int(line.split()[1]) * 1024 / 1e6)
            break
    cap = min(caps)
    validate_integral_inputs(
        h1, h2, e_core, norb=norb, nelec=nelec, context=context, seed_data=seed_data, seed=seed
    )
    reference.validate()
    if (reference.hamiltonian_id, reference.frame_id) != (context.hamiltonian_id, context.frame_id):
        raise ProvenanceMismatchError("Comparison reference has a different Hamiltonian/frame")
    validate_samples(samples, norb=norb, nelec=nelec, shots=len(samples))
    dims = _strings(sqd.ci_strings, norb, nelec)
    if dims != sqd.subspace_dims or any(
        (1 << n) - 1 not in s for n, s in zip(nelec, sqd.ci_strings, strict=False)
    ):
        raise ComparisonUnavailableError("Final SQD spin dimensions or HF inclusion invalid")
    if sqd.amplitudes.dtype.kind not in "fiu":
        raise ReferenceNumericalError("SQD amplitudes must be real numeric values")
    amplitudes = np.asarray(sqd.amplitudes, dtype=np.float64)
    if (
        amplitudes.shape != dims
        or not np.all(np.isfinite(amplitudes))
        or not np.isfinite(np.linalg.norm(amplitudes))
        or abs(np.linalg.norm(amplitudes) - 1) > 1e-8
    ):
        raise ReferenceNumericalError("Invalid SQD amplitudes")
    _finite(sqd.energy, "sqd.energy")
    _finite(sqd.electronic_energy, "sqd.electronic_energy")
    from q2m3.sqd.ansatz import _SeedInventory

    # Fork inherits all parent inputs; both process baselines stay live.
    retained = retained_mb + 2 * _rss_mb(os.getpid())
    guard_allocation(
        norb,
        nelec,
        stage="ccsd",
        model=_SeedInventory(norb),
        retained_mb=retained,
        host_available_mb=cap,
        rss_budget_mb=cap,
    )
    model = load_resource_model(profile={"max_space": 12, "shots": len(samples)})
    full_dims = tuple(math.comb(norb, n) for n in nelec)
    pool_mb = (
        512 * len(samples)
        + 128 * math.prod(full_dims)
        + 64 * sum(full_dims)
        + 4 * sqd.amplitudes.nbytes
    ) / 1e6
    guard_allocation(
        norb,
        nelec,
        stage="comparison",
        model=model,
        subspace_dims=full_dims,
        retained_mb=retained + pool_mb,
        solver_method="selected_ci_pyscf",
        host_available_mb=cap,
        rss_budget_mb=cap,
    )
    if monotonic() >= deadline:
        raise ReferenceTimeoutError("Comparison shared wall deadline exhausted before selection")
    raw, _, peak = _supervise(
        _ComparisonWorker(),
        (h1, h2, e_core),
        dict(
            norb=norb,
            nelec=nelec,
            context=context,
            seed_data=seed_data,
            sqd=sqd,
            samples=samples,
            seed=seed,
            cap=cap,
            retained_mb=retained,
            deadline=deadline,
        ),
        deadline,
        cap,
    )
    scalar = summarize_observables(
        sqd_energy=sqd.energy,
        sci_energy=raw["sci_energy"],
        random_energy=raw["random_energy"],
        hf_energy=seed_data.hf_energy,
        baseline_energy=reference.energy,
        baseline_tier=reference.tier,
        random_reason=raw["random_reason"],
    )
    scalar["diagnostics"].update(
        sci_selection=raw["trace"],
        random_seed=seed,
        sqd_projected_residual_ha=raw["sqd_residual_ha"],
    )
    return ComparisonResult(
        sqd_energy=sqd.energy,
        iso_active_space_ccsd_energy=seed_data.ccsd_energy,
        iso_ndet_sci_energy=raw["sci_energy"],
        iso_ndet_random_energy=raw["random_energy"],
        subspace_dims=dims,
        subspace_dim=math.prod(dims),
        unique_dets_vs_shots=raw["curve"],
        hamiltonian_id=context.hamiltonian_id,
        frame_id=context.frame_id,
        seed=seed,
        ci_strings=raw["ci_strings"],
        allocation_audit=raw["audit"],
        comparison_wall_s=monotonic() - started,
        peak_rss_mb=peak,
        **scalar,
    )
