"""Bounded public molecular and supplied-integral SQD workflows."""

from __future__ import annotations

import math
import multiprocessing
import os
import platform
import signal
import threading
import warnings
from dataclasses import fields, replace
from importlib.metadata import version
from pathlib import Path
from time import monotonic, sleep
from typing import Literal

import numpy as np

from q2m3.molecule import MoleculeConfig
from q2m3.sqd import extend
from q2m3.sqd.ansatz import build_ccsd_seed, build_lucj_from_integrals
from q2m3.sqd.config import (
    CCSDSeed,
    ExtSQDSettings,
    IntegralContext,
    LUCJConfig,
    ReferenceConfig,
    RunMode,
    SQDConfig,
    validate_integral_inputs,
    validate_molecular_inputs,
)
from q2m3.sqd.diagonalize import diagonalize_samples
from q2m3.sqd.exceptions import (
    AllocationGuardError,
    ReferenceNumericalError,
    ReferenceTimeoutError,
    ResourceLimitError,
)
from q2m3.sqd.integrals import build_integrals
from q2m3.sqd.observables import run_comparisons
from q2m3.sqd.reference import _rss_mb, _tree_pids, run_reference
from q2m3.sqd.resources import check_orbital_limit, guard_allocation, resource_model_for
from q2m3.sqd.result import SQDResult
from q2m3.sqd.sampling import FfsimSampler
from q2m3.sqd.transport import decode_result, encode_result

_transport_state = threading.local()
_EXT_CODE_PATH_ID = "q2m3.sqd.extend.run_ext_sqd"


_wire = encode_result


def _unwire(value):
    # receive() looks this name up at call time; the cancellation tests replace it.
    return decode_result(value, cancelled=getattr(_transport_state, "cancelled", None))


def _worker(sender, function, args, kwargs):
    os.setsid()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            result = function(*args, **kwargs)
        result = replace(
            result,
            warnings=tuple(dict.fromkeys((*result.warnings, *(str(w.message) for w in caught)))),
        )
        sender.send(("result", _wire(result)))
    except Exception as exc:
        sender.send(("error", exc))
    finally:
        sender.close()


def _supervise(function, args, kwargs, *, deadline, cap, finalize=None):
    """Enforce a shared deadline and simultaneous parent/descendant RSS cap.

    Reader cancellation is cooperative and waits at most 50 ms. An in-flight
    native unpickle or constructor cannot be preempted and may briefly remain
    in the daemon reader after cancellation.
    """
    ctx = multiprocessing.get_context("fork")
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_worker, args=(sender, function, args, kwargs))
    peak = _rss_mb(os.getpid())
    if peak >= cap:
        receiver.close()
        sender.close()
        raise AllocationGuardError("SQD parent already reaches RSS cap")
    process.start()
    sender.close()
    descendants = set()
    cancelled = threading.Event()
    received = threading.Event()
    delivery = []

    def receive():
        _transport_state.cancelled = cancelled
        try:
            kind, value = receiver.recv()
            if cancelled.is_set():
                return
            if kind == "error":
                raise value
            result = _unwire(value)
            if cancelled.is_set():
                return
            if finalize is not None:
                result = finalize(result, peak)
            if not cancelled.is_set():
                delivery.append(("result", result))
        except (EOFError, OSError) as exc:
            if not cancelled.is_set():
                delivery.append(
                    (
                        "error",
                        ReferenceNumericalError(
                            f"SQD worker closed without a complete result: {exc}"
                        ),
                    )
                )
        except Exception as exc:
            if not cancelled.is_set():
                delivery.append(("error", exc))
        finally:
            received.set()
            del _transport_state.cancelled

    reader = threading.Thread(target=receive, name="sqd-result-receiver", daemon=True)
    reader.start()
    try:
        while True:
            descendants.update(_tree_pids(process.pid))
            peak = max(peak, sum(_rss_mb(pid) for pid in _tree_pids(os.getpid())))
            if peak >= cap:
                raise ResourceLimitError(f"SQD process tree RSS={peak:.6f} MB >= {cap}")
            if monotonic() >= deadline:
                raise ReferenceTimeoutError("SQD shared wall deadline exhausted")
            if received.is_set():
                kind, value = delivery.pop()
                if kind == "error":
                    raise value
                return value, peak
            sleep(0.01)
    finally:
        cancelled.set()
        # Nested reference workers create their own sessions, so process-group
        # termination alone would leave those solvers alive after interruption.
        descendants.update(_tree_pids(process.pid))
        for pid in sorted(descendants, reverse=True):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.is_alive():
            process.kill()
        process.join(timeout=1)
        reader.join(timeout=0.05)
        receiver.close()
        process.close()


def _run(function, args, kwargs, reference, initial_elapsed=0.0):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("SQD supervisor requires Linux x86_64 /proc and fork")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"SQD requires {name}=1 before interpreter start")
    available = next(
        int(line.split()[1]) * 1024 / 1e6
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    cap = min(reference.rss_budget_mb, available, 8192.0)
    start = monotonic()
    worker_cap = cap - _rss_mb(os.getpid())
    if worker_cap <= 0:
        raise AllocationGuardError("SQD parent already reaches RSS cap")
    kwargs = kwargs | {
        "cap": worker_cap,
        "deadline": start + reference.wall_budget_s - initial_elapsed,
    }

    def finalize(result, peak):
        diagnostics = dict(result.diagnostics)
        diagnostics["resources"] = {
            "peak_rss_mb": peak,
            "rss_budget_mb": cap,
            "wall_budget_s": reference.wall_budget_s,
            "poll_interval_s": 0.01,
            "scope": "parent_and_simultaneous_descendants",
            "bound_kind": "polled_rss_not_arbitrary_transient_peak",
            "measurement_end": "start_of_final_result_construction",
            "timing_scope": "total ends at the same boundary; final construction remains "
            "under deadline and RSS supervision",
            "policy": "existing_component_caps_remain_at_most_8192_MB",
        }
        return replace(
            result,
            diagnostics=diagnostics,
            timings_s=dict(result.timings_s) | {"total": monotonic() - start + initial_elapsed},
        ).validate()

    result, _ = _supervise(
        function, args, kwargs, deadline=kwargs["deadline"], cap=cap, finalize=finalize
    )
    for message in result.warnings:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    return result


def run_sqd(
    symbols: list[str],
    coordinates: np.ndarray,
    *,
    active_electrons: int,
    active_orbitals: int,
    charge: int = 0,
    basis: str = "sto-3g",
    lucj: LUCJConfig | None = None,
    reference: ReferenceConfig | None = None,
    mm_charges: np.ndarray | None = None,
    mm_coords: np.ndarray | None = None,
    embedding_mode: Literal["diagonal", "full_oneelectron"] = "diagonal",
    seed: int = 0,
    allow_large: bool = False,
    verbose: bool = True,
    mode: RunMode = "full",
) -> SQDResult:
    """Run SQD from geometry in Angstrom with an explicit closed-shell active space.

    Args:
        symbols: Atomic element symbols.
        coordinates: Atomic coordinates in Angstrom.
        active_electrons: Even active electron count.
        active_orbitals: Active spatial orbital count.
        charge: Molecular charge.
        basis: PySCF basis name.
        lucj: Ansatz and sampling configuration.
        reference: Baseline policy and shared run resource budgets.
        mm_charges: Optional point charges in atomic charge units.
        mm_coords: Point-charge positions in Angstrom.
        embedding_mode: Fixed-frame one-electron embedding treatment.
        seed: Nonnegative seed forwarded to all stochastic operations.
        allow_large: Forward soft-policy override where supported. Existing
            component/model caps remain in force, including 8192 MB.
        verbose: Print a compact energy/status summary.
        mode: Full SQD or explicitly requested reference-only calculation.

    Returns:
        Validated immutable energies, metadata and resource/timing audit.

    Raises:
        ValueError: Inputs or execution environment are unsupported.
        ResourceModelDomainError: The active space has more than 14 orbitals.
        ResourceLimitError: Allocation or observed process RSS reaches a cap.
        ReferenceTimeoutError: The total shared deadline expires.
    """
    check_orbital_limit(active_orbitals)
    started = monotonic()
    molecule = MoleculeConfig(
        "sqd",
        list(symbols),
        np.array(coordinates, copy=True),
        charge,
        active_electrons,
        active_orbitals,
        basis,
    )
    config = SQDConfig(
        molecule,
        lucj or LUCJConfig(),
        reference or ReferenceConfig(),
        embedding_mode,
        seed,
        allow_large,
        mode,
    ).validate()
    validate_molecular_inputs(
        config.molecule, mm_charges=mm_charges, mm_coords=mm_coords, embedding_mode=embedding_mode
    )
    if not isinstance(verbose, bool):
        raise ValueError("verbose must be bool")
    geometry_elapsed = monotonic() - started
    result = _run(
        _molecular,
        (config,),
        {
            "mm_charges": None if mm_charges is None else np.array(mm_charges, copy=True),
            "mm_coords": None if mm_coords is None else np.array(mm_coords, copy=True),
            "geometry_elapsed": geometry_elapsed,
        },
        config.reference,
        initial_elapsed=geometry_elapsed,
    )
    if verbose:
        print(
            f"SQD {result.status}: baseline {result.baseline_method} "
            f"{result.baseline_energy:.12f} Ha; SQD {result.sqd_energy}"
        )
    return result


def _molecular(config, *, mm_charges, mm_coords, cap, deadline, geometry_elapsed):
    if config.mode == "full":
        _full_budget(
            config.molecule.active_orbitals,
            (config.molecule.active_electrons // 2,) * 2,
            config.lucj,
            cap,
            config.allow_large,
        )
    start = monotonic()
    data = build_integrals(
        config.molecule,
        mm_charges=mm_charges,
        mm_coords=mm_coords,
        embedding_mode=config.embedding_mode,
        host_available_mb=cap,
        rss_budget_mb=cap,
        allow_large=config.allow_large,
    )
    timings = {"geometry": geometry_elapsed, "integrals": monotonic() - start}
    seed_data = None
    if config.mode == "full":
        start = monotonic()
        seed_data = build_ccsd_seed(
            data, host_available_mb=cap, rss_budget_mb=cap, allow_large=config.allow_large
        )
        timings["ccsd"] = monotonic() - start
    h1, h2, e_core = data.h1, data.h2, data.e_core
    norb, nelec, context = data.norb, data.nelec, data.context
    del data  # Complete molecular orbitals are not retained during the active-space run.
    return _execute(
        h1,
        h2,
        e_core,
        norb=norb,
        nelec=nelec,
        context=context,
        seed_data=seed_data,
        lucj=config.lucj,
        reference=config.reference,
        seed=config.seed,
        allow_large=config.allow_large,
        mode=config.mode,
        cap=cap,
        deadline=deadline,
        timings=timings,
        provenance={"entry": "geometry", "configuration": config.snapshot()},
    )


def run_sqd_from_integrals(
    h1: np.ndarray,
    h2: np.ndarray,
    e_core: float,
    *,
    norb: int,
    nelec: tuple[int, int],
    context: IntegralContext,
    seed_data: CCSDSeed | None,
    lucj: LUCJConfig | None = None,
    reference: ReferenceConfig | None = None,
    seed: int = 0,
    allow_large: bool = False,
    mode: RunMode = "full",
    ext_sqd: ExtSQDSettings | None = None,
) -> SQDResult:
    """Run the same workflow from real chemist integrals and authenticated provenance.

    Args:
        h1: Real one-electron integrals in Hartree.
        h2: Real chemist two-electron tensor, without any additional transposition.
        e_core: Core constant in Hartree, already including embedding corrections.
        norb: Active spatial orbital count.
        nelec: Balanced alpha/beta electron counts.
        context: Integral/frame identifiers and constant decomposition.
        seed_data: Same-frame CCSD seed. Use None only in reference-only mode.
        lucj: Ansatz and sampling configuration.
        reference: Baseline policy and total shared resource budgets.
        seed: Nonnegative run seed.
        allow_large: Forward supported overrides within existing component caps.
        mode: Full SQD or explicit reference-only calculation.
        ext_sqd: Configuration-space extension settings. When set in full mode, the
            extension stage runs on the SQD seed space after the comparisons.

    Returns:
        Validated immutable result, using exactly one copy of e_core in energies.

    Raises:
        ValueError: The declared integral contract or ``ext_sqd`` is invalid.
        ResourceModelDomainError: The active space has more than 14 orbitals.
        ResourceLimitError: Predicted or actual resources exceed the budget.
        ReferenceTimeoutError: The total shared deadline expires.
    """
    check_orbital_limit(norb)
    if ext_sqd is not None:
        if not isinstance(ext_sqd, ExtSQDSettings):
            raise ValueError(f"ext_sqd must be ExtSQDSettings or None, got {type(ext_sqd)}")
        ext_sqd.validate()
    reference = (reference or ReferenceConfig()).validate()
    return _run(
        _execute,
        (h1, h2, e_core),
        dict(
            norb=norb,
            nelec=nelec,
            context=context,
            seed_data=seed_data,
            lucj=lucj or LUCJConfig(),
            reference=reference,
            seed=seed,
            allow_large=allow_large,
            mode=mode,
            timings={},
            provenance={"entry": "integrals"},
            ext_sqd=ext_sqd,
        ),
        reference,
    )


def _full_budget(norb, nelec, lucj, cap, allow_large):
    model = resource_model_for(norb, profile={"n_reps": lucj.n_reps, "shots": lucj.shots})
    for stage in ("prepare", "sample"):
        guard_allocation(
            norb,
            nelec,
            stage=stage,
            model=model,
            host_available_mb=cap,
            rss_budget_mb=cap,
            allow_large=allow_large,
        )


def _remaining(deadline):
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise ReferenceTimeoutError("SQD shared wall deadline exhausted")
    return remaining


def _execute(
    h1,
    h2,
    e_core,
    *,
    norb,
    nelec,
    context,
    seed_data,
    lucj,
    reference,
    seed,
    allow_large,
    mode,
    cap,
    deadline,
    timings,
    provenance,
    ext_sqd=None,
):
    validate_integral_inputs(
        h1,
        h2,
        e_core,
        norb=norb,
        nelec=nelec,
        context=context,
        seed_data=seed_data,
        lucj=lucj,
        reference=reference,
        seed=seed,
        allow_large=allow_large,
        mode=mode,
    )
    o = nelec[0]
    hf = float(
        e_core
        + 2 * np.trace(h1[:o, :o], dtype=np.float64)
        + 2 * np.einsum("iijj", h2[:o, :o, :o, :o], dtype=np.float64)
        - np.einsum("ijji", h2[:o, :o, :o, :o], dtype=np.float64)
    )
    comparison = None
    diagonalized = None
    ext_record = None
    audit = []
    if mode == "full":
        _full_budget(norb, nelec, lucj, cap, allow_large)
        start = monotonic()
        operator = build_lucj_from_integrals(
            h1,
            h2,
            e_core,
            norb=norb,
            nelec=nelec,
            context=context,
            seed_data=seed_data,
            lucj=lucj,
            host_available_mb=cap,
            rss_budget_mb=cap,
            allow_large=allow_large,
        )
        sampler = FfsimSampler(host_available_mb=cap, rss_budget_mb=cap)
        state = sampler.prepare(operator, norb, nelec)
        timings["prepare"] = monotonic() - start
        start = monotonic()
        samples = sampler.sample(state, norb, nelec, shots=lucj.shots, seed=seed)
        timings["sample"] = monotonic() - start
        del state, operator
        start = monotonic()
        diagonalized = diagonalize_samples(
            h1,
            h2,
            e_core,
            samples,
            norb=norb,
            nelec=nelec,
            host_available_mb=cap,
            rss_budget_mb=cap,
            allow_large=allow_large,
            samples_per_batch=lucj.samples_per_batch,
            seed=seed,
        )
        timings["diagonalize"] = monotonic() - start
    start = monotonic()
    retained_reference_mb = 0.0
    if diagonalized is not None:
        retained_reference_mb = _retained_mb(samples, diagonalized)
    ref = run_reference(
        h1,
        h2,
        e_core,
        norb=norb,
        nelec=nelec,
        context=context,
        config=reference,
        seed_data=seed_data,
        seed=seed,
        host_available_mb=cap,
        rss_budget_mb=cap,
        retained_mb=retained_reference_mb,
        remaining_wall_s=_remaining(deadline),
    )
    timings["reference"] = monotonic() - start
    if mode == "full":
        start = monotonic()
        comparison = run_comparisons(
            h1,
            h2,
            e_core,
            norb=norb,
            nelec=nelec,
            context=context,
            seed_data=seed_data,
            reference=ref,
            sqd=diagonalized,
            samples=samples,
            seed=seed,
            host_available_mb=cap,
            rss_budget_mb=cap,
            remaining_wall_s=_remaining(deadline),
        )
        timings["comparison"] = monotonic() - start
        ext_record = _ext_stage(
            h1,
            h2,
            e_core,
            ext_sqd,
            samples=samples,
            diagonalized=diagonalized,
            norb=norb,
            nelec=nelec,
            cap=cap,
            allow_large=allow_large,
            audit=audit,
            deadline=deadline,
        )
    return _assemble(
        norb,
        nelec,
        context,
        seed_data,
        lucj,
        reference,
        seed,
        mode,
        hf,
        ref,
        timings,
        provenance,
        comparison,
        diagonalized,
        ext_record,
    )


def _retained_mb(samples, diagonalized):
    """Return the MB of SQD samples, amplitudes and CI strings that stay live."""
    return (
        samples.nbytes
        + diagonalized.amplitudes.nbytes
        + sum(strings.nbytes for strings in diagonalized.ci_strings)
    ) / 1e6


def _ext_stage(
    h1,
    h2,
    e_core,
    settings,
    *,
    samples,
    diagonalized,
    norb,
    nelec,
    cap,
    allow_large,
    audit,
    deadline,
):
    """Run the quantum extended-SQD arm on the SQD seed space.

    A guard rejection or a numerical failure of the arm gives a null arm, and the run
    continues. Other exceptions propagate.

    Note:
        Residual risk. The arm runs in the worker, which ``_supervise`` monitors against
        the deadline and the RSS of the process tree. If the real time is more than 2
        times the estimate, or the worker RSS exceeds the cap, the whole run fails and
        the SQD and T0 results obtained so far are lost. ``t_arm`` comes from only 1
        N2 point with 14 orbitals and excludes the time of CISD and SCI selection.

    Returns:
        The extension record, or None when ``settings`` is None.
    """
    if settings is None:
        return None
    start = monotonic()
    try:
        arm = extend.run_ext_sqd(
            h1,
            h2,
            e_core,
            (diagonalized.ci_strings, diagonalized.amplitudes),
            settings,
            seed_source="quantum",
            norb=norb,
            nelec=nelec,
            host_available_mb=cap,
            rss_budget_mb=cap,
            retained_mb=_retained_mb(samples, diagonalized),
            allow_large=allow_large,
            audit=audit,
            deadline=deadline,
        )
    except AllocationGuardError as exc:
        arm = _null_ext_arm(diagonalized.ci_strings, f"ext_rss_budget: {exc}", start)
    except ReferenceNumericalError as exc:
        arm = _null_ext_arm(diagonalized.ci_strings, f"ext_numerical: {exc}", start)
    return extend.ExtSQDRecord(settings, _EXT_CODE_PATH_ID, (arm,))


def _null_ext_arm(ci_strings, reason, start):
    """Return the null quantum arm for an ext solve that raised.

    The exception carries the dimensions and the predicted RSS only as message text,
    which is kept in ``reason``. The other fields are placeholders, not measurements:
    ``truncated_counts``, ``extended_counts`` and ``subspace_dims`` are (0, 0),
    ``coverage`` is 0.0, ``only_hf`` is False and ``predicted_rss_mb`` is None, so
    that strict JSON (``allow_nan=False``) can write the arm.
    """
    return extend.ExtSQDArm(
        seed_source="quantum",
        energy=None,
        seed_counts=(len(ci_strings[0]), len(ci_strings[1])),
        truncated_counts=(0, 0),
        extended_counts=(0, 0),
        subspace_dims=(0, 0),
        coverage=0.0,
        only_hf=False,
        seed_space_energy=None,
        zero_weight_strings=None,
        sci_baseline_energy=None,
        sci_baseline_reason="sci_baseline_not_implemented",
        wall_s=monotonic() - start,
        predicted_rss_mb=None,
        process_peak_rss_mb=extend._peak_rss_mb(),
        null_reason=reason,
    )


def _assemble(
    norb,
    nelec,
    context,
    seed_data,
    lucj,
    reference,
    seed,
    mode,
    hf,
    ref,
    timings,
    provenance,
    comparison=None,
    diagonalized=None,
    ext_sqd=None,
):
    values = {f.name: None for f in fields(SQDResult)}
    values["ext_sqd"] = ext_sqd
    values.update(
        schema_version="sqd.result.v1",
        status="reference_only",
        hf_energy=hf,
        hf_reference_kind=context.hf_reference_kind,
        baseline_energy=ref.energy,
        baseline_tier=ref.tier,
        baseline_method=ref.method,
        baseline_uncertainty_mHa=ref.uncertainty_mHa,
        baseline_uncertainty_kind=ref.uncertainty_kind,
        baseline_downgrade_reason=ref.downgrade_reason,
        baseline_untrustworthy=ref.untrustworthy,
        baseline_t1_residual=ref.t1_residual,
        t1_diagnostic=None if ref.t2 is None else ref.t2.t1_diagnostic,
        t2_diagnostics=ref.t2,
        reference_attempts=ref.attempts,
        active_space=(sum(nelec), norb),
        seed=seed,
        full_ci_dim=math.prod(math.comb(norb, n) for n in nelec),
        embedding_mode=context.embedding_mode,
        fixed_mo=context.fixed_mo,
        two_electron_tensor_fixed=context.two_electron_tensor_fixed,
        versions={
            name: version(name) for name in ("numpy", "scipy", "pyscf", "ffsim", "qiskit-addon-sqd")
        },
        warnings=ref.warnings,
        diagnostics={},
        provenance=provenance
        | {
            "context": context,
            "seed": seed,
            "requested_lucj": lucj,
            "reference_config": reference,
            "ccsd_seed": seed_data,
        },
    )
    if seed_data is not None:
        values["iso_active_space_ccsd_energy"] = seed_data.ccsd_energy
    elif ref.t2 is not None:
        values["iso_active_space_ccsd_energy"] = hf + ref.t2.ccsd_correlation_energy_ha
    if comparison is not None:
        for name in (
            "sqd_energy",
            "iso_active_space_ccsd_energy",
            "iso_ndet_sci_energy",
            "iso_ndet_random_energy",
            "delta_mHa",
            "delta_vs_sci_mHa",
            "ratio_sqd_over_sci",
            "subspace_dims",
            "subspace_dim",
            "unique_dets_vs_shots",
        ):
            values[name] = getattr(comparison, name)
        values.update(
            status="completed",
            n_reps=lucj.n_reps,
            shots=lucj.shots,
            samples_per_batch=diagonalized.samples_per_batch,
            backend="ffsim",
            diagnostics=dict(comparison.diagnostics)
            | {
                "diagonalization_allocations": diagonalized.allocation_audit,
                "comparison_allocations": comparison.allocation_audit,
                "comparison_ci_strings": comparison.ci_strings,
            },
        )
    stage_names = (
        "geometry",
        "integrals",
        "ccsd",
        "prepare",
        "sample",
        "diagonalize",
        "reference",
        "comparison",
        "total",
    )
    times = {name: timings.get(name) for name in stage_names}
    times["total"] = sum(t for t in times.values() if t is not None)
    reasons = {name: "not_applicable" for name, value in values.items() if value is None}
    for name in (
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
        "samples_per_batch",
        "ext_sqd",
    ):
        if mode == "reference_only":
            reasons[name] = "reference_only"
    if mode == "full" and ext_sqd is None:
        reasons["ext_sqd"] = "ext_sqd_disabled"
    if comparison is not None:
        reasons.update(comparison.null_reasons)
    if ref.uncertainty_reason:
        reasons["baseline_uncertainty_mHa"] = ref.uncertainty_reason
    reasons.update({f"timings_s.{name}": "not_executed" for name, t in times.items() if t is None})
    values.update(timings_s=times, null_reasons=reasons)
    return SQDResult(**values).validate()
