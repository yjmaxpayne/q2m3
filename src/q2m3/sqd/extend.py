"""String-level seed truncation and excitation for extended SQD.

The truncation and excitation functions act on per-spin CI string sets. They run
no diagonalization and no resource guard. ``run_ext_sqd`` chains them, guards the
fixed-space solve and runs it. The ``ExtSQDArm`` and ``ExtSQDRecord`` records hold
the per-seed-source results.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import numpy as np

from q2m3.sqd.config import ExtSQDSettings
from q2m3.sqd.diagonalize import kernel_fixed_space
from q2m3.sqd.exceptions import AllocationGuardError
from q2m3.sqd.observables import _sector
from q2m3.sqd.resources import (
    HARD_RSS_MB,
    REJECT_RSS_MB,
    estimate_rss_mb,
    guard_allocation,
    resource_model_for,
)

_ORDERS = (1, 2, 3)
_CHUNK_BYTES = 8_000_000
# Wall-time scale of the ext precheck: 306 s for a 11,778,624-determinant space (ADD:184).
_WALL_REFERENCE_S = 306.0
_WALL_REFERENCE_DIM = 11_778_624
# An arm's null_reason starts with one of these (03e:12).
_NULL_REASON_PREFIXES = ("ext_wall_budget", "ext_rss_budget:", "ext_numerical:")


@dataclass(frozen=True)
class Truncation:
    """Seed strings kept by amplitude truncation.

    Attributes:
        ci_strings: Ascending unique int64 alpha and beta strings. Each contains
            the HF string.
        only_hf: True when truncation dropped every non-HF string on both spins.
    """

    ci_strings: tuple[np.ndarray, np.ndarray]
    only_hf: bool


def _require_pairing(value: object, reason: object, value_name: str, reason_name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is None exactly when ``reason`` is set."""
    if (value is None) != (reason is not None):
        raise ValueError(
            f"{value_name} must be None exactly when {reason_name} is set, "
            f"got {value_name}={value!r}, {reason_name}={reason!r}"
        )


@dataclass(frozen=True)
class ExtSQDArm:
    """Result of one extended-SQD arm, that is one seed source.

    The arm keeps scalars, counts and one optional string array. It keeps no
    amplitude arrays.

    A null arm (``null_reason`` is not None) has no solve result. Check
    ``null_reason`` before you use its counts, ``coverage`` or ``predicted_rss_mb``.
    There are two null-arm forms:

    - ``ext_wall_budget``: the time precheck stopped the arm before the solve. The
      counts, ``subspace_dims`` and ``coverage`` are the real extended values, and
      ``predicted_rss_mb`` is the finite estimate. The solve did not run.
    - ``ext_rss_budget:`` or ``ext_numerical:``: the orchestrator mapped an exception
      from the ext solve. ``truncated_counts``, ``extended_counts`` and
      ``subspace_dims`` are (0, 0), ``coverage`` is 0.0, ``only_hf`` is False and
      ``predicted_rss_mb`` is None. The dimensions are only in the reason text.

    Attributes:
        seed_source: Seed source of the arm, for example ``quantum`` or ``cisd``.
        energy: Total energy in Hartree, or None when the arm has no result.
        seed_counts: Alpha and beta seed string counts before truncation.
        truncated_counts: Alpha and beta string counts after truncation.
        extended_counts: Alpha and beta string counts after excitation.
        subspace_dims: Alpha and beta dimensions of the fixed-space solve.
        coverage: Extended strings per spin divided by C(norb, n_alpha).
        only_hf: True when truncation kept only the HF string on both spins.
        seed_space_energy: Energy in the seed space, or None for the quantum arm.
        zero_weight_strings: Int64 strings with zero seed weight. Only the cisd
            arm sets this value.
        sci_baseline_energy: SCI baseline energy, or None with a reason.
        sci_baseline_reason: Reason for a missing SCI baseline, or None.
        wall_s: Wall time of the arm in seconds.
        predicted_rss_mb: Predicted RSS of the ext solve in decimal MB, or None for
            an arm mapped from an exception.
        process_peak_rss_mb: Peak RSS of the process so far (``VmHWM``) at the end of
            the arm, in decimal MB. It is not the peak of this arm only.
        null_reason: Reason for a missing energy, or None.
    """

    seed_source: str
    energy: float | None
    seed_counts: tuple[int, int]
    truncated_counts: tuple[int, int]
    extended_counts: tuple[int, int]
    subspace_dims: tuple[int, int]
    coverage: float
    only_hf: bool
    seed_space_energy: float | None
    zero_weight_strings: np.ndarray | None
    sci_baseline_energy: float | None
    sci_baseline_reason: str | None
    wall_s: float
    predicted_rss_mb: float | None
    process_peak_rss_mb: float
    null_reason: str | None

    def validate(self) -> ExtSQDArm:
        """Check that each nullable value pairs with its reason in both directions.

        Returns:
            This arm.

        Raises:
            ValueError: If ``energy`` and ``null_reason``, or ``sci_baseline_energy``
                and ``sci_baseline_reason``, are both set or both None, or if
                ``null_reason`` does not start with a known ext prefix.
        """
        _require_pairing(self.energy, self.null_reason, "energy", "null_reason")
        if self.null_reason is not None and not self.null_reason.startswith(_NULL_REASON_PREFIXES):
            raise ValueError(
                f"null_reason must start with one of {_NULL_REASON_PREFIXES}, "
                f"got {self.null_reason!r}"
            )
        _require_pairing(
            self.sci_baseline_energy,
            self.sci_baseline_reason,
            "sci_baseline_energy",
            "sci_baseline_reason",
        )
        return self


@dataclass(frozen=True)
class ExtSQDRecord:
    """Extended-SQD settings, code path and one arm per seed source.

    Attributes:
        settings: Settings of the extended-SQD run.
        code_path_id: Identifier of the code path that produced the arms.
        arms: One arm per entry of ``settings.seed_sources``, in the same order.
    """

    settings: ExtSQDSettings
    code_path_id: str
    arms: tuple[ExtSQDArm, ...]

    def validate(self) -> ExtSQDRecord:
        """Check the arm order against the settings, then check each arm.

        Returns:
            This record.

        Raises:
            ValueError: If the ``seed_source`` sequence of the arms is not equal to
                ``settings.seed_sources``, or if an arm is not valid.
        """
        sources = tuple(arm.seed_source for arm in self.arms)
        expected = tuple(self.settings.seed_sources)
        if sources != expected:
            raise ValueError(
                f"arm seed_source sequence {sources} must equal settings.seed_sources {expected}"
            )
        for arm in self.arms:
            arm.validate()
        return self


def _hf(electrons: int) -> int:
    return (1 << electrons) - 1


def _string_array(strings, electrons: int) -> np.ndarray:
    strings = np.asarray(strings)
    if strings.ndim != 1 or strings.dtype.kind not in "iu" or not strings.size:
        raise ValueError("CI strings must be a nonempty one-dimensional integer array")
    strings = strings.astype(np.int64)
    if np.any(strings < 0):
        raise ValueError("CI strings violate the orbital range")
    if np.any(np.bitwise_count(strings) != electrons):
        raise ValueError("CI strings violate the particle count")
    return strings


def _seed_array(seeds, norb: int, nelec_spin: int) -> np.ndarray:
    seeds = _string_array(seeds, nelec_spin)
    if np.any(seeds >= 1 << norb):
        raise ValueError("CI strings violate the orbital range")
    return seeds


def excite_strings(seeds, *, norb: int, nelec_spin: int, order: int) -> np.ndarray:
    """Return every legal string within ``order`` excitations of some seed.

    The excitation distance between two strings is popcount(a XOR b) / 2. Seeds
    are processed in chunks. The XOR array of a chunk holds at most 8 MB while one
    seed column fits in 8 MB, which holds for every sector up to 14 orbitals.

    Args:
        seeds: Integer seed strings of one spin. Order and duplicates do not matter.
        norb: Spatial orbitals.
        nelec_spin: Electrons of this spin.
        order: Maximum excitation order: 1, 2 or 3.

    Returns:
        Ascending unique int64 strings.

    Raises:
        ValueError: Empty or illegal seeds, or ``order`` not in {1, 2, 3}.
    """
    if order not in _ORDERS:
        raise ValueError(f"Excitation order must be 1, 2 or 3, got {order!r}")
    seeds = _seed_array(seeds, norb, nelec_spin)
    pool = _sector(norb, nelec_spin).astype(np.int64)
    rows = max(1, _CHUNK_BYTES // (pool.size * pool.itemsize))
    keep = np.zeros(pool.size, dtype=bool)
    for start in range(0, seeds.size, rows):
        block = seeds[start : start + rows]
        # One expression, so each XOR block is freed before the next one exists.
        keep |= (np.bitwise_count(pool[:, None] ^ block[None, :]) <= 2 * order).any(axis=1)
    return pool[keep]


def truncate_strings(
    ci_strings: tuple[np.ndarray, np.ndarray],
    amplitudes: np.ndarray,
    threshold: float,
    *,
    nelec: tuple[int, int],
) -> Truncation:
    """Keep the strings whose amplitude reaches ``threshold``.

    The amplitude of an alpha string is the largest |c| in its row. The amplitude
    of a beta string is the largest |c| in its column. The HF string is always
    in the output, also when ``ci_strings`` does not contain it.

    Args:
        ci_strings: Alpha and beta integer strings. Each string has ``nelec`` set bits
            for its spin.
        amplitudes: Finite alpha-major CI coefficients of shape (len(alpha), len(beta)).
        threshold: Inclusive amplitude threshold, finite and nonnegative.
        nelec: Alpha and beta electron counts.

    Returns:
        Kept strings per spin and the ``only_hf`` flag.

    Raises:
        ValueError: Threshold negative or not finite; empty, non-integer, negative or
            wrong-count strings; amplitude shape mismatch or non-finite amplitudes.

    Warns:
        RuntimeWarning: Every non-HF string was dropped on both spins.
    """
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be finite and nonnegative, got {threshold!r}")
    alpha, beta = (_string_array(s, n) for s, n in zip(ci_strings, nelec, strict=True))
    weights = np.abs(np.asarray(amplitudes))
    if weights.shape != (alpha.size, beta.size):
        raise ValueError(f"Amplitude shape {weights.shape} does not match the CI strings")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Amplitudes must be finite")
    kept = tuple(
        np.union1d(sector[maxima >= threshold], _hf(electrons))
        for sector, maxima, electrons in (
            (alpha, weights.max(axis=1), nelec[0]),
            (beta, weights.max(axis=0), nelec[1]),
        )
    )
    only_hf = all(sector.size == 1 for sector in kept)
    if only_hf:
        warnings.warn(
            f"Truncation at threshold {threshold} kept only the HF string",
            RuntimeWarning,
            stacklevel=2,
        )
    return Truncation(kept, only_hf)


def extend_seed_space(
    seeds: tuple[np.ndarray, np.ndarray],
    *,
    norb: int,
    nelec: tuple[int, int],
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Excite the alpha and beta seeds separately and share their union.

    Both returned spins hold the same strings, as for closed shells.

    Args:
        seeds: Truncated alpha and beta seed strings.
        norb: Spatial orbitals.
        nelec: Alpha and beta electron counts. They must be equal.
        order: Maximum excitation order: 1, 2 or 3.

    Returns:
        Equal ascending unique int64 alpha and beta strings.

    Raises:
        ValueError: Open-shell populations, or a seed set that ``excite_strings`` rejects.
    """
    if nelec[0] != nelec[1]:
        raise ValueError(f"Extension supports closed-shell populations only, got {nelec}")
    alpha, beta = (
        excite_strings(s, norb=norb, nelec_spin=n, order=order)
        for s, n in zip(seeds, nelec, strict=True)
    )
    union = np.union1d(alpha, beta)
    return union, union.copy()


def _peak_rss_mb() -> float:
    """Return the process peak RSS so far (``VmHWM``) in decimal MB."""
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024 / 1e6
    raise RuntimeError("VmHWM is missing from /proc/self/status")


def _over_wall_budget(deadline: float | None, seed_source: str, dims: tuple[int, int]) -> bool:
    """Return True when less than twice the predicted arm time remains before ``deadline``.

    ``t_arm = k * max(1 s, 306 s * D / 11,778,624)`` with ``k = 2`` for the quantum arm
    and ``k = 3`` for the other arms. ``D`` is the extended product dimension.
    """
    if deadline is None:
        return False
    k = 2 if seed_source == "quantum" else 3
    t_arm = k * max(1.0, _WALL_REFERENCE_S * dims[0] * dims[1] / _WALL_REFERENCE_DIM)
    return deadline - time.monotonic() < 2 * t_arm


def _budget_mb(host_available_mb: float, rss_budget_mb: float, allow_large: bool) -> float:
    """Return the guard budget, ``min(caps)`` with the caps of ``guard_allocation``.

    This copies the budget formula of ``resources.py`` lines 166-175. Keep the two in
    sync when ``guard_allocation`` changes its caps.
    """
    caps = [HARD_RSS_MB, host_available_mb, rss_budget_mb]
    if not allow_large:
        caps.append(REJECT_RSS_MB)
    return min(caps)


def _guard_ext(
    norb: int,
    nelec: tuple[int, int],
    dims: tuple[int, int],
    predicted: float,
    phase: str,
    audit: list[dict],
    resources: dict,
) -> float:
    """Run the ext guard and append one ``stage="ext"`` audit record on pass or rejection.

    Returns:
        The predicted RSS that the guard returns, in decimal MB.

    Raises:
        AllocationGuardError: The guard rejects; the message starts with ``stage=ext``.
    """
    model = resource_model_for(norb, profile={"max_space": 12})
    record = {
        "stage": "ext",
        "phase": phase,
        "subspace_dims": dims,
        "retained_mb": resources["retained_mb"],
        "predicted_rss_mb": predicted,
        "budget_mb": _budget_mb(
            resources["host_available_mb"], resources["rss_budget_mb"], resources["allow_large"]
        ),
        "model_id": model.model_id,
    }
    try:
        guarded = guard_allocation(
            norb, nelec, stage="diagonalize", model=model, subspace_dims=dims, **resources
        )
    except AllocationGuardError as exc:
        audit.append(record)
        raise AllocationGuardError(f"stage=ext, subspace_dims={dims}: {exc}") from exc
    audit.append(record | {"predicted_rss_mb": guarded})
    return guarded


def run_ext_sqd(
    h1: np.ndarray,
    h2: np.ndarray,
    e_core: float,
    seed_space: tuple[tuple[np.ndarray, np.ndarray], np.ndarray],
    settings: ExtSQDSettings,
    *,
    seed_source: str,
    norb: int,
    nelec: tuple[int, int],
    host_available_mb: float,
    rss_budget_mb: float,
    retained_mb: float,
    allow_large: bool,
    audit: list[dict],
    deadline: float | None,
) -> ExtSQDArm:
    """Truncate the seeds, excite them, guard and solve one fixed-space problem.

    The order is truncation, excitation, wall-time precheck, resource guard,
    ``kernel_fixed_space``. No configuration recovery occurs. The solver repeats the
    guard as a second check with the same resource arguments. The guard appends one
    ``stage="ext"`` record to ``audit`` when it passes and when it rejects.

    Args:
        h1: Active-space one-electron integrals.
        h2: Active-space two-electron integrals.
        e_core: Core energy in Hartree. The arm energy includes it.
        seed_space: Alpha and beta seed strings and their alpha-major amplitudes.
        settings: Validated extended-SQD settings (threshold and order).
        seed_source: Seed source of the arm.
        norb: Active spatial orbitals.
        nelec: Alpha and beta electron counts.
        host_available_mb: Available host memory in decimal MB.
        rss_budget_mb: RSS budget in decimal MB.
        retained_mb: Memory that the caller keeps during the solve, in decimal MB.
        allow_large: Lift the reject cap of the resource guard.
        audit: Allocation audit list of the run.
        deadline: ``time.monotonic()`` deadline of the run, or None.

    Returns:
        The arm with the total energy, string counts and the guard prediction. When
        less than twice the predicted arm time remains before ``deadline``, the arm
        has no energy, ``null_reason="ext_wall_budget"``, and neither the guard nor
        the solver runs.

    Raises:
        ValueError: Seed strings, amplitudes or populations that the string
            functions reject.
        AllocationGuardError: The resource guard rejects the extended space. The
            message starts with ``stage=ext``.
        ResourceLimitError: No resource bound applies to the extended space.
        ReferenceNumericalError: The fixed-space solve fails numerically.
    """
    start = time.monotonic()
    ci_strings, amplitudes = seed_space
    truncation = truncate_strings(ci_strings, amplitudes, settings.threshold, nelec=nelec)
    extended = extend_seed_space(
        truncation.ci_strings, norb=norb, nelec=nelec, order=settings.order
    )
    dims = (int(extended[0].size), int(extended[1].size))
    resources = {
        "host_available_mb": host_available_mb,
        "rss_budget_mb": rss_budget_mb,
        "retained_mb": retained_mb,
        "allow_large": allow_large,
    }
    predicted = estimate_rss_mb(
        norb,
        nelec,
        stage="diagonalize",
        subspace_dims=dims,
        retained_mb=retained_mb,
        model=resource_model_for(norb, profile={"max_space": 12}),
    )
    energy, null_reason = None, "ext_wall_budget"
    if not _over_wall_budget(deadline, seed_source, dims):
        phase = f"ext_{seed_source}"
        predicted = _guard_ext(norb, nelec, dims, predicted, phase, audit, resources)
        solved = kernel_fixed_space(h1, h2, e_core, extended, norb=norb, nelec=nelec, **resources)
        energy, null_reason = float(solved.energy), None
    return ExtSQDArm(
        seed_source=seed_source,
        energy=energy,
        seed_counts=(len(ci_strings[0]), len(ci_strings[1])),
        truncated_counts=tuple(int(s.size) for s in truncation.ci_strings),
        extended_counts=dims,
        subspace_dims=dims,
        coverage=dims[0] / math.comb(norb, nelec[0]),
        only_hf=truncation.only_hf,
        seed_space_energy=None,
        zero_weight_strings=None,
        sci_baseline_energy=None,
        sci_baseline_reason="sci_baseline_not_implemented",
        wall_s=time.monotonic() - start,
        predicted_rss_mb=predicted,
        process_peak_rss_mb=_peak_rss_mb(),
        null_reason=null_reason,
    )
