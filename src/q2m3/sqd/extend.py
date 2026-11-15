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
from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from q2m3.sqd.config import ExtSQDSettings
from q2m3.sqd.diagonalize import kernel_fixed_space
from q2m3.sqd.exceptions import AllocationGuardError, ReferenceNumericalError
from q2m3.sqd.observables import (
    _random_strings,
    _run_selected_ci,
    _sector,
    _take_spin_strings,
)
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
# Single-link grouping: two neighbouring CISD weights share one group when their gap
# is at most this fraction of max(w).
_CISD_GROUP_RTOL = 1e-6


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


def assert_same_ext_config(records: Iterable[ExtSQDRecord]) -> None:
    """Check that the records share each settings field and the code path.

    All arms of one point must use the same settings and the same code path. When they
    differ, the check fails and names the differing fields.
    No production caller uses it yet. The benchmark records will use it later.

    Args:
        records: Extended-SQD records to compare.

    Raises:
        ValueError: If an element is not an ``ExtSQDRecord``, or if a field of
            ``settings`` or ``code_path_id`` differs between records. The message
            names only the differing fields.
    """
    items = list(records)
    for record in items:
        if not isinstance(record, ExtSQDRecord):
            raise ValueError(f"expected ExtSQDRecord elements, got {type(record).__name__}")
    if not items:
        return
    first, rest = items[0], items[1:]
    differing = [
        f.name
        for f in fields(first.settings)
        if any(getattr(r.settings, f.name) != getattr(first.settings, f.name) for r in rest)
    ]
    if any(r.code_path_id != first.code_path_id for r in rest):
        differing.append("code_path_id")
    if differing:
        raise ValueError(f"records differ in: {', '.join(differing)}")


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


def hf_seed_strings(norb: int, nelec: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return the HF seed: one HF string per spin.

    Args:
        norb: Number of active orbitals.
        nelec: Alpha and beta electron counts.

    Returns:
        Alpha and beta int64 arrays, each with the HF string of that spin only.

    Raises:
        ValueError: If an electron count is larger than ``norb``.
    """
    alpha, beta = (_seed_array([_hf(n)], norb, n) for n in nelec)
    return alpha, beta


def random_seed_strings(
    norb: int, nelec: tuple[int, int], dims: tuple[int, int], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the random seed: the SQD random baseline draw for the same seed.

    Args:
        norb: Number of active orbitals.
        nelec: Alpha and beta electron counts.
        dims: Number of strings per spin, HF included.
        seed: Seed of the random generator.

    Returns:
        Alpha and beta ascending int64 arrays that contain the HF string.

    Raises:
        ComparisonUnavailableError: If an entry of ``dims`` is less than 1 or larger
            than the legal sector of that spin.
    """
    # _random_strings already sorts; the sort here only keeps the contract explicit.
    alpha, beta = (
        np.sort(np.asarray(s, dtype=np.int64)) for s in _random_strings(norb, nelec, dims, seed)
    )
    return alpha, beta


def cisd_seed_strings(
    strings: np.ndarray,
    weights: np.ndarray,
    diagonal: np.ndarray,
    *,
    nelec_spin: int,
    size: int,
) -> tuple[np.ndarray, int]:
    """Return the CISD seed of one spin: HF plus the size - 1 top-ranked other strings.

    Ranking: sort the non-HF strings by descending weight, then join neighbours
    whose weight gap is at most ``_CISD_GROUP_RTOL * max(weights)`` into one group
    (single link). Groups keep the descending weight order. In a group, the lower
    diagonal energy comes first, then the lower integer.

    When fewer than ``size - 1`` non-HF strings are available, the seed holds all of
    them, so it has fewer than ``size`` strings. The HF string is added even when
    ``strings`` does not contain it. In production the CISD pool always contains HF.

    Args:
        strings: Candidate CI strings of one spin.
        weights: CISD weight of each string, aligned with ``strings``.
        diagonal: Diagonal energy of each string, aligned with ``strings``.
        nelec_spin: Electron count of the spin.
        size: Seed size, HF included.

    Returns:
        The ascending int64 seed strings, and the number of selected strings with a
        weight of exactly 0.0.

    Raises:
        ValueError: If the strings are invalid, if ``weights`` or ``diagonal`` does
            not align with ``strings``, or if ``size`` is less than 1.
    """
    if size < 1:
        raise ValueError(f"CISD seed size must be at least 1, got {size}")
    strings = _string_array(strings, nelec_spin)
    weights = np.asarray(weights, dtype=float)
    diagonal = np.asarray(diagonal, dtype=float)
    if weights.shape != strings.shape or diagonal.shape != strings.shape:
        raise ValueError("CISD weights and diagonals must align with the CI strings")
    hf = _hf(nelec_spin)
    others = strings != hf
    by_weight = np.argsort(-weights[others], kind="stable")
    ranked, w, d = (a[others][by_weight] for a in (strings, weights, diagonal))
    tol = _CISD_GROUP_RTOL * float(weights.max())
    group = np.cumsum(np.diff(w, prepend=w[:1]) < -tol)
    chosen = ranked[np.lexsort((ranked, d, group))][: size - 1]
    seed = np.union1d(chosen, [hf]).astype(np.int64)
    zero_count = int(np.count_nonzero(weights[np.isin(strings, seed)] == 0.0))
    return seed, zero_count


def cisd_state(
    h1: np.ndarray, h2: np.ndarray, e_core: float, *, norb: int, nelec: tuple[int, int]
) -> tuple[float, np.ndarray]:
    """Return the CISD ground state of an active-space Hamiltonian in its own MO basis.

    The RHF reference uses identity MO coefficients, so the CI vector stays in the
    basis of ``h1`` and ``h2``. No SCF runs, so degenerate orbitals do not rotate.

    Args:
        h1: One-electron integrals, shape (norb, norb).
        h2: Two-electron integrals in chemist notation, shape (norb,) * 4.
        e_core: Core energy added to the total energy.
        norb: Number of active orbitals.
        nelec: Alpha and beta electron counts. They must be equal.

    Returns:
        The total CISD energy (``e_core`` included), and the FCI-shaped CI vector of
        shape (alpha string count, beta string count).

    Raises:
        ValueError: If the alpha and beta electron counts are not equal.
        ReferenceNumericalError: If CISD does not converge to ``conv_tol``.
    """
    from pyscf import ao2mo, ci, gto, scf

    if nelec[0] != nelec[1]:
        raise ValueError("CISD seeds need a closed-shell reference (equal alpha and beta)")
    mol = gto.M(verbose=0)
    mol.nelectron = sum(nelec)
    mol.incore_anyway = True
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *_: np.asarray(h1, dtype=np.float64)
    mf.get_ovlp = lambda *_: np.eye(norb)
    mf.energy_nuc = lambda *_: float(e_core)
    mf._eri = ao2mo.restore(8, np.asarray(h2, dtype=np.float64), norb)
    mf.mo_coeff = np.eye(norb)
    mf.mo_occ = np.where(np.arange(norb) < nelec[0], 2.0, 0.0)
    dm = mf.make_rdm1()
    mf.mo_energy = np.diag(mf.get_fock(dm=dm))
    mf.e_tot = mf.energy_tot(dm=dm)
    solver = ci.CISD(mf)
    solver.conv_tol = 1e-12
    solver.kernel()
    if not solver.converged:
        raise ReferenceNumericalError(
            f"CISD did not converge to conv_tol={solver.conv_tol:g}; the CI vector is not valid"
        )
    return float(solver.e_tot), ci.cisd.to_fcivec(solver.ci, norb, nelec)


def _guard_seed(
    norb: int,
    nelec: tuple[int, int],
    dims: tuple[int, int],
    phase: str,
    audit: list[dict],
    resources: dict,
) -> None:
    """Price a ``dims`` product space with the ext guard and record ``phase`` in ``audit``."""
    predicted = estimate_rss_mb(
        norb,
        nelec,
        stage="diagonalize",
        subspace_dims=dims,
        retained_mb=resources["retained_mb"],
        model=resource_model_for(norb, profile={"max_space": 12}),
    )
    _guard_ext(norb, nelec, dims, predicted, phase, audit, resources)


def _cisd_seed(
    h1: np.ndarray,
    h2: np.ndarray,
    e_core: float,
    *,
    norb: int,
    nelec: tuple[int, int],
    dims: tuple[int, int],
    audit: list[dict],
    resources: dict,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Return the CISD seed strings per spin and the selected zero-weight strings."""
    from pyscf.fci import cistring, direct_spin1

    full = (math.comb(norb, nelec[0]), math.comb(norb, nelec[1]))
    _guard_seed(norb, nelec, full, "seed_cisd", audit, resources)
    _, vector = cisd_state(h1, h2, e_core, norb=norb, nelec=nelec)
    hdiag = direct_spin1.make_hdiag(h1, h2, norb, nelec).reshape(vector.shape)
    marginals = ((vector**2).sum(axis=1), (vector**2).sum(axis=0))
    diagonals = (hdiag[:, 0], hdiag[0, :])
    seeds, zero_weight = [], []
    for n, w, diag, size in zip(nelec, marginals, diagonals, dims, strict=True):
        pool = cistring.make_strings(range(norb), n)
        chosen, _ = cisd_seed_strings(pool, w, diag, nelec_spin=n, size=size)
        seeds.append(chosen)
        zero_weight.append(chosen[w[np.searchsorted(pool, chosen)] == 0.0])
    return (seeds[0], seeds[1]), np.concatenate(zero_weight).astype(np.int64)


def build_seed_space(
    source: str,
    h1: np.ndarray,
    h2: np.ndarray,
    e_core: float,
    *,
    norb: int,
    nelec: tuple[int, int],
    dims: tuple[int, int],
    seed: int,
    audit: list[dict],
    resources: dict,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], np.ndarray], float, np.ndarray | None]:
    """Build a seed and diagonalize the Hamiltonian in its alpha x beta product space.

    The ``hf`` seed is always the single HF determinant; it ignores ``dims``. The
    ``random`` seed is the SQD random draw for ``seed``. The ``cisd`` seed ranks
    strings by CISD marginal weight; the guard prices the full sector before CISD runs.

    Args:
        source: Seed source: ``"hf"``, ``"random"`` or ``"cisd"``.
        h1: One-electron integrals, shape (norb, norb).
        h2: Two-electron integrals in chemist notation.
        e_core: Core energy in Hartree.
        norb: Number of active orbitals.
        nelec: Electron counts (alpha, beta).
        dims: Seed sizes (alpha, beta), HF string included.
        seed: Random seed for the ``random`` source.
        audit: List that receives one ext-guard record per guarded phase.
        resources: Guard and solver resources: ``host_available_mb`` and optional
            ``rss_budget_mb``, ``retained_mb``, ``allow_large``.

    Returns:
        ``((alpha, beta), amplitudes)`` with alpha-major amplitudes, the ground-state
        energy including ``e_core``, and the selected zero-weight strings (alpha then
        beta, int64) for ``cisd`` or ``None`` for the other sources.

    Raises:
        ValueError: If ``source`` is not ``"hf"``, ``"random"`` or ``"cisd"``.
        AllocationGuardError: If the ext guard rejects a phase.
    """
    resources = {"rss_budget_mb": 8192.0, "retained_mb": 0.0, "allow_large": False} | resources
    zero_weight = None
    if source == "hf":
        strings = hf_seed_strings(norb, nelec)
    elif source == "random":
        strings = random_seed_strings(norb, nelec, dims, seed)
    elif source == "cisd":
        strings, zero_weight = _cisd_seed(
            h1, h2, e_core, norb=norb, nelec=nelec, dims=dims, audit=audit, resources=resources
        )
    else:
        raise ValueError(f"seed source must be 'hf', 'random' or 'cisd', got {source!r}")
    space = (len(strings[0]), len(strings[1]))
    _guard_seed(norb, nelec, space, "seed_space", audit, resources)
    solved = kernel_fixed_space(h1, h2, e_core, strings, norb=norb, nelec=nelec, **resources)
    return (solved.ci_strings, solved.amplitudes), float(solved.energy), zero_weight


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


def sci_ranking(
    h1: np.ndarray,
    h2: np.ndarray,
    e_core: float,
    *,
    norb: int,
    nelec: tuple[int, int],
    host_available_mb: float,
    rss_budget_mb: float,
    retained_mb: float,
    audit: list[dict],
) -> tuple[float, Any, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Guard and run the one selected-CI solve that ranks strings for the SCI baseline.

    The solve diagonalizes the full sector, so the guard prices the full sector
    ``(C(norb, n_alpha), C(norb, n_beta))`` and records ``phase="sci_ranking"`` in ``audit``.

    Args:
        h1: Active-space one-electron integrals.
        h2: Active-space two-electron integrals.
        e_core: Core energy that is added to the selected-CI energy.
        norb: Number of active orbitals.
        nelec: Active electrons as ``(n_alpha, n_beta)``.
        host_available_mb: Available host memory in decimal MB.
        rss_budget_mb: RSS budget of the guard in decimal MB.
        retained_mb: Memory that the caller holds during the solve, in decimal MB.
        audit: List that receives one ext guard record.

    Returns:
        ``(energy, ci, pools, diags)``: the total selected-CI energy, the CI vector,
        and the per-spin string pools and diagonals.

    Raises:
        AllocationGuardError: The guard rejects the full sector. The solve does not run.
    """
    dims = (math.comb(norb, nelec[0]), math.comb(norb, nelec[1]))
    resources = {
        "host_available_mb": host_available_mb,
        "rss_budget_mb": rss_budget_mb,
        "retained_mb": retained_mb,
        "allow_large": False,
    }
    _guard_seed(norb, nelec, dims, "sci_ranking", audit, resources)
    energy, ci, pools, diags = _run_selected_ci(h1, h2, norb, nelec)
    return float(energy) + e_core, ci, pools, diags


def _nbytes_mb(*arrays: np.ndarray) -> float:
    """Return the resident size of ``arrays`` in decimal MB, for a guard's ``retained_mb``."""
    return sum(array.nbytes for array in arrays) / 1e6


def _sci_baseline(
    h1: np.ndarray,
    h2: np.ndarray,
    e_core: float,
    sci_ranking: (
        tuple[float, Any, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None
    ),
    dims: tuple[int, int],
    *,
    norb: int,
    nelec: tuple[int, int],
    audit: list[dict],
    **resources: Any,
) -> tuple[float | None, str | None]:
    """Solve the space of the top-ranked SCI strings with the per-spin dims of the arm.

    Returns:
        ``(energy, None)`` when the baseline runs. ``(None, reason)`` when no ranking
        is available, the guard rejects the baseline space, or the baseline solve fails
        numerically. A rejection appends one ``phase="sci_baseline"`` audit record and
        does not raise.
    """
    if sci_ranking is None:
        return None, "sci_ranking_unavailable"
    _, ci, pools, diags = sci_ranking
    space = tuple(
        np.sort(_take_spin_strings(ci, spin, pools[spin], diags[spin], nelec[spin], dims[spin])[0])
        for spin in (0, 1)
    )
    predicted = estimate_rss_mb(
        norb,
        nelec,
        stage="diagonalize",
        subspace_dims=dims,
        retained_mb=resources["retained_mb"],
        model=resource_model_for(norb, profile={"max_space": 12}),
    )
    try:
        _guard_ext(norb, nelec, dims, predicted, "sci_baseline", audit, resources)
    except AllocationGuardError as exc:
        return None, f"sci_baseline_rss_budget: {exc}"
    try:
        solved = kernel_fixed_space(h1, h2, e_core, space, norb=norb, nelec=nelec, **resources)
    except ReferenceNumericalError as exc:
        return None, f"sci_baseline_numerical: {exc}"
    return float(solved.energy), None


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
    sci_ranking: (
        tuple[float, Any, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None
    ) = None,
    seed_space_energy: float | None = None,
    zero_weight_strings: np.ndarray | None = None,
) -> ExtSQDArm:
    """Truncate the seeds, excite them, guard and solve one fixed-space problem.

    The order is truncation, excitation, wall-time precheck, resource guard,
    ``kernel_fixed_space``, SCI baseline. No configuration recovery occurs. The solver
    repeats the guard as a second check with the same resource arguments. The guard
    appends one ``stage="ext"`` record to ``audit`` when it passes and when it rejects.

    Retained memory accumulates. Before the guard, ``retained_mb`` gets the MB of
    the ``sci_ranking`` arrays (CI vector, pools, diagonals) when ``sci_ranking`` is
    given. Before the SCI baseline, it also gets the MB of the extended strings and
    the extended-space amplitudes.

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
        sci_ranking: The ``sci_ranking`` result for the same Hamiltonian, or None.
        seed_space_energy: Energy in the seed space, written to the arm unchanged.
            None for the quantum arm.
        zero_weight_strings: Int64 strings with zero seed weight, written to the arm
            unchanged, or None.

    Returns:
        The arm with the total energy, string counts and the guard prediction. When
        less than twice the predicted arm time remains before ``deadline``, the arm
        has no energy, ``null_reason="ext_wall_budget"``, and neither the guard nor
        the solver runs. A solved arm also has the SCI baseline energy in the same
        per-spin dims, or the reason why the baseline did not run.

    Raises:
        ValueError: Seed strings, amplitudes or populations that the string
            functions reject.
        AllocationGuardError: The resource guard rejects the extended space. The
            message starts with ``stage=ext``.
        ResourceLimitError: No resource bound applies to the extended space.
        ReferenceNumericalError: The extended-space solve fails numerically. A
            numerical failure of the SCI baseline solve does not raise.
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
    if sci_ranking is not None:
        _, ci, pools, diags = sci_ranking
        resources["retained_mb"] += _nbytes_mb(ci, *pools, *diags)
    predicted = estimate_rss_mb(
        norb,
        nelec,
        stage="diagonalize",
        subspace_dims=dims,
        retained_mb=resources["retained_mb"],
        model=resource_model_for(norb, profile={"max_space": 12}),
    )
    energy, null_reason = None, "ext_wall_budget"
    baseline, baseline_reason = None, "sci_baseline_arm_null"
    if not _over_wall_budget(deadline, seed_source, dims):
        phase = f"ext_{seed_source}"
        predicted = _guard_ext(norb, nelec, dims, predicted, phase, audit, resources)
        solved = kernel_fixed_space(h1, h2, e_core, extended, norb=norb, nelec=nelec, **resources)
        energy, null_reason = float(solved.energy), None
        resources["retained_mb"] += _nbytes_mb(*extended, solved.amplitudes)
        baseline, baseline_reason = _sci_baseline(
            h1, h2, e_core, sci_ranking, dims, norb=norb, nelec=nelec, audit=audit, **resources
        )
    return ExtSQDArm(
        seed_source=seed_source,
        energy=energy,
        seed_counts=(len(ci_strings[0]), len(ci_strings[1])),
        truncated_counts=tuple(int(s.size) for s in truncation.ci_strings),
        extended_counts=dims,
        subspace_dims=dims,
        coverage=dims[0] / math.comb(norb, nelec[0]),
        only_hf=truncation.only_hf,
        seed_space_energy=seed_space_energy,
        zero_weight_strings=zero_weight_strings,
        sci_baseline_energy=baseline,
        sci_baseline_reason=baseline_reason,
        wall_s=time.monotonic() - start,
        predicted_rss_mb=predicted,
        process_peak_rss_mb=_peak_rss_mb(),
        null_reason=null_reason,
    )
