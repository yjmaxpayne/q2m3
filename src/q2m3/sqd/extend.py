"""String-level seed truncation and excitation for extended SQD.

The functions act on per-spin CI string sets. They run no diagonalization and
no resource guard. Callers guard the fixed-space solve that follows them.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from q2m3.sqd.observables import _sector

_ORDERS = (1, 2, 3)
_CHUNK_BYTES = 8_000_000


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
