"""Guarded ffsim preparation and beta-left/alpha-right big-endian sampling.

Numerical calls require the frozen resource-model environment and an independent
process-tree RSS/deadline supervisor. ffsim 0.0.84 expresses big-endian order with
BitstringType.BIT_ARRAY (there is no ``order`` keyword); orbital zero is rightmost in
each spin block, compatible with downstream ``order='big'`` decoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from q2m3.sqd.config import _integer, validate_active_space
from q2m3.sqd.exceptions import SamplingIntegrityError
from q2m3.sqd.resources import guard_allocation, load_resource_model

if TYPE_CHECKING:
    import ffsim


def _check_state(state: np.ndarray, norb: int, nelec: tuple[int, int], stage: str) -> None:
    dimension = math.prod(math.comb(norb, count) for count in nelec)
    if (
        not isinstance(state, np.ndarray)
        or state.shape != (dimension,)
        or state.dtype.kind not in "fci"
        or not np.all(np.isfinite(state))
    ):
        raise SamplingIntegrityError(
            f"backend=ffsim stage={stage}: invalid state shape/dtype/finite"
        )
    # Integer dot products can wrap huge amplitudes into a unit norm.
    # Widen before multiplication; a normalized integer vector can only contain
    # one +/-1, so subsequent backend arithmetic is also safe for that domain.
    norm_state = state.astype(np.complex128, copy=False)
    norm_squared = float(np.vdot(norm_state, norm_state).real)
    if not np.isfinite(norm_squared) or abs(norm_squared - 1.0) > 1e-10:
        raise SamplingIntegrityError(
            f"backend=ffsim stage={stage}: state norm squared={norm_squared}, expected 1"
        )


def validate_samples(
    samples: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    shots: int,
    backend: str = "ffsim",
    stage: str = "sample",
) -> np.ndarray:
    """Check raw backend output before any conversion can hide invalid values.

    Args:
        samples: Raw bool matrix, beta block on the left and alpha on the right.
        norb: Spatial orbital count.
        nelec: Alpha and beta populations.
        shots: Exact requested number of nonempty samples.
        backend: Backend name for diagnostic context.
        stage: Stage name for diagnostic context.

    Returns:
        The validated matrix, without coercion or repair.

    Raises:
        SamplingIntegrityError: Invalid shape, dtype, finiteness or particle count.
    """
    validate_active_space(norb, nelec)
    _integer(shots, "shots", 1)
    context = f"backend={backend} stage={stage}"
    if not isinstance(samples, np.ndarray) or samples.shape != (shots, 2 * norb):
        raise SamplingIntegrityError(f"{context}: invalid or empty sample shape")
    if samples.dtype != np.dtype(bool) or not np.all(np.isfinite(samples)):
        raise SamplingIntegrityError(f"{context}: samples must have finite bool dtype")
    if not (
        np.all(samples[:, :norb].sum(axis=1) == nelec[1])
        and np.all(samples[:, norb:].sum(axis=1) == nelec[0])
    ):
        raise SamplingIntegrityError(f"{context}: beta/alpha Hamming weights differ from nelec")
    return samples


@dataclass(frozen=True)
class FfsimSampler:
    """A supported backend with mandatory caps and extra retained-input memory.

    ``retained_mb`` covers extra live inputs beyond the calibrated serial active-
    space schedule, e.g. a larger AO/full-MO integral source kept by the caller.
    This low-level sampler does not verify seed provenance; obtain its operator
    through ``build_lucj`` to enforce same-Hamiltonian/frame CCSD acceptance.
    """

    host_available_mb: float
    rss_budget_mb: float
    retained_mb: float = 0.0

    def _guard(self, norb: int, nelec: tuple[int, int], stage: str, **profile: int) -> None:
        model = load_resource_model(profile=profile)
        guard_allocation(
            norb,
            nelec,
            stage=stage,
            model=model,
            host_available_mb=self.host_available_mb,
            rss_budget_mb=self.rss_budget_mb,
            retained_mb=self.retained_mb,
        )

    def prepare(
        self, operator: ffsim.UCJOpSpinBalanced, norb: int, nelec: tuple[int, int]
    ) -> np.ndarray:
        """Apply an actual UCJ operator to HF and verify its full-CI state.

        Args:
            operator: Spin-balanced operator from build_lucj after seed acceptance.
            norb: Active spatial orbital count matching the operator.
            nelec: Balanced alpha/beta populations.

        Returns:
            Finite normalized full-CI state in ffsim address order.

        Raises:
            SamplingIntegrityError: Backend returns an invalid state.
            ValueError: Operator or spin domain is unsupported.
        """
        validate_active_space(norb, nelec)
        import ffsim

        if not isinstance(operator, ffsim.UCJOpSpinBalanced) or operator.norb != norb:
            raise ValueError("prepare requires a matching UCJOpSpinBalanced operator")
        self._guard(norb, nelec, "prepare", n_reps=operator.n_reps)
        try:
            state = ffsim.hartree_fock_state(norb, nelec)
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            raise SamplingIntegrityError(f"backend=ffsim stage=prepare: {exc}") from exc
        _check_state(state, norb, nelec, "prepare")
        try:
            state = ffsim.apply_unitary(state, operator, norb=norb, nelec=nelec)
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            raise SamplingIntegrityError(f"backend=ffsim stage=prepare: {exc}") from exc
        _check_state(state, norb, nelec, "prepare")
        return state

    def sample(
        self,
        state: np.ndarray,
        norb: int,
        nelec: tuple[int, int],
        *,
        shots: int = 100_000,
        seed: int = 0,
    ) -> np.ndarray:
        """Draw reproducible bool samples in beta/alpha big-endian block order.

        Args:
            state: Finite normalized full-CI vector.
            norb: Active spatial orbital count.
            nelec: Balanced alpha/beta populations.
            shots: Statistical sampling budget, within the calibrated profile.
            seed: Nonnegative integer for a fresh reproducible random stream.

        Returns:
            Bool array with shape (shots, 2*norb), beta left and alpha right.

        Raises:
            SamplingIntegrityError: State or backend output fails integrity checks.
            ValueError: Invalid counts or seed.
        """
        validate_active_space(norb, nelec)
        _integer(shots, "shots", 1)
        _integer(seed, "seed")
        self._guard(norb, nelec, "sample", shots=shots)
        _check_state(state, norb, nelec, "sample")
        import ffsim

        try:
            samples = ffsim.sample_state_vector(
                state,
                norb=norb,
                nelec=nelec,
                shots=shots,
                seed=seed,
                concatenate=True,
                bitstring_type=ffsim.BitstringType.BIT_ARRAY,
            )
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            raise SamplingIntegrityError(f"backend=ffsim stage=sample: {exc}") from exc
        return validate_samples(samples, norb, nelec, shots=shots)
