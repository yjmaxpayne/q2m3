"""Bounded published SQD recovery and a fixed-space PySCF adapter.

Callers must enforce an independent process-tree RSS cap and wall deadline.
The direct adapter performs no configuration selection or recovery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q2m3.sqd.config import _finite, _integer, _real_array, validate_active_space
from q2m3.sqd.exceptions import ReferenceNumericalError, SamplingIntegrityError
from q2m3.sqd.resources import addon_subspace_bound, guard_allocation, load_resource_model
from q2m3.sqd.sampling import validate_samples


@dataclass(frozen=True)
class AllocationAudit:
    """One pre-allocation estimate; iteration -1 denotes preprocessing bound."""

    iteration: int
    batch: int
    subspace_dims: tuple[int, int]
    predicted_rss_mb: float


@dataclass(frozen=True)
class DiagonalizationResult:
    """Lowest returned energy and alpha-major/beta-minor fixed-space state."""

    energy: float
    electronic_energy: float
    ci_strings: tuple[np.ndarray, np.ndarray]
    amplitudes: np.ndarray
    subspace_dims: tuple[int, int]
    allocation_audit: tuple[AllocationAudit, ...]


def _integrals(h1, h2, e_core, norb, nelec):
    validate_active_space(norb, nelec)
    h1 = _real_array(h1, (norb, norb), "h1")
    h2 = _real_array(h2, (norb,) * 4, "h2")
    _finite(e_core, "e_core")
    for array, transpose in (
        (h1, h1.T),
        (h2, h2.transpose(1, 0, 2, 3)),
        (h2, h2.transpose(0, 1, 3, 2)),
        (h2, h2.transpose(2, 3, 0, 1)),
    ):
        if not np.allclose(array, transpose, atol=1e-12, rtol=0):
            raise ValueError("Integrals violate real chemist permutation symmetry")
    return np.asarray(h1, dtype=np.float64, order="C"), np.asarray(h2, dtype=np.float64, order="C")


def _strings(strings, norb, nelec):
    if not isinstance(strings, tuple) or len(strings) != 2:
        raise ValueError("ci_strings must be an alpha/beta pair")
    for sector, electrons in zip(strings, nelec, strict=True):
        if (
            not isinstance(sector, np.ndarray)
            or sector.ndim != 1
            or not sector.size
            or sector.dtype.kind not in "iu"
        ):
            raise ValueError("CI sectors must be nonempty one-dimensional integer arrays")
        if any(
            int(s) < 0 or int(s) >= 1 << norb or int(s).bit_count() != electrons for s in sector
        ):
            raise ValueError("CI strings violate orbital range or particle count")
        if np.any(sector[1:] <= sector[:-1]):
            raise ValueError("CI strings must be unique and ascending")
    return tuple(len(s) for s in strings)


def _result(energy, constant, strings, amplitudes, audit):
    dims = tuple(len(s) for s in strings)
    if (
        not np.isfinite(energy)
        or not np.isfinite(energy + constant)
        or amplitudes.shape != dims
        or not np.all(np.isfinite(amplitudes))
        or abs(np.linalg.norm(amplitudes) - 1) > 1e-8
    ):
        raise ReferenceNumericalError("Nonfinite or unnormalized diagonalization result")

    def frozen(a):
        return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)

    return DiagonalizationResult(
        float(energy + constant),
        float(energy),
        tuple(frozen(s) for s in strings),
        frozen(amplitudes),
        dims,
        tuple(audit),
    )


def kernel_fixed_space(
    h1: np.ndarray,
    h2: np.ndarray,
    e_core: float,
    ci_strings: tuple[np.ndarray, np.ndarray],
    *,
    norb: int,
    nelec: tuple[int, int],
    host_available_mb: float,
    rss_budget_mb: float = 8192.0,
    retained_mb: float = 0.0,
    allow_large: bool = False,
) -> DiagonalizationResult:
    """Solve a specified Cartesian CI basis without selection or recovery.

    Args:
        h1: Real symmetric one-electron integrals in Hartree.
        h2: Real chemist two-electron tensor in the same frame.
        e_core: Constant added exactly once to the electronic energy.
        ci_strings: Ascending unique alpha and beta integer determinant arrays.
        norb: Spatial orbitals.
        nelec: Balanced spin populations.
        host_available_mb: Executor's total memory cap, decimal MB.
        rss_budget_mb: User memory cap, decimal MB.
        retained_mb: Additional live inputs outside the calibrated inventory.
        allow_large: Override the soft memory threshold only.

    Returns:
        Energy, immutable fixed-space state, and pre-allocation audit.

    Raises:
        ValueError: Invalid integrals or CI strings.
        ResourceLimitError: Workload or memory budget is unsupported.
        ReferenceNumericalError: Nonfinite, unconverged or invalid solver output.
    """
    validate_active_space(norb, nelec)
    dims = _strings(ci_strings, norb, nelec)
    model = load_resource_model(profile={"max_space": 12})
    estimate = guard_allocation(
        norb,
        nelec,
        stage="diagonalize",
        model=model,
        subspace_dims=dims,
        host_available_mb=host_available_mb,
        rss_budget_mb=rss_budget_mb,
        retained_mb=retained_mb,
        allow_large=allow_large,
    )
    h1, h2 = _integrals(h1, h2, e_core, norb, nelec)
    from pyscf.fci import selected_ci

    ci_strings = tuple(np.asarray(s, dtype=np.int64) for s in ci_strings)
    solver = selected_ci.SelectedCI()
    energy, state = selected_ci.kernel_fixed_space(
        solver,
        h1,
        h2,
        norb,
        nelec,
        ci_strs=ci_strings,
        ecore=0,
        max_space=12,
        max_cycle=100,
        tol=1e-12,
    )
    if not solver.converged:
        raise ReferenceNumericalError("Fixed-space Davidson solver did not converge")
    return _result(
        energy, e_core, state._strs, np.asarray(state), [AllocationAudit(0, 0, dims, estimate)]
    )


def diagonalize_samples(
    h1: np.ndarray,
    h2: np.ndarray,
    e_core: float,
    samples: np.ndarray,
    *,
    norb: int,
    nelec: tuple[int, int],
    host_available_mb: float,
    rss_budget_mb: float = 8192.0,
    retained_mb: float = 0.0,
    allow_large: bool = False,
    samples_per_batch: int | None = None,
    num_batches: int = 2,
    max_iterations: int = 2,
    seed: int = 0,
    symmetrize_spin: bool = False,
) -> DiagonalizationResult:
    """Run published addon recovery, batching and self-consistency on strict samples.

    Args:
        h1: Real symmetric one-electron integrals in Hartree.
        h2: Chemist two-electron integrals in the same orbital frame.
        e_core: Constant added exactly once.
        samples: Nonempty bool matrix, beta-left/alpha-right, orbital zero rightmost.
        norb: Spatial orbitals.
        nelec: Balanced spin populations; every sample must conserve both.
        host_available_mb: Executor memory cap, decimal MB.
        rss_budget_mb: User memory cap, decimal MB.
        retained_mb: Extra live inputs beyond the model inventory.
        allow_large: Override the soft threshold only.
        samples_per_batch: Sample count per batch; defaults to min(300, shots).
        num_batches: One or two serial batches.
        max_iterations: One or two recovery iterations.
        seed: Nonnegative random seed.
        symmetrize_spin: Merge spin sectors before solving, when requested.

    Returns:
        Best published-addon state with all candidate allocation audits. HF is
        included in every candidate; final dimensions describe the best state,
        which need not belong to the last iteration.

    Raises:
        SamplingIntegrityError: Invalid sample shape, dtype or particle counts.
        ValueError: Invalid input tensors or parameters.
        ResourceLimitError: Unsupported workload or insufficient memory budget.
        ReferenceNumericalError: Invalid numerical output.
    """
    shots = samples.shape[0] if isinstance(samples, np.ndarray) and samples.ndim == 2 else 1
    validate_samples(samples, norb, nelec, shots=shots, backend="addon", stage="diagonalize_input")
    return _diagonalize_recoverable_samples(
        h1,
        h2,
        e_core,
        samples,
        norb=norb,
        nelec=nelec,
        host_available_mb=host_available_mb,
        rss_budget_mb=rss_budget_mb,
        retained_mb=retained_mb,
        allow_large=allow_large,
        samples_per_batch=samples_per_batch,
        num_batches=num_batches,
        max_iterations=max_iterations,
        seed=seed,
        symmetrize_spin=symmetrize_spin,
    )


def _diagonalize_recoverable_samples(
    h1,
    h2,
    e_core,
    samples,
    *,
    norb,
    nelec,
    host_available_mb,
    rss_budget_mb=8192.0,
    retained_mb=0.0,
    allow_large=False,
    samples_per_batch=None,
    num_batches=2,
    max_iterations=2,
    seed=0,
    symmetrize_spin=False,
    initial_occupancies=None,
):
    """Internal recovery contract seam: raw weights may be invalid, structure may not."""
    validate_active_space(norb, nelec)
    if (
        not isinstance(samples, np.ndarray)
        or samples.ndim != 2
        or samples.shape[1] != 2 * norb
        or not len(samples)
        or samples.dtype != np.dtype(bool)
    ):
        raise SamplingIntegrityError("Recovery input must be a nonempty bool matrix")
    _integer(seed, "seed", 0)
    _integer(num_batches, "num_batches", 1)
    _integer(max_iterations, "max_iterations", 1)
    if not isinstance(symmetrize_spin, bool):
        raise ValueError("symmetrize_spin must be bool")
    samples_per_batch = min(300, len(samples)) if samples_per_batch is None else samples_per_batch
    _integer(samples_per_batch, "samples_per_batch", 1)
    if samples_per_batch > len(samples):
        raise ValueError("samples_per_batch cannot exceed shots")
    model = load_resource_model(
        profile=dict(
            shots=len(samples),
            num_batches=num_batches,
            max_iterations=max_iterations,
            max_space=12,
            carryover_threshold="0",
        )
    )
    audit = []

    def guard(dims, iteration, batch):
        estimate = guard_allocation(
            norb,
            nelec,
            stage="diagonalize",
            model=model,
            subspace_dims=dims,
            host_available_mb=host_available_mb,
            rss_budget_mb=rss_budget_mb,
            retained_mb=retained_mb,
            allow_large=allow_large,
        )
        audit.append(AllocationAudit(iteration, batch, dims, estimate))

    guard(addon_subspace_bound(norb, nelec), -1, -1)
    h1, h2 = _integrals(h1, h2, e_core, norb, nelec)
    if initial_occupancies is not None:
        if (
            not isinstance(initial_occupancies, tuple)
            or len(initial_occupancies) != 2
            or any(
                not isinstance(o, np.ndarray)
                or o.shape != (norb,)
                or not np.all(np.isfinite(o))
                or np.any(o < 0)
                or np.any(o > 1)
                for o in initial_occupancies
            )
        ):
            raise ValueError("initial_occupancies must contain finite orbital probabilities")
    from qiskit.primitives.containers import BitArray
    from qiskit_addon_sqd import fermion

    iteration = 0

    def guarded_solver(batches, one, two, n, electrons):
        nonlocal iteration
        if not batches:
            raise ValueError("Empty candidate batch list")
        # Validate and guard ALL batches before the first native workspace allocation.
        for batch, strings in enumerate(batches):
            guard(_strings(strings, n, electrons), iteration, batch)
        results = [
            fermion.solve_sci(
                strings, one, two, n, electrons, max_space=12, max_cycle=100, tol=1e-12
            )
            for strings in batches
        ]
        for result in results:
            _result(
                result.energy,
                e_core,
                (result.sci_state.ci_strs_a, result.sci_state.ci_strs_b),
                result.sci_state.amplitudes,
                (),
            )
            from pyscf.fci import direct_spin1, selected_ci

            state = result.sci_state
            vector = np.asarray(state.amplitudes).view(selected_ci.SCIvector)
            vector._strs = (state.ci_strs_a, state.ci_strs_b)
            effective = direct_spin1.absorb_h1e(one, two, n, electrons, 0.5)
            action = selected_ci.contract_2e(effective, vector, n, electrons)
            residual = float(np.linalg.norm(action - result.energy * vector))
            if not np.isfinite(residual) or residual > 1e-6:
                raise ReferenceNumericalError(
                    f"Projected eigenvector residual {residual} Ha exceeds 1e-6 Ha"
                )
        iteration += 1
        return results

    result = fermion.diagonalize_fermionic_hamiltonian(
        h1,
        h2,
        BitArray.from_bool_array(samples, order="big"),
        samples_per_batch,
        norb,
        nelec,
        num_batches=num_batches,
        max_iterations=max_iterations,
        sci_solver=guarded_solver,
        symmetrize_spin=symmetrize_spin,
        include_configurations=([2 ** nelec[0] - 1], [2 ** nelec[1] - 1]),
        initial_occupancies=initial_occupancies,
        carryover_threshold=0,
        seed=seed,
    )
    state = result.sci_state
    return _result(
        result.energy, e_core, (state.ci_strs_a, state.ci_strs_b), state.amplitudes, audit
    )
