"""Fixed-frame RCCSD seeds and deterministic ffsim LUCJ construction.

The allocation inventory is a planning estimate with an unvalidated native
reserve. Every caller must independently supervise process-tree RSS and timeout.
No SCF optimization or orbital-frame change occurs in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from numbers import Integral
from typing import Any

import numpy as np

from q2m3.molecule import MoleculeConfig
from q2m3.sqd.config import (
    CCSDSeed,
    IntegralContext,
    LUCJConfig,
    _validate_pairs,
    validate_integral_inputs,
)
from q2m3.sqd.exceptions import (
    CCSDConvergenceError,
    ProvenanceMismatchError,
    ResourceModelDomainError,
)
from q2m3.sqd.integrals import IntegralData, _frame_id, hamiltonian_id
from q2m3.sqd.resources import Stage, guard_allocation, load_resource_model


@dataclass(frozen=True)
class _SeedInventory:
    """Serial real RCCSD, DIIS6, <=200 cycles, or rank-2 UCJ decomposition.

    128 float64 n^4 arrays cover ERIs, t2/DIIS, simultaneous receiver inputs,
    update scratch and double-factorization matrices; 64 n^2 arrays cover
    singles/Fock/orbitals. Full retained MO is counted separately. The 1024 MB
    import/native reserve is not a certified bound and needs external enforcement.
    """

    n_mo: int
    model_id: str = "fixed-frame-rccsd-inventory-v1"

    def upper_bound_mb(
        self,
        norb: int,
        nelec: tuple[int, int],
        *,
        stage: Stage,
        subspace_dims: tuple[int, int] | None,
        retained_mb: float,
        solver_method: str | None,
    ) -> float:
        if stage != "ccsd" or not 2 <= norb <= 10 or not norb <= self.n_mo <= 32:
            raise ResourceModelDomainError("Fixed-frame RCCSD inventory domain exceeded")
        return 1024 + retained_mb + 8 * (128 * norb**4 + 64 * norb**2 + self.n_mo**2) / 1e6


@dataclass(frozen=True)
class _ActiveHamiltonian:
    h1: np.ndarray
    h2: np.ndarray
    e_core: float
    norb: int
    nelec: tuple[int, int]
    context: IntegralContext


def _guard_active(
    data: IntegralData | _ActiveHamiltonian,
    n_mo: int,
    host_available_mb: float,
    rss_budget_mb: float,
    allow_large: bool,
) -> None:
    if version("pyscf") != "2.14.0":
        raise ResourceModelDomainError("Fixed-frame RCCSD inventory requires PySCF 2.14.0")
    guard_allocation(
        data.norb,
        data.nelec,
        stage="ccsd",
        model=_SeedInventory(n_mo),
        host_available_mb=host_available_mb,
        rss_budget_mb=rss_budget_mb,
        allow_large=allow_large,
    )


def _guard(
    data: IntegralData,
    host_available_mb: float,
    rss_budget_mb: float,
    allow_large: bool,
) -> None:
    if not isinstance(data, IntegralData):
        raise ValueError("data must be IntegralData")
    if data.mo_coeff.ndim != 2 or data.mo_coeff.shape[0] != data.mo_coeff.shape[1]:
        raise ValueError("Complete square MO frame is required")
    _guard_active(data, len(data.mo_coeff), host_available_mb, rss_budget_mb, allow_large)


def _authenticate_active(
    data: IntegralData | _ActiveHamiltonian,
    seed_data: CCSDSeed | None = None,
) -> None:
    validate_integral_inputs(
        data.h1,
        data.h2,
        data.e_core,
        norb=data.norb,
        nelec=data.nelec,
        context=data.context,
        seed_data=seed_data,
        mode="reference_only" if seed_data is None else "full",
    )
    actual = hamiltonian_id(data.h1, data.h2, data.e_core, norb=data.norb, nelec=data.nelec)
    if actual != data.context.hamiltonian_id:
        raise ProvenanceMismatchError("Actual Hamiltonian hash differs from context")
    if seed_data is not None and not np.allclose(
        seed_data.t2, seed_data.t2.transpose(1, 0, 3, 2), atol=1e-12, rtol=0
    ):
        raise ValueError("RCCSD t2 simultaneous exchange symmetry is required")


def _authenticate(data: IntegralData, seed_data: CCSDSeed | None = None) -> None:
    _authenticate_active(data, seed_data)
    source = data.context.source
    try:
        molecule = MoleculeConfig(
            "frame-verification",
            list(source["symbols"]),
            source["coordinates_angstrom"],
            source["charge"],
            2 * data.nelec[0],
            data.norb,
            basis=source["basis"],
        )
        frame = _frame_id(
            molecule, data.mo_coeff, data.context.active_indices, data.context.n_core_orbitals
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProvenanceMismatchError("Complete molecular frame source is required") from exc
    if frame != data.context.frame_id:
        raise ProvenanceMismatchError("Actual orbital frame hash differs from context")


def _hf_fock(data: IntegralData | _ActiveHamiltonian) -> tuple[float, np.ndarray]:
    o = data.nelec[0]
    h2 = data.h2
    fock = data.h1 + 2 * np.einsum("pqii->pq", h2[:, :, :o, :o])
    fock -= np.einsum("piiq->pq", h2[:, :o, :o, :])
    hf = data.e_core + 2 * np.trace(data.h1[:o, :o])
    hf += 2 * np.einsum("iijj", h2[:o, :o, :o, :o]) - np.einsum("ijji", h2[:o, :o, :o, :o])
    return float(hf), fock


def _solver(
    data: IntegralData | _ActiveHamiltonian,
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    from pyscf import ao2mo, cc, gto, scf

    hf, fock = _hf_fock(data)
    mol = gto.M(verbose=0)
    mol.nelectron = sum(data.nelec)
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: data.h1
    mf.get_ovlp = lambda *args: np.eye(data.norb)
    mf._eri = ao2mo.restore(8, data.h2, data.norb)
    mf.mo_coeff = np.eye(data.norb)
    mf.mo_occ = np.array([2.0] * data.nelec[0] + [0.0] * (data.norb - data.nelec[0]))
    mf.mo_energy = np.diag(fock)
    mf.e_tot = hf
    mf.converged = True
    solver = cc.CCSD(mf)
    solver.conv_tol = 1e-13
    solver.conv_tol_normt = 1e-11
    solver.diis_space = 6
    solver.max_memory = 1024
    solver.incore_complete = True
    solver.level_shift = 0
    eris = solver.ao2mo()
    o = data.nelec[0]
    denominators = eris.mo_energy[:o, None] - eris.mo_energy[None, o:]
    doubles = denominators[:, None, :, None] + denominators[None, :, None, :]
    if np.any(np.abs(denominators) < 1e-12) or np.any(np.abs(doubles) < 1e-12):
        raise CCSDConvergenceError("Singular fixed-frame CCSD Jacobi denominator")
    return solver, eris, denominators, doubles


def _physical_values(
    data: IntegralData | _ActiveHamiltonian,
    t1: np.ndarray,
    t2: np.ndarray,
) -> tuple[float, float, float]:
    """Independently contract HF/Ecc; undo Jacobi denominators for Ha equations.

    PySCF 2.14 RCCSD update_amps divides the equation numerator by eia/eijab.
    Multiplying its update defect by these unshifted denominators recovers the
    actual projected CC equations in Ha, not the dimensionless update defect.
    Tests compare this against explicit e^-T H e^T alpha/alpha-beta projections.
    """
    hf, fock = _hf_fock(data)
    if not np.isfinite(hf) or not np.all(np.isfinite(fock)):
        raise CCSDConvergenceError("Nonfinite HF energy or Fock matrix")
    if not t1.size:
        return hf, hf, 0.0
    o = data.nelec[0]
    g = data.h2[:o, o:, :o, o:].transpose(0, 2, 1, 3)
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    energy = hf + 2 * np.einsum("ia,ia", fock[:o, o:], t1)
    energy += np.einsum("ijab,ijab", tau, 2 * g - g.transpose(0, 1, 3, 2))
    solver, eris, d1, d2 = _solver(data)
    next_t1, next_t2 = solver.update_amps(t1, t2, eris)
    if not np.all(np.isfinite(next_t1)) or not np.all(np.isfinite(next_t2)):
        raise CCSDConvergenceError("Nonfinite CCSD equation update")
    r1, r2 = (next_t1 - t1) * d1, (next_t2 - t2) * d2
    if not np.all(np.isfinite(r1)) or not np.all(np.isfinite(r2)):
        raise CCSDConvergenceError("Nonfinite CCSD equation residual array")
    residual = max(np.max(np.abs(r1)), np.max(np.abs(r2)))
    if not np.isfinite(energy) or not np.isfinite(residual):
        raise CCSDConvergenceError("Nonfinite CCSD energy or physical residual")
    return hf, float(energy), float(residual)


def validate_ccsd_seed(
    data: IntegralData,
    seed_data: CCSDSeed,
    *,
    host_available_mb: float,
    rss_budget_mb: float = 8192.0,
    allow_large: bool = False,
) -> None:
    """Authenticate arrays/frame and independently check a full-run seed.

    Args:
        data: Same-frame real chemist Hamiltonian including molecular provenance.
        seed_data: Claimed converged RCCSD amplitudes and energies.
        host_available_mb: Executor whole-run cap in decimal MB.
        rss_budget_mb: User whole-run cap in decimal MB.
        allow_large: Override only the resource soft boundary.

    Raises:
        ProvenanceMismatchError: Hash or independently contracted energy differs.
        CCSDConvergenceError: Actual or declared residual violates the contract.
        ResourceLimitError: Guard refuses the operation before workspaces allocate.
    """
    _guard(data, host_available_mb, rss_budget_mb, allow_large)
    _authenticate(data, seed_data)
    _check_physics(data, seed_data)
    hf, _ = _hf_fock(data)
    if not np.isfinite(data.hf_energy) or abs(data.hf_energy - hf) > 1e-10:
        raise ProvenanceMismatchError("Independent HF energy disagrees with IntegralData")


def _check_physics(data: IntegralData | _ActiveHamiltonian, seed_data: CCSDSeed) -> None:
    hf, energy, residual = _physical_values(data, seed_data.t1, seed_data.t2)
    if not np.all(np.isfinite((hf, energy, residual))):
        raise CCSDConvergenceError("Nonfinite CCSD physical acceptance values")
    if abs(seed_data.hf_energy - hf) > 1e-10:
        raise ProvenanceMismatchError("Independent HF energy disagrees with declaration")
    if abs(energy - seed_data.ccsd_energy) > 1e-10:
        raise ProvenanceMismatchError("Independent CCSD energy disagrees with declaration")
    if residual > 1e-7 or abs(residual - seed_data.residual_max_abs_ha) > 1e-10:
        raise CCSDConvergenceError(
            "Actual CCSD equation residual (Ha) disagrees or exceeds tolerance"
        )


def validate_ccsd_seed_from_integrals(
    h1: np.ndarray,
    h2: np.ndarray,
    e_core: float,
    *,
    norb: int,
    nelec: tuple[int, int],
    context: IntegralContext,
    seed_data: CCSDSeed,
    host_available_mb: float,
    rss_budget_mb: float = 8192.0,
    allow_large: bool = False,
) -> None:
    """Receive the low-level integral contract without requiring molecular MOs.

    Actual Hamiltonian bytes and seed equations are verified. Frame provenance
    is the equality of the supplied context and seed identifiers; only the
    molecular wrapper can additionally reconstruct the frame digest from MOs.

    Args:
        h1: Real one-electron integrals in the declared frame.
        h2: Real chemist two-electron tensor in the declared frame.
        e_core: Same-Hamiltonian core constant in Hartree.
        norb: Active spatial orbital count.
        nelec: Balanced spin populations.
        context: Supplied frame and Hamiltonian provenance; source may be empty.
        seed_data: Converged same-frame RCCSD seed.
        host_available_mb: Executor whole-run cap in decimal MB.
        rss_budget_mb: User whole-run cap in decimal MB.
        allow_large: Override only the resource soft boundary.

    Raises:
        ProvenanceMismatchError: Hash, frame identifier or energy mismatch.
        CCSDConvergenceError: Actual/declarative convergence or residual failure.
        ResourceLimitError: Inventory guard refuses allocation.
    """
    data = _ActiveHamiltonian(h1, h2, e_core, norb, nelec, context)
    _guard_active(data, norb, host_available_mb, rss_budget_mb, allow_large)
    _authenticate_active(data, seed_data)
    _check_physics(data, seed_data)


def build_ccsd_seed(
    data: IntegralData,
    *,
    host_available_mb: float,
    rss_budget_mb: float = 8192.0,
    allow_large: bool = False,
    max_cycle: int = 200,
) -> CCSDSeed:
    """Solve RCCSD in the supplied fixed frame without rerunning SCF.

    Args:
        data: Immutable Hamiltonian and complete molecular-frame source.
        host_available_mb: Executor whole-run cap in decimal MB.
        rss_budget_mb: User whole-run cap in decimal MB.
        allow_large: Override only the resource soft boundary.
        max_cycle: Iteration bound, 0..200; zero exercises convergence refusal.

    Returns:
        Converged read-only amplitudes with independently verified Ha residual.

    Raises:
        CCSDConvergenceError: Iterations fail or the physical residual is unacceptable.
        ProvenanceMismatchError: Supplied Hamiltonian/frame/energy is inconsistent.
        ResourceLimitError: Inventory guard refuses allocation.
    """
    if (
        isinstance(max_cycle, bool)
        or not isinstance(max_cycle, Integral)
        or not 0 <= max_cycle <= 200
    ):
        raise ValueError("max_cycle must be an integer in [0, 200]")
    _guard(data, host_available_mb, rss_budget_mb, allow_large)
    _authenticate(data)
    o, v = data.nelec[0], data.norb - data.nelec[0]
    if v:
        solver, eris, _, _ = _solver(data)
        solver.max_cycle = int(max_cycle)
        _, t1, t2 = solver.kernel(eris=eris)
        if not solver.converged:
            raise CCSDConvergenceError("Same-frame RCCSD did not converge")
    else:
        t1, t2 = np.empty((o, 0)), np.empty((o, o, 0, 0))
    hf, energy, residual = _physical_values(data, t1, t2)
    result = CCSDSeed(
        t1,
        t2,
        hf,
        energy,
        True,
        residual,
        data.context.frame_id,
        data.context.hamiltonian_id,
        "pyscf.RCCSD.fixed_frame",
        version("pyscf"),
    )
    validate_ccsd_seed(
        data,
        result,
        host_available_mb=host_available_mb,
        rss_budget_mb=rss_budget_mb,
        allow_large=allow_large,
    )
    return result


def build_lucj(
    data: IntegralData,
    seed_data: CCSDSeed,
    *,
    lucj: LUCJConfig | None = None,
    host_available_mb: float,
    rss_budget_mb: float = 8192.0,
    allow_large: bool = False,
) -> Any:
    """Build the real ffsim spin-balanced operator from an authenticated seed.

    Args:
        data: Same Hamiltonian/frame as the seed.
        seed_data: Converged RCCSD seed.
        lucj: Repetitions and actual same-/opposite-spin interaction restrictions.
        host_available_mb: Executor whole-run cap in decimal MB.
        rss_budget_mb: User whole-run cap in decimal MB.
        allow_large: Override only the resource soft boundary.

    Returns:
        ffsim UCJOpSpinBalanced with requested repetitions and masked interactions.
        Initialization uses deterministic factorization with optimize=False; no RNG.

    Raises:
        ResourceModelDomainError: Unsupported environment, profile or no virtuals.
        CCSDConvergenceError: Seed fails physical acceptance; no fallback occurs.
    """
    lucj = lucj or LUCJConfig()
    _validate_pairs(lucj, data.norb)
    load_resource_model(profile={"n_reps": lucj.n_reps, "shots": lucj.shots})
    validate_ccsd_seed(
        data,
        seed_data,
        host_available_mb=host_available_mb,
        rss_budget_mb=rss_budget_mb,
        allow_large=allow_large,
    )
    _guard(data, host_available_mb, rss_budget_mb, allow_large)
    return _ucj_from_validated_seed(seed_data, lucj)


def build_lucj_from_integrals(
    h1: np.ndarray,
    h2: np.ndarray,
    e_core: float,
    *,
    norb: int,
    nelec: tuple[int, int],
    context: IntegralContext,
    seed_data: CCSDSeed,
    lucj: LUCJConfig | None = None,
    host_available_mb: float,
    rss_budget_mb: float = 8192.0,
    allow_large: bool = False,
) -> Any:
    """Build LUCJ from the raw integral contract after physical seed acceptance.

    Args:
        h1: Real one-electron integrals in the declared frame.
        h2: Real chemist two-electron integrals in the declared frame.
        e_core: Same-Hamiltonian constant in Hartree.
        norb: Active spatial orbital count.
        nelec: Balanced spin populations.
        context: Hamiltonian/frame provenance, without requiring molecular MOs.
        seed_data: Claimed converged RCCSD seed.
        lucj: Repetitions and actual interaction restrictions.
        host_available_mb: Executor whole-run cap in decimal MB.
        rss_budget_mb: User whole-run cap in decimal MB.
        allow_large: Override only the resource soft boundary.

    Returns:
        Deterministically initialized ffsim UCJOpSpinBalanced.

    Raises:
        ProvenanceMismatchError: Seed fails hash/frame/energy authentication.
        CCSDConvergenceError: Seed fails physical residual acceptance.
        ResourceLimitError: Guard rejects the environment/profile or allocation.
    """
    lucj = lucj or LUCJConfig()
    _validate_pairs(lucj, norb)
    load_resource_model(profile={"n_reps": lucj.n_reps, "shots": lucj.shots})
    validate_ccsd_seed_from_integrals(
        h1,
        h2,
        e_core,
        norb=norb,
        nelec=nelec,
        context=context,
        seed_data=seed_data,
        host_available_mb=host_available_mb,
        rss_budget_mb=rss_budget_mb,
        allow_large=allow_large,
    )
    data = _ActiveHamiltonian(h1, h2, e_core, norb, nelec, context)
    _guard_active(data, norb, host_available_mb, rss_budget_mb, allow_large)
    return _ucj_from_validated_seed(seed_data, lucj)


def _ucj_from_validated_seed(seed_data: CCSDSeed, lucj: LUCJConfig) -> Any:
    if not seed_data.t1.shape[1]:
        raise ResourceModelDomainError("ffsim amplitude decomposition requires virtual orbitals")
    import ffsim

    pairs = (
        None
        if lucj.interaction_pairs is None
        else tuple(None if channel is None else list(channel) for channel in lucj.interaction_pairs)
    )
    return ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        seed_data.t2, t1=seed_data.t1, n_reps=lucj.n_reps, interaction_pairs=pairs, optimize=False
    )


__all__ = [
    "build_ccsd_seed",
    "validate_ccsd_seed",
    "validate_ccsd_seed_from_integrals",
    "build_lucj",
    "build_lucj_from_integrals",
]
