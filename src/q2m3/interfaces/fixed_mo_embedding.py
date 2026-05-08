# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Fixed-MO one-electron embedding integrals for explicit-MM QM/MM workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q2m3.constants import ANGSTROM_TO_BOHR


@dataclass(frozen=True)
class FixedMOEmbeddingDiagnostics:
    """Scalar diagnostics for a fixed-MO one-electron MM perturbation.

    Args:
        delta_h_diag_fro: Frobenius norm of the diagonal active-space perturbation.
        delta_h_offdiag_fro: Frobenius norm of the off-diagonal active-space perturbation.
        delta_h_offdiag_to_diag: Ratio of off-diagonal to diagonal Frobenius norms.
        delta_h_hermitian_max_abs: Maximum absolute anti-Hermitian residual.
        delta_h_trace_ha: Trace of the active-space one-electron perturbation in Hartree.
        delta_nuclear_mm_ha: MM-induced nuclear constant difference in Hartree.
        delta_core_constant_ha: Frozen-core one-electron plus nuclear MM constant in Hartree.
        n_mm_charges: Number of MM point charges.
        fixed_mo: Whether the perturbation is represented in the vacuum MO frame.
        two_electron_tensor_fixed: Whether the two-electron tensor is held at vacuum values.
    """

    delta_h_diag_fro: float
    delta_h_offdiag_fro: float
    delta_h_offdiag_to_diag: float | None
    delta_h_hermitian_max_abs: float
    delta_h_trace_ha: float
    delta_nuclear_mm_ha: float
    delta_core_constant_ha: float
    n_mm_charges: int
    fixed_mo: bool = True
    two_electron_tensor_fixed: bool = True


@dataclass(frozen=True)
class FixedMOEmbeddingResult:
    """Fixed-MO one-electron embedding tensors and active-space diagnostics.

    Args:
        one_electron_vacuum: Vacuum active-space one-electron tensor in Hartree.
        two_electron: Vacuum active-space two-electron tensor in PennyLane convention.
        delta_h_active: Full active-space MM one-electron perturbation in Hartree.
        delta_h_diag: Diagonal part of ``delta_h_active``.
        delta_h_offdiag: Off-diagonal part of ``delta_h_active``.
        delta_nuclear_mm: MM-induced nuclear constant difference in Hartree.
        delta_core_constant: Frozen-core one-electron plus nuclear MM constant in Hartree.
        active_indices: Vacuum MO indices included in the active space.
        n_core_orbitals: Number of frozen doubly occupied core orbitals.
        diagnostics: Scalar diagnostics for logging and public API metadata.
    """

    one_electron_vacuum: np.ndarray
    two_electron: np.ndarray
    delta_h_active: np.ndarray
    delta_h_diag: np.ndarray
    delta_h_offdiag: np.ndarray
    delta_nuclear_mm: float
    delta_core_constant: float
    active_indices: tuple[int, ...]
    n_core_orbitals: int
    diagnostics: FixedMOEmbeddingDiagnostics


def build_fixed_mo_embedding_integrals(
    symbols: list[str],
    coords: np.ndarray,
    *,
    mm_charges: np.ndarray,
    mm_coords: np.ndarray,
    charge: int = 0,
    basis: str = "sto-3g",
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
) -> FixedMOEmbeddingResult:
    """Build fixed-MO one-electron embedding tensors for MM point charges.

    The helper runs vacuum RHF once, keeps its canonical MO frame fixed, and
    computes ``Delta h_AO = h_core_MM - h_core_vac`` from a PySCF
    ``qmmm.mm_charge`` mean-field object. The returned two-electron tensor is
    the vacuum active-space tensor; no orbital relaxation or MM-updated
    two-electron tensor is included.

    Args:
        symbols: QM atomic symbols.
        coords: QM coordinates in Angstrom, shape ``(n_atoms, 3)`` or flattened.
        mm_charges: MM point charges in elementary charge units, shape ``(n_mm,)``.
        mm_coords: MM point-charge coordinates in Angstrom, shape ``(n_mm, 3)``.
        charge: Total QM molecular charge.
        basis: PySCF basis set name.
        active_electrons: Number of active electrons. Must be paired with
            ``active_orbitals`` when provided.
        active_orbitals: Number of active spatial orbitals. Must be paired with
            ``active_electrons`` when provided.

    Returns:
        Fixed-MO embedding tensors and diagnostics.

    Raises:
        ValueError: If coordinates, MM arrays, or active-space parameters are invalid.
        RuntimeError: If the vacuum or MM-embedded RHF calculation does not converge.
    """
    from pyscf import ao2mo, gto, qmmm, scf

    qm_coords = _validate_qm_coords(symbols, coords)
    mm_charges_arr, mm_coords_arr = _validate_mm_inputs(mm_charges, mm_coords)

    mol = gto.M(
        atom=[(symbol, tuple(coord)) for symbol, coord in zip(symbols, qm_coords, strict=True)],
        basis=basis,
        charge=charge,
        unit="Angstrom",
    )

    mf_vac = scf.RHF(mol)
    mf_vac.verbose = 0
    mf_vac.run()
    _ensure_converged(mf_vac, "vacuum")

    mf_mm = scf.RHF(mol)
    mf_mm.verbose = 0
    mf_mm = qmmm.mm_charge(mf_mm, mm_coords_arr * ANGSTROM_TO_BOHR, mm_charges_arr)
    mf_mm.run()
    _ensure_converged(mf_mm, "MM-embedded")

    mo_coeff = _canonicalize_mo_signs(np.asarray(mf_vac.mo_coeff, dtype=float))
    n_orbitals = mo_coeff.shape[1]
    active_indices, n_core = _resolve_active_space(
        n_electrons=mol.nelectron,
        n_orbitals=n_orbitals,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )
    core_indices = tuple(range(n_core))

    h_vac_ao = np.asarray(mf_vac.get_hcore(), dtype=float)
    h_mm_ao = np.asarray(mf_mm.get_hcore(), dtype=float)
    h_vac_mo = mo_coeff.T @ h_vac_ao @ mo_coeff
    delta_h_mo = mo_coeff.T @ (h_mm_ao - h_vac_ao) @ mo_coeff

    two_electron_chemist = ao2mo.kernel(mol, mo_coeff, compact=False).reshape(
        n_orbitals, n_orbitals, n_orbitals, n_orbitals
    )

    one_electron_vacuum = _active_one_electron_with_core(
        h_vac_mo,
        two_electron_chemist,
        active_indices,
        core_indices,
    )
    two_electron = two_electron_chemist[
        np.ix_(active_indices, active_indices, active_indices, active_indices)
    ].transpose(0, 2, 3, 1)

    delta_h_active = delta_h_mo[np.ix_(active_indices, active_indices)]
    delta_h_diag = np.diag(np.diag(delta_h_active))
    delta_h_offdiag = delta_h_active - delta_h_diag
    delta_nuclear_mm = float(mf_mm.energy_nuc() - mol.energy_nuc())
    delta_core_constant = _delta_core_constant(delta_h_mo, core_indices, delta_nuclear_mm)

    diagnostics = _build_diagnostics(
        delta_h_active=delta_h_active,
        delta_h_diag=delta_h_diag,
        delta_h_offdiag=delta_h_offdiag,
        delta_nuclear_mm=delta_nuclear_mm,
        delta_core_constant=delta_core_constant,
        n_mm_charges=mm_charges_arr.size,
    )

    return FixedMOEmbeddingResult(
        one_electron_vacuum=np.asarray(one_electron_vacuum, dtype=float),
        two_electron=np.asarray(two_electron, dtype=float),
        delta_h_active=np.asarray(delta_h_active, dtype=float),
        delta_h_diag=np.asarray(delta_h_diag, dtype=float),
        delta_h_offdiag=np.asarray(delta_h_offdiag, dtype=float),
        delta_nuclear_mm=delta_nuclear_mm,
        delta_core_constant=delta_core_constant,
        active_indices=active_indices,
        n_core_orbitals=n_core,
        diagnostics=diagnostics,
    )


def _validate_qm_coords(symbols: list[str], coords: np.ndarray) -> np.ndarray:
    if not symbols:
        raise ValueError("symbols must contain at least one atom")

    coords_arr = np.asarray(coords, dtype=float)
    if coords_arr.ndim == 1:
        if coords_arr.size != len(symbols) * 3:
            raise ValueError(
                f"Flattened coords length ({coords_arr.size}) must be symbols * 3 "
                f"= {len(symbols) * 3}"
            )
        coords_arr = coords_arr.reshape(-1, 3)
    elif coords_arr.ndim == 2:
        if coords_arr.shape != (len(symbols), 3):
            raise ValueError(f"coords must have shape ({len(symbols)}, 3), got {coords_arr.shape}")
    else:
        raise ValueError(f"coords must be 1D or 2D array, got {coords_arr.ndim}D")

    if not np.all(np.isfinite(coords_arr)):
        raise ValueError("coords must contain only finite values")
    return coords_arr


def _validate_mm_inputs(
    mm_charges: np.ndarray,
    mm_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    charges_arr = np.asarray(mm_charges, dtype=float)
    coords_arr = np.asarray(mm_coords, dtype=float)

    if charges_arr.ndim != 1:
        raise ValueError(f"mm_charges must be a 1D array, got {charges_arr.ndim}D")
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError(f"mm_coords must have shape (n_mm, 3), got {coords_arr.shape}")
    if charges_arr.size == 0:
        raise ValueError("fixed-MO embedding requires at least one MM charge")
    if coords_arr.shape[0] != charges_arr.size:
        raise ValueError(
            f"mm_coords rows ({coords_arr.shape[0]}) must match "
            f"mm_charges length ({charges_arr.size})"
        )
    if not np.all(np.isfinite(charges_arr)) or not np.all(np.isfinite(coords_arr)):
        raise ValueError("mm_charges and mm_coords must contain only finite values")

    return charges_arr, coords_arr


def _ensure_converged(mean_field, label: str) -> None:
    if not bool(getattr(mean_field, "converged", False)):
        raise RuntimeError(f"{label} RHF calculation did not converge")


def _canonicalize_mo_signs(mo_coeff: np.ndarray) -> np.ndarray:
    canonical = mo_coeff.copy()
    for col in range(canonical.shape[1]):
        pivot = int(np.argmax(np.abs(canonical[:, col])))
        if canonical[pivot, col] < 0:
            canonical[:, col] *= -1.0
    return canonical


def _resolve_active_space(
    *,
    n_electrons: int,
    n_orbitals: int,
    active_electrons: int | None,
    active_orbitals: int | None,
) -> tuple[tuple[int, ...], int]:
    if (active_electrons is None) != (active_orbitals is None):
        raise ValueError("active_electrons and active_orbitals must be provided together")

    if active_electrons is None:
        active_electrons = n_electrons
        active_orbitals = n_orbitals

    if active_electrons <= 0:
        raise ValueError(f"active_electrons must be positive, got {active_electrons}")
    if active_orbitals is None or active_orbitals <= 0:
        raise ValueError(f"active_orbitals must be positive, got {active_orbitals}")
    if active_electrons > n_electrons:
        raise ValueError(
            f"active_electrons ({active_electrons}) cannot exceed n_electrons ({n_electrons})"
        )
    if (n_electrons - active_electrons) % 2 != 0:
        raise ValueError(
            f"Active space requires (n_electrons - active_electrons) to be even; "
            f"got n_electrons={n_electrons}, active_electrons={active_electrons}"
        )

    n_core = (n_electrons - active_electrons) // 2
    if active_orbitals * 2 < active_electrons:
        raise ValueError(
            f"active_orbitals ({active_orbitals}) cannot hold active_electrons "
            f"({active_electrons})"
        )
    if n_core + active_orbitals > n_orbitals:
        raise ValueError(
            f"Requested active space needs orbitals through index {n_core + active_orbitals - 1}, "
            f"but only {n_orbitals} molecular orbitals are available"
        )

    active_indices = tuple(range(n_core, n_core + active_orbitals))
    return active_indices, n_core


def _active_one_electron_with_core(
    h_mo: np.ndarray,
    two_electron_chemist: np.ndarray,
    active_indices: tuple[int, ...],
    core_indices: tuple[int, ...],
) -> np.ndarray:
    active = np.array(active_indices, dtype=int)
    one_electron = h_mo[np.ix_(active, active)].copy()
    for core in core_indices:
        for row, p in enumerate(active_indices):
            for col, q in enumerate(active_indices):
                one_electron[row, col] += (
                    2.0 * two_electron_chemist[p, q, core, core]
                    - two_electron_chemist[p, core, core, q]
                )
    return one_electron


def _delta_core_constant(
    delta_h_mo: np.ndarray,
    core_indices: tuple[int, ...],
    delta_nuclear_mm: float,
) -> float:
    if not core_indices:
        return delta_nuclear_mm
    core = np.array(core_indices, dtype=int)
    delta_core_block = delta_h_mo[np.ix_(core, core)]
    return float(2.0 * np.trace(delta_core_block) + delta_nuclear_mm)


def _build_diagnostics(
    *,
    delta_h_active: np.ndarray,
    delta_h_diag: np.ndarray,
    delta_h_offdiag: np.ndarray,
    delta_nuclear_mm: float,
    delta_core_constant: float,
    n_mm_charges: int,
) -> FixedMOEmbeddingDiagnostics:
    diag_fro = float(np.linalg.norm(delta_h_diag, ord="fro"))
    offdiag_fro = float(np.linalg.norm(delta_h_offdiag, ord="fro"))
    offdiag_to_diag = offdiag_fro / diag_fro if diag_fro > 0.0 else None
    hermitian_residual = delta_h_active - delta_h_active.T.conj()

    return FixedMOEmbeddingDiagnostics(
        delta_h_diag_fro=diag_fro,
        delta_h_offdiag_fro=offdiag_fro,
        delta_h_offdiag_to_diag=offdiag_to_diag,
        delta_h_hermitian_max_abs=float(np.max(np.abs(hermitian_residual))),
        delta_h_trace_ha=float(np.trace(delta_h_active)),
        delta_nuclear_mm_ha=delta_nuclear_mm,
        delta_core_constant_ha=delta_core_constant,
        n_mm_charges=n_mm_charges,
    )


__all__ = [
    "FixedMOEmbeddingDiagnostics",
    "FixedMOEmbeddingResult",
    "build_fixed_mo_embedding_integrals",
]
