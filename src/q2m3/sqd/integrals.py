"""Real chemist integrals in a single, immutable vacuum RHF orbital frame.

Geometry work has a separate bounded AO allocation inventory, not the calibrated
synthetic active-space model. Callers must enforce an independent process-tree
RSS cap and wall timeout; the inventory is a conservative planning estimate, not
a certified bound on arbitrary native-library allocations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from q2m3.interfaces.fixed_mo_embedding import (
    _canonicalize_mo_signs,
    build_fixed_mo_embedding_integrals,
    resolve_active_space,
)
from q2m3.molecule import MoleculeConfig
from q2m3.sqd.config import (
    IntegralContext,
    validate_active_space,
    validate_integral_inputs,
    validate_molecular_inputs,
)
from q2m3.sqd.exceptions import ResourceModelDomainError
from q2m3.sqd.resources import Stage, guard_allocation


def _snapshot(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError("Integral and orbital arrays must be finite and real")
    canonical = np.asarray(array, dtype="<f8", order="C")
    return np.frombuffer(canonical.tobytes(), dtype="<f8").reshape(array.shape)


def _update_array(digest: Any, name: str, value: np.ndarray | float) -> None:
    array = _snapshot(value)
    header = json.dumps([name, list(array.shape), "<f8"], separators=(",", ":"))
    digest.update(header.encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))


def hamiltonian_id(
    h1: np.ndarray, h2: np.ndarray, e_core: float, *, norb: int, nelec: tuple[int, int]
) -> str:
    """Hash actual integrals using the ``q2m3.hamiltonian.v1`` byte format.

    The digest starts with the version and NUL, then compact ASCII JSON
    ``[norb,[nalpha,nbeta]]`` and NUL. Each h1, h2, e_core field has a compact
    ``[name,shape,"<f8"]`` JSON header and NUL, followed by little-endian float64
    C-order bytes; e_core is a scalar with shape []. No labels enter the digest.

    Args:
        h1: Real one-electron matrix in Hartree.
        h2: Real chemist two-electron tensor in Hartree.
        e_core: Core plus nuclear constant in Hartree.
        norb: Active spatial orbital count.
        nelec: Balanced spin populations.

    Returns:
        Lowercase SHA256 hex digest, invariant to storage order and endianness.

    Raises:
        ValueError: Shape, spin populations, or real/finite representation is invalid.
    """
    validate_active_space(norb, nelec)
    if np.shape(h1) != (norb, norb) or np.shape(h2) != (norb,) * 4 or np.ndim(e_core):
        raise ValueError("Hamiltonian shapes do not match norb")
    digest = hashlib.sha256(b"q2m3.hamiltonian.v1\0")
    digest.update(
        json.dumps([int(norb), [int(n) for n in nelec]], separators=(",", ":")).encode() + b"\0"
    )
    for name, value in (("h1", h1), ("h2", h2), ("e_core", e_core)):
        _update_array(digest, name, value)
    return digest.hexdigest()


@dataclass(frozen=True)
class IntegralData:
    """Immutable chemist Hamiltonian, complete MO snapshot and its provenance.

    ``hf_energy`` is the expectation of the first ``nelec[0]`` active orbitals,
    including e_core. It is not a relaxed-MM SCF minimum. No CCSD amplitudes are
    constructed here; downstream seed solvers must consume this same Hamiltonian.
    """

    h1: np.ndarray
    h2: np.ndarray
    e_core: float
    norb: int
    nelec: tuple[int, int]
    mo_coeff: np.ndarray
    hf_energy: float
    context: IntegralContext

    def __post_init__(self) -> None:
        for name in ("h1", "h2", "mo_coeff"):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))


@dataclass(frozen=True)
class _MolecularIntegralInventory:
    """AO-sized planning inventory; external RSS/time enforcement is mandatory.

    Domain: <=32 AO/MOs, <=32 atoms, <=256 MM charges, serial real RHF using
    default DIIS space/cycle settings. The 1024 decimal-MB reserve covers imports
    and native overhead empirically checked only by supervised validation runs.
    Array allowance: 32 dense AO^4 arrays (ERI transforms/copies/scratch), 64 AO^2
    arrays (SCF/DIIS/MO), and four n_mm*AO^2 charge blocks, all float64. This is
    deliberately separate from the synthetic active-space calibration.
    """

    n_ao: int
    n_atoms: int
    n_mm: int
    model_id: str = "molecular-ao-inventory-v1"

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
        if (
            stage != "integrals"
            or not 1 <= self.n_ao <= 32
            or not 1 <= self.n_atoms <= 32
            or not 0 <= self.n_mm <= 256
            or norb > self.n_ao
        ):
            raise ResourceModelDomainError("Molecular AO inventory domain exceeded")
        return (
            1024.0
            + retained_mb
            + 8 * (32 * self.n_ao**4 + 64 * self.n_ao**2 + 4 * self.n_mm * self.n_ao**2) / 1e6
        )


def _frame_id(
    molecule: MoleculeConfig,
    mo: np.ndarray,
    active: tuple[int, ...],
    n_core: int,
) -> str:
    # The molecular name and MM charges are deliberately absent: zero MM and
    # vacuum have the same frame. Full signed columns distinguish orbital gauges.
    header = [
        molecule.symbols,
        int(molecule.charge),
        molecule.basis,
        [int(i) for i in active],
        int(n_core),
    ]
    digest = hashlib.sha256(b"q2m3.orbital-frame.v1\0")
    digest.update(json.dumps(header, separators=(",", ":")).encode() + b"\0")
    _update_array(digest, "coordinates_angstrom", molecule.coords_array)
    _update_array(digest, "mo_coeff", mo)
    return digest.hexdigest()


def _vacuum_integrals(
    mol: Any,
    active: tuple[int, ...],
    n_core: int,
    nelec: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    from pyscf import ao2mo, mcscf, scf

    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.run()
    if not mf.converged:
        raise RuntimeError("vacuum RHF calculation did not converge")
    mo = _canonicalize_mo_signs(mf.mo_coeff)
    cas = mcscf.CASCI(mf, len(active), nelec)
    cas.ncore = n_core
    h1, core = cas.get_h1eff(mo_coeff=mo)
    h2 = ao2mo.restore(1, cas.get_h2eff(mo_coeff=mo), len(active))
    return h1, h2, float(core), 0.0, mo


def _embedded_integrals(
    molecule: MoleculeConfig,
    charges: np.ndarray,
    coords: np.ndarray,
    mode: str,
    active: tuple[int, ...],
    n_core: int,
    n_ao: int,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    result = build_fixed_mo_embedding_integrals(
        molecule.symbols,
        molecule.coords_array,
        mm_charges=charges,
        mm_coords=coords,
        charge=molecule.charge,
        basis=molecule.basis,
        active_electrons=molecule.active_electrons,
        active_orbitals=molecule.active_orbitals,
    )
    if result.mo_coeff is None:
        raise ValueError("Fixed-MO helper did not supply mo_coeff")
    if result.vacuum_core_constant is None:
        raise ValueError("Fixed-MO helper did not supply vacuum_core_constant")
    if result.mo_coeff.shape != (n_ao, n_ao):
        raise ValueError("Fixed-MO helper must supply the complete mo_coeff matrix")
    if result.active_indices != active or result.n_core_orbitals != n_core:
        raise ValueError("Fixed-MO helper active space does not match the requested frame")
    delta = result.delta_h_diag if mode == "diagonal" else result.delta_h_active
    return (
        result.one_electron_vacuum + delta,
        result.two_electron.transpose(0, 3, 1, 2),
        result.vacuum_core_constant,
        result.delta_core_constant,
        result.mo_coeff,
    )


def build_integrals(
    molecule: MoleculeConfig,
    *,
    mm_charges: np.ndarray | None = None,
    mm_coords: np.ndarray | None = None,
    embedding_mode: Literal["diagonal", "full_oneelectron"] = "diagonal",
    host_available_mb: float,
    rss_budget_mb: float = 8192.0,
    allow_large: bool = False,
) -> IntegralData:
    """Build a guarded molecular Hamiltonian in its one vacuum RHF frame.

    Args:
        molecule: Molecular geometry in Angstrom and explicit closed-shell space.
        mm_charges: Optional nonempty point-charge array in elementary charges.
        mm_coords: Corresponding coordinates in Angstrom.
        embedding_mode: Diagonal or full active-space one-electron perturbation.
        host_available_mb: Executor's conservative whole-run memory cap in decimal MB.
        rss_budget_mb: User's whole-run memory cap in decimal MB.
        allow_large: Override only the standard soft resource rejection boundary.

    Returns:
        Read-only integrals/MOs and same-frame HF expectation with provenance.

    Raises:
        ValueError: Malformed inputs, missing helper frame, or invalid integrals.
        RuntimeError: RHF did not converge.
        ResourceLimitError: The pre-allocation estimate reaches the run cap.
        ResourceModelDomainError: Real-AO work is outside the bounded inventory.

    An external process-tree RSS monitor and wall timeout are required. Molecular
    basis metadata is built first to count AOs; the inventory guard then runs
    before any RHF or integral transformation. Its estimate is recorded in source
    and does not certify the later sampling/reference stages or arbitrary hosts.
    """
    from pyscf import __version__ as pyscf_version
    from pyscf import gto

    validate_molecular_inputs(
        molecule, mm_charges=mm_charges, mm_coords=mm_coords, embedding_mode=embedding_mode
    )
    n_mm = 0 if mm_charges is None else len(mm_charges)
    if mm_charges is not None and not n_mm:
        raise ValueError("MM inputs must contain at least one point charge")
    if len(molecule.symbols) > 32 or n_mm > 256:
        raise ResourceModelDomainError("Molecular atom/point-charge inventory domain exceeded")
    if not isinstance(molecule.basis, str) or not molecule.basis.strip():
        raise ValueError("basis must be a nonempty PySCF basis name")
    mol = gto.M(
        atom=list(zip(molecule.symbols, molecule.coords, strict=True)),
        basis=molecule.basis,
        charge=molecule.charge,
        unit="Angstrom",
        verbose=0,
    )
    active, n_core = resolve_active_space(
        n_electrons=mol.nelectron,
        n_orbitals=mol.nao_nr(),
        active_electrons=molecule.active_electrons,
        active_orbitals=molecule.active_orbitals,
    )
    norb, nelec = len(active), (molecule.active_electrons // 2,) * 2
    inventory = _MolecularIntegralInventory(mol.nao_nr(), mol.natm, n_mm)
    estimate = guard_allocation(
        norb,
        nelec,
        stage="integrals",
        model=inventory,
        host_available_mb=host_available_mb,
        rss_budget_mb=rss_budget_mb,
        allow_large=allow_large,
    )
    if mm_charges is None:
        h1, h2, vacuum_core, delta_core, mo = _vacuum_integrals(mol, active, n_core, nelec)
        mode, kind = "vacuum", "canonical_rhf"
    else:
        h1, h2, vacuum_core, delta_core, mo = _embedded_integrals(
            molecule, mm_charges, mm_coords, embedding_mode, active, n_core, mol.nao_nr()
        )
        mode, kind = embedding_mode, "fixed_frame_determinant"
    e_core = float(vacuum_core + delta_core)
    context = IntegralContext(
        frame_id=_frame_id(molecule, mo, active, n_core),
        hamiltonian_id=hamiltonian_id(h1, h2, e_core, norb=norb, nelec=nelec),
        active_indices=active,
        n_core_orbitals=n_core,
        vacuum_core_constant=float(vacuum_core),
        delta_core_constant=float(delta_core),
        embedding_mode=mode,
        fixed_mo=True,
        two_electron_tensor_fixed=True,
        hf_reference_kind=kind,
        source={
            "producer": "build_integrals",
            "frame_version": "q2m3.orbital-frame.v1",
            "hamiltonian_version": "q2m3.hamiltonian.v1",
            "basis": molecule.basis,
            "symbols": molecule.symbols,
            "coordinates_angstrom": molecule.coords,
            "charge": molecule.charge,
            "mm_charges": mm_charges,
            "mm_coords_angstrom": mm_coords,
            "pyscf_version": pyscf_version,
            "frame_source": "vacuum_rhf",
            "tensor_convention": "chemist",
            "resource_model": inventory.model_id,
            "estimated_rss_mb": estimate,
            "n_ao": mol.nao_nr(),
            "n_mm": n_mm,
            "resource_bound_kind": "allocation_inventory_with_unvalidated_native_reserve",
            "external_rss_and_timeout_required": True,
        },
    )
    validate_integral_inputs(
        h1,
        h2,
        e_core,
        norb=norb,
        nelec=nelec,
        context=context,
        seed_data=None,
        mode="reference_only",
    )
    occupied = range(nelec[0])
    hf_energy = e_core + 2 * sum(h1[i, i] for i in occupied)
    hf_energy += sum(2 * h2[i, i, j, j] - h2[i, j, j, i] for i in occupied for j in occupied)
    return IntegralData(h1, h2, e_core, norb, nelec, mo, float(hf_energy), context)


__all__ = ["IntegralData", "build_integrals", "hamiltonian_id"]
