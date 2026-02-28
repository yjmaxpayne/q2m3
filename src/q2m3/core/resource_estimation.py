# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
EFTQC resource estimation module.

Provides structured dataclass results for Early Fault-Tolerant Quantum Computer
(EFTQC) resource estimation, wrapping PySCFPennyLaneConverter.estimate_qpe_resources().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q2m3.interfaces import PySCFPennyLaneConverter


@dataclass(frozen=True)
class EFTQCResources:
    """Resource estimation result for a single EFTQC configuration.

    Attributes:
        hamiltonian_1norm: Lambda (1-norm) of the Hamiltonian in Hartree.
        logical_qubits: Number of logical qubits required.
        toffoli_gates: Non-Clifford (Toffoli) gate count.
        n_terms: Number of Hamiltonian terms. None when DoubleFactorization
            resource estimation is used (it does not construct the full
            PennyLane Hamiltonian object).
        target_error: Target energy error in Hartree.
        n_system_qubits: System register size = n_orbitals * 2 (Jordan-Wigner).
        basis: Basis set used (e.g. "sto-3g").
        n_mm_charges: Number of MM point charges; 0 for vacuum.
    """

    hamiltonian_1norm: float
    logical_qubits: int
    toffoli_gates: int
    n_terms: int | None
    target_error: float
    n_system_qubits: int
    basis: str
    n_mm_charges: int


@dataclass(frozen=True)
class ResourceComparisonResult:
    """Comparison of EFTQC resources between vacuum and solvated systems.

    Attributes:
        vacuum: Resource estimate with no MM embedding.
        solvated: Resource estimate with MM point charge embedding.
        delta_lambda_percent: Percentage change in Hamiltonian 1-norm
            due to solvation: (solvated - vacuum) / vacuum * 100.
        delta_gates_percent: Percentage change in Toffoli gate count
            due to solvation: (solvated - vacuum) / vacuum * 100.
    """

    vacuum: EFTQCResources
    solvated: EFTQCResources
    delta_lambda_percent: float
    delta_gates_percent: float


def estimate_resources(
    symbols: list[str],
    coords: np.ndarray,
    charge: int = 0,
    basis: str = "sto-3g",
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    target_error: float = 0.0016,
    mm_charges: np.ndarray | None = None,
    mm_coords: np.ndarray | None = None,
) -> EFTQCResources:
    """Estimate EFTQC resource requirements for a molecular system.

    Wraps PySCFPennyLaneConverter.estimate_qpe_resources() and returns a
    structured, immutable EFTQCResources dataclass instead of a raw dict.

    Args:
        symbols: List of atomic symbols (e.g. ['H', 'H']).
        coords: Atomic coordinates in Angstrom, shape (n_atoms, 3).
        charge: Total molecular charge (default: 0).
        basis: Basis set name (default: "sto-3g").
        active_electrons: Active space electrons (optional, reserved for future).
        active_orbitals: Active space orbitals (optional, reserved for future).
        target_error: Target energy error in Hartree (default: 0.0016 = 1 kcal/mol).
        mm_charges: MM point charges array (optional, enables solvated Hamiltonian).
        mm_coords: MM charge coordinates in Angstrom, shape (n_mm, 3).

    Returns:
        EFTQCResources dataclass with all resource fields populated.
    """
    converter = PySCFPennyLaneConverter(basis=basis, mapping="jordan_wigner")
    raw = converter.estimate_qpe_resources(
        symbols=symbols,
        coords=coords,
        charge=charge,
        mm_charges=mm_charges,
        mm_coords=mm_coords,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        target_error=target_error,
    )
    return EFTQCResources(
        hamiltonian_1norm=float(raw["hamiltonian_1norm"]),
        logical_qubits=int(raw["logical_qubits"]),
        toffoli_gates=int(raw["toffoli_gates"]),
        n_terms=None,  # Not available from DoubleFactorization resource estimation
        target_error=float(raw["target_error"]),
        n_system_qubits=int(raw["n_orbitals"]) * 2,  # JW: 2 qubits per spatial orbital
        basis=str(raw["basis"]),
        n_mm_charges=int(raw["n_mm_charges"]),
    )


def compare_vacuum_solvated(
    symbols: list[str],
    coords: np.ndarray,
    charge: int = 0,
    basis: str = "sto-3g",
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    mm_charges: np.ndarray | None = None,
    mm_coords: np.ndarray | None = None,
    target_error: float = 0.0016,
) -> ResourceComparisonResult:
    """Compare EFTQC resource requirements between vacuum and solvated systems.

    Calls estimate_resources() twice: once without MM charges (vacuum) and
    once with MM charges (solvated), then computes the percentage deltas.

    Args:
        symbols: List of atomic symbols.
        coords: Atomic coordinates in Angstrom, shape (n_atoms, 3).
        charge: Total molecular charge (default: 0).
        basis: Basis set name (default: "sto-3g").
        active_electrons: Active space electrons (optional).
        active_orbitals: Active space orbitals (optional).
        mm_charges: MM point charges for the solvated estimate.
        mm_coords: MM charge coordinates in Angstrom for the solvated estimate.
        target_error: Target energy error in Hartree (default: 0.0016).

    Returns:
        ResourceComparisonResult with both estimates and derived deltas.
    """
    common = dict(
        symbols=symbols,
        coords=coords,
        charge=charge,
        basis=basis,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        target_error=target_error,
    )
    vacuum = estimate_resources(**common)
    solvated = estimate_resources(**common, mm_charges=mm_charges, mm_coords=mm_coords)

    delta_lambda = (
        (solvated.hamiltonian_1norm - vacuum.hamiltonian_1norm) / vacuum.hamiltonian_1norm * 100
    )
    delta_gates = (solvated.toffoli_gates - vacuum.toffoli_gates) / vacuum.toffoli_gates * 100

    return ResourceComparisonResult(
        vacuum=vacuum,
        solvated=solvated,
        delta_lambda_percent=delta_lambda,
        delta_gates_percent=delta_gates,
    )
