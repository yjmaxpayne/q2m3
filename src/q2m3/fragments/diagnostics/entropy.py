"""Orbital entropies and mutual information of an active space from a block2 DMRG run."""

from __future__ import annotations

import itertools
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from q2m3._lazy import extra_available
from q2m3.fragments.counterpoise import Subsystem
from q2m3.fragments.integrals import ActiveSpaceIntegrals
from q2m3.fragments.molecule import Molecule

CONVERGENCE_THRESHOLD_HA = 1e-3
ASSIGNMENT_RULE = "A if the Lowdin population on the atoms of fragment A is >= 0.5, else B"
SWEEP_NOISE = 1e-4


@dataclass(frozen=True)
class DMRGConvergence:
    """Sweep record of one DMRG run and its convergence verdict.

    A tier is a run of consecutive sweeps with one bond dimension. Its energy is the
    energy after its last sweep. The run is unconverged when the energies of the last two
    tiers differ by more than ``threshold_ha``.

    Raises:
        ValueError: The sweeps use fewer than two bond dimensions, or the two sequences
            differ in length.
    """

    sweep_bond_dims: tuple[int, ...]
    sweep_energies: tuple[float, ...]
    threshold_ha: float = CONVERGENCE_THRESHOLD_HA

    def __post_init__(self) -> None:
        if len(self._tiers()) < 2:
            raise ValueError("the sweep schedule needs at least two bond dimensions")

    def _tiers(self) -> list[tuple[int, float]]:
        sweeps = zip(self.sweep_bond_dims, self.sweep_energies, strict=True)
        return [
            (bond_dim, list(group)[-1][1])
            for bond_dim, group in itertools.groupby(sweeps, key=lambda sweep: sweep[0])
        ]

    @property
    def tier_bond_dims(self) -> tuple[int, ...]:
        return tuple(bond_dim for bond_dim, _ in self._tiers())

    @property
    def tier_energies(self) -> tuple[float, ...]:
        return tuple(energy for _, energy in self._tiers())

    @property
    def last_tier_difference_ha(self) -> float:
        energies = self.tier_energies
        return abs(energies[-1] - energies[-2])

    @property
    def status(self) -> str:
        return "unconverged" if self.last_tier_difference_ha > self.threshold_ha else "converged"


@dataclass(frozen=True)
class EntropyDiagnostics:
    """Orbital entropies, mutual information and fragment assignment of active orbitals.

    Arrays follow the active-orbital order of the integrals. ``fragment_weights`` holds the
    Lowdin population of each orbital on the atoms of fragment A, and
    ``orbital_fragments`` the label that ``assignment_rule`` gives it.
    """

    single_orbital_entropy: np.ndarray
    mutual_information: np.ndarray
    fragment_weights: np.ndarray
    orbital_fragments: tuple[str, ...]
    assignment_rule: str
    convergence: DMRGConvergence


def mutual_information(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Return ``0.5 * (s1_i + s1_j - s2_ij)`` with a zero diagonal.

    Args:
        s1: Single-orbital entropies, shape ``(n,)``.
        s2: Two-orbital entropies, shape ``(n, n)``.
    """
    s1 = np.asarray(s1, dtype=float)
    mi = 0.5 * (s1[:, None] + s1[None, :] - np.asarray(s2, dtype=float))
    np.fill_diagonal(mi, 0.0)
    return mi


def assign_orbitals(
    molecule: Molecule, subsystem: Subsystem, orbitals: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Assign each orbital to a fragment by its Lowdin population on fragment A.

    Args:
        molecule: Dimer whose first fragment is fragment A.
        subsystem: Counterpoise subsystem that defines the AO basis.
        orbitals: MO coefficients, one orbital per column.

    Returns:
        The population of each orbital on the atoms of fragment A, and its label.
    """
    mol = subsystem.mol
    w, v = np.linalg.eigh(mol.intor_symmetric("int1e_ovlp"))
    orthogonal = (v * np.sqrt(w)) @ v.T @ orbitals
    slices = mol.aoslice_by_atom()
    aos = [ao for atom in molecule.fragments[0] for ao in range(*slices[atom, 2:4])]
    weights = np.sum(orthogonal[aos] ** 2, axis=0)
    return weights, tuple("A" if weight >= 0.5 else "B" for weight in weights)


def entropy_diagnostics(
    molecule: Molecule,
    subsystem: Subsystem,
    integrals: ActiveSpaceIntegrals,
    mo_coeff: np.ndarray,
    *,
    bond_dims: Sequence[int] = (100, 250),
    sweeps_per_tier: int = 4,
    n_threads: int = 1,
) -> EntropyDiagnostics:
    """Run SZ-symmetry DMRG on an active space and diagnose its orbital entanglement.

    Each bond dimension runs ``sweeps_per_tier`` sweeps; the last sweep of each tier has
    no noise. All sweeps run, so the record holds every tier. The DMRG energy is not a
    reference value.

    Args:
        molecule: Dimer whose first fragment is fragment A.
        subsystem: Counterpoise subsystem of the integrals.
        integrals: Active-space integrals of the subsystem.
        mo_coeff: MO coefficients of the integrals, core orbitals first.
        bond_dims: Bond dimension of each tier, in sweep order.
        sweeps_per_tier: Number of sweeps for each bond dimension.
        n_threads: Number of block2 threads.

    Raises:
        ImportError: The optional ``dmrg`` backend is not installed.
        ValueError: ``bond_dims`` has fewer than two distinct consecutive values.
    """
    if not extra_available("dmrg"):
        raise ImportError(
            "Entropy diagnostics require the optional 'dmrg' backend. "
            "Install it with: uv sync --extra dmrg"
        )
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    record = integrals.record
    schedule = [bond_dim for bond_dim in bond_dims for _ in range(sweeps_per_tier)]
    noises = [0.0 if (i + 1) % sweeps_per_tier == 0 else SWEEP_NOISE for i in range(len(schedule))]
    with tempfile.TemporaryDirectory() as scratch:
        # block2 computes orbital entropies under SZ symmetry only.
        driver = DMRGDriver(scratch=scratch, symm_type=SymmetryTypes.SZ, n_threads=n_threads)
        driver.initialize_system(n_sites=record.ncas, n_elec=record.nelecas, spin=0)
        mpo = driver.get_qc_mpo(
            h1e=integrals.h1, g2e=integrals.h2, ecore=integrals.e_core, iprint=0
        )
        ket = driver.get_random_mps(tag="KET", bond_dim=schedule[0], nroots=1)
        driver.dmrg(mpo, ket, n_sweeps=len(schedule), bond_dims=schedule, noises=noises, tol=0.0)
        # block2 documents ``_dmrg`` as the holder of the per-sweep statistics.
        sweeps = driver._dmrg
        convergence = DMRGConvergence(
            tuple(int(bond_dim) for bond_dim in sweeps.bond_dims),
            tuple(float(np.ravel(energy)[0]) for energy in sweeps.energies),
        )
        s1 = np.asarray(driver.get_orbital_entropies(ket, orb_type=1))
        s2 = np.asarray(driver.get_orbital_entropies(ket, orb_type=2))
    active = mo_coeff[:, record.ncore : record.ncore + record.ncas]
    weights, fragments = assign_orbitals(molecule, subsystem, active)
    return EntropyDiagnostics(
        single_orbital_entropy=s1,
        mutual_information=mutual_information(s1, s2),
        fragment_weights=weights,
        orbital_fragments=fragments,
        assignment_rule=ASSIGNMENT_RULE,
        convergence=convergence,
    )
