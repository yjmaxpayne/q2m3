"""Full-space reference energies of the counterpoise subsystems."""

from __future__ import annotations

from pyscf import mp
from pyscf.data.elements import chemcore

from q2m3.fragments.counterpoise import Subsystem


def frozen_core_mp2(subsystem: Subsystem) -> tuple[float, int]:
    """Return the full-space MP2 total energy (Ha) and the number of frozen orbitals.

    The frozen orbitals are the chemical core of the atoms with a nucleus. Ghost atoms
    add no core orbitals, so a monomer in the dimer basis freezes only its own core.
    """
    nfrozen = chemcore(subsystem.mol)
    solver = mp.MP2(subsystem.scf, frozen=nfrozen)
    solver.kernel()
    return float(solver.e_tot), nfrozen
