"""Counterpoise subsystems: RHF for the dimer and for each monomer in the dimer basis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyscf import gto, scf

from q2m3.fragments.errors import FragmentWorkflowError
from q2m3.fragments.molecule import Molecule

SCF_CONV_TOL = 1e-10


@dataclass(frozen=True)
class Subsystem:
    """One counterpoise subsystem and its converged RHF.

    ``mol`` lists every atom of the molecule in input order. Atoms in ``ghost_atoms``
    are ``ghost-<element>`` atoms: they keep their basis functions but have no nucleus
    and no electrons.
    """

    name: str
    mol: gto.Mole
    scf: scf.hf.RHF
    real_atoms: tuple[int, ...]
    ghost_atoms: tuple[int, ...]


def counterpoise(molecule: Molecule, *, basis: str) -> dict[str, Subsystem]:
    """Run RHF for the counterpoise subsystems of ``molecule``.

    A dimer gives ``"AB"``, ``"A"`` (fragment B as ghost atoms) and ``"B"`` (fragment A
    as ghost atoms), so all three share one AO basis. A single molecule gives ``"M"``.
    Each RHF uses exact integrals (no density fitting), spherical basis functions and
    ``conv_tol=SCF_CONV_TOL``.

    Raises:
        FragmentWorkflowError: The RHF of a subsystem does not converge.
    """
    all_atoms = tuple(range(len(molecule.symbols)))
    if molecule.fragments is None:
        plan = [("M", all_atoms, molecule.charge)]
    else:
        a, b = molecule.fragments
        plan = [
            ("AB", all_atoms, molecule.charge),
            ("A", a, molecule.charges[0]),
            ("B", b, molecule.charges[1]),
        ]
    return {name: _run_rhf(molecule, name, real, charge, basis) for name, real, charge in plan}


def _run_rhf(
    molecule: Molecule, name: str, real: tuple[int, ...], charge: int, basis: str
) -> Subsystem:
    atoms = [
        (symbol if i in real else f"ghost-{symbol}", xyz)
        for i, (symbol, xyz) in enumerate(
            zip(molecule.symbols, molecule.coordinates_angstrom, strict=True)
        )
    ]
    mol = gto.M(
        atom=atoms, basis=basis, unit="Angstrom", charge=charge, spin=0, cart=False, verbose=0
    )
    mf = scf.RHF(mol)
    mf.conv_tol = SCF_CONV_TOL
    last: dict[str, Any] = {}
    mf.callback = last.update
    mf.kernel()
    mf.callback = None
    if not mf.converged:
        change = last.get("e_tot", float("nan")) - last.get("last_hf_e", float("nan"))
        raise FragmentWorkflowError(
            f"RHF for subsystem {name} did not converge: last energy change {change:.3e} Ha"
        )
    ghost = tuple(i for i in range(len(atoms)) if i not in real)
    return Subsystem(name, mol, mf, tuple(real), ghost)
