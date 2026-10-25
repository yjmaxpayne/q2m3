"""Counterpoise subsystems: the dimer and both monomers in the full dimer basis.

The reference energies come from PySCF with explicit ``ghost-`` atoms. Both sides use
PySCF, so this checks the subsystem construction, not agreement between programs.
"""

from __future__ import annotations

import pytest
from pyscf import gto, scf
from pyscf.scf import hf

from q2m3.fragments.counterpoise import SCF_CONV_TOL, counterpoise
from q2m3.fragments.errors import FragmentWorkflowError
from q2m3.fragments.molecule import Molecule

DIMER_FRAGMENTS = ((0, 1, 2), (3, 4, 5))


def _molecule(atoms, **kwargs) -> Molecule:
    lines = [str(len(atoms)), ""] + [f"{s} {x!r} {y!r} {z!r}" for s, (x, y, z) in atoms]
    return Molecule.from_xyz("\n".join(lines) + "\n", **kwargs)


def _explicit_rhf_energy(atoms, real, *, ghost_others: bool) -> float:
    """RHF energy of the atoms in ``real``; with ``ghost_others`` the rest keep basis only."""
    spec = []
    for i, (symbol, xyz) in enumerate(atoms):
        if i in real:
            spec.append((symbol, xyz))
        elif ghost_others:
            spec.append((f"ghost-{symbol}", xyz))
    mol = gto.M(atom=spec, basis="cc-pvdz", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    return mf.kernel()


def _explicit_interaction(atoms, *, ghost_others: bool) -> float:
    a, b = DIMER_FRAGMENTS
    e_ab = _explicit_rhf_energy(atoms, range(6), ghost_others=False)
    e_a = _explicit_rhf_energy(atoms, a, ghost_others=ghost_others)
    e_b = _explicit_rhf_energy(atoms, b, ghost_others=ghost_others)
    return e_ab - e_a - e_b


def test_counterpoise_interaction_matches_explicit_ghost_atoms(water_dimer):
    subsystems = counterpoise(_molecule(water_dimer, fragments=DIMER_FRAGMENTS), basis="cc-pvdz")
    energies = {name: sub.scf.e_tot for name, sub in subsystems.items()}
    tested = energies["AB"] - energies["A"] - energies["B"]

    reference = _explicit_interaction(water_dimer, ghost_others=True)
    without_ghosts = _explicit_interaction(water_dimer, ghost_others=False)

    assert abs(tested - reference) < 1e-8
    # The check must reject monomers computed in their own basis (no basis-set superposition).
    assert abs(without_ghosts - reference) >= 1e-8


def test_subsystems_share_atom_order_and_basis(water_dimer):
    # Even electron counts in each monomer (8 and 12), so every subsystem is a singlet.
    molecule = _molecule(water_dimer, fragments=DIMER_FRAGMENTS, charges=(2, -2))
    subsystems = counterpoise(molecule, basis="sto-3g")

    assert list(subsystems) == ["AB", "A", "B"]
    expected = {"AB": ((0, 1, 2, 3, 4, 5), ()), "A": DIMER_FRAGMENTS, "B": DIMER_FRAGMENTS[::-1]}
    for name, (real, ghost) in expected.items():
        sub = subsystems[name]
        assert (sub.name, sub.real_atoms, sub.ghost_atoms) == (name, real, ghost)
        labels = [sub.mol.atom_symbol(i) for i in range(sub.mol.natm)]
        assert labels == [f"GHOST-{s}" if i in ghost else s for i, (s, _) in enumerate(water_dimer)]
        assert sub.mol.nao == subsystems["AB"].mol.nao
        assert sub.mol.unit == "Angstrom" and sub.mol.cart is False and sub.mol.spin == 0
        assert sub.scf.conv_tol == SCF_CONV_TOL == 1e-10
        assert not hasattr(sub.scf, "with_df")
        assert sub.scf.converged
        # The convergence monitor holds the SCF loop locals; it must not outlive the run.
        assert sub.scf.callback is None
    assert [subsystems[n].mol.charge for n in ("AB", "A", "B")] == [0, 2, -2]


def test_single_molecule_builds_one_subsystem(water_dimer):
    subsystems = counterpoise(_molecule(water_dimer[:3]), basis="sto-3g")

    assert list(subsystems) == ["M"]
    assert subsystems["M"].real_atoms == (0, 1, 2) and subsystems["M"].ghost_atoms == ()


def test_unconverged_rhf_names_the_subsystem(water_dimer, monkeypatch):
    monkeypatch.setattr(hf.SCF, "max_cycle", 2)
    molecule = _molecule(water_dimer, fragments=DIMER_FRAGMENTS)

    with pytest.raises(FragmentWorkflowError, match=r"AB.*energy change"):
        counterpoise(molecule, basis="sto-3g")
