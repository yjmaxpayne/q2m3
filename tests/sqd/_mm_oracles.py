"""Independent PySCF oracles for MM-embedded SQD benchmarks.

These helpers share only PySCF and the vacuum MO sign rule with q2m3. They do
not call q2m3 integral code, so they can check it. This module must not import
``q2m3.sqd`` at module level: the core test profile collects its importers.
"""

from __future__ import annotations

import numpy as np


def mm_embedding(fixture: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return TIP3P point charges (Angstrom, e) built from the fixture state vectors."""
    from q2m3.solvation.solvent import TIP3P_WATER, SolventMolecule, get_mm_embedding_data

    molecules = [
        SolventMolecule.from_state_vector(TIP3P_WATER, np.asarray(state, dtype=float))
        for state in fixture["water_states"]
    ]
    return get_mm_embedding_data(molecules)


def _vacuum_frame(symbols, coords, charge):
    """Return the QM molecule, vacuum RHF and its MOs with the largest |c| positive."""
    from pyscf import gto, scf

    mol = gto.M(
        atom=list(zip(symbols, np.asarray(coords, dtype=float).tolist(), strict=True)),
        basis="sto-3g",
        charge=charge,
        unit="Angstrom",
        verbose=0,
    )
    vacuum = scf.RHF(mol)
    vacuum.verbose = 0
    vacuum.run()
    if not vacuum.converged:
        raise RuntimeError("vacuum RHF did not converge")
    mo = vacuum.mo_coeff.copy()
    for column in mo.T:
        if column[np.argmax(np.abs(column))] < 0:
            column *= -1
    return mol, vacuum, mo


def _h1eff(symbols, coords, charge, mm_coords, mm_charges, ncas, nelecas):
    """Return vacuum and embedded CASCI h1eff, core energies and the vacuum h2."""
    from pyscf import ao2mo, mcscf, qmmm, scf

    mol, vacuum, mo = _vacuum_frame(symbols, coords, charge)
    cas_vacuum = mcscf.CASCI(vacuum, ncas, nelecas)
    h1_vacuum, core_vacuum = cas_vacuum.get_h1eff(mo_coeff=mo)
    h2 = ao2mo.restore(1, cas_vacuum.get_h2eff(mo_coeff=mo), ncas)
    if len(mm_charges):
        embedded = qmmm.mm_charge(
            scf.RHF(mol), np.asarray(mm_coords), np.asarray(mm_charges), unit="Angstrom"
        )
        h1_full, core_full = mcscf.CASCI(embedded, ncas, nelecas).get_h1eff(mo_coeff=mo)
    else:
        h1_full, core_full = h1_vacuum, core_vacuum
    return h1_vacuum, h1_full, core_full, h2


def independent_casci(symbols, coords, charge, mm_coords, mm_charges, ncas, nelecas, mode):
    """Return the CASCI energy (Ha) of the MM mean field in the vacuum MO frame.

    Args:
        mode: ``"full_oneelectron"`` keeps the full embedded h1eff.
            ``"diagonal"`` keeps the vacuum h1eff plus the embedding diagonal.
    """
    from pyscf import fci

    if mode not in ("diagonal", "full_oneelectron"):
        raise ValueError(f"unknown embedding mode: {mode!r}")
    h1_vacuum, h1_full, core_full, h2 = _h1eff(
        symbols, coords, charge, mm_coords, mm_charges, ncas, nelecas
    )
    if mode == "diagonal":
        h1 = h1_vacuum + np.diag(np.diag(h1_full - h1_vacuum))
    else:
        h1 = h1_full
    half = nelecas // 2
    energy = fci.direct_spin1.kernel(h1, h2, ncas, (half, half), ecore=core_full, tol=1e-12)[0]
    return float(energy)


def independent_offdiag_delta_h1(symbols, coords, charge, mm_coords, mm_charges, ncas, nelecas):
    """Return the off-diagonal block of embedded minus vacuum h1eff in the vacuum MO frame."""
    h1_vacuum, h1_full, _, _ = _h1eff(symbols, coords, charge, mm_coords, mm_charges, ncas, nelecas)
    delta = h1_full - h1_vacuum
    return delta - np.diag(np.diag(delta))
