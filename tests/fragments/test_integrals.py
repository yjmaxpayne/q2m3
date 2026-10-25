"""Active-space integrals and integral contexts of the counterpoise subsystems.

Each subsystem gives h1, h2 (chemist order) and e_core from a CASCI object over its AVAS
orbitals. The context names the orbital frame by a SHA256 digest of the geometry, the
real-atom mask, the active window and the MO coefficients, and the Hamiltonian by the
digest of the integrals.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest
from pyscf import fci, mcscf

from q2m3.fragments.active_space import select_active_spaces
from q2m3.fragments.counterpoise import counterpoise
from q2m3.fragments.integrals import ActiveSpaceIntegrals, build_integrals, frame_id
from q2m3.fragments.molecule import Molecule
from q2m3.sqd.config import validate_integral_inputs
from q2m3.sqd.integrals import hamiltonian_id
from tests.fragments.conftest import AVAS_SETTINGS, WATER_DIMER

DIMER_FRAGMENTS = ((0, 1, 2), (3, 4, 5))
LABELS, THRESHOLD = AVAS_SETTINGS["S6"]
SUBSYSTEMS = ("AB", "A", "B")
# Full CI of the integrals against an independent CASCI over the same orbitals.
ENERGY_TOLERANCE_HA = 1e-8


def _xyz(atoms) -> str:
    lines = [str(len(atoms)), ""] + [f"{s} {x!r} {y!r} {z!r}" for s, (x, y, z) in atoms]
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def s6():
    molecule = Molecule.from_xyz(_xyz(WATER_DIMER), fragments=DIMER_FRAGMENTS)
    subsystems = counterpoise(molecule, basis="cc-pvdz")
    spaces = select_active_spaces(subsystems, LABELS, THRESHOLD)
    built = {
        name: build_integrals(
            molecule, subsystems[name], record, mo, avas_labels=LABELS, avas_threshold=THRESHOLD
        )
        for name, (record, mo) in spaces.items()
    }
    return molecule, subsystems, spaces, built


def _full_ci(h1, h2, e_core, integrals: ActiveSpaceIntegrals) -> float:
    norb = integrals.record.ncas
    return fci.direct_spin1.kernel(h1, h2, norb, integrals.nelec, ecore=e_core)[0]


def _casci(subsystem, record, mo) -> float:
    casci = mcscf.CASCI(subsystem.scf, record.ncas, record.nelecas)
    casci.verbose = 0
    return casci.kernel(mo)[0]


def _expected_frame_id(molecule, subsystem, mo, ncore, ncas, *, mask=True) -> str:
    # Independent reading of the fragment frame byte format.
    header = {
        "symbols": list(molecule.symbols),
        "coordinates_angstrom": [list(xyz) for xyz in molecule.coordinates_angstrom],
        "basis": subsystem.mol.basis,
        "cart": subsystem.mol.cart,
        "real_atom_mask": [i in subsystem.real_atoms for i in range(len(molecule.symbols))],
        "charge": subsystem.mol.charge,
        "active_indices": list(range(ncore, ncore + ncas)),
        "n_core": ncore,
    }
    if not mask:
        del header["real_atom_mask"]
    digest = hashlib.sha256(b"q2m3.fragment-frame.v1\0")
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    mo = np.ascontiguousarray(mo, dtype="<f8")
    digest.update(json.dumps(["mo_coeff", list(mo.shape), "<f8"], separators=(",", ":")).encode())
    digest.update(b"\0" + mo.tobytes(order="C"))
    return digest.hexdigest()


@pytest.mark.parametrize("name", SUBSYSTEMS)
def test_full_ci_of_integrals_matches_casci(s6, name, record_property):
    _molecule, subsystems, spaces, built = s6
    integrals = built[name]
    record, mo = spaces[name]

    energy = _full_ci(integrals.h1, integrals.h2, integrals.e_core, integrals)
    diff = abs(energy - _casci(subsystems[name], record, mo))

    record_property("abs_diff_ha", float(diff))
    assert diff <= ENERGY_TOLERANCE_HA


@pytest.mark.parametrize("name", SUBSYSTEMS)
def test_physicist_order_tensor_breaks_agreement(s6, name):
    _molecule, subsystems, spaces, built = s6
    integrals = built[name]
    record, mo = spaces[name]

    wrong = np.swapaxes(integrals.h2, 1, 2)
    energy = _full_ci(integrals.h1, wrong, integrals.e_core, integrals)

    assert abs(energy - _casci(subsystems[name], record, mo)) > ENERGY_TOLERANCE_HA


@pytest.mark.parametrize("name", SUBSYSTEMS)
def test_one_electron_matrix_is_exactly_symmetric(s6, name):
    # The raw matrix is asymmetric near 1e-15 Ha, below every downstream tolerance.
    h1 = s6[3][name].h1
    assert np.array_equal(h1, h1.T)


@pytest.mark.parametrize("name", SUBSYSTEMS)
def test_frame_id_follows_byte_format(s6, name):
    molecule, subsystems, spaces, built = s6
    record, mo = spaces[name]
    args = (molecule, subsystems[name], mo, record.ncore, record.ncas)

    actual = built[name].context.frame_id

    assert actual == _expected_frame_id(*args)
    assert actual != _expected_frame_id(*args, mask=False)


def test_frame_id_uses_subsystem_charge_and_basis(water_dimer):
    molecule = Molecule.from_xyz(_xyz(water_dimer), fragments=DIMER_FRAGMENTS, charges=(2, -2))
    subsystems = counterpoise(molecule, basis="sto-3g")
    # Fixture premise: singlet monomers, and fragment charges that differ from the total.
    assert [subsystems[n].mol.nelectron % 2 for n in ("A", "B")] == [0, 0]
    assert {subsystems["A"].mol.charge, subsystems["B"].mol.charge, molecule.charge} == {2, -2, 0}

    for name in SUBSYSTEMS:
        sub = subsystems[name]
        actual = frame_id(molecule, sub, sub.scf.mo_coeff, ncore=1, ncas=2)
        assert actual == _expected_frame_id(molecule, sub, sub.scf.mo_coeff, 1, 2)


@pytest.mark.parametrize("name", SUBSYSTEMS)
def test_hamiltonian_id_is_digest_of_returned_integrals(s6, name):
    integrals = s6[3][name]
    norb, nelec = integrals.record.ncas, integrals.nelec
    h1, h2, e_core = integrals.h1, integrals.h2, integrals.e_core

    expected = hamiltonian_id(h1, h2, e_core, norb=norb, nelec=nelec)
    shifted = h1.copy()
    shifted[0, 0] += 1e-9

    assert integrals.context.hamiltonian_id == expected
    assert hamiltonian_id(shifted, h2, e_core, norb=norb, nelec=nelec) != expected


@pytest.mark.parametrize("name", SUBSYSTEMS)
def test_context_passes_integral_input_validation(s6, name):
    molecule, _subsystems, spaces, built = s6
    integrals = built[name]
    record = spaces[name][0]
    context = integrals.context
    norb = record.ncas

    validate_integral_inputs(
        integrals.h1,
        integrals.h2,
        integrals.e_core,
        norb=norb,
        nelec=integrals.nelec,
        context=context,
        seed_data=None,
        mode="reference_only",
    )

    assert integrals.nelec == (record.nelecas // 2, record.nelecas // 2)
    assert type(integrals.e_core) is float
    assert context.active_indices == tuple(range(record.ncore, record.ncore + norb))
    assert context.n_core_orbitals == record.ncore
    assert (context.vacuum_core_constant, context.delta_core_constant) == (integrals.e_core, 0.0)
    assert (context.embedding_mode, context.fixed_mo, context.two_electron_tensor_fixed) == (
        "vacuum",
        True,
        True,
    )
    assert context.hf_reference_kind == "fixed_frame_determinant"
    assert dict(context.source) == {
        "producer": "q2m3.fragments",
        "frame_version": "q2m3.fragment-frame.v1",
        "hamiltonian_version": "q2m3.hamiltonian.v1",
        "subsystem": name,
        "geometry_sha256": molecule.geometry_sha256(),
        "basis": "cc-pvdz",
        "avas_labels": LABELS,
        "avas_threshold": THRESHOLD,
    }


@pytest.mark.parametrize("name", SUBSYSTEMS)
def test_record_carries_frame_and_hamiltonian_ids(s6, name):
    record = s6[2][name][0]
    context = s6[3][name].context

    expected = dataclasses.replace(
        record, frame_id=context.frame_id, hamiltonian_id=context.hamiltonian_id
    )

    assert s6[3][name].record == expected
