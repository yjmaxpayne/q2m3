"""Active-space integrals and integral contexts of the counterpoise subsystems.

The integrals come from a CASCI object over the AVAS orbitals of one subsystem. The context
names the orbital frame by the ``q2m3.fragment-frame.v1`` digest and the Hamiltonian by the
``q2m3.hamiltonian.v1`` digest of :mod:`q2m3.sqd.integrals`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence

import numpy as np
from pyscf import ao2mo, mcscf

from q2m3.fragments.counterpoise import Subsystem
from q2m3.fragments.molecule import Molecule
from q2m3.fragments.result import ActiveSpaceRecord
from q2m3.sqd.config import IntegralContext
from q2m3.sqd.integrals import hamiltonian_id

FRAME_VERSION = "q2m3.fragment-frame.v1"
HAMILTONIAN_VERSION = "q2m3.hamiltonian.v1"


@dataclasses.dataclass(frozen=True)
class ActiveSpaceIntegrals:
    """Active-space Hamiltonian of one subsystem in the input format of the SQD solver.

    ``h1`` and ``h2`` (chemist order) are in Hartree over ``record.ncas`` orbitals.
    ``e_core`` is the frozen-core plus nuclear energy. ``record`` carries the frame and
    Hamiltonian digests of ``context``.
    """

    h1: np.ndarray
    h2: np.ndarray
    e_core: float
    nelec: tuple[int, int]
    context: IntegralContext
    record: ActiveSpaceRecord


def frame_id(
    molecule: Molecule, subsystem: Subsystem, mo_coeff: np.ndarray, *, ncore: int, ncas: int
) -> str:
    """Hash the orbital frame of one subsystem using the ``q2m3.fragment-frame.v1`` format.

    The digest starts with the version and NUL. Then come compact ASCII JSON with sorted
    keys and NUL: ``symbols``, ``coordinates_angstrom``, ``basis``, ``cart``,
    ``real_atom_mask``, ``charge`` (of the subsystem), ``active_indices`` and ``n_core``.
    Last come the compact ``["mo_coeff",shape,"<f8"]`` JSON header and NUL, and the
    little-endian float64 C-order bytes of ``mo_coeff``.

    The digest depends on the bits of ``mo_coeff``. Use it only to pair a context with its
    seed in one run. Compare recomputed results by energy.

    Returns:
        Lowercase SHA256 hex digest.
    """
    mol = subsystem.mol
    header = {
        "symbols": list(molecule.symbols),
        "coordinates_angstrom": [list(xyz) for xyz in molecule.coordinates_angstrom],
        "basis": mol.basis,
        "cart": bool(mol.cart),
        "real_atom_mask": [i in subsystem.real_atoms for i in range(mol.natm)],
        "charge": int(mol.charge),
        "active_indices": list(range(ncore, ncore + ncas)),
        "n_core": int(ncore),
    }
    mo = np.ascontiguousarray(mo_coeff, dtype="<f8")
    mo_header = json.dumps(["mo_coeff", list(mo.shape), "<f8"], separators=(",", ":"))
    digest = hashlib.sha256(FRAME_VERSION.encode("ascii") + b"\0")
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0" + mo_header.encode("ascii") + b"\0")
    digest.update(mo.tobytes(order="C"))
    return digest.hexdigest()


def build_integrals(
    molecule: Molecule,
    subsystem: Subsystem,
    record: ActiveSpaceRecord,
    mo_coeff: np.ndarray,
    *,
    avas_labels: Sequence[str],
    avas_threshold: float,
) -> ActiveSpaceIntegrals:
    """Build the active-space integrals and the integral context of one subsystem.

    Args:
        molecule: Input molecule of the counterpoise subsystems.
        subsystem: Subsystem with its converged RHF.
        record: Active space that AVAS selected for the subsystem.
        mo_coeff: AVAS MO coefficients, core orbitals first, then the active window.
        avas_labels: AO labels given to AVAS, as the user wrote them.
        avas_threshold: AVAS threshold.

    Returns:
        Integrals, context and the record with its frame and Hamiltonian digests.
    """
    ncas, ncore = record.ncas, record.ncore
    casci = mcscf.CASCI(subsystem.scf, ncas, record.nelecas)
    h1, e_core = casci.get_h1eff(mo_coeff)
    h1 = 0.5 * (h1 + h1.T)
    h2 = ao2mo.restore(1, casci.get_h2eff(mo_coeff), ncas)
    e_core = float(e_core)
    nelec = (record.nelecas // 2, record.nelecas // 2)
    context = IntegralContext(
        frame_id=frame_id(molecule, subsystem, mo_coeff, ncore=ncore, ncas=ncas),
        hamiltonian_id=hamiltonian_id(h1, h2, e_core, norb=ncas, nelec=nelec),
        active_indices=tuple(range(ncore, ncore + ncas)),
        n_core_orbitals=ncore,
        vacuum_core_constant=e_core,
        delta_core_constant=0.0,
        embedding_mode="vacuum",
        fixed_mo=True,
        two_electron_tensor_fixed=True,
        # AVAS orbitals are not the canonical RHF orbitals.
        hf_reference_kind="fixed_frame_determinant",
        source={
            "producer": "q2m3.fragments",
            "frame_version": FRAME_VERSION,
            "hamiltonian_version": HAMILTONIAN_VERSION,
            "subsystem": subsystem.name,
            "geometry_sha256": molecule.geometry_sha256(),
            "basis": subsystem.mol.basis,
            "avas_labels": list(avas_labels),
            "avas_threshold": float(avas_threshold),
        },
    )
    filled = dataclasses.replace(
        record, frame_id=context.frame_id, hamiltonian_id=context.hamiltonian_id
    )
    return ActiveSpaceIntegrals(h1, h2, e_core, nelec, context, filled)
