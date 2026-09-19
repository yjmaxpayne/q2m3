"""Independent molecular-integral and fixed-frame scientific oracles."""

from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np
import pytest

from q2m3.interfaces import fixed_mo_embedding as embedding
from q2m3.molecule import MoleculeConfig


def test_active_space_public_and_private_alias():
    from q2m3.interfaces import resolve_active_space

    assert resolve_active_space is embedding._resolve_active_space
    assert resolve_active_space(
        n_electrons=10, n_orbitals=7, active_electrons=4, active_orbitals=4
    ) == ((3, 4, 5, 6), 3)
    assert resolve_active_space(
        n_electrons=3, n_orbitals=4, active_electrons=None, active_orbitals=None
    ) == (
        (0, 1, 2, 3),
        0,
    )  # Legacy helper does not impose the SQD spin domain.


@pytest.mark.parametrize(
    "electrons,orbitals,match",
    [
        (None, 4, "together"),
        (0, 4, "positive"),
        (4, 0, "positive"),
        (12, 6, "exceed"),
        (3, 4, "even"),
        (6, 2, "cannot hold"),
        (4, 5, "available"),
    ],
)
def test_active_space_boundaries(electrons, orbitals, match):
    from q2m3.interfaces import resolve_active_space

    with pytest.raises(ValueError, match=match):
        resolve_active_space(
            n_electrons=10,
            n_orbitals=7,
            active_electrons=electrons,
            active_orbitals=orbitals,
        )


@pytest.fixture(scope="module")
def water():
    return MoleculeConfig(
        "water",
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.13, 0.1, 0.96], [0.88, -0.12, -0.29]],
        0,
        4,
        4,
    )


@pytest.fixture(scope="module")
def mm():
    return np.array([0.31, -0.19]), np.array([[2.3, 0.4, 1.2], [-1.7, 2.1, 0.3]])


@pytest.fixture(scope="module")
def helper(water, mm):
    return embedding.build_fixed_mo_embedding_integrals(
        water.symbols,
        water.coords_array,
        mm_charges=mm[0],
        mm_coords=mm[1],
        active_electrons=4,
        active_orbitals=4,
    )


def _mean_field(water):
    from pyscf import gto, scf

    mol = gto.M(
        atom=list(zip(water.symbols, water.coords, strict=True)),
        basis=water.basis,
        charge=water.charge,
        unit="Angstrom",
        verbose=0,
    )
    return scf.RHF(mol)


def _casci(mf, mo):
    from pyscf import ao2mo, mcscf

    cas = mcscf.CASCI(mf, 4, (2, 2))
    h1, core = cas.get_h1eff(mo_coeff=mo)
    h2 = ao2mo.restore(1, cas.get_h2eff(mo_coeff=mo), 4)
    return h1, h2, core


def test_helper_full_frame_and_vacuum_core_match_independent_casci(water, helper):
    assert helper.mo_coeff.shape == (7, 7)
    assert not helper.mo_coeff.flags.writeable
    with pytest.raises(ValueError):
        helper.mo_coeff.setflags(write=True)
    h1, h2, core = _casci(_mean_field(water), helper.mo_coeff)
    np.testing.assert_allclose(helper.one_electron_vacuum, h1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(helper.two_electron, h2.transpose(0, 2, 3, 1), rtol=0, atol=1e-12)
    assert abs(helper.vacuum_core_constant - core) <= 1e-10
    assert abs(core) > 10
    legacy = replace(helper, mo_coeff=None, vacuum_core_constant=None)
    assert legacy.mo_coeff is None


def test_helper_mm_units_match_independent_point_charge_potential(water, mm, helper):
    from pyscf import qmmm

    vac = _mean_field(water)
    embedded = qmmm.mm_charge(_mean_field(water), mm[1], mm[0], unit="Angstrom")
    mo = helper.mo_coeff
    delta = mo.T @ (embedded.get_hcore() - vac.get_hcore()) @ mo
    np.testing.assert_allclose(helper.delta_h_active, delta[3:, 3:], rtol=0, atol=1e-12)
    nuclear_delta = embedded.energy_nuc() - vac.energy_nuc()
    assert abs(helper.delta_nuclear_mm - nuclear_delta) <= 1e-12
    assert abs(nuclear_delta) > 1e-3


@pytest.fixture(scope="module")
def integrals_module():
    return importlib.import_module("q2m3.sqd.integrals")


@pytest.fixture(scope="module")
def vacuum(water, integrals_module):
    return integrals_module.build_integrals(water, host_available_mb=4096)


@pytest.fixture(scope="module", params=["diagonal", "full_oneelectron"])
def assembled(request, water, mm, integrals_module):
    return integrals_module.build_integrals(
        water,
        mm_charges=mm[0],
        mm_coords=mm[1],
        embedding_mode=request.param,
        host_available_mb=4096,
    )


def test_vacuum_matches_independent_casci(water, vacuum, record_property):
    h1, h2, core = _casci(_mean_field(water), vacuum.mo_coeff)
    np.testing.assert_allclose(vacuum.h1, h1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(vacuum.h2, h2, rtol=0, atol=1e-12)
    assert abs(vacuum.e_core - core) <= 1e-10
    assert vacuum.nelec == (2, 2)
    assert vacuum.norb == 4
    assert vacuum.context.active_indices == (3, 4, 5, 6)
    assert vacuum.context.n_core_orbitals == 3
    assert vacuum.context.embedding_mode == "vacuum"
    assert vacuum.context.hf_reference_kind == "canonical_rhf"
    assert vacuum.context.delta_core_constant == 0
    record_property("vacuum_h2_maxabs", float(np.max(np.abs(vacuum.h2 - h2))))
    record_property("vacuum_core_abs_error", abs(vacuum.e_core - core))
    mf = _mean_field(water)
    dm = 2 * vacuum.mo_coeff[:, :5] @ vacuum.mo_coeff[:, :5].T
    assert abs(vacuum.hf_energy - mf.energy_tot(dm=dm)) <= 1e-10


def test_mm_same_frame_casci_and_energy_golden(water, mm, assembled, record_property):
    from pyscf import fci, qmmm

    mf = _mean_field(water)
    embedded = qmmm.mm_charge(_mean_field(water), mm[1], mm[0], unit="Angstrom")
    h1_vac, h2, core_vac = _casci(mf, assembled.mo_coeff)
    h1_full, _, core_full = _casci(embedded, assembled.mo_coeff)
    full = assembled.context.embedding_mode == "full_oneelectron"
    h1 = h1_full if full else h1_vac + np.diag(np.diag(h1_full - h1_vac))
    np.testing.assert_allclose(assembled.h1, h1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(assembled.h2, h2, rtol=0, atol=1e-12)
    assert abs(assembled.e_core - core_full) <= 1e-10
    assert abs(assembled.context.vacuum_core_constant - core_vac) <= 1e-10
    assert abs(assembled.context.delta_core_constant - (core_full - core_vac)) <= 1e-10
    assert assembled.context.hf_reference_kind == "fixed_frame_determinant"
    assert assembled.context.fixed_mo and assembled.context.two_electron_tensor_fixed
    expected_energy = fci.direct_spin1.kernel(h1, h2, 4, (2, 2), ecore=core_full)[0]
    actual_energy = fci.direct_spin1.kernel(
        assembled.h1, assembled.h2, 4, (2, 2), ecore=assembled.e_core
    )[0]
    assert abs(actual_energy - expected_energy) <= 1e-10
    dm = 2 * assembled.mo_coeff[:, :5] @ assembled.mo_coeff[:, :5].T
    assert abs(assembled.hf_energy - embedded.energy_tot(dm=dm)) <= 1e-10
    assert np.max(np.abs(h1_full - h1_vac - np.diag(np.diag(h1_full - h1_vac)))) > 1e-4
    assert np.max(np.abs(h2 - h2.transpose(0, 2, 3, 1))) > 1e-2
    record_property("mm_h2_maxabs", float(np.max(np.abs(assembled.h2 - h2))))
    record_property("mm_reference_energy_abs_error", abs(actual_energy - expected_energy))
    record_property("mm_core_abs_error", abs(assembled.e_core - core_full))
    record_property("mm_hf_ao_abs_error", abs(assembled.hf_energy - embedded.energy_tot(dm=dm)))


def test_zero_mm_is_vacuum_in_identical_frame(water, mm, vacuum, integrals_module):
    zero = integrals_module.build_integrals(
        water,
        mm_charges=np.zeros(2),
        mm_coords=mm[1],
        host_available_mb=4096,
    )
    np.testing.assert_allclose(zero.mo_coeff, vacuum.mo_coeff, rtol=0, atol=1e-13)
    for name in ("h1", "h2"):
        np.testing.assert_allclose(getattr(zero, name), getattr(vacuum, name), rtol=0, atol=1e-12)
    assert abs(zero.e_core - vacuum.e_core) <= 1e-10
    assert abs(zero.hf_energy - vacuum.hf_energy) <= 1e-10
    assert zero.context.frame_id == vacuum.context.frame_id
    # Roundoff in the independently contracted core need not have identical bytes.


def test_integral_snapshots_cannot_be_mutated(vacuum):
    for array in (vacuum.mo_coeff, vacuum.h1, vacuum.h2):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        vacuum.context.source["basis"] = "changed"


def test_hamiltonian_hash_is_canonical_and_sensitive(vacuum, integrals_module):
    fn = integrals_module.hamiltonian_id
    kwargs = dict(norb=4, nelec=(2, 2))
    expected = fn(vacuum.h1, vacuum.h2, vacuum.e_core, **kwargs)
    assert expected == vacuum.context.hamiltonian_id
    assert (
        fn(
            np.asfortranarray(vacuum.h1).astype(">f8"),
            vacuum.h2.astype(">f8"),
            vacuum.e_core,
            **kwargs,
        )
        == expected
    )
    changed = vacuum.h1.copy()
    changed[0, 0] += 1e-6
    assert fn(changed, vacuum.h2, vacuum.e_core, **kwargs) != expected
    assert fn(vacuum.h1, vacuum.h2, vacuum.e_core + 1e-6, **kwargs) != expected
    assert fn(vacuum.h1, vacuum.h2, vacuum.e_core, norb=4, nelec=(1, 1)) != expected
    with pytest.raises(ValueError):
        fn(vacuum.h1.astype(complex), vacuum.h2, vacuum.e_core, **kwargs)


def test_mm_builder_uses_helper_frame_once(monkeypatch, water, mm, helper, integrals_module):
    calls = []

    def supplied(*args, **kwargs):
        calls.append(1)
        return helper

    from pyscf import scf

    monkeypatch.setattr(integrals_module, "build_fixed_mo_embedding_integrals", supplied)
    monkeypatch.setattr(scf, "RHF", lambda *a, **k: pytest.fail("second vacuum RHF"))
    result = integrals_module.build_integrals(
        water,
        mm_charges=mm[0],
        mm_coords=mm[1],
        host_available_mb=4096,
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(result.mo_coeff, helper.mo_coeff)


@pytest.mark.parametrize("field", ["mo_coeff", "vacuum_core_constant"])
def test_mm_builder_rejects_legacy_missing_frame_fields(
    monkeypatch,
    water,
    mm,
    helper,
    integrals_module,
    field,
):
    monkeypatch.setattr(
        integrals_module,
        "build_fixed_mo_embedding_integrals",
        lambda *a, **k: replace(helper, **{field: None}),
    )
    with pytest.raises(ValueError, match=field):
        integrals_module.build_integrals(
            water, mm_charges=mm[0], mm_coords=mm[1], host_available_mb=4096
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_mode": "full_oneelectron"},
        {"mm_charges": np.ones(1)},
        {"mm_charges": np.zeros(0), "mm_coords": np.zeros((0, 3))},
    ],
)
def test_molecular_mm_boundary_checks(water, integrals_module, kwargs):
    with pytest.raises(ValueError):
        integrals_module.build_integrals(water, host_available_mb=4096, **kwargs)


def test_allocation_guard_rejects_before_scf(monkeypatch, water, integrals_module):
    from pyscf import scf

    from q2m3.sqd.exceptions import ResourceLimitError

    monkeypatch.setattr(scf, "RHF", lambda *a, **k: pytest.fail("SCF before resource guard"))
    with pytest.raises(ResourceLimitError, match="stage=integrals"):
        integrals_module.build_integrals(water, host_available_mb=1)


def test_resource_domain_rejects_before_scf(monkeypatch, water, integrals_module):
    from pyscf import scf

    from q2m3.sqd.exceptions import ResourceModelDomainError

    monkeypatch.setattr(scf, "RHF", lambda *a, **k: pytest.fail("SCF before resource guard"))
    with pytest.raises(ResourceModelDomainError):
        integrals_module.build_integrals(replace(water, basis="cc-pvtz"), host_available_mb=4096)


def test_signed_helper_frame_is_consumed_without_recanonicalizing(
    monkeypatch,
    water,
    mm,
    assembled,
    integrals_module,
):
    from pyscf import fci, qmmm

    original = embedding._canonicalize_mo_signs

    def signed(mo):
        result = original(mo)
        result[:, 5] *= -1
        return result

    monkeypatch.setattr(embedding, "_canonicalize_mo_signs", signed)
    changed = integrals_module.build_integrals(
        water,
        mm_charges=mm[0],
        mm_coords=mm[1],
        embedding_mode=assembled.context.embedding_mode,
        host_available_mb=4096,
    )
    assert changed.context.frame_id != assembled.context.frame_id
    assert changed.context.hamiltonian_id != assembled.context.hamiltonian_id
    signs = np.array([1, 1, -1, 1])
    np.testing.assert_allclose(
        changed.h1, np.einsum("p,q,pq->pq", signs, signs, assembled.h1), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        changed.h2,
        np.einsum("p,q,r,s,pqrs->pqrs", signs, signs, signs, signs, assembled.h2),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(changed.mo_coeff[:, 5], -assembled.mo_coeff[:, 5], rtol=0, atol=0)
    embedded = qmmm.mm_charge(_mean_field(water), mm[1], mm[0], unit="Angstrom")
    _, golden_h2, _ = _casci(embedded, changed.mo_coeff)
    np.testing.assert_allclose(changed.h2, golden_h2, rtol=0, atol=1e-12)
    energies = [
        fci.direct_spin1.kernel(x.h1, x.h2, 4, (2, 2), ecore=x.e_core)[0]
        for x in (assembled, changed)
    ]
    assert abs(energies[0] - energies[1]) <= 1e-10


def test_real_rccsd_oracle_detects_amplitude_only_frame_change(vacuum, record_property):
    """Independent oracle only; physical seed reception belongs to the seed solver."""
    from pyscf import ao2mo, cc, gto, scf

    mol = gto.M(verbose=0)
    mol.nelectron = 4
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: vacuum.h1
    mf.get_ovlp = lambda *args: np.eye(4)
    mf._eri = ao2mo.restore(8, vacuum.h2, 4)
    mf.mo_coeff = np.eye(4)
    mf.mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
    mf.mo_energy = np.diag(mf.get_fock())
    mf.e_tot = vacuum.hf_energy
    mf.converged = True
    solver = cc.CCSD(mf)
    solver.conv_tol = 1e-13
    solver.conv_tol_normt = 1e-11
    solver.max_cycle = 200
    correlation, t1, t2 = solver.kernel()
    assert solver.converged
    eris = solver.ao2mo()
    assert abs(solver.energy(t1, t2, eris) - correlation) <= 1e-12
    signs = np.array([-1, 1])
    wrong_t1 = t1 * signs[None, :]
    wrong_t2 = t2 * signs[None, None, :, None] * signs[None, None, None, :]
    # Labels can remain unchanged; actual amplitudes in the wrong virtual gauge
    # do not solve these integrals, and change their independently evaluated energy.
    mismatch = abs(solver.energy(wrong_t1, wrong_t2, eris) - correlation)
    assert mismatch > 1e-5
    next_t1, next_t2 = solver.update_amps(wrong_t1, wrong_t2, eris)
    update_defect = max(np.max(np.abs(next_t1 - wrong_t1)), np.max(np.abs(next_t2 - wrong_t2)))
    assert update_defect > 1e-4
    record_property("amplitude_only_frame_energy_error_ha", mismatch)
    record_property("amplitude_only_frame_update_defect", float(update_defect))


def test_numpy_integer_molecule_counts_preserve_frame(water, vacuum, integrals_module):
    result = integrals_module.build_integrals(
        replace(
            water, charge=np.int64(0), active_electrons=np.int64(4), active_orbitals=np.int64(4)
        ),
        host_available_mb=4096,
    )
    assert result.context.frame_id == vacuum.context.frame_id
    assert result.context.hamiltonian_id == vacuum.context.hamiltonian_id
