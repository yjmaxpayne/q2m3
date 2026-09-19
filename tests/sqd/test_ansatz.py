"""Same-frame RCCSD reception checked against an independent Fock-space oracle."""

from __future__ import annotations

import importlib
from dataclasses import replace
from itertools import combinations, product

import numpy as np
import pytest

from q2m3.molecule import MoleculeConfig
from q2m3.sqd.config import LUCJConfig
from q2m3.sqd.exceptions import CCSDConvergenceError, ProvenanceMismatchError, ResourceLimitError
from q2m3.sqd.integrals import build_integrals

pytestmark = pytest.mark.sqd


@pytest.fixture(scope="module")
def module():
    return importlib.import_module("q2m3.sqd.ansatz")


@pytest.fixture(scope="module", params=["h2", "water_mm"])
def data(request):
    if request.param == "h2":
        molecule = MoleculeConfig("h2", ["H", "H"], [[0, 0, 0], [0, 0, 0.74]], 0, 2, 2)
        return build_integrals(molecule, host_available_mb=4096)
    molecule = MoleculeConfig(
        "water", ["O", "H", "H"], [[0, 0, 0], [0.13, 0.1, 0.96], [0.88, -0.12, -0.29]], 0, 4, 4
    )
    return build_integrals(
        molecule,
        host_available_mb=4096,
        mm_charges=np.array([0.31, -0.19]),
        mm_coords=np.array([[2.3, 0.4, 1.2], [-1.7, 2.1, 0.3]]),
        embedding_mode="full_oneelectron",
    )


def fock_oracle(data, t1, t2):
    """Explicit fermionic matrices; no PySCF CC energy/update or integral conversion."""
    from scipy.linalg import expm

    n, o = data.norb, data.nelec[0]
    strings = [sum(1 << i for i in c) for c in combinations(range(n), o)]
    basis = [a | (b << n) for a, b in product(strings, repeat=2)]
    index = {bits: i for i, bits in enumerate(basis)}
    size = len(basis)

    def operator(actions):
        matrix = np.zeros((size, size))
        for col, bits in enumerate(basis):
            sign = 1
            for orbital, creation in reversed(actions):
                occupied = bool(bits & (1 << orbital))
                if occupied == creation:
                    break
                sign *= (-1) ** ((bits & ((1 << orbital) - 1)).bit_count())
                bits ^= 1 << orbital
            else:
                if bits in index:
                    matrix[index[bits], col] += sign
        return matrix

    h = np.eye(size) * data.e_core
    excitations = {}
    for p, q in product(range(n), repeat=2):
        e = sum(operator([(p + s * n, True), (q + s * n, False)]) for s in range(2))
        h += data.h1[p, q] * e
        if p >= o and q < o:
            excitations[p - o, q] = e
    for p, q, r, s in product(range(n), repeat=4):
        for spin, other in product(range(2), repeat=2):
            h += (
                0.5
                * data.h2[p, q, r, s]
                * operator(
                    [
                        (p + spin * n, True),
                        (r + other * n, True),
                        (s + other * n, False),
                        (q + spin * n, False),
                    ]
                )
            )
    t = np.zeros_like(h)
    for i, a in product(range(o), range(n - o)):
        t += t1[i, a] * excitations[a, i]
    for i, j, a, b in product(range(o), range(o), range(n - o), range(n - o)):
        t += 0.5 * t2[i, j, a, b] * excitations[a, i] @ excitations[b, j]
    hf = (1 << o) - 1 | (((1 << o) - 1) << n)
    hf_idx = index[hf]
    column = (expm(-t) @ h @ expm(t))[:, hf_idx]
    projected = [
        abs(column[k])
        for k, bits in enumerate(basis)
        if ((hf & ~bits & ((1 << n) - 1)).bit_count(), ((hf & ~bits) >> n).bit_count())
        in ((1, 0), (1, 1))
    ]
    return h[hf_idx, hf_idx], column[hf_idx], max(projected, default=0)


def test_same_hamiltonian_seed_has_physical_ha_residual(module, data, monkeypatch, record_property):
    from pyscf.scf.hf import RHF

    monkeypatch.setattr(RHF, "kernel", lambda *a, **k: pytest.fail("SCF rerun"))
    seed = module.build_ccsd_seed(data, host_available_mb=4096)
    hf, energy, residual = fock_oracle(data, seed.t1, seed.t2)
    assert abs(seed.hf_energy - hf) < 1e-10
    assert abs(seed.ccsd_energy - energy) < 1e-10
    assert residual <= 1e-7
    assert abs(seed.residual_max_abs_ha - residual) <= 1e-10
    assert seed.hamiltonian_id == data.context.hamiltonian_id
    assert seed.frame_id == data.context.frame_id
    module.validate_ccsd_seed(data, seed, host_available_mb=4096)
    record_property("residual_ha", residual)
    record_property("energy_error_ha", abs(seed.ccsd_energy - energy))


def test_receiver_checks_actual_arrays_and_energies(module, data, record_property):
    seed = module.build_ccsd_seed(data, host_available_mb=4096)
    for wrong in [
        replace(seed, hf_energy=seed.hf_energy + 0.01),
        replace(seed, ccsd_energy=seed.ccsd_energy + 0.01),
        replace(seed, frame_id="wrong"),
    ]:
        with pytest.raises(ProvenanceMismatchError):
            module.validate_ccsd_seed(data, wrong, host_available_mb=4096)
    changed = data.h1.copy()
    changed[0, 0] += 0.01
    with pytest.raises(ProvenanceMismatchError):
        module.validate_ccsd_seed(replace(data, h1=changed), seed, host_available_mb=4096)
    wrong_t1 = seed.t1.copy()
    wrong_t1[0, 0] += 0.02
    hf, energy, residual = fock_oracle(data, wrong_t1, seed.t2)
    assert residual > 1e-4
    values = module._physical_values(data, wrong_t1, seed.t2)
    assert abs(values[1] - energy) < 1e-10
    assert abs(values[2] - residual) < 1e-10
    record_property("perturbed_residual_ha", residual)
    record_property("perturbed_residual_oracle_error_ha", abs(values[2] - residual))
    record_property("perturbed_energy_oracle_error_ha", abs(values[1] - energy))
    with pytest.raises(CCSDConvergenceError):
        module.validate_ccsd_seed(
            data, replace(seed, t1=wrong_t1, ccsd_energy=energy), host_available_mb=4096
        )


def test_no_convergence_fails_fast(module, data):
    with pytest.raises(CCSDConvergenceError):
        module.build_ccsd_seed(data, host_available_mb=4096, max_cycle=0)


def test_guard_precedes_solver_and_seed_verification(module, data, monkeypatch):
    from pyscf import cc

    seed = module.build_ccsd_seed(data, host_available_mb=4096)
    monkeypatch.setattr(cc, "CCSD", lambda *a, **k: pytest.fail("allocation before guard"))
    with pytest.raises(ResourceLimitError):
        module.build_ccsd_seed(data, host_available_mb=1)
    with pytest.raises(ResourceLimitError):
        module.validate_ccsd_seed(data, seed, host_available_mb=1)


@pytest.mark.sqd
def test_actual_ffsim_connectivity_is_consumed(module, data):
    seed = module.build_ccsd_seed(data, host_available_mb=4096)
    full = module.build_lucj(data, seed, host_available_mb=4096)
    empty = module.build_lucj(
        data, seed, host_available_mb=4096, lucj=LUCJConfig(interaction_pairs=((), ()))
    )
    assert full.n_reps == empty.n_reps == 2
    assert np.max(np.abs(full.diag_coulomb_mats)) > 1e-5
    np.testing.assert_allclose(empty.diag_coulomb_mats, 0, atol=1e-12)


def test_receiver_rejects_t2_symmetry_and_changed_mo(module, data):
    seed = module.build_ccsd_seed(data, host_available_mb=4096)
    mo = data.mo_coeff.copy()
    mo[:, -1] *= -1
    with pytest.raises(ProvenanceMismatchError):
        module.validate_ccsd_seed(replace(data, mo_coeff=mo), seed, host_available_mb=4096)
    if data.norb > 2:
        t2 = seed.t2.copy()
        t2[0, 1, 0, 1] += 0.01
        with pytest.raises(ValueError, match="symmetry"):
            module.validate_ccsd_seed(data, replace(seed, t2=t2), host_available_mb=4096)


def test_full_occupied_seed_is_exact_determinant(module, data):
    if data.norb != 2:
        return
    from q2m3.sqd.integrals import hamiltonian_id

    context = replace(
        data.context,
        hamiltonian_id=hamiltonian_id(data.h1, data.h2, data.e_core, norb=2, nelec=(2, 2)),
    )
    hf = (
        data.e_core
        + 2 * np.trace(data.h1)
        + sum(2 * data.h2[i, i, j, j] - data.h2[i, j, j, i] for i, j in product(range(2), repeat=2))
    )
    filled = replace(data, nelec=(2, 2), context=context, hf_energy=hf)
    seed = module.build_ccsd_seed(filled, host_available_mb=4096)
    assert seed.t1.shape == (2, 0)
    assert seed.ccsd_energy == seed.hf_energy
    assert seed.residual_max_abs_ha == 0
    module.validate_ccsd_seed(filled, seed, host_available_mb=4096)


def test_raw_receiver_supports_low_level_contract_without_molecular_source(module, data):
    seed = module.build_ccsd_seed(data, host_available_mb=4096)
    context = replace(data.context, source={})
    kwargs = dict(
        norb=data.norb, nelec=data.nelec, context=context, seed_data=seed, host_available_mb=4096
    )
    module.validate_ccsd_seed_from_integrals(data.h1, data.h2, data.e_core, **kwargs)
    for field, value in [("frame_id", "wrong"), ("hamiltonian_id", "0" * 64)]:
        with pytest.raises(ProvenanceMismatchError):
            module.validate_ccsd_seed_from_integrals(
                data.h1,
                data.h2,
                data.e_core,
                **{**kwargs, "context": replace(context, **{field: value})},
            )
    with pytest.raises(ResourceLimitError):
        module.validate_ccsd_seed_from_integrals(
            data.h1, data.h2, data.e_core, **{**kwargs, "host_available_mb": 1}
        )


@pytest.mark.parametrize("which,value", [(0, np.nan), (1, np.nan), (1, np.inf)])
def test_residual_nan_is_rejected_before_maximum(module, data, monkeypatch, which, value):
    from pyscf.cc.ccsd import CCSD

    seed = module.build_ccsd_seed(data, host_available_mb=4096)

    def broken(self, t1, t2, eris):
        result = [t1.copy(), t2.copy()]
        result[which].flat[0] = value
        return tuple(result)

    monkeypatch.setattr(CCSD, "update_amps", broken)
    with pytest.raises(CCSDConvergenceError, match="Nonfinite"):
        module.validate_ccsd_seed(data, seed, host_available_mb=4096)


def test_raw_lucj_matches_molecular_operator_without_mo_metadata(module, data):
    seed = module.build_ccsd_seed(data, host_available_mb=4096)
    lucj = LUCJConfig(interaction_pairs=(None, ((0, 0), (0, 1))))
    molecular = module.build_lucj(data, seed, lucj=lucj, host_available_mb=4096)
    raw = module.build_lucj_from_integrals(
        data.h1,
        data.h2,
        data.e_core,
        norb=data.norb,
        nelec=data.nelec,
        context=replace(data.context, source={}),
        seed_data=seed,
        lucj=lucj,
        host_available_mb=4096,
    )
    np.testing.assert_allclose(raw.diag_coulomb_mats, molecular.diag_coulomb_mats, atol=0, rtol=0)
    np.testing.assert_allclose(raw.orbital_rotations, molecular.orbital_rotations, atol=0, rtol=0)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_nonfinite_physical_scalar_never_passes_acceptance(module, data, monkeypatch, which):
    seed = module.build_ccsd_seed(data, host_available_mb=4096)
    values = [seed.hf_energy, seed.ccsd_energy, seed.residual_max_abs_ha]
    values[which] = np.nan
    monkeypatch.setattr(module, "_physical_values", lambda *args: tuple(values))
    with pytest.raises(CCSDConvergenceError, match="Nonfinite"):
        module.validate_ccsd_seed(data, seed, host_available_mb=4096)


def test_nonfinite_doubles_residual_product_cannot_hide_in_max(module, data, monkeypatch):
    seed = module.build_ccsd_seed(data, host_available_mb=4096)
    original = module._solver

    def overflowed_denominator(data):
        solver, eris, d1, d2 = original(data)
        solver.update_amps = lambda t1, t2, eris: (t1.copy(), t2.copy())
        return solver, eris, d1, np.full_like(d2, np.inf)

    monkeypatch.setattr(module, "_solver", overflowed_denominator)
    with np.errstate(invalid="ignore"):
        with pytest.raises(CCSDConvergenceError, match="Nonfinite"):
            module.validate_ccsd_seed(data, seed, host_available_mb=4096)


def test_fully_occupied_contraction_overflow_is_rejected(module, data):
    n = data.norb
    huge = replace(data, h1=np.eye(n) * 1e308, nelec=(n, n))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(CCSDConvergenceError, match="Nonfinite"):
            module._physical_values(huge, np.empty((n, 0)), np.empty((n, n, 0, 0)))
