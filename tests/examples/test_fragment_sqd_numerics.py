"""Real SQD regressions for fixed examples and disconnected fragments."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from examples.fragments import _common, _inputs
from examples.fragments._accuracy import assess_accuracy
from q2m3.constants import HARTREE_TO_KCAL_MOL

pytestmark = [pytest.mark.slow, pytest.mark.sqd]


@pytest.fixture(scope="module", params=["water", "crystal"])
def system(request, tmp_path_factory):
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit_addon_sqd")
    from q2m3.fragments.active_space import check_correspondence, select_active_spaces
    from q2m3.fragments.counterpoise import counterpoise
    from q2m3.fragments.integrals import build_integrals
    from q2m3.fragments.structure import normalize_xh, read_dimer_cif

    if request.param == "water":
        molecule = _inputs.water_input(tmp_path_factory.mktemp("water-input"))
        settings = _inputs.WATER_SETTINGS
    else:
        molecule = normalize_xh(
            read_dimer_cif(_inputs.DATA / "cod_2010678.cif", name=_inputs.CRYSTAL_ID)
        )
        settings = _inputs.CRYSTAL_SETTINGS
    basis, labels, threshold = settings
    subsystems = counterpoise(molecule, basis=basis)
    spaces = select_active_spaces(subsystems, labels, threshold)
    check_correspondence(spaces["AB"], spaces["A"], spaces["B"], subsystems["AB"].scf.get_ovlp())
    integrals = {
        name: build_integrals(
            molecule, sub, *spaces[name], avas_labels=labels, avas_threshold=threshold
        )
        for name, sub in subsystems.items()
    }
    args = _common.parser("test", mode="full").parse_args([])
    return request.param, integrals, _common.calculation_options(args, *settings)["lucj"]


def solve(data, lucj, seed):
    from q2m3.sqd import build_ccsd_seed_from_integrals, run_sqd_from_integrals

    norb = len(data.h1)
    seed_data = build_ccsd_seed_from_integrals(
        data.h1,
        data.h2,
        data.e_core,
        norb=norb,
        nelec=data.nelec,
        context=data.context,
        host_available_mb=6000,
    )
    result = run_sqd_from_integrals(
        data.h1,
        data.h2,
        data.e_core,
        norb=norb,
        nelec=data.nelec,
        context=data.context,
        seed_data=seed_data,
        lucj=lucj,
        seed=seed,
    )
    result.validate()
    assert result.baseline_tier == "T0"
    return result


@pytest.mark.parametrize("seed", range(5))
def test_full_factor_examples_across_seeds(system, seed, record_property):
    name, integrals, lucj = system
    results = {key: solve(value, lucj, seed) for key, value in integrals.items()}
    subsystems = {
        key: SimpleNamespace(e_sqd=value.sqd_energy, e_casci=value.baseline_energy, sqd=value)
        for key, value in results.items()
    }
    delta_e = {
        method: getattr(subsystems["AB"], f"e_{method}")
        - getattr(subsystems["A"], f"e_{method}")
        - getattr(subsystems["B"], f"e_{method}")
        for method in ("sqd", "casci")
    }
    evidence = assess_accuracy(SimpleNamespace(subsystems=subsystems, delta_e=delta_e))
    record_property(f"{name}_seed_{seed}_accuracy", evidence)
    assert evidence["passed"], evidence
    for result in results.values():
        assert result.n_reps == lucj.n_reps
        assert result.diagnostics["resources"]["peak_rss_mb"] < 8192


def test_disconnected_fragment_additivity(system, record_property):
    name, integrals, lucj = system
    if name != "crystal":
        pytest.skip("Use the crystal monomer that exposed the truncation bias")
    from q2m3.sqd.integrals import hamiltonian_id

    mono = integrals["A"]
    n0, occupied = len(mono.h1), mono.nelec[0]
    n, nelec = 2 * n0, (2 * occupied, 2 * occupied)
    virtual = n0 - occupied
    blocks = [
        list(range(occupied)) + list(range(2 * occupied, 2 * occupied + virtual)),
        list(range(occupied, 2 * occupied)) + list(range(2 * occupied + virtual, n)),
    ]
    h1, h2 = np.zeros((n, n)), np.zeros((n,) * 4)
    for block in blocks:
        h1[np.ix_(block, block)] = mono.h1
        h2[np.ix_(block, block, block, block)] = mono.h2
    e_core = 2 * mono.e_core
    context = replace(
        mono.context,
        frame_id="disconnected-crystal-A-product",
        hamiltonian_id=hamiltonian_id(h1, h2, e_core, norb=n, nelec=nelec),
        active_indices=tuple(range(n)),
        n_core_orbitals=0,
        vacuum_core_constant=e_core,
        source={"test": "Two monomers with exactly zero cross-fragment integrals"},
    )
    product = SimpleNamespace(h1=h1, h2=h2, e_core=e_core, nelec=nelec, context=context)
    a, ab = solve(mono, lucj, 0), solve(product, lucj, 0)
    assert abs(ab.baseline_energy - 2 * a.baseline_energy) < 1e-8
    delta = (ab.sqd_energy - 2 * a.sqd_energy) * HARTREE_TO_KCAL_MOL
    record_property("disconnected_interaction_kcal_mol", delta)
    assert abs(delta) < 0.05


def test_factor_capacity_reconstructs_ccsd_doubles(system, record_property):
    import ffsim
    from ffsim.linalg.double_factorized_decomposition import reconstruct_t2

    from q2m3.sqd import build_ccsd_seed_from_integrals

    system_name, integrals, lucj = system
    for name, data in integrals.items():
        seed = build_ccsd_seed_from_integrals(
            data.h1,
            data.h2,
            data.e_core,
            norb=len(data.h1),
            nelec=data.nelec,
            context=data.context,
            host_available_mb=6000,
        )
        mats, rotations = ffsim.linalg.double_factorized_t2(
            seed.t2, max_terms=lucj.n_reps, optimize=False
        )
        reconstructed = reconstruct_t2(mats, rotations, data.nelec[0])
        relative_error = np.linalg.norm(reconstructed - seed.t2) / np.linalg.norm(seed.t2)
        record_property(f"{system_name}_{name}_doubles_relative_error", float(relative_error))
        assert relative_error < 1e-8
