"""Four-arm observables: independent arithmetic and real non-full spaces."""

import importlib
from dataclasses import replace

import numpy as np
import pytest

from q2m3.sqd.exceptions import ComparisonUnavailableError, ReferenceNumericalError

pytestmark = pytest.mark.sqd


def mod():
    return importlib.import_module("q2m3.sqd.observables")


def test_manual_gaps_ratio_and_random_diagnostics():
    out = mod().summarize_observables(
        sqd_energy=-1.95,
        sci_energy=-1.9,
        random_energy=-1.8,
        hf_energy=-1.7,
        baseline_energy=-2.0,
        baseline_tier="T0",
    )
    assert out["delta_mHa"] == pytest.approx(50)
    assert out["delta_vs_sci_mHa"] == pytest.approx(-50)
    assert out["ratio_sqd_over_sci"] == pytest.approx(0.5)
    assert out["diagnostics"]["sampling_quality"] == pytest.approx(1.5)
    assert "sampling_quality" not in {k for k in out if k != "diagnostics"}


@pytest.mark.parametrize("tier", ["T1", "T1+", "T2"])
def test_nonexact_and_degenerate(tier):
    out = mod().summarize_observables(
        sqd_energy=-1.95,
        sci_energy=-1.9,
        random_energy=-1.7,
        hf_energy=-1.7,
        baseline_energy=-2,
        baseline_tier=tier,
    )
    assert out["ratio_sqd_over_sci"] is None
    assert out["null_reasons"]["ratio_sqd_over_sci"] == "non_exact_reference"
    assert out["diagnostics"]["random_degenerate"] is True
    assert out["diagnostics"]["sampling_quality"] is None


def test_ratio_boundaries_and_absence():
    m = mod()
    args = dict(
        sqd_energy=-2.0 - 5e-11,
        sci_energy=-1.9,
        random_energy=None,
        hf_energy=-1.7,
        baseline_energy=-2,
        baseline_tier="T0",
    )
    out = m.summarize_observables(**args, random_reason="budget")
    assert out["ratio_sqd_over_sci"] == 0
    assert out["delta_mHa"] < 0
    assert out["diagnostics"]["ratio_tolerance_ha"] == 1e-10
    assert out["null_reasons"]["iso_ndet_random_energy"] == "budget"
    out = m.summarize_observables(**(args | {"sci_energy": -2.0}), random_reason="budget")
    assert out["ratio_sqd_over_sci"] is None
    with pytest.raises(ReferenceNumericalError):
        m.summarize_observables(**(args | {"sqd_energy": -2.001}), random_reason="budget")
    with pytest.raises(ValueError):
        m.summarize_observables(**args)
    out = m.summarize_observables(
        sqd_energy=None,
        sci_energy=None,
        random_energy=None,
        hf_energy=-1.7,
        baseline_energy=-2,
        baseline_tier="T0",
        mode="reference_only",
    )
    assert out["delta_mHa"] is None
    assert set(out["null_reasons"].values()) == {"reference_only"}


def test_prefix_curve_counts_pairs_not_cartesian():
    samples = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 1, 0]], dtype=bool)
    assert mod().unique_dets_vs_shots(samples, checkpoints=(1, 2, 3)) == (
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 3),
    )
    with pytest.raises(ValueError):
        mod().unique_dets_vs_shots(samples.astype(int))


@pytest.fixture(scope="module")
def four_orbitals():
    from q2m3.molecule import MoleculeConfig
    from q2m3.sqd.ansatz import build_ccsd_seed
    from q2m3.sqd.diagonalize import kernel_fixed_space
    from q2m3.sqd.integrals import build_integrals
    from q2m3.sqd.reference import run_reference

    data = build_integrals(
        MoleculeConfig("h2", ["H", "H"], [[0, 0, 0], [0, 0, 0.9]], 0, 2, 4, basis="6-31g"),
        host_available_mb=4096,
    )
    seed = build_ccsd_seed(data, host_available_mb=4096)
    ref = run_reference(
        data.h1,
        data.h2,
        data.e_core,
        norb=4,
        nelec=(1, 1),
        context=data.context,
        host_available_mb=4096,
    )
    sqd = kernel_fixed_space(
        data.h1,
        data.h2,
        data.e_core,
        (np.array([1, 2]), np.array([1, 2, 4])),
        norb=4,
        nelec=(1, 1),
        host_available_mb=4096,
    )
    samples = np.array(
        [[bool(int(s) & (1 << i)) for i in range(7, -1, -1)] for s in [17, 34, 17, 65]], dtype=bool
    )
    return data, seed, ref, sqd, samples


def run(fixture, **kwargs):
    data, seed, ref, sqd, samples = fixture
    return mod().run_comparisons(
        data.h1,
        data.h2,
        data.e_core,
        norb=4,
        nelec=(1, 1),
        context=data.context,
        seed_data=seed,
        reference=ref,
        sqd=sqd,
        samples=samples,
        seed=7,
        host_available_mb=4096,
        **kwargs,
    )


def test_real_four_arms_nonfull_and_audit(four_orbitals):
    out = run(four_orbitals)
    data, seed, ref, sqd, _ = four_orbitals
    assert out.subspace_dims == (2, 3)
    assert out.subspace_dim == 6 < 16
    assert out.iso_active_space_ccsd_energy == seed.ccsd_energy
    assert out.sqd_energy == sqd.energy
    assert out.delta_mHa == pytest.approx(1000 * (sqd.energy - ref.energy))
    assert out.unique_dets_vs_shots[-1] == (4, 3)
    for arm in ("sci", "random"):
        strings = out.ci_strings[arm]
        assert tuple(map(len, strings)) == (2, 3)
        assert all(1 in sector and len(set(sector)) == len(sector) for sector in strings)
    assert out.diagnostics["sci_selection"]["method"] == "pyscf_selected_ci_marginal_weight"
    assert out.diagnostics["sci_selection"]["cutoff"] == 1e-4
    assert out.comparison_wall_s > 0
    assert out.peak_rss_mb > 0
    assert out.allocation_audit[0]["stage"] == "comparison"
    assert out.allocation_audit[0]["subspace_dims"] == (4, 4)
    assert out.allocation_audit[0]["retained_mb"] > 0
    other = run(four_orbitals)
    assert other.ci_strings == out.ci_strings
    assert other.iso_ndet_random_energy == pytest.approx(out.iso_ndet_random_energy, abs=1e-12)


def test_reject_energy_frame_and_spin_dimension_forgery(four_orbitals):
    data, seed, ref, sqd, samples = four_orbitals
    with pytest.raises(ReferenceNumericalError):
        run((data, seed, ref, replace(sqd, energy=sqd.energy + 0.01), samples))
    with pytest.raises(ValueError):
        run((data, seed, replace(ref, frame_id="vacuum-other-frame"), sqd, samples))
    with pytest.raises(ComparisonUnavailableError):
        run((data, seed, ref, replace(sqd, subspace_dims=(3, 2)), samples))


def test_guard_before_pool_and_timeout(four_orbitals):
    from q2m3.sqd.exceptions import ReferenceTimeoutError, ResourceLimitError

    with pytest.raises(ResourceLimitError):
        run(four_orbitals, rss_budget_mb=1)
    with pytest.raises(ReferenceTimeoutError):
        run(four_orbitals, remaining_wall_s=1e-12)


def test_sci_expansion_and_random_hf_are_deterministic():
    m = mod()
    h1 = np.diag([-2.0, -1.0, 0.0, 1.0])
    h2 = np.zeros((4,) * 4)
    strings, trace = m._select_sci(h1, h2, 4, (1, 1), (3, 2))
    assert tuple(map(len, strings)) == (3, 2)
    assert any(trace["expanded_strings"])
    assert all(1 in sector for sector in strings)
    for seed in range(20):
        strings = m._random_strings(4, (1, 1), (2, 3), seed)
        assert tuple(map(len, strings)) == (2, 3)
        assert all(1 in sector for sector in strings)


def test_random_predicted_budget_is_explicit_null(four_orbitals, monkeypatch):
    monkeypatch.setattr(mod(), "_random_wall_estimate", lambda *args: 1e9)
    out = run(four_orbitals)
    assert out.iso_ndet_sci_energy is not None
    assert out.iso_ndet_random_energy is None
    assert out.ci_strings["random"] is None
    assert out.null_reasons["iso_ndet_random_energy"] == "random_wall_budget"
    assert out.diagnostics["sampling_quality"] is None


def test_independent_fock_energies(four_orbitals):
    from tests.sqd.test_diagonalize import projected_matrix

    data, *_ = four_orbitals
    out = run(four_orbitals)
    for arm, energy in (("sci", out.iso_ndet_sci_energy), ("random", out.iso_ndet_random_energy)):
        matrix = projected_matrix(data.h1, data.h2, data.e_core, out.ci_strings[arm])
        assert np.linalg.eigvalsh(matrix)[0] == pytest.approx(energy, abs=1e-10)


def test_mm_cannot_accept_relabelled_vacuum_seed(four_orbitals):
    from q2m3.molecule import MoleculeConfig
    from q2m3.sqd.exceptions import CCSDConvergenceError, ProvenanceMismatchError
    from q2m3.sqd.integrals import build_integrals

    _, vacuum_seed, ref, sqd, samples = four_orbitals
    data = build_integrals(
        MoleculeConfig("h2", ["H", "H"], [[0, 0, 0], [0, 0, 0.9]], 0, 2, 4, basis="6-31g"),
        host_available_mb=4096,
        mm_charges=np.array([0.2, -0.1]),
        mm_coords=np.array([[2.0, 0.3, 0.1], [-1.0, 0.5, 0.2]]),
        embedding_mode="full_oneelectron",
    )
    seed = replace(
        vacuum_seed, frame_id=data.context.frame_id, hamiltonian_id=data.context.hamiltonian_id
    )
    from q2m3.sqd.diagonalize import kernel_fixed_space
    from q2m3.sqd.reference import run_reference

    reference = run_reference(
        data.h1,
        data.h2,
        data.e_core,
        norb=4,
        nelec=(1, 1),
        context=data.context,
        host_available_mb=4096,
    )
    sqd = kernel_fixed_space(
        data.h1, data.h2, data.e_core, sqd.ci_strings, norb=4, nelec=(1, 1), host_available_mb=4096
    )
    with pytest.raises((CCSDConvergenceError, ProvenanceMismatchError)):
        run((data, seed, reference, sqd, samples))


def test_sci_expansion_cannot_silently_reduce_target():
    with pytest.raises(ComparisonUnavailableError):
        mod()._select_sci(np.diag([-2.0, -1.0, 0.0, 1.0]), np.zeros((4,) * 4), 4, (1, 1), (5, 2))


def test_wrong_sqd_vector_is_rejected(four_orbitals):
    data, seed, ref, sqd, samples = four_orbitals
    bad = np.ones(sqd.subspace_dims) / np.sqrt(sqd.amplitudes.size)
    with pytest.raises(ReferenceNumericalError):
        run((data, seed, ref, replace(sqd, amplitudes=bad), samples))


def test_nonfinite_derived_values_and_sample_domain(four_orbitals):
    with pytest.raises(ReferenceNumericalError):
        mod().summarize_observables(
            sqd_energy=1e308,
            sci_energy=1.0,
            random_energy=2.0,
            hf_energy=3.0,
            baseline_energy=-1e308,
            baseline_tier="T0",
        )
    from q2m3.sqd.exceptions import ResourceModelDomainError

    data, seed, ref, sqd, samples = four_orbitals
    too_many = np.tile(samples[0], (100001, 1))
    with pytest.raises(ResourceModelDomainError):
        run((data, seed, ref, sqd, too_many))
