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


COMPARISON_FIELDS = (
    "iso_ndet_sci_energy",
    "iso_ndet_random_energy",
    "delta_vs_sci_mHa",
    "ratio_sqd_over_sci",
    "unique_dets_vs_shots",
)


def assert_null_comparison(out, fixture, reason_part):
    data, seed, ref, sqd, _ = fixture
    assert (out.hamiltonian_id, out.frame_id) == (
        data.context.hamiltonian_id,
        data.context.frame_id,
    )
    assert out.seed == 7
    assert out.ci_strings["sqd"] == tuple(tuple(map(int, s)) for s in sqd.ci_strings)
    for name in COMPARISON_FIELDS:
        assert getattr(out, name) is None
        assert out.null_reasons[name].startswith("comparison_rss_budget: ")
        assert reason_part in out.null_reasons[name]
    assert out.sqd_energy == sqd.energy
    assert out.iso_active_space_ccsd_energy == seed.ccsd_energy
    assert out.delta_mHa == pytest.approx(1000 * (sqd.energy - ref.energy), abs=1e-12)
    assert out.subspace_dims == sqd.subspace_dims == (2, 3)
    assert out.subspace_dim == 6
    assert out.allocation_audit == ()
    assert out.ci_strings["sci"] is None and out.ci_strings["random"] is None


def test_guard_before_pool_and_timeout(four_orbitals):
    from q2m3.sqd.exceptions import ReferenceTimeoutError

    out = run(four_orbitals, rss_budget_mb=1)
    assert_null_comparison(out, four_orbitals, "stage=ccsd")
    with pytest.raises(ReferenceTimeoutError):
        run(four_orbitals, remaining_wall_s=1e-12)


def test_parent_comparison_guard_rejection_returns_null_comparison(four_orbitals, monkeypatch):
    from q2m3.sqd.exceptions import AllocationGuardError

    original = mod().guard_allocation

    def reject_comparison(*args, **kwargs):
        if kwargs["stage"] == "comparison":
            raise AllocationGuardError("predicted=9000.0 MB, stage=comparison")
        return original(*args, **kwargs)

    monkeypatch.setattr(mod(), "guard_allocation", reject_comparison)
    assert_null_comparison(run(four_orbitals), four_orbitals, "stage=comparison")


def test_worker_guard_rejection_returns_null_comparison(four_orbitals, monkeypatch):
    from q2m3.sqd.exceptions import AllocationGuardError

    def reject(*args, **kwargs):
        raise AllocationGuardError("worker guard rejects the candidate space")

    monkeypatch.setattr(mod()._ComparisonWorker, "solve", reject)
    assert_null_comparison(run(four_orbitals), four_orbitals, "worker guard")


def test_supervisor_parent_cap_rejection_returns_null_comparison(four_orbitals, monkeypatch):
    from q2m3.sqd import reference

    # Only the supervisor's parent check sees this RSS; the parent guards use their own import.
    monkeypatch.setattr(reference, "_rss_mb", lambda pid: 1e9)
    out = run(four_orbitals)
    assert_null_comparison(out, four_orbitals, "Reference parent already reaches RSS cap")


@pytest.mark.parametrize("tier,gap", [("T1", -0.01), ("T0", -5e-11)])
def test_budget_rejected_checks_only_exact_baseline_beyond_tolerance(four_orbitals, tier, gap):
    from time import monotonic
    from types import SimpleNamespace

    data, seed, _, sqd, _ = four_orbitals
    reference = SimpleNamespace(tier=tier, energy=sqd.energy - gap)
    out = mod()._budget_rejected(
        sqd, seed, reference, data.context, sqd.subspace_dims, 7, monotonic(), "guard"
    )
    assert out.delta_mHa == pytest.approx(1000 * gap, abs=1e-12)


def test_null_comparison_keeps_variational_check(four_orbitals):
    data, seed, ref, sqd, samples = four_orbitals
    assert ref.tier == "T0"
    below = replace(sqd, energy=ref.energy - 0.01)
    with pytest.raises(ReferenceNumericalError, match="below exact baseline"):
        run((data, seed, ref, below, samples), rss_budget_mb=1)


def test_measured_rss_overrun_in_comparison_still_raises(four_orbitals, monkeypatch):
    from q2m3.sqd.exceptions import AllocationGuardError, ResourceLimitError

    def overrun(*args, **kwargs):
        raise ResourceLimitError("Reference process tree RSS=9000.0 MB >= 8192.0")

    monkeypatch.setattr(mod(), "_supervise", overrun)
    with pytest.raises(ResourceLimitError, match="process tree") as caught:
        run(four_orbitals)
    assert not isinstance(caught.value, AllocationGuardError)


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


def _coupled_tied_input():
    # Dyadic values keep every integral sum exact in binary floating point.
    norb = 6
    h1 = np.diag([-2.0, -1.0, -1.0, -1.0, 0.5, 0.5])
    h1[0, 3] = h1[3, 0] = 0.125
    h2 = np.zeros((norb,) * 4)
    for p in range(norb):
        h2[p, p, p, p] = 0.5
    for q in range(1, norb):
        h2[0, 0, q, q] = h2[q, q, 0, 0] = 0.25
    for idx in ((0, 3, 0, 3), (3, 0, 0, 3), (0, 3, 3, 0), (3, 0, 3, 0)):
        h2[idx] = 0.0625
    # Orbitals 1 and 2: (0q|55) makes the SCI selection bound nonzero, but no
    # determinant with nonzero amplitude occupies orbital 5, so the strings
    # 2 and 4 enter the SCI pool with exactly zero weight (exact tie).
    for q in (1, 2):
        for idx in ((0, q, 5, 5), (q, 0, 5, 5), (5, 5, 0, q), (5, 5, q, 0)):
            h2[idx] = 0.25
    return h1, h2, norb, (1, 1), (3, 5)


def test_select_sci_characterization_input_has_coupling_and_ties():
    """Pin the input premises: without h2 coupling and exact ties across both
    cut points, identical HEAD output cannot discriminate tie-breaking changes."""
    h1, h2, norb, nelec, dims = _coupled_tied_input()
    strings, trace = mod()._select_sci(h1, h2, norb, nelec, dims)
    # Orbital-mixing (0q|55) coupling, not only the diagonal (pp|pp) terms.
    assert h2[0, 1, 5, 5] != 0.0 and h2[0, 2, 5, 5] != 0.0
    # Alpha truncation: strings 2 and 4 tie at exactly 0.0 and compete for 1 slot.
    weights = dict(trace["rankings"][0])
    assert weights[2] == weights[4] == 0.0
    assert sum(w > 0.0 for s, w in weights.items() if s != 1) == dims[0] - 2
    assert (2 in strings[0]) != (4 in strings[0])
    # Beta expansion: strings 16 and 32 have exactly equal diagonal energies.
    assert h1[0, 0] + h1[4, 4] + h2[0, 0, 4, 4] == h1[0, 0] + h1[5, 5] + h2[0, 0, 5, 5]
    assert trace["expanded_strings"][0] == ()
    assert trace["expanded_strings"][1]
    assert (16 in strings[1]) != (32 in strings[1])


def test_select_sci_strings_match_head_on_coupled_tied_input():
    """Pin the selected strings. On an exact tie, the lower bit-string integer
    is kept: string 2 over 4 at alpha truncation, 16 over 32 at beta expansion."""
    strings, _ = mod()._select_sci(*_coupled_tied_input())
    for got, want in zip(strings, ([1, 2, 8], [1, 2, 4, 8, 16]), strict=True):
        assert got.dtype == np.int64
        assert np.array_equal(got, want)


def test_select_sci_trace_matches_head_on_coupled_tied_input():
    """Pin the trace from HEAD exactly, without tolerance, so the split cannot
    reorder tied rankings or change the SCI pool energy."""
    _, trace = mod()._select_sci(*_coupled_tied_input())
    assert trace["rankings"] == (
        ((1, 0.9796054189854604), (8, 0.0203945810145397), (2, 0.0), (4, 0.0)),
        ((1, 0.9796054189854605), (8, 0.020394581014539653), (2, 0.0), (4, 0.0)),
    )
    assert trace["expanded_strings"] == ((), (16,))
    assert trace["pool_electronic_energy"] == -3.537112155077698


def _pruned_pool_input():
    # Dyadic integer draws times 2**-8, 8-fold symmetrized; /8 keeps values dyadic.
    # The rng call order (off first, then h2) fixes the integrals; do not reorder.
    rng = np.random.default_rng(3)
    norb = 6
    h1 = np.diag([-2.0, -1.5, -1.0, 0.5, 1.0, 1.5])
    off = rng.integers(-4, 5, size=(norb, norb)) * 2.0**-8
    h1 = h1 + (off + off.T) / 2
    h2 = rng.integers(-4, 5, size=(norb,) * 4) * 2.0**-8
    h2 = h2 + h2.transpose(1, 0, 2, 3)
    h2 = h2 + h2.transpose(0, 1, 3, 2)
    h2 = (h2 + h2.transpose(2, 3, 0, 1)) / 8
    for p in range(norb):
        h2[p, p, p, p] += 0.5
    return h1, h2, norb, (3, 3), (6, 6)


def test_select_sci_trace_matches_head_on_pruned_pool_input():
    """On the K1 input the SCI pool is the full space, so `cutoff` and `tol` have no
    effect there; this input has a pool smaller than the full space and pins both."""
    strings, trace = mod()._select_sci(*_pruned_pool_input())
    # Premise: C(6, 3) = 20 strings per spin; a smaller pool means cutoff prunes.
    assert all(d < 20 for d in trace["pool_dims"])
    for got in strings:
        assert np.array_equal(got, [7, 11, 13, 14, 21, 38])
    assert trace["pool_dims"] == (18, 18)
    assert trace["rankings"] == (
        (
            (7, 0.9992007543277182),
            (11, 0.00046585716274922775),
            (13, 0.00014971500697290264),
            (14, 6.514542999003772e-05),
            (38, 4.4185171164327853e-05),
            (21, 4.368504971191569e-05),
            (37, 6.384201928269321e-06),
            (22, 5.7664584615078155e-06),
            (19, 5.493472591325182e-06),
            (35, 5.265819859827898e-06),
            (25, 3.5032753861757706e-06),
            (28, 1.5578189004694645e-06),
            (49, 9.304198793177327e-07),
            (26, 8.583511202886355e-07),
            (42, 8.26155735812623e-07),
            (52, 5.1661262457089113e-08),
            (50, 1.730268824636675e-08),
            (44, 2.913879905722822e-09),
        ),
        (
            (7, 0.9992007543277179),
            (11, 0.00046585716274922883),
            (13, 0.00014971500697290264),
            (14, 6.514542999003757e-05),
            (38, 4.4185171164327826e-05),
            (21, 4.3685049711915756e-05),
            (37, 6.384201928269326e-06),
            (22, 5.7664584615078206e-06),
            (19, 5.493472591325183e-06),
            (35, 5.265819859827895e-06),
            (25, 3.503275386175767e-06),
            (28, 1.557818900469466e-06),
            (49, 9.304198793177322e-07),
            (26, 8.583511202886348e-07),
            (42, 8.261557358126234e-07),
            (52, 5.166126245708911e-08),
            (50, 1.7302688246366748e-08),
            (44, 2.9138799057228282e-09),
        ),
    )
    assert trace["expanded_strings"] == ((), ())
    assert isinstance(trace["pool_electronic_energy"], float)
    assert trace["pool_electronic_energy"] == -7.392812805283875
