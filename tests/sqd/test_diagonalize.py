"""Independent fermionic projection and guarded recovery contracts."""

from unittest.mock import Mock

import numpy as np
import pytest

pytestmark = pytest.mark.sqd
BUDGET = dict(host_available_mb=4096, rss_budget_mb=4096)


def fixture_space():
    rng = np.random.default_rng(731)
    n = 4
    one = rng.normal(size=(n, n))
    one = (one + one.T) / 3
    factors = rng.normal(size=(3, n, n))
    factors += factors.transpose(0, 2, 1)
    two = np.einsum("xpq,xrs->pqrs", factors, factors) / 19
    strings = (np.array([3, 6, 9]), np.array([3, 5, 10, 12]))
    return one, two, 1.731, strings


def projected_matrix(one, two, constant, strings):
    """Apply second-quantized operators explicitly in alpha-then-beta Fock order."""
    n = len(one)
    basis = [int(a) | (int(b) << n) for a in strings[0] for b in strings[1]]
    rows = {bits: i for i, bits in enumerate(basis)}
    matrix = np.eye(len(basis)) * constant

    def apply(bits, ops):
        sign = 1
        for orbital, create in ops:
            occupied = (bits >> orbital) & 1
            if occupied == create:
                return None, 0
            sign *= (-1) ** ((bits & ((1 << orbital) - 1)).bit_count())
            bits ^= 1 << orbital
        return bits, sign

    for j, bits in enumerate(basis):
        for p, q in np.ndindex(n, n):
            for spin in (0, n):
                out, sign = apply(bits, [(q + spin, False), (p + spin, True)])
                if out in rows:
                    matrix[rows[out], j] += sign * one[p, q]
            for r, s in np.ndindex(n, n):
                for spin in (0, n):
                    for other in (0, n):
                        out, sign = apply(
                            bits,
                            [
                                (q + spin, False),
                                (s + other, False),
                                (r + other, True),
                                (p + spin, True),
                            ],
                        )
                        if out in rows:
                            matrix[rows[out], j] += 0.5 * sign * two[p, q, r, s]
    return matrix


def test_direct_and_addon_match_independent_projection():
    from qiskit_addon_sqd.fermion import solve_sci

    from q2m3.sqd.diagonalize import kernel_fixed_space

    one, two, constant, strings = fixture_space()
    expected, vectors = np.linalg.eigh(projected_matrix(one, two, constant, strings))
    result = kernel_fixed_space(one, two, constant, strings, norb=4, nelec=(2, 2), **BUDGET)
    addon = solve_sci(strings, one, two, 4, (2, 2), max_space=12, max_cycle=100, tol=1e-12)
    assert result.energy == pytest.approx(expected[0], abs=1e-10)
    assert result.electronic_energy + constant == pytest.approx(result.energy, abs=1e-12)
    assert addon.energy + constant == pytest.approx(result.energy, abs=1e-10)
    assert abs(np.vdot(vectors[:, 0], result.amplitudes.ravel())) == pytest.approx(1, abs=1e-10)
    assert result.subspace_dims == (3, 4)
    for actual, wanted in zip(result.ci_strings, strings, strict=False):
        np.testing.assert_array_equal(actual, wanted)


def samples_for(strings, n=4):
    return np.array(
        [
            [c == "1" for c in f"{int(b):0{n}b}{int(a):0{n}b}"]
            for a in strings[0]
            for b in strings[1]
        ],
        dtype=bool,
    )


def test_public_samples_preserve_asymmetric_basis_and_constant():
    from q2m3.sqd.diagonalize import diagonalize_samples

    one, two, constant, strings = fixture_space()
    result = diagonalize_samples(
        one, two, constant, samples_for(strings), norb=4, nelec=(2, 2), max_iterations=1, **BUDGET
    )
    expected = np.linalg.eigvalsh(projected_matrix(one, two, constant, strings))[0]
    assert result.energy == pytest.approx(expected, abs=1e-10)
    assert result.subspace_dims == (3, 4)
    assert result.allocation_audit[-1].subspace_dims == (3, 4)


def test_recovery_consumes_invalid_raw_strings_and_conserves_particles(monkeypatch):
    from qiskit_addon_sqd import fermion

    from q2m3.sqd.diagonalize import _diagonalize_recoverable_samples, diagonalize_samples
    from q2m3.sqd.exceptions import SamplingIntegrityError

    one, two, constant, _ = fixture_space()
    raw = np.zeros((12, 8), dtype=bool)
    raw[:, 0] = True  # Wrong weights in both sectors: cannot survive postselection.
    recovered = []
    real_recover = fermion.recover_configurations

    def capture(*args, **kwargs):
        assert np.any(args[0][:, :4].sum(axis=1) != 2)
        bits, probs = real_recover(*args, **kwargs)
        assert np.all(bits[:, :4].sum(axis=1) == 2)
        assert np.all(bits[:, 4:].sum(axis=1) == 2)
        recovered.append(bits.copy())
        return bits, probs

    monkeypatch.setattr(fermion, "recover_configurations", capture)
    with pytest.raises(SamplingIntegrityError):
        diagonalize_samples(one, two, constant, raw, norb=4, nelec=(2, 2), **BUDGET)
    result = _diagonalize_recoverable_samples(
        one,
        two,
        constant,
        raw,
        norb=4,
        nelec=(2, 2),
        seed=9,
        initial_occupancies=(np.array([0.1, 0.7, 0.9, 0.3]), np.array([0.8, 0.2, 0.6, 0.4])),
        **BUDGET,
    )
    assert recovered
    assert len([a for a in result.allocation_audit if a.iteration >= 0]) == 4
    assert all(int(s).bit_count() == 2 for sector in result.ci_strings for s in sector)
    recovered_alpha = {int("".join(map(str, row[4:].astype(int))), 2) for row in recovered[0]}
    assert any(s != 3 for s in recovered_alpha)
    assert set(result.ci_strings[0]) & (recovered_alpha - {3})


def test_every_batch_is_guarded_before_any_solver(monkeypatch):
    from qiskit_addon_sqd import fermion

    import q2m3.sqd.diagonalize as module
    from q2m3.sqd.exceptions import ResourceLimitError

    one, two, constant, strings = fixture_space()
    solver = Mock(side_effect=AssertionError("Native solver must not run"))
    real_guard = module.guard_allocation
    calls = []

    def guard(*args, **kwargs):
        calls.append(kwargs["subspace_dims"])
        if len(calls) == 3:  # Full-sector bound, then first and second real batch.
            raise ResourceLimitError("second candidate exceeds budget")
        return real_guard(*args, **kwargs)

    monkeypatch.setattr(module, "guard_allocation", guard)
    monkeypatch.setattr(fermion, "solve_sci", solver)
    with pytest.raises(ResourceLimitError, match="second candidate"):
        module.diagonalize_samples(
            one, two, constant, samples_for(strings), norb=4, nelec=(2, 2), **BUDGET
        )
    assert len(calls) == 3
    solver.assert_not_called()


def test_recovery_expansion_upper_bound_rejects_before_preprocessing(monkeypatch, record_property):
    from qiskit_addon_sqd import fermion

    from q2m3.sqd.diagonalize import _diagonalize_recoverable_samples
    from q2m3.sqd.exceptions import ResourceLimitError
    from q2m3.sqd.resources import estimate_rss_mb, load_resource_model

    one, two, constant, _ = fixture_space()
    model = load_resource_model()
    small = estimate_rss_mb(4, (2, 2), stage="diagonalize", subspace_dims=(1, 1), model=model)
    full = estimate_rss_mb(4, (2, 2), stage="diagonalize", subspace_dims=(6, 6), model=model)
    raw = np.zeros((12, 8), dtype=bool)
    occupancies = (np.ones(4) / 2, np.ones(4) / 2)
    actual = _diagonalize_recoverable_samples(
        one,
        two,
        constant,
        raw,
        norb=4,
        nelec=(2, 2),
        initial_occupancies=occupancies,
        seed=0,
        **BUDGET,
    )
    candidates = [entry for entry in actual.allocation_audit if entry.iteration >= 0]
    dimensions = [entry.subspace_dims for entry in candidates]
    actual_peak = max(entry.predicted_rss_mb for entry in candidates)
    record_property("recovered_candidate_dims", dimensions)
    record_property("recovered_candidate_peak_rss_mb", actual_peak)
    assert any(alpha * beta > 1 for alpha, beta in dimensions), dimensions
    assert small < actual_peak <= full
    rejecting_budget = (small + actual_peak) / 2
    assert small < rejecting_budget < actual_peak <= full
    recovery = Mock(side_effect=AssertionError("Preprocessing must not run"))
    solver = Mock(side_effect=AssertionError("Native solver must not run"))
    monkeypatch.setattr(fermion, "recover_configurations", recovery)
    monkeypatch.setattr(fermion, "solve_sci", solver)
    with pytest.raises(ResourceLimitError):
        _diagonalize_recoverable_samples(
            one,
            two,
            constant,
            raw,
            norb=4,
            nelec=(2, 2),
            initial_occupancies=occupancies,
            seed=0,
            host_available_mb=4096,
            rss_budget_mb=rejecting_budget,
        )
    recovery.assert_not_called()
    solver.assert_not_called()


@pytest.mark.parametrize("change", ["empty", "shape", "dtype", "weight"])
def test_invalid_public_samples_fail_before_native_solver(monkeypatch, change):
    from qiskit_addon_sqd import fermion

    from q2m3.sqd.diagonalize import diagonalize_samples

    one, two, constant, strings = fixture_space()
    samples = samples_for(strings)
    samples = {
        "empty": samples[:0],
        "shape": samples[:, :-1],
        "dtype": samples.astype(float),
        "weight": np.zeros_like(samples),
    }[change]
    solver = Mock(side_effect=AssertionError("Native solver must not run"))
    monkeypatch.setattr(fermion, "solve_sci", solver)
    with pytest.raises((ValueError, RuntimeError)):
        diagonalize_samples(one, two, constant, samples, norb=4, nelec=(2, 2), **BUDGET)
    solver.assert_not_called()


@pytest.mark.parametrize("change", ["empty", "shape", "duplicate", "unsorted", "weight", "range"])
def test_invalid_fixed_space_fails_before_native_solver(monkeypatch, change):
    from pyscf.fci import selected_ci

    from q2m3.sqd.diagonalize import kernel_fixed_space

    one, two, constant, strings = fixture_space()
    alpha = {
        "empty": np.array([], dtype=int),
        "shape": np.array([[3]]),
        "duplicate": np.array([3, 3]),
        "unsorted": np.array([6, 3]),
        "weight": np.array([1]),
        "range": np.array([17]),
    }[change]
    solver = Mock(side_effect=AssertionError("Native solver must not run"))
    monkeypatch.setattr(selected_ci, "kernel_fixed_space", solver)
    with pytest.raises(ValueError):
        kernel_fixed_space(one, two, constant, (alpha, strings[1]), norb=4, nelec=(2, 2), **BUDGET)
    solver.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"max_iterations": 3},
        {"num_batches": 3},
        {"seed": -1},
        {"samples_per_batch": 0},
        {"samples_per_batch": 100},
        {"symmetrize_spin": 1},
    ],
)
def test_unsupported_workload_is_rejected(options):
    from q2m3.sqd.diagonalize import diagonalize_samples

    one, two, constant, strings = fixture_space()
    with pytest.raises((ValueError, RuntimeError)):
        diagonalize_samples(
            one, two, constant, samples_for(strings), norb=4, nelec=(2, 2), **options, **BUDGET
        )


@pytest.mark.parametrize(
    "change", ["h1_shape", "h2_shape", "h1_nan", "h2_inf", "core_nan", "h1_asym", "h2_asym"]
)
def test_invalid_integrals_fail_before_native_solver(monkeypatch, change):
    from pyscf.fci import selected_ci

    from q2m3.sqd.diagonalize import kernel_fixed_space

    one, two, constant, strings = fixture_space()
    if change == "h1_shape":
        one = one[:-1]
    elif change == "h2_shape":
        two = two[:-1]
    elif change == "h1_nan":
        one[0, 0] = np.nan
    elif change == "h2_inf":
        two[0, 0, 0, 0] = np.inf
    elif change == "core_nan":
        constant = np.nan
    elif change == "h1_asym":
        one[0, 1] += 1
    else:
        two[0, 1, 2, 3] += 1
    solver = Mock(side_effect=AssertionError("Native solver must not run"))
    monkeypatch.setattr(selected_ci, "kernel_fixed_space", solver)
    with pytest.raises(ValueError):
        kernel_fixed_space(one, two, constant, strings, norb=4, nelec=(2, 2), **BUDGET)
    solver.assert_not_called()


@pytest.mark.parametrize("corruption", ["finite_nonstationary", "nan_energy", "nan_state"])
def test_invalid_addon_state_is_rejected(monkeypatch, corruption):
    from dataclasses import replace

    from qiskit_addon_sqd import fermion

    from q2m3.sqd.diagonalize import diagonalize_samples
    from q2m3.sqd.exceptions import ReferenceNumericalError

    one, two, constant, strings = fixture_space()
    real_solver = fermion.solve_sci

    def corrupt(*args, **kwargs):
        result = real_solver(*args, **kwargs)
        if corruption == "nan_energy":
            return replace(result, energy=np.nan)
        amplitudes = result.sci_state.amplitudes.copy()
        if corruption == "nan_state":
            amplitudes[0, 0] = np.nan
        else:
            amplitudes = np.roll(amplitudes, 1, axis=0)
        return replace(result, sci_state=replace(result.sci_state, amplitudes=amplitudes))

    monkeypatch.setattr(fermion, "solve_sci", corrupt)
    with pytest.raises(ReferenceNumericalError):
        diagonalize_samples(
            one,
            two,
            constant,
            samples_for(strings),
            norb=4,
            nelec=(2, 2),
            max_iterations=1,
            **BUDGET,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_native_buffers_are_canonicalized_after_validation(dtype):
    from q2m3.sqd.diagonalize import kernel_fixed_space

    one, two, constant, strings = fixture_space()
    one = one.astype(dtype)
    two = two.astype(dtype)
    strings = tuple(s.astype(np.int32) for s in strings)
    expected = np.linalg.eigvalsh(projected_matrix(one, two, constant, strings))[0]
    result = kernel_fixed_space(one, two, constant, strings, norb=4, nelec=(2, 2), **BUDGET)
    assert result.energy == pytest.approx(expected, abs=1e-10)
    assert all(sector.dtype == np.dtype(np.int64) for sector in result.ci_strings)
    with pytest.raises(ValueError):
        result.amplitudes.setflags(write=True)


def test_spin_symmetrization_unions_both_sectors_before_guard():
    from q2m3.sqd.diagonalize import diagonalize_samples

    one, two, constant, strings = fixture_space()
    union = np.union1d(*strings)
    expected = np.linalg.eigvalsh(projected_matrix(one, two, constant, (union, union)))[0]
    result = diagonalize_samples(
        one,
        two,
        constant,
        samples_for(strings),
        norb=4,
        nelec=(2, 2),
        max_iterations=1,
        symmetrize_spin=True,
        **BUDGET,
    )
    assert result.energy == pytest.approx(expected, abs=1e-10)
    assert result.subspace_dims == (6, 6)
    assert all(entry.subspace_dims == (6, 6) for entry in result.allocation_audit)
    for sector in result.ci_strings:
        np.testing.assert_array_equal(sector, union)


def test_hf_is_included_when_absent_from_valid_samples():
    from q2m3.sqd.diagonalize import diagonalize_samples

    one, two, constant, strings = fixture_space()
    without_hf = tuple(sector[sector != 3] for sector in strings)
    samples = samples_for(without_hf)
    assert all(3 not in sector for sector in without_hf)
    result = diagonalize_samples(
        one,
        two,
        constant,
        samples,
        norb=4,
        nelec=(2, 2),
        max_iterations=1,
        **BUDGET,
    )
    expected = np.linalg.eigvalsh(projected_matrix(one, two, constant, strings))[0]
    assert result.energy == pytest.approx(expected, abs=1e-10)
    for actual, wanted in zip(result.ci_strings, strings, strict=True):
        np.testing.assert_array_equal(actual, wanted)


def test_intermediate_recovery_expansion_is_guarded_when_best_space_stays_small(
    monkeypatch, record_property
):
    from qiskit_addon_sqd import fermion

    from q2m3.sqd.diagonalize import _diagonalize_recoverable_samples
    from q2m3.sqd.exceptions import ResourceLimitError
    from q2m3.sqd.resources import estimate_rss_mb, load_resource_model

    one = np.zeros((4, 4))
    two = np.zeros((4,) * 4)
    constant = 1.731
    # The second row has a valid alpha determinant 5 and invalid beta weight 1.
    raw = np.array([[c == "1" for c in row] for row in ("00110011", "00010101")])
    actual = _diagonalize_recoverable_samples(
        one, two, constant, raw, norb=4, nelec=(2, 2), seed=0, **BUDGET
    )
    candidates = [entry for entry in actual.allocation_audit if entry.iteration >= 0]
    first = [entry for entry in candidates if entry.iteration == 0]
    recovered = [entry for entry in candidates if entry.iteration == 1]
    assert first and all(entry.subspace_dims == (1, 1) for entry in first)
    assert recovered and max(np.prod(entry.subspace_dims) for entry in recovered) > 1
    assert actual.subspace_dims == (1, 1)
    assert actual.energy == pytest.approx(constant, abs=1e-12)
    actual_peak = max(entry.predicted_rss_mb for entry in candidates)
    small = estimate_rss_mb(
        4, (2, 2), stage="diagonalize", subspace_dims=(1, 1), model=load_resource_model()
    )
    assert actual_peak > small
    record_property("intermediate_candidate_dims", [entry.subspace_dims for entry in candidates])
    record_property("final_best_dims", actual.subspace_dims)
    record_property("intermediate_candidate_peak_rss_mb", actual_peak)
    rejecting_budget = (small + actual_peak) / 2
    recovery = Mock(side_effect=AssertionError("Recovery must not run after budget rejection"))
    solver = Mock(side_effect=AssertionError("Solver must not run after budget rejection"))
    monkeypatch.setattr(fermion, "recover_configurations", recovery)
    monkeypatch.setattr(fermion, "solve_sci", solver)
    with pytest.raises(ResourceLimitError):
        _diagonalize_recoverable_samples(
            one,
            two,
            constant,
            raw,
            norb=4,
            nelec=(2, 2),
            seed=0,
            host_available_mb=4096,
            rss_budget_mb=rejecting_budget,
        )
    recovery.assert_not_called()
    solver.assert_not_called()
