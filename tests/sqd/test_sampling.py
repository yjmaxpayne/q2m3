"""Sampling integrity and independent determinant/bit-order oracles."""

from unittest.mock import Mock

import numpy as np
import pytest

pytestmark = pytest.mark.sqd


def test_h2_prepare_and_seeded_samples():
    import ffsim

    from q2m3.sqd.sampling import FfsimSampler

    operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(np.full((1, 1, 1, 1), 0.2), n_reps=2)
    sampler = FfsimSampler(host_available_mb=4096, rss_budget_mb=4096)
    state = sampler.prepare(operator, 2, (1, 1))
    assert state.shape == (4,)
    assert np.linalg.norm(state) == pytest.approx(1, abs=1e-12)
    first = sampler.sample(state, 2, (1, 1), shots=4000, seed=17)
    second = sampler.sample(state, 2, (1, 1), shots=4000, seed=17)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == bool
    assert first.shape == (4000, 4)
    for seed in (18, 19):
        samples = sampler.sample(state, 2, (1, 1), shots=4000, seed=seed)
        # Each spin sector has a single electron: its orbital is encoded big endian.
        addresses = (samples[:, 2] * 2 + samples[:, 0]).astype(int)
        empirical = np.bincount(addresses, minlength=4) / len(samples)
        np.testing.assert_allclose(empirical, np.abs(state) ** 2, atol=0.04)


def test_asymmetric_determinant_round_trip_big_endian():
    import ffsim
    from pyscf.fci import cistring

    from q2m3.sqd.sampling import FfsimSampler

    # Independent CI address from PySCF, not ffsim's sample conversion helper.
    norb, count = 5, 2
    alpha, beta = (0, 3), (1, 4)
    a = cistring.str2addr(norb, count, sum(1 << i for i in alpha))
    b = cistring.str2addr(norb, count, sum(1 << i for i in beta))
    state = np.zeros(100, dtype=complex)
    state[a * 10 + b] = 1
    sampler = FfsimSampler(host_available_mb=4096, rss_budget_mb=4096)
    samples = sampler.sample(state, norb, (count, count), shots=31, seed=25)
    expected = np.array([int(bit) for bit in "1001001001"], dtype=bool)
    np.testing.assert_array_equal(samples, np.tile(expected, (31, 1)))
    assert not np.array_equal(expected, np.r_[expected[5:], expected[:5]])
    assert not np.array_equal(expected, np.r_[expected[:5][::-1], expected[5:][::-1]])
    np.testing.assert_array_equal(
        ffsim.strings_to_addresses(samples, norb, (count, count)),
        np.full(31, a * 10 + b),
    )


@pytest.mark.parametrize("defect", ["nan", "infinity", "unnormalized", "shape"])
def test_prepare_rejects_illegal_state_before_diagonalization(monkeypatch, defect):
    import ffsim

    from q2m3.sqd.exceptions import SamplingIntegrityError
    from q2m3.sqd.sampling import FfsimSampler

    operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(np.zeros((1, 1, 1, 1)), n_reps=2)
    bad = np.array([1, 0, 0, 0], dtype=complex)
    if defect == "shape":
        bad = bad[:3]
    else:
        bad[0] = {"nan": np.nan, "infinity": np.inf, "unnormalized": 2}[defect]
    monkeypatch.setattr(ffsim, "apply_unitary", lambda *args, **kwargs: bad)
    diagonalize = Mock()
    with pytest.raises(SamplingIntegrityError, match="backend=ffsim.*stage=prepare"):
        state = FfsimSampler(host_available_mb=4096, rss_budget_mb=4096).prepare(
            operator, 2, (1, 1)
        )
        diagonalize(state)
    diagonalize.assert_not_called()


@pytest.mark.parametrize(
    "bad",
    [
        np.empty((0, 4), dtype=bool),
        np.ones((3, 3), dtype=bool),
        np.array([[0, 1, 0, np.nan]] * 3),
        np.array([[0, 1, 0, 1]] * 3),
        np.ones((3, 4), dtype=bool),
        np.array([[1, 1, 0, 0]] * 3, dtype=bool),
        np.array([[0, 1, 0, 1]] * 4, dtype=bool),
    ],
)
def test_backend_output_checked_before_bool_conversion(monkeypatch, bad):
    import ffsim

    from q2m3.sqd.exceptions import SamplingIntegrityError
    from q2m3.sqd.sampling import FfsimSampler

    monkeypatch.setattr(ffsim, "sample_state_vector", lambda *args, **kwargs: bad)
    diagonalize = Mock()
    with pytest.raises(SamplingIntegrityError, match="backend=ffsim.*stage=sample"):
        samples = FfsimSampler(host_available_mb=4096, rss_budget_mb=4096).sample(
            np.array([1, 0, 0, 0]), 2, (1, 1), shots=3, seed=0
        )
        diagonalize(samples)
    diagonalize.assert_not_called()


def test_bad_external_state_and_unsupported_spin_rejected():
    from q2m3.sqd.exceptions import SamplingIntegrityError
    from q2m3.sqd.sampling import FfsimSampler

    sampler = FfsimSampler(host_available_mb=4096, rss_budget_mb=4096)
    with pytest.raises(SamplingIntegrityError, match="stage=sample"):
        sampler.sample(np.array([np.nan, 0, 0, 0]), 2, (1, 1), shots=1)
    with pytest.raises(ValueError, match="closed-shell"):
        sampler.sample(np.ones(6) / np.sqrt(6), 3, (1, 2), shots=1)


def test_resource_limits_precede_backend_allocations(monkeypatch):
    import ffsim

    from q2m3.sqd.exceptions import ResourceLimitError, ResourceModelDomainError
    from q2m3.sqd.sampling import FfsimSampler

    operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(np.zeros((1, 1, 1, 1)), n_reps=2)
    allocate = Mock(side_effect=AssertionError("allocation must not happen"))
    monkeypatch.setattr(ffsim, "hartree_fock_state", allocate)
    with pytest.raises(ResourceLimitError):
        FfsimSampler(host_available_mb=1, rss_budget_mb=1).prepare(operator, 2, (1, 1))
    sampler = FfsimSampler(host_available_mb=4096, rss_budget_mb=4096)
    with pytest.raises(ResourceModelDomainError, match="shots"):
        sampler.sample(np.array([1, 0, 0, 0]), 2, (1, 1), shots=100001)
    too_many_reps = ffsim.UCJOpSpinBalanced.from_t_amplitudes(np.zeros((1, 1, 1, 1)), n_reps=3)
    with pytest.raises(ResourceModelDomainError, match="n_reps"):
        sampler.prepare(too_many_reps, 2, (1, 1))
    allocate.assert_not_called()


@pytest.mark.parametrize("block", [0, 1])
def test_each_spin_block_is_validated(block):
    from q2m3.sqd.exceptions import SamplingIntegrityError
    from q2m3.sqd.sampling import validate_samples

    bad = np.array([[False, True, False, True]])
    bad[0, 2 * block] = True
    with pytest.raises(SamplingIntegrityError, match="backend=custom.*stage=consumer.*Hamming"):
        validate_samples(bad, 2, (1, 1), shots=1, backend="custom", stage="consumer")


@pytest.mark.parametrize("stage", ["prepare", "sample"])
def test_backend_numerical_error_has_context(monkeypatch, stage):
    import ffsim

    from q2m3.sqd.exceptions import SamplingIntegrityError
    from q2m3.sqd.sampling import FfsimSampler

    failure = ValueError("native numerical failure")
    sampler = FfsimSampler(host_available_mb=4096, rss_budget_mb=4096)
    operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(np.zeros((1, 1, 1, 1)), n_reps=2)
    monkeypatch.setattr(
        ffsim,
        "apply_unitary" if stage == "prepare" else "sample_state_vector",
        Mock(side_effect=failure),
    )
    with pytest.raises(SamplingIntegrityError, match=f"backend=ffsim stage={stage}") as error:
        if stage == "prepare":
            sampler.prepare(operator, 2, (1, 1))
        else:
            sampler.sample(np.array([1, 0, 0, 0]), 2, (1, 1), shots=2)
    assert error.value.__cause__ is failure


def test_integer_norm_overflow_rejected_before_backend(monkeypatch):
    import ffsim

    from q2m3.sqd.exceptions import SamplingIntegrityError
    from q2m3.sqd.sampling import FfsimSampler

    sample_backend = Mock(side_effect=AssertionError("invalid state reached sampling"))
    monkeypatch.setattr(ffsim, "sample_state_vector", sample_backend)
    state = np.array([2**32, 1, 0, 0], dtype=np.int64)
    with pytest.raises(SamplingIntegrityError, match="backend=ffsim stage=sample.*norm"):
        FfsimSampler(host_available_mb=4096, rss_budget_mb=4096).sample(state, 2, (1, 1), shots=2)
    sample_backend.assert_not_called()


def test_hf_generation_numerical_error_has_context(monkeypatch):
    import ffsim

    from q2m3.sqd.exceptions import SamplingIntegrityError
    from q2m3.sqd.sampling import FfsimSampler

    failure = ValueError("invalid HF state construction")
    operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(np.zeros((1, 1, 1, 1)), n_reps=2)
    monkeypatch.setattr(ffsim, "hartree_fock_state", Mock(side_effect=failure))
    with pytest.raises(SamplingIntegrityError, match="backend=ffsim stage=prepare") as error:
        FfsimSampler(host_available_mb=4096, rss_budget_mb=4096).prepare(operator, 2, (1, 1))
    assert error.value.__cause__ is failure
