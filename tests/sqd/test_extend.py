"""String-level seed truncation and excitation for extended SQD.

The reference sets enumerate legal strings with ``itertools.combinations`` and
measure excitation distance with ``int.bit_count``. They never call the
production sector helper or the excitation function under test.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tracemalloc
import warnings
from importlib.util import resolve_name
from itertools import combinations, product
from math import comb
from pathlib import Path

import numpy as np
import pytest

from q2m3.sqd import extend
from q2m3.sqd.extend import Truncation, excite_strings, extend_seed_space, truncate_strings
from tests.sqd.test_imports import boundary_violations

SRC = Path(__file__).resolve().parents[2] / "src"
TAU = 1e-3


def legal(norb: int, n: int) -> list[int]:
    """All strings with ``n`` occupied orbitals out of ``norb``, ascending."""
    return sorted(sum(1 << i for i in occ) for occ in combinations(range(norb), n))


def distance(a: int, b: int) -> int:
    """Excitation distance between two strings with the same particle count."""
    return (int(a) ^ int(b)).bit_count() // 2


def reference(seeds, norb: int, n: int, order: int) -> list[int]:
    """Every legal string within ``order`` excitations of some seed."""
    return [s for s in legal(norb, n) if any(distance(s, t) <= order for t in seeds)]


def assert_strings(actual: np.ndarray, expected) -> None:
    """Exact ascending int64 string set: value, dtype and order all matter."""
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.int64
    assert np.all(np.diff(actual) > 0)
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.int64))


def strings(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.int64)


# ---------------------------------------------------------------- excitation


@pytest.mark.parametrize(("norb", "n", "count"), [(6, 3, 19), (8, 3, 46), (10, 5, 126)])
def test_hf_seed_double_excitation_counts_match_closed_formula(norb, n, count):
    """A lone HF seed with k = 2 gives 1 + o*v + C(o,2)*C(v,2) strings per spin."""
    o, v = n, norb - n
    assert count == 1 + o * v + comb(o, 2) * comb(v, 2)
    hf = (1 << n) - 1
    out = excite_strings(strings(hf), norb=norb, nelec_spin=n, order=2)
    assert len(out) == count
    assert_strings(out, reference([hf], norb, n, 2))


@pytest.mark.parametrize(("norb", "n", "count"), [(6, 3, 19), (8, 3, 46), (10, 5, 126)])
def test_non_hf_seed_excites_relative_to_the_seed(norb, n, count):
    """Excitations are measured from the seed, not from the HF string."""
    seed = legal(norb, n)[-1]
    hf = (1 << n) - 1
    assert distance(seed, hf) > 2  # the HF-relative shortcut gives another set
    out = excite_strings(strings(seed), norb=norb, nelec_spin=n, order=2)
    assert len(out) == count
    assert all(distance(s, seed) <= 2 for s in out)
    assert_strings(out, reference([seed], norb, n, 2))


@pytest.mark.parametrize("seed_set", ["three-random", "single-non-hf"])
def test_string_level_extension_matches_exhaustive_determinant_projection(seed_set):
    """Per-spin SD extension equals the determinant-level SD extension projected on each spin.

    The determinant reference walks all legal (alpha, beta) pairs and keeps those
    within total distance 2 of a seed determinant. Seeds are symmetric (alpha =
    beta), as required for the D8 union to coincide with the projection. Three
    seeds fill the whole (6e,6o) sector; one non-HF seed leaves one string out.
    """
    norb, n = 6, 3
    pool = legal(norb, n)
    if seed_set == "three-random":
        picks = np.random.default_rng(7).choice(len(pool), 3, replace=False)
        seeds = [pool[i] for i in picks]
    else:
        seeds = [pool[-1]]
    seed_dets = [(s, s) for s in seeds]
    kept = [
        (a, b)
        for a, b in product(pool, pool)
        if any(distance(a, sa) + distance(b, sb) <= 2 for sa, sb in seed_dets)
    ]
    projected_alpha = sorted({a for a, _ in kept})
    projected_beta = sorted({b for _, b in kept})
    assert projected_alpha != reference(seeds, norb, n, 1)  # singles-only differs
    if seed_set == "single-non-hf":
        assert len(projected_alpha) == len(pool) - 1  # the whole sector differs
    alpha, beta = extend_seed_space(
        (strings(*seeds), strings(*seeds)), norb=norb, nelec=(n, n), order=2
    )
    assert_strings(alpha, projected_alpha)
    assert_strings(beta, projected_beta)


def test_excitation_order_selects_single_and_triple_counts():
    """k = 1 and k = 3 give 16 and 56 strings for a lone HF seed in (6e,8o)."""
    hf = 0b111
    single = excite_strings(strings(hf), norb=8, nelec_spin=3, order=1)
    triple = excite_strings(strings(hf), norb=8, nelec_spin=3, order=3)
    assert (len(single), len(triple)) == (16, 56)
    assert_strings(single, reference([hf], 8, 3, 1))
    assert_strings(triple, reference([hf], 8, 3, 3))


def test_extension_rejects_empty_seeds_and_unsupported_orders():
    """Empty seeds and orders outside {1, 2, 3} fail before any work."""
    empty = np.array([], dtype=np.int64)
    hf = strings(0b111)
    with pytest.raises(ValueError, match="nonempty"):
        excite_strings(empty, norb=6, nelec_spin=3, order=2)
    with pytest.raises(ValueError, match="nonempty"):
        extend_seed_space((empty, empty), norb=6, nelec=(3, 3), order=2)
    for order in (0, 4):
        with pytest.raises(ValueError, match="order"):
            excite_strings(hf, norb=6, nelec_spin=3, order=order)
        with pytest.raises(ValueError, match="order"):
            extend_seed_space((hf, hf), norb=6, nelec=(3, 3), order=order)


@pytest.mark.parametrize(
    ("seed", "error"),
    [
        (np.array([7.0, 11.0]), "integer"),
        (np.int64(7), "one-dimensional"),
        (strings(7, 11).reshape(1, 2), "one-dimensional"),
        (strings(7, -7), "range"),
        (strings(7, 0b1000011), "range"),
        (strings(7, 0b1), "particle"),
    ],
    ids=["float", "zero-dim", "two-dim", "negative", "beyond-norb", "wrong-count"],
)
def test_excitation_rejects_illegal_seed_strings(seed, error):
    """Each input breaks exactly one seed rule: dtype, shape, sign, orbital range, or count.

    A legal seed precedes the illegal one, so a check on the first element only fails.
    """
    with pytest.raises(ValueError, match=error):
        excite_strings(seed, norb=6, nelec_spin=3, order=2)


def test_excitation_peak_temporary_memory_stays_chunked():
    """All 3432 seeds of (14e,14o) stay within the chunked temporary budget.

    An unchunked distance matrix needs about 94 MB for the XOR array alone.
    """
    seeds = np.array(legal(14, 7), dtype=np.int64)
    tracemalloc.start()
    try:
        out = excite_strings(seeds, norb=14, nelec_spin=7, order=2)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert len(out) == 3432
    assert peak < 12_000_000


def test_excitation_covers_seeds_in_every_chunk(monkeypatch):
    """A short last chunk still contributes its seed's excitations."""
    norb, n = 8, 3
    pool = legal(norb, n)
    monkeypatch.setattr(extend, "_CHUNK_BYTES", len(pool) * 8 * 2)  # two seeds per chunk
    picks = np.random.default_rng(3).choice(len(pool), 5, replace=False)
    seeds = [pool[i] for i in picks]
    assert set(reference(seeds[-1:], norb, n, 1)) - set(reference(seeds[:-1], norb, n, 1))
    out = excite_strings(strings(*seeds), norb=norb, nelec_spin=n, order=1)
    assert_strings(out, reference(seeds, norb, n, 1))


# ---------------------------------------------------------------- D8 union


def test_extension_takes_the_union_of_alpha_and_beta_extensions():
    """Different alpha and beta seeds give one shared union on both spins (D8)."""
    norb, n = 6, 3
    first, last = legal(norb, n)[0], legal(norb, n)[-1]
    union = sorted(set(reference([first], norb, n, 1)) | set(reference([last], norb, n, 1)))
    assert union != reference([first], norb, n, 1)
    alpha, beta = extend_seed_space(
        (strings(first), strings(last)), norb=norb, nelec=(n, n), order=1
    )
    assert_strings(alpha, union)
    assert_strings(beta, union)
    assert not np.shares_memory(alpha, beta)  # a caller may edit one spin safely
    # In (6e,6o) the k = 1 union already fills the sector, so a wider sector checks
    # that the requested order reaches both excitations.
    wide = legal(8, 3)
    seeds = [wide[0], wide[-1]]
    assert reference(seeds, 8, 3, 1) != reference(seeds, 8, 3, 2)
    alpha, beta = extend_seed_space(
        (strings(seeds[0]), strings(seeds[1])), norb=8, nelec=(3, 3), order=1
    )
    assert_strings(alpha, reference(seeds, 8, 3, 1))
    assert_strings(beta, reference(seeds, 8, 3, 1))


def test_extension_rejects_open_shell_populations():
    """Only closed-shell populations are supported."""
    with pytest.raises(ValueError, match="closed-shell"):
        extend_seed_space((strings(0b111), strings(0b11)), norb=6, nelec=(3, 2), order=1)


# ---------------------------------------------------------------- truncation


def sector_4o2e() -> np.ndarray:
    return strings(*legal(4, 2))  # [3, 5, 6, 9, 10, 12]; HF = 3 at index 0


def fixture_row_maximum() -> np.ndarray:
    """Symmetric 6x6 amplitudes. Row 0 = HF; s1 = row 1; s2 = row 2; s3 = row 3."""
    m = np.zeros((6, 6))
    m[0, 0] = 0.9
    m[1, 1] = 0.8 * TAU
    m[1, 5] = m[5, 1] = 0.8 * TAU  # s1: exactly two entries of 0.8 tau
    m[2, 2] = TAU  # s2: one entry of tau
    m[3, 3] = TAU * (1 - 1e-9)  # s3: one entry just below tau
    return m


def test_truncation_keeps_all_at_zero_and_only_hf_above_all_amplitudes():
    """tau = 0 drops nothing (even all-zero rows); tau above all |c| keeps HF and warns."""
    ci = (sector_4o2e(), sector_4o2e())
    m = fixture_row_maximum()
    assert not m[4].any()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        everything = truncate_strings(ci, m, 0.0, nelec=(2, 2))
    assert isinstance(everything, Truncation)
    assert_strings(everything.ci_strings[0], legal(4, 2))
    assert_strings(everything.ci_strings[1], legal(4, 2))
    assert everything.only_hf is False
    with pytest.warns(RuntimeWarning, match="only the HF"):
        hf_only = truncate_strings(ci, m, 10.0, nelec=(2, 2))
    assert_strings(hf_only.ci_strings[0], [3])
    assert_strings(hf_only.ci_strings[1], [3])
    assert hf_only.only_hf is True


def test_truncation_uses_row_maximum_and_inclusive_threshold():
    """s1 (two 0.8 tau) is dropped, s2 (= tau) kept, s3 (just below tau) dropped."""
    m = fixture_row_maximum()
    assert np.linalg.norm(m[1]) >= TAU  # an L2 rule would keep s1
    assert m[3, 3] >= 0.9 * TAU  # a 0.9 tau rule would keep s3
    kept = truncate_strings((sector_4o2e(), sector_4o2e()), m, TAU, nelec=(2, 2))
    assert_strings(kept.ci_strings[0], [3, 6])
    assert_strings(kept.ci_strings[1], [3, 6])
    assert kept.only_hf is False


def test_truncation_does_not_sum_marginal_weights():
    """A row with four entries of 0.6 tau is dropped although sum |c|^2 >= tau^2."""
    m = np.zeros((6, 6))
    m[0, 0] = 0.9
    for j in (1, 2, 3, 5):
        m[4, j] = m[j, 4] = 0.6 * TAU
    assert np.sum(m[4] ** 2) >= TAU**2
    with pytest.warns(RuntimeWarning, match="only the HF"):
        kept = truncate_strings((sector_4o2e(), sector_4o2e()), m, TAU, nelec=(2, 2))
    assert_strings(kept.ci_strings[0], [3])


@pytest.mark.parametrize("threshold", [-1.0, float("nan"), float("inf")])
def test_truncation_rejects_negative_or_nonfinite_thresholds(threshold):
    """tau must be finite and nonnegative."""
    with pytest.raises(ValueError, match="threshold"):
        truncate_strings(
            (sector_4o2e(), sector_4o2e()), fixture_row_maximum(), threshold, nelec=(2, 2)
        )


def test_truncation_reads_alpha_rows_and_beta_columns():
    """alpha uses row maxima, beta column maxima; only_hf needs both spins at HF only."""
    ci = (strings(3, 5, 6), strings(3, 5, 6))
    m = np.array([[0.9, 0.0, 0.3], [0.05, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kept = truncate_strings(ci, m, 0.1, nelec=(2, 2))
    assert_strings(kept.ci_strings[0], [3])
    assert_strings(kept.ci_strings[1], [3, 6])
    assert kept.only_hf is False


def test_truncation_adds_hf_string_missing_from_the_subspace():
    """The HF string is always in the output, even when the input lacks it."""
    ci = (strings(5, 6), strings(5, 6))
    m = np.array([[0.9, 0.0], [0.0, 0.05]])
    kept = truncate_strings(ci, m, 0.1, nelec=(2, 2))
    assert_strings(kept.ci_strings[0], [3, 5])
    assert_strings(kept.ci_strings[1], [3, 5])
    assert kept.only_hf is False


@pytest.mark.parametrize(
    ("ci", "amplitudes", "nelec", "error"),
    [
        ((strings(3, 5, 6), strings(3, 5, 6)), np.eye(3), (1, 1), "particle"),
        ((np.array([3.0, 5.0]), strings(3, 5)), np.eye(2), (2, 2), "integer"),
        ((strings(3, 5).reshape(1, 2), strings(3, 5)), np.eye(2), (2, 2), "one-dimensional"),
        ((np.array([], dtype=np.int64), strings(3)), np.zeros((0, 1)), (2, 2), "nonempty"),
        ((strings(3, -3), strings(3, 5)), np.eye(2), (2, 2), "range"),
        (
            (strings(3, 5, 6), strings(3, 5, 6)),
            np.array([[0.9, np.nan, 0.1], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]]),
            (2, 2),
            "finite",
        ),
    ],
    ids=["wrong-count", "float", "two-dim", "empty", "negative", "nan-amplitude"],
)
def test_truncation_rejects_illegal_strings_and_nonfinite_amplitudes(ci, amplitudes, nelec, error):
    """Illegal strings or a NaN amplitude fail loudly instead of dropping strings silently."""
    with pytest.raises(ValueError, match=error):
        truncate_strings(ci, amplitudes, 0.4, nelec=nelec)


def test_truncation_reads_each_spin_population():
    """With unequal populations, each spin gets its own HF string and particle check."""
    ci = (strings(5, 6), strings(2, 4))
    m = np.array([[0.9, 0.0], [0.0, 0.05]])
    kept = truncate_strings(ci, m, 0.1, nelec=(2, 1))
    assert_strings(kept.ci_strings[0], [3, 5])
    assert_strings(kept.ci_strings[1], [1, 2])
    assert kept.only_hf is False


def test_truncation_rejects_amplitudes_with_mismatched_shape():
    """The amplitude matrix must be (len(alpha), len(beta))."""
    with pytest.raises(ValueError, match="shape"):
        truncate_strings((strings(3, 5, 6), strings(3, 5, 6, 9)), np.eye(3), 0.1, nelec=(2, 2))


# ---------------------------------------------------------------- import boundary


def test_extend_module_imports_only_allowed_sqd_modules():
    """extend.py may use only the diagonalize, resources and observables SQD modules.

    The forbidden-name scan also sees relative imports and literal module paths
    passed to ``importlib.import_module`` or ``__import__``.
    """
    source = Path(extend.__file__).read_text()
    forbidden = ("pennylane", "catalyst", "q2m3.fragments")
    assert not boundary_violations(source, "q2m3.sqd", forbidden)
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(resolve_name("." * node.level + (node.module or ""), "q2m3.sqd"))
    internal = {name for name in names if name.split(".")[0] == "q2m3"}
    assert internal <= {"q2m3.sqd.diagonalize", "q2m3.sqd.resources", "q2m3.sqd.observables"}


def test_importing_sqd_package_does_not_load_extend_module():
    """A fresh ``import q2m3.sqd`` leaves q2m3.sqd.extend unloaded."""
    env = dict(os.environ, PYTHONPATH=str(SRC))
    process = subprocess.run(
        [sys.executable, "-c", "import sys, q2m3.sqd; print('q2m3.sqd.extend' in sys.modules)"],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "False"
