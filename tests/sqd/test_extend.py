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
from tests.sqd.test_result import ext_arm_data, ext_record, ext_record_parts

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


_ALLOWED_SQD_IMPORTS = {
    "q2m3.sqd.diagonalize",
    "q2m3.sqd.resources",
    "q2m3.sqd.observables",
    "q2m3.sqd.exceptions",
    "q2m3.sqd.config",
}


def _internal_imports(source: str) -> set[str]:
    """Return the q2m3 modules that ``source`` imports, resolved inside ``q2m3.sqd``."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = resolve_name("." * node.level + (node.module or ""), "q2m3.sqd")
            if (SRC / base.replace(".", "/")).is_dir():
                # ``from package import name`` loads the submodule ``package.name``.
                names.update(f"{base}.{alias.name}" for alias in node.names)
            else:
                names.add(base)
    return {name for name in names if name.split(".")[0] == "q2m3"}


def test_extend_module_imports_only_allowed_sqd_modules():
    """extend.py may use only the diagonalize, resources, observables, exceptions and
    config SQD modules.

    The forbidden-name scan also sees relative imports and literal module paths
    passed to ``importlib.import_module`` or ``__import__``.
    """
    source = Path(extend.__file__).read_text()
    forbidden = ("pennylane", "catalyst", "q2m3.fragments")
    assert not boundary_violations(source, "q2m3.sqd", forbidden)
    assert _internal_imports(source) <= _ALLOWED_SQD_IMPORTS


def test_import_scan_records_full_module_for_from_package_import():
    """``from q2m3.sqd import exceptions`` imports the exceptions module.

    Recording only the package name would make the allow-list reject a permitted
    module, and would hide which submodule a ``from package import`` loads.
    """
    assert _internal_imports("from q2m3.sqd import exceptions\n") == {"q2m3.sqd.exceptions"}


def test_import_scan_records_full_name_for_aliased_module_import():
    """``import q2m3.sqd.exceptions as e`` is recorded by module path, not by alias."""
    imported = _internal_imports("import q2m3.sqd.exceptions as e\n")
    assert imported == {"q2m3.sqd.exceptions"}


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


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        pytest.param({"energy": None}, "null_reason", id="energy_missing_without_reason"),
        pytest.param(
            {"null_reason": "ext_numerical: eigensolver failed"},
            "null_reason",
            id="energy_with_null_reason",
        ),
        pytest.param(
            {"sci_baseline_energy": None}, "sci_baseline", id="baseline_missing_without_reason"
        ),
        pytest.param(
            {"sci_baseline_reason": "sci_baseline_arm_null"},
            "sci_baseline",
            id="baseline_with_reason",
        ),
        pytest.param(
            {"energy": None, "null_reason": "bogus: x"},
            "must start with one of",
            id="null_reason_bad_prefix",
        ),
        pytest.param(
            {"energy": None, "null_reason": "ext_rss_budget"},
            "must start with one of",
            id="null_reason_prefix_without_colon",
        ),
    ],
)
def test_ext_arm_validate_rejects_broken_null_pairing(changes, field):
    """Kills RA1-RA5b (in id order): each null value pairs with its reason in both directions,
    and an arm's null reason starts with one of the known ext prefixes."""
    from q2m3.sqd.extend import ExtSQDArm

    data = ext_arm_data() | changes
    with pytest.raises(ValueError, match=field):
        ExtSQDArm(**data).validate()


@pytest.mark.parametrize(
    ("edit", "field"),
    [
        pytest.param(lambda arms: arms[::-1], "seed_source", id="arms_reordered"),
        pytest.param(lambda arms: arms[:2], "seed_source", id="arm_missing"),
        pytest.param(lambda arms: [*arms, ext_arm_data("random")], "seed_source", id="extra_arm"),
        pytest.param(
            lambda arms: [arms[0] | {"energy": None}, *arms[1:]],
            "null_reason",
            id="inner_arm_invalid",
        ),
    ],
)
def test_ext_record_validate_rejects_arms_that_break_its_contract(edit, field):
    """Kills RR1-RR4 (in id order): arms follow the settings order exactly and each is checked."""
    arms = edit(ext_record_parts())
    with pytest.raises(ValueError, match=field):
        ext_record(arms).validate()


def test_ext_record_with_three_arms_passes_validate():
    """Kills P1: a valid three-arm record with a null middle arm passes both validators."""
    record = ext_record()
    assert record.validate() is record
    assert all(arm.validate() is arm for arm in record.arms)


# ---------------------------------------------------------------- run_ext_sqd
# Matrix B2 (S3.2b1, owner S4.2a). run_ext_sqd is called in process, so the recorders
# see its calls. Each test imports run_ext_sqd itself: `import q2m3.sqd` must not load
# the extend module, and a missing name must fail only these tests.

RESOURCES = {"host_available_mb": 1e6, "rss_budget_mb": 9000.0, "retained_mb": 123.0}
E3_ALPHA = (7, 11, 13, 56)
E3_BETA = (7, 14, 19)
# Row maxima [0.9, 0.05, 1e-4, 2e-4]; column maxima [0.9, 5e-4, 5e-4]. At TAU the
# spins keep different sets: alpha {7, 11}, beta {7}.
E3_AMPLITUDES = np.array(
    [[0.9, 5e-4, 5e-4], [0.05, 1e-5, 1e-5], [1e-4, 1e-5, 1e-5], [2e-4, 1e-5, 1e-5]]
)


@pytest.fixture(scope="module")
def h6_chain():
    """Linear H6, 1.0 Angstrom spacing, STO-3G RHF: norb 6, nelec (3, 3), |e_core| > 1 Ha."""
    from pyscf import ao2mo, gto, scf

    atoms = [("H", (0.0, 0.0, float(i))) for i in range(6)]
    mol = gto.M(atom=atoms, basis="sto-3g", unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    assert mf.converged, "SCF did not converge"
    coeff = mf.mo_coeff
    h1 = coeff.T @ mf.get_hcore() @ coeff
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, coeff), 6)
    return h1, h2, mol.energy_nuc()


def _record(monkeypatch, name: str, real) -> list[dict]:
    """Replace ``extend.<name>`` with a pass-through that records bound arguments and result."""
    import inspect

    signature = inspect.signature(real)
    calls: list[dict] = []

    def recorder(*args, **kwargs):
        call = {"arguments": signature.bind(*args, **kwargs).arguments}
        calls.append(call)
        call["result"] = real(*args, **kwargs)
        return call["result"]

    # raising=False keeps the Red failure on the run_ext_sqd import. A missed patch point
    # still fails: every test that uses a recorder reads or counts its calls.
    monkeypatch.setattr(extend, name, recorder, raising=False)
    return calls


@pytest.fixture
def solver_calls(monkeypatch):
    from q2m3.sqd.diagonalize import kernel_fixed_space

    return _record(monkeypatch, "kernel_fixed_space", kernel_fixed_space)


@pytest.fixture
def guard_calls(monkeypatch):
    from q2m3.sqd.resources import guard_allocation

    return _record(monkeypatch, "guard_allocation", guard_allocation)


def _ext_inputs(alpha, beta, amplitudes, *, threshold: float, order: int):
    from q2m3.sqd.config import ExtSQDSettings

    seed_space = ((strings(*alpha), strings(*beta)), np.asarray(amplitudes, dtype=float))
    return seed_space, ExtSQDSettings(order=order, threshold=threshold).validate()


def _call_ext(run_ext_sqd, h6, seed_space, settings, **overrides):
    h1, h2, e_core = h6
    kwargs = {"seed_source": "quantum", "norb": 6, "nelec": (3, 3), **RESOURCES}
    kwargs |= {"allow_large": False, "audit": [], "deadline": None} | overrides
    return run_ext_sqd(h1, h2, e_core, seed_space, settings, **kwargs)


def _e3_extended() -> list[int]:
    """Brute force: keep row or column max |c| >= TAU plus HF, then add order-1 neighbours."""
    rows, cols = np.abs(E3_AMPLITUDES).max(axis=1), np.abs(E3_AMPLITUDES).max(axis=0)
    kept = {s for s, m in zip(E3_ALPHA, rows, strict=True) if m >= TAU}
    kept |= {s for s, m in zip(E3_BETA, cols, strict=True) if m >= TAU} | {0b000111}
    return reference(kept, 6, 3, 1)


def test_run_ext_sqd_keeps_isolated_seeds_and_does_not_raise_energy(h6_chain, solver_calls):
    """Kills XG15: seeds 7 and 56 are 3 excitations apart, so an order-1 extension that
    starts from HF only, or that drops order 0, loses 56 (or 7 and 56). The energy must
    also respect the variational bound E_ext <= E_SQD (03e:37, O-S1.2)."""
    from q2m3.sqd.diagonalize import kernel_fixed_space
    from q2m3.sqd.extend import run_ext_sqd

    seed_space, settings = _ext_inputs(
        (7, 56), (7, 56), [[0.9, 0.1], [0.1, 0.05]], threshold=0.0, order=1
    )
    arm = _call_ext(run_ext_sqd, h6_chain, seed_space, settings)

    assert len(solver_calls) == 1
    for spin in solver_calls[0]["arguments"]["ci_strings"]:
        assert {7, 56} <= set(spin.tolist())
    h1, h2, e_core = h6_chain
    seeds = strings(7, 56)
    e_sqd = kernel_fixed_space(
        h1, h2, e_core, (seeds, seeds), norb=6, nelec=(3, 3), **RESOURCES
    ).energy
    assert arm.energy <= e_sqd + 1e-10


def test_run_ext_sqd_full_seed_space_matches_pyscf_fci_with_core_energy(h6_chain):
    """Kills XF1 and XF2: with all 20 strings as seeds the solved space is the full CI
    space, so the arm energy is the PySCF FCI total energy, e_core included (03e:38)."""
    from pyscf import fci

    from q2m3.sqd.extend import run_ext_sqd

    h1, h2, e_core = h6_chain
    assert abs(e_core) > 1.0  # premise: losing e_core moves the energy far beyond 1e-8
    every = legal(6, 3)
    seed_space, settings = _ext_inputs(
        every, every, np.full((20, 20), 0.05), threshold=0.0, order=1
    )
    arm = _call_ext(run_ext_sqd, h6_chain, seed_space, settings)

    exact, _ = fci.direct_spin1.kernel(h1, h2, 6, (3, 3), ecore=e_core, tol=1e-13)
    assert abs(arm.energy - exact) <= 1e-8


def test_run_ext_sqd_solves_exactly_the_truncated_and_extended_strings(h6_chain, solver_calls):
    """Kills XF3 (threshold not passed: 56 and its neighbours appear), XF4 (order not
    passed: distance-2 strings appear), XF13 (resource arguments not passed to the solver)
    and XG14 (configuration recovery instead of one fixed-space solve; O-S1.7). XF5
    (alpha/beta swap) is not killed here; S5.2a killed it in
    test_run_ext_sqd_arm_reports_counts_coverage_and_guard_prediction."""
    from q2m3.sqd.extend import run_ext_sqd

    seed_space, settings = _ext_inputs(E3_ALPHA, E3_BETA, E3_AMPLITUDES, threshold=TAU, order=1)
    _call_ext(run_ext_sqd, h6_chain, seed_space, settings)

    assert len(solver_calls) == 1
    arguments = solver_calls[0]["arguments"]
    for spin in arguments["ci_strings"]:
        assert_strings(spin, _e3_extended())
    h1, h2, e_core = h6_chain
    np.testing.assert_array_equal(arguments["h1"], h1)
    np.testing.assert_array_equal(arguments["h2"], h2)
    assert arguments["e_core"] == e_core
    assert (arguments["norb"], arguments["nelec"]) == (6, (3, 3))
    assert {key: arguments[key] for key in RESOURCES} == RESOURCES
    assert arguments["allow_large"] is False


def test_run_ext_sqd_arm_reports_counts_coverage_and_guard_prediction(h6_chain, guard_calls):
    """Kills XF6 (counts in beta-alpha order), XF7 (truncated counts copied from the seeds),
    XF8 (extended counts copied from the truncation), XF9 (coverage over 2**norb) and
    XF10 (predicted RSS not taken from the guard). Also kills XF5 (alpha/beta swap; S5.2a).
    The deadline one hour ahead is the positive control of the time precheck."""
    import time

    from q2m3.sqd.extend import run_ext_sqd

    seed_space, settings = _ext_inputs(E3_ALPHA, E3_BETA, E3_AMPLITUDES, threshold=TAU, order=1)
    arm = _call_ext(run_ext_sqd, h6_chain, seed_space, settings, deadline=time.monotonic() + 3600)

    extended = len(_e3_extended())
    assert len(guard_calls) == 1
    assert arm.seed_source == "quantum"
    assert arm.energy is not None
    assert arm.seed_counts == (4, 3)
    assert arm.truncated_counts == (2, 1)
    assert arm.extended_counts == (extended, extended)
    assert arm.subspace_dims == (extended, extended)
    assert arm.coverage == extended / comb(6, 3)
    assert arm.only_hf is False
    assert arm.seed_space_energy is None
    assert arm.zero_weight_strings is None
    assert arm.predicted_rss_mb == guard_calls[0]["result"]
    assert arm.null_reason is None
    assert arm.wall_s >= 0
    assert arm.process_peak_rss_mb > 0
    assert arm.validate() is arm


EXT_AUDIT_KEYS = {
    "stage",
    "phase",
    "subspace_dims",
    "retained_mb",
    "predicted_rss_mb",
    "budget_mb",
    "model_id",
}


def _ext_model():
    from q2m3.sqd.resources import resource_model_for

    return resource_model_for(6, profile={"max_space": 12})


def _e3_run(run_ext_sqd, h6, **overrides):
    seed_space, settings = _ext_inputs(E3_ALPHA, E3_BETA, E3_AMPLITUDES, threshold=TAU, order=1)
    return _call_ext(run_ext_sqd, h6, seed_space, settings, **overrides)


@pytest.mark.parametrize(
    "caps",
    [
        pytest.param({"rss_budget_mb": 1.0}, id="rss_cap"),
        pytest.param({"host_available_mb": 1.0}, id="host_cap"),
    ],
)
def test_run_ext_sqd_guard_rejects_before_solver(h6_chain, solver_calls, caps):
    """Kills XG4 (no own guard, or a guard after the solve), XG5 (rejection turned into a
    null arm), XG6 (base ResourceLimitError raised), XG7 (guard error re-raised without
    the stage=ext prefix), XG8 (audit only on pass), XG10 (audit stage "diagonalize") and,
    in [host_cap], XG11 (budget_mb copied from rss_budget_mb=9000). O-S4.1; Q-i: the
    audited prediction is estimate_rss_mb with the guard arguments."""
    from q2m3.sqd.exceptions import AllocationGuardError, ResourceLimitError
    from q2m3.sqd.extend import run_ext_sqd
    from q2m3.sqd.resources import estimate_rss_mb

    audit: list[dict] = []
    with pytest.raises(AllocationGuardError) as caught:
        _e3_run(run_ext_sqd, h6_chain, audit=audit, **caps)

    extended = len(_e3_extended())
    dims = (extended, extended)
    predicted = estimate_rss_mb(
        6, (3, 3), stage="diagonalize", subspace_dims=dims, retained_mb=123.0, model=_ext_model()
    )
    assert isinstance(caught.value, ResourceLimitError)
    assert solver_calls == []
    message = str(caught.value)
    assert message.startswith("stage=ext")
    assert str(dims) in message
    assert "predicted=" in message
    assert "budget=1.000000" in message
    assert len(audit) == 1
    record = audit[0]
    assert set(record) == EXT_AUDIT_KEYS
    assert record["stage"] == "ext"
    assert record["budget_mb"] == 1.0
    assert record["predicted_rss_mb"] == predicted
    assert record["predicted_rss_mb"] > record["budget_mb"]
    # XG16 (S7d, G3b F2): the audit keeps the extended dims, the caller's retained_mb
    # (RESOURCES) and the diagonalize model, not values from another source.
    assert tuple(record["subspace_dims"]) == dims
    assert record["retained_mb"] == 123.0
    assert record["model_id"] == _ext_model().model_id


def test_run_ext_sqd_guard_uses_diagonalize_model_on_extended_dims(h6_chain, guard_calls):
    """Kills XF11 (retained_mb dropped), XF12 (truncated instead of extended dims) and
    XG13 in its stage="comparison" form. The form without the max_space profile is
    equivalent here: both rows give the same model_id (S3.2b2 check). O-S4.1."""
    from q2m3.sqd.extend import run_ext_sqd

    _e3_run(run_ext_sqd, h6_chain)

    extended = len(_e3_extended())
    assert len(guard_calls) == 1
    arguments = guard_calls[0]["arguments"]
    assert (arguments["norb"], arguments["nelec"]) == (6, (3, 3))
    assert arguments["stage"] == "diagonalize"
    assert tuple(arguments["subspace_dims"]) == (extended, extended)
    assert arguments["model"].model_id == _ext_model().model_id
    assert arguments.get("retained_mb", 0.0) == 123.0
    assert arguments["host_available_mb"] == 1e6
    assert arguments.get("rss_budget_mb") == 9000.0
    assert arguments.get("allow_large", False) is False


def test_run_ext_sqd_appends_one_ext_audit_record_when_guard_passes(h6_chain):
    """Kills XG9 (audit only on rejection, or the solver allocation_audit appended too),
    XG10 (stage "diagonalize"), XG11 (budget_mb copied from rss_budget_mb=9000) and XG12
    (REJECT_RSS_MB missing from the caps: the minimum is then 9000, not 8192). Q12-3:
    only the audit list is read; no result metadata key is checked."""
    from q2m3.sqd.extend import run_ext_sqd

    earlier = {"stage": "diagonalize", "phase": "0:0"}
    audit = [earlier]
    arm = _e3_run(run_ext_sqd, h6_chain, audit=audit)

    assert audit[0] is earlier
    added = audit[1:]
    assert [record["stage"] for record in added] == ["ext"]
    assert set(added[0]) == EXT_AUDIT_KEYS
    assert added[0]["budget_mb"] == 8192.0
    assert added[0]["predicted_rss_mb"] == arm.predicted_rss_mb


def test_run_ext_sqd_skips_solver_when_deadline_has_passed(h6_chain, solver_calls, guard_calls):
    """Kills XG2 (no precheck: the solver runs) and XG3 (guard before the precheck: the
    guard runs and the audit grows). O-T5c run layer; Q-h: a failed precheck returns a
    null arm and does not raise."""
    import time

    from q2m3.sqd.extend import run_ext_sqd

    audit: list[dict] = []
    arm = _e3_run(run_ext_sqd, h6_chain, audit=audit, deadline=time.monotonic() - 1)

    assert arm.energy is None
    assert arm.null_reason.startswith("ext_wall_budget")
    assert arm.validate() is arm
    assert solver_calls == []
    assert guard_calls == []
    assert audit == []


@pytest.mark.filterwarnings("ignore:Truncation at threshold:RuntimeWarning")
def test_run_ext_sqd_time_precheck_uses_extended_dimension(monkeypatch, guard_calls):
    """Kills XG1 (precheck before the extension, so D is unknown and the 1 s floor
    applies). Q12-4 large-D fixture, recomputed in S3.2b2: HF seed, norb 12, order 3
    gives 662 strings per spin, D = 438,244. For a quantum arm (k = 2) the threshold is
    2 * 2 * 306 s * D / 11,778,624 = 45.54 s; with the floor it is 4 s. A deadline 20 s
    ahead must give a null arm. A wrong implementation reaches the solver stub."""
    import time

    from q2m3.sqd.extend import run_ext_sqd

    class SolverReached(AssertionError):
        pass

    def solver(*args, **kwargs):
        raise SolverReached("the precheck let a 45.5 s arm start with 20 s left")

    monkeypatch.setattr(extend, "kernel_fixed_space", solver, raising=False)
    noise = np.random.default_rng(12).normal(size=(12, 12))
    hamiltonian = ((noise + noise.T) / 2, np.zeros((12,) * 4), 0.0)
    hf = 0b111111
    seed_space, settings = _ext_inputs((hf,), (hf,), [[1.0]], threshold=0.0, order=3)
    audit: list[dict] = []
    arm = _call_ext(
        run_ext_sqd,
        hamiltonian,
        seed_space,
        settings,
        norb=12,
        nelec=(6, 6),
        audit=audit,
        deadline=time.monotonic() + 20,
    )

    assert arm.subspace_dims == (662, 662)  # the extension ran before the precheck
    assert arm.energy is None
    assert arm.null_reason.startswith("ext_wall_budget")
    assert arm.validate() is arm
    assert guard_calls == []
    assert audit == []


def test_run_ext_sqd_reports_only_hf_from_truncation(h6_chain):
    """Kills XF14 (only_hf hard-coded False). Every non-HF amplitude is below TAU on both
    spins, so the arm must report the truncation result only_hf=True; E4 covers False."""
    from q2m3.sqd.extend import run_ext_sqd

    seed_space, settings = _ext_inputs(
        (7, 11), (7, 14), [[0.9, 1e-4], [1e-4, 1e-5]], threshold=TAU, order=1
    )
    with pytest.warns(RuntimeWarning, match="only the HF"):
        arm = _call_ext(run_ext_sqd, h6_chain, seed_space, settings)

    assert arm.only_hf is True
    assert (arm.seed_counts, arm.truncated_counts) == ((2, 2), (1, 1))


def test_run_ext_sqd_does_not_call_diagonalize_samples(h6_chain, monkeypatch):
    """Kills XG14b: ext-SQD solves one fixed space and does no configuration recovery
    (O-S1.7), so a call to diagonalize_samples is an error. Only calls through the module
    attributes are observed."""
    from q2m3.sqd.extend import run_ext_sqd

    class RecoveryCalled(Exception):
        pass

    def forbidden(*args, **kwargs):
        raise RecoveryCalled("run_ext_sqd called diagonalize_samples")

    monkeypatch.setattr("q2m3.sqd.diagonalize.diagonalize_samples", forbidden)
    monkeypatch.setattr("q2m3.sqd.extend.diagonalize_samples", forbidden, raising=False)
    seed_space, settings = _ext_inputs(E3_ALPHA, E3_BETA, E3_AMPLITUDES, threshold=TAU, order=1)
    arm = _call_ext(run_ext_sqd, h6_chain, seed_space, settings)

    assert arm.energy is not None


# ---- seed sources

CISD_POOL = legal(4, 2)  # [3, 5, 6, 9, 10, 12]; HF = 3
RANDOM_ARGS = (8, (4, 3), (5, 4))  # norb, nelec, dims: asymmetric on purpose
RANDOM_SEEDS = (0, 1, 2)


def cisd_select(weights: dict, diagonal: dict, size):
    """Run cisd_seed_strings on CISD_POOL with per-string weights and diagonals."""
    return extend.cisd_seed_strings(
        np.asarray(CISD_POOL, dtype=np.int64),
        np.asarray([weights[s] for s in CISD_POOL], dtype=float),
        np.asarray([diagonal[s] for s in CISD_POOL], dtype=float),
        nelec_spin=2,
        size=size,
    )


def test_hf_seed_strings_hold_only_the_per_spin_hf_string():
    """Kills B2a-1: an unbalanced nelec shows a beta seed built from the alpha HF string."""
    alpha, beta = extend.hf_seed_strings(5, (3, 1))

    assert_strings(alpha, [0b111])
    assert_strings(beta, [0b1])


def test_random_seed_strings_are_legal_ascending_and_contain_hf():
    """Kills B2a-2: the random arm seed must be a legal ascending set of size dims with HF."""
    norb, nelec, dims = RANDOM_ARGS
    strings = extend.random_seed_strings(norb, nelec, dims, 0)

    for spin in range(2):
        actual = strings[spin]
        assert_strings(actual, sorted(actual.tolist()))
        assert len(actual) == dims[spin]
        assert set(actual.tolist()) <= set(legal(norb, nelec[spin]))
        assert (1 << nelec[spin]) - 1 in actual.tolist()


def test_random_seed_strings_repeat_for_one_seed_and_differ_across_seeds():
    """Kills B2a-3 and B2a-4: the random null arm must be reproducible yet depend on seed."""
    norb, nelec, dims = RANDOM_ARGS
    draws = set()
    for seed in RANDOM_SEEDS:
        first = extend.random_seed_strings(norb, nelec, dims, seed)
        second = extend.random_seed_strings(norb, nelec, dims, seed)
        for spin in range(2):
            np.testing.assert_array_equal(first[spin], second[spin])
        draws.add(tuple(tuple(s.tolist()) for s in first))

    assert len(draws) == len(RANDOM_SEEDS)


def test_random_seed_strings_match_the_sqd_random_baseline_draw():
    """Kills B2a-5: 03f fixes the random seed source to the SQD random baseline draw."""
    random_strings = sys.modules["q2m3.sqd.observables"]._random_strings
    for seed in RANDOM_SEEDS:
        actual = extend.random_seed_strings(*RANDOM_ARGS, seed)
        expected = random_strings(*RANDOM_ARGS, seed)
        for spin in range(2):
            assert_strings(actual[spin], np.sort(expected[spin]))


def test_cisd_seed_strings_group_weights_by_single_link_relative_tolerance():
    """Kills B2a-6 to B2a-10: near-equal weights form one single-link group that sorts by
    diagonal (D11), so exact, pairwise, anchored and diagonal-only keys all pick wrong."""
    step = 2.0**-21
    weights = {3: 1.0, 5: 0.5, 9: 0.5 - step, 6: 0.5 - 2 * step, 10: 0.5 - 3 * step, 12: 0.0}
    diagonal = {3: -1.0, 5: 3.0, 6: 4.0, 9: 2.0, 10: 1.0, 12: 0.5}
    strings, _ = cisd_select(weights, diagonal, size=3)

    assert_strings(strings, [3, 9, 10])  # ranked order is [3, 10, 9], so output must be sorted


def test_cisd_seed_strings_tolerance_is_relative_to_the_largest_weight():
    """Kills B2a-11: with max(w) = 2^-10 a gap of 2^-25 splits groups; an absolute 1e-6
    tolerance would merge them and let the lower diagonal win."""
    top = 2.0**-10
    weights = {3: 2.0**-11, 5: top, 6: top - 2.0**-25, 9: 0.0, 10: 0.0, 12: 0.0}
    diagonal = {3: 0.0, 5: 2.0, 6: 1.0, 9: 3.0, 10: 4.0, 12: 5.0}
    strings, _ = cisd_select(weights, diagonal, size=2)

    assert_strings(strings, [3, 5])


def test_cisd_seed_strings_break_equal_weight_and_diagonal_by_integer():
    """Kills B2a-12: the last key is the ascending integer, which makes the seed deterministic."""
    weights = {3: 0.5, 5: 0.125, 6: 0.125, 9: 0.25, 10: 0.25, 12: 0.25}
    strings, _ = cisd_select(weights, dict.fromkeys(CISD_POOL, 1.0), size=3)

    assert_strings(strings, [3, 9, 10])


def test_cisd_seed_strings_always_keep_hf():
    """Kills B2a-13 and B2a-14: HF stays in the seed and the seed size stays equal to size."""
    weights = {3: 0.0625, 5: 0.5, 6: 0.25, 9: 0.125, 10: 0.09375, 12: 0.03125}
    strings, _ = cisd_select(weights, dict.fromkeys(CISD_POOL, 1.0), size=2)

    assert_strings(strings, [3, 5])


def test_cisd_seed_strings_fill_zero_weight_strings_by_diagonal_and_count_them():
    """Kills B2a-15 and B2a-16: zero-weight fill follows the diagonal, and the arm field
    zero_weight_strings counts only the selected zero-weight strings."""
    weights = {3: 0.5, 5: 0.25, 6: 0.0, 9: 0.0, 10: 0.0, 12: 0.0}
    diagonal = {3: 0.0, 5: 0.0, 6: 4.0, 9: 1.0, 10: 2.0, 12: 3.0}
    strings, zero_count = cisd_select(weights, diagonal, size=5)

    assert_strings(strings, [3, 5, 9, 10, 12])
    assert type(zero_count) is int
    assert zero_count == 3


def test_cisd_seed_strings_count_only_exact_zero_weights():
    """zero_weight_strings is an arm field, so a tiny positive weight must not count as zero."""
    weights = {3: 0.5, 5: 2.0**-40, 6: 0.0, 9: 0.0, 10: 0.0, 12: 0.0}
    diagonal = {3: 0.0, 5: 0.5, 6: 4.0, 9: 1.0, 10: 2.0, 12: 3.0}
    strings, zero_count = cisd_select(weights, diagonal, size=3)

    assert_strings(strings, [3, 5, 9])
    assert zero_count == 1


def test_cisd_seed_strings_return_whole_pool_when_size_exceeds_it():
    """A size above the pool returns every string, so callers must not assume len == size."""
    weights = {3: 0.5, 5: 0.25, 6: 0.125, 9: 0.0, 10: 0.0, 12: 0.0}
    strings, zero_count = cisd_select(weights, dict.fromkeys(CISD_POOL, 1.0), size=8)

    assert_strings(strings, CISD_POOL)
    assert zero_count == 3


def test_cisd_seed_strings_reject_misaligned_weights():
    """Kills B2a-17: a silent zip truncation would rank strings against the wrong weights."""
    pool = np.asarray(CISD_POOL, dtype=np.int64)
    with pytest.raises(ValueError, match="must align with the CI strings"):
        extend.cisd_seed_strings(
            pool, np.full(len(pool) - 1, 0.25), np.zeros(len(pool)), nelec_spin=2, size=2
        )


@pytest.mark.parametrize("size", [0, -1])
def test_cisd_seed_strings_reject_size_below_one(size):
    """Kills S7a2 S2: ``[: size - 1]`` wraps around below 1 and silently returns extra strings."""
    weights = dict(zip(CISD_POOL, (0.9, 0.5, 0.4, 0.3, 0.2, 0.1), strict=True))
    with pytest.raises(ValueError, match="size must be at least 1"):
        cisd_select(weights, dict.fromkeys(CISD_POOL, 0.0), size)


# ---- cisd seeds (B2b group 1)


@pytest.fixture(scope="module")
def n2_sym():
    """N2 at 1.10 Angstrom, STO-3G RHF with point-group symmetry: norb 10, nelec (7, 7)."""
    from pyscf import ao2mo, gto, scf

    atoms = [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.10))]
    mol = gto.M(atom=atoms, basis="sto-3g", unit="Angstrom", symmetry=True, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    assert mf.converged, "SCF did not converge"
    coeff = mf.mo_coeff
    h1 = coeff.T @ mf.get_hcore() @ coeff
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, coeff), 10)
    return h1, h2, mol.energy_nuc()


def _cisd_marginals(h1, h2, e_core, *, norb, nelec):
    """Per spin: (string pool, CISD marginal weights, diagonal energies on the HF row/column)."""
    from pyscf.fci import cistring, direct_spin1

    _, vec = extend.cisd_state(h1, h2, e_core, norb=norb, nelec=nelec)
    hdiag = direct_spin1.make_hdiag(h1, h2, norb, nelec).reshape(vec.shape)
    pools = [cistring.make_strings(range(norb), n) for n in nelec]
    weights = ((vec**2).sum(axis=1), (vec**2).sum(axis=0))
    return list(zip(pools, weights, (hdiag[:, 0], hdiag[0, :]), strict=True))


def _cisd_expected(marginals, nelec, dims):
    """Per spin: the (strings, zero_count) that cisd_seed_strings selects from the marginals."""
    return [
        extend.cisd_seed_strings(pool, w, e, nelec_spin=n, size=d)
        for (pool, w, e), n, d in zip(marginals, nelec, dims, strict=True)
    ]


def _seed(source, system, *, norb, nelec, dims):
    h1, h2, e_core = system
    return extend.build_seed_space(
        source,
        h1,
        h2,
        e_core,
        norb=norb,
        nelec=nelec,
        dims=dims,
        seed=0,
        audit=[],
        resources=dict(RESOURCES),
    )


def test_cisd_state_matches_independent_sd_diagonalization(h6_chain):
    """O-T6/K9: the CISD seed must be the exact ground state of the singles-and-doubles space."""
    from pyscf.fci import cistring, direct_spin1

    h1, h2, e_core = h6_chain
    norb, nelec = 6, (3, 3)
    pool = cistring.make_strings(range(norb), 3)
    h2e = direct_spin1.absorb_h1e(h1, h2, norb, nelec, 0.5)
    columns = [
        direct_spin1.contract_2e(h2e, unit.reshape(len(pool), -1), norb, nelec).ravel()
        for unit in np.eye(len(pool) ** 2)
    ]
    ham = np.column_stack(columns)
    order = np.array([bin(int(s) ^ int(pool[0])).count("1") // 2 for s in pool])
    sd = (order[:, None] + order[None, :]).ravel() <= 2
    e_sd = np.linalg.eigvalsh(ham[np.ix_(sd, sd)])[0] + e_core
    e_hf = ham[0, 0] + e_core
    assert abs(e_sd - e_hf) > 1e-6

    e_cisd, _ = extend.cisd_state(h1, h2, e_core, norb=norb, nelec=nelec)

    assert abs(e_cisd - e_sd) <= 1e-10
    assert abs(e_cisd - e_hf) > 1e-6


def test_build_seed_space_cisd_respects_unequal_spin_dims(h6_chain):
    """K7: alpha and beta seed sizes are independent, so dims must not be swapped or shared."""
    norb, nelec, dims = 6, (3, 3), (5, 3)
    marginals = _cisd_marginals(*h6_chain, norb=norb, nelec=nelec)
    expected = _cisd_expected(marginals, nelec, dims)
    # h6 is closed shell, so alpha and beta marginals match: only the per-spin length
    # detects swapped or shared dims.

    ((alpha, beta), _), _, _ = _seed("cisd", h6_chain, norb=norb, nelec=nelec, dims=dims)

    for got, (want, _), size in zip((alpha, beta), expected, dims, strict=True):
        assert len(got) == size and 7 in got.tolist()
        np.testing.assert_array_equal(got, want)


def test_build_seed_space_cisd_pads_zero_weight_strings_by_ranking(n2_sym):
    """K8: when d exceeds the CISD support, zero-weight strings fill the seed in ranking order."""
    norb, nelec, dims = 10, (7, 7), (82, 82)
    marginals = _cisd_marginals(*n2_sym, norb=norb, nelec=nelec)
    nonzero = [pool[w != 0.0] for pool, w, _ in marginals]
    assert all(len(nz) < d for nz, d in zip(nonzero, dims, strict=True))
    expected = _cisd_expected(marginals, nelec, dims)

    ((alpha, beta), _), _, zero_strings = _seed("cisd", n2_sym, norb=norb, nelec=nelec, dims=dims)

    for got, (want, _), nz in zip((alpha, beta), expected, nonzero, strict=True):
        np.testing.assert_array_equal(got, want)
        assert set(nz.tolist()) <= set(got.tolist())
    zero = [
        want[w[np.searchsorted(pool, want)] == 0.0]
        for (want, _), (pool, w, _) in zip(expected, marginals, strict=True)
    ]
    assert zero_strings.dtype == np.int64
    np.testing.assert_array_equal(zero_strings, np.concatenate(zero))
    assert len(zero_strings) == sum(count for _, count in expected)


def test_build_seed_space_cisd_orders_n2_zero_weight_tie_by_diagonal(n2_sym):
    """K6 (relaxed, M B2b-0 (a)): the tie group that holds exact zeros is cut by diagonal
    energy, not by integer order or by raw weight order."""
    norb, nelec, d = 10, (7, 7), 22
    marginals = _cisd_marginals(*n2_sym, norb=norb, nelec=nelec)
    pool, w, diag = marginals[0]
    ranked = np.sort(w)[::-1]
    assert ranked[-1] == 0.0 and np.all(-np.diff(ranked[d - 1 :]) <= 1e-6 * ranked[0])
    i, j = np.searchsorted(pool, [367, 223])
    assert abs(w[i] - w[j]) <= 1e-6 * ranked[0] and abs(diag[i] - diag[j]) <= 1e-9
    expected = _cisd_expected(marginals, nelec, (d, d))
    want = set(expected[0][0].tolist())
    flat, _ = extend.cisd_seed_strings(pool, w, np.zeros_like(diag), nelec_spin=7, size=d)
    assert want != set(flat.tolist())
    assert want != set(pool[np.argsort(-w, kind="stable")[:d]].tolist())

    ((alpha, beta), _), _, _ = _seed("cisd", n2_sym, norb=norb, nelec=nelec, dims=(d, d))

    np.testing.assert_array_equal(alpha, expected[0][0])
    np.testing.assert_array_equal(beta, expected[1][0])


# ---- seed space and guards (B2b group 2)

H6 = {"norb": 6, "nelec": (3, 3)}
H6_HF_STRING = 0b000111


def _product_hamiltonian(system, alpha, beta, *, norb, nelec):
    """Electronic Hamiltonian (no e_core) on the alpha x beta product, alpha-major order."""
    from pyscf.fci import cistring, direct_spin1

    h1, h2, _ = system
    pairs = np.array(
        list(
            product(
                cistring.strs2addr(norb, nelec[0], np.asarray(alpha)),
                cistring.strs2addr(norb, nelec[1], np.asarray(beta)),
            )
        )
    )
    full = (comb(norb, nelec[0]), comb(norb, nelec[1]))
    h2e = direct_spin1.absorb_h1e(h1, h2, norb, nelec, 0.5)
    columns = []
    for a, b in pairs:
        unit = np.zeros(full)
        unit[a, b] = 1.0
        columns.append(
            direct_spin1.contract_2e(h2e, unit, norb, nelec).reshape(full)[tuple(pairs.T)]
        )
    return np.array(columns).T


def _h6_hf_energy(system) -> float:
    return _product_hamiltonian(system, [H6_HF_STRING], [H6_HF_STRING], **H6)[0, 0] + system[2]


def _seed_h6(source, system, *, dims, audit, **caps):
    resources = RESOURCES | caps
    return extend.build_seed_space(
        source, *system, **H6, dims=dims, seed=0, audit=audit, resources=resources
    )


def test_build_seed_space_hf_is_single_hf_determinant(h6_chain):
    """The hf arm is the null control for seed quality, so it must hold only the HF determinant."""
    ((alpha, beta), amplitudes), energy, zero_weight = _seed("hf", h6_chain, **H6, dims=(1, 1))

    np.testing.assert_array_equal(alpha, [H6_HF_STRING])
    np.testing.assert_array_equal(beta, [H6_HF_STRING])
    np.testing.assert_allclose(np.abs(np.ravel(amplitudes)), [1.0])
    assert abs(energy - _h6_hf_energy(h6_chain)) <= 1e-10
    assert zero_weight is None


def test_build_seed_space_hf_ignores_dims(h6_chain):
    """The hf arm must ignore dims by ruling, and dims=(1, 1) cannot tell hf from random."""
    ((alpha, beta), amplitudes), energy, zero_weight = _seed("hf", h6_chain, **H6, dims=(5, 3))

    np.testing.assert_array_equal(alpha, [H6_HF_STRING])
    np.testing.assert_array_equal(beta, [H6_HF_STRING])
    np.testing.assert_allclose(np.abs(np.ravel(amplitudes)), [1.0])
    assert abs(energy - _h6_hf_energy(h6_chain)) <= 1e-10
    assert zero_weight is None


@pytest.mark.parametrize("source", ["random", "cisd"])
def test_build_seed_space_energy_and_amplitudes_diagonalize_product_space(h6_chain, source):
    """seed_space_energy is the D-P1 reference, so it must be the exact product-space minimum."""
    ((alpha, beta), amplitudes), energy, _ = _seed(source, h6_chain, **H6, dims=(5, 3))
    assert (len(alpha), len(beta)) == (5, 3)
    ham = _product_hamiltonian(h6_chain, alpha, beta, **H6)
    exact = np.linalg.eigvalsh(ham)[0] + h6_chain[2]
    assert abs(exact - _h6_hf_energy(h6_chain)) > 1e-6

    assert abs(energy - exact) <= 1e-10
    vector = np.ravel(amplitudes)
    assert abs(vector @ ham @ vector / (vector @ vector) + h6_chain[2] - energy) <= 1e-10


def test_build_seed_space_guards_cisd_full_sector_before_cisd(h6_chain, monkeypatch):
    """CISD allocates the full sector, so the guard must price that sector before CISD runs."""
    from q2m3.sqd.exceptions import AllocationGuardError

    audit: list[dict] = []
    _seed_h6("cisd", h6_chain, dims=(5, 3), audit=audit)
    phases = [(entry["phase"], tuple(entry["subspace_dims"])) for entry in audit]
    assert phases == [("seed_cisd", (20, 20)), ("seed_space", (5, 3))]

    def stub(*args, **kwargs):
        raise AssertionError("cisd_state ran before the seed_cisd guard rejected")

    monkeypatch.setattr(extend, "cisd_state", stub)
    with pytest.raises(AllocationGuardError):
        _seed_h6("cisd", h6_chain, dims=(5, 3), audit=[], rss_budget_mb=1.0)


@pytest.mark.parametrize(("source", "dims"), [("hf", (1, 1)), ("random", (5, 3))])
def test_build_seed_space_hf_and_random_skip_seed_cisd_guard(h6_chain, source, dims):
    """Only CISD allocates the full sector; a seed_cisd guard on hf or random would reject valid runs."""
    audit: list[dict] = []
    _seed_h6(source, h6_chain, dims=dims, audit=audit)

    assert [entry["phase"] for entry in audit] == ["seed_space"]


def test_build_seed_space_rejects_unknown_source(h6_chain):
    """quantum seeds come from sampling, so the dispatcher must refuse them and typos, not fall back."""
    for source in ("quantum", "bogus"):
        with pytest.raises(ValueError, match="seed source must be"):
            _seed(source, h6_chain, **H6, dims=(5, 3))


# ---- sci baseline (B3 group 1)

E3_SCI_PREFIX = [7, 11, 13, 14, 19, 21, 22, 25, 28, 35, 37, 38, 42, 49]  # S3.3b-0 probe (2)


def _ranking(h6, *, audit=None, **overrides):
    h1, h2, e_core = h6
    kwargs = {"norb": 6, "nelec": (3, 3), "audit": [] if audit is None else audit, **RESOURCES}
    return extend.sci_ranking(h1, h2, e_core, **(kwargs | overrides))


def test_sci_ranking_guards_full_sector(h6_chain):
    """Ranking diagonalizes the full sector, so its guard must see the full sector, not arm dims."""
    full = (comb(6, 3), comb(6, 3))
    assert full != (len(_e3_extended()),) * 2
    audit: list[dict] = []
    _ranking(h6_chain, audit=audit)
    assert len(audit) == 1
    assert audit[0]["phase"] == "sci_ranking"
    assert tuple(audit[0]["subspace_dims"]) == full


def test_sci_ranking_guard_rejection_raises_allocation_guard_error(h6_chain):
    """An over-budget ranking must stop before the full-sector allocation (D8), not run anyway."""
    from q2m3.sqd.exceptions import AllocationGuardError

    audit: list[dict] = []
    with pytest.raises(AllocationGuardError):
        _ranking(h6_chain, audit=audit, rss_budget_mb=1149.0)
    assert audit[-1]["phase"] == "sci_ranking"
    assert audit[-1]["predicted_rss_mb"] > 1149.0


def test_run_ext_sqd_sci_baseline_uses_ranked_prefix_not_arm_strings(h6_chain, monkeypatch):
    """The baseline must spend the arm dims on SCI-ranked strings, else D-P1 compares the arm to itself."""
    from pyscf.fci import cistring

    from q2m3.sqd import observables
    from q2m3.sqd.diagonalize import kernel_fixed_space
    from q2m3.sqd.observables import _select_sci

    h1, h2, e_core = h6_chain
    ranking = _ranking(h6_chain)
    calls: list[str] = []
    for module in (extend, observables):

        def counted(*args, _module=module, _run=module._run_selected_ci, **kwargs):
            calls.append(_module.__name__)
            return _run(*args, **kwargs)

        monkeypatch.setattr(module, "_run_selected_ci", counted)
    arm = _e3_run(extend.run_ext_sqd, h6_chain, sci_ranking=ranking)
    # A full-sector SCI rerun inside run_ext_sqd would skip the ranking guard (D8).
    assert calls == []
    oracle, _ = _select_sci(h1, h2, 6, (3, 3), tuple(arm.extended_counts))
    lexical = set(cistring.make_strings(range(6), 3)[:14].tolist())
    for spin in oracle:
        assert set(spin.tolist()) == set(E3_SCI_PREFIX)
        assert set(spin.tolist()) not in (set(_e3_extended()), lexical)
    e_oracle = kernel_fixed_space(
        h1, h2, e_core, tuple(oracle), norb=6, nelec=(3, 3), **RESOURCES
    ).energy
    assert e_oracle == pytest.approx(-3.2338790886971536, abs=1e-10)
    assert arm.energy == pytest.approx(-3.2329912656142668, abs=1e-10)
    assert abs(arm.sci_baseline_energy - e_oracle) <= 1e-10
    assert abs(arm.sci_baseline_energy - arm.energy) > 1e-6
    assert arm.sci_baseline_reason is None
    arm.validate()


def test_run_ext_sqd_sci_baseline_matches_extended_dims_per_spin(h6_chain):
    """The baseline gets the extended dims, not the truncated dims, so both sides have equal space."""
    audit: list[dict] = []
    arm = _e3_run(extend.run_ext_sqd, h6_chain, audit=audit, sci_ranking=_ranking(h6_chain))
    assert (tuple(arm.extended_counts), tuple(arm.truncated_counts)) == ((14, 14), (2, 1))
    baseline = [record for record in audit if record["phase"] == "sci_baseline"]
    assert len(baseline) == 1
    assert tuple(baseline[0]["subspace_dims"]) == tuple(arm.extended_counts)
    assert not [record for record in audit if record["phase"] == "sci_ranking"]


# ---- sci baseline (B3 group 2)


def _resident_mb(h6) -> tuple[float, float]:
    """(ranking MB, extended + amplitudes MB) for E3, per the D12 formula in M B3-0."""
    from q2m3.sqd.observables import _run_selected_ci

    h1, h2, _ = h6
    _, ci, pools, diags = _run_selected_ci(h1, h2, 6, (3, 3))
    ranking = ci.nbytes + sum(array.nbytes for array in (*pools, *diags))
    n = len(_e3_extended())
    solved = 2 * np.zeros(n, np.int64).nbytes + np.zeros((n, n)).nbytes
    assert ranking > 0 and solved > 0
    return ranking / 1e6, solved / 1e6


def test_run_ext_sqd_baseline_guard_rejection_keeps_arm_energy(h6_chain):
    """A baseline over budget must not erase the extended energy that is already solved (B3-9)."""
    from q2m3.sqd.resources import estimate_rss_mb

    n = len(_e3_extended())
    ranking_mb, solved_mb = _resident_mb(h6_chain)
    ext_retained = RESOURCES["retained_mb"] + ranking_mb
    predicted = [
        estimate_rss_mb(
            6, (3, 3), stage="diagonalize", subspace_dims=(n, n), retained_mb=r, model=_ext_model()
        )
        for r in (ext_retained, ext_retained + solved_mb)
    ]
    budget = sum(predicted) / 2
    audit: list[dict] = []
    arm = _e3_run(
        extend.run_ext_sqd,
        h6_chain,
        audit=audit,
        rss_budget_mb=budget,
        sci_ranking=_ranking(h6_chain),
    )
    assert [record["phase"] for record in audit] == ["ext_quantum", "sci_baseline"]
    assert audit[0]["predicted_rss_mb"] <= budget < audit[1]["predicted_rss_mb"]
    assert arm.energy == pytest.approx(-3.2329912656142668, abs=1e-10)
    assert arm.null_reason is None
    assert arm.sci_baseline_energy is None
    assert arm.sci_baseline_reason.startswith("sci_baseline_rss_budget: ")
    arm.validate()


def test_run_ext_sqd_baseline_numerical_error_keeps_arm_energy(h6_chain, monkeypatch):
    """A baseline solve that fails numerically must not erase the solved extended energy."""
    from q2m3.sqd.exceptions import ReferenceNumericalError

    original = extend.kernel_fixed_space
    calls = []

    def solver(*args, **kwargs):
        calls.append(len(calls))
        if len(calls) == 2:
            raise ReferenceNumericalError("STUB baseline numerical")
        return original(*args, **kwargs)

    monkeypatch.setattr(extend, "kernel_fixed_space", solver)
    audit: list[dict] = []
    arm = _e3_run(extend.run_ext_sqd, h6_chain, audit=audit, sci_ranking=_ranking(h6_chain))
    assert len(calls) == 2
    assert [record["phase"] for record in audit] == ["ext_quantum", "sci_baseline"]
    assert arm.energy == pytest.approx(-3.2329912656142668, abs=1e-10)
    assert arm.null_reason is None
    assert arm.sci_baseline_energy is None
    assert arm.sci_baseline_reason.startswith("sci_baseline_numerical: ")
    assert "STUB baseline numerical" in arm.sci_baseline_reason
    arm.validate()


def test_cisd_state_rejects_unconverged_cisd(h6_chain, monkeypatch):
    """An unconverged CISD vector must not seed the arm silently."""
    from pyscf.ci import cisd

    from q2m3.sqd.exceptions import ReferenceNumericalError

    original = cisd.CISD.kernel

    def unconverged(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.converged = False
        return result

    monkeypatch.setattr(cisd.CISD, "kernel", unconverged)
    h1, h2, e_core = h6_chain
    with pytest.raises(ReferenceNumericalError, match="not converge.*conv_tol"):
        extend.cisd_state(h1, h2, e_core, norb=6, nelec=(3, 3))


def test_run_ext_sqd_guards_count_resident_ranking_and_solution(h6_chain):
    """Guards must count the ranking and the extended solution that stay resident, else D12 under-predicts RSS."""
    ranking_mb, solved_mb = _resident_mb(h6_chain)
    audit: list[dict] = []
    _e3_run(extend.run_ext_sqd, h6_chain, audit=audit, sci_ranking=_ranking(h6_chain))
    # A list keeps a repeated phase visible; a dict would keep only its last record.
    assert [record["phase"] for record in audit] == ["ext_quantum", "sci_baseline"]
    retained = {record["phase"]: record["retained_mb"] for record in audit}
    ext_retained = RESOURCES["retained_mb"] + ranking_mb
    assert retained["ext_quantum"] == pytest.approx(ext_retained, rel=1e-12)
    assert retained["sci_baseline"] == pytest.approx(ext_retained + solved_mb, rel=1e-12)


def test_run_ext_sqd_wall_budget_arm_records_baseline_reason(h6_chain):
    """A null arm has no extended space, so the baseline must record why it did not run."""
    import time

    audit: list[dict] = []
    arm = _e3_run(
        extend.run_ext_sqd,
        h6_chain,
        audit=audit,
        deadline=time.monotonic(),
        sci_ranking=_ranking(h6_chain),
    )
    assert arm.null_reason == "ext_wall_budget"
    assert arm.sci_baseline_energy is None
    assert arm.sci_baseline_reason == "sci_baseline_arm_null"
    assert not [record for record in audit if record["phase"] == "sci_baseline"]


def test_run_ext_sqd_without_ranking_records_baseline_reason(h6_chain):
    """Without a ranking the baseline cannot run, so the arm must say so and keep its energy."""
    audit: list[dict] = []
    arm = _e3_run(extend.run_ext_sqd, h6_chain, audit=audit)
    assert arm.sci_baseline_reason == "sci_ranking_unavailable"
    assert not [r for r in audit if r["phase"] in ("sci_ranking", "sci_baseline")]
    assert arm.energy == pytest.approx(-3.2329912656142668, abs=1e-10)
    assert arm.sci_baseline_energy is None
    assert arm.null_reason is None
    arm.validate()


def test_sci_ranking_energy_includes_e_core(h6_chain):
    """The ranking energy is public, |e_core| > 1 Ha here, and no other test reads it, so e_core loss went unseen."""
    from pyscf import fci

    h1, h2, e_core = h6_chain
    assert abs(e_core) > 1.0
    energy = _ranking(h6_chain)[0]
    e_fci, _ = fci.direct_spin1.kernel(h1, h2, 6, (3, 3), ecore=e_core, tol=1e-13)
    assert e_fci - 1e-8 <= energy <= e_fci + 0.1


def test_run_ext_sqd_ext_guard_rejection_audit_counts_resident_ranking(h6_chain):
    """The SCI ranking stays resident during the ext stage, so the ext guard must add it to RSS.

    A guard that leaves the ranking out under-predicts RSS and can admit an OOM. The guard
    writes its prediction to the audit only on a rejection, and no other test rejects there.
    """
    ranking_mb, _ = _resident_mb(h6_chain)
    assert ranking_mb > 0
    records = []
    for ranking in (_ranking(h6_chain), None):
        audit: list[dict] = []
        with pytest.raises(extend.AllocationGuardError):
            _e3_run(
                extend.run_ext_sqd,
                h6_chain,
                audit=audit,
                # Half of 1149.6328 MB, the smaller ext_quantum prediction (no ranking,
                # S5b.3-c1). ext_quantum is the first guard, so both runs reject there.
                rss_budget_mb=574.8164,
                sci_ranking=ranking,
            )
        (record,) = [record for record in audit if record["phase"] == "ext_quantum"]
        records.append(record)
    with_ranking, without_ranking = records
    assert without_ranking["retained_mb"] == pytest.approx(RESOURCES["retained_mb"], rel=1e-12)
    assert with_ranking["retained_mb"] == pytest.approx(
        RESOURCES["retained_mb"] + ranking_mb, rel=1e-12
    )
    assert with_ranking["predicted_rss_mb"] == pytest.approx(
        without_ranking["predicted_rss_mb"] + ranking_mb, rel=1e-12
    )


# ---- arm comparison (B4 group 2, P1)


def test_ext_config_comparison_accepts_records_with_equal_settings_and_code_path():
    """O-S5.1: the arms of one point share settings and code path, so they must compare clean."""
    records = []
    for shift in range(4):
        arms = ext_record_parts()
        arms[0]["energy"] -= 0.01 * shift  # arm data differ; settings and code path do not
        records.append(ext_record(arms=arms))
    assert len({record.arms[0].energy for record in records}) == 4
    assert extend.assert_same_ext_config(records) is None


def test_ext_config_comparison_reports_a_differing_threshold_by_field_name():
    """K2: a tau that differs between arms must fail the comparison and name only that field."""
    from dataclasses import fields, replace

    base = ext_record()
    other = replace(base, settings=replace(base.settings, threshold=base.settings.threshold * 10))
    names = [field.name for field in fields(base.settings)]
    differing = [n for n in names if getattr(base.settings, n) != getattr(other.settings, n)]
    assert differing == ["threshold"] and other.code_path_id == base.code_path_id
    with pytest.raises(ValueError, match="threshold") as excinfo:
        extend.assert_same_ext_config([base, other])
    unrelated = [n for n in names if n != "threshold"] + ["code_path_id"]
    assert [n for n in unrelated if n in str(excinfo.value)] == []


def test_ext_config_comparison_reports_a_differing_code_path_id():
    """Arms from different code paths must not compare equal even when the settings match."""
    from dataclasses import replace

    base = ext_record()
    other = replace(base, code_path_id="ext-sqd-other")
    assert other.settings == base.settings
    with pytest.raises(ValueError, match="code_path_id"):
        extend.assert_same_ext_config([base, other])


def test_ext_config_comparison_rejects_a_non_record_with_value_error():
    """I-013: a wrong input type must raise ValueError, not TypeError or AttributeError."""
    inputs = [ext_record(), ext_arm_data()]
    with pytest.raises(ValueError, match="expected ExtSQDRecord elements, got dict"):
        extend.assert_same_ext_config(inputs)
