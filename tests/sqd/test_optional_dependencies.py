"""Executable smoke coverage for the installed SQD optional dependency profile."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_local_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.sqd
def test_optional_sqd_sampling_preserves_support_and_seed(sqd_modules):
    import numpy as np

    replay = load_local_module(
        "sqd_optional_replay", Path(__file__).resolve().parents[2] / "tools/sqd/replay/replay.py"
    )

    ffsim, _ = sqd_modules
    assert replay.interaction_pair_checks() == 8

    # Two determinants with unequal weights make a constant-output sampler fail.
    state = np.array([0.5, 0.0, 0.0, np.sqrt(0.75)], dtype=complex)
    options = {
        "norb": 2,
        "nelec": (1, 1),
        "shots": 512,
        "bitstring_type": ffsim.BitstringType.BIT_ARRAY,
        "seed": 20260917,
    }
    bits = np.asarray(ffsim.sample_state_vector(state, **options))
    repeated = np.asarray(ffsim.sample_state_vector(state, **options))
    np.testing.assert_array_equal(bits, repeated)
    assert bits.shape == (512, 4)
    assert np.all(bits[:, :2].sum(axis=1) == 1)
    assert np.all(bits[:, 2:].sum(axis=1) == 1)
    outcomes, counts = np.unique(bits, axis=0, return_counts=True)
    assert outcomes.shape == (2, 4)
    # Basis states 0 and 3 place both spins in the same spatial orbital.
    np.testing.assert_array_equal(outcomes[:, :2], outcomes[:, 2:])
    np.testing.assert_allclose(np.sort(counts / 512), [0.25, 0.75], atol=0.08, rtol=0)


@pytest.mark.parametrize("entrypoint", ["marker", "fixture"])
def test_optional_guard_skips_absent_extra(monkeypatch, entrypoint):
    guards = load_local_module("sqd_guard_checks", Path(__file__).with_name("conftest.py"))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(pytest.skip.Exception, match="SQD extra is not installed"):
        if entrypoint == "marker":
            item = SimpleNamespace(get_closest_marker=lambda name: object())
            guards.pytest_runtest_setup(item)
        else:
            guards.sqd_modules.__wrapped__()


def test_optional_fixture_propagates_broken_installed_import(monkeypatch):
    guards = load_local_module("sqd_guard_checks", Path(__file__).with_name("conftest.py"))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    def broken_import(name):
        raise ImportError(f"Broken binary dependency in {name}")

    monkeypatch.setattr(importlib, "import_module", broken_import)
    with pytest.raises(ImportError, match="Broken binary dependency in ffsim"):
        guards.sqd_modules.__wrapped__()
