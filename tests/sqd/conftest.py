"""Optional SQD fixtures; missing extras never break test collection."""

from __future__ import annotations

import importlib.util

import pytest


def pytest_runtest_setup(item):
    """Skip only marked SQD tests when the optional distributions are absent."""
    if item.get_closest_marker("sqd") is not None:
        _require_sqd_extra()


def _require_sqd_extra():
    missing = [
        name for name in ("ffsim", "qiskit_addon_sqd") if importlib.util.find_spec(name) is None
    ]
    if missing:
        pytest.skip(f"SQD extra is not installed: {', '.join(missing)}")


@pytest.fixture(scope="session")
def sqd_modules():
    """Import optional dependencies at execution time, after the availability guard."""
    import importlib

    _require_sqd_extra()
    return tuple(importlib.import_module(name) for name in ("ffsim", "qiskit_addon_sqd"))
