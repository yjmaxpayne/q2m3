"""Path-sensitive migration contracts, independent of numerical algorithms."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "module",
    [
        "examples.resources.resource_estimation_survey",
        "examples.performance.ir_qre_correlation_analysis",
        "examples.performance.ir_qre_trotter5_compile_survey",
        "examples.performance.h3o_dynamic_trotter_oom_scan",
    ],
)
def test_output_directory_stays_at_repository_root(module):
    loaded = importlib.import_module(module)
    directory = getattr(loaded, "OUTPUT_DIR", getattr(loaded, "DEFAULT_OUTPUT_DIR", None))
    assert directory == ROOT / "data/output"


@pytest.mark.parametrize(
    "module",
    [
        "tools.sqd.calibrate_resources",
        "tools.sqd.connectivity_comparison",
        "tools.sqd.orbital_basis_scan",
        "tools.sqd.molecular_benchmark",
    ],
)
def test_tool_root_is_checkout(module):
    assert importlib.import_module(module).ROOT == ROOT


def test_replay_root_and_inventory():
    module = importlib.import_module("tools.sqd.replay.replay")
    assert module.ROOT == ROOT
    assert module.validate_manifest()["origins"]


def test_sibling_tool_is_importable():
    module = importlib.import_module("tools.sqd.compare_fixed_space")
    assert module.ROOT == ROOT
