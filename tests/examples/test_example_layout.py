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
