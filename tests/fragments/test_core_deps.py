"""Core dependencies exclude external quantum-chemistry programs."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
EXTERNAL_PROGRAMS = frozenset({"block2", "psi4", "xtb", "orca"})


def _requirement_name(requirement: str) -> str:
    return re.split(r"[\s<>=!~\[;]", requirement, maxsplit=1)[0].lower()


def test_core_dependencies_exclude_external_programs():
    with PYPROJECT.open("rb") as handle:
        requirements = tomllib.load(handle)["project"]["dependencies"]
    names = {_requirement_name(requirement) for requirement in requirements}
    assert "pyscf" in names, names
    assert sorted(names & EXTERNAL_PROGRAMS) == []
