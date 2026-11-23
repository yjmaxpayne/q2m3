"""Dependency rules for the SQD-driven MC solvation module."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.sqd.test_imports import boundary_violations

SQD_MC_PATH = Path(__file__).resolve().parents[2] / "src/q2m3/solvation/sqd_mc.py"
PACKAGE = "q2m3.solvation"
BACKENDS = ("q2m3.sqd", "ffsim", "qiskit_addon_sqd", "catalyst", "jax", "pennylane")
QPE_ORCHESTRATION = (
    "q2m3.solvation.orchestrator",
    "q2m3.solvation.circuit_builder",
    "q2m3.solvation.ir_cache",
    # The package __init__ loads the orchestrator lazily for these names.
    "q2m3.solvation.run_solvation",
    "q2m3.solvation.replay_quantum_trajectory",
    "q2m3.solvation.*",
)
# catalyst, jax and pennylane are absent: `import q2m3` already loads them.
UNLOADED_AFTER_IMPORT = (
    "q2m3.solvation.orchestrator",
    "q2m3.solvation.circuit_builder",
    "q2m3.solvation.ir_cache",
    "q2m3.sqd",
    "q2m3.sqd.orchestrator",
    "ffsim",
    "qiskit_addon_sqd",
)
LOADED_MODULES_PROBE = r"""
import json
import sys

import q2m3.solvation.sqd_mc

watched = json.loads(sys.argv[1])
print(json.dumps(sorted(name for name in watched if name in sys.modules)))
"""

REJECT = True
ACCEPT = False


def _definition_time_parts(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> list:
    """Return the parts of a definition that Python evaluates when it runs the def."""
    args = node.args
    parts = [*args.defaults, *(default for default in args.kw_defaults if default is not None)]
    if not isinstance(node, ast.Lambda):
        parameters = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
        parts += node.decorator_list
        parts += [p.annotation for p in parameters if p is not None and p.annotation is not None]
        if node.returns is not None:
            parts.append(node.returns)
    return parts


def import_time_nodes(tree: ast.AST) -> Iterator[ast.AST]:
    """Yield nodes evaluated at import time.

    Function and lambda bodies are skipped. Decorators, defaults, keyword defaults
    and annotations are visited, because Python 3.12 evaluates them at def time.
    """
    stack = list(ast.iter_child_nodes(tree))
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            stack.extend(_definition_time_parts(node))
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def module_level_violations(source: str) -> list[str]:
    """Return backend modules that the source imports at import time."""
    violations = []
    for node in import_time_nodes(ast.parse(source)):
        if isinstance(node, ast.Import | ast.ImportFrom):
            statement = node
        elif isinstance(node, ast.Call):
            statement = ast.Expr(value=node)
        else:
            continue
        snippet = ast.unparse(ast.Module(body=[statement], type_ignores=[]))
        violations += boundary_violations(snippet, PACKAGE, BACKENDS)
    return violations


def qpe_orchestration_violations(source: str) -> list[str]:
    """Return QPE orchestration modules that the source references anywhere."""
    return boundary_violations(source, PACKAGE, QPE_ORCHESTRATION)


def test_sqd_mc_has_no_module_level_backend_imports():
    assert module_level_violations(SQD_MC_PATH.read_text()) == []


def test_sqd_mc_never_imports_qpe_orchestration():
    assert qpe_orchestration_violations(SQD_MC_PATH.read_text()) == []


def test_importing_sqd_mc_leaves_backends_and_qpe_orchestration_unloaded():
    # A fresh interpreter also catches import-time loads that the AST scan misses.
    process = subprocess.run(
        [sys.executable, "-c", LOADED_MODULES_PROBE, json.dumps(UNLOADED_AFTER_IMPORT)],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(process.stdout.splitlines()[-1]) == []


@pytest.mark.parametrize(
    ("code", "rejected"),
    [
        pytest.param("import ffsim", REJECT, id="import"),
        pytest.param("import jax.numpy as jnp", REJECT, id="import-submodule-alias"),
        pytest.param("from q2m3.sqd import run_sqd", REJECT, id="from-module"),
        pytest.param("from q2m3 import sqd", REJECT, id="from-root-import"),
        pytest.param("from .. import sqd", REJECT, id="relative-parent"),
        pytest.param("try:\n import catalyst\nexcept ImportError:\n pass", REJECT, id="try-block"),
        pytest.param(
            "if TYPE_CHECKING:\n from q2m3.sqd.result import SQDResult",
            REJECT,
            id="type-checking-block",
        ),
        pytest.param("class C:\n import ffsim", REJECT, id="class-body"),
        pytest.param(
            "import importlib\nimportlib.import_module('ffsim')", REJECT, id="import-module-call"
        ),
        pytest.param("__import__('pennylane')", REJECT, id="dunder-import-call"),
        pytest.param("@importlib.import_module('jax')\ndef f():\n pass", REJECT, id="decorator"),
        pytest.param("def f(x=__import__('jax')):\n pass", REJECT, id="positional-default"),
        pytest.param("def f():\n from q2m3.sqd import run_sqd", ACCEPT, id="function-body"),
        pytest.param("async def f():\n import ffsim", ACCEPT, id="async-function-body"),
        pytest.param("g = lambda: __import__('ffsim')", ACCEPT, id="lambda-body"),
        pytest.param("from q2m3.sqdx import y", ACCEPT, id="similar-prefix-absolute"),
        pytest.param("from .sqd_helpers import x", ACCEPT, id="similar-prefix-relative"),
        pytest.param("from .energy import StepResult", ACCEPT, id="sibling-submodule"),
        pytest.param("def f(x: __import__('jax').Array):\n pass", REJECT, id="argument-annotation"),
        pytest.param("def f() -> __import__('ffsim').X:\n pass", REJECT, id="return-annotation"),
        # Known gap: a module-level call runs the body at import time. The fresh
        # interpreter test above observes this case through sys.modules.
        pytest.param(
            "def _load():\n import ffsim\n_load()", ACCEPT, id="module-level-call-known-gap"
        ),
        pytest.param("def f(*, x=__import__('jax')):\n pass", REJECT, id="keyword-only-default"),
        pytest.param("g = lambda x=__import__('jax'): x", REJECT, id="lambda-default"),
        pytest.param("def f(*a: __import__('jax').X):\n pass", REJECT, id="vararg-annotation"),
        pytest.param("def f(**k: __import__('jax').X):\n pass", REJECT, id="kwarg-annotation"),
        pytest.param(
            "def f(*, x: __import__('jax').X):\n pass", REJECT, id="keyword-only-annotation"
        ),
        pytest.param(
            "def f(x: __import__('jax').X, /):\n pass",
            REJECT,
            id="positional-only-annotation",
        ),
        pytest.param("if __name__ == '__main__':\n import ffsim", REJECT, id="main-guard"),
        pytest.param("for _ in ():\n import ffsim", REJECT, id="for-block"),
        pytest.param("with open(__file__):\n import ffsim", REJECT, id="with-block"),
        pytest.param("x = [__import__('ffsim') for _ in ()]", REJECT, id="comprehension"),
        pytest.param("from ..sqd import run_sqd", REJECT, id="relative-parent-module"),
        pytest.param(
            "from importlib import import_module as load\nload('ffsim')",
            REJECT,
            id="aliased-import-module",
        ),
        pytest.param("while False:\n import ffsim", REJECT, id="while-block"),
        pytest.param(
            "from .energy import StepResult\n\ndef f():\n pass\n\nimport ffsim",
            REJECT,
            id="appended-after-definitions",
        ),
        pytest.param(
            "if TYPE_CHECKING:\n from .energy import StepResult\n"
            " from q2m3.sqd import SQDResult",
            REJECT,
            id="type-checking-second-statement",
        ),
        pytest.param("match x:\n case _:\n  import ffsim", REJECT, id="match-block"),
    ],
)
def test_module_level_scanner_negative_space(code, rejected):
    assert bool(module_level_violations(code)) is rejected


@pytest.mark.parametrize(
    ("code", "rejected"),
    [
        pytest.param("from .orchestrator import run_solvation", REJECT, id="relative-orchestrator"),
        pytest.param("from . import circuit_builder", REJECT, id="relative-package-import"),
        pytest.param(
            "def f():\n from q2m3.solvation import ir_cache", REJECT, id="function-body-from"
        ),
        pytest.param(
            "def f():\n importlib.import_module('q2m3.solvation.orchestrator')",
            REJECT,
            id="function-body-import-module",
        ),
        pytest.param(
            "from q2m3.solvation import orchestrator as o", REJECT, id="from-package-alias"
        ),
        pytest.param("import q2m3.solvation.mc_loop", ACCEPT, id="allowed-absolute-submodule"),
        pytest.param("from .energy import StepResult", ACCEPT, id="allowed-relative-submodule"),
        # Known gap: a computed module name is invisible to a static scan.
        pytest.param(
            "def f(n):\n importlib.import_module('q2m3.solvation.' + n)",
            ACCEPT,
            id="dynamic-name-known-gap",
        ),
        pytest.param("from . import run_solvation", REJECT, id="lazy-run-solvation"),
        pytest.param(
            "from q2m3.solvation import replay_quantum_trajectory",
            REJECT,
            id="lazy-replay-trajectory",
        ),
        pytest.param("from . import *", REJECT, id="relative-star"),
        pytest.param("from q2m3.solvation import *", REJECT, id="absolute-star"),
        pytest.param("from .energy import *", ACCEPT, id="sibling-star"),
        pytest.param("import q2m3.solvation.orchestrator", REJECT, id="import-orchestrator"),
        pytest.param(
            "from ..solvation import orchestrator", REJECT, id="relative-parent-orchestrator"
        ),
        pytest.param(
            "import q2m3.solvation\nq2m3.solvation.run_solvation", REJECT, id="attribute-access"
        ),
        pytest.param(
            "from importlib import import_module as load\nload('q2m3.solvation.orchestrator')",
            REJECT,
            id="aliased-import-module",
        ),
        pytest.param("import q2m3.solvation.ir_cache as ic", REJECT, id="import-alias"),
        pytest.param(
            "if TYPE_CHECKING:\n from .orchestrator import run_solvation",
            REJECT,
            id="type-checking-block",
        ),
        pytest.param(
            "def f():\n from .energy import StepResult\n from .orchestrator import run_solvation",
            REJECT,
            id="function-body-second-statement",
        ),
    ],
)
def test_qpe_orchestration_scanner_negative_space(code, rejected):
    assert bool(qpe_orchestration_violations(code)) is rejected
