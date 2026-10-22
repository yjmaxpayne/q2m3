"""Import-direction rules for ``q2m3.fragments``, checked by an AST scan of the source tree.

The scan cannot see dynamic imports (``importlib.import_module``, ``__import__``), names a
star import may load lazily, or private attributes read from an imported public module.
These cases are outside the checked surface.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "q2m3"
FRAGMENTS = "q2m3.fragments"
FORBIDDEN = ("q2m3.solvation", "q2m3.sampling", "q2m3.profiling")
OPTIONAL_BACKENDS = frozenset({"psi4", "pyblock2", "block2", "ase"})
PRIVATE_EXEMPT = frozenset({"_lazy"})
# Runs inside the Psi4 interpreter, so it may import psi4 at module level.
TOP_LEVEL_EXEMPT = Path("fragments", "external", "_psi4_driver.py")


@dataclass(frozen=True)
class Violation:
    path: Path
    lineno: int
    rule: str
    module: str
    name: str | None = None

    def __str__(self) -> str:
        tail = f" {self.name}" if self.name else ""
        return f"{self.path}:{self.lineno} {self.rule} {self.module}{tail}"


def _within(module: str, package: str) -> bool:
    return module == package or module.startswith(package + ".")


def _module_name(root: Path, path: Path) -> str:
    parts = [root.name, *path.relative_to(root).with_suffix("").parts]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _targets(
    node: ast.Import | ast.ImportFrom, module: str, is_package: bool
) -> list[tuple[str, str | None]]:
    """Absolute ``(module, name)`` pairs loaded by one import statement."""
    if isinstance(node, ast.Import):
        return [(alias.name, None) for alias in node.names]
    base = node.module or ""
    if node.level:
        # The package of an __init__.py is the module itself, not its parent.
        package = module if is_package else module.rpartition(".")[0]
        anchor = package.rsplit(".", node.level - 1)[0]
        base = f"{anchor}.{base}" if base else anchor
    return [(base, None if alias.name == "*" else alias.name) for alias in node.names]


def _is_private(part: str) -> bool:
    dunder = part.startswith("__") and part.endswith("__")
    return part.startswith("_") and not dunder and part not in PRIVATE_EXEMPT


def _rules(target: str, inside: bool) -> list[str]:
    if not inside:
        return ["reverse"] if _within(target, FRAGMENTS) else []
    rules = []
    if any(_within(target, forbidden) for forbidden in FORBIDDEN):
        rules.append("forbidden")
    if (
        _within(target, "q2m3")
        and not _within(target, FRAGMENTS)
        and any(_is_private(part) for part in target.split(".")[1:])
    ):
        rules.append("private")
    return rules


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _top_level_imports(nodes: Iterable[ast.AST]) -> Iterator[ast.Import | ast.ImportFrom]:
    """Imports outside function and lambda bodies; skips the body of ``if TYPE_CHECKING:``."""
    for node in nodes:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        if isinstance(node, ast.Import | ast.ImportFrom):
            yield node
        elif isinstance(node, ast.If) and _is_type_checking(node.test):
            yield from _top_level_imports(node.orelse)
        else:
            yield from _top_level_imports(ast.iter_child_nodes(node))


def scan(root: Path) -> list[Violation]:
    """Check every module under ``root`` (a ``q2m3`` package directory) against the rules.

    Rules: ``reverse`` (a module outside the fragments package imports it), ``forbidden``
    (fragments imports solvation, sampling or profiling), ``private`` (fragments imports an
    underscore name from another q2m3 package), ``optional top-level`` (fragments imports an
    optional backend outside a function body).
    """
    violations = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = _module_name(root, path)
        is_package = path.name == "__init__.py"
        inside = _within(module, FRAGMENTS)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Import | ast.ImportFrom):
                continue
            for base, name in _targets(node, module, is_package):
                target = f"{base}.{name}" if name else base
                for rule in _rules(target, inside):
                    violations.append(Violation(path, node.lineno, rule, base, name))
        if not inside or path.relative_to(root) == TOP_LEVEL_EXEMPT:
            continue
        for node in _top_level_imports(tree.body):
            bases = dict.fromkeys(base for base, _ in _targets(node, module, is_package))
            for base in bases:
                if base.split(".")[0] in OPTIONAL_BACKENDS:
                    violations.append(Violation(path, node.lineno, "optional top-level", base))
    return violations


def _write_tree(tmp_path: Path, relative: str, source: str) -> tuple[Path, Path]:
    root = tmp_path / "q2m3"
    for package in (root, root / "fragments", root / "sqd"):
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").touch()
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return root, target


FRAGMENT = "fragments/mod.py"
SQD = "sqd/mod.py"

# (relative path, source, expected rule or None for an accepted import)
CASES = {
    "absolute-public-name": (FRAGMENT, "from q2m3.sqd import run_sqd_from_integrals\n", None),
    "import-forbidden-module": (FRAGMENT, "import q2m3.solvation\n", "forbidden"),
    "relative-forbidden-module": (FRAGMENT, "from ..solvation import x\n", "forbidden"),
    "relative-private-name": (FRAGMENT, "from ..sqd.orchestrator import _unwire\n", "private"),
    "absolute-private-name": (FRAGMENT, "from q2m3.sqd.orchestrator import _unwire\n", "private"),
    "top-level-optional": (FRAGMENT, "import psi4\n", "optional top-level"),
    "optional-in-top-level-try": (
        FRAGMENT,
        "try:\n    import psi4\nexcept ImportError:\n    psi4 = None\n",
        "optional top-level",
    ),
    "optional-under-type-checking": (
        FRAGMENT,
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import psi4\n",
        None,
    ),
    "optional-in-function": (FRAGMENT, "def f():\n    import psi4\n    return psi4\n", None),
    "from-package-import-forbidden": (FRAGMENT, "from q2m3 import solvation\n", "forbidden"),
    "top-level-optional-submodule": (
        FRAGMENT,
        "from pyblock2.driver.core import DMRGDriver\n",
        "optional top-level",
    ),
    "exempt-psi4-driver": ("fragments/external/_psi4_driver.py", "import psi4\n", None),
    "reverse-import": (SQD, "import q2m3.fragments\n", "reverse"),
    "reverse-relative-import": (SQD, "from .. import fragments\n", "reverse"),
    "optional-in-top-level-with": (
        FRAGMENT,
        "from contextlib import suppress\n\nwith suppress(ImportError):\n    import ase\n",
        "optional top-level",
    ),
    "optional-in-class-body": (FRAGMENT, "class Driver:\n    import psi4\n", "optional top-level"),
    "optional-in-except-branch": (
        FRAGMENT,
        "try:\n    pass\nexcept ImportError:\n    import psi4\n",
        "optional top-level",
    ),
    "optional-in-type-checking-else": (
        FRAGMENT,
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    pass\nelse:\n    import psi4\n",
        "optional top-level",
    ),
    "optional-in-version-branch": (
        FRAGMENT,
        "import sys\n\nif sys.version_info >= (3, 11):\n    import psi4\n",
        "optional top-level",
    ),
    "import-private-module-as": (FRAGMENT, "import q2m3.sqd._x as y\n", "private"),
    "from-private-module": (FRAGMENT, "from q2m3.sqd._x import pub\n", "private"),
    "lazy-helper-exempt": (FRAGMENT, "from q2m3 import _lazy\n", None),
    "dunder-name": (FRAGMENT, "from q2m3 import __version__\n", None),
    "fragment-internal-private": (FRAGMENT, "from ._sqd_child import X\n", None),
    "package-init-relative": ("fragments/__init__.py", "from .workflow import run\n", None),
    "reverse-from-package": (SQD, "from q2m3 import fragments\n", "reverse"),
}


@pytest.mark.parametrize("relative, source, rule", CASES.values(), ids=CASES.keys())
def test_rule_table(tmp_path, relative, source, rule):
    root, target = _write_tree(tmp_path, relative, source)
    violations = [str(violation) for violation in scan(root)]
    if rule is None:
        assert violations == []
    else:
        assert len(violations) == 1, violations
        assert violations[0].startswith(f"{target}:"), violations
        assert f" {rule} " in violations[0], violations


def test_package_init_relative_import_resolves_to_package(tmp_path):
    # A wrong anchor for __init__.py would resolve this to q2m3._x and report a private import.
    root, _ = _write_tree(tmp_path, "fragments/__init__.py", "from ._x import y\n")
    assert [str(violation) for violation in scan(root)] == []
    root, target = _write_tree(tmp_path, "fragments/__init__.py", "from .._x import y\n")
    assert [str(violation) for violation in scan(root)] == [f"{target}:1 private q2m3._x y"]


def test_source_tree_has_no_violations():
    assert (SRC_ROOT / "fragments" / "__init__.py").is_file()
    assert [str(violation) for violation in scan(SRC_ROOT)] == []
