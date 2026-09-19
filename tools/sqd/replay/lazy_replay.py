"""Restore a disposable package and check lazy exports in fresh interpreters."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PACKAGE = """from .lazy_helpers import available_exports, lazy_getattr
_LAZY = {"run_sqd": (__name__ + ".sqd", "sqd"),
         "run_solvation": (__name__ + ".solvation", "catalyst")}
__all__ = available_exports(_LAZY)
def __getattr__(name):
    return lazy_getattr(__name__, globals(), _LAZY, name)
def __dir__():
    return sorted(set(globals()) | set(_LAZY))
"""
CHILD = """import importlib.abc, importlib.util, json, sys
blocked = sys.argv[1].split(",") if sys.argv[1] else []
requirements = {"run_sqd": ("ffsim", "qiskit_addon_sqd"),
                "run_solvation": ("catalyst", "jax")}
# Compute the oracle from the installed environment before importing the helper
# or inserting our blocker; do not consume available_exports or package __all__.
expected = {name for name, deps in requirements.items()
            if all(dep not in blocked and importlib.util.find_spec(dep) is not None
                   for dep in deps)}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ImportError("blocked dependency")
sys.meta_path.insert(0, Block())
import replay_package as pkg
checks = []
def check(label, value):
    checks.append({"check": label, "pass": bool(value)})
    assert value, label
check("package import", pkg.__name__ == "replay_package")
check("no optional import", not set(sys.modules).intersection(
    {"ffsim", "qiskit", "qiskit_addon_sqd", "catalyst", "jax", "pennylane"}))
check("all equals expected and dir covers all",
      set(pkg.__all__) == expected and expected <= set(dir(pkg)))
ns = {}
exec("from replay_package import *", ns)
check("star equals expected", {name for name in ns if not name.startswith("__")} == expected)
resolved = []
for name, extra in (("run_sqd", "sqd"), ("run_solvation", "catalyst")):
    try:
        value = getattr(pkg, name)
        resolved.append(name in expected and callable(value))
    except ImportError as exc:
        resolved.append(name not in expected and "uv sync --extra " + extra in str(exc))
check("explicit imports or actionable errors", all(resolved))
try:
    getattr(pkg, "unknown_export")
except AttributeError:
    unknown_ok = True
else:
    unknown_ok = False
check("unknown attribute", unknown_ok)
print(json.dumps({"blocked": blocked, "expected_exports": sorted(expected), "checks": checks, "n_total": 6, "n_pass": 6}))
"""


def run() -> dict:
    """Exercise six assertions per actual-install and missing-extra profile."""
    profiles = []
    with tempfile.TemporaryDirectory(prefix="sqd-lazy-") as temp:
        root = Path(temp)
        package = root / "replay_package"
        package.mkdir()
        (package / "__init__.py").write_text(PACKAGE)
        shutil.copyfile(Path(__file__).with_name("lazy_helpers.py"), package / "lazy_helpers.py")
        (package / "sqd.py").write_text("import ffsim, qiskit_addon_sqd\ndef run_sqd(): pass\n")
        (package / "solvation.py").write_text("import catalyst, jax\ndef run_solvation(): pass\n")
        for blocked in (
            "",
            "ffsim,qiskit_addon_sqd",
            "catalyst",
            "ffsim,qiskit_addon_sqd,catalyst,jax",
        ):
            try:
                process = subprocess.run(
                    [sys.executable, "-c", CHILD, blocked],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                print(f"Lazy profile blocked={blocked!r} failed:\n{exc.stderr}", file=sys.stderr)
                raise
            profiles.append(json.loads(process.stdout.splitlines()[-1]))
    return {
        "profiles": profiles,
        "n_total": 24,
        "n_pass": 24,
        "scope": "temporary restored prototype package; not production root exports",
    }
