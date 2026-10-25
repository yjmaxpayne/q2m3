"""Importing ``q2m3.fragments`` does not load optional quantum-chemistry backends."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
OPTIONAL_BACKENDS = ("psi4", "pyblock2", "block2", "ase")
# Executed by other interpreters: the SQD child process and the Psi4 environment.
EXCLUDED = frozenset({"q2m3.fragments._sqd_child", "q2m3.fragments.external._psi4_driver"})

WALK = """
import importlib, json, pkgutil, sys
stubs, excluded, backends = sys.argv[1], set(json.loads(sys.argv[2])), json.loads(sys.argv[3])
sys.path.insert(0, stubs)
import q2m3
import q2m3.fragments

def reraise(name):
    raise

imported = []
for info in pkgutil.walk_packages(q2m3.fragments.__path__, "q2m3.fragments.", onerror=reraise):
    if info.name not in excluded:
        importlib.import_module(info.name)
        imported.append(info.name)
loaded = sorted(set(backends) & set(sys.modules))
print(json.dumps({"origin": q2m3.__file__, "imported": imported, "loaded": loaded}))
"""


def test_excluded_modules_are_private():
    assert all(name.rpartition(".")[2].startswith("_") for name in EXCLUDED)


def test_fragment_modules_do_not_load_optional_backends(tmp_path):
    stubs = tmp_path / "stubs"
    for backend in OPTIONAL_BACKENDS:
        (stubs / backend).mkdir(parents=True)
        (stubs / backend / "__init__.py").touch()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC), env.get("PYTHONPATH")]))
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            WALK,
            str(stubs),
            json.dumps(sorted(EXCLUDED)),
            json.dumps(OPTIONAL_BACKENDS),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout.splitlines()[-1])
    assert Path(report["origin"]).is_relative_to(SRC), report["origin"]
    assert "q2m3.fragments.errors" in report["imported"]
    assert report["loaded"] == []


def test_sqd_worker_helpers_import_without_the_sqd_backend(tmp_path):
    script = """
import importlib.abc
import sys

blocked = {"ffsim", "qiskit", "qiskit_addon_sqd"}

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError("blocked: " + fullname, name=fullname)

sys.meta_path.insert(0, Block())
from q2m3.fragments import _sqd_child
assert callable(_sqd_child.host_available_mb)
assert not blocked.intersection(sys.modules)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC), env.get("PYTHONPATH")]))
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
