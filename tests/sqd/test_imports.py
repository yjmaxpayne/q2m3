"""Optional exports are consistent in fresh Python interpreters."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

IMPORT_PROFILE = r"""
import builtins
import importlib.abc
import importlib.util
import json
import sys

blocked = set(json.loads(sys.argv[1]))
has_catalyst = all(
    name not in blocked and importlib.util.find_spec(name) is not None
    for name in ('catalyst', 'jax')
)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('blocked: ' + fullname, name=fullname)
sys.meta_path.insert(0, Block())

imports = []
original_import = builtins.__import__
def trace_import(name, *args, **kwargs):
    imports.append(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = trace_import

import q2m3
root_modules = sorted(sys.modules)
import q2m3.sqd as sqd
import q2m3.solvation as solvation
packages = (q2m3, sqd, solvation)
assert all(package.__name__ in sys.modules for package in packages)
assert 'q2m3.solvation.orchestrator' not in sys.modules, 'eager orchestrator'
for name in imports + list(sys.modules):
    assert name.split('.')[0] not in ('ffsim', 'qiskit', 'qiskit_addon_sqd'), name
optional = {
    'run_solvation', 'replay_quantum_trajectory', 'MoleculeConfig',
    'QPEConfig', 'SolvationConfig', 'SolventModel', 'TIP3P_WATER', 'SPC_E_WATER',
}
assert ('run_solvation' in q2m3.__all__) == has_catalyst
assert (optional & set(solvation.__all__)) == (optional if has_catalyst else set())
for package in packages:
    assert set(package.__all__) <= set(dir(package))
    namespace = {}
    exec('from ' + package.__name__ + ' import *', namespace)
    assert set(namespace) - {'__builtins__'} == set(package.__all__)
    try:
        getattr(package, 'unknown_export')
    except AttributeError as exc:
        assert 'unknown_export' in str(exc)
    else:
        raise AssertionError('unknown attribute resolved')
for package, names in ((q2m3, {'run_solvation'}), (solvation, optional)):
    for name in names:
        if has_catalyst:
            value = getattr(package, name)
            assert value is package.__dict__[name], 'resolved value not cached'
        else:
            try:
                getattr(package, name)
            except ImportError as exc:
                assert 'uv sync --extra catalyst' in str(exc)
                assert package.__name__ + '.' + name in str(exc)
            else:
                raise AssertionError('unavailable export resolved: ' + name)
if has_catalyst:
    from q2m3.solvation.orchestrator import run_solvation
    assert callable(q2m3.run_solvation)
    assert q2m3.run_solvation is solvation.run_solvation is run_solvation
print(json.dumps({'checks': 6, 'blocked': sorted(blocked),
                  'catalyst_available': has_catalyst, 'root_modules': root_modules}))
"""


def run_import_profile(blocked: tuple[str, ...], source: Path | None = None) -> dict:
    """Check a package tree without reusing the pytest process's imports."""
    env = dict(os.environ)
    if source is not None:
        env["PYTHONPATH"] = str(source)
    process = subprocess.run(
        [sys.executable, "-c", IMPORT_PROFILE, json.dumps(blocked)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return json.loads(process.stdout.splitlines()[-1])


@pytest.mark.parametrize(
    "blocked",
    [(), ("ffsim", "qiskit_addon_sqd"), ("catalyst",), ("ffsim", "qiskit_addon_sqd", "catalyst")],
    ids=["installed", "without-sqd", "without-catalyst", "without-extras"],
)
def test_public_import_profile(blocked):
    assert run_import_profile(blocked)["checks"] == 6


@pytest.mark.parametrize(
    "extra, missing",
    [("sqd", "ffsim"), ("sqd", "qiskit_addon_sqd"), ("catalyst", "catalyst"), ("catalyst", "jax")],
)
@pytest.mark.parametrize("failure", [None, ImportError, ValueError])
def test_missing_probe_excludes_exports_and_prevents_loading(monkeypatch, extra, missing, failure):
    from q2m3 import _lazy

    def find_spec(name):
        if name == missing:
            if failure is not None:
                raise failure("unavailable probe")
            return None
        return object()

    def unexpected_import(name):
        pytest.fail(f"Loaded {name} before checking the extra")

    monkeypatch.setattr(_lazy.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(_lazy.importlib, "import_module", unexpected_import)
    exports = {"entry": ("optional_backend", extra)}
    assert not _lazy.extra_available(extra)
    assert _lazy.available_exports(exports) == []
    namespace = {}
    with pytest.raises(ImportError, match=f"uv sync --extra {extra}"):
        _lazy.lazy_getattr("package", namespace, exports, "entry")
    assert namespace == {}


def test_available_probe_does_not_import_and_resolution_caches_value(monkeypatch):
    from q2m3 import _lazy

    calls = []
    value = object()
    monkeypatch.setattr(_lazy.importlib.util, "find_spec", lambda name: object())

    def import_module(name):
        calls.append(name)
        return SimpleNamespace(entry=value)

    monkeypatch.setattr(_lazy.importlib, "import_module", import_module)
    exports = {"entry": ("optional_backend", "sqd")}
    assert _lazy.extra_available("sqd")
    assert _lazy.available_exports(exports) == ["entry"]
    assert calls == []
    namespace = {}
    assert _lazy.lazy_getattr("package", namespace, exports, "entry") is value
    assert namespace["entry"] is value
    assert calls == ["optional_backend"]


@pytest.mark.parametrize(
    "error",
    [
        ImportError("broken backend"),
        ModuleNotFoundError("internal module", name="internal_module"),
        AttributeError("missing entry"),
    ],
)
def test_backend_errors_are_preserved(monkeypatch, error):
    from q2m3 import _lazy

    monkeypatch.setattr(_lazy.importlib.util, "find_spec", lambda name: object())

    def import_module(name):
        raise error

    monkeypatch.setattr(_lazy.importlib, "import_module", import_module)
    namespace = {}
    with pytest.raises(type(error)) as caught:
        _lazy.lazy_getattr("package", namespace, {"entry": ("backend", "sqd")}, "entry")
    assert caught.value is error
    assert namespace == {}


def test_unknown_attribute_does_not_probe_or_import(monkeypatch):
    from q2m3 import _lazy

    def unexpected_call(*args):
        pytest.fail("Unknown attributes must not load optional backends")

    monkeypatch.setattr(_lazy.importlib.util, "find_spec", unexpected_call)
    monkeypatch.setattr(_lazy.importlib, "import_module", unexpected_call)
    with pytest.raises(AttributeError, match="module 'package' has no attribute 'unknown'"):
        _lazy.lazy_getattr("package", {}, {}, "unknown")
