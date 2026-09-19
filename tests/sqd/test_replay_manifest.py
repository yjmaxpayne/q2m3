"""The replay bundle must survive loss or corruption of its source files."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_replay():
    spec = importlib.util.spec_from_file_location("sqd_replay", ROOT / "tools/sqd/replay/replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_covers_all_probes():
    module = load_replay()
    manifest = module.validate_manifest(ROOT / "tools/sqd/replay")
    assert set(manifest["probes"]) == {"h2", "reference", "transpose", "lazy"}


@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_manifest_rejects_bad_source(tmp_path, mutation):
    import shutil

    module = load_replay()
    shutil.copytree(ROOT / "tools/sqd/replay", tmp_path / "bundle")
    source = tmp_path / "bundle" / "reference_history.py"
    if mutation == "missing":
        source.unlink()
    else:
        source.write_text("corrupted")
    with pytest.raises((FileNotFoundError, ValueError)):
        module.validate_manifest(tmp_path / "bundle")


@pytest.mark.parametrize("mutation", ["missing_entry", "empty", "wrong_path", "escape", "symlink"])
def test_manifest_rejects_incomplete_or_escaping_inventory(tmp_path, mutation):
    import json
    import shutil

    module = load_replay()
    bundle = tmp_path / "bundle"
    shutil.copytree(ROOT / "tools/sqd/replay", bundle)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "empty":
        manifest["files"] = {}
    elif mutation == "symlink":
        target = tmp_path / "outside.py"
        source = bundle / "reference_history.py"
        shutil.copyfile(source, target)
        source.unlink()
        source.symlink_to(target)
    else:
        checksum = manifest["files"].pop("reference_history.py")
        if mutation == "wrong_path":
            manifest["files"]["absent.py"] = checksum
        elif mutation == "escape":
            shutil.copyfile(bundle / "reference_history.py", tmp_path / "outside.py")
            manifest["files"]["../outside.py"] = checksum
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        module.validate_manifest(bundle)


@pytest.mark.parametrize("mutation", ["none", "empty_exports", "always_hint", "empty_and_hint"])
def test_lazy_oracle_rejects_degenerate_helpers(tmp_path, mutation):
    import subprocess
    import sys

    replay = load_replay()
    lazy = replay.load_file("lazy_oracle_test", ROOT / "tools/sqd/replay/lazy_replay.py")
    package = tmp_path / "replay_package"
    package.mkdir()
    (package / "__init__.py").write_text(lazy.PACKAGE)
    helper = (ROOT / "tools/sqd/replay/lazy_helpers.py").read_text()
    if mutation in ("empty_exports", "empty_and_hint"):
        helper += "\ndef available_exports(lazy):\n    return []\n"
    if mutation in ("always_hint", "empty_and_hint"):
        helper += (
            "\ndef lazy_getattr(package, namespace, lazy, name):\n"
            "    if name not in lazy: raise AttributeError(name)\n"
            "    raise ImportError('uv sync --extra ' + lazy[name][1])\n"
        )
    (package / "lazy_helpers.py").write_text(helper)
    (package / "sqd.py").write_text("import ffsim, qiskit_addon_sqd\ndef run_sqd(): pass\n")
    (package / "solvation.py").write_text("import catalyst, jax\ndef run_solvation(): pass\n")
    for dependency in ("ffsim", "qiskit_addon_sqd", "catalyst", "jax"):
        (tmp_path / f"{dependency}.py").write_text("# lightweight dependency stub\n")
    process = subprocess.run(
        [sys.executable, "-c", lazy.CHILD, ""], cwd=tmp_path, capture_output=True, text=True
    )
    if mutation == "none":
        assert process.returncode == 0, process.stderr
    else:
        assert process.returncode != 0, "Degenerate lazy implementation escaped the oracle"


def test_lazy_profiles_preserve_shared_jax(tmp_path, monkeypatch):
    import subprocess

    replay = load_replay()
    lazy = replay.load_file("lazy_profiles_test", ROOT / "tools/sqd/replay/lazy_replay.py")
    actual_run = subprocess.run

    def run_with_dependency_stubs(command, **kwargs):
        root = Path(kwargs["cwd"])
        for dependency in ("qiskit_addon_sqd", "catalyst", "jax"):
            (root / f"{dependency}.py").write_text("# dependency stub\n")
        (root / "ffsim.py").write_text("import jax\n")
        return actual_run(command, **kwargs)

    monkeypatch.setattr(lazy.subprocess, "run", run_with_dependency_stubs)
    result = lazy.run()
    sqd_only = [p for p in result["profiles"] if p["expected_exports"] == ["run_sqd"]]
    assert len(sqd_only) == 1
    assert sqd_only[0]["blocked"] == ["catalyst"]
    assert result["n_pass"] == 24


def test_lazy_child_failure_prints_stderr(monkeypatch, capsys):
    import subprocess

    replay = load_replay()
    lazy = replay.load_file("lazy_failure_test", ROOT / "tools/sqd/replay/lazy_replay.py")

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="diagnostic-sentinel")

    monkeypatch.setattr(lazy.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        lazy.run()
    assert "diagnostic-sentinel" in capsys.readouterr().err
