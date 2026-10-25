"""CLI evidence, fixed geometries, and real classical fragment calculations."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.fragments import (
    _common,
    _inputs,
    crystal_dimer_interaction,
    crystal_dimer_prepare,
    fragment_report,
    water_dimer_classical,
    water_dimer_full,
)
from q2m3.constants import HARTREE_TO_KCAL_MOL
from q2m3.fragments import Molecule

ROOT = Path(__file__).resolve().parents[2]
MODULES = (
    "water_dimer_classical",
    "water_dimer_full",
    "crystal_dimer_prepare",
    "crystal_dimer_interaction",
    "fragment_report",
)


def read_json(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("module", MODULES)
def test_module_help(module):
    completed = subprocess.run(
        [sys.executable, "-m", f"examples.fragments.{module}", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--output" in completed.stdout
    if module in ("water_dimer_full", "crystal_dimer_interaction", "fragment_report"):
        for option in ("--shots", "--seed", "--ccsd-t-wall-budget", "--rss-budget"):
            assert option in completed.stdout


@pytest.mark.parametrize(
    "module,mode",
    zip(
        MODULES,
        (
            "classical_only",
            "full",
            None,
            "classical_only",
            "classical_only",
        ),
        strict=True,
    ),
)
def test_entry_defaults(module, mode, monkeypatch):
    entry = importlib.import_module(f"examples.fragments.{module}")
    captured = {}

    def capture(args, name, calculate):
        captured.update(vars(args))
        return 0

    monkeypatch.setattr(entry, "execute", capture)
    assert entry.main([]) == 0
    assert captured.get("mode") == mode
    assert captured["output"] is None
    if "shots" in captured:
        assert captured["shots"] == 100_000
        assert captured["seed"] == 0
        assert captured["wall_budget"] == 900
        assert captured["rss_budget"] == 8192
        assert captured["ccsd_t_wall_budget"] == 7200
        assert captured["ccsd_t_rss_budget"] == 20000


@pytest.mark.parametrize(
    "option,value",
    [
        ("--seed", "-1"),
        ("--shots", "0"),
        ("--wall-budget", "nan"),
        ("--rss-budget", "inf"),
        ("--ccsd-t-wall-budget", "-1"),
    ],
)
def test_invalid_arguments_fail_before_output(option, value, tmp_path):
    output = tmp_path / "unused"
    with pytest.raises(SystemExit) as error:
        water_dimer_full.main([option, value, "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_exclusive_output_and_unique_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(_common, "OUTPUT_ROOT", tmp_path)
    args = argparse.Namespace(output=None)
    for _ in range(2):
        assert _common.execute(args, "new", lambda *_: None) == 0
    directories = list(tmp_path.iterdir())
    assert len(directories) == 2
    existing = directories[0]
    before = (existing / "run.json").read_bytes()

    def forbidden(*_):
        pytest.fail("Existing output must prevent execution")

    assert _common.execute(argparse.Namespace(output=existing), "new", forbidden) == 1
    assert (existing / "run.json").read_bytes() == before


def test_failure_preserves_input_and_traceback(tmp_path, monkeypatch):
    def fail(*_, **__):
        raise RuntimeError("test solver failure")

    monkeypatch.setattr("q2m3.fragments.run", fail)
    output = tmp_path / "failed"
    assert water_dimer_classical.main(["--output", str(output)]) == 1
    assert (output / "water-dimer.xyz").read_bytes() == (
        _inputs.DATA / "water-dimer.xyz"
    ).read_bytes()
    assert read_json(output / "input.json")["options"]["mode"] == "classical_only"
    failure = read_json(output / "failure.json")
    assert failure["exception_type"] == "RuntimeError"
    assert "test solver failure" in failure["traceback"]
    assert read_json(output / "run.json")["status"] == "failed"


def test_full_options_use_library_budgets(tmp_path, monkeypatch):
    captured = {}

    def capture(molecule, **options):
        captured.update(options)
        raise RuntimeError("stop before solver")

    monkeypatch.setattr("q2m3.fragments.run", capture)
    assert (
        water_dimer_full.main(
            [
                "--output",
                str(tmp_path / "full"),
                "--seed",
                "17",
                "--shots",
                "23",
                "--wall-budget",
                "45",
                "--rss-budget",
                "678",
                "--ccsd-t-wall-budget",
                "90",
                "--ccsd-t-rss-budget",
                "123",
            ]
        )
        == 1
    )
    assert captured["mode"] == "full"
    assert captured["seed"] == 17
    assert captured["lucj"].shots == 23
    assert captured["reference"].wall_budget_s == 45
    assert captured["reference"].rss_budget_mb == 678
    assert captured["ccsd_t_budget"].wall_s == 90
    assert captured["ccsd_t_budget"].rss_mb == 123


def test_bundled_inputs_match_repository_sources():
    for name, original in (
        ("water-dimer.xyz", ROOT / "doc/source/tutorials/water-dimer.xyz"),
        ("cod_2010678.cif", ROOT / "tests/fragments/data/cod_2010678.cif"),
    ):
        assert (_inputs.DATA / name).read_bytes() == original.read_bytes()


def test_crystal_structure_outputs(tmp_path):
    pytest.importorskip("ase")
    output = tmp_path / "prepare"
    assert crystal_dimer_prepare.main(["--output", str(output)]) == 0
    assert (output / "cod_2010678.cif").read_bytes() == (
        _inputs.DATA / "cod_2010678.cif"
    ).read_bytes()
    original = Molecule.from_xyz(output / "original.xyz")
    normalized = Molecule.from_xyz(output / "normalized.xyz")
    before = np.asarray(original.coordinates_angstrom)
    after = np.asarray(normalized.coordinates_angstrom)
    heavy = np.array(original.symbols) != "H"
    np.testing.assert_array_equal(before[heavy], after[heavy])
    fragments = read_json(output / "fragments.json")["fragments"]
    assert Counter(original.symbols[i] for i in fragments["A"]) == Counter(C=6, H=5, N=1, O=3)
    assert Counter(original.symbols[i] for i in fragments["B"]) == Counter(C=6, H=7, N=1)
    with (output / "xh_bonds.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    for row in rows:
        h, parent = int(row["h_index"]), int(row["parent_index"])
        length = np.linalg.norm(after[h] - after[parent])
        assert length == pytest.approx(float(row["target_angstrom"]), abs=1e-12)
        assert length == pytest.approx(float(row["after_angstrom"]), abs=1e-12)
        np.testing.assert_allclose(
            (before[h] - before[parent]) / float(row["before_angstrom"]),
            (after[h] - after[parent]) / length,
            atol=1e-12,
        )


@pytest.fixture(scope="module")
def classical(tmp_path_factory):
    pytest.importorskip("ase")
    from pyscf import lib

    previous = lib.num_threads()
    lib.num_threads(1)
    records = {}
    try:
        for name, module in (
            (_inputs.WATER_ID, water_dimer_classical),
            (_inputs.CRYSTAL_ID, crystal_dimer_interaction),
        ):
            directory = tmp_path_factory.mktemp(name)
            args = _common.parser("test", mode="classical_only", choose_mode=True).parse_args([])
            result = module.calculate(args, directory)
            records[name] = (result, directory)
        yield records
    finally:
        lib.num_threads(previous)


@pytest.mark.parametrize(
    "name,spaces",
    [
        (_inputs.WATER_ID, [(12, 8), (6, 4), (6, 4)]),
        (_inputs.CRYSTAL_ID, [(12, 9), (6, 4), (6, 5)]),
    ],
)
def test_real_classical_results_and_report_consistency(classical, name, spaces):
    result, directory = classical[name]
    assert result.validate() is result
    assert [
        (result.subsystems[s].active_space.nelecas, result.subsystems[s].active_space.ncas)
        for s in ("AB", "A", "B")
    ] == spaces
    payload = read_json(directory / "result.json")
    report = (directory / "report.md").read_text()
    with (directory / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    for row in rows:
        layer = row["method"]
        value = payload["delta_e"][layer]
        if value is None:
            assert row["delta_e_ha"] == ""
            assert row["null_reason"] == payload["null_reasons"][f"delta_e.{layer}"]
            assert row["null_reason"] in report
        else:
            energies = [payload["subsystems"][s][f"e_{layer}"] for s in ("AB", "A", "B")]
            assert value == pytest.approx(energies[0] - energies[1] - energies[2], abs=1e-12)
            assert float(row["delta_e_ha"]) == value
            kcal = value * HARTREE_TO_KCAL_MOL
            assert float(row["delta_e_kcal_mol"]) == pytest.approx(kcal)
            assert f"| {value:.6f} | {kcal:.3f} |" in report
    if name == _inputs.WATER_ID:
        assert result.delta_e["hf"] * HARTREE_TO_KCAL_MOL == pytest.approx(-3.682, abs=0.001)
        assert result.delta_e["mp2"] * HARTREE_TO_KCAL_MOL == pytest.approx(-3.935, abs=0.001)


@pytest.mark.parametrize("failed", [_inputs.WATER_ID, _inputs.CRYSTAL_ID, "both"])
def test_batch_partial_failure_continues(classical, tmp_path, monkeypatch, failed):
    visited = []

    def calculator(name):
        def calculate(args, directory, **kwargs):
            visited.append(name)
            _common.write_json(directory / "input.json", {"fragment_id": name})
            if failed in (name, "both"):
                raise RuntimeError("test item failure")
            return _common.save_result(directory, classical[name][0])

        return calculate

    monkeypatch.setattr(water_dimer_classical, "calculate", calculator(_inputs.WATER_ID))
    monkeypatch.setattr(crystal_dimer_interaction, "calculate", calculator(_inputs.CRYSTAL_ID))
    output = tmp_path / "batch"
    assert fragment_report.main(["--output", str(output)]) == 1
    assert visited == [_inputs.WATER_ID, _inputs.CRYSTAL_ID]
    aggregate = read_json(output / "results.json")
    failures = visited if failed == "both" else [failed]
    assert aggregate["failures"] == failures
    assert len(aggregate["results"]) == 2 - len(failures)
    for name in failures:
        assert (output / name / "failure.json").exists()
        assert f"{name}/failure.json" in (output / "report.md").read_text()


def test_reference_import_and_ks_report(classical, tmp_path):
    references = tmp_path / "references.csv"
    with references.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("fragment_id", "method", "basis", "delta_e", "unit"))
        for name, (result, _) in classical.items():
            # Reuse measured HF values to exercise import. These are not external benchmarks.
            writer.writerow((name, "HF", result.basis, result.delta_e["hf"], "Ha"))
    ks = tmp_path / "ks.json"
    ks.write_text('{"test-only measurement label": 1.25}')
    output = tmp_path / "references-run"
    assert (
        fragment_report.main(
            [
                "--output",
                str(output),
                "--references",
                str(references),
                "--ks-table",
                str(ks),
            ]
        )
        == 0
    )
    report = (output / "report.md").read_text()
    assert "| test-only measurement label | 1.25 |" in report
    assert "q2m3 不对 Ks 做统计结论。" in report
    assert "基组 cc-pvdz" in report and "基组 sto-3g" in report
    assert (output / "references.csv").read_bytes() == references.read_bytes()
    assert (output / "ks-table.json").read_bytes() == ks.read_bytes()
    for name, (result, _) in classical.items():
        saved = read_json(output / name / "result.json")
        assert saved["references"][0]["delta_e"] == result.delta_e["hf"]


def test_missing_reference_stops_before_calculations(tmp_path, monkeypatch):
    source = tmp_path / "missing.csv"
    source.write_text("fragment_id,method,basis,delta_e,unit\nwater-dimer,HF,cc-pvdz,0,Ha\n")

    def forbidden(*_, **__):
        pytest.fail("Missing coverage must prevent calculations")

    monkeypatch.setattr(water_dimer_classical, "calculate", forbidden)
    output = tmp_path / "missing"
    assert fragment_report.main(["--output", str(output), "--references", str(source)]) == 1
    assert _inputs.CRYSTAL_ID in read_json(output / "failure.json")["message"]
    assert not (output / _inputs.WATER_ID).exists()


@pytest.mark.parametrize("text", ["[]", '{"x": true}', '{"x": NaN}', '{"x": "2"}', '{"": 2}'])
def test_invalid_ks_json(text, tmp_path):
    source = tmp_path / "ks.json"
    source.write_text(text)
    with pytest.raises(ValueError, match="Ks JSON"):
        _common.load_ks(source)


def test_warnings_survive_all_outputs(classical, tmp_path):
    result = replace(classical[_inputs.WATER_ID][0], warnings=("test warning",))
    _common.save_result(tmp_path, result)
    assert read_json(tmp_path / "result.json")["warnings"] == ["test warning"]
    assert "test warning" in (tmp_path / "summary.csv").read_text()
    assert "test warning" in (tmp_path / "report.md").read_text()


def test_subsystem_warnings_survive_summary_and_report(classical, tmp_path):
    result = classical[_inputs.WATER_ID][0]
    # A lightweight stand-in exercises report preservation without another SQD run.
    subsystem = replace(
        result.subsystems["AB"], sqd=SimpleNamespace(warnings=("test SQD warning",))
    )
    report_input = replace(result, subsystems=dict(result.subsystems) | {"AB": subsystem})
    _common.save_report(tmp_path, [report_input])
    assert "AB: test SQD warning" in (tmp_path / "summary.csv").read_text()
    assert "AB: test SQD warning" in (tmp_path / "report.md").read_text()


def test_ks_report_rejection_stops_before_calculations(tmp_path, monkeypatch):
    source = tmp_path / "ks.json"
    source.write_text('{"quantum advantage": 1}')

    def forbidden(*_, **__):
        pytest.fail("Rejected report labels must prevent calculations")

    monkeypatch.setattr(water_dimer_classical, "calculate", forbidden)
    output = tmp_path / "rejected-ks"
    assert fragment_report.main(["--output", str(output), "--ks-table", str(source)]) == 1
    assert "banned term" in read_json(output / "failure.json")["message"]
    assert not (output / _inputs.WATER_ID).exists()
