"""Psi4 adapter: scratch directories, interpreter discovery, size limit and the dual-track check.

Most tests replace the Psi4 interpreter with ``FakePsi4``. Tests marked ``external`` run the
real Psi4 environment. Local reproduction (from the checkout root):

    Q2M3_PSI4_PYTHON=/abs/path/to/psi4-env/bin/python \
        uv run --no-sync pytest tests/fragments/test_psi4_adapter.py -m external -n 0
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pyscf import scf

import q2m3.fragments.counterpoise as counterpoise_module
from q2m3.fragments import ExternalProgramError, Molecule
from q2m3.fragments.counterpoise import counterpoise
from q2m3.fragments.external import psi4 as psi4_module
from q2m3.fragments.external.psi4 import (
    N_MAX_PSI4_ATOMS,
    HFCounterpoise,
    Psi4Adapter,
    compare_hf_counterpoise,
    pyscf_hf_counterpoise,
)
from tests.fragments.conftest import WATER_DIMER

PSI4_PATH = "/opt/psi4/bin/python"
DRIVER = Path(psi4_module.__file__).with_name("_psi4_driver.py")
ENERGIES = {"AB": -152.0, "A": -76.0, "B": -75.99}
EXACT_OPTIONS = {"scf_type": "pk", "puream": True, "e_convergence": 1e-10, "d_convergence": 1e-10}

# S66 system 18 (water...pyridine) in Angstrom, as used by the Psi4 versus PySCF prototype runs.
S66_18_WATER = (
    ("O", (-0.55283102, -0.10169749, -0.00049879)),
    ("H", (-0.87175963, 0.8017922, 0.0001444)),
    ("H", (0.4126595, -0.00183225, -0.00025181)),
)
S66_18_PYRIDINE = (
    ("N", (2.36402099, 0.09662268, 0.0001468)),
    ("C", (3.05992763, 0.06265189, 1.14489465)),
    ("H", (2.47525508, 0.08626283, 2.05576267)),
    ("C", (4.44895122, -0.00253054, 1.19489071)),
    ("H", (4.9548576, -0.0273847, 2.14921983)),
    ("C", (5.16011436, -0.03565634, -2.044e-05)),
    ("H", (6.23995431, -0.08742989, -0.00010086)),
    ("C", (4.44880607, -0.0025972, -1.19482173)),
    ("H", (4.95460301, -0.02747022, -2.14922033)),
    ("C", (3.05977605, 0.06259779, -1.14467547)),
    ("H", (2.47500717, 0.08619845, -2.05546803)),
)


@dataclass(frozen=True)
class Call:
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    timeout: float


@dataclass
class FakePsi4:
    """Stands in for the Psi4 interpreter.

    Like ``import psi4``, every run writes ``timer.dat`` into its working directory.
    """

    identity_returncode: int = 0
    identity_stderr: str = ""
    driver_returncode: int = 0
    driver_stderr: str = ""
    driver_error: BaseException | None = None
    options: dict = field(default_factory=lambda: dict(EXACT_OPTIONS))
    calls: list[Call] = field(default_factory=list)
    inputs: list[dict] = field(default_factory=list)

    def run(self, argv, *, cwd, env, timeout):
        self.calls.append(Call(tuple(argv), Path(cwd), dict(env), timeout))
        Path(cwd, "timer.dat").write_text("psi4 timers\n")
        if "-c" in argv:
            return subprocess.CompletedProcess(
                argv, self.identity_returncode, "1.11\n", self.identity_stderr
            )
        if self.driver_error is not None:
            raise self.driver_error
        self.inputs.append(json.loads(Path(argv[-2]).read_text()))
        output = {"psi4_version": "1.11", "energies": ENERGIES, "options": self.options}
        Path(argv[-1]).write_text(json.dumps(output))
        return subprocess.CompletedProcess(argv, self.driver_returncode, "", self.driver_stderr)


def _xyz(atoms) -> str:
    lines = [f"{symbol} {x} {y} {z}" for symbol, (x, y, z) in atoms]
    return f"{len(lines)}\n\n" + "\n".join(lines)


def _dimer(atoms_a, atoms_b, **kwargs) -> Molecule:
    n_a = len(atoms_a)
    fragments = (range(n_a), range(n_a, n_a + len(atoms_b)))
    return Molecule.from_xyz(_xyz((*atoms_a, *atoms_b)), fragments=fragments, **kwargs)


def _chain_dimer(formula_a: str, formula_b: str) -> Molecule:
    """Dimer with the given element counts; atoms sit 1.5 Angstrom apart on the x axis."""
    symbols = [_expand(formula_a), _expand(formula_b)]
    atoms = [
        (symbol, (1.5 * i, 0.0, 0.0))
        for i, symbol in enumerate(symbol for part in symbols for symbol in part)
    ]
    return _dimer(atoms[: len(symbols[0])], atoms[len(symbols[0]) :])


def _expand(formula: str) -> list[str]:
    """``"C6H6O3"`` -> six C, six H and three O symbols."""
    symbols, element, digits = [], "", ""
    for char in formula + "X":
        if char.isupper():
            if element:
                symbols += [element] * int(digits or "1")
            element, digits = char, ""
        elif char.islower():
            element += char
        else:
            digits += char
    return symbols


@pytest.fixture
def water() -> Molecule:
    return _dimer(WATER_DIMER[:3], WATER_DIMER[3:])


@pytest.fixture
def fake() -> FakePsi4:
    return FakePsi4()


def _adapter(fake: FakePsi4, **kwargs) -> Psi4Adapter:
    return Psi4Adapter(PSI4_PATH, runner=fake, **kwargs)


def _record(program: str = "psi4", *, shift: float = 0.0, **changes) -> HFCounterpoise:
    energies = {**ENERGIES, "AB": ENERGIES["AB"] + shift}
    base = HFCounterpoise(
        program=program,
        basis="cc-pvdz",
        geometry_sha256="0" * 64,
        energies=energies,
        density_fitting=False,
        scf_conv_tol=1e-10,
        provenance={},
    )
    return dataclasses.replace(base, **changes)


# --- scratch directory --------------------------------------------------------------------


def test_adapter_leaves_the_caller_directory_unchanged(tmp_path, monkeypatch, fake, water):
    caller = tmp_path / "caller"
    caller.mkdir()
    (caller / "input.xyz").write_text("kept\n")
    monkeypatch.chdir(caller)
    before = sorted(os.listdir(caller))

    _adapter(fake).hf_counterpoise(water)

    assert sorted(os.listdir(caller)) == before
    # The fake did write timer.dat; it went to the scratch directory, not to the caller.
    assert {call.cwd for call in fake.calls} and caller not in {call.cwd for call in fake.calls}


def test_one_call_runs_in_one_removed_scratch_directory(fake, water):
    _adapter(fake).hf_counterpoise(water)

    scratch = {call.cwd for call in fake.calls}
    assert len(scratch) == 1 and len(fake.calls) == 2
    (directory,) = scratch
    assert directory.name.startswith("q2m3-psi4-")
    assert all(call.env["PSI_SCRATCH"] == str(directory) for call in fake.calls)
    assert not directory.exists()


# --- driver input and record --------------------------------------------------------------


def test_adapter_returns_the_psi4_energies_and_settings(fake, water):
    record = _adapter(fake).hf_counterpoise(water)

    assert record.program == "psi4"
    assert record.energies == ENERGIES
    assert record.delta_e == ENERGIES["AB"] - ENERGIES["A"] - ENERGIES["B"]
    assert record.density_fitting is False
    assert record.scf_conv_tol == 1e-10
    assert record.basis == "cc-pvdz"
    assert record.geometry_sha256 == water.geometry_sha256()
    assert record.provenance == {
        "psi4_version": "1.11",
        "python_executable": PSI4_PATH,
        "options": EXACT_OPTIONS,
    }


@pytest.mark.parametrize(
    ("scf_type", "density_fitting"),
    [("pk", False), ("direct", False), ("out_of_core", False), ("df", True), ("cd", True)],
)
def test_psi4_record_marks_density_fitting_from_the_scf_type(
    fake, water, scf_type, density_fitting
):
    fake.options["scf_type"] = scf_type

    assert _adapter(fake).hf_counterpoise(water).density_fitting is density_fitting


def test_psi4_record_keeps_the_looser_convergence_setting(fake, water):
    fake.options["d_convergence"] = 1e-6

    assert _adapter(fake).hf_counterpoise(water).scf_conv_tol == 1e-6


def test_driver_input_holds_the_fragments_basis_and_convergence(fake):
    molecule = _dimer(WATER_DIMER[3:], WATER_DIMER[:3], charges=(2, -2))

    _adapter(fake).hf_counterpoise(molecule, basis="sto-3g")

    (job,) = fake.inputs
    assert job["basis"] == "sto-3g"
    assert job["conv_tol"] == 1e-10
    assert job["fragments"] == [
        {"charge": 2, "multiplicity": 1, "atoms": [[s, *xyz] for s, xyz in WATER_DIMER[3:]]},
        {"charge": -2, "multiplicity": 1, "atoms": [[s, *xyz] for s, xyz in WATER_DIMER[:3]]},
    ]
    # Isolated mode (-I) keeps the driver directory and PYTHONPATH out of sys.path. The
    # driver directory holds the adapter module psi4.py, which would shadow the psi4 package.
    assert fake.calls[0].argv[:3] == (PSI4_PATH, "-I", "-c")
    assert fake.calls[1].argv[:3] == (PSI4_PATH, "-I", str(DRIVER))


def test_adapter_forwards_the_time_limit_to_the_driver_run(fake, water):
    _adapter(fake, timeout_s=12.5).hf_counterpoise(water)

    assert fake.calls[1].timeout == 12.5


# --- interpreter discovery ----------------------------------------------------------------


def test_adapter_needs_an_interpreter_path(monkeypatch):
    monkeypatch.delenv("Q2M3_PSI4_PYTHON", raising=False)

    with pytest.raises(ExternalProgramError, match="Q2M3_PSI4_PYTHON"):
        Psi4Adapter()


@pytest.mark.parametrize("path", ["envs/psi4/bin/python", "python"])
def test_adapter_rejects_a_relative_path_or_a_bare_name(path):
    with pytest.raises(ExternalProgramError, match="absolute path"):
        Psi4Adapter(path)


def test_environment_variable_names_the_interpreter(monkeypatch, fake, water):
    monkeypatch.setenv("Q2M3_PSI4_PYTHON", "/env/psi4/bin/python")

    Psi4Adapter(runner=fake).hf_counterpoise(water)

    assert {call.argv[0] for call in fake.calls} == {"/env/psi4/bin/python"}


def test_argument_takes_precedence_over_the_environment_variable(monkeypatch, fake, water):
    monkeypatch.setenv("Q2M3_PSI4_PYTHON", "/env/psi4/bin/python")

    _adapter(fake).hf_counterpoise(water)

    assert {call.argv[0] for call in fake.calls} == {PSI4_PATH}


def test_failed_identity_check_names_the_path_and_the_reason(fake, water):
    fake.identity_returncode = 1
    fake.identity_stderr = "Traceback ...\nModuleNotFoundError: No module named 'psi4'\n"

    with pytest.raises(ExternalProgramError, match="cannot import psi4") as caught:
        _adapter(fake).hf_counterpoise(water)

    assert PSI4_PATH in str(caught.value)
    assert "No module named 'psi4'" in str(caught.value)
    assert len(fake.calls) == 1


def test_interpreter_without_psi4_is_rejected_by_a_real_run(tmp_path, water):
    # A stand-in interpreter that fails like a Python without psi4, independent of the
    # packages installed in the test environment.
    interpreter = tmp_path / "python"
    interpreter.write_text(
        "#!/bin/sh\necho \"ModuleNotFoundError: No module named 'psi4'\" >&2\nexit 1\n"
    )
    interpreter.chmod(0o755)

    with pytest.raises(ExternalProgramError, match="cannot import psi4"):
        Psi4Adapter(str(interpreter)).hf_counterpoise(water)


def test_missing_interpreter_file_is_rejected_by_a_real_run(tmp_path, water):
    with pytest.raises(ExternalProgramError, match="cannot run"):
        Psi4Adapter(str(tmp_path / "no-such-python")).hf_counterpoise(water)


# --- driver failures ----------------------------------------------------------------------


def test_failed_driver_run_names_the_exit_code_and_the_reason(fake, water):
    fake.driver_returncode = 3
    fake.driver_stderr = "PsiException: SCF did not converge\n"

    with pytest.raises(ExternalProgramError, match="driver failed") as caught:
        _adapter(fake).hf_counterpoise(water)

    assert "exit code 3" in str(caught.value)
    assert "SCF did not converge" in str(caught.value)


def test_driver_run_over_the_time_limit_is_an_external_program_error(fake, water):
    fake.driver_error = subprocess.TimeoutExpired(cmd="psi4", timeout=12.5)

    with pytest.raises(ExternalProgramError, match="time limit"):
        _adapter(fake, timeout_s=12.5).hf_counterpoise(water)


# --- size limit ---------------------------------------------------------------------------


def test_size_limit_allows_the_pyrogallol_nicotinamide_dimer(fake):
    molecule = _chain_dimer("C6H6O3", "C6H6N2O")
    assert len(molecule.symbols) == N_MAX_PSI4_ATOMS == 30

    _adapter(fake).hf_counterpoise(molecule)

    assert len(fake.calls) == 2


@pytest.mark.parametrize(
    ("formula_a", "formula_b", "n_atoms"),
    [("C6H6O3", "C6H7N2O", 31), ("C15H10O5", "C6H6N2O", 45)],
    ids=["one-above-limit", "baicalein-nicotinamide"],
)
def test_size_limit_rejects_larger_dimers_before_any_run(fake, formula_a, formula_b, n_atoms):
    molecule = _chain_dimer(formula_a, formula_b)
    assert len(molecule.symbols) == n_atoms

    with pytest.raises(ValueError, match=f"at most 30 atoms.*has {n_atoms} atoms"):
        _adapter(fake).hf_counterpoise(molecule)

    assert fake.calls == []


def test_adapter_needs_two_fragments(fake):
    molecule = Molecule.from_xyz(_xyz(WATER_DIMER[:3]))

    with pytest.raises(ValueError, match="two fragments"):
        _adapter(fake).hf_counterpoise(molecule)

    assert fake.calls == []


# --- dual-track comparison ----------------------------------------------------------------


@pytest.mark.parametrize("shift", [0.9e-8, -0.9e-8])
def test_comparison_accepts_a_difference_below_the_limit(shift):
    difference = compare_hf_counterpoise(_record(shift=shift), _record("pyscf"))

    assert difference == pytest.approx(shift, abs=1e-12)


@pytest.mark.parametrize("shift", [1.1e-8, -1.1e-8])
def test_comparison_rejects_a_difference_above_the_limit(shift):
    with pytest.raises(ValueError, match="differ by"):
        compare_hf_counterpoise(_record(shift=shift), _record("pyscf"))


@pytest.mark.parametrize("side", [0, 1])
def test_comparison_rejects_loose_convergence_on_either_side(side):
    records = [_record(), _record("pyscf")]
    records[side] = dataclasses.replace(records[side], scf_conv_tol=1e-6)

    with pytest.raises(ValueError, match="convergence"):
        compare_hf_counterpoise(*records)


@pytest.mark.parametrize("side", [0, 1])
def test_comparison_rejects_density_fitting_on_either_side(side):
    records = [_record(), _record("pyscf")]
    records[side] = dataclasses.replace(records[side], density_fitting=True)

    with pytest.raises(ValueError, match="density fitting"):
        compare_hf_counterpoise(*records)


def test_comparison_rejects_different_geometries():
    with pytest.raises(ValueError, match="different geometries"):
        compare_hf_counterpoise(_record(), _record("pyscf", geometry_sha256="1" * 64))


def test_comparison_rejects_different_basis_sets():
    with pytest.raises(ValueError, match="different basis sets"):
        compare_hf_counterpoise(_record(), _record("pyscf", basis="aug-cc-pvdz"))


# --- PySCF track --------------------------------------------------------------------------


def test_pyscf_record_holds_the_counterpoise_energies(water):
    import pyscf

    record = pyscf_hf_counterpoise(water, basis="sto-3g")
    subsystems = counterpoise(water, basis="sto-3g")

    assert record.program == "pyscf"
    # Two SCF runs agree to the convergence threshold, not bit for bit, with several threads.
    expected = {name: sub.scf.e_tot for name, sub in subsystems.items()}
    assert record.energies == pytest.approx(expected, abs=1e-9)
    assert record.density_fitting is False
    assert record.scf_conv_tol == 1e-10
    assert record.basis == "sto-3g"
    assert record.geometry_sha256 == water.geometry_sha256()
    assert record.provenance == {"pyscf_version": pyscf.__version__}


def test_loose_pyscf_convergence_is_rejected(monkeypatch, water):
    monkeypatch.setattr(counterpoise_module, "SCF_CONV_TOL", 1e-6)
    loose = pyscf_hf_counterpoise(water, basis="sto-3g")
    exact = dataclasses.replace(loose, program="psi4", scf_conv_tol=1e-10)

    assert loose.scf_conv_tol == 1e-6
    with pytest.raises(ValueError, match="convergence"):
        compare_hf_counterpoise(exact, loose)


def test_density_fitted_pyscf_is_rejected(monkeypatch, water):
    exact_rhf = scf.RHF
    monkeypatch.setattr(scf, "RHF", lambda mol: exact_rhf(mol).density_fit())
    fitted = pyscf_hf_counterpoise(water, basis="sto-3g")
    exact = dataclasses.replace(fitted, program="psi4", density_fitting=False)

    assert fitted.density_fitting is True
    with pytest.raises(ValueError, match="density fitting"):
        compare_hf_counterpoise(exact, fitted)


# --- real Psi4 environment ----------------------------------------------------------------


@pytest.mark.external
def test_s66_18_psi4_and_pyscf_agree_and_the_caller_directory_is_unchanged(tmp_path, monkeypatch):
    molecule = _dimer(S66_18_WATER, S66_18_PYRIDINE, name="S66-18")
    monkeypatch.chdir(tmp_path)

    psi4_record = Psi4Adapter().hf_counterpoise(molecule)
    pyscf_record = pyscf_hf_counterpoise(molecule)
    difference = compare_hf_counterpoise(psi4_record, pyscf_record)

    assert abs(difference) < 1e-8
    # Swapped monomer labels leave the interaction energy unchanged, so check each energy.
    assert psi4_record.energies == pytest.approx(pyscf_record.energies, abs=1e-8)
    assert psi4_record.density_fitting is False
    assert os.listdir(tmp_path) == []


@pytest.mark.external
def test_charged_fragments_agree_with_pyscf():
    # H3O+ ... OH-: a ghost fragment must carry neither the charge nor the electrons of its
    # real counterpart. Both monomers have 10 electrons, so both are closed-shell singlets.
    molecule = Molecule.from_xyz(
        _xyz(WATER_DIMER), fragments=([0, 1, 2, 4], [3, 5]), charges=(1, -1)
    )

    psi4_record = Psi4Adapter().hf_counterpoise(molecule, basis="sto-3g")
    pyscf_record = pyscf_hf_counterpoise(molecule, basis="sto-3g")
    difference = compare_hf_counterpoise(psi4_record, pyscf_record)

    assert abs(difference) < 1e-8
    assert psi4_record.energies == pytest.approx(pyscf_record.energies, abs=1e-8)


@pytest.mark.external
def test_importing_psi4_writes_timer_dat_into_the_working_directory(tmp_path):
    # The hazard the scratch directory exists for: a bare Psi4 import writes into its cwd.
    subprocess.run([os.environ["Q2M3_PSI4_PYTHON"], "-c", "import psi4"], cwd=tmp_path, check=True)

    assert (tmp_path / "timer.dat").exists()
