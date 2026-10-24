"""Psi4 adapter: HF counterpoise energies from a separate Psi4 environment.

q2m3 does not import psi4. The adapter runs ``_psi4_driver.py`` with the interpreter of
the Psi4 environment and exchanges JSON files with it in a scratch directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyscf

from q2m3.fragments.counterpoise import SCF_CONV_TOL, counterpoise
from q2m3.fragments.errors import ExternalProgramError
from q2m3.fragments.external import ExternalProgramRunner, SubprocessRunner
from q2m3.fragments.molecule import Molecule

PSI4_PYTHON_ENV = "Q2M3_PSI4_PYTHON"
# Baicalein alone has 30 atoms, so a full baicalein dimer with any partner is rejected.
N_MAX_PSI4_ATOMS = 30
MAX_COMPARABLE_CONV_TOL = 1e-10
DUAL_TRACK_THRESHOLD_HA = 1e-8
EXACT_SCF_TYPES = frozenset({"pk", "direct", "out_of_core"})
IDENTITY_TIMEOUT_S = 120.0
DRIVER = Path(__file__).with_name("_psi4_driver.py")
# Psi4 runs in isolated mode (-I): sys.path gets neither PYTHONPATH nor the driver directory.
# The driver directory holds this module, psi4.py, which would shadow the psi4 package.
_IDENTITY_SCRIPT = "import psi4; print(psi4.__version__)"


@dataclass(frozen=True)
class HFCounterpoise:
    """HF counterpoise energies of one dimer from one program, in Ha.

    ``energies`` holds ``"AB"``, ``"A"`` and ``"B"``, all three in the dimer basis.
    ``scf_conv_tol`` is the loosest SCF convergence threshold the program used.
    """

    program: str
    basis: str
    geometry_sha256: str
    energies: Mapping[str, float]
    density_fitting: bool
    scf_conv_tol: float
    provenance: Mapping[str, Any]

    @property
    def delta_e(self) -> float:
        """E(AB) - E(A) - E(B)."""
        return self.energies["AB"] - self.energies["A"] - self.energies["B"]


class Psi4Adapter:
    """Runs Psi4 HF counterpoise calculations through an interpreter of a Psi4 environment.

    ``python_executable`` (or the ``Q2M3_PSI4_PYTHON`` variable when it is ``None``) must be
    the absolute path of that interpreter. Each calculation runs in a new scratch directory,
    so the working directory of the caller does not change.

    Raises:
        ExternalProgramError: No interpreter path is given, or the path is not absolute.
    """

    def __init__(
        self,
        python_executable: str | os.PathLike[str] | None = None,
        *,
        runner: ExternalProgramRunner | None = None,
        timeout_s: float = 3600.0,
    ) -> None:
        path = (
            python_executable if python_executable is not None else os.environ.get(PSI4_PYTHON_ENV)
        )
        if not path:
            raise ExternalProgramError(
                f"Set {PSI4_PYTHON_ENV} or give python_executable. "
                "The value must be the absolute path of the Psi4 interpreter."
            )
        if not os.path.isabs(path):
            raise ExternalProgramError(
                f"The Psi4 interpreter must be an absolute path, not {os.fspath(path)!r}."
            )
        self.python_executable = os.fspath(path)
        self.runner = runner if runner is not None else SubprocessRunner()
        self.timeout_s = timeout_s

    def hf_counterpoise(self, molecule: Molecule, *, basis: str = "cc-pvdz") -> HFCounterpoise:
        """Run RHF for the dimer and for each monomer in the dimer basis.

        Psi4 uses exact integrals (``scf_type pk``), spherical basis functions and
        convergence thresholds of ``SCF_CONV_TOL``.

        Raises:
            ValueError: The molecule does not have two fragments, or it has more than
                ``N_MAX_PSI4_ATOMS`` atoms.
            ExternalProgramError: The interpreter cannot import psi4, or the Psi4 run
                fails or exceeds ``timeout_s``.
        """
        _check_size(molecule)
        with tempfile.TemporaryDirectory(prefix="q2m3-psi4-") as scratch:
            env = {**os.environ, "PSI_SCRATCH": scratch}
            self._identify(scratch, env)
            job, output = Path(scratch, "job.json"), Path(scratch, "output.json")
            job.write_text(json.dumps(_job(molecule, basis)))
            argv = [self.python_executable, "-I", str(DRIVER), str(job), str(output)]
            completed = self._run(argv, scratch, env, self.timeout_s)
            if completed.returncode != 0:
                raise ExternalProgramError(
                    f"The Psi4 driver failed with exit code {completed.returncode} "
                    f"({self.python_executable}): {_reason(completed)}"
                )
            data = json.loads(output.read_text())
        return _psi4_record(molecule, basis, data, self.python_executable)

    def _identify(self, scratch: str, env: Mapping[str, str]) -> None:
        argv = [self.python_executable, "-I", "-c", _IDENTITY_SCRIPT]
        completed = self._run(argv, scratch, env, IDENTITY_TIMEOUT_S)
        if completed.returncode != 0:
            raise ExternalProgramError(
                f"The Psi4 interpreter {self.python_executable} cannot import psi4: "
                f"{_reason(completed)}"
            )

    def _run(
        self, argv: Sequence[str], cwd: str, env: Mapping[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self.runner.run(argv, cwd=cwd, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise ExternalProgramError(
                f"The Psi4 run did not finish in the time limit of {timeout} s "
                f"({self.python_executable})."
            ) from None
        except OSError as exc:
            raise ExternalProgramError(
                f"The adapter cannot run the Psi4 interpreter {self.python_executable}: {exc}"
            ) from exc


def pyscf_hf_counterpoise(molecule: Molecule, *, basis: str = "cc-pvdz") -> HFCounterpoise:
    """HF counterpoise energies from PySCF, with the SCF settings read from each RHF."""
    subsystems = counterpoise(molecule, basis=basis)
    return HFCounterpoise(
        program="pyscf",
        basis=basis,
        geometry_sha256=molecule.geometry_sha256(),
        energies={name: float(sub.scf.e_tot) for name, sub in subsystems.items()},
        density_fitting=any(
            getattr(sub.scf, "with_df", None) is not None for sub in subsystems.values()
        ),
        scf_conv_tol=max(sub.scf.conv_tol for sub in subsystems.values()),
        provenance={"pyscf_version": pyscf.__version__},
    )


def compare_hf_counterpoise(
    first: HFCounterpoise,
    second: HFCounterpoise,
    *,
    threshold_ha: float = DUAL_TRACK_THRESHOLD_HA,
) -> float:
    """Return ``first.delta_e - second.delta_e`` after the dual-track checks.

    Raises:
        ValueError: A record uses density fitting or an SCF convergence threshold above
            ``MAX_COMPARABLE_CONV_TOL``, the records differ in geometry or basis set, or
            the absolute difference is ``threshold_ha`` or more.
    """
    for record in (first, second):
        if record.density_fitting:
            raise ValueError(
                f"The comparison needs exact integrals. The {record.program} energies "
                "use density fitting."
            )
        if record.scf_conv_tol > MAX_COMPARABLE_CONV_TOL:
            raise ValueError(
                f"The comparison needs SCF convergence thresholds of "
                f"{MAX_COMPARABLE_CONV_TOL:.0e} or less. The {record.program} threshold "
                f"is {record.scf_conv_tol:.1e}."
            )
    if first.geometry_sha256 != second.geometry_sha256:
        raise ValueError(
            "The comparison needs one geometry. The records have different geometries."
        )
    if first.basis != second.basis:
        raise ValueError(
            "The comparison needs one basis set. The records have different basis sets: "
            f"{first.basis!r} and {second.basis!r}."
        )
    difference = first.delta_e - second.delta_e
    if abs(difference) >= threshold_ha:
        raise ValueError(
            f"The HF counterpoise energies differ by {difference:.3e} Ha. "
            f"The limit is {threshold_ha:.1e} Ha."
        )
    return difference


def _check_size(molecule: Molecule) -> None:
    if molecule.fragments is None:
        raise ValueError("The Psi4 adapter needs a molecule with two fragments.")
    n_atoms = len(molecule.symbols)
    if n_atoms > N_MAX_PSI4_ATOMS:
        raise ValueError(
            f"The Psi4 adapter accepts at most {N_MAX_PSI4_ATOMS} atoms. "
            f"The molecule has {n_atoms} atoms."
        )


def _job(molecule: Molecule, basis: str) -> dict[str, Any]:
    assert molecule.fragments is not None
    fragments = [
        {
            "charge": charge,
            "multiplicity": multiplicity,
            "atoms": [[molecule.symbols[i], *molecule.coordinates_angstrom[i]] for i in part],
        }
        for part, charge, multiplicity in zip(
            molecule.fragments, molecule.charges, molecule.multiplicities, strict=True
        )
    ]
    return {"basis": basis, "conv_tol": SCF_CONV_TOL, "fragments": fragments}


def _psi4_record(
    molecule: Molecule, basis: str, data: Mapping[str, Any], python_executable: str
) -> HFCounterpoise:
    options = data["options"]
    return HFCounterpoise(
        program="psi4",
        basis=basis,
        geometry_sha256=molecule.geometry_sha256(),
        energies=data["energies"],
        density_fitting=options["scf_type"].lower() not in EXACT_SCF_TYPES,
        scf_conv_tol=max(options["e_convergence"], options["d_convergence"]),
        provenance={
            "psi4_version": data["psi4_version"],
            "python_executable": python_executable,
            "options": options,
        },
    )


def _reason(completed: subprocess.CompletedProcess[str]) -> str:
    lines = [line for line in (completed.stderr or "").splitlines() if line.strip()]
    return lines[-1].strip() if lines else "no error output"
