"""Dispersion-corrected geometry optimization and fragment identity guards."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pyscf import geomopt

from q2m3._lazy import extra_available
from q2m3.fragments import structure
from q2m3.fragments.molecule import Molecule


@pytest.fixture
def dimer(water_dimer):
    return Molecule(
        tuple(symbol for symbol, _ in water_dimer),
        tuple(xyz for _, xyz in water_dimer),
        ((0, 1, 2), (3, 4, 5)),
        (0, 0),
        (1, 1),
        "water dimer",
    )


@pytest.fixture
def solver(monkeypatch):
    """Replace only the optimizer; keep PySCF's molecule and DFT configuration real."""
    state = types.SimpleNamespace(
        converged=True, scf_converged=True, summary={"dispersion": -0.002}, calls=[], callback=True
    )
    module = types.ModuleType("pyscf.geomopt.geometric_solver")

    def kernel(method, **kwargs):
        state.calls.append((method, kwargs))
        method.scf_summary = state.summary
        method.disp = getattr(state, "dispersion", method.disp)
        scanner = types.SimpleNamespace(base=method, converged=state.scf_converged)
        if state.callback:
            kwargs["callback"]({"g_scanner": scanner})
        optimized = method.mol.copy()
        xyz = optimized.atom_coords(unit="Angstrom")
        xyz[1, 1] += 0.02
        optimized.set_geom_(xyz, unit="Angstrom")
        return state.converged, getattr(state, "result", optimized)

    module.kernel = kernel
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(geomopt, "geometric_solver", module, raising=False)
    monkeypatch.setattr(structure, "extra_available", lambda extra: extra == "geometry")
    return state


@pytest.mark.parametrize("charges", [(0, 0), (1, 1)])
def test_optimizer_preserves_identity_and_records_the_effective_method(dimer, solver, charges):
    dimer = replace(dimer, charges=charges)
    original = dimer.geometry_sha256()
    result = structure.optimize_fragment_geometry(
        dimer, functional="pbe", basis="sto-3g", maxsteps=7
    )
    method, options = solver.calls[0]
    assert method.xc == "pbe"
    assert method.disp == "d3bj"
    assert method.mol.basis == "sto-3g"
    assert method.mol.charge == sum(charges)
    assert method.mol.spin == 0
    np.testing.assert_allclose(method.mol.atom_coords(unit="Angstrom"), dimer.coordinates_angstrom)
    assert options["maxsteps"] == 7
    assert options["assert_convergence"] is True
    assert result.molecule == replace(
        dimer, coordinates_angstrom=result.molecule.coordinates_angstrom
    )
    assert result.molecule.coordinates_angstrom != dimer.coordinates_angstrom
    assert dimer.geometry_sha256() == original
    assert result.provenance["functional"] == "pbe"
    assert result.provenance["basis"] == "sto-3g"
    assert result.provenance["dispersion"] == "D3(BJ)"
    assert result.provenance["converged"] is True
    assert result.provenance["scf_converged"] is True
    assert result.provenance["dispersion_energy_hartree"] == -0.002
    assert result.provenance["maxsteps"] == 7
    json.dumps(result.provenance, allow_nan=False)


def test_unconverged_optimization_is_recorded_without_claiming_success(dimer, solver):
    solver.converged = False
    result = structure.optimize_fragment_geometry(dimer)
    assert result.provenance["converged"] is False
    assert result.provenance["scf_converged"] is True


@pytest.mark.parametrize("dispersion", [None, "", "d4", "d3zero"])
def test_dispersion_cannot_be_disabled_or_replaced(dimer, solver, dispersion):
    with pytest.raises(ValueError, match="D3\\(BJ\\)"):
        structure.optimize_fragment_geometry(dimer, dispersion=dispersion)
    assert not solver.calls


@pytest.mark.parametrize("summary", [{}, {"dispersion": float("nan")}])
def test_missing_or_nonfinite_dispersion_evidence_is_rejected(dimer, solver, summary):
    solver.summary = summary
    with pytest.raises(ValueError, match="dispersion"):
        structure.optimize_fragment_geometry(dimer)


def test_unconverged_scf_is_rejected(dimer, solver):
    solver.scf_converged = False
    with pytest.raises(RuntimeError, match="SCF.*converge"):
        structure.optimize_fragment_geometry(dimer)


def test_backend_cannot_replace_requested_dispersion(dimer, solver):
    solver.dispersion = "d4"
    with pytest.raises(ValueError, match="dispersion evidence"):
        structure.optimize_fragment_geometry(dimer)


def test_backend_must_evaluate_the_dispersion_corrected_gradient(dimer, solver):
    solver.callback = False
    with pytest.raises(ValueError, match="no D3\\(BJ\\) dispersion evidence"):
        structure.optimize_fragment_geometry(dimer)


def test_functional_cannot_embed_a_different_dispersion_scheme(dimer, solver):
    with pytest.raises(ValueError, match="dispersion suffix"):
        structure.optimize_fragment_geometry(dimer, functional="pbe-d4")
    assert not solver.calls


def test_hydrogen_transfer_changes_fragment_membership_and_is_rejected(dimer, solver):
    from pyscf import gto

    xyz = np.array(dimer.coordinates_angstrom)
    xyz[2] = xyz[3] + (-0.9, 0.0, 0.0)
    # The transferred H is bonded to B and is more than 2 Angstrom from A's oxygen.
    assert np.linalg.norm(xyz[2] - xyz[3]) == pytest.approx(0.9)
    assert np.linalg.norm(xyz[2] - xyz[0]) > 2.0
    solver.result = gto.M(
        atom=list(zip(dimer.symbols, xyz, strict=True)), basis="sto-3g", verbose=0
    )
    with pytest.raises(ValueError, match="fragment membership"):
        structure.optimize_fragment_geometry(dimer)


def test_changed_atom_order_is_rejected(dimer, solver):
    solver.result = types.SimpleNamespace(
        natm=6,
        atom_pure_symbol=lambda i: dimer.symbols[(i + 1) % 6],
        atom_coords=lambda unit: np.array(dimer.coordinates_angstrom),
    )
    with pytest.raises(ValueError, match="atom order"):
        structure.optimize_fragment_geometry(dimer)


def test_changed_atom_count_is_rejected(dimer, solver):
    solver.result = types.SimpleNamespace(natm=5)
    with pytest.raises(ValueError, match="atom count"):
        structure.optimize_fragment_geometry(dimer)


def test_nonfinite_output_coordinates_are_rejected(dimer, solver):
    xyz = np.array(dimer.coordinates_angstrom)
    xyz[0, 0] = np.inf
    solver.result = types.SimpleNamespace(
        natm=6,
        atom_pure_symbol=lambda i: dimer.symbols[i],
        atom_coords=lambda unit: xyz,
    )
    with pytest.raises(ValueError, match="finite"):
        structure.optimize_fragment_geometry(dimer)


def test_input_partition_must_match_the_covalent_fragments(dimer, solver):
    wrong = replace(dimer, fragments=((0, 1), (2, 3, 4, 5)))
    with pytest.raises(ValueError, match="fragment membership"):
        structure.optimize_fragment_geometry(wrong)
    assert not solver.calls


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"fragments": None}, "two fragments"),
        ({"multiplicities": (1, 3)}, "singlet"),
        ({"charges": (0,)}, "charges"),
    ],
)
def test_only_dimers_with_two_singlet_fragments_are_supported(dimer, solver, changes, message):
    with pytest.raises(ValueError, match=message):
        structure.optimize_fragment_geometry(replace(dimer, **changes))
    assert not solver.calls


@pytest.mark.parametrize("maxsteps", [0, -1, 1.5, True])
def test_iteration_limit_requires_a_positive_integer(dimer, solver, maxsteps):
    with pytest.raises(ValueError, match="maxsteps"):
        structure.optimize_fragment_geometry(dimer, maxsteps=maxsteps)
    assert not solver.calls


def test_geometry_extra_has_an_actionable_missing_dependency_error(dimer, monkeypatch):
    monkeypatch.setattr(structure, "extra_available", lambda extra: False)
    with pytest.raises(ImportError, match="uv sync --extra geometry"):
        structure.optimize_fragment_geometry(dimer)


@pytest.mark.geometry
def test_real_dft_d3_optimization_and_iteration_limit(tmp_path):
    if not extra_available("geometry"):
        pytest.skip("geometry extra is not installed")
    # A fresh interpreter avoids native-thread state leaking into later fork tests.
    script = r"""
import json
import numpy as np
from pyscf import dft, gto
from q2m3.fragments.molecule import Molecule
from q2m3.fragments.structure import optimize_fragment_geometry

molecule = Molecule(
    ("O", "H", "H", "O", "H", "H"),
    ((-1.551007, -0.114520, 0), (-1.934259, 0.762503, 0), (-0.599677, 0.040712, 0),
     (1.350625, 0.111469, 0), (1.680398, -0.373741, -0.758561),
     (1.680398, -0.373741, 0.758561)),
    ((0, 1, 2), (3, 4, 5)), (0, 0), (1, 1), "water dimer",
)
limited = optimize_fragment_geometry(molecule, functional="pbe", basis="sto-3g", maxsteps=1)
assert limited.provenance["converged"] is False
result = optimize_fragment_geometry(molecule, functional="pbe", basis="sto-3g")
assert result.provenance["converged"] is True
assert result.provenance["dispersion"] == "D3(BJ)"
assert result.molecule.fragments == molecule.fragments
assert np.linalg.norm(np.array(result.molecule.coordinates_angstrom) - molecule.coordinates_angstrom) > 1e-3

# Independent single-point calculations at the returned geometry establish that
# the recorded correction was applied and that the DFT-D3 gradient is small.
mol = gto.M(atom=list(zip(result.molecule.symbols, result.molecule.coordinates_angstrom)),
            basis="sto-3g", unit="Angstrom", verbose=0)
plain = dft.RKS(mol, xc="pbe").run()
corrected = dft.RKS(mol, xc="pbe")
corrected.disp = "d3bj"
corrected.kernel()
assert plain.converged and corrected.converged
delta = corrected.e_tot - plain.e_tot
assert abs(delta) > 1e-5
assert abs(delta - result.provenance["dispersion_energy_hartree"]) < 1e-8
gradient = corrected.nuc_grad_method().kernel()
assert np.linalg.norm(gradient, axis=1).max() < 6e-4
print(json.dumps({"delta_hartree": delta, "gradient_max": float(np.linalg.norm(gradient, axis=1).max()),
                  "converged": result.provenance["converged"]}))
"""
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    run = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    evidence = json.loads(run.stdout.splitlines()[-1])
    assert evidence["converged"] is True


def test_geometry_extra_is_registered_without_loading_optional_backends():
    import tomllib

    from q2m3._lazy import EXTRA_PROBES

    config = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    dependencies = config["project"]["optional-dependencies"]["geometry"]
    assert any(dep.startswith("pyscf-dispersion>=") for dep in dependencies)
    assert any(dep.startswith("geometric>=") for dep in dependencies)
    assert EXTRA_PROBES["geometry"] == ("pyscf.dispersion", "geometric")
