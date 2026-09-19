"""Executable, fail-closed dependency and numerical compatibility checks.

Invoke directly with ``python tests/sqd/dependency_gate.py --help``. Scientific
imports remain inside the probe so core-profile checks need no SQD extra.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
from pathlib import Path

TOLERANCE_HA = 1e-10


def assert_core_profile(exported: str, installed: list[dict]) -> dict:
    """Reject SQD distributions in both requirements export and installed inventory."""
    assert exported.strip() and installed, "Empty core-profile evidence"
    names = [item["name"] for item in installed]
    for line in exported.splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)", line.strip())
        if match:
            names.append(match.group(1))
    forbidden = sorted(
        {
            name
            for name in names
            if (normalized := re.sub(r"[-_.]+", "-", name).lower()) == "ffsim"
            or normalized.startswith("qiskit")
        }
    )
    assert not forbidden, f"Core profile contains optional SQD distributions: {forbidden}"
    return {"isolated": True, "installed_count": len(installed)}


def require_catalyst() -> dict:
    """Require an actual Catalyst import; metadata or a skipped test is insufficient."""
    importlib.import_module("catalyst")
    return {"imported": True}


def _flatten(value):
    if isinstance(value, list):
        for element in value:
            yield from _flatten(element)
    else:
        yield value


def _shape(value):
    if not isinstance(value, list):
        return ()
    assert value, "Empty integral shape"
    child = _shape(value[0])
    assert all(_shape(element) == child for element in value), "Ragged integral shape"
    return (len(value), *child)


def _validate_oracles(report: dict) -> None:
    energies = report["energies_ha"]
    assert set(energies) == {"fci", "ccsd", "casci", "selected_ci"}, "Incomplete oracle"
    assert all(math.isfinite(value) for value in energies.values()), "Nonfinite energy"
    for method, energy in energies.items():
        assert abs(energy - energies["fci"]) <= TOLERANCE_HA, f"FCI oracle failed: {method}"
    error = report["ao2mo_max_error"]
    assert math.isfinite(error) and 0 <= error <= TOLERANCE_HA, "AO transform oracle failed"
    assert set(report["arrays"]) == {"h1", "h2"}, "Incomplete integral arrays"
    for array in report["arrays"].values():
        values = list(_flatten(array))
        assert values and all(math.isfinite(value) for value in values), "Nonfinite/empty array"


def compare_numerics(baseline: dict, candidate: dict) -> dict:
    """Compare identical-input results only after each passes its scientific oracles."""
    assert baseline["input"] == candidate["input"], "Different numerical probe input"
    for report in (baseline, candidate):
        _validate_oracles(report)
    deltas = {
        key: abs(value - candidate["energies_ha"][key])
        for key, value in baseline["energies_ha"].items()
    }
    assert max(deltas.values()) <= TOLERANCE_HA, f"Energy drift: {deltas}"
    array_deltas = {}
    for name, array in baseline["arrays"].items():
        assert _shape(array) == _shape(
            candidate["arrays"][name]
        ), f"Integral shape mismatch: {name}"
        left, right = list(_flatten(array)), list(_flatten(candidate["arrays"][name]))
        assert len(left) == len(right), f"Integral shape mismatch: {name}"
        array_deltas[name] = max(abs(a - b) for a, b in zip(left, right, strict=True))
    assert max(array_deltas.values()) <= TOLERANCE_HA, f"Integral drift: {array_deltas}"
    return {"max_energy_delta_ha": max(deltas.values()), "integral_max_deltas": array_deltas}


def numerical_probe() -> dict:
    """Probe two-electron H2 using exact FCI and explicit AO transformation oracles."""
    import numpy as np
    from pyscf import ao2mo, cc, fci, gto, mcscf, scf

    inputs = {
        "atom": "H 0.13 -0.21 0.07; H 0.42 0.18 0.67",
        "basis": "sto-3g",
        "unit": "Angstrom",
        "charge": 0,
        "spin": 0,
        "seed": None,
        "scf_conv_tol": 1e-13,
        "solver_conv_tol": 1e-13,
    }
    mol = gto.M(
        **{key: inputs[key] for key in ("atom", "basis", "unit", "charge", "spin")}, verbose=0
    )
    mf = scf.RHF(mol).run(conv_tol=inputs["scf_conv_tol"])
    assert mf.converged, "SCF did not converge"
    # Fix arbitrary MO column signs before comparing A/B integral arrays.
    coeff = mf.mo_coeff.copy()
    signs = np.sign(coeff[np.argmax(np.abs(coeff), axis=0), np.arange(coeff.shape[1])])
    coeff *= signs
    mf.mo_coeff = coeff
    norb = coeff.shape[1]
    h1 = coeff.T @ mf.get_hcore() @ coeff
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, coeff), norb)
    explicit = np.einsum("pqrs,pi,qj,rk,sl->ijkl", mol.intor("int2e"), coeff, coeff, coeff, coeff)
    exact, _ = fci.direct_spin1.kernel(h1, h2, norb, mol.nelec, ecore=mol.energy_nuc(), tol=1e-13)
    coupled = cc.CCSD(mf).run(conv_tol=1e-13, conv_tol_normt=1e-11, max_cycle=200)
    assert coupled.converged, "CCSD did not converge"
    cas = mcscf.CASCI(mf, norb, mol.nelectron)
    cas.fcisolver.conv_tol = 1e-13
    cas.kernel()
    selected = fci.selected_ci.SelectedCI(mol)
    selected.select_cutoff = 1e-12
    selected.ci_coeff_cutoff = 1e-12
    sci, _ = selected.kernel(h1, h2, norb, mol.nelec, ecore=mol.energy_nuc(), tol=1e-13)
    report = {
        "input": inputs,
        "energies_ha": {
            "fci": float(exact),
            "ccsd": float(coupled.e_tot),
            "casci": float(cas.e_tot),
            "selected_ci": float(sci),
        },
        "ao2mo_max_error": float(np.max(np.abs(h2 - explicit))),
        "arrays": {"h1": h1.tolist(), "h2": h2.tolist()},
    }
    _validate_oracles(report)
    return report


def main() -> None:
    """Run one gate and emit machine-readable JSON; assertion/import errors exit nonzero."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalyst")
    commands.add_parser("numerics")
    core = commands.add_parser("core-profile")
    core.add_argument("export", type=Path)
    core.add_argument("installed", type=Path, help="uv pip list --format json output")
    compare = commands.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.command == "catalyst":
        result = require_catalyst()
    elif args.command == "numerics":
        result = numerical_probe()
    elif args.command == "core-profile":
        result = assert_core_profile(
            args.export.read_text(), json.loads(args.installed.read_text())
        )
    else:
        result = compare_numerics(
            json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text())
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
