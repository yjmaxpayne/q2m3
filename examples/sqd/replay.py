"""Reproducible dependency probes, with no scientific imports during collection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(bundle: Path = BUNDLE) -> dict:
    """Reject incomplete or modified replay bundles before running a probe."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    if set(manifest["probes"]) != {"h2", "reference", "transpose", "lazy"}:
        raise ValueError("Replay probe set is incomplete")
    required = {
        "README.md",
        "replay.py",
        "reference_history.py",
        "lazy_replay.py",
        "lazy_helpers.py",
    }
    if set(manifest["files"]) != required:
        raise ValueError("Replay inventory must contain exactly the five required files")
    for name, expected in manifest["files"].items():
        path = bundle / name
        if path.resolve().parent != bundle.resolve():
            raise ValueError(f"Replay source escapes bundle: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if digest(path) != expected:
            raise ValueError(f"Replay source hash mismatch: {name}")
    return manifest


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def h2() -> dict:
    import ffsim
    import numpy as np
    from pyscf import cc, gto, mcscf, scf
    from qiskit.primitives import BitArray
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    coupled = cc.CCSD(mf).run(conv_tol=1e-12)
    assert mf.converged and coupled.converged
    ham = ffsim.MolecularData.from_scf(mf, active_space=[0, 1]).hamiltonian
    # Both channels require upper-triangular pairs, including the diagonal.
    pairs = [(i, j) for i in range(2) for j in range(i, 2)]
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        coupled.t2, t1=coupled.t1, n_reps=2, interaction_pairs=(pairs, pairs)
    )
    vec = ffsim.apply_unitary(ffsim.hartree_fock_state(2, (1, 1)), op, norb=2, nelec=(1, 1))
    assert abs(float(np.vdot(vec, vec).real) - 1) < 1e-10
    bits = np.asarray(
        ffsim.sample_state_vector(
            vec,
            norb=2,
            nelec=(1, 1),
            shots=50_000,
            bitstring_type=ffsim.BitstringType.BIT_ARRAY,
            seed=20260721,
        )
    )
    assert bits.shape == (50_000, 4)
    assert np.all(bits[:, :2].sum(axis=1) == 1) and np.all(bits[:, 2:].sum(axis=1) == 1)
    result = diagonalize_fermionic_hamiltonian(
        ham.one_body_tensor,
        ham.two_body_tensor,
        BitArray.from_bool_array(bits, order="big"),
        samples_per_batch=100,
        norb=2,
        nelec=(1, 1),
        num_batches=3,
        max_iterations=5,
        symmetrize_spin=True,
        seed=20260721,
    )
    exact = float(mcscf.CASCI(mf, 2, 2).run().e_tot)
    energy = float(result.energy + ham.constant)
    assert np.isfinite(energy) and abs(energy - exact) < 1e-3
    assert "pennylane" not in sys.modules
    pair_checks = interaction_pair_checks()
    return {
        "n_total": 6 + pair_checks,
        "n_pass": 6 + pair_checks,
        "energy_ha": energy,
        "exact_ha": exact,
        "error_ha": energy - exact,
        "interaction_pairs": [pairs, pairs],
        "inputs": {
            "atom": "H 0 0 0; H 0 0 0.74",
            "basis": "sto-3g",
            "active_electrons": 2,
            "active_orbitals": 2,
            "seed": 20260721,
            "shots": 50_000,
            "n_reps": 2,
        },
    }


def interaction_pair_checks() -> int:
    """Check channel masks and rejection semantics using nonzero amplitudes."""
    import ffsim
    import numpy as np

    amplitudes = np.array([[[[-0.2]]]])
    pairs = ([(0, 0), (0, 1)], [(1, 1)])
    op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(amplitudes, n_reps=2, interaction_pairs=pairs)
    count = 0
    for channel, allowed in enumerate(pairs):
        mask = np.zeros((2, 2), dtype=bool)
        for i, j in allowed:
            mask[i, j] = mask[j, i] = True
        values = op.diag_coulomb_mats[:, channel]
        assert np.all(values[:, ~mask] == 0)
        count += 1
        assert all(np.any(np.abs(values[:, i, j]) > 1e-12) for i, j in allowed)
        count += 1
        for invalid in ([(1, 0)], [(0, 0), (0, 0)]):
            trial = list(pairs)
            trial[channel] = invalid
            try:
                ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                    amplitudes, n_reps=2, interaction_pairs=tuple(trial)
                )
            except ValueError:
                count += 1
            else:
                raise AssertionError("Invalid interaction pairs accepted")
    return count


def reference() -> dict:
    result = load_file("reference_history", BUNDLE / "reference_history.py").part_E()
    assert result["n_pass"] == result["n_total"]
    result["scope"] = "historical 120s/GB model; not T01 production reference semantics"
    result["inputs"] = "21 embedded deterministic historical boundary cases; no RNG"
    return result


def transpose() -> dict:
    # A scoped namespace avoids the existing root's eager PennyLane import debt.
    # Load the real constants rather than inventing a conversion factor.
    import types

    import numpy as np
    from pyscf import ao2mo, gto, mcscf, scf

    saved = {name: sys.modules.get(name) for name in ("q2m3", "q2m3.constants")}
    try:
        package = types.ModuleType("q2m3")
        package.__path__ = [str(ROOT / "src/q2m3")]
        sys.modules["q2m3"] = package
        load_file("q2m3.constants", ROOT / "src/q2m3/constants.py")
        fmo = load_file("fixed_mo_probe", ROOT / "src/q2m3/interfaces/fixed_mo_embedding.py")
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.757, 0.587], [0.0, -0.757, 0.587]])
    result = fmo.build_fixed_mo_embedding_integrals(
        ["O", "H", "H"],
        coords,
        mm_charges=np.array([-0.834, 0.417, 0.417]),
        mm_coords=np.array([[0.0, 0.0, 3.0], [0.0, 0.76, 3.59], [0.0, -0.76, 3.59]]),
        basis="sto-3g",
        active_electrons=4,
        active_orbitals=4,
    )
    mol = gto.M(atom=list(zip(["O", "H", "H"], coords, strict=False)), basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    assert mf.converged
    mo = fmo._canonicalize_mo_signs(np.asarray(mf.mo_coeff))
    n = mo.shape[1]
    chem = ao2mo.kernel(mol, mo, compact=False).reshape(n, n, n, n)
    active = np.array(result.active_indices)
    expected = chem[np.ix_(active, active, active, active)]
    converted = result.two_electron.transpose(0, 3, 1, 2)
    err = float(np.max(np.abs(converted - expected)))
    wrong = float(np.max(np.abs(result.two_electron - expected)))
    assert err < 1e-10
    assert wrong > 1e-3  # The molecule distinguishes an omitted inverse permutation.
    mc = mcscf.CASCI(mf, 4, 4)
    mc.mo_coeff = mo
    h1, constant = mc.get_h1eff()
    assert np.max(np.abs(result.one_electron_vacuum - h1)) < 1e-10
    assert np.max(np.abs(converted - ao2mo.restore(1, mc.get_h2eff(), 4))) < 1e-10
    assert np.count_nonzero(np.abs(result.delta_h_offdiag) > 1e-10) > 0
    return {
        "n_total": 6,
        "n_pass": 6,
        "inverse_error_ha": err,
        "unconverted_error_ha": wrong,
        "vacuum_core_constant_ha": float(constant),
        "inputs": {
            "symbols": ["O", "H", "H"],
            "coords_angstrom": coords.tolist(),
            "active_electrons": 4,
            "active_orbitals": 4,
            "basis": "sto-3g",
            "mm_charges": [-0.834, 0.417, 0.417],
            "mm_coords_angstrom": [[0, 0, 3], [0, 0.76, 3.59], [0, -0.76, 3.59]],
        },
        "producer_sha256": digest(ROOT / "src/q2m3/interfaces/fixed_mo_embedding.py"),
    }


def lazy() -> dict:
    return load_file("lazy_replay", BUNDLE / "lazy_replay.py").run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=("h2", "reference", "transpose", "lazy"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = validate_manifest()
    start = time.monotonic()
    result = globals()[args.probe]()
    versions = {}
    for name in ("numpy", "pyscf", "ffsim", "qiskit", "qiskit-addon-sqd", "pennylane-catalyst"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    output = {
        "probe": args.probe,
        "result": result,
        "versions": versions,
        "python": sys.version,
        "command": sys.argv,
        "wall_s": time.monotonic() - start,
        "source_hashes": manifest["files"],
        "exit_code": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
