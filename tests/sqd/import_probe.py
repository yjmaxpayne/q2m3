"""Fresh-process SQD execution with import and cached-call boundary tracing."""

from __future__ import annotations

import builtins
import importlib
import importlib.abc
import json
import os
import sys
import threading
import types
from pathlib import Path


def main():
    """Run real high/low workflows under tracing, optionally bypassing root init."""
    source, mode, event_path = sys.argv[1:]
    forbidden = {"pennylane", "catalyst"}
    if mode == "without-catalyst":

        class NoCatalyst(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] == "catalyst":
                    raise ModuleNotFoundError("blocked catalyst", name=fullname)

        sys.meta_path.insert(0, NoCatalyst())
    if mode == "isolated":
        # This harness bypasses only package initializers, never algorithm code.
        for name, relative in (("q2m3", "q2m3"), ("q2m3.interfaces", "q2m3/interfaces")):
            package = types.ModuleType(name)
            package.__path__ = [str(Path(source) / relative)]
            sys.modules[name] = package
        preloaded = []
    else:
        import q2m3

        preloaded = sorted(n for n in sys.modules if n.split(".")[0] in forbidden)
        if mode == "without-catalyst":
            assert not any(n.split(".")[0] == "catalyst" for n in preloaded)

    events = os.open(event_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)

    def record(kind, name):
        os.write(events, (json.dumps([os.getpid(), kind, name]) + "\n").encode())

    def reject(kind, name):
        record(kind, name)
        raise AssertionError("SQD boundary violation: " + kind + " " + name)

    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def traced_import(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            reject("import", name)
        return original_import(name, *args, **kwargs)

    def traced_import_module(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            reject("import_module", name)
        return original_import_module(name, *args, **kwargs)

    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in forbidden:
                reject("finder", fullname)

    seen = set()
    watched = {"_run_full", "sample", "diagonalize_samples", "run_reference", "run_comparisons"}

    def profile(frame, event, arg):
        if event == "call":
            module = frame.f_globals.get("__name__", "")
            name = frame.f_code.co_name
        elif event == "c_call":
            module = getattr(arg, "__module__", "") or ""
            name = getattr(arg, "__name__", "")
        else:
            return
        if module.split(".")[0] in forbidden:
            reject("call", module + "." + name)
        key = (os.getpid(), module, name)
        if module.startswith("q2m3.sqd.") and name in watched and key not in seen:
            seen.add(key)
            record("stage", module + "." + name)

    builtins.__import__ = traced_import
    importlib.import_module = traced_import_module
    sys.meta_path.insert(0, Block())
    sys.setprofile(profile)
    threading.setprofile(profile)

    import numpy as np

    from q2m3.molecule import MoleculeConfig
    from q2m3.sqd import LUCJConfig, run_sqd, run_sqd_from_integrals
    from q2m3.sqd.ansatz import build_ccsd_seed
    from q2m3.sqd.integrals import build_integrals

    if mode != "isolated":
        assert q2m3.run_sqd is run_sqd
        high_entry = q2m3.run_sqd
    else:
        high_entry = run_sqd
    molecule = MoleculeConfig("H2", ["H", "H"], [[0, 0, 0], [0, 0, 0.74]], 0, 2, 2)
    lucj = LUCJConfig(n_reps=2, shots=128)
    high = high_entry(
        molecule.symbols,
        np.asarray(molecule.coords),
        active_electrons=2,
        active_orbitals=2,
        lucj=lucj,
        seed=31,
        verbose=False,
    )
    data = build_integrals(molecule, host_available_mb=4096)
    seed_data = build_ccsd_seed(data, host_available_mb=4096)
    low = run_sqd_from_integrals(
        data.h1,
        data.h2,
        data.e_core,
        norb=data.norb,
        nelec=data.nelec,
        context=data.context,
        seed_data=seed_data,
        lucj=lucj,
        seed=31,
    )
    for result in (high, low):
        result.validate()
        assert result.backend == "ffsim" and result.shots == 128
        assert result.baseline_tier == "T0"
        assert abs(result.sqd_energy - result.baseline_energy) < 1e-10
        assert result.sqd_energy < result.hf_energy - 0.01
    assert abs(high.sqd_energy - low.sqd_energy) < 1e-10
    assert high.unique_dets_vs_shots == low.unique_dets_vs_shots
    assert high.provenance["context"]["hamiltonian_id"] == data.context.hamiltonian_id
    sys.setprofile(None)
    threading.setprofile(None)
    builtins.__import__ = original_import
    importlib.import_module = original_import_module
    os.close(events)
    if mode == "isolated":
        assert not forbidden & {name.split(".")[0] for name in sys.modules}
    print(
        json.dumps(
            {
                "mode": mode,
                "preloaded": preloaded,
                "energy_ha": high.sqd_energy,
                "low_energy_ha": low.sqd_energy,
                "hf_ha": high.hf_energy,
                "shots": high.shots,
                "seed": high.seed,
                "subspace_dims": high.subspace_dims,
                "hamiltonian_id": data.context.hamiltonian_id,
            }
        )
    )


if __name__ == "__main__":
    main()
