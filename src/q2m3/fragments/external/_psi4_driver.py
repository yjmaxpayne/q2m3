"""Psi4 side of the q2m3 Psi4 adapter: HF counterpoise energies of one dimer.

The interpreter of the Psi4 environment runs this file as
``python _psi4_driver.py <job.json> <output.json>`` in a scratch directory. No q2m3 module
imports it, so it may import psi4 at module level.
"""

import json
import sys

import psi4

# Counterpoise subsystems: indices of the fragments that become ghost atoms.
GHOSTS = {"AB": (), "A": (1,), "B": (0,)}


def geometry(fragments: list[dict], ghosts: tuple[int, ...]) -> str:
    blocks = []
    for index, fragment in enumerate(fragments):
        ghost = index in ghosts
        # A ghost fragment keeps its basis functions but has no nuclei and no electrons.
        lines = ["0 1" if ghost else f"{fragment['charge']} {fragment['multiplicity']}"]
        for symbol, x, y, z in fragment["atoms"]:
            label = f"Gh({symbol})" if ghost else symbol
            lines.append(f"{label} {x:.10f} {y:.10f} {z:.10f}")
        blocks.append("\n".join(lines))
    return "\n--\n".join(blocks) + "\nunits angstrom\nno_com\nno_reorient\nsymmetry c1\n"


def main(job_path: str, output_path: str) -> None:
    with open(job_path) as handle:
        job = json.load(handle)
    psi4.core.set_output_file("psi4_output.dat", False)
    psi4.set_options(
        {
            "basis": job["basis"],
            "reference": "rhf",
            "scf_type": "pk",
            "puream": True,
            "e_convergence": job["conv_tol"],
            "d_convergence": job["conv_tol"],
        }
    )
    energies = {}
    for name, ghosts in GHOSTS.items():
        psi4.core.clean()
        molecule = psi4.geometry(geometry(job["fragments"], ghosts))
        energies[name] = psi4.energy("scf", molecule=molecule)
    options = {
        "scf_type": psi4.core.get_global_option("SCF_TYPE").lower(),
        "puream": bool(psi4.core.get_global_option("PUREAM")),
        "e_convergence": psi4.core.get_option("SCF", "E_CONVERGENCE"),
        "d_convergence": psi4.core.get_option("SCF", "D_CONVERGENCE"),
    }
    output = {"psi4_version": psi4.__version__, "energies": energies, "options": options}
    with open(output_path, "w") as handle:
        json.dump(output, handle)


if __name__ == "__main__":
    main(*sys.argv[1:])
