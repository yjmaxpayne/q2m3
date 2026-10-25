"""Experimental dimer input, X-H normalization, and DFT-D3(BJ) geometry optimization."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from scipy.sparse.csgraph import connected_components

from q2m3._lazy import extra_available
from q2m3.fragments.molecule import Molecule

# Standard neutron-diffraction X-H bond lengths (Angstrom), keyed by the parent element.
# Source: Allen, Kennard, Watson, Brammer, Orpen & Taylor, J. Chem. Soc., Perkin Trans. 2,
# 1987, S1-S19, doi:10.1039/p298700000s1. The values are transcribed from Bergedahl et al.,
# Int. J. Mol. Sci. 2025, 26, 1671, doi:10.3390/ijms26041671 (CC BY 4.0), Methods; the
# 1987 tables were not read.
XH_BOND_TARGETS_ANGSTROM: Final[Mapping[str, float]] = MappingProxyType(
    {"C": 1.083, "N": 1.009, "O": 0.983}
)


@dataclass(frozen=True)
class GeometryOptimizationResult:
    """Optimized molecular geometry and JSON-compatible calculation provenance.

    Inspect ``provenance["converged"]`` before using the geometry. An exhausted
    optimization returns its last evaluated geometry with this flag set to False.
    """

    molecule: Molecule
    provenance: dict[str, Any]


def optimize_fragment_geometry(
    molecule: Molecule,
    *,
    functional: str = "b3lyp",
    basis: str = "def2-svp",
    dispersion: str = "d3bj",
    maxsteps: int = 100,
) -> GeometryOptimizationResult:
    """Optimize a singlet dimer in vacuum with PySCF DFT-D3(BJ) and geomeTRIC.

    This optimizes all atoms without counterpoise. It does not constrain fragment
    motion. The input and output must each have the two declared covalent components.
    A bond exists below 1.2 times the sum of the PySCF covalent radii. This geometric
    test rejects atom transfer, fragmentation, and interfragment covalent bonding;
    it does not infer a chemical bond order. Atom order and fragment metadata stay fixed.

    Args:
        molecule: Dimer geometry in Angstrom, with two singlet fragments.
        functional: PySCF functional name without an embedded dispersion suffix.
        basis: PySCF basis name.
        dispersion: Must be ``"d3bj"``. Dispersion cannot be disabled.
        maxsteps: Positive maximum number of optimization steps.

    Returns:
        New geometry and provenance with functional, dispersion, and convergence flags.
        An iteration limit sets ``converged=False``. It does not imply convergence.

    Raises:
        ImportError: The ``geometry`` extra is missing.
        ValueError: Input is invalid, dispersion is absent, coordinates are nonfinite,
            or the optimizer changes atom order or fragment membership.
        RuntimeError: An electronic SCF calculation does not converge.
    """
    if dispersion != "d3bj":
        raise ValueError("geometry optimization requires D3(BJ) dispersion (dispersion='d3bj')")
    if not isinstance(maxsteps, int) or isinstance(maxsteps, bool) or maxsteps < 1:
        raise ValueError("maxsteps must be a positive integer")
    if molecule.fragments is None or len(molecule.fragments) != 2:
        raise ValueError("geometry optimization requires two fragments")
    if molecule.multiplicities != (1, 1) or len(molecule.charges) != 2:
        raise ValueError("geometry optimization requires two singlet fragments and two charges")
    if not extra_available("geometry"):
        raise ImportError("geometry optimization needs: uv sync --extra geometry")

    from pyscf import dft, gto
    from pyscf.geomopt.geometric_solver import kernel
    from pyscf.scf.dispersion import parse_disp

    if parse_disp(functional)[1] is not None:
        raise ValueError("functional must not contain a dispersion suffix; use dispersion='d3bj'")
    _check_fragment_membership(molecule, np.asarray(molecule.coordinates_angstrom))
    mol = gto.M(
        atom=list(zip(molecule.symbols, molecule.coordinates_angstrom, strict=True)),
        basis=basis,
        unit="Angstrom",
        charge=molecule.charge,
        spin=0,
        verbose=0,
    )
    method = dft.RKS(mol, xc=functional)
    method.disp = dispersion
    dispersion_energies: list[float] = []

    def record_step(envs: dict[str, Any]) -> None:
        scanner = envs["g_scanner"]
        if not scanner.converged:
            raise RuntimeError("DFT SCF did not converge during geometry optimization")
        correction = scanner.base.scf_summary.get("dispersion")
        if scanner.base.disp != "d3bj" or correction is None or not np.isfinite(correction):
            raise ValueError("optimizer step lacks finite D3(BJ) dispersion evidence")
        dispersion_energies.append(float(correction))

    converged, optimized = kernel(
        method, maxsteps=maxsteps, assert_convergence=True, callback=record_step
    )
    if not dispersion_energies:
        raise ValueError("optimizer returned no D3(BJ) dispersion evidence")
    if (
        optimized.natm != len(molecule.symbols)
        or tuple(optimized.atom_pure_symbol(i) for i in range(optimized.natm)) != molecule.symbols
    ):
        raise ValueError("geometry optimization changed atom order or atom count")
    xyz = optimized.atom_coords(unit="Angstrom")
    _check_fragment_membership(molecule, xyz)
    geometry = replace(
        molecule, coordinates_angstrom=tuple(tuple(float(c) for c in row) for row in xyz)
    )
    return GeometryOptimizationResult(
        geometry,
        {
            "backend": "PySCF",
            "optimizer": "geomeTRIC",
            "functional": method.xc,
            "basis": basis,
            "dispersion": "D3(BJ)",
            "converged": bool(converged),
            "scf_converged": True,
            "maxsteps": maxsteps,
            "gradient_evaluations": len(dispersion_energies),
            "dispersion_energy_hartree": dispersion_energies[-1],
        },
    )


def _check_fragment_membership(molecule: Molecule, xyz: np.ndarray) -> None:
    """Check covalent components against the declared atom partition."""
    from pyscf.data import elements, radii
    from pyscf.lib import param

    if xyz.shape != (len(molecule.symbols), 3) or not np.isfinite(xyz).all():
        raise ValueError("geometry coordinates must be finite with shape (n_atoms, 3)")
    radii_angstrom = np.array([radii.COVALENT[elements.charge(s)] for s in molecule.symbols])
    radii_angstrom *= param.BOHR
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    bonds = distances < 1.2 * (radii_angstrom[:, None] + radii_angstrom[None, :])
    count, labels = connected_components(bonds, directed=False)
    actual = {frozenset(np.flatnonzero(labels == i)) for i in range(count)}
    expected = {frozenset(fragment) for fragment in molecule.fragments}
    indices = sorted(i for fragment in molecule.fragments for i in fragment)
    if actual != expected or indices != list(range(len(molecule.symbols))):
        raise ValueError("covalent connectivity does not preserve fragment membership")


def read_dimer_cif(source: str | os.PathLike[str], *, name: str | None = None) -> Molecule:
    """Read the asymmetric unit of a CIF as a neutral singlet dimer.

    The atoms of the first data block with a structure are taken as listed: no symmetry
    expansion and no periodic images. The two fragments are the two covalently bonded
    molecules (ASE covalent-radius cutoffs). Fragment A holds the first listed atom.

    Raises:
        ImportError: ASE (the optional ``viz`` extra) is not installed.
        ValueError: The CIF has no structure, or its asymmetric unit is not exactly two
            covalently bonded molecules. For example, a molecule is split across the cell
            boundary, or the dimer needs a symmetry operation.
    """
    try:
        from ase.io.cif import parse_cif
        from ase.neighborlist import build_neighbor_list, natural_cutoffs
    except ImportError:
        raise ImportError(
            "read_dimer_cif needs ASE, which is not installed. "
            "Install it with: uv sync --extra viz"
        ) from None
    block = next((b for b in parse_cif(os.fspath(source)) if b.has_structure()), None)
    if block is None:
        raise ValueError(f"{os.fspath(source)} has no data block with a structure")
    atoms = block.get_unsymmetrized_structure()
    # ASE 3.26 already returns pbc=False here. Set it so that no version bonds periodic images.
    atoms.pbc = False
    bonds = build_neighbor_list(atoms, cutoffs=natural_cutoffs(atoms))
    count, labels = connected_components(bonds.get_connectivity_matrix(), directed=False)
    if count != 2:
        raise ValueError(
            f"the asymmetric unit holds {count} covalently bonded molecules, expected 2"
        )
    in_a = labels == labels[0]
    fragments = (
        tuple(int(i) for i in np.flatnonzero(in_a)),
        tuple(int(i) for i in np.flatnonzero(~in_a)),
    )
    coordinates = tuple(tuple(float(c) for c in xyz) for xyz in atoms.positions)
    return Molecule(
        tuple(atoms.get_chemical_symbols()), coordinates, fragments, (0, 0), (1, 1), name
    )


def normalize_xh(molecule: Molecule) -> Molecule:
    """Move each H along its X-H bond to the target length in ``XH_BOND_TARGETS_ANGSTROM``.

    The parent X of an H is the nearest other atom. Only H atoms move. The result keeps the
    fragments, charges, multiplicities and name.

    Known limit: normalization corrects the short X-H bonds of X-ray H positions. It does
    not correct the polarization of X-H bonds in hydrogen bonds, so a normalized H is not
    the true H position in a hydrogen bond.

    Raises:
        ValueError: The parent of an H has no target length.
    """
    xyz = np.array(molecule.coordinates_angstrom, dtype=float)
    normalized = xyz.copy()
    for h, symbol in enumerate(molecule.symbols):
        if symbol != "H":
            continue
        distances = np.linalg.norm(xyz - xyz[h], axis=1)
        distances[h] = np.inf
        parent = int(np.argmin(distances))
        element = molecule.symbols[parent]
        if element not in XH_BOND_TARGETS_ANGSTROM:
            raise ValueError(f"no X-H target for {element} (atom {parent}, parent of H atom {h})")
        bond = (xyz[h] - xyz[parent]) / distances[parent]
        normalized[h] = xyz[parent] + XH_BOND_TARGETS_ANGSTROM[element] * bond
    coordinates = tuple(tuple(float(c) for c in row) for row in normalized)
    return replace(molecule, coordinates_angstrom=coordinates)
