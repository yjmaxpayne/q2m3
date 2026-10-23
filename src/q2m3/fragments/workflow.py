"""One-call entry point of the fragment interaction-energy workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from typing import Any

from pyscf import lib

from q2m3._lazy import extra_available
from q2m3.fragments.counterpoise import SCF_CONV_TOL, Subsystem, counterpoise
from q2m3.fragments.molecule import Molecule
from q2m3.fragments.references import frozen_core_mp2
from q2m3.fragments.result import (
    COUNTERPOISE_SUBSYSTEMS,
    ENERGY_LAYERS,
    ErrorDecomposition,
    FragmentInteractionResult,
    SubsystemResult,
    interaction_energy,
)

MODES = ("full", "classical_only")
VERSIONED_PACKAGES = ("q2m3", "pyscf", "numpy", "ffsim", "qiskit-addon-sqd", "pennylane")

_NO_SQD = "mode='classical_only' does not run the SQD stage"
_SUBSYSTEM_REASONS = {
    "e_ccsd_t": "CCSD(T) reference not computed",
    "e_casci": _NO_SQD,
    "e_sqd": _NO_SQD,
    "sqd": _NO_SQD,
    "active_space": "active space not resolved",
    "resources.qpe": "resource estimate not computed",
    "resources.sqd": "resource estimate not computed",
}
_RESULT_REASONS = {
    "errors.algorithm": "delta_e.sqd or delta_e.casci is null",
    "errors.active_space": "no same-basis full-space reference",
    "errors.basis": "no same-basis full-space reference and no external reference",
    "errors.model": "no external reference",
    "correspondence": "active space not resolved",
}


def _make_runner() -> Any:
    """Return the runner of the SQD stage."""
    raise NotImplementedError("the SQD stage runner is not available yet")


def run(
    molecule: Molecule,
    *,
    avas_labels: Sequence[str],
    avas_threshold: float,
    basis: str = "cc-pvdz",
    mode: str = "full",
    references: Sequence[Any] | None = None,
    seed: int = 0,
    lucj: Any = None,
    reference: Any = None,
    ccsd_t_budget: Any = None,
) -> FragmentInteractionResult | SubsystemResult:
    """Compute counterpoise-corrected interaction energies of ``molecule``.

    A dimer gives a ``FragmentInteractionResult``. A single molecule gives the
    ``SubsystemResult`` of subsystem ``"M"``. ``mode="classical_only"`` computes HF and
    frozen-core MP2 and never builds the SQD stage runner.

    Raises:
        TypeError: ``avas_labels`` is not a sequence of strings, or ``avas_threshold``
            is not a real number.
        ValueError: ``mode`` is not ``"full"`` or ``"classical_only"``.
        ImportError: ``mode="full"`` and the optional ``sqd`` backend is missing.
        NotImplementedError: ``mode="full"`` with the ``sqd`` backend, or
            ``references`` is not ``None``.
        FragmentWorkflowError: The RHF of a subsystem does not converge.
    """
    labels, threshold = _checked_avas(avas_labels, avas_threshold)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "full":
        if not extra_available("sqd"):
            raise ImportError(
                "run(mode='full') requires the optional 'sqd' backend. "
                "Install it with: uv sync --extra sqd"
            )
        raise NotImplementedError("mode='full' is not available yet. Use mode='classical_only'.")
    if references is not None:
        raise NotImplementedError("imported references are not supported yet")

    subsystems = counterpoise(molecule, basis=basis)
    mp2 = {name: frozen_core_mp2(sub) for name, sub in subsystems.items()}
    results = {name: _classical_result(sub, mp2[name][0]) for name, sub in subsystems.items()}
    if molecule.fragments is None:
        return results["M"].validate()

    delta_e = {layer: interaction_energy(results, layer) for layer in ENERGY_LAYERS}
    null_reasons = {
        f"delta_e.{layer}": _null_layer_reason(results, layer)
        for layer, value in delta_e.items()
        if value is None
    }
    provenance = {
        "geometry_sha256": molecule.geometry_sha256(),
        "basis": basis,
        "cart": False,
        "avas_labels": labels,
        "avas_threshold": threshold,
        "seed": seed,
        "scf_conv_tol": SCF_CONV_TOL,
        "frozen_core": {name: nfrozen for name, (_, nfrozen) in mp2.items()},
        "versions": {package: _installed_version(package) for package in VERSIONED_PACKAGES},
        "threads": {"classical_stage": lib.num_threads()},
    }
    return FragmentInteractionResult(
        fragment_id=molecule.name or molecule.geometry_sha256(),
        mode=mode,
        basis=basis,
        subsystems=results,
        delta_e=delta_e,
        references=(),
        errors=ErrorDecomposition(),
        correspondence=None,
        provenance=provenance,
        null_reasons=null_reasons | _RESULT_REASONS,
    ).validate()


def _checked_avas(labels: Any, threshold: Any) -> tuple[tuple[str, ...], float]:
    if isinstance(labels, str) or not isinstance(labels, Sequence):
        raise TypeError(f"avas_labels must be a sequence of strings, got {labels!r}")
    if not all(isinstance(label, str) for label in labels):
        raise TypeError(f"avas_labels must contain only strings, got {labels!r}")
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError(f"avas_threshold must be a real number, got {threshold!r}")
    return tuple(labels), float(threshold)


def _classical_result(sub: Subsystem, e_mp2: float) -> SubsystemResult:
    return SubsystemResult(
        name=sub.name,
        real_atoms=sub.real_atoms,
        ghost_atoms=sub.ghost_atoms,
        n_ao=sub.mol.nao,
        e_hf=float(sub.scf.e_tot),
        e_mp2=e_mp2,
        e_ccsd_t=None,
        e_casci=None,
        e_sqd=None,
        active_space=None,
        sqd=None,
        resources={},
        null_reasons=dict(_SUBSYSTEM_REASONS),
    )


def _null_layer_reason(results: Mapping[str, SubsystemResult], layer: str) -> str:
    field = f"e_{layer}"
    nulls = [name for name in COUNTERPOISE_SUBSYSTEMS if getattr(results[name], field) is None]
    return f"{field} is null for {', '.join(nulls)}: {results[nulls[0]].null_reasons[field]}"


def _installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None
