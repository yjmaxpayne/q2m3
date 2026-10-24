"""One-call entry point of the fragment interaction-energy workflow."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from os import PathLike
from typing import Any

from pyscf import lib

from q2m3._lazy import extra_available
from q2m3.fragments.active_space import check_correspondence, select_active_spaces
from q2m3.fragments.counterpoise import SCF_CONV_TOL, Subsystem, counterpoise
from q2m3.fragments.errors import ReferenceImportError
from q2m3.fragments.importers import import_references
from q2m3.fragments.integrals import ActiveSpaceIntegrals, build_integrals
from q2m3.fragments.molecule import Molecule
from q2m3.fragments.references import CCSDTBudget, CCSDTStage, frozen_core_mp2, run_ccsd_t_stage
from q2m3.fragments.resources import qpe_estimate, sqd_estimate
from q2m3.fragments.result import (
    COUNTERPOISE_SUBSYSTEMS,
    ENERGY_LAYERS,
    ActiveSpaceRecord,
    CorrespondenceReport,
    ErrorDecomposition,
    ExternalReference,
    FragmentInteractionResult,
    SubsystemResult,
    decompose_errors,
    interaction_energy,
    promote_sqd_energies,
    references_by_category,
)
from q2m3.fragments.sqd_stage import SQDJob, SQDStageOutcome, SubprocessSQDRunner
from q2m3.sqd.config import LUCJConfig, ReferenceConfig
from q2m3.sqd.result import SQDResult

MODES = ("full", "classical_only")
VERSIONED_PACKAGES = ("q2m3", "pyscf", "numpy", "ffsim", "qiskit-addon-sqd", "pennylane")

_NO_SQD = "mode='classical_only' does not run the SQD stage"
_SUBSYSTEM_REASONS = {
    "e_ccsd_t": "CCSD(T) reference not computed",
    "e_casci": _NO_SQD,
    "e_sqd": _NO_SQD,
    "sqd": _NO_SQD,
    "resources.qpe": "resource estimate not computed",
    "resources.sqd": "resource estimate not computed",
}
_SAME_BASIS = "same-basis full-space reference"
_EXTERNAL = "external reference"
# The two energies each error term subtracts.
_ERROR_INPUTS = {
    "algorithm": ("delta_e.sqd", "delta_e.casci"),
    "active_space": ("delta_e.casci", _SAME_BASIS),
    "basis": (_SAME_BASIS, _EXTERNAL),
    "model": ("delta_e.casci", _EXTERNAL),
}


def _make_runner() -> Any:
    """Return the runner of the SQD stage."""
    return SubprocessSQDRunner()


def run(
    molecule: Molecule,
    *,
    avas_labels: Sequence[str],
    avas_threshold: float,
    basis: str = "cc-pvdz",
    mode: str = "full",
    references: Sequence[str | PathLike[str] | ExternalReference] | None = None,
    seed: int = 0,
    lucj: Any = None,
    reference: Any = None,
    ccsd_t_budget: Any = None,
) -> FragmentInteractionResult | SubsystemResult:
    """Compute counterpoise-corrected interaction energies of ``molecule``.

    A dimer gives a ``FragmentInteractionResult``. A single molecule gives the
    ``SubsystemResult`` of subsystem ``"M"``. Both modes select the AVAS active spaces
    and, for a dimer, check their correspondence before any correlated step. Both modes
    compute HF and frozen-core MP2. ``mode="classical_only"`` never builds the SQD stage
    runner. ``mode="full"`` also runs the frozen-core CCSD(T) reference when its predicted
    cost fits ``ccsd_t_budget`` (default ``CCSDTBudget()``), builds the active-space
    integrals, runs SQD for all subsystems in one child interpreter, and estimates the QPE
    and SQD resources. A rejected CCSD(T) leaves ``e_ccsd_t`` null with the reason.

    ``references`` holds reference files, read by ``import_references`` with the
    fragment id and ``basis`` of this run, and ``ExternalReference`` records. They form
    the active-space, basis and model error terms of a dimer result.

    Raises:
        TypeError: ``avas_labels`` is not a sequence of strings, ``avas_threshold`` is
            not a real number, or ``references`` is a single string or path.
        ValueError: ``mode`` is not ``"full"`` or ``"classical_only"``, an AVAS label is
            malformed or matches no AO, or ``references`` is given for a single molecule.
        ImportError: ``mode="full"`` and the optional ``sqd`` backend is missing.
        ReferenceImportError: A reference file violates the import format, or two
            references share a category.
        FragmentWorkflowError: The RHF or the CCSD of a subsystem does not converge, or
            label resolution, the ghost guard or the preflight rejects an active space.
        ActiveSpaceMismatchError: The monomer active spaces do not correspond to the
            dimer active space.
    """
    labels, threshold = _checked_avas(avas_labels, avas_threshold)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "full" and not extra_available("sqd"):
        raise ImportError(
            "run(mode='full') requires the optional 'sqd' backend. "
            "Install it with: uv sync --extra sqd"
        )
    fragment_id = molecule.name or molecule.geometry_sha256()
    imported = _imported_references(references, molecule, fragment_id, basis)

    subsystems = counterpoise(molecule, basis=basis)
    spaces, correspondence = _active_spaces(subsystems, labels, threshold)
    mp2 = {name: frozen_core_mp2(sub) for name, sub in subsystems.items()}
    results = {
        name: _classical_result(sub, mp2[name][0], spaces[name][0])
        for name, sub in subsystems.items()
    }
    outcome = ccsd_t = None
    if mode == "full":
        nfrozen = {name: frozen for name, (_, frozen) in mp2.items()}
        ccsd_t = run_ccsd_t_stage(subsystems, nfrozen, ccsd_t_budget or CCSDTBudget())
        integrals = {
            name: build_integrals(
                molecule, sub, *spaces[name], avas_labels=labels, avas_threshold=threshold
            )
            for name, sub in subsystems.items()
        }
        lucj = lucj or LUCJConfig()
        outcome = _run_sqd_stage(integrals, lucj, reference or ReferenceConfig(), seed)
        results = {
            name: _full_result(results[name], integrals[name], sqd, basis, lucj, ccsd_t)
            for name, sqd in zip(integrals, outcome.results, strict=True)
        }
    if molecule.fragments is None:
        return results["M"].validate()

    delta_e = {layer: interaction_energy(results, layer) for layer in ENERGY_LAYERS}
    null_reasons = {
        f"delta_e.{layer}": _null_layer_reason(results, layer)
        for layer, value in delta_e.items()
        if value is None
    }
    errors, same_basis_source = decompose_errors(delta_e, imported)
    null_reasons |= _error_reasons(errors, delta_e, same_basis_source, imported)
    threads: dict[str, Any] = {"classical_stage": lib.num_threads()}
    warnings: tuple[str, ...] = ()
    if outcome is not None:
        threads["sqd_stage"] = outcome.thread_records
        warnings = (*outcome.warnings, *ccsd_t.warnings)
    provenance = {
        "geometry_sha256": molecule.geometry_sha256(),
        "basis": basis,
        "cart": False,
        "avas_labels": labels,
        "avas_threshold": threshold,
        "cos_threshold": correspondence.cos_threshold,
        "seed": seed,
        "scf_conv_tol": SCF_CONV_TOL,
        "frozen_core": {name: nfrozen for name, (_, nfrozen) in mp2.items()},
        "versions": {package: _installed_version(package) for package in VERSIONED_PACKAGES},
        "threads": threads,
        "same_basis_reference_source": same_basis_source,
    }
    if ccsd_t is not None:
        provenance["ccsd_t_cost"] = ccsd_t.cost
    return FragmentInteractionResult(
        fragment_id=fragment_id,
        mode=mode,
        basis=basis,
        subsystems=results,
        delta_e=delta_e,
        references=imported,
        errors=errors,
        correspondence=correspondence,
        provenance=provenance,
        null_reasons=null_reasons,
        warnings=warnings,
    ).validate()


def _checked_avas(labels: Any, threshold: Any) -> tuple[tuple[str, ...], float]:
    if isinstance(labels, str) or not isinstance(labels, Sequence):
        raise TypeError(f"avas_labels must be a sequence of strings, got {labels!r}")
    if not all(isinstance(label, str) for label in labels):
        raise TypeError(f"avas_labels must contain only strings, got {labels!r}")
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError(f"avas_threshold must be a real number, got {threshold!r}")
    return tuple(labels), float(threshold)


def _imported_references(
    references: Any, molecule: Molecule, fragment_id: str, basis: str
) -> tuple[ExternalReference, ...]:
    if references is None:
        return ()
    if isinstance(references, str | PathLike):
        raise TypeError(f"references must be a sequence of paths or records, got {references!r}")
    if molecule.fragments is None:
        raise ValueError("references need a dimer; a single molecule has no interaction energy")
    found: list[ExternalReference] = []
    for item in references:
        if isinstance(item, ExternalReference):
            found.append(item)
        else:
            found.extend(import_references(item, fragment_id=fragment_id, basis=basis))
    try:
        references_by_category(found)
    except ValueError as exc:
        raise ReferenceImportError(str(exc)) from exc
    return tuple(found)


def _error_reasons(
    errors: ErrorDecomposition,
    delta_e: Mapping[str, float | None],
    same_basis_source: str | None,
    references: Sequence[ExternalReference],
) -> dict[str, str]:
    available = {
        "delta_e.sqd": delta_e["sqd"] is not None,
        "delta_e.casci": delta_e["casci"] is not None,
        _SAME_BASIS: same_basis_source is not None,
        _EXTERNAL: any(reference.category == "external" for reference in references),
    }
    return {
        f"errors.{term}": "missing " + " and ".join(n for n in inputs if not available[n])
        for term, inputs in _ERROR_INPUTS.items()
        if getattr(errors, term) is None
    }


def _active_spaces(
    subsystems: Mapping[str, Subsystem], labels: tuple[str, ...], threshold: float
) -> tuple[dict[str, tuple[ActiveSpaceRecord, Any]], CorrespondenceReport | None]:
    spaces = select_active_spaces(subsystems, labels, threshold)
    if "AB" not in spaces:
        return spaces, None
    overlap = subsystems["AB"].mol.intor("int1e_ovlp")
    return spaces, check_correspondence(spaces["AB"], spaces["A"], spaces["B"], overlap)


def _classical_result(sub: Subsystem, e_mp2: float, record: ActiveSpaceRecord) -> SubsystemResult:
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
        active_space=record,
        sqd=None,
        resources={},
        null_reasons=dict(_SUBSYSTEM_REASONS),
    )


def _full_result(
    classical: SubsystemResult,
    integrals: ActiveSpaceIntegrals,
    sqd: SQDResult,
    basis: str,
    lucj: LUCJConfig,
    ccsd_t: CCSDTStage,
) -> SubsystemResult:
    e_sqd, e_casci, sqd_reasons = promote_sqd_energies(sqd)
    resources = {
        "qpe": qpe_estimate(integrals, basis=basis),
        "sqd": sqd_estimate(integrals, lucj=lucj),
    }
    reasons = dict(sqd_reasons)
    if ccsd_t.reason is not None:
        reasons["e_ccsd_t"] = ccsd_t.reason
    return dataclasses.replace(
        classical,
        e_ccsd_t=ccsd_t.energies[classical.name],
        e_casci=e_casci,
        e_sqd=e_sqd,
        active_space=integrals.record,
        sqd=sqd,
        resources=resources,
        null_reasons=reasons,
    )


def _run_sqd_stage(
    integrals: Mapping[str, ActiveSpaceIntegrals],
    lucj: LUCJConfig,
    reference: ReferenceConfig,
    seed: int,
) -> SQDStageOutcome:
    jobs = [
        SQDJob(i.h1, i.h2, i.e_core, i.record.ncas, i.nelec, i.context, lucj, reference, seed)
        for i in integrals.values()
    ]
    return _make_runner().run(jobs)


def _null_layer_reason(results: Mapping[str, SubsystemResult], layer: str) -> str:
    field = f"e_{layer}"
    nulls = [name for name in COUNTERPOISE_SUBSYSTEMS if getattr(results[name], field) is None]
    return f"{field} is null for {', '.join(nulls)}: {results[nulls[0]].null_reasons[field]}"


def _installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None
