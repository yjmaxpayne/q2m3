"""One-call entry point of the fragment interaction-energy workflow."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from typing import Any

from pyscf import lib

from q2m3._lazy import extra_available
from q2m3.fragments.active_space import check_correspondence, select_active_spaces
from q2m3.fragments.counterpoise import SCF_CONV_TOL, Subsystem, counterpoise
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
    FragmentInteractionResult,
    SubsystemResult,
    interaction_energy,
    promote_sqd_energies,
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
_RESULT_REASONS = {
    "errors.algorithm": "delta_e.sqd or delta_e.casci is null",
    "errors.active_space": "no same-basis full-space reference",
    "errors.basis": "no same-basis full-space reference and no external reference",
    "errors.model": "no external reference",
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
    references: Sequence[Any] | None = None,
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

    Raises:
        TypeError: ``avas_labels`` is not a sequence of strings, or ``avas_threshold``
            is not a real number.
        ValueError: ``mode`` is not ``"full"`` or ``"classical_only"``, or an AVAS label
            is malformed or matches no AO.
        ImportError: ``mode="full"`` and the optional ``sqd`` backend is missing.
        NotImplementedError: ``references`` is not ``None``.
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
    if references is not None:
        raise NotImplementedError("imported references are not supported yet")

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
    } | _RESULT_REASONS
    algorithm = None
    if delta_e["sqd"] is not None and delta_e["casci"] is not None:
        algorithm = delta_e["sqd"] - delta_e["casci"]
        del null_reasons["errors.algorithm"]
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
    }
    if ccsd_t is not None:
        provenance["ccsd_t_cost"] = ccsd_t.cost
    return FragmentInteractionResult(
        fragment_id=molecule.name or molecule.geometry_sha256(),
        mode=mode,
        basis=basis,
        subsystems=results,
        delta_e=delta_e,
        references=(),
        errors=ErrorDecomposition(algorithm=algorithm),
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
