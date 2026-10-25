"""Result records of the fragment interaction-energy workflow.

Interaction energies are derived values: ``validate()`` recomputes them from the
subsystem energies and rejects stored values that disagree.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from q2m3.sqd.result import SQDResult

SCHEMA_VERSION = "fragments.result.v1"
ENERGY_LAYERS = ("hf", "mp2", "casci", "sqd", "ccsd_t")
ERROR_TERMS = ("algorithm", "active_space", "basis", "model")
RESOURCE_KEYS = ("qpe", "sqd")
QPE_ESTIMATOR = "pennylane.estimator.DoubleFactorization"
SQD_ESTIMATOR = "q2m3.sqd.resources"
RESOURCE_ESTIMATORS = {"qpe": QPE_ESTIMATOR, "sqd": SQD_ESTIMATOR}
COUNTERPOISE_SUBSYSTEMS = ("AB", "A", "B")
DELTA_E_TOLERANCE_HA = 1e-12
REFERENCE_CATEGORIES = ("same_basis_full_space", "external")
REQUIRED_PROVENANCE = (
    "geometry_sha256",
    "basis",
    "cart",
    "avas_labels",
    "avas_threshold",
    "cos_threshold",
    "seed",
    "scf_conv_tol",
    "frozen_core",
    "versions",
    "threads",
)
FULL_MODE_PROVENANCE = ("ccsd_t_cost",)

_LAYER_FIELDS = {
    "hf": "e_hf",
    "mp2": "e_mp2",
    "casci": "e_casci",
    "sqd": "e_sqd",
    "ccsd_t": "e_ccsd_t",
}


@dataclass(frozen=True)
class ActiveSpaceRecord:
    """Active space of one subsystem: ``nelecas`` electrons in ``ncas`` orbitals.

    ``ao_labels`` are the anchored AO label patterns given to AVAS, and ``minao_indices``
    the matched AOs of the MINAO copy. ``frame_id`` and ``hamiltonian_id`` stay ``None``
    until the active-space integrals are built.
    """

    nelecas: int
    ncas: int
    ncore: int
    ao_labels: tuple[str, ...]
    minao_indices: tuple[int, ...]
    frame_id: str | None = None
    hamiltonian_id: str | None = None


@dataclass(frozen=True)
class CorrespondenceReport:
    """Whether the monomer active spaces together span the dimer active space."""

    counts_conserved: bool
    min_cos_occupied: float
    min_cos_virtual: float
    rank_occupied: int
    rank_virtual: int
    cos_threshold: float
    corresponds: bool


@dataclass(frozen=True)
class ResourceEstimate:
    """One resource estimate of a subsystem and the estimator that produced it.

    ``versions`` maps package names to installed versions, ``values`` holds the estimated
    quantities, and ``scope`` states what the quantities describe.
    """

    estimator: str
    versions: Mapping[str, str]
    values: Mapping[str, Any]
    scope: str


@dataclass(frozen=True)
class ExternalReference:
    """Imported interaction energy of one fragment dimer, in Ha.

    ``category`` is ``"same_basis_full_space"`` when ``basis`` is the workflow basis and
    ``"external"`` otherwise.
    """

    fragment_id: str
    method: str
    basis: str
    delta_e: float
    category: str


@dataclass(frozen=True)
class ErrorDecomposition:
    """Signed error terms of the interaction energy in Ha; ``None`` when not formed."""

    algorithm: float | None = None
    active_space: float | None = None
    basis: float | None = None
    model: float | None = None


@dataclass(frozen=True, kw_only=True)
class SubsystemResult:
    """Energies and records of one subsystem: ``AB``, ``A``, ``B``, or ``M`` (one molecule).

    Every ``None`` field has a ``null_reasons`` entry under its field name; a missing
    resource estimate uses ``resources.<qpe|sqd>``. Each resource slot holds only an
    estimate from that slot's estimator. ``e_sqd`` and ``e_casci`` are derived from
    ``sqd`` by :func:`promote_sqd_energies`.
    """

    name: Literal["AB", "A", "B", "M"]
    real_atoms: tuple[int, ...]
    ghost_atoms: tuple[int, ...]
    n_ao: int
    e_hf: float | None
    e_mp2: float | None
    e_ccsd_t: float | None
    e_casci: float | None
    e_sqd: float | None
    active_space: ActiveSpaceRecord | None
    sqd: SQDResult | None
    resources: Mapping[str, ResourceEstimate | None]
    null_reasons: Mapping[str, str]

    def validate(self) -> SubsystemResult:
        """Check the null reasons, the resource slots and the SQD record.

        Raises:
            ValueError: A null field has no ``null_reasons`` entry; a resource key is not
                ``qpe`` or ``sqd``; a resource slot holds something other than an
                estimate from that slot's estimator; ``sqd`` is not exactly an
                ``SQDResult`` or fails its own validation; the active space is not the
                one of the SQD run; or ``e_sqd`` or ``e_casci`` differs from the value
                derived from ``sqd``.
        """
        owner = f"subsystem {self.name}"
        _require_reasons(owner, self._null_keys(), self.null_reasons)
        self._check_sqd(owner)
        unknown = [key for key in self.resources if key not in RESOURCE_KEYS]
        if unknown:
            raise ValueError(f"{owner}: unknown resource keys: {', '.join(unknown)}")
        for key, estimate in self.resources.items():
            if estimate is None:
                continue
            estimator = estimate.estimator if isinstance(estimate, ResourceEstimate) else None
            if estimator != RESOURCE_ESTIMATORS[key]:
                raise ValueError(
                    f"{owner}: resources[{key!r}] holds an estimate from {estimator!r}; "
                    f"expected {RESOURCE_ESTIMATORS[key]!r}"
                )
        return self

    def _check_sqd(self, owner: str) -> None:
        if self.sqd is not None:
            if type(self.sqd) is not SQDResult:
                raise ValueError(
                    f"{owner}: sqd must be an SQDResult, got {type(self.sqd).__name__}"
                )
            self.sqd.validate()
            expected = self.sqd.provenance["context"]["hamiltonian_id"]
            actual = None if self.active_space is None else self.active_space.hamiltonian_id
            if actual != expected:
                raise ValueError(
                    f"{owner}: active-space hamiltonian_id {actual!r} differs from the SQD "
                    f"context hamiltonian_id {expected!r}"
                )
        e_sqd, e_casci, _ = promote_sqd_energies(self.sqd)
        for field, expected_energy in (("e_sqd", e_sqd), ("e_casci", e_casci)):
            stored = getattr(self, field)
            if stored != expected_energy:
                raise ValueError(
                    f"{owner}: {field} = {stored!r} differs from {expected_energy!r}, "
                    "the value derived from sqd"
                )

    def _null_keys(self) -> Iterator[str]:
        nullable = ("e_hf", "e_mp2", "e_ccsd_t", "e_casci", "e_sqd", "active_space", "sqd")
        yield from (name for name in nullable if getattr(self, name) is None)
        yield from (f"resources.{key}" for key in RESOURCE_KEYS if self.resources.get(key) is None)


@dataclass(frozen=True, kw_only=True)
class FragmentInteractionResult:
    """Counterpoise-corrected interaction energies of one fragment dimer, in Ha.

    ``delta_e`` maps each layer in ``ENERGY_LAYERS`` to E(AB) - E(A) - E(B), all three in
    the dimer basis. Null entries of ``delta_e`` and ``errors`` have ``null_reasons``
    entries under dotted keys such as ``delta_e.casci`` and ``errors.model``.
    """

    schema_version: str = SCHEMA_VERSION
    fragment_id: str
    mode: str
    basis: str
    subsystems: Mapping[str, SubsystemResult]
    delta_e: Mapping[str, float | None]
    references: tuple[ExternalReference, ...]
    errors: ErrorDecomposition
    correspondence: CorrespondenceReport | None
    provenance: Mapping[str, Any]
    null_reasons: Mapping[str, str]
    warnings: tuple[str, ...] = ()

    def validate(self) -> FragmentInteractionResult:
        """Recompute the interaction energies and check the null reasons.

        Raises:
            ValueError: The subsystems are not ``AB``, ``A``, ``B``; a classical-only
                result holds an SQD record, or a full result lacks one; the provenance
                lacks a required key; a stored interaction energy or error term differs
                from the recomputed one by more than ``DELTA_E_TOLERANCE_HA``; the model
                error is not the sum of the active-space and basis errors; the recorded
                same-basis reference source is not the one used; a reference category is
                unknown or repeated; or a null field has no reason.
        """
        if set(self.subsystems) != set(COUNTERPOISE_SUBSYSTEMS):
            raise ValueError(f"subsystems must be AB, A, B; got {', '.join(self.subsystems)}")
        self._check_sqd_records()
        self._check_provenance()
        for subsystem in self.subsystems.values():
            subsystem.validate()
        expected = {layer: interaction_energy(self.subsystems, layer) for layer in ENERGY_LAYERS}
        for layer, energy in expected.items():
            stored = self.delta_e.get(layer)
            if not _agrees(stored, energy):
                raise ValueError(
                    f"delta_e[{layer!r}] = {stored!r} does not match the subsystem "
                    f"energies, which give {energy!r}"
                )
        self._check_errors(expected)
        _require_reasons("result", self._null_keys(), self.null_reasons)
        return self

    def _check_provenance(self) -> None:
        required = REQUIRED_PROVENANCE + (FULL_MODE_PROVENANCE if self.mode == "full" else ())
        missing = [key for key in required if key not in self.provenance]
        if missing:
            raise ValueError(f"provenance lacks required keys: {', '.join(missing)}")

    def _check_errors(self, delta_e: Mapping[str, float | None]) -> None:
        expected, source = decompose_errors(delta_e, self.references)
        for term in ERROR_TERMS:
            stored, value = getattr(self.errors, term), getattr(expected, term)
            if not _agrees(stored, value):
                raise ValueError(
                    f"errors.{term} = {stored!r} does not match the recomputed value {value!r}"
                )
        terms = (self.errors.model, self.errors.active_space, self.errors.basis)
        if None not in terms and not _agrees(terms[0], terms[1] + terms[2]):
            raise ValueError(
                f"errors.model = {terms[0]!r} differs from errors.active_space + errors.basis "
                f"= {terms[1] + terms[2]!r}"
            )
        recorded = self.provenance.get("same_basis_reference_source")
        if recorded != source:
            raise ValueError(
                f"provenance same_basis_reference_source = {recorded!r}; the errors use {source!r}"
            )

    def _check_sqd_records(self) -> None:
        if self.mode == "classical_only":
            holding = [name for name, sub in self.subsystems.items() if sub.sqd is not None]
            if holding:
                raise ValueError(
                    f"mode='classical_only' result holds an SQD record for {', '.join(holding)}"
                )
        if self.mode == "full":
            missing = [
                name for name in COUNTERPOISE_SUBSYSTEMS if self.subsystems[name].sqd is None
            ]
            if missing:
                raise ValueError(f"mode='full' result has no SQD record for {', '.join(missing)}")

    def _null_keys(self) -> Iterator[str]:
        yield from (
            f"delta_e.{layer}" for layer in ENERGY_LAYERS if self.delta_e.get(layer) is None
        )
        yield from (f"errors.{term}" for term in ERROR_TERMS if getattr(self.errors, term) is None)
        if self.correspondence is None:
            yield "correspondence"


def interaction_energy(subsystems: Mapping[str, SubsystemResult], layer: str) -> float | None:
    """Return E(AB) - E(A) - E(B) for ``layer``, or ``None`` if any of the three is null."""
    energies = [getattr(subsystems[name], _LAYER_FIELDS[layer]) for name in COUNTERPOISE_SUBSYSTEMS]
    if any(energy is None for energy in energies):
        return None
    e_ab, e_a, e_b = energies
    return e_ab - e_a - e_b


def references_by_category(
    references: Sequence[ExternalReference],
) -> dict[str, ExternalReference]:
    """Map each reference category to its single reference.

    Raises:
        ValueError: A category is not in ``REFERENCE_CATEGORIES``, or two references share
            a category.
    """
    found: dict[str, ExternalReference] = {}
    for reference in references:
        if reference.category not in REFERENCE_CATEGORIES:
            raise ValueError(
                f"reference category must be one of {REFERENCE_CATEGORIES}, "
                f"got {reference.category!r}"
            )
        if reference.category in found:
            raise ValueError(f"more than one {reference.category!r} reference")
        found[reference.category] = reference
    return found


def decompose_errors(
    delta_e: Mapping[str, float | None], references: Sequence[ExternalReference]
) -> tuple[ErrorDecomposition, str | None]:
    """Return the error terms and the source of the same-basis reference.

    The same-basis full-space reference is the PySCF CCSD(T) interaction energy
    (``"pyscf_ccsd_t"``) or, when that is null, the imported same-basis reference
    (``"imported"``). A term is ``None`` when one of its two energies is missing.

    Raises:
        ValueError: As :func:`references_by_category`.
    """
    by_category = references_by_category(references)
    same, source = delta_e.get("ccsd_t"), "pyscf_ccsd_t"
    if same is None and "same_basis_full_space" in by_category:
        same, source = by_category["same_basis_full_space"].delta_e, "imported"
    if same is None:
        source = None
    external = by_category["external"].delta_e if "external" in by_category else None
    casci = delta_e.get("casci")
    errors = ErrorDecomposition(
        algorithm=_difference(delta_e.get("sqd"), casci),
        active_space=_difference(casci, same),
        basis=_difference(same, external),
        model=_difference(casci, external),
    )
    return errors, source


def promote_sqd_energies(
    sqd: SQDResult | None,
) -> tuple[float | None, float | None, dict[str, str]]:
    """Return ``e_sqd``, ``e_casci`` and the null reasons of null ones, derived from ``sqd``.

    ``e_casci`` is the baseline energy only for an exact-CASCI T0 baseline. With no SQD
    record both energies are ``None`` and the caller gives the reasons.
    """
    if sqd is None:
        return None, None, {}
    reasons = {}
    e_casci = None
    if sqd.baseline_tier == "T0" and sqd.baseline_method == "exact_casci":
        e_casci = sqd.baseline_energy
    else:
        reasons["e_casci"] = f"baseline_tier={sqd.baseline_tier}"
    if sqd.sqd_energy is None:
        reasons["e_sqd"] = f"sqd_energy is null: {sqd.null_reasons.get('sqd_energy')}"
    return sqd.sqd_energy, e_casci, reasons


def _difference(minuend: float | None, subtrahend: float | None) -> float | None:
    if minuend is None or subtrahend is None:
        return None
    return minuend - subtrahend


def _agrees(stored: float | None, expected: float | None) -> bool:
    if stored is None or expected is None:
        return stored is None and expected is None
    return abs(stored - expected) <= DELTA_E_TOLERANCE_HA


def _require_reasons(owner: str, null_keys: Iterator[str], reasons: Mapping[str, str]) -> None:
    missing = [key for key in null_keys if key not in reasons]
    if missing:
        raise ValueError(f"{owner}: null field without a reason: {', '.join(missing)}")
