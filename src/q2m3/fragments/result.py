"""Result records of the fragment interaction-energy workflow.

Interaction energies are derived values: ``validate()`` recomputes them from the
subsystem energies and rejects stored values that disagree.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
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
    estimate from that slot's estimator.
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
        """Check the null reasons and the resource slots.

        Raises:
            ValueError: A null field has no ``null_reasons`` entry; a resource key is not
                ``qpe`` or ``sqd``; or a resource slot holds something other than an
                estimate from that slot's estimator.
        """
        owner = f"subsystem {self.name}"
        _require_reasons(owner, self._null_keys(), self.null_reasons)
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
    references: tuple[Any, ...]
    errors: ErrorDecomposition
    correspondence: CorrespondenceReport | None
    provenance: Mapping[str, Any]
    null_reasons: Mapping[str, str]
    warnings: tuple[str, ...] = ()

    def validate(self) -> FragmentInteractionResult:
        """Recompute the interaction energies and check the null reasons.

        Raises:
            ValueError: The subsystems are not ``AB``, ``A``, ``B``; a classical-only
                result holds an SQD record; a stored interaction energy differs from the
                recomputed one by more than ``DELTA_E_TOLERANCE_HA``; or a null field
                has no reason.
        """
        if set(self.subsystems) != set(COUNTERPOISE_SUBSYSTEMS):
            raise ValueError(f"subsystems must be AB, A, B; got {', '.join(self.subsystems)}")
        if self.mode == "classical_only":
            holding = [name for name, sub in self.subsystems.items() if sub.sqd is not None]
            if holding:
                raise ValueError(
                    f"mode='classical_only' result holds an SQD record for {', '.join(holding)}"
                )
        for subsystem in self.subsystems.values():
            subsystem.validate()
        for layer in ENERGY_LAYERS:
            expected = interaction_energy(self.subsystems, layer)
            stored = self.delta_e.get(layer)
            consistent = (stored is None and expected is None) or (
                stored is not None
                and expected is not None
                and abs(stored - expected) <= DELTA_E_TOLERANCE_HA
            )
            if not consistent:
                raise ValueError(
                    f"delta_e[{layer!r}] = {stored!r} does not match the subsystem "
                    f"energies, which give {expected!r}"
                )
        _require_reasons("result", self._null_keys(), self.null_reasons)
        return self

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


def _require_reasons(owner: str, null_keys: Iterator[str], reasons: Mapping[str, str]) -> None:
    missing = [key for key in null_keys if key not in reasons]
    if missing:
        raise ValueError(f"{owner}: null field without a reason: {', '.join(missing)}")
