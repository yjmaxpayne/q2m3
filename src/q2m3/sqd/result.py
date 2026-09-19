"""Immutable, finite result records for sampled quantum diagonalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import comb, isclose, isfinite
from numbers import Integral, Real
from typing import Any, Literal

from q2m3.sqd.config import _freeze_snapshot

Tier = Literal["T0", "T1", "T1+", "T2"]
_METHODS = {
    "T0": ("exact_casci",),
    "T1": ("selected_ci_pyscf",),
    "T1+": ("shci_dice", "dmrg_block2"),
    "T2": ("ccsd_t",),
}
_REFERENCE_ONLY_FIELDS = (
    "sqd_energy",
    "delta_mHa",
    "delta_vs_sci_mHa",
    "ratio_sqd_over_sci",
    "iso_ndet_sci_energy",
    "iso_ndet_random_energy",
    "unique_dets_vs_shots",
    "subspace_dim",
    "subspace_dims",
    "n_reps",
    "shots",
    "backend",
)
_TIMING_STAGES = (
    "geometry",
    "integrals",
    "ccsd",
    "prepare",
    "sample",
    "diagonalize",
    "reference",
    "comparison",
    "total",
)


def _number(value: Any, name: str, *, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _integer(value: Any, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _reason(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} requires a nonempty reason or identifier")


def _same(actual: float, expected: float, name: str) -> None:
    _number(actual, name)
    if not isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError(f"{name} is inconsistent with its source values")


def _reference_contract(tier, method, uncertainty, kind, reason, downgrade, warnings):
    if reason is not None:
        _reason(reason, "uncertainty_reason")
    if tier not in _METHODS or method not in _METHODS[tier]:
        raise ValueError("baseline tier/method mismatch")
    if kind not in ("exact_active_space", "solver_estimate", "unknown"):
        raise ValueError("invalid uncertainty kind")
    if uncertainty is not None:
        _number(uncertainty, "uncertainty", nonnegative=True)
    if kind == "unknown":
        if uncertainty is not None:
            raise ValueError("unknown uncertainty must be None")
        _reason(reason, "uncertainty")
    elif uncertainty is None:
        raise ValueError("known uncertainty requires a value")
    if tier == "T0":
        if kind != "exact_active_space" or uncertainty != 0 or downgrade is not None:
            raise ValueError("T0 requires exact_active_space uncertainty=0 and no downgrade")
    else:
        _reason(downgrade, "baseline_downgrade_reason")
        if not warnings:
            raise ValueError("downgraded baseline requires warnings")
        if kind == "exact_active_space":
            raise ValueError("only T0 can claim exact_active_space uncertainty")
    if tier in ("T1", "T2") and kind != "unknown":
        raise ValueError(f"{tier} uncertainty must be unknown")
    if tier == "T1" and reason != "tight_cutoff_error_unknown":
        raise ValueError("T1 uncertainty reason must be tight_cutoff_error_unknown")


@dataclass(frozen=True)
class ReferenceAttempt:
    """One auditable reference selection or execution outcome."""

    tier: Tier
    method: str
    outcome: Literal[
        "selected", "disabled", "unavailable", "out_of_domain", "over_budget", "timeout", "success"
    ]
    reason: str | None
    wall_s: float
    peak_rss_mb: float | None

    def validate(self) -> ReferenceAttempt:
        """Validate the outcome and decimal-MB measurement, returning this record."""
        if self.tier not in _METHODS or self.method not in _METHODS[self.tier]:
            raise ValueError("reference attempt tier/method mismatch")
        if self.outcome not in (
            "selected",
            "disabled",
            "unavailable",
            "out_of_domain",
            "over_budget",
            "timeout",
            "success",
        ):
            raise ValueError("invalid reference attempt outcome")
        if self.reason is not None:
            _reason(self.reason, "reference attempt reason")
        if self.outcome not in ("selected", "success"):
            _reason(self.reason, "reference attempt reason")
        _number(self.wall_s, "wall_s", nonnegative=True)
        if self.peak_rss_mb is not None:
            _number(self.peak_rss_mb, "peak_rss_mb", nonnegative=True)
        return self


@dataclass(frozen=True)
class T1Residual:
    """Adjacent-cutoff residual, attributed to the looser executed point."""

    loose_cutoff: float | None
    tight_cutoff: float
    loose_energy: float | None
    tight_energy: float
    loose_residual_mHa: float | None
    reason: str | None

    def validate(self) -> T1Residual:
        """Check cutoff ordering and the residual, returning this record."""
        if self.reason is not None:
            _reason(self.reason, "residual reason")
        _number(self.tight_cutoff, "tight_cutoff")
        _number(self.tight_energy, "tight_energy")
        if not 0 < self.tight_cutoff < 1:
            raise ValueError("tight_cutoff must lie in (0, 1)")
        if self.loose_cutoff is None:
            if (
                self.loose_energy is not None
                or self.loose_residual_mHa is not None
                or self.reason != "paired_cutoff_not_run"
            ):
                raise ValueError(
                    "missing loose cutoff requires absent values and paired_cutoff_not_run"
                )
        else:
            if self.reason is not None:
                raise ValueError("paired residual must have reason=None")
            _number(self.loose_cutoff, "loose_cutoff")
            _number(self.loose_energy, "loose_energy")
            if not self.tight_cutoff < self.loose_cutoff < 1:
                raise ValueError("loose_cutoff must exceed tight_cutoff and be below 1")
            _number(self.loose_residual_mHa, "loose_residual_mHa", nonnegative=True)
            _same(
                self.loose_residual_mHa,
                1000 * abs(self.loose_energy - self.tight_energy),
                "loose_residual_mHa",
            )
        return self


@dataclass(frozen=True)
class T2Diagnostics:
    """Independent coupled-cluster reliability flags for a T2 baseline."""

    t1_diagnostic: float
    ccsd_converged: bool
    triples_correction_ha: float
    ccsd_correlation_energy_ha: float
    triples_ratio: float | None
    high_t1: bool
    ccsd_not_converged: bool
    high_triples: bool
    ratio_reason: str | None

    def validate(self) -> T2Diagnostics:
        """Check each flag against its own diagnostic, returning this record."""
        _number(self.t1_diagnostic, "t1_diagnostic", nonnegative=True)
        _number(self.triples_correction_ha, "triples_correction_ha")
        _number(self.ccsd_correlation_energy_ha, "ccsd_correlation_energy_ha")
        for name in ("ccsd_converged", "high_t1", "ccsd_not_converged", "high_triples"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        if self.high_t1 != (self.t1_diagnostic > 0.02) or self.ccsd_not_converged != (
            not self.ccsd_converged
        ):
            raise ValueError("T2 diagnostic flags are inconsistent")
        if abs(self.ccsd_correlation_energy_ha) <= 1e-12:
            if (
                self.triples_ratio is not None
                or not self.high_triples
                or self.ratio_reason != "correlation_denominator_too_small"
            ):
                raise ValueError(
                    "small correlation denominator requires unknown ratio and high_triples"
                )
        else:
            ratio = abs(self.triples_correction_ha / self.ccsd_correlation_energy_ha)
            _number(self.triples_ratio, "triples_ratio", nonnegative=True)
            _same(self.triples_ratio, ratio, "triples_ratio")
            if self.high_triples != (ratio > 0.10) or self.ratio_reason is not None:
                raise ValueError("triples diagnostic flag/reason is inconsistent")
        return self


@dataclass(frozen=True)
class ReferenceResult:
    """A reference energy with explicit uncertainty and execution provenance."""

    energy: float
    tier: Tier
    method: str
    uncertainty_mHa: float | None
    uncertainty_kind: Literal["exact_active_space", "solver_estimate", "unknown"]
    uncertainty_reason: str | None
    downgrade_reason: str | None
    t1_residual: T1Residual | None
    t2: T2Diagnostics | None
    untrustworthy: bool
    hamiltonian_id: str
    frame_id: str
    attempts: tuple[ReferenceAttempt, ...]
    warnings: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "attempts", tuple(self.attempts))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def validate(self) -> ReferenceResult:
        """Validate reference metadata and diagnostics, returning this record."""
        _number(self.energy, "energy")
        _reason(self.hamiltonian_id, "hamiltonian_id")
        _reason(self.frame_id, "frame_id")
        _reference_contract(
            self.tier,
            self.method,
            self.uncertainty_mHa,
            self.uncertainty_kind,
            self.uncertainty_reason,
            self.downgrade_reason,
            self.warnings,
        )
        _diagnostics(self.tier, self.energy, self.t1_residual, self.t2, self.untrustworthy)
        for attempt in self.attempts:
            if not isinstance(attempt, ReferenceAttempt):
                raise ValueError("attempts must contain ReferenceAttempt records")
            attempt.validate()
        for warning in self.warnings:
            _reason(warning, "warning")
        return self


def _diagnostics(tier, energy, residual, t2, untrustworthy):
    if not isinstance(untrustworthy, bool):
        raise ValueError("untrustworthy must be bool")
    if tier == "T1":
        if not isinstance(residual, T1Residual):
            raise ValueError("T1 requires baseline_t1_residual")
        residual.validate()
        _same(residual.tight_energy, energy, "tight_energy")
    elif residual is not None:
        raise ValueError("baseline_t1_residual is only valid for T1")
    if tier == "T2":
        if not isinstance(t2, T2Diagnostics):
            raise ValueError("T2 requires t2_diagnostics")
        t2.validate()
        if untrustworthy != (t2.high_t1 or t2.ccsd_not_converged or t2.high_triples):
            raise ValueError("baseline_untrustworthy must reflect all three T2 flags")
    elif t2 is not None:
        raise ValueError("t2_diagnostics is only valid for T2")


@dataclass(frozen=True)
class SQDResult:
    """Complete SQD report, with absent work represented by None and a reason."""

    schema_version: Literal["sqd.result.v1"]
    status: Literal["completed", "reference_only"]
    sqd_energy: float | None
    hf_energy: float
    hf_reference_kind: str
    baseline_energy: float
    iso_active_space_ccsd_energy: float | None
    iso_ndet_sci_energy: float | None
    iso_ndet_random_energy: float | None
    baseline_tier: Tier
    baseline_method: str
    baseline_uncertainty_mHa: float | None
    baseline_uncertainty_kind: str
    baseline_downgrade_reason: str | None
    baseline_untrustworthy: bool
    baseline_t1_residual: T1Residual | None
    t1_diagnostic: float | None
    t2_diagnostics: T2Diagnostics | None
    reference_attempts: tuple[ReferenceAttempt, ...]
    delta_mHa: float | None
    delta_vs_sci_mHa: float | None
    ratio_sqd_over_sci: float | None
    unique_dets_vs_shots: tuple[tuple[int, int], ...] | None
    active_space: tuple[int, int]
    n_reps: int | None
    shots: int | None
    seed: int
    subspace_dim: int | None
    subspace_dims: tuple[int, int] | None
    full_ci_dim: int
    embedding_mode: str
    two_electron_tensor_fixed: bool
    fixed_mo: bool
    backend: str | None
    versions: Mapping[str, str]
    diagnostics: Mapping[str, Any]
    provenance: Mapping[str, Any]
    timings_s: Mapping[str, float | None]
    null_reasons: Mapping[str, str]
    warnings: tuple[str, ...]

    def __post_init__(self):
        for name in ("versions", "diagnostics", "provenance", "timings_s", "null_reasons"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping")
            object.__setattr__(self, name, _freeze_snapshot(value))
        for name in ("active_space", "subspace_dims", "unique_dets_vs_shots", "warnings"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _freeze_snapshot(getattr(self, name)))
        object.__setattr__(self, "reference_attempts", tuple(self.reference_attempts))

    def validate(self) -> SQDResult:
        """Validate finite data, reference semantics and execution consistency.

        Returns:
            This immutable result.

        Raises:
            ValueError: A field contradicts the result schema or executed mode.
        """
        if self.schema_version != "sqd.result.v1" or self.status not in (
            "completed",
            "reference_only",
        ):
            raise ValueError("invalid schema_version or status")
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.endswith("energy") or field.name in (
                "delta_mHa",
                "delta_vs_sci_mHa",
                "t1_diagnostic",
                "ratio_sqd_over_sci",
            ):
                if value is not None:
                    _number(value, field.name)
            if value is None and field.name != "baseline_downgrade_reason":
                _reason(self.null_reasons.get(field.name), field.name)
        _number(self.hf_energy, "hf_energy")
        _number(self.baseline_energy, "baseline_energy")
        _reference_contract(
            self.baseline_tier,
            self.baseline_method,
            self.baseline_uncertainty_mHa,
            self.baseline_uncertainty_kind,
            self.null_reasons.get("baseline_uncertainty_mHa"),
            self.baseline_downgrade_reason,
            self.warnings,
        )
        _diagnostics(
            self.baseline_tier,
            self.baseline_energy,
            self.baseline_t1_residual,
            self.t2_diagnostics,
            self.baseline_untrustworthy,
        )
        if self.t2_diagnostics is not None:
            _same(self.t1_diagnostic, self.t2_diagnostics.t1_diagnostic, "t1_diagnostic")
        elif self.t1_diagnostic is not None:
            raise ValueError("t1_diagnostic is only public for T2")
        self._validate_metadata()
        self._validate_mode()
        return self

    def _validate_metadata(self):
        _integer(self.seed, "seed", 0)
        if len(self.active_space) != 2:
            raise ValueError("active_space must contain electrons and orbitals")
        nelec, norb = self.active_space
        _integer(nelec, "active_electrons")
        _integer(norb, "active_orbitals")
        if nelec % 2 or nelec > 2 * norb:
            raise ValueError("active_space must be closed-shell and within capacity")
        _integer(self.full_ci_dim, "full_ci_dim")
        if self.full_ci_dim != comb(norb, nelec // 2) ** 2:
            raise ValueError("full_ci_dim differs from active_space")
        if self.embedding_mode not in ("vacuum", "diagonal", "full_oneelectron"):
            raise ValueError("invalid embedding_mode")
        if self.hf_reference_kind not in ("canonical_rhf", "fixed_frame_determinant"):
            raise ValueError("invalid hf_reference_kind")
        if self.embedding_mode != "vacuum" and self.hf_reference_kind != "fixed_frame_determinant":
            raise ValueError("MM embedding requires fixed_frame_determinant")
        for name in ("fixed_mo", "two_electron_tensor_fixed"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        for key, value in self.versions.items():
            _reason(value, f"versions.{key}")
        for key, value in self.null_reasons.items():
            _reason(value, f"null_reasons.{key}")
        for warning in self.warnings:
            _reason(warning, "warning")
        for attempt in self.reference_attempts:
            if not isinstance(attempt, ReferenceAttempt):
                raise ValueError("reference_attempts must contain ReferenceAttempt records")
            attempt.validate()
        for name in _TIMING_STAGES:
            if name not in self.timings_s:
                raise ValueError(f"timings_s missing {name}")
        for name, duration in self.timings_s.items():
            if duration is None:
                _reason(self.null_reasons.get(f"timings_s.{name}"), f"timings_s.{name}")
            else:
                _number(duration, f"timings_s.{name}", nonnegative=True)

        if self.timings_s["reference"] is None:
            raise ValueError("timings_s.reference is required for the baseline energy")
        total = self.timings_s["total"]
        if total is None:
            raise ValueError("timings_s.total is required")
        elapsed = sum(
            value for name, value in self.timings_s.items() if name != "total" and value is not None
        )
        if elapsed > total + 1e-9:
            raise ValueError("timings_s.total must cover disjoint executed stages")

    def _validate_mode(self):
        if self.status == "reference_only":
            for name in _REFERENCE_ONLY_FIELDS:
                if (
                    getattr(self, name) is not None
                    or self.null_reasons.get(name) != "reference_only"
                ):
                    raise ValueError(f"{name} must be None with reference_only reason")
            for name in ("prepare", "sample", "diagonalize", "comparison"):
                if self.timings_s[name] is not None:
                    raise ValueError(f"timings_s.{name} must be None in reference_only")
            return
        required = tuple(
            name
            for name in _REFERENCE_ONLY_FIELDS
            if name not in ("ratio_sqd_over_sci", "iso_ndet_random_energy")
        ) + ("iso_active_space_ccsd_energy",)
        for stage in (
            "prepare",
            "sample",
            "diagonalize",
            "reference",
            "comparison",
        ):
            if self.timings_s[stage] is None:
                raise ValueError(f"timings_s.{stage} is required for completed results")
        for name in required:
            if getattr(self, name) is None:
                raise ValueError(f"{name} is required for completed results")
        _integer(self.n_reps, "n_reps")
        _integer(self.shots, "shots")
        _reason(self.backend, "backend")
        if len(self.subspace_dims) != 2:
            raise ValueError("subspace_dims must contain alpha and beta dimensions")
        for dim in self.subspace_dims:
            _integer(dim, "subspace_dims")
            if dim > comb(self.active_space[1], self.active_space[0] // 2):
                raise ValueError("subspace_dims exceeds the spin sector")
        _integer(self.subspace_dim, "subspace_dim")
        if self.subspace_dim != self.subspace_dims[0] * self.subspace_dims[1]:
            raise ValueError("subspace_dim must equal alpha * beta")
        self._validate_curve()
        _same(self.delta_mHa, 1000 * (self.sqd_energy - self.baseline_energy), "delta_mHa")
        _same(
            self.delta_vs_sci_mHa,
            1000 * (self.sqd_energy - self.iso_ndet_sci_energy),
            "delta_vs_sci_mHa",
        )
        self._validate_ratio()

    def _validate_curve(self):
        last_shots, last_unique = 0, 0
        for point in self.unique_dets_vs_shots:
            if len(point) != 2:
                raise ValueError("unique_dets_vs_shots requires (shots, pairs) points")
            shots, unique = point
            _integer(shots, "curve shots")
            _integer(unique, "curve unique pairs")
            if shots <= last_shots or unique < last_unique or unique > min(shots, self.full_ci_dim):
                raise ValueError("unique_dets_vs_shots must be cumulative sampled pairs")
            last_shots, last_unique = shots, unique
        if last_shots != self.shots:
            raise ValueError("unique_dets_vs_shots must end at total shots")

    def _validate_ratio(self):
        if self.ratio_sqd_over_sci is not None:
            _number(self.ratio_sqd_over_sci, "ratio_sqd_over_sci", nonnegative=True)
        if self.baseline_tier != "T0":
            if (
                self.ratio_sqd_over_sci is not None
                or self.null_reasons.get("ratio_sqd_over_sci") != "non_exact_reference"
            ):
                raise ValueError("ratio_sqd_over_sci requires non_exact_reference absence")
            return
        sqd_gap = self.sqd_energy - self.baseline_energy
        sci_gap = self.iso_ndet_sci_energy - self.baseline_energy
        if sqd_gap < -1e-10 or sci_gap < -1e-10:
            raise ValueError("energy below exact variational bound")
        if -1e-10 <= sqd_gap < 0 and self.diagnostics.get("ratio_tolerance_ha") != 1e-10:
            raise ValueError("tiny negative SQD gap requires ratio_tolerance_ha=1e-10")
        if sci_gap > 1e-10:
            _same(self.ratio_sqd_over_sci, max(0.0, sqd_gap) / sci_gap, "ratio_sqd_over_sci")
        elif self.ratio_sqd_over_sci is not None:
            raise ValueError("ratio_sqd_over_sci undefined at small SCI denominator")
