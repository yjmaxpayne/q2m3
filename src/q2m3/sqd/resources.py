"""Versioned empirical RSS estimates and allocation guards, in decimal MB.

Estimates describe a bounded serial workload, not a mathematical memory guarantee.
Executors must also enforce a process-tree RSS cap and a wall deadline. Loading a
model checks the environment; estimating with an already loaded model is pure.
"""

from __future__ import annotations

import json
import math
import platform
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from pathlib import Path
from typing import Literal, Protocol

from q2m3.sqd.config import validate_active_space
from q2m3.sqd.exceptions import ResourceLimitError, ResourceModelDomainError

Stage = Literal["integrals", "ccsd", "prepare", "sample", "diagonalize", "reference", "comparison"]
STAGES = ("integrals", "ccsd", "prepare", "sample", "diagonalize", "reference", "comparison")
WARN_RSS_MB = 2048.0
REJECT_RSS_MB = 8192.0
HARD_RSS_MB = 12288.0


class ResourceModel(Protocol):
    """Pure upper-estimate provider for a specified workload and environment."""

    model_id: str

    def upper_bound_mb(
        self,
        norb: int,
        nelec: tuple[int, int],
        *,
        stage: Stage,
        subspace_dims: tuple[int, int] | None,
        retained_mb: float,
        solver_method: str | None,
    ) -> float:
        """Return a finite decimal-MB estimate including retained allocations."""
        ...


def _finite_nonnegative(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite nonnegative number")


def _dimensions(norb: int, nelec: tuple[int, int], dims: tuple[int, int]) -> None:
    if (
        not isinstance(dims, tuple | list)
        or len(dims) != 2
        or any(
            isinstance(d, bool) or not isinstance(d, Integral) or not 1 <= d <= math.comb(norb, n)
            for d, n in zip(dims, nelec, strict=True)
        )
    ):
        raise ResourceModelDomainError(
            f"Invalid subspace_dims={dims} for norb={norb}, nelec={nelec}"
        )


def estimate_rss_mb(
    norb: int,
    nelec: tuple[int, int],
    *,
    stage: Stage,
    subspace_dims: tuple[int, int] | None = None,
    retained_mb: float = 0.0,
    solver_method: str | None = None,
    model: ResourceModel,
) -> float:
    """Estimate total simultaneously resident memory for one serial stage.

    Args:
        norb: Active spatial orbitals.
        nelec: Balanced alpha/beta populations.
        stage: Operation about to allocate memory.
        subspace_dims: Candidate spin-sector sizes, required for fixed spaces.
        retained_mb: Additional live inputs outside the model's workload inventory.
        solver_method: Required reference solver identity.
        model: Versioned model already checked against its runtime environment.

    Returns:
        Finite decimal MB, including imports, workspace and retained inputs.

    Raises:
        ValueError: Malformed counts or retained memory.
        ResourceModelDomainError: Missing dimensions, unsupported stage or invalid bound.
    """
    validate_active_space(norb, nelec)
    _finite_nonnegative(retained_mb, "retained_mb")
    if stage not in STAGES:
        raise ResourceModelDomainError(f"Unknown stage={stage}")
    if stage in ("diagonalize", "comparison") and subspace_dims is None:
        raise ResourceModelDomainError(f"stage={stage} requires candidate subspace_dims")
    if stage == "reference" and not solver_method:
        raise ResourceModelDomainError("reference requires solver_method")
    if subspace_dims is not None:
        _dimensions(norb, nelec, subspace_dims)
    bound = model.upper_bound_mb(
        norb,
        nelec,
        stage=stage,
        subspace_dims=subspace_dims,
        retained_mb=retained_mb,
        solver_method=solver_method,
    )
    try:
        _finite_nonnegative(bound, "predicted_rss_mb")
    except ValueError as exc:
        raise ResourceModelDomainError(str(exc)) from exc
    if bound < retained_mb:
        raise ResourceModelDomainError("Model estimate omits retained memory")
    return float(bound)


def guard_allocation(
    norb: int,
    nelec: tuple[int, int],
    *,
    stage: Stage,
    model: ResourceModel,
    host_available_mb: float,
    subspace_dims: tuple[int, int] | None = None,
    retained_mb: float = 0.0,
    solver_method: str | None = None,
    rss_budget_mb: float | None = None,
    allow_large: bool = False,
) -> float:
    """Reject an unsafe operation before allocating its working space.

    Args:
        norb: Active spatial orbitals.
        nelec: Balanced alpha/beta populations.
        stage: Stage about to run.
        model: Applicable empirical model.
        host_available_mb: Conservative total run cap supplied by the executor.
        subspace_dims: Actual candidates, or a proven pre-call dimension bound.
        retained_mb: Additional simultaneously live memory outside the inventory.
        solver_method: Reference solver identity, when applicable.
        rss_budget_mb: Optional user cap; override never bypasses it.
        allow_large: Permit estimates above the default soft rejection boundary.

    Returns:
        The checked estimate in decimal MB.

    Raises:
        ResourceLimitError: Estimate reaches any applicable rejection boundary.
        ResourceModelDomainError: No applicable bound exists, even with override.
    """
    if not isinstance(allow_large, bool):
        raise ValueError("allow_large must be bool")
    caps = [HARD_RSS_MB, host_available_mb]
    if not allow_large:
        caps.append(REJECT_RSS_MB)
    if rss_budget_mb is not None:
        caps.append(rss_budget_mb)
    for cap in caps:
        _finite_nonnegative(cap, "RSS cap")
        if cap == 0:
            raise ValueError("RSS cap must be positive")
    budget = min(caps)
    try:
        predicted = estimate_rss_mb(
            norb,
            nelec,
            stage=stage,
            subspace_dims=subspace_dims,
            retained_mb=retained_mb,
            solver_method=solver_method,
            model=model,
        )
    except ResourceModelDomainError as exc:
        raise ResourceModelDomainError(
            f"predicted=unavailable, budget={budget:.6f} MB, norb={norb}, "
            f"nelec={nelec}, subspace_dims={subspace_dims}, stage={stage}: {exc}"
        ) from exc
    context = (
        f"predicted={predicted:.6f} MB, budget={budget:.6f} MB, norb={norb}, "
        f"nelec={nelec}, subspace_dims={subspace_dims}, stage={stage}, model={model.model_id}"
    )
    if predicted >= budget:
        raise ResourceLimitError(context)
    if predicted >= WARN_RSS_MB:
        warnings.warn(
            f"RSS reaches 2048 MB warning boundary: {context}", RuntimeWarning, stacklevel=2
        )
    return predicted


def addon_subspace_bound(
    norb: int,
    nelec: tuple[int, int],
    *,
    max_dim: int | tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Bound every recovery/batch/spin expansion by the complete spin sectors.

    Args:
        norb: Active spatial orbitals.
        nelec: Balanced alpha/beta populations.
        max_dim: Optional dimension cap actually passed to the supported addon.

    Returns:
        Per-sector bound independent of sample uniqueness or recovery history.
        Batch count and workspace lifetimes must separately match the model profile.
    """
    validate_active_space(norb, nelec)
    full = tuple(math.comb(norb, n) for n in nelec)
    if max_dim is None:
        return full
    dims = (max_dim, max_dim) if isinstance(max_dim, Integral) else max_dim
    if (
        not isinstance(dims, tuple | list)
        or len(dims) != 2
        or any(isinstance(d, bool) or not isinstance(d, Integral) or d < 1 for d in dims)
    ):
        raise ValueError("max_dim must contain positive integer sector limits")
    return tuple(min(d, f) for d, f in zip(dims, full, strict=True))


def workspace_features(
    norb: int,
    nelec: tuple[int, int],
    dims: tuple[int, int] | None,
) -> tuple[float, ...]:
    """Return intercept, CI, integral, subspace and link-table memory features.

    All nonconstant features are decimal MB. The shape feature bounds selected-CI
    double-annihilation link tables by 16*(da+db)*occupied**2*virtual**2 bytes.
    The subspace feature budgets Davidson vectors, batch states and carryover
    copies. These are allocation priors, not separately measured stage peaks.
    Coefficient floors preserve these
    allocation priors when small-domain regression cannot identify a slope.
    """
    da, db = (0, 0) if dims is None else dims
    occupied = nelec[0]
    return (
        1.0,
        64 * math.prod(math.comb(norb, n) for n in nelec) / 1e6,
        1500 * norb**4 / 1e6,
        512 * da * db / 1e6,
        16 * (da + db) * occupied**2 * (norb - occupied) ** 2 / 1e6,
    )


@dataclass(frozen=True)
class CalibratedResourceModel:
    """Immutable coefficients for a version-checked serial workload."""

    model_id: str
    coefficients: tuple[tuple[str, tuple[float, ...]], ...]
    max_norb: int
    max_ci_dim: int
    max_sector_dim: int

    def upper_bound_mb(
        self,
        norb: int,
        nelec: tuple[int, int],
        *,
        stage: Stage,
        subspace_dims: tuple[int, int] | None,
        retained_mb: float,
        solver_method: str | None,
    ) -> float:
        """Evaluate the stage model, rejecting extrapolation outside its domain."""
        if (
            not 2 <= norb <= self.max_norb
            or math.prod(math.comb(norb, n) for n in nelec) > self.max_ci_dim
        ):
            raise ResourceModelDomainError(
                f"model={self.model_id}: norb={norb}, nelec={nelec} outside calibration"
            )
        if stage == "reference":
            if solver_method not in ("exact_casci", "selected_ci_pyscf"):
                raise ResourceModelDomainError(f"Uncalibrated solver_method={solver_method}")
            subspace_dims = addon_subspace_bound(norb, nelec)
        if subspace_dims and max(subspace_dims) > self.max_sector_dim:
            raise ResourceModelDomainError(f"Uncalibrated subspace_dims={subspace_dims}")
        coefficients = dict(self.coefficients)[stage]
        return retained_mb + sum(
            c * x
            for c, x in zip(
                coefficients, workspace_features(norb, nelec, subspace_dims), strict=True
            )
        )


def load_resource_model(
    path: str | Path | None = None,
    *,
    versions: Mapping[str, str] | None = None,
    profile: Mapping[str, object] | None = None,
) -> CalibratedResourceModel:
    """Load a validated model and reject incompatible dependencies or workload.

    Args:
        path: Calibration artifact; defaults to the shipped coefficient file.
        versions: Exact explicit manifest, or installed distribution versions. Only
            absent optional Catalyst is accepted when inspecting the installation;
            an installed Catalyst version must still match the calibration.
        profile: Workload overrides to check against the artifact's bounded profile.

    Returns:
        An immutable model. Executors must honor the recorded live-array schedule,
        solver settings and profile; arbitrary geometry/AO-basis work is not covered.

    Raises:
        ResourceModelDomainError: Missing artifact, unvalidated model or runtime mismatch.
    """
    path = Path(path) if path is not None else Path(__file__).with_name("resource_calibration.json")
    try:
        data = json.loads(path.read_text())
        if data["schema"] != "sqd.rss.v1" or data["status"] != "validated_holdout":
            raise ValueError("unvalidated calibration")
        expected = dict(data["versions"])
        actual = dict(versions) if versions is not None else {}
        if versions is None:
            for key in tuple(expected):
                try:
                    actual[key] = version(key)
                except PackageNotFoundError:
                    if key != "pennylane-catalyst":
                        raise
                    # SQD never calls Catalyst. Keep the calibrated conservative
                    # import allowance, without inventing an installed version.
                    del expected[key]
        if actual != expected:
            raise ValueError(f"dependency versions differ: expected={expected}, actual={actual}")
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ValueError("calibrated platform is Linux x86_64")
        for key, value in (profile or {}).items():
            limit = data["profile"][key]
            if type(value) is not type(limit) or (
                value > limit if isinstance(limit, int) else value != limit
            ):
                raise ValueError(f"uncalibrated workload {key}={value}; limit={limit}")
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"{key} must be positive")
        rows = []
        for stage in STAGES:
            coefficients = tuple(data["coefficients"][stage])
            if len(coefficients) != 5 or any(not math.isfinite(c) or c < 0 for c in coefficients):
                raise ValueError("invalid coefficients")
            rows.append((stage, coefficients))
        return CalibratedResourceModel(data["model_id"], tuple(rows), **data["domain"])
    except (OSError, ValueError, KeyError, TypeError, PackageNotFoundError) as exc:
        raise ResourceModelDomainError(f"Cannot load RSS model: {exc}") from exc
