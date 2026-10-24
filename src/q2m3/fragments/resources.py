"""QPE and SQD resource estimates of one subsystem, from its active-space integrals.

Both estimates use the integrals that the SQD stage solves, so the estimated and the
solved Hamiltonians have the same active space.
"""

from __future__ import annotations

import dataclasses
import math
from importlib.metadata import version

from q2m3.core.resource_estimation import estimate_qpe_resources_from_integrals
from q2m3.fragments.integrals import ActiveSpaceIntegrals
from q2m3.fragments.result import QPE_ESTIMATOR, SQD_ESTIMATOR, ResourceEstimate
from q2m3.sqd.config import LUCJConfig
from q2m3.sqd.resources import estimate_rss_mb, load_resource_model

VERSIONED_PACKAGES = ("q2m3", "pennylane")
QPE_SCOPE = (
    "Double-factorization cost estimate of the active-space Hamiltonian; "
    "not a QPEEngine circuit result."
)
SQD_SCOPE = (
    "SQD workload of the active space and the calibrated RSS model prediction for the "
    "prepare and sample stages, in decimal MB."
)


def qpe_estimate(
    integrals: ActiveSpaceIntegrals, *, basis: str, target_error: float = 0.0016
) -> ResourceEstimate:
    """Return the double-factorization QPE cost of the subsystem Hamiltonian.

    Args:
        integrals: Active-space integrals of the subsystem.
        basis: Basis set label recorded in the estimate.
        target_error: Target energy error in Hartree.
    """
    resources = estimate_qpe_resources_from_integrals(
        integrals.h1, integrals.h2, basis=basis, target_error=target_error
    )
    return ResourceEstimate(QPE_ESTIMATOR, _versions(), dataclasses.asdict(resources), QPE_SCOPE)


def sqd_estimate(
    integrals: ActiveSpaceIntegrals, *, lucj: LUCJConfig | None = None
) -> ResourceEstimate:
    """Return the SQD workload and predicted stage memory of the subsystem.

    Args:
        integrals: Active-space integrals of the subsystem.
        lucj: Ansatz and sampling configuration of the SQD stage.

    Raises:
        ResourceModelDomainError: The workload or the environment is outside the
            calibration of the RSS model.
    """
    lucj = lucj or LUCJConfig()
    norb, nelec = integrals.record.ncas, integrals.nelec
    model = load_resource_model(profile={"n_reps": lucj.n_reps, "shots": lucj.shots})
    values = {
        "model_id": model.model_id,
        "qubits": 2 * norb,
        "shots": lucj.shots,
        "n_reps": lucj.n_reps,
        "full_ci_dim": math.comb(norb, nelec[0]) * math.comb(norb, nelec[1]),
        "rss_mb": {
            stage: estimate_rss_mb(norb, nelec, stage=stage, model=model)
            for stage in ("prepare", "sample")
        },
    }
    return ResourceEstimate(SQD_ESTIMATOR, _versions(), values, SQD_SCOPE)


def _versions() -> dict[str, str]:
    return {package: version(package) for package in VERSIONED_PACKAGES}
