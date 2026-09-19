"""SQD contracts and lazily loaded workflows.

Data contracts are independent of optional sampling backends. Workflow exports
are listed in __all__ when the SQD extra is available and load on first access.
"""

from typing import Any as _Any

from q2m3._lazy import available_exports as _available_exports
from q2m3._lazy import lazy_getattr as _lazy_getattr
from q2m3.sqd.config import CCSDSeed, IntegralContext, LUCJConfig, ReferenceConfig, SQDConfig
from q2m3.sqd.exceptions import (
    BaselineUnavailableError,
    CCSDConvergenceError,
    ComparisonUnavailableError,
    ProvenanceMismatchError,
    ReferenceNumericalError,
    ReferenceTimeoutError,
    ReferenceUnavailableError,
    ResourceLimitError,
    ResourceModelDomainError,
    SamplingIntegrityError,
)
from q2m3.sqd.result import ReferenceAttempt, ReferenceResult, SQDResult, T1Residual, T2Diagnostics

__all__ = [
    "CCSDSeed",
    "IntegralContext",
    "LUCJConfig",
    "ReferenceConfig",
    "SQDConfig",
    "ReferenceAttempt",
    "ReferenceResult",
    "SQDResult",
    "T1Residual",
    "T2Diagnostics",
    "BaselineUnavailableError",
    "CCSDConvergenceError",
    "ComparisonUnavailableError",
    "ProvenanceMismatchError",
    "ReferenceNumericalError",
    "ReferenceTimeoutError",
    "ReferenceUnavailableError",
    "ResourceLimitError",
    "ResourceModelDomainError",
    "SamplingIntegrityError",
]

_LAZY_EXPORTS = {
    "run_sqd": ("q2m3.sqd.orchestrator", "sqd"),
    "run_sqd_from_integrals": ("q2m3.sqd.orchestrator", "sqd"),
}
__all__ += _available_exports(_LAZY_EXPORTS)


def __getattr__(name: str) -> _Any:
    return _lazy_getattr(__name__, globals(), _LAZY_EXPORTS, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
