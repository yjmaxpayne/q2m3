"""Typed SQD failures shared by optional numerical backends."""


class ResourceLimitError(RuntimeError):
    """A run exceeds its allowed resource budget."""


class ResourceModelDomainError(ResourceLimitError):
    """No calibrated resource bound applies to the requested operation."""


class CCSDConvergenceError(RuntimeError):
    """The ansatz seed did not converge."""


class BaselineUnavailableError(RuntimeError):
    """No permitted reference solver can supply a baseline."""


class ReferenceUnavailableError(RuntimeError):
    """An optional reference solver is unavailable."""


class ReferenceTimeoutError(TimeoutError):
    """A reference solve exhausted its time budget."""


class ReferenceNumericalError(RuntimeError):
    """A reference solve failed its numerical acceptance checks."""


class SamplingIntegrityError(RuntimeError):
    """Samples violate the sampling contract."""


class ComparisonUnavailableError(RuntimeError):
    """A required matched comparison cannot be produced."""


class ProvenanceMismatchError(ValueError):
    """Inputs refer to different orbital frames or Hamiltonians."""
