"""
MC Solvation module for q2m3.

Requires Catalyst and JAX. Install with:
    uv sync --extra solvation
"""

try:
    import catalyst  # noqa: F401
    import jax  # noqa: F401
except ImportError as e:
    raise ImportError(
        "q2m3.solvation requires Catalyst and JAX. " "Install with: uv sync --extra solvation"
    ) from e

# Public API will be populated as modules are implemented
__all__: list[str] = []
