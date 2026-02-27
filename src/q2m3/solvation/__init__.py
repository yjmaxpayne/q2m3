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

from q2m3.solvation.config import MoleculeConfig, QPEConfig, SolvationConfig
from q2m3.solvation.solvent import SPC_E_WATER, TIP3P_WATER

__all__ = [
    "MoleculeConfig",
    "QPEConfig",
    "SolvationConfig",
    "TIP3P_WATER",
    "SPC_E_WATER",
]
