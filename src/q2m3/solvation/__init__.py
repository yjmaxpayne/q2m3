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

from q2m3.solvation.circuit_builder import (
    MAX_TROTTER_STEPS_RUNTIME,
    QPECircuitBundle,
    build_qpe_circuit,
)
from q2m3.solvation.config import MoleculeConfig, QPEConfig, SolvationConfig
from q2m3.solvation.energy import (
    StepResult,
    compute_hf_energy_solvated,
    compute_hf_energy_vacuum,
    compute_mulliken_charges,
    create_coefficient_callback,
    create_hf_corrected_step_callback,
    create_step_callback,
    precompute_vacuum_cache,
)
from q2m3.solvation.mc_loop import MCResult, create_mc_loop
from q2m3.solvation.phase_extraction import extract_energy_from_probs, extract_energy_from_shots
from q2m3.solvation.solvent import SPC_E_WATER, TIP3P_WATER

__all__ = [
    "MoleculeConfig",
    "QPEConfig",
    "SolvationConfig",
    "TIP3P_WATER",
    "SPC_E_WATER",
    "extract_energy_from_probs",
    "extract_energy_from_shots",
    "QPECircuitBundle",
    "build_qpe_circuit",
    "MAX_TROTTER_STEPS_RUNTIME",
    "StepResult",
    "precompute_vacuum_cache",
    "create_coefficient_callback",
    "create_step_callback",
    "create_hf_corrected_step_callback",
    "compute_hf_energy_vacuum",
    "compute_hf_energy_solvated",
    "compute_mulliken_charges",
    "MCResult",
    "create_mc_loop",
]
