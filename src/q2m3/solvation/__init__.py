"""MC Solvation module for q2m3.

Analysis functions (pure NumPy) are always available.
MC simulation requires Catalyst and JAX.
"""

# --- Pure NumPy analysis (always available, no Catalyst dependency) ---
from q2m3.solvation.analysis import (
    DeltaCorrPolResult,
    EnergyPhaseResult,
    EquilibrationResult,
    ModeComparisonResult,
    QPEHFConsistencyResult,
    analyze_energy_phases,
    compute_delta_corr_pol,
    compute_qpe_hf_consistency,
    detect_equilibration,
    run_mode_comparison,
)

# --- Catalyst-dependent MC simulation (graceful degradation) ---
_HAS_CATALYST = True
try:
    import catalyst  # noqa: F401
    import jax  # noqa: F401
except ImportError:
    _HAS_CATALYST = False

if _HAS_CATALYST:
    from q2m3.solvation.config import MoleculeConfig, QPEConfig, SolvationConfig
    from q2m3.solvation.orchestrator import run_solvation
    from q2m3.solvation.solvent import SPC_E_WATER, TIP3P_WATER, SolventModel

__all__ = [
    # Pure NumPy analysis (no Catalyst required)
    "DeltaCorrPolResult",
    "EnergyPhaseResult",
    "EquilibrationResult",
    "ModeComparisonResult",
    "QPEHFConsistencyResult",
    "analyze_energy_phases",
    "compute_delta_corr_pol",
    "compute_qpe_hf_consistency",
    "detect_equilibration",
    "run_mode_comparison",
    # Catalyst-dependent MC simulation (only when Catalyst installed)
    "run_solvation",
    "MoleculeConfig",
    "QPEConfig",
    "SolvationConfig",
    "SolventModel",
    "TIP3P_WATER",
    "SPC_E_WATER",
]
