# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""MC Solvation module for q2m3.

Analysis functions (pure NumPy) are always available.
MC simulation requires Catalyst and JAX and is loaded on first access.
Check membership in __all__ to discover available optional exports.
"""

from typing import Any as _Any

from q2m3._lazy import available_exports as _available_exports
from q2m3._lazy import lazy_getattr as _lazy_getattr

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
]

_LAZY_EXPORTS = {
    "run_solvation": ("q2m3.solvation.orchestrator", "catalyst"),
    "replay_quantum_trajectory": ("q2m3.solvation.orchestrator", "catalyst"),
    "MoleculeConfig": ("q2m3.solvation.config", "catalyst"),
    "QPEConfig": ("q2m3.solvation.config", "catalyst"),
    "SolvationConfig": ("q2m3.solvation.config", "catalyst"),
    "SolventModel": ("q2m3.solvation.solvent", "catalyst"),
    "TIP3P_WATER": ("q2m3.solvation.solvent", "catalyst"),
    "SPC_E_WATER": ("q2m3.solvation.solvent", "catalyst"),
}
__all__ += _available_exports(_LAZY_EXPORTS)


def __getattr__(name: str) -> _Any:
    return _lazy_getattr(__name__, globals(), _LAZY_EXPORTS, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
