"""Solvation analysis functions for δ_corr-pol and related statistics.

Pure NumPy module — no Catalyst or JAX dependency.
All public functions are pure functions: take NumPy arrays, return frozen dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeltaCorrPolResult:
    """δ_corr-pol analysis result.

    Note: frozen=True prevents field reassignment but does NOT prevent mutation
    of mutable fields like np.ndarray (e.g., obj.per_step_delta[0] = 999 is still
    possible). This is an accepted trade-off: frozen provides intent-level immutability
    and hashability for simple fields, while ndarray copy-on-read is not enforced
    to avoid performance overhead on large arrays. Consumers should treat all fields
    as read-only.
    """

    mean_ha: float  # mean δ (Hartree)
    std_ha: float  # standard deviation (Hartree)
    sem_ha: float  # standard error of the mean (Hartree)
    t_statistic: float  # t-statistic
    n_samples: int  # number of paired samples
    is_significant: bool  # |t| >= 2.0
    per_step_delta: np.ndarray  # per-step differences (Ha) — mutable, treat as read-only


@dataclass(frozen=True)
class EnergyPhaseResult:
    """Energy trend phase analysis result (early vs late phase)."""

    early_mean: float
    early_std: float
    late_mean: float
    late_std: float
    n_per_phase: int


@dataclass(frozen=True)
class QPEHFConsistencyResult:
    """QPE-HF trajectory consistency metrics."""

    mean_offset_mha: float  # mean(QPE - HF) in mHa
    pearson_correlation: float
    early_offset_mha: float
    late_offset_mha: float
    offset_drift_mha: float  # late_offset - early_offset
    n_samples: int


@dataclass(frozen=True)
class EquilibrationResult:
    """Equilibration diagnostic result (windowed monotonicity)."""

    frac_decreasing: float  # fraction of adjacent window pairs with decreasing mean
    n_windows: int
    qpe_drift_mha: float  # energy drift from first to last window (mHa)
    is_monotonic: bool  # frac_decreasing > 0.8


@dataclass(frozen=True)
class ModeComparisonResult:
    """Full three-mode comparison analysis result."""

    delta_corr_pol: DeltaCorrPolResult
    energy_phases: dict[str, EnergyPhaseResult]
    qpe_hf_consistency: QPEHFConsistencyResult | None
    equilibration: EquilibrationResult | None
    energy_distributions: dict[str, dict]
    trotter_bias_mha: float


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_KEYS: frozenset[str] = frozenset({"quantum_energies", "hf_energies"})


def _validate_result_dict(result: dict, label: str) -> None:
    """Validate that result dict contains required keys from run_solvation()."""
    missing = _REQUIRED_KEYS - result.keys()
    if missing:
        raise ValueError(
            f"{label} missing required keys: {missing}. "
            f"Expected keys from run_solvation() return dict."
        )


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------


def compute_delta_corr_pol(
    quantum_energies_fixed: np.ndarray,
    quantum_energies_dynamic: np.ndarray,
) -> DeltaCorrPolResult:
    """Compute per-step δ_corr-pol from paired fixed/dynamic QPE energies.

    δ_corr-pol(R) = E_QPE(dynamic, R) - E_QPE(fixed, R)
                  ≈ E_corr(solvated, R) - E_corr(vacuum)

    Args:
        quantum_energies_fixed: QPE energies from fixed Hamiltonian mode.
        quantum_energies_dynamic: QPE energies from dynamic Hamiltonian mode.
            Must be the same length as quantum_energies_fixed.

    Returns:
        DeltaCorrPolResult with paired-sample statistics.
    """
    delta = quantum_energies_dynamic - quantum_energies_fixed
    n = len(delta)
    mean_ha = float(np.mean(delta))
    std_ha = float(np.std(delta, ddof=1) if n > 1 else 0.0)
    sem_ha = float(std_ha / np.sqrt(n) if n > 0 else 0.0)
    t_stat = float(mean_ha / sem_ha if sem_ha > 0 else 0.0)

    return DeltaCorrPolResult(
        mean_ha=mean_ha,
        std_ha=std_ha,
        sem_ha=sem_ha,
        t_statistic=t_stat,
        n_samples=n,
        is_significant=abs(t_stat) >= 2.0,
        per_step_delta=delta,
    )


def analyze_energy_phases(
    quantum_energies: np.ndarray,
    n_phases: int = 3,
) -> EnergyPhaseResult:
    """Analyze early/late energy trends across MC sampling phases.

    Splits the energy trajectory into n_phases equal segments and compares
    the first (early) and last (late) segments.

    Args:
        quantum_energies: Energy trajectory from QPE-driven MC simulation.
        n_phases: Number of phases to divide the trajectory into.

    Returns:
        EnergyPhaseResult with mean and std for early and late phases.
    """
    n = len(quantum_energies)
    n_per_phase = max(1, n // n_phases)
    early = quantum_energies[:n_per_phase]
    late = quantum_energies[-n_per_phase:]

    return EnergyPhaseResult(
        early_mean=float(np.mean(early)),
        early_std=float(np.std(early)),
        late_mean=float(np.mean(late)),
        late_std=float(np.std(late)),
        n_per_phase=n_per_phase,
    )


def compute_qpe_hf_consistency(
    quantum_energies: np.ndarray,
    hf_energies: np.ndarray,
) -> QPEHFConsistencyResult:
    """Compute QPE-HF trajectory consistency metrics.

    Measures how well the QPE and HF energy trajectories track each other.
    A positive, high Pearson correlation indicates consistent trends.

    Args:
        quantum_energies: QPE energy trajectory.
        hf_energies: HF energy trajectory (same length as quantum_energies).

    Returns:
        QPEHFConsistencyResult with offset and correlation statistics.
    """
    n = len(quantum_energies)
    diff = quantum_energies - hf_energies
    mean_offset_mha = float(np.mean(diff) * 1000)

    if np.std(quantum_energies) > 0 and np.std(hf_energies) > 0:
        pearson = float(np.corrcoef(quantum_energies, hf_energies)[0, 1])
    else:
        pearson = 0.0

    n_third = max(1, n // 3)
    early_diff = diff[:n_third]
    late_diff = diff[-n_third:]
    early_offset_mha = float(np.mean(early_diff) * 1000)
    late_offset_mha = float(np.mean(late_diff) * 1000)
    offset_drift_mha = late_offset_mha - early_offset_mha

    return QPEHFConsistencyResult(
        mean_offset_mha=mean_offset_mha,
        pearson_correlation=pearson,
        early_offset_mha=early_offset_mha,
        late_offset_mha=late_offset_mha,
        offset_drift_mha=offset_drift_mha,
        n_samples=n,
    )


def detect_equilibration(
    quantum_energies: np.ndarray,
    min_samples: int = 10,
) -> EquilibrationResult | None:
    """Windowed monotonicity diagnostic for MC equilibration.

    Splits energies into 5 equal windows and checks if window means
    are monotonically decreasing (suggesting non-equilibration).

    Args:
        quantum_energies: QPE energy trajectory.
        min_samples: Minimum number of samples required. Returns None if
            len(quantum_energies) < min_samples.

    Returns:
        EquilibrationResult, or None if insufficient samples.
    """
    n = len(quantum_energies)
    if n < min_samples:
        return None

    window = max(1, n // 5)
    n_windows = n // window
    window_means = [
        float(np.mean(quantum_energies[i * window : (i + 1) * window])) for i in range(n_windows)
    ]

    n_decreasing = sum(
        1 for i in range(1, len(window_means)) if window_means[i] < window_means[i - 1]
    )
    frac_decreasing = n_decreasing / max(1, len(window_means) - 1)

    qpe_drift_mha = float((window_means[-1] - window_means[0]) * 1000)

    return EquilibrationResult(
        frac_decreasing=frac_decreasing,
        n_windows=n_windows,
        qpe_drift_mha=qpe_drift_mha,
        is_monotonic=frac_decreasing > 0.8,
    )


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------


def run_mode_comparison(
    result_fixed: dict,
    result_dynamic: dict,
    result_hf_corrected: dict | None = None,
    e_vacuum: float = 0.0,
) -> ModeComparisonResult:
    """Orchestrate full three-mode comparison analysis.

    Combines lower-level analysis functions to produce a complete
    ModeComparisonResult from run_solvation() return dicts.

    Args:
        result_fixed: run_solvation() result dict for fixed Hamiltonian mode.
        result_dynamic: run_solvation() result dict for dynamic Hamiltonian mode.
        result_hf_corrected: Optional result dict for hf_corrected mode.
        e_vacuum: Vacuum reference energy (Ha) for Trotter bias calculation.

    Returns:
        ModeComparisonResult with all analysis fields populated.

    Raises:
        ValueError: If any result dict is missing required keys.
    """
    # Step 0: validate inputs
    _validate_result_dict(result_fixed, "result_fixed")
    _validate_result_dict(result_dynamic, "result_dynamic")
    if result_hf_corrected is not None:
        _validate_result_dict(result_hf_corrected, "result_hf_corrected")

    # Step 1: extract and filter NaN values
    q_fixed = np.array(result_fixed["quantum_energies"])
    q_fixed = q_fixed[~np.isnan(q_fixed)]

    q_dyn = np.array(result_dynamic["quantum_energies"])
    q_dyn = q_dyn[~np.isnan(q_dyn)]

    hf_dyn = np.array(result_dynamic["hf_energies"])
    hf_dyn = hf_dyn[~np.isnan(hf_dyn)]

    # Step 2b: align fixed/dynamic arrays to same length
    n_eval = min(len(q_fixed), len(q_dyn))
    q_fixed = q_fixed[:n_eval]
    q_dyn = q_dyn[:n_eval]

    # Step 3: compute δ_corr-pol
    delta_result = compute_delta_corr_pol(q_fixed, q_dyn)

    # Step 4: energy phase analysis for each mode
    energy_phases: dict[str, EnergyPhaseResult] = {
        "fixed": analyze_energy_phases(q_fixed),
        "dynamic": analyze_energy_phases(q_dyn),
    }

    # Step 5: QPE-HF consistency and equilibration for dynamic mode
    n_common = min(len(q_dyn), len(hf_dyn))
    qpe_hf = compute_qpe_hf_consistency(q_dyn[:n_common], hf_dyn[:n_common])
    equilibration = detect_equilibration(q_dyn)

    # Step 6: Trotter bias (mHa) from fixed mode
    trotter_bias_mha = float((np.mean(q_fixed) - e_vacuum) * 1000)

    # Step 6b: energy distributions for each available mode
    energy_distributions: dict[str, dict] = {}
    modes_to_summarize: list[tuple[str, np.ndarray]] = [
        ("fixed", q_fixed),
        ("dynamic", q_dyn),
    ]
    if result_hf_corrected is not None:
        q_hf = np.array(result_hf_corrected["quantum_energies"])
        q_hf = q_hf[~np.isnan(q_hf)]
        modes_to_summarize.append(("hf_corrected", q_hf))

    for mode_name, q_arr in modes_to_summarize:
        energy_distributions[mode_name] = {
            "mean": float(np.mean(q_arr)),
            "std": float(np.std(q_arr, ddof=1) if len(q_arr) > 1 else 0.0),
            "min": float(np.min(q_arr)),
            "max": float(np.max(q_arr)),
            "n_evals": len(q_arr),
        }

    # Step 7: build result
    return ModeComparisonResult(
        delta_corr_pol=delta_result,
        energy_phases=energy_phases,
        qpe_hf_consistency=qpe_hf,
        equilibration=equilibration,
        energy_distributions=energy_distributions,
        trotter_bias_mha=trotter_bias_mha,
    )
