# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Monte Carlo loop for QPE-driven solvation simulations.

Implements a pure-Python Metropolis MC loop (ADR-003: no @qjit) that
delegates energy evaluation to pre-compiled step callbacks.

Three Hamiltonian modes determine acceptance energy:
    - hf_corrected: E_HF_ref + E_MM(sol-sol)  (fast, approximate)
    - fixed / dynamic: E_QPE + E_MM(sol-sol)   (full quantum)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .config import SolvationConfig
from .energy import StepResult


@dataclass
class MCResult:
    """Complete MC simulation result."""

    initial_energy: float
    final_energy: float
    best_energy: float  # Best acceptance energy
    best_qpe_energy: float  # Best E_QPE component
    acceptance_rate: float
    avg_energy: float  # Mean acceptance energy across trajectory
    quantum_energies: np.ndarray  # E_QPE per step (NaN for non-QPE steps)
    hf_energies: np.ndarray  # E_HF reference per step
    callback_times: np.ndarray  # PySCF time per step
    quantum_times: np.ndarray  # QPE time per step (0.0 for non-QPE)
    final_solvent_states: np.ndarray
    best_solvent_states: np.ndarray
    best_qpe_solvent_states: np.ndarray  # Config at lowest QPE energy
    trajectory_solvent_states: np.ndarray  # Current solvent configuration after each MC step
    n_quantum_evaluations: int  # Actual QPE call count
    n_accepted: int


def _propose_move(
    state: np.ndarray,
    rng: np.random.Generator,
    translation_step: float,
    rotation_step: float,
) -> np.ndarray:
    """Propose random translation + rotation for a solvent molecule.

    Args:
        state: Current molecule state [x, y, z, roll, pitch, yaw].
        rng: NumPy random generator.
        translation_step: Max displacement per axis in Angstrom.
        rotation_step: Max rotation per axis in radians.

    Returns:
        New state array (does not mutate input).
    """
    new_state = state.copy()
    new_state[:3] += rng.uniform(-translation_step, translation_step, 3)
    new_state[3:] += rng.uniform(-rotation_step, rotation_step, 3)
    return new_state


def create_mc_loop(
    config: SolvationConfig,
    step_callback: Callable[[np.ndarray, np.ndarray], StepResult],
) -> Callable[..., MCResult]:
    """
    Factory: create QPE-driven MC loop (pure Python).

    Returns a closure that runs the MC simulation with Metropolis acceptance.

    Args:
        config: Solvation simulation configuration.
        step_callback: Energy evaluator for each MC trial.
            Signature: (solvent_states, qm_coords) -> StepResult

    Returns:
        mc_loop(initial_solvents, qm_coords, seed, initial_energy) -> MCResult
    """
    n_waters = config.n_waters
    n_mc_steps = config.n_mc_steps
    translation_step = config.translation_step
    rotation_step = config.rotation_step
    kt = config.kt
    is_hf_corrected = config.hamiltonian_mode == "hf_corrected"

    def mc_loop(
        initial_solvents: np.ndarray,
        qm_coords: np.ndarray,
        seed: int,
        initial_energy: float,
    ) -> MCResult:
        """Run MC solvation loop.

        Args:
            initial_solvents: Solvent states, shape (n_waters, 6).
            qm_coords: QM atom coordinates (flat or 2D).
            seed: Random seed for reproducibility.
            initial_energy: Pre-computed initial energy.

        Returns:
            MCResult with full trajectory data.
        """
        rng = np.random.default_rng(seed)
        solvents = initial_solvents.copy()
        current_energy = float(initial_energy)

        # Trajectory arrays
        quantum_energies = np.full(n_mc_steps, np.nan)
        hf_energies = np.zeros(n_mc_steps)
        callback_times = np.zeros(n_mc_steps)
        quantum_times = np.zeros(n_mc_steps)
        trajectory_solvent_states = np.zeros((n_mc_steps, n_waters, 6), dtype=solvents.dtype)

        # Best tracking
        best_energy = current_energy
        best_solvents = solvents.copy()
        best_qpe_energy = float("inf")
        best_qpe_solvents = solvents.copy()

        n_accepted = 0
        energy_sum = 0.0

        for step in range(n_mc_steps):
            # 1. Select random solvent molecule
            mol_idx = rng.integers(0, n_waters)

            # 2. Propose move
            new_solvents = solvents.copy()
            new_solvents[mol_idx] = _propose_move(
                solvents[mol_idx], rng, translation_step, rotation_step
            )

            # 3. Compute energy via step callback
            result = step_callback(new_solvents, qm_coords)

            # 4. Acceptance energy depends on mode
            if is_hf_corrected:
                new_energy = result.e_hf_ref + result.e_mm_sol_sol
            else:
                new_energy = result.e_qpe + result.e_mm_sol_sol

            # 5. Metropolis criterion
            delta_e = new_energy - current_energy
            if delta_e <= 0 or rng.random() < np.exp(-delta_e / kt):
                solvents = new_solvents
                current_energy = new_energy
                n_accepted += 1

            # 6. Best QPE tracking (only when QPE actually ran)
            if not np.isnan(result.e_qpe) and result.e_qpe < best_qpe_energy:
                best_qpe_energy = result.e_qpe
                best_qpe_solvents = solvents.copy()

            # 7. Best total energy tracking
            if current_energy < best_energy:
                best_energy = current_energy
                best_solvents = solvents.copy()

            # 8. Record trajectory
            quantum_energies[step] = result.e_qpe
            hf_energies[step] = result.e_hf_ref
            callback_times[step] = result.callback_time
            quantum_times[step] = result.qpe_time
            trajectory_solvent_states[step] = solvents
            energy_sum += current_energy

        n_quantum_evaluations = int(np.sum(~np.isnan(quantum_energies)))

        return MCResult(
            initial_energy=float(initial_energy),
            final_energy=current_energy,
            best_energy=best_energy,
            best_qpe_energy=best_qpe_energy,
            acceptance_rate=n_accepted / n_mc_steps,
            avg_energy=energy_sum / n_mc_steps,
            quantum_energies=quantum_energies,
            hf_energies=hf_energies,
            callback_times=callback_times,
            quantum_times=quantum_times,
            final_solvent_states=solvents,
            best_solvent_states=best_solvents,
            best_qpe_solvent_states=best_qpe_solvents,
            trajectory_solvent_states=trajectory_solvent_states,
            n_quantum_evaluations=n_quantum_evaluations,
            n_accepted=n_accepted,
        )

    return mc_loop
