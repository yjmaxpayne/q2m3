# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
QJIT-Compiled Monte Carlo Solvation Loop

This module provides a factory function to create @qjit compiled MC loops
for solvation simulations with quantum algorithm validation.

Architecture:
    create_mc_loop() [factory function]
        |
        +-- Captures: compiled_circuit, solver_config, energy_config
        |
        +-- Returns: @qjit mc_solvation_loop()
                |
                +-- catalyst.for_loop (MC iterations)
                |       |
                |       +-- propose_move() (JAX: translation + rotation)
                |       +-- pure_callback(compute_energy) -> PySCF HF + MM
                |       +-- metropolis_accept() (Boltzmann criterion)
                |       +-- catalyst.cond(step % interval == 0):
                |               +-- quantum_circuit() -> QPE/VQE
                |               +-- extract_energy() -> phase analysis
                |
                +-- Return results with quantum energies and timing

Key QJIT Integration:
    - pure_callback: Wraps PySCF HF energy (non-JAX code)
    - for_loop: Replaces Python for-loop for MC iterations
    - cond: Conditional quantum algorithm execution
    - debug.print: Real-time output inside @qjit

Random Number Generation:
    - Uses Linear Congruential Generator (LCG) for @qjit compatibility
    - Parameters from Numerical Recipes: a=1664525, c=1013904223, m=2^32
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from catalyst import cond, debug, for_loop, pure_callback, qjit

from .config import SolvationConfig
from .constants import BOLTZMANN_CONSTANT

# =============================================================================
# MC Step Functions (JAX-compatible)
# =============================================================================


def _propose_translation(
    position: jnp.ndarray,
    rng_state: int,
    step_size: float,
) -> tuple[jnp.ndarray, int]:
    """
    Propose a translation move.

    Uses LCG random number generator for @qjit compatibility.

    Args:
        position: Current position array (3,)
        rng_state: Current RNG state (LCG)
        step_size: Maximum translation per axis

    Returns:
        Tuple of (new_position, new_rng_state)
    """
    a, c, m = 1664525, 1013904223, 2**32

    rng_state = (a * rng_state + c) % m
    tx = ((rng_state / m) * 2 - 1) * step_size
    rng_state = (a * rng_state + c) % m
    ty = ((rng_state / m) * 2 - 1) * step_size
    rng_state = (a * rng_state + c) % m
    tz = ((rng_state / m) * 2 - 1) * step_size

    new_position = position + jnp.array([tx, ty, tz])
    return new_position, rng_state


def _propose_rotation(
    euler_angles: jnp.ndarray,
    rng_state: int,
    step_size: float,
) -> tuple[jnp.ndarray, int]:
    """
    Propose a rotation move.

    Args:
        euler_angles: Current Euler angles (roll, pitch, yaw)
        rng_state: Current RNG state (LCG)
        step_size: Maximum rotation per axis (radians)

    Returns:
        Tuple of (new_angles, new_rng_state)
    """
    a, c, m = 1664525, 1013904223, 2**32

    rng_state = (a * rng_state + c) % m
    d_roll = ((rng_state / m) * 2 - 1) * step_size
    rng_state = (a * rng_state + c) % m
    d_pitch = ((rng_state / m) * 2 - 1) * step_size
    rng_state = (a * rng_state + c) % m
    d_yaw = ((rng_state / m) * 2 - 1) * step_size

    new_angles = euler_angles + jnp.array([d_roll, d_pitch, d_yaw])
    # Wrap to [-π, π]
    new_angles = jnp.mod(new_angles + jnp.pi, 2 * jnp.pi) - jnp.pi

    return new_angles, rng_state


# =============================================================================
# MC Loop Factory
# =============================================================================


def create_mc_loop(
    config: SolvationConfig,
    compiled_circuit: Callable[[], Any],
    extract_energy: Callable[[np.ndarray, float], float],
    compute_energy_impl: Callable[[np.ndarray, np.ndarray], np.ndarray],
    base_time: float,
    e_vacuum: float = 0.0,
    compute_mm_correction_impl: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> Callable:
    """
    Factory function to create @qjit compiled MC solvation loop.

    This function captures the pre-compiled quantum circuit and configuration
    via closure, allowing direct circuit calls inside @qjit without additional
    pure_callback wrapping.

    Energy Calculation (vacuum_correction mode):
        E_QPE(total) = delta_e(QPE) + e_vacuum + delta_e_mm

        Where:
        - delta_e(QPE): Phase-estimated energy from QPE circuit
        - e_vacuum: Pre-computed vacuum HF energy (reference)
        - delta_e_mm: MM correction = E_HF(solvated) - E_HF(vacuum)

    Args:
        config: Solvation simulation configuration
        compiled_circuit: Pre-compiled @qjit quantum circuit callable
        extract_energy: Function to extract energy from circuit samples
        compute_energy_impl: Pure Python function for total energy computation.
            Signature: (qm_coords_flat, solvent_states) -> [energy, elapsed_time]
        base_time: Base evolution time for quantum energy extraction
        e_vacuum: Vacuum HF energy (reference for energy-shifted QPE)
        compute_mm_correction_impl: Function to compute MM correction.
            Signature: (solvent_states, qm_coords_flat) -> delta_e_mm
            If None, delta_e_mm = 0 (no MM correction)

    Returns:
        @qjit compiled MC solvation loop function

    Example:
        config = SolvationConfig(...)
        compiled_qpe, base_time = build_qpe_circuit(...)

        mc_loop = create_mc_loop(
            config,
            compiled_qpe,
            extract_qpe_energy,
            compute_total_energy_impl,
            base_time,
            e_vacuum=vacuum_hf_energy,
            compute_mm_correction_impl=mm_correction_callback,
        )

        result = mc_loop(initial_solvent_states, qm_coords_flat, seed)
    """
    # Extract configuration constants (captured by closure)
    n_solvent = config.n_waters
    n_mc_steps = config.n_mc_steps
    qpe_interval = config.qpe_config.qpe_interval
    n_qpe_evaluations = n_mc_steps // qpe_interval
    translation_step = config.translation_step
    rotation_step = config.rotation_step
    kt = BOLTZMANN_CONSTANT * config.temperature

    @qjit
    def mc_solvation_loop(
        initial_solvent_states,  # NumPy array, shape (n_solvent, 6)
        qm_coords_flat,  # NumPy array, shape (n_atoms * 3,)
        rng_seed,  # Python int or NumPy int
    ):
        """
        Run complete MC sampling with quantum algorithm validation.

        All operations are QJIT-compiled for performance. PySCF energy
        computations use pure_callback, while quantum circuit calls
        are direct (pre-compiled via closure).

        Args:
            initial_solvent_states: Initial solvent configurations, shape (n_mol, 6)
            qm_coords_flat: Flattened QM region coordinates
            rng_seed: Random seed (integer)

        Returns:
            Dictionary with MC results:
                - initial_energy: Starting total energy
                - final_energy: Final total energy
                - best_energy: Lowest energy found
                - best_solvent_states: Configuration at best energy
                - final_solvent_states: Final configuration
                - acceptance_rate: Fraction of accepted moves
                - avg_energy: Average energy over MC steps
                - n_accepted: Total accepted moves
                - quantum_energies: Array of quantum algorithm energies
                - n_quantum_evaluations: Number of quantum evaluations performed
                - hf_times: HF computation times per step
                - quantum_times: Quantum evaluation times
                - best_qpe_energy: Lowest QPE-validated energy
                - best_qpe_solvent_states: Configuration at best QPE energy
        """
        # Define pure_callbacks inside @qjit
        energy_timing_struct = jax.ShapeDtypeStruct((2,), jnp.float64)

        @pure_callback
        def compute_energy_with_timing(qc, ss) -> energy_timing_struct:  # type: ignore
            """Compute total energy with timing: returns [energy, elapsed]."""
            return compute_energy_impl(qc, ss)

        @pure_callback
        def extract_quantum_energy(samples) -> float:
            """Extract delta_e from quantum algorithm samples."""
            return extract_energy(samples, base_time)

        @pure_callback
        def compute_mm_correction(ss, qc) -> float:
            """Compute MM correction: delta_e_mm = E_HF(solvated) - E_HF(vacuum)."""
            if compute_mm_correction_impl is not None:
                return compute_mm_correction_impl(ss, qc)
            return np.float64(0.0)

        @pure_callback
        def get_timestamp() -> float:
            """Get current timestamp for timing measurements."""
            import time

            return np.float64(time.perf_counter())

        # LCG constants (Numerical Recipes)
        a, c, m = 1664525, 1013904223, 2**32

        # Initialize RNG state
        rng = rng_seed

        # Compute initial energy
        init_result = compute_energy_with_timing(qm_coords_flat, initial_solvent_states)
        initial_energy = init_result[0]

        # Arrays to store quantum energies and timing
        quantum_energies = jnp.zeros(n_qpe_evaluations)
        quantum_idx = 0
        hf_times = jnp.zeros(n_mc_steps)
        quantum_times = jnp.zeros(n_qpe_evaluations)

        # Track best QPE-validated energy and configuration
        # Initialize with large value; will be updated on first QPE evaluation
        best_qpe_energy = jnp.float64(1e10)
        best_qpe_solvents = jnp.array(initial_solvent_states)

        # MC step function
        def mc_step(step_idx, state):
            (
                solvents,
                rng,
                energy,
                best_solvents,
                best_energy,
                n_accepted,
                energies_sum,
                quantum_energies,
                quantum_idx,
                hf_times,
                quantum_times,
                best_qpe_energy,
                best_qpe_solvents,
            ) = state

            # Select random solvent molecule
            rng = (a * rng + c) % m
            mol_idx = jnp.int32((rng / m) * n_solvent) % n_solvent

            # Get current state
            old_state = solvents[mol_idx]
            position = old_state[:3]
            angles = old_state[3:]

            # Propose translation
            rng = (a * rng + c) % m
            tx = ((rng / m) * 2 - 1) * translation_step
            rng = (a * rng + c) % m
            ty = ((rng / m) * 2 - 1) * translation_step
            rng = (a * rng + c) % m
            tz = ((rng / m) * 2 - 1) * translation_step
            new_position = position + jnp.array([tx, ty, tz])

            # Propose rotation
            rng = (a * rng + c) % m
            d_roll = ((rng / m) * 2 - 1) * rotation_step
            rng = (a * rng + c) % m
            d_pitch = ((rng / m) * 2 - 1) * rotation_step
            rng = (a * rng + c) % m
            d_yaw = ((rng / m) * 2 - 1) * rotation_step
            new_angles = angles + jnp.array([d_roll, d_pitch, d_yaw])
            new_angles = jnp.mod(new_angles + jnp.pi, 2 * jnp.pi) - jnp.pi

            # Create new state
            new_state = jnp.concatenate([new_position, new_angles])
            new_solvents = solvents.at[mol_idx].set(new_state)

            # Compute new energy with timing
            energy_result = compute_energy_with_timing(qm_coords_flat, new_solvents)
            new_energy = energy_result[0]
            hf_elapsed = energy_result[1]
            hf_times = hf_times.at[step_idx].set(hf_elapsed)

            # Metropolis acceptance criterion
            delta_e = new_energy - energy
            rng = (a * rng + c) % m
            random_val = rng / m
            accept_prob = jnp.exp(-delta_e / kt)
            accept = (delta_e <= 0.0) | (random_val < accept_prob)

            # Update state based on acceptance
            solvents = jnp.where(accept, new_solvents, solvents)
            energy = jnp.where(accept, new_energy, energy)
            n_accepted = n_accepted + jnp.where(accept, 1, 0)

            # Track best configuration
            is_new_best = energy < best_energy
            best_solvents = jnp.where(is_new_best, solvents, best_solvents)
            best_energy = jnp.where(is_new_best, energy, best_energy)

            energies_sum = energies_sum + energy

            # Quantum evaluation at specified intervals
            should_run_quantum = ((step_idx + 1) % qpe_interval) == 0

            @cond(should_run_quantum)
            def run_quantum_conditional():
                # Record start time
                q_start = get_timestamp()

                # Step 1: Call pre-compiled quantum circuit (direct call, not pure_callback)
                samples = compiled_circuit()

                # Step 2: Extract delta_e from QPE samples
                delta_e = extract_quantum_energy(samples)

                # Step 3: Compute vacuum QPE energy: E_QPE(vacuum) = delta_e + e_ref
                e_qpe_vacuum = delta_e + e_vacuum

                # Step 4: Compute MM correction: delta_e_mm = E_HF(solvated) - E_HF(vacuum)
                delta_e_mm = compute_mm_correction(solvents, qm_coords_flat)

                # Step 5: Total QPE energy with vacuum correction
                # E_total = E_QPE(vacuum) + delta_e_mm
                q_energy = e_qpe_vacuum + delta_e_mm

                # Record end time
                q_end = get_timestamp()
                q_elapsed = q_end - q_start

                # Debug output
                debug.print(
                    "  [Quantum] Step {step}: HF={hf:.6f} Ha, QPE={q:.6f} Ha ({t:.1f} ms)",
                    step=step_idx + 1,
                    hf=energy,
                    q=q_energy,
                    t=q_elapsed * 1000.0,
                )
                return jnp.array([q_energy, q_elapsed])

            @run_quantum_conditional.otherwise
            def no_quantum():
                return jnp.array([0.0, 0.0])

            quantum_result = run_quantum_conditional()
            q_energy_evaluated = quantum_result[0]

            # Store quantum results if evaluated
            new_quantum_idx = jnp.where(should_run_quantum, quantum_idx + 1, quantum_idx)
            quantum_energies = jnp.where(
                should_run_quantum,
                quantum_energies.at[quantum_idx].set(quantum_result[0]),
                quantum_energies,
            )
            quantum_times = jnp.where(
                should_run_quantum,
                quantum_times.at[quantum_idx].set(quantum_result[1]),
                quantum_times,
            )

            # Track best QPE-validated energy and its configuration
            # Only update when QPE was actually evaluated AND energy is lower
            is_new_best_qpe = should_run_quantum & (q_energy_evaluated < best_qpe_energy)
            new_best_qpe_energy = jnp.where(is_new_best_qpe, q_energy_evaluated, best_qpe_energy)
            new_best_qpe_solvents = jnp.where(is_new_best_qpe, solvents, best_qpe_solvents)

            return (
                solvents,
                rng,
                energy,
                best_solvents,
                best_energy,
                n_accepted,
                energies_sum,
                quantum_energies,
                new_quantum_idx,
                hf_times,
                quantum_times,
                new_best_qpe_energy,
                new_best_qpe_solvents,
            )

        # Initialize state tuple
        solvents_jnp = jnp.array(initial_solvent_states)
        init_state = (
            solvents_jnp,
            rng,
            initial_energy,
            solvents_jnp.copy(),
            initial_energy,
            0,
            0.0,
            quantum_energies,
            quantum_idx,
            hf_times,
            quantum_times,
            best_qpe_energy,
            best_qpe_solvents,
        )

        # Run MC loop
        final_state = for_loop(0, n_mc_steps, 1)(mc_step)(init_state)

        (
            final_solvents,
            final_rng,
            final_energy,
            best_solvents,
            best_energy,
            n_accepted,
            energies_sum,
            quantum_energies,
            final_quantum_idx,
            hf_times,
            quantum_times,
            final_best_qpe_energy,
            final_best_qpe_solvents,
        ) = final_state

        return {
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "best_energy": best_energy,
            "best_solvent_states": best_solvents,
            "final_solvent_states": final_solvents,
            "acceptance_rate": n_accepted / n_mc_steps,
            "avg_energy": energies_sum / n_mc_steps,
            "n_accepted": n_accepted,
            "quantum_energies": quantum_energies,
            "n_quantum_evaluations": final_quantum_idx,
            "hf_times": hf_times,
            "quantum_times": quantum_times,
            "best_qpe_energy": final_best_qpe_energy,
            "best_qpe_solvent_states": final_best_qpe_solvents,
        }

    return mc_solvation_loop


# =============================================================================
# Simplified MC Loop (No Quantum Validation)
# =============================================================================


def create_classical_mc_loop(
    config: SolvationConfig,
    compute_energy_impl: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Callable:
    """
    Create @qjit compiled MC loop without quantum validation.

    Useful for equilibration runs or pure classical MC sampling.

    Args:
        config: Solvation simulation configuration
        compute_energy_impl: Energy computation callback

    Returns:
        @qjit compiled classical MC loop function
    """
    n_solvent = config.n_waters
    n_mc_steps = config.n_mc_steps
    translation_step = config.translation_step
    rotation_step = config.rotation_step
    kt = BOLTZMANN_CONSTANT * config.temperature

    @qjit
    def classical_mc_loop(
        initial_solvent_states,  # NumPy array, shape (n_solvent, 6)
        qm_coords_flat,  # NumPy array, shape (n_atoms * 3,)
        rng_seed,  # Python int or NumPy int
    ):
        """Classical MC sampling without quantum evaluation."""
        energy_timing_struct = jax.ShapeDtypeStruct((2,), jnp.float64)

        @pure_callback
        def compute_energy_with_timing(qc, ss) -> energy_timing_struct:  # type: ignore
            return compute_energy_impl(qc, ss)

        a, c, m = 1664525, 1013904223, 2**32
        rng = rng_seed

        init_result = compute_energy_with_timing(qm_coords_flat, initial_solvent_states)
        initial_energy = init_result[0]

        hf_times = jnp.zeros(n_mc_steps)

        def mc_step(step_idx, state):
            (
                solvents,
                rng,
                energy,
                best_solvents,
                best_energy,
                n_accepted,
                energies_sum,
                hf_times,
            ) = state

            rng = (a * rng + c) % m
            mol_idx = jnp.int32((rng / m) * n_solvent) % n_solvent

            old_state = solvents[mol_idx]
            position = old_state[:3]
            angles = old_state[3:]

            # Propose move
            rng = (a * rng + c) % m
            tx = ((rng / m) * 2 - 1) * translation_step
            rng = (a * rng + c) % m
            ty = ((rng / m) * 2 - 1) * translation_step
            rng = (a * rng + c) % m
            tz = ((rng / m) * 2 - 1) * translation_step
            new_position = position + jnp.array([tx, ty, tz])

            rng = (a * rng + c) % m
            d_roll = ((rng / m) * 2 - 1) * rotation_step
            rng = (a * rng + c) % m
            d_pitch = ((rng / m) * 2 - 1) * rotation_step
            rng = (a * rng + c) % m
            d_yaw = ((rng / m) * 2 - 1) * rotation_step
            new_angles = angles + jnp.array([d_roll, d_pitch, d_yaw])
            new_angles = jnp.mod(new_angles + jnp.pi, 2 * jnp.pi) - jnp.pi

            new_state = jnp.concatenate([new_position, new_angles])
            new_solvents = solvents.at[mol_idx].set(new_state)

            energy_result = compute_energy_with_timing(qm_coords_flat, new_solvents)
            new_energy = energy_result[0]
            hf_elapsed = energy_result[1]
            hf_times = hf_times.at[step_idx].set(hf_elapsed)

            delta_e = new_energy - energy
            rng = (a * rng + c) % m
            random_val = rng / m
            accept_prob = jnp.exp(-delta_e / kt)
            accept = (delta_e <= 0.0) | (random_val < accept_prob)

            solvents = jnp.where(accept, new_solvents, solvents)
            energy = jnp.where(accept, new_energy, energy)
            n_accepted = n_accepted + jnp.where(accept, 1, 0)

            is_new_best = energy < best_energy
            best_solvents = jnp.where(is_new_best, solvents, best_solvents)
            best_energy = jnp.where(is_new_best, energy, best_energy)

            energies_sum = energies_sum + energy

            return (
                solvents,
                rng,
                energy,
                best_solvents,
                best_energy,
                n_accepted,
                energies_sum,
                hf_times,
            )

        solvents_jnp = jnp.array(initial_solvent_states)
        init_state = (
            solvents_jnp,
            rng,
            initial_energy,
            solvents_jnp.copy(),
            initial_energy,
            0,
            0.0,
            hf_times,
        )

        final_state = for_loop(0, n_mc_steps, 1)(mc_step)(init_state)

        (
            final_solvents,
            _,
            final_energy,
            best_solvents,
            best_energy,
            n_accepted,
            energies_sum,
            hf_times,
        ) = final_state

        return {
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "best_energy": best_energy,
            "best_solvent_states": best_solvents,
            "final_solvent_states": final_solvents,
            "acceptance_rate": n_accepted / n_mc_steps,
            "avg_energy": energies_sum / n_mc_steps,
            "n_accepted": n_accepted,
            "hf_times": hf_times,
        }

    return classical_mc_loop
