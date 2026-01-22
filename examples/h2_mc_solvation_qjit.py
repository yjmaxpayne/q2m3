#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
H2 + TIP3P Water: End-to-End QJIT-Compiled MC Solvation with QPE Validation

This example demonstrates Catalyst @qjit end-to-end compilation for hybrid
classical-quantum workflows. The entire workflow is JIT-compiled:
- MC sampling loop with PySCF HF energy evaluation
- QPE quantum circuit evaluation every 10 MC steps

Key QJIT Integration Points:
1. `pure_callback`: Wraps PySCF HF energy computation (non-JAX code)
2. `catalyst.for_loop`: Replaces Python for-loop for MC iterations
3. `catalyst.cond`: Conditional QPE execution every 10 steps
4. `catalyst.debug.print`: Real-time output of QPE results
5. **Pre-compiled QPE QNode**: Circuit compiled ONCE before MC loop, reused 10x

Architecture:
    run_qjit_mc_solvation():
        |
        +-- [Outside @qjit] Build H2 vacuum Hamiltonian (fixed structure)
        +-- [Outside @qjit] @qjit compile QPE QNode (ONCE)
        |
        +-- @qjit mc_solvation_with_qpe_loop()  [via closure]
                |
                +-- catalyst.for_loop (100 MC steps)
                |       |
                |       +-- propose_move() (JAX: translations + rotations)
                |       +-- pure_callback(compute_hf_energy)  -> PySCF HF
                |       +-- pure_callback(compute_mm_energy)  -> TIP3P force field
                |       +-- metropolis_accept() (JAX: Boltzmann criterion)
                |       +-- catalyst.cond(step % 10 == 9):
                |               +-- Call pre-compiled QPE QNode (NO pure_callback!)
                |               +-- catalyst.debug.print() -> Real-time output
                |
                +-- Return results with QPE energies

Key Optimization:
    - QPE circuit is compiled ONCE before MC loop
    - Same compiled circuit is reused for all 10 QPE evaluations
    - Expected speedup: 10-50x on QPE execution time

Reference:
    - Catalyst Callbacks: https://docs.pennylane.ai/projects/catalyst/en/stable/dev/callbacks.html
    - Catalyst cond: https://docs.pennylane.ai/projects/catalyst/en/stable/code/api/catalyst.cond.html
"""

import time
import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from catalyst import cond, debug, for_loop, pure_callback, qjit

warnings.filterwarnings("ignore")

from pyscf import gto, qmmm, scf  # noqa: E402

# =============================================================================
# Constants (Python floats, NOT JAX scalars)
# =============================================================================
HARTREE_TO_KCAL_MOL = 627.5094
ANGSTROM_TO_BOHR = 1.8897259886
BOLTZMANN_CONSTANT = 3.1668114e-6  # Hartree/K

# H2 geometry (fixed solute)
H2_BOND_LENGTH = 0.74  # Angstrom
H2_SYMBOLS = ["H", "H"]
H2_COORDS_LIST = [[0.0, 0.0, 0.0], [0.0, 0.0, H2_BOND_LENGTH]]

# TIP3P water model parameters
TIP3P_OH_BOND_LENGTH = 0.9572  # Angstrom
TIP3P_HOH_ANGLE = 104.52  # degrees
TIP3P_OXYGEN_CHARGE = -0.834  # e
TIP3P_HYDROGEN_CHARGE = 0.417  # e
TIP3P_SIGMA_OO = 3.15061  # Angstrom
TIP3P_EPSILON_OO = 0.152  # kcal/mol
COULOMB_CONSTANT = 332.0637  # kcal/mol * Angstrom / e^2
KCAL_TO_HARTREE = 1.0 / 627.5094

# MC parameters (Python constants)
N_WATERS = 10
N_MC_STEPS = 1000
TEMPERATURE = 300.0  # Kelvin
TRANSLATION_STEP = 0.3  # Angstrom
ROTATION_STEP = 0.2618  # ~15 degrees in radians
KT = BOLTZMANN_CONSTANT * TEMPERATURE

# QPE parameters
QPE_INTERVAL = 10  # Run QPE every 10 MC steps
N_QPE_EVALUATIONS = N_MC_STEPS // QPE_INTERVAL  # Total QPE evaluations
N_ESTIMATION_WIRES = 4
N_TROTTER_STEPS = 10
N_QPE_SHOTS = 50

# Solvation constraints
INITIAL_WATER_DISTANCE = 4.0  # Angstrom


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class SolvationConfig:
    """Configuration for solvation optimization."""

    n_waters: int = N_WATERS
    n_mc_steps: int = N_MC_STEPS
    temperature: float = TEMPERATURE
    translation_step: float = TRANSLATION_STEP
    rotation_step: float = ROTATION_STEP
    random_seed: int = 42


# =============================================================================
# Energy Computation Functions (Called via pure_callback)
# =============================================================================


def _compute_hf_energy_impl(qm_coords_flat, mm_coords_flat, mm_charges):
    """Pure Python function for HF energy computation with MM embedding."""
    qm_coords = np.asarray(qm_coords_flat).reshape(-1, 3)
    mm_coords_flat = np.asarray(mm_coords_flat)
    mm_charges = np.asarray(mm_charges)

    atom_str = "; ".join(
        f"{s} {c[0]} {c[1]} {c[2]}" for s, c in zip(H2_SYMBOLS, qm_coords, strict=True)
    )
    mol = gto.M(atom=atom_str, basis="sto-3g", unit="Angstrom")

    mf = scf.RHF(mol)
    mf.verbose = 0

    if len(mm_charges) > 0:
        mm_coords = mm_coords_flat.reshape(-1, 3)
        mm_coords_bohr = mm_coords * ANGSTROM_TO_BOHR
        mf = qmmm.mm_charge(mf, mm_coords_bohr, mm_charges)

    mf.run()
    return np.float64(mf.e_tot)


def _compute_mm_energy_impl(water_states):
    """Pure Python function for TIP3P MM energy computation."""
    water_states = np.asarray(water_states)
    n_waters = water_states.shape[0]
    if n_waters < 2:
        return np.float64(0.0)

    e_lj_kcal = 0.0
    e_coulomb_kcal = 0.0

    all_coords = []
    for i in range(n_waters):
        water_state = water_states[i]
        position = water_state[:3]
        euler_angles = water_state[3:]

        half_angle_rad = np.radians(TIP3P_HOH_ANGLE / 2)
        h1_local = np.array(
            [
                TIP3P_OH_BOND_LENGTH * np.cos(half_angle_rad),
                TIP3P_OH_BOND_LENGTH * np.sin(half_angle_rad),
                0.0,
            ]
        )
        h2_local = np.array(
            [
                TIP3P_OH_BOND_LENGTH * np.cos(half_angle_rad),
                -TIP3P_OH_BOND_LENGTH * np.sin(half_angle_rad),
                0.0,
            ]
        )

        roll, pitch, yaw = euler_angles
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        Ry = np.array(
            [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
        )
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
        )
        R = Rz @ Ry @ Rx

        o_pos = position
        h1_pos = position + R @ h1_local
        h2_pos = position + R @ h2_local
        all_coords.append(np.array([o_pos, h1_pos, h2_pos]))

    charges = np.array([TIP3P_OXYGEN_CHARGE, TIP3P_HYDROGEN_CHARGE, TIP3P_HYDROGEN_CHARGE])

    for i in range(n_waters):
        for j in range(i + 1, n_waters):
            o_i = all_coords[i][0]
            o_j = all_coords[j][0]
            r_oo = np.linalg.norm(o_j - o_i)

            sigma_r = TIP3P_SIGMA_OO / r_oo
            sigma_r_6 = sigma_r**6
            sigma_r_12 = sigma_r_6**2
            e_lj_kcal += 4.0 * TIP3P_EPSILON_OO * (sigma_r_12 - sigma_r_6)

            for ai in range(3):
                for aj in range(3):
                    r_ij = np.linalg.norm(all_coords[j][aj] - all_coords[i][ai])
                    e_coulomb_kcal += COULOMB_CONSTANT * charges[ai] * charges[aj] / r_ij

    return np.float64((e_lj_kcal + e_coulomb_kcal) * KCAL_TO_HARTREE)


def _get_mm_from_water_states_impl(water_states):
    """Extract MM coordinates and charges from water states."""
    water_states = np.asarray(water_states)
    n_waters = water_states.shape[0]
    all_coords = []
    all_charges = []

    for i in range(n_waters):
        water_state = water_states[i]
        position = water_state[:3]
        euler_angles = water_state[3:]

        half_angle_rad = np.radians(TIP3P_HOH_ANGLE / 2)
        h1_local = np.array(
            [
                TIP3P_OH_BOND_LENGTH * np.cos(half_angle_rad),
                TIP3P_OH_BOND_LENGTH * np.sin(half_angle_rad),
                0.0,
            ]
        )
        h2_local = np.array(
            [
                TIP3P_OH_BOND_LENGTH * np.cos(half_angle_rad),
                -TIP3P_OH_BOND_LENGTH * np.sin(half_angle_rad),
                0.0,
            ]
        )

        roll, pitch, yaw = euler_angles
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        Ry = np.array(
            [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
        )
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
        )
        R = Rz @ Ry @ Rx

        o_pos = position
        h1_pos = position + R @ h1_local
        h2_pos = position + R @ h2_local

        all_coords.extend([o_pos, h1_pos, h2_pos])
        all_charges.extend([TIP3P_OXYGEN_CHARGE, TIP3P_HYDROGEN_CHARGE, TIP3P_HYDROGEN_CHARGE])

    return np.array(all_coords).flatten(), np.array(all_charges)


def _compute_total_energy_impl(qm_coords_flat, water_states):
    """Combined energy callback for @qjit."""
    mm_coords_flat, mm_charges = _get_mm_from_water_states_impl(water_states)
    e_qm = _compute_hf_energy_impl(qm_coords_flat, mm_coords_flat, mm_charges)
    e_mm = _compute_mm_energy_impl(water_states)
    return np.float64(e_qm + e_mm)


def _get_timestamp_impl():
    """Get current timestamp for timing measurements inside @qjit."""
    return np.float64(time.perf_counter())


def _compute_total_energy_with_timing_impl(qm_coords_flat, water_states):
    """Combined energy callback with timing for @qjit."""
    start_time = time.perf_counter()
    mm_coords_flat, mm_charges = _get_mm_from_water_states_impl(water_states)
    e_qm = _compute_hf_energy_impl(qm_coords_flat, mm_coords_flat, mm_charges)
    e_mm = _compute_mm_energy_impl(water_states)
    elapsed = time.perf_counter() - start_time
    return np.array([e_qm + e_mm, elapsed], dtype=np.float64)


# =============================================================================
# Pre-compiled QPE Circuit Builder (Outside @qjit)
# =============================================================================


def build_precompiled_qpe(qm_coords: np.ndarray, hf_energy_estimate: float):
    """
    Build a pre-compiled QPE circuit for H2 vacuum Hamiltonian.

    This function is called ONCE before MC loop to:
    1. Build H2 vacuum Hamiltonian using PennyLane qchem
    2. Create QPE QNode and compile it with @qjit

    The compiled circuit can then be reused for all 10 QPE evaluations
    in the MC loop without recompilation.

    Args:
        qm_coords: QM region coordinates in Angstrom, shape (n_atoms, 3)
        hf_energy_estimate: Estimated HF energy for optimal base_time

    Returns:
        tuple: (compiled_qpe_circuit, base_time, qpe_engine)
            - compiled_qpe_circuit: @qjit compiled QNode
            - base_time: Base time for energy extraction
            - qpe_engine: QPEEngine instance for energy extraction
    """
    from q2m3.core import QPEEngine
    from q2m3.interfaces import PySCFPennyLaneConverter

    # Build H2 vacuum Hamiltonian (fixed structure, no MM embedding)
    # This allows the circuit to be compiled once and reused
    converter = PySCFPennyLaneConverter(basis="sto-3g", mapping="jordan_wigner")
    H, n_qubits, hf_state = converter.pyscf_to_pennylane_hamiltonian(
        symbols=H2_SYMBOLS,
        coords=qm_coords,
        charge=0,
        active_electrons=2,
        active_orbitals=2,
    )

    # Create QPE engine with Catalyst enabled
    qpe_engine = QPEEngine(
        n_qubits=n_qubits,
        n_iterations=8,
        mapping="jordan_wigner",
        use_catalyst=True,  # Enable @qjit compilation
    )

    # Compute optimal base_time from HF energy estimate
    base_time = QPEEngine.compute_optimal_base_time(hf_energy_estimate)

    # Build and compile QPE circuit
    # This returns a @qjit compiled QNode
    compiled_circuit = qpe_engine._build_standard_qpe_circuit(
        H,
        hf_state,
        n_estimation_wires=N_ESTIMATION_WIRES,
        base_time=base_time,
        n_trotter_steps=N_TROTTER_STEPS,
        n_shots=N_QPE_SHOTS,
    )

    return compiled_circuit, base_time, qpe_engine


def _extract_qpe_energy_from_samples(samples, base_time: float) -> float:
    """
    Extract energy from QPE samples (standalone function for use in @qjit).

    Physics derivation:
    - QPE measures eigenphase of U = exp(-iHt)
    - If H|ψ⟩ = E|ψ⟩, then U|ψ⟩ = exp(-iEt)|ψ⟩ = exp(i*2π*φ)|ψ⟩
    - Therefore: -Et = 2πφ (mod 2π), which gives φ = -Et/(2π) mod 1
    - Inverting: E = -2πφ/t

    Args:
        samples: Binary samples from estimation register
        base_time: Base evolution time used in QPE

    Returns:
        Estimated ground state energy in Hartree
    """
    from collections import Counter

    # Handle single sample or multiple shots
    samples = np.asarray(samples, dtype=np.int64)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)

    n_bits = samples.shape[1]
    phase_indices = []
    for sample in samples:
        idx = sum(int(bit) * (2 ** (n_bits - 1 - k)) for k, bit in enumerate(sample))
        phase_indices.append(idx)

    # Find mode (most frequent phase index)
    counter = Counter(phase_indices)
    mode_idx, _ = counter.most_common(1)[0]

    # Convert to energy
    mode_phase = mode_idx / (2**n_bits)
    energy = -2 * np.pi * mode_phase / base_time

    return energy


# =============================================================================
# QJIT-Compiled MC Solvation Loop with Pre-compiled QPE
# =============================================================================


def create_mc_solvation_loop(compiled_qpe_circuit, base_time: float):
    """
    Factory function to create MC solvation loop with pre-compiled QPE.

    This uses closure to capture the pre-compiled QPE circuit, allowing
    it to be called directly inside the @qjit function without pure_callback.

    Args:
        compiled_qpe_circuit: Pre-compiled @qjit QPE QNode
        base_time: Base time for QPE energy extraction

    Returns:
        @qjit compiled MC solvation function
    """

    @qjit
    def mc_solvation_with_qpe_loop(
        initial_water_states,  # NumPy array, shape (n_waters, 6)
        qm_coords_flat,  # NumPy array, shape (6,) for H2
        rng_seed,  # Python int or NumPy int
    ):
        """
        Run the complete MC sampling loop with QPE validation, all QJIT-compiled.

        Every 10 MC steps, a QPE energy estimation is performed using the
        pre-compiled QPE circuit (passed via closure).

        Args:
            initial_water_states: Initial water configurations, shape (n_waters, 6)
            qm_coords_flat: Flattened QM coordinates
            rng_seed: Random seed (integer)

        Returns:
            Dictionary with final results including QPE energies and timing data
        """

        # Define pure_callbacks INSIDE the qjit function
        # Use jax.ShapeDtypeStruct for array return types
        energy_timing_struct = jax.ShapeDtypeStruct((2,), jnp.float64)

        @pure_callback
        def compute_total_energy_with_timing(qc, ws) -> energy_timing_struct:  # type: ignore
            """Compute energy and return [energy, elapsed_time]."""
            return _compute_total_energy_with_timing_impl(qc, ws)

        @pure_callback
        def get_timestamp() -> float:
            """Get current timestamp for QPE timing."""
            return _get_timestamp_impl()

        @pure_callback
        def extract_qpe_energy(samples) -> float:
            """Extract energy from QPE samples (pure callback for Counter)."""
            return _extract_qpe_energy_from_samples(samples, base_time)

        # LCG constants (Numerical Recipes)
        a = 1664525
        c = 1013904223
        m = 2**32

        # Initialize RNG state
        rng = rng_seed

        # Compute initial energy (with timing)
        init_result = compute_total_energy_with_timing(qm_coords_flat, initial_water_states)
        initial_energy = init_result[0]

        # Arrays to store QPE energies and timing data
        qpe_energies = jnp.zeros(N_QPE_EVALUATIONS)
        qpe_idx = 0

        # Timing arrays: HF times for each MC step, QPE times for each evaluation
        hf_times = jnp.zeros(N_MC_STEPS)
        qpe_times = jnp.zeros(N_QPE_EVALUATIONS)

        # MC step function
        def mc_step(step_idx, state):
            (
                waters,
                rng,
                energy,
                best_waters,
                best_energy,
                n_accepted,
                energies_sum,
                qpe_energies,
                qpe_idx,
                hf_times,
                qpe_times,
            ) = state

            # Generate random numbers for this step
            rng = (a * rng + c) % m
            water_idx = jnp.int32((rng / m) * N_WATERS) % N_WATERS

            # Translation
            rng = (a * rng + c) % m
            tx = ((rng / m) * 2 - 1) * TRANSLATION_STEP
            rng = (a * rng + c) % m
            ty = ((rng / m) * 2 - 1) * TRANSLATION_STEP
            rng = (a * rng + c) % m
            tz = ((rng / m) * 2 - 1) * TRANSLATION_STEP

            # Rotation
            rng = (a * rng + c) % m
            r_roll = ((rng / m) * 2 - 1) * ROTATION_STEP
            rng = (a * rng + c) % m
            r_pitch = ((rng / m) * 2 - 1) * ROTATION_STEP
            rng = (a * rng + c) % m
            r_yaw = ((rng / m) * 2 - 1) * ROTATION_STEP

            # Apply move
            old_state = waters[water_idx]
            new_position = old_state[:3] + jnp.array([tx, ty, tz])
            new_angles = old_state[3:] + jnp.array([r_roll, r_pitch, r_yaw])
            new_angles = jnp.mod(new_angles + jnp.pi, 2 * jnp.pi) - jnp.pi
            new_state = jnp.concatenate([new_position, new_angles])
            new_waters = waters.at[water_idx].set(new_state)

            # Compute new energy (with timing)
            energy_result = compute_total_energy_with_timing(qm_coords_flat, new_waters)
            new_energy = energy_result[0]
            hf_elapsed = energy_result[1]
            hf_times = hf_times.at[step_idx].set(hf_elapsed)

            # Metropolis criterion
            delta_e = new_energy - energy
            rng = (a * rng + c) % m
            random_val = rng / m
            accept_prob = jnp.exp(-delta_e / KT)
            accept = (delta_e <= 0.0) | (random_val < accept_prob)

            # Update state
            waters = jnp.where(accept, new_waters, waters)
            energy = jnp.where(accept, new_energy, energy)
            n_accepted = n_accepted + jnp.where(accept, 1, 0)

            # Update best
            is_new_best = energy < best_energy
            best_waters = jnp.where(is_new_best, waters, best_waters)
            best_energy = jnp.where(is_new_best, energy, best_energy)

            energies_sum = energies_sum + energy

            # QPE evaluation every 10 steps (at step 9, 19, 29, ...)
            should_run_qpe = ((step_idx + 1) % QPE_INTERVAL) == 0

            @cond(should_run_qpe)
            def run_qpe_conditional():
                # Record QPE start time
                qpe_start = get_timestamp()

                # Call pre-compiled QPE circuit directly (NO pure_callback!)
                # This is key optimization: circuit is already @qjit compiled
                samples = compiled_qpe_circuit()

                # Extract energy from samples
                qpe_e = extract_qpe_energy(samples)

                # Record QPE end time
                qpe_end = get_timestamp()
                qpe_elapsed = qpe_end - qpe_start

                # Print QPE result with timing
                debug.print(
                    "  [QPE] Step {step}: HF={hf:.6f} Ha, QPE={qpe:.6f} Ha ({elapsed:.1f} ms)",
                    step=step_idx + 1,
                    hf=energy,
                    qpe=qpe_e,
                    elapsed=qpe_elapsed * 1000.0,
                )
                return jnp.array([qpe_e, qpe_elapsed])

            @run_qpe_conditional.otherwise
            def no_qpe():
                return jnp.array([0.0, 0.0])  # Placeholder, won't be stored

            qpe_result = run_qpe_conditional()

            # Store QPE energy and timing if we ran QPE
            new_qpe_idx = jnp.where(should_run_qpe, qpe_idx + 1, qpe_idx)
            qpe_energies = jnp.where(
                should_run_qpe, qpe_energies.at[qpe_idx].set(qpe_result[0]), qpe_energies
            )
            qpe_times = jnp.where(
                should_run_qpe, qpe_times.at[qpe_idx].set(qpe_result[1]), qpe_times
            )

            return (
                waters,
                rng,
                energy,
                best_waters,
                best_energy,
                n_accepted,
                energies_sum,
                qpe_energies,
                new_qpe_idx,
                hf_times,
                qpe_times,
            )

        # Initial state
        waters_jnp = jnp.array(initial_water_states)
        init_state = (
            waters_jnp,
            rng,
            initial_energy,
            waters_jnp.copy(),
            initial_energy,
            0,
            0.0,
            qpe_energies,
            qpe_idx,
            hf_times,
            qpe_times,
        )

        # Run MC loop
        final_state = for_loop(0, N_MC_STEPS, 1)(mc_step)(init_state)

        (
            final_waters,
            final_rng,
            final_energy,
            best_waters,
            best_energy,
            n_accepted,
            energies_sum,
            qpe_energies,
            final_qpe_idx,
            hf_times,
            qpe_times,
        ) = final_state

        return {
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "best_energy": best_energy,
            "best_waters": best_waters,
            "final_waters": final_waters,
            "acceptance_rate": n_accepted / N_MC_STEPS,
            "avg_energy": energies_sum / N_MC_STEPS,
            "n_accepted": n_accepted,
            "qpe_energies": qpe_energies,
            "n_qpe_evaluations": final_qpe_idx,
            "hf_times": hf_times,
            "qpe_times": qpe_times,
        }

    return mc_solvation_with_qpe_loop


# =============================================================================
# Initialization Functions
# =============================================================================


def initialize_water_states(
    n_waters: int,
    qm_center: np.ndarray,
    initial_distance: float = INITIAL_WATER_DISTANCE,
    random_seed: int = 42,
) -> np.ndarray:
    """Initialize water molecules in a ring around QM center."""
    np.random.seed(random_seed)
    water_states = []

    for i in range(n_waters):
        angle = 2 * np.pi * i / n_waters
        x = float(qm_center[0]) + initial_distance * np.cos(angle)
        y = float(qm_center[1]) + initial_distance * np.sin(angle)
        z = float(qm_center[2])

        position = np.array([x, y, z])
        euler_angles = np.random.uniform(-np.pi, np.pi, size=3)

        state = np.concatenate([position, euler_angles])
        water_states.append(state)

    return np.array(water_states)


# =============================================================================
# Time Statistics Formatting
# =============================================================================


def format_time_statistics(
    qpe_compile_time: float,
    mc_loop_time: float,
    hf_times: np.ndarray,
    qpe_times: np.ndarray,
) -> str:
    """
    Format time statistics in a detailed table format (方案 B).

    Args:
        qpe_compile_time: Time to compile QPE circuit (one-time)
        mc_loop_time: Total MC loop execution time (includes JIT compilation)
        hf_times: Array of HF energy computation times per MC step
        qpe_times: Array of QPE evaluation times

    Returns:
        Formatted string with time statistics table
    """
    # Calculate statistics
    hf_total = np.sum(hf_times)
    hf_avg = np.mean(hf_times) * 1000  # Convert to ms
    hf_std = np.std(hf_times) * 1000

    n_qpe = len(qpe_times[qpe_times > 0])  # Count non-zero entries
    qpe_valid = qpe_times[qpe_times > 0]

    if n_qpe > 0:
        qpe_total = np.sum(qpe_valid)
        qpe_avg = np.mean(qpe_valid) * 1000  # Convert to ms
        qpe_std = np.std(qpe_valid) * 1000
        qpe_first = qpe_valid[0] * 1000 if len(qpe_valid) > 0 else 0
        qpe_subsequent = qpe_valid[1:] if len(qpe_valid) > 1 else np.array([])
        qpe_subsequent_avg = np.mean(qpe_subsequent) * 1000 if len(qpe_subsequent) > 0 else 0
        qpe_subsequent_std = np.std(qpe_subsequent) * 1000 if len(qpe_subsequent) > 0 else 0
    else:
        qpe_total = qpe_avg = qpe_std = qpe_first = qpe_subsequent_avg = qpe_subsequent_std = 0

    # Estimate MC loop JIT compilation overhead (first run - steady state)
    mc_jit_overhead = mc_loop_time - hf_total - qpe_total
    if mc_jit_overhead < 0:
        mc_jit_overhead = 0  # Numerical precision protection

    total_wall_time = qpe_compile_time + mc_loop_time

    n_mc_steps = len(hf_times)

    # Format output - use shorter column widths to fit within 100 chars
    lines = [
        "",
        "  " + "═" * 56,
        "  Time Statistics",
        "  " + "═" * 56,
        "  1. Compilation Phase:",
        f"     - QPE Circuit @qjit:  {qpe_compile_time:.2f} s (one-time)",
        f"     - MC Loop @qjit:      ~{mc_jit_overhead:.2f} s (first-run overhead)",
        "",
        f"  2. Execution Phase ({n_mc_steps} MC steps, {n_qpe} QPE evals):",
        "     ┌────────────────┬────────┬────────┬────────┐",
        "     │ Component      │ Total  │ Avg    │ StdDev │",
        "     ├────────────────┼────────┼────────┼────────┤",
        f"     │ HF ({n_mc_steps:4d}x)     │ {hf_total:5.1f}s │ {hf_avg:5.1f}ms│ {hf_std:5.1f}ms│",
        f"     │ QPE ({n_qpe:4d}x)    │ {qpe_total:5.1f}s │ {qpe_avg:5.1f}ms│ {qpe_std:5.1f}ms│",
    ]

    if n_qpe > 1:
        qpe_subsequent_total = np.sum(qpe_subsequent)  # Already in seconds
        lines.extend(
            [
                f"     │  - First eval  │ {qpe_first:5.1f}ms│   -    │   -    │",
                f"     │  - Subsequent  │ {qpe_subsequent_total:5.2f}s │ {qpe_subsequent_avg:5.1f}ms│"
                f" {qpe_subsequent_std:5.1f}ms│",
            ]
        )

    lines.extend(
        [
            "     └────────────────┴────────┴────────┴────────┘",
            "",
            f"  Total wall time: {total_wall_time:.2f} s",
        ]
    )

    return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================


def run_qjit_mc_solvation(config: SolvationConfig) -> dict:
    """Run QJIT-compiled Monte Carlo solvation optimization with QPE validation."""
    print("=" * 70)
    print("  H2 + TIP3P Water: QJIT-Compiled MC Solvation with QPE Validation")
    print("=" * 70)
    print()

    # Initialize system
    qm_coords = np.array(H2_COORDS_LIST)
    qm_center = np.mean(qm_coords, axis=0)
    water_states = initialize_water_states(
        config.n_waters, qm_center, random_seed=config.random_seed
    )

    print("[Step 1] System Initialization")
    print("-" * 70)
    print("  Solute: H2 (fixed geometry)")
    print(f"  Solvent: {config.n_waters} TIP3P water molecules")
    print("  Initial water positions:")
    for i in range(config.n_waters):
        pos = water_states[i, :3]
        print(f"    Water {i + 1}: O at [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    print()

    # Step 2: Pre-compile QPE circuit (KEY OPTIMIZATION)
    print("[Step 2] Pre-compiling QPE Circuit")
    print("-" * 70)
    print("  Building H2 vacuum Hamiltonian...")

    # Get initial HF energy estimate for optimal base_time
    qm_coords_flat_np = np.array(qm_coords.flatten(), dtype=np.float64)
    water_states_np = np.array(water_states, dtype=np.float64)
    initial_hf_energy = _compute_hf_energy_impl(
        qm_coords_flat_np,
        np.array([]),  # No MM charges for initial estimate
        np.array([]),
    )
    print(f"  Initial HF energy estimate: {initial_hf_energy:.6f} Ha")

    print("  Compiling QPE QNode with @qjit...")
    compile_start = time.perf_counter()
    compiled_qpe_circuit, base_time, qpe_engine = build_precompiled_qpe(
        qm_coords, initial_hf_energy
    )
    # Trigger compilation by running once
    _ = compiled_qpe_circuit()
    compile_time = time.perf_counter() - compile_start
    print(f"  QPE circuit compiled in {compile_time:.2f} s")
    print(f"  Base time: {base_time:.6f}")
    print()

    # Create MC loop with pre-compiled QPE
    mc_solvation_with_qpe_loop = create_mc_solvation_loop(compiled_qpe_circuit, base_time)

    print("[Step 3] Running QJIT-Compiled MC Sampling with QPE")
    print("-" * 70)
    print(f"  MC steps:          {N_MC_STEPS}")
    print(f"  QPE interval:      Every {QPE_INTERVAL} steps ({N_QPE_EVALUATIONS} evaluations)")
    print(f"  Temperature:       {TEMPERATURE} K")
    print(f"  Translation step:  {TRANSLATION_STEP:.2f} Angstrom")
    print(f"  Rotation step:     {np.degrees(ROTATION_STEP):.1f} degrees")
    print()
    print("  QJIT Features:")
    print("  - catalyst.pure_callback for PySCF HF energy")
    print("  - PRE-COMPILED QPE circuit (compiled once, reused 10x)")
    print("  - catalyst.for_loop for MC iterations")
    print("  - catalyst.cond for conditional QPE execution")
    print("  - catalyst.debug.print for real-time QPE output")
    print()
    print("  First run includes MC loop JIT compilation...")
    print()

    # First run (includes JIT compilation of MC loop)
    run_start = time.perf_counter()
    result = mc_solvation_with_qpe_loop(water_states_np, qm_coords_flat_np, config.random_seed)
    run_time = time.perf_counter() - run_start

    print()
    print(f"  MC sampling completed in {run_time:.2f} s")
    print(f"  Acceptance rate: {float(result['acceptance_rate']) * 100:.1f}%")

    # Display time statistics (方案 B: 分层详细式)
    hf_times = np.array(result["hf_times"])
    qpe_times = np.array(result["qpe_times"])
    time_stats = format_time_statistics(compile_time, run_time, hf_times, qpe_times)
    print(time_stats)
    print()

    # Report results
    print("[Step 4] MC Energy Results")
    print("-" * 70)
    print(f"  Initial energy:  {float(result['initial_energy']):.6f} Ha")
    print(f"  Final energy:    {float(result['final_energy']):.6f} Ha")
    print(f"  Best energy:     {float(result['best_energy']):.6f} Ha")
    print(f"  Average energy:  {float(result['avg_energy']):.6f} Ha")
    print()

    energy_change = (
        float(result["best_energy"]) - float(result["initial_energy"])
    ) * HARTREE_TO_KCAL_MOL
    print(f"  Energy change: {energy_change:+.4f} kcal/mol")
    if energy_change < 0:
        print("  [SUCCESS] Found lower energy configuration!")
    else:
        print("  [INFO] No lower energy found")
    print()

    print("[Step 5] Best Configuration")
    print("-" * 70)
    best_waters = np.array(result["best_waters"])
    for i in range(config.n_waters):
        pos = best_waters[i, :3]
        dist = float(np.linalg.norm(pos - qm_center))
        print(
            f"  Water {i + 1}: O at [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], dist = {dist:.2f} A"
        )
    print()
    print("=" * 70)

    return result


def main():
    """Run the QJIT MC solvation optimization example with QPE validation."""
    config = SolvationConfig(
        n_waters=N_WATERS,
        n_mc_steps=N_MC_STEPS,
        temperature=TEMPERATURE,
        random_seed=42,
    )

    result = run_qjit_mc_solvation(config)
    return result


if __name__ == "__main__":
    main()
