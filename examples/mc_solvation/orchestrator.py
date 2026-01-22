# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
MC Solvation Orchestrator

Main workflow coordinator for QM/MM Monte Carlo solvation simulations
with quantum algorithm validation. Supports two QPE modes:

1. vacuum_correction: E_total = E_QPE(vacuum) + ΔE_MM(HF)
   - Fast: QPE circuit pre-compiled once, reused for all evaluations
   - Approximate: Ignores correlation-polarization coupling

2. mm_embedded: E_total = E_QPE(with_MM_embedding)
   - Rigorous: MM charges included in QPE Hamiltonian
   - Slow: Requires dynamic Hamiltonian reconstruction per evaluation

Usage:
    from mc_solvation import SolvationConfig, MoleculeConfig, run_solvation

    molecule = MoleculeConfig(
        name="H2",
        symbols=["H", "H"],
        coords=[[0, 0, 0], [0, 0, 0.74]],
        charge=0,
        active_electrons=2,
        active_orbitals=2,
    )

    config = SolvationConfig(
        molecule=molecule,
        qpe_mode="vacuum_correction",
        n_waters=10,
        n_mc_steps=100,
    )

    result = run_solvation(config)
"""

import time
from collections import Counter
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel

from q2m3.core import QPEEngine
from q2m3.interfaces import PySCFPennyLaneConverter

from .config import SolvationConfig
from .constants import HARTREE_TO_KCAL_MOL
from .energy import compute_hf_energy_vacuum, compute_mm_correction
from .mc_loop import create_mc_loop
from .solvent import TIP3P_WATER, initialize_solvent_ring, molecules_to_state_array
from .statistics import create_timing_data_from_result, print_time_statistics

console = Console()


# =============================================================================
# Energy Callback Implementations (for @qjit pure_callback)
# =============================================================================


def _create_energy_callback_vacuum_correction(config: SolvationConfig, e_vacuum: float):
    """
    Create energy callback for vacuum_correction mode.

    Energy decomposition: E_total = E_HF(solvated) + E_MM(solvent-solvent)
    """
    from .energy import _compute_total_energy_impl

    symbols = config.molecule.symbols
    charge = config.molecule.charge
    basis = config.molecule.basis

    def compute_energy_with_timing(qm_coords_flat, solvent_states):
        """Returns [energy, elapsed_time]."""
        start = time.perf_counter()
        energy = _compute_total_energy_impl(
            symbols,
            qm_coords_flat,
            charge,
            basis,
            solvent_states,
            "TIP3P",
        )
        elapsed = time.perf_counter() - start
        return np.array([energy, elapsed], dtype=np.float64)

    return compute_energy_with_timing


def _create_mm_correction_callback(config: SolvationConfig, e_vacuum: float):
    """
    Create MM correction callback for vacuum_correction mode.

    MM correction: delta_e_mm = E_HF(solvated) - E_HF(vacuum)

    This correction captures the electrostatic effect of MM charges on the QM region.
    """
    from .energy import _compute_hf_energy_solvated_impl
    from .solvent import SOLVENT_MODELS, get_mm_embedding_data, state_array_to_molecules

    symbols = config.molecule.symbols
    charge = config.molecule.charge
    basis = config.molecule.basis

    def compute_mm_correction(solvent_states, qm_coords_flat):
        """Compute delta_e_mm = E_HF(solvated) - E_HF(vacuum)."""
        # Convert state array to molecules to get MM embedding data
        model = SOLVENT_MODELS["TIP3P"]
        solvent_molecules = state_array_to_molecules(model, np.asarray(solvent_states))

        # Get MM embedding coordinates and charges
        mm_coords, mm_charges = get_mm_embedding_data(solvent_molecules)

        # Compute solvated HF energy
        e_solvated = _compute_hf_energy_solvated_impl(
            symbols,
            np.asarray(qm_coords_flat),
            charge,
            basis,
            mm_coords.flatten(),
            mm_charges,
        )

        # MM correction
        delta_e_mm = e_solvated - e_vacuum
        return np.float64(delta_e_mm)

    return compute_mm_correction


def _create_energy_callback_mm_embedded(config: SolvationConfig):
    """
    Create energy callback for mm_embedded mode.

    This mode requires dynamic Hamiltonian construction with MM embedding
    for each QPE evaluation. Currently implemented as placeholder.
    """
    raise NotImplementedError(
        "mm_embedded mode requires dynamic Hamiltonian construction. "
        "This mode is planned for future implementation."
    )


# =============================================================================
# QPE Circuit Builder
# =============================================================================


def build_vacuum_qpe_circuit(
    config: SolvationConfig,
    qm_coords: np.ndarray,
    hf_energy_estimate: float,
):
    """
    Build pre-compiled QPE circuit for vacuum Hamiltonian.

    This function creates a QPE circuit that can be reused for all evaluations
    in vacuum_correction mode.

    Args:
        config: Solvation configuration
        qm_coords: QM region coordinates in Angstrom, shape (n_atoms, 3)
        hf_energy_estimate: Estimated HF energy for optimal base_time

    Returns:
        Tuple of (compiled_circuit, base_time, qpe_engine)
    """
    mol = config.molecule
    qpe = config.qpe_config

    # Build vacuum Hamiltonian
    converter = PySCFPennyLaneConverter(basis=mol.basis, mapping="jordan_wigner")
    H, n_qubits, hf_state = converter.pyscf_to_pennylane_hamiltonian(
        symbols=mol.symbols,
        coords=qm_coords,
        charge=mol.charge,
        active_electrons=mol.active_electrons,
        active_orbitals=mol.active_orbitals,
    )

    # Create QPE engine
    qpe_engine = QPEEngine(
        n_qubits=n_qubits,
        n_iterations=8,
        mapping="jordan_wigner",
        use_catalyst=qpe.use_catalyst,
    )

    # Compute optimal parameters using energy-shifted QPE
    params = QPEEngine.compute_shifted_qpe_params(
        target_resolution=qpe.target_resolution,
        energy_range=qpe.energy_range,
    )
    base_time = params["base_time"]

    n_estimation_wires = qpe.n_estimation_wires
    if n_estimation_wires <= 0:
        n_estimation_wires = params["n_estimation_wires"]

    # Build and compile circuit
    compiled_circuit = qpe_engine._build_standard_qpe_circuit(
        H,
        hf_state,
        n_estimation_wires=n_estimation_wires,
        base_time=base_time,
        n_trotter_steps=qpe.n_trotter_steps,
        n_shots=qpe.n_shots,
    )

    return compiled_circuit, base_time, qpe_engine


def extract_qpe_energy_from_samples(samples: np.ndarray, base_time: float) -> float:
    """
    Extract energy from QPE measurement samples.

    Physics: QPE measures phase φ where U|ψ⟩ = exp(i2πφ)|ψ⟩
    For U = exp(-iHt), we have φ = -Et/(2π), so E = -2πφ/t
    """
    samples = np.asarray(samples, dtype=np.int64)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)

    n_bits = samples.shape[1]
    phase_indices = []
    for sample in samples:
        idx = sum(int(bit) * (2 ** (n_bits - 1 - k)) for k, bit in enumerate(sample))
        phase_indices.append(idx)

    counter = Counter(phase_indices)
    mode_idx, _ = counter.most_common(1)[0]

    mode_phase = mode_idx / (2**n_bits)
    energy = -2 * np.pi * mode_phase / base_time

    return np.float64(energy)


# =============================================================================
# Main Orchestrator
# =============================================================================


def run_solvation(config: SolvationConfig, show_plots: bool = True) -> dict[str, Any]:
    """
    Run MC solvation simulation with quantum algorithm validation.

    This is the main entry point that orchestrates the entire workflow:
    1. Initialize system (QM region + solvent)
    2. Pre-compile QPE circuit (for vacuum_correction mode)
    3. Run @qjit compiled MC loop with periodic QPE evaluation
    4. Report results and statistics

    Args:
        config: Complete solvation simulation configuration
        show_plots: Whether to display energy trajectory plots

    Returns:
        Dictionary with simulation results including:
            - initial_energy: Starting total energy
            - final_energy: Final total energy
            - best_energy: Lowest energy found
            - acceptance_rate: Fraction of accepted MC moves
            - quantum_energies: Array of QPE energy estimates
            - timing: TimingData object with performance statistics
    """
    config.validate()
    mol = config.molecule

    console.print(
        Panel(
            f"[bold]{mol.name}[/bold] + {config.n_waters} TIP3P Waters\n"
            f"Mode: [cyan]{config.qpe_mode}[/cyan]",
            title="MC Solvation Simulation",
            border_style="blue",
        )
    )

    # ==========================================================================
    # Step 1: Initialize System
    # ==========================================================================
    console.print("\n[bold]Step 1:[/bold] System Initialization")

    qm_coords = mol.coords_array
    qm_center = mol.center

    # Initialize solvent molecules
    solvent_molecules = initialize_solvent_ring(
        model=TIP3P_WATER,
        n_molecules=config.n_waters,
        center=qm_center,
        radius=config.initial_water_distance,
        random_seed=config.random_seed,
    )
    solvent_states = molecules_to_state_array(solvent_molecules)

    console.print(f"  Solute: {mol.name} ({mol.n_atoms} atoms)")
    console.print(f"  Solvent: {config.n_waters} TIP3P water molecules")
    console.print(f"  Active space: ({mol.active_electrons}e, {mol.active_orbitals}o)")

    # ==========================================================================
    # Step 2: Compute Vacuum Reference
    # ==========================================================================
    console.print("\n[bold]Step 2:[/bold] Computing Vacuum Reference")

    e_vacuum = compute_hf_energy_vacuum(mol)
    console.print(f"  Vacuum HF energy: {e_vacuum:.6f} Ha")

    # ==========================================================================
    # Step 3: Pre-compile QPE Circuit
    # ==========================================================================
    console.print("\n[bold]Step 3:[/bold] Pre-compiling QPE Circuit")

    compile_start = time.perf_counter()
    compiled_circuit, base_time, qpe_engine = build_vacuum_qpe_circuit(config, qm_coords, e_vacuum)
    # Trigger compilation
    _ = compiled_circuit()
    compile_time = time.perf_counter() - compile_start

    console.print(f"  Compilation time: {compile_time:.2f}s")
    console.print(f"  Base time (shifted QPE): {base_time:.6f}")

    # ==========================================================================
    # Step 4: Create and Run MC Loop
    # ==========================================================================
    console.print("\n[bold]Step 4:[/bold] Running MC Sampling")
    console.print(f"  MC steps: {config.n_mc_steps}")
    console.print(f"  QPE interval: every {config.qpe_config.qpe_interval} steps")
    console.print(f"  Temperature: {config.temperature} K")

    # Create energy callback based on mode
    if config.qpe_mode == "vacuum_correction":
        energy_callback = _create_energy_callback_vacuum_correction(config, e_vacuum)
        mm_correction_callback = _create_mm_correction_callback(config, e_vacuum)
    else:
        energy_callback = _create_energy_callback_mm_embedded(config)
        mm_correction_callback = None  # Not needed for mm_embedded mode

    # Create MC loop with pre-compiled QPE
    # Pass e_vacuum and mm_correction_callback for vacuum_correction mode
    mc_loop = create_mc_loop(
        config=config,
        compiled_circuit=compiled_circuit,
        extract_energy=extract_qpe_energy_from_samples,
        compute_energy_impl=energy_callback,
        base_time=base_time,
        e_vacuum=e_vacuum,
        compute_mm_correction_impl=mm_correction_callback,
    )

    # Run MC loop
    qm_coords_flat = qm_coords.flatten().astype(np.float64)
    solvent_states_np = solvent_states.astype(np.float64)

    loop_start = time.perf_counter()
    result = mc_loop(solvent_states_np, qm_coords_flat, config.random_seed)
    loop_time = time.perf_counter() - loop_start

    console.print(f"\n  MC sampling completed in {loop_time:.2f}s")
    console.print(f"  Acceptance rate: {float(result['acceptance_rate']) * 100:.1f}%")

    # ==========================================================================
    # Step 5: Report Results
    # ==========================================================================
    console.print("\n[bold]Step 5:[/bold] Results")

    initial_e = float(result["initial_energy"])
    final_e = float(result["final_energy"])
    best_e = float(result["best_energy"])
    energy_change = (best_e - initial_e) * HARTREE_TO_KCAL_MOL

    console.print(f"  Initial energy: {initial_e:.6f} Ha")
    console.print(f"  Final energy:   {final_e:.6f} Ha")
    console.print(f"  Best HF energy: {best_e:.6f} Ha")
    console.print(f"  HF energy change: {energy_change:+.4f} kcal/mol")

    # QPE-validated best energy (more accurate with electron correlation)
    best_qpe_e = float(result["best_qpe_energy"])
    n_eval = int(result["n_quantum_evaluations"])
    if n_eval > 0 and best_qpe_e < 1e9:  # Check if any QPE evaluation occurred
        qpe_change = (best_qpe_e - initial_e) * HARTREE_TO_KCAL_MOL
        console.print()
        console.print("[bold cyan]  QPE-Validated Best (Recommended):[/bold cyan]")
        console.print(f"    Best QPE energy: {best_qpe_e:.6f} Ha")
        console.print(f"    QPE energy change: {qpe_change:+.4f} kcal/mol")

        # Display best QPE configuration (all solvent molecules)
        best_qpe_solvents = np.array(result["best_qpe_solvent_states"])
        console.print(f"    Configuration (solvent O positions):")
        for i in range(config.n_waters):
            pos = best_qpe_solvents[i, :3]
            console.print(
                f"      Water {i + 1:2d}: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}] Å"
            )

    # Timing statistics
    timing = create_timing_data_from_result(result, compile_time, loop_time)
    print_time_statistics(timing, console)

    # Plots
    if show_plots:
        from .plotting import plot_energy_trajectory

        quantum_energies = np.array(result["quantum_energies"])
        n_eval = int(result["n_quantum_evaluations"])
        if n_eval > 0:
            quantum_steps = [(i + 1) * config.qpe_config.qpe_interval for i in range(n_eval)]
            mc_steps = list(range(1, config.n_mc_steps + 1))
            # For HF energies, we'd need to track them separately in mc_loop
            # For now, just plot quantum energies
            plot_energy_trajectory(
                mc_steps=quantum_steps,
                hf_energies=quantum_energies[:n_eval],  # Placeholder
                quantum_steps=quantum_steps,
                quantum_energies=quantum_energies[:n_eval],
                title=f"{mol.name} MC Solvation - Energy Trajectory",
                show=show_plots,
            )

    # Add timing to result
    result["timing"] = timing
    result["e_vacuum"] = e_vacuum

    console.print("\n" + "=" * 60)

    return result
