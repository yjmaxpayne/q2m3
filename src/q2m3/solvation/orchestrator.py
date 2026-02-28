"""
MC Solvation Orchestrator.

End-to-end workflow for QPE-driven Monte Carlo solvation simulations.
Supports three Hamiltonian modes:

    - hf_corrected: E_HF + E_MM for acceptance, interval-based QPE diagnostics
    - fixed: Compile-once vacuum Hamiltonian, QPE every step
    - dynamic: Per-step MM-embedded Hamiltonian, QPE every step

Usage:
    from q2m3.solvation import SolvationConfig, MoleculeConfig, run_solvation

    config = SolvationConfig(
        molecule=MoleculeConfig(name="H2", symbols=["H","H"],
                                coords=[[0,0,0],[0,0,0.74]], charge=0),
        hamiltonian_mode="fixed",
        n_waters=10,
        n_mc_steps=100,
    )
    result = run_solvation(config, show_plots=False)
"""

import os
import time
from typing import Any

import numpy as np
from rich.console import Console
from rich.panel import Panel

from q2m3.constants import HARTREE_TO_KCAL_MOL

from .circuit_builder import build_qpe_circuit
from .config import SolvationConfig
from .energy import (
    compute_hf_energy_vacuum,
    compute_mulliken_charges,
    create_hf_corrected_step_callback,
    create_step_callback,
    precompute_vacuum_cache,
)
from .mc_loop import MCResult, create_mc_loop
from .solvent import TIP3P_WATER, initialize_solvent_ring, molecules_to_state_array
from .statistics import create_timing_data_from_result, print_time_statistics

console = Console()


def run_solvation(config: SolvationConfig, show_plots: bool = True) -> dict[str, Any]:
    """
    Execute complete MC solvation workflow.

    Workflow:
    1. Validate config
    2. Initialize solvent ring
    3. Compute vacuum HF energy
    4. Build QPE circuit (build_qpe_circuit -> QPECircuitBundle)
    5. Precompute vacuum cache
    6. Create step callback (mode-dependent)
    7. Compute initial energy (first step_callback call -> triggers @qjit)
    8. Create MC loop (create_mc_loop)
    9. Execute MC loop
    10. Compute Mulliken charges (diagnostic)
    11. Print statistics + plot (if requested)
    12. Return result dict

    Args:
        config: Complete solvation simulation configuration
        show_plots: Whether to display energy trajectory plots

    Returns:
        dict with Layer 1 (MCResult) + Layer 2 (orchestrator) fields
    """
    config.validate()
    mol = config.molecule
    mode = config.hamiltonian_mode

    if config.verbose:
        console.print(
            Panel(
                f"[bold]{mol.name}[/bold] + {config.n_waters} TIP3P Waters\n"
                f"Mode: [cyan]{mode}[/cyan]",
                title="MC Solvation Simulation",
                border_style="blue",
            )
        )

    # === Step 1: Initialize System ===
    qm_coords = mol.coords_array
    qm_center = qm_coords.mean(axis=0)

    solvent_molecules = initialize_solvent_ring(
        model=TIP3P_WATER,
        n_molecules=config.n_waters,
        center=qm_center,
        radius=config.initial_water_distance,
        random_seed=config.random_seed,
    )
    solvent_states = molecules_to_state_array(solvent_molecules)

    if config.verbose:
        console.print(f"  Solute: {mol.name} ({len(mol.symbols)} atoms)")
        console.print(f"  Solvent: {config.n_waters} TIP3P water molecules")
        console.print(f"  Active space: ({mol.active_electrons}e, {mol.active_orbitals}o)")

    # === Step 2: Vacuum HF Reference ===
    e_vacuum = compute_hf_energy_vacuum(mol)
    if config.verbose:
        console.print(f"  Vacuum HF energy: {e_vacuum:.6f} Ha")

    # === Step 3: Build QPE Circuit ===
    # Redirect Catalyst @qjit compilation artifacts to cache directory.
    # Catalyst captures os.getcwd() at decoration time for intermediate files.
    if config.ir_cache_enabled:
        from .ir_cache import cache_path_for_config

        _cache_dir = cache_path_for_config(config).parent
        _cache_dir.mkdir(parents=True, exist_ok=True)
        _original_cwd = os.getcwd()
        os.chdir(str(_cache_dir))
    try:
        circuit_bundle = build_qpe_circuit(config, qm_coords, e_vacuum)
    finally:
        if config.ir_cache_enabled:
            os.chdir(_original_cwd)

    # === Step 3.5: IR Cache Resolution ===
    cache_stats: dict = {"is_cache_hit": False}
    if config.ir_cache_enabled:
        from .ir_cache import resolve_compiled_circuit

        circuit_bundle, cache_stats = resolve_compiled_circuit(config, circuit_bundle)
        if config.verbose:
            if cache_stats.get("is_cache_hit"):
                t = cache_stats.get("phase_b_time_s", 0)
                console.print(f"  [green]IR cache hit[/green] ({t:.2f}s)")
            elif cache_stats.get("fallback"):
                console.print("  [yellow]IR cache miss (fallback to normal compile)[/yellow]")
            else:
                ta = cache_stats.get("phase_a_time_s", 0)
                tb = cache_stats.get("phase_b_time_s", 0)
                console.print(
                    f"  [cyan]IR cache miss[/cyan] " f"(Phase A: {ta:.1f}s, Phase B: {tb:.2f}s)"
                )

    # Circuit metadata (all modes share same bundle structure)
    n_trotter_requested = config.qpe_config.n_trotter_steps
    energy_formula_map = {
        "hf_corrected": "E_HF(R)+E_MM",
        "fixed": "E_QPE(H_vac)+E_MM",
        "dynamic": "E_QPE(H_eff)+E_MM",
    }
    circuit_metadata = {
        "hamiltonian_mode": mode,
        "n_system_qubits": circuit_bundle.n_system_qubits,
        "n_estimation_wires": circuit_bundle.n_estimation_wires,
        "total_qubits": circuit_bundle.n_system_qubits + circuit_bundle.n_estimation_wires,
        "n_hamiltonian_terms": len(circuit_bundle.base_coeffs),
        "n_trotter_steps": circuit_bundle.n_trotter_steps,
        "n_trotter_steps_requested": n_trotter_requested,
        "base_time": circuit_bundle.base_time,
        "energy_formula": energy_formula_map.get(mode, "unknown"),
        "energy_shift": circuit_bundle.energy_shift,
    }

    if config.verbose:
        n_bins = 2**circuit_bundle.n_estimation_wires
        phase_res = 2 * np.pi / (circuit_bundle.base_time * n_bins)
        console.print(
            f"  Base time: {circuit_bundle.base_time:.6f} a.u. "
            f"(resolution: {phase_res * 1000:.1f} mHa/bin, {n_bins} bins)"
        )

    # === Step 4: Precompute Vacuum Cache ===
    vacuum_cache = precompute_vacuum_cache(config)

    # === Step 5: Create Step Callback (mode-dependent) ===
    if mode == "hf_corrected":
        step_callback = create_hf_corrected_step_callback(circuit_bundle, config, vacuum_cache)
    else:
        # fixed and dynamic both use create_step_callback
        step_callback = create_step_callback(circuit_bundle, config, vacuum_cache)

    # === Step 6: Compute Initial Energy (triggers @qjit) ===
    qm_coords_flat = qm_coords.flatten().astype(np.float64)
    solvent_states_np = solvent_states.astype(np.float64)

    if config.verbose:
        console.print("  [dim]Computing initial energy (triggers @qjit compilation)...[/dim]")

    precompute_start = time.perf_counter()
    init_result = step_callback(solvent_states_np, qm_coords_flat)
    precompute_time = time.perf_counter() - precompute_start

    if mode == "hf_corrected":
        initial_energy = init_result.e_hf_ref + init_result.e_mm_sol_sol
    else:
        initial_energy = init_result.e_qpe + init_result.e_mm_sol_sol

    if config.verbose:
        console.print(f"  Initial energy computation: {precompute_time:.2f}s")
        console.print(f"  Initial energy: {initial_energy:.6f} Ha")

    # === Step 7: Create MC Loop ===
    mc_loop = create_mc_loop(config, step_callback)

    # === Step 8: Execute MC Loop ===
    if config.verbose:
        console.print(f"  MC steps: {config.n_mc_steps}")
        console.print(f"  Temperature: {config.temperature} K")

    loop_start = time.perf_counter()
    mc_result: MCResult = mc_loop(
        solvent_states_np, qm_coords_flat, config.random_seed, initial_energy
    )
    loop_time = time.perf_counter() - loop_start

    if config.verbose:
        console.print(f"  MC sampling completed in {loop_time:.2f}s")
        console.print(f"  Acceptance rate: {mc_result.acceptance_rate * 100:.1f}%")

    # === Step 9: Mulliken Charges (diagnostic) ===
    mulliken_charges = compute_mulliken_charges(mol)

    # === Step 10: Timing Statistics ===
    # Build result dict from MCResult (Layer 1)
    result: dict[str, Any] = {
        "initial_energy": mc_result.initial_energy,
        "final_energy": mc_result.final_energy,
        "best_energy": mc_result.best_energy,
        "best_qpe_energy": mc_result.best_qpe_energy,
        "acceptance_rate": mc_result.acceptance_rate,
        "avg_energy": mc_result.avg_energy,
        "quantum_energies": mc_result.quantum_energies,
        "hf_energies": mc_result.hf_energies,
        "hf_times": mc_result.callback_times,  # Alias: callback_times -> hf_times
        "quantum_times": mc_result.quantum_times,
        "final_solvent_states": mc_result.final_solvent_states,
        "best_solvent_states": mc_result.best_solvent_states,
        "best_qpe_solvent_states": mc_result.best_qpe_solvent_states,
        "n_quantum_evaluations": mc_result.n_quantum_evaluations,
        "n_accepted": mc_result.n_accepted,
    }

    # Layer 2 — Orchestrator injected fields
    timing = create_timing_data_from_result(
        result, compile_time=precompute_time, loop_time=loop_time, hamiltonian_mode=mode
    )
    result["timing"] = timing
    result["e_vacuum"] = e_vacuum
    result["circuit_metadata"] = circuit_metadata
    result["mulliken_charges"] = mulliken_charges
    result["cache_stats"] = cache_stats

    if config.verbose:
        # Print results summary
        best_e = mc_result.best_energy
        energy_change = (best_e - mc_result.initial_energy) * HARTREE_TO_KCAL_MOL
        console.print(f"\n  Best energy: {best_e:.6f} Ha")
        console.print(f"  Energy change: {energy_change:+.4f} kcal/mol")
        print_time_statistics(timing, console)

    # === Step 11: Plots ===
    if show_plots:
        from .plotting import plot_energy_trajectory

        quantum_energies = mc_result.quantum_energies
        n_eval = mc_result.n_quantum_evaluations
        if n_eval > 0:
            valid_mask = ~np.isnan(quantum_energies)
            valid_steps = np.where(valid_mask)[0] + 1  # 1-indexed
            plot_energy_trajectory(
                mc_steps=list(range(1, config.n_mc_steps + 1)),
                hf_energies=mc_result.hf_energies,
                quantum_steps=valid_steps.tolist(),
                quantum_energies=quantum_energies[valid_mask].tolist(),
                title=f"{mol.name} MC Solvation - Energy Trajectory ({mode})",
                show=show_plots,
            )

    return result
