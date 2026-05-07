# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
MC Solvation Orchestrator.

End-to-end workflow for QPE-driven Monte Carlo solvation simulations.
Supports three Hamiltonian modes:

* hf_corrected: E_HF + E_MM for acceptance, interval-based QPE diagnostics
* fixed: Compile-once vacuum Hamiltonian, QPE every step
* dynamic: Per-step MM-embedded Hamiltonian, QPE every step

Usage::

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


def _build_runtime_qpe_assets(
    config: SolvationConfig,
    qm_coords: np.ndarray,
    e_vacuum: float,
    *,
    verbose: bool,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Build and resolve fixed or dynamic QPE assets."""
    energy_formula_map = {
        "hf_corrected": "E_HF(R)+E_MM",
        "fixed": "E_QPE(H_vac)+E_MM",
        "dynamic": "E_QPE(H_eff)+E_MM",
    }

    if config.ir_cache_enabled:
        from .ir_cache import cache_path_for_config

        cache_dir = cache_path_for_config(config).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        original_cwd = os.getcwd()
        os.chdir(str(cache_dir))
    try:
        circuit_bundle = build_qpe_circuit(config, qm_coords, e_vacuum)
    finally:
        if config.ir_cache_enabled:
            os.chdir(original_cwd)

    cache_stats: dict[str, Any] = {"is_cache_hit": False}
    if config.ir_cache_enabled:
        from .ir_cache import resolve_compiled_circuit

        circuit_bundle, cache_stats = resolve_compiled_circuit(config, circuit_bundle)
        if verbose:
            if cache_stats.get("is_cache_hit"):
                t = cache_stats.get("phase_b_time_s", 0)
                console.print(f"  [green]IR cache hit[/green] ({t:.2f}s)")
            elif cache_stats.get("fallback"):
                console.print("  [yellow]IR cache miss (fallback to normal compile)[/yellow]")
            else:
                ta = cache_stats.get("phase_a_time_s", 0)
                tb = cache_stats.get("phase_b_time_s", 0)
                console.print(
                    f"  [cyan]IR cache miss[/cyan] (Phase A: {ta:.1f}s, Phase B: {tb:.2f}s)"
                )

    n_trotter_requested = config.qpe_config.n_trotter_steps
    circuit_metadata = {
        "hamiltonian_mode": config.hamiltonian_mode,
        "n_system_qubits": circuit_bundle.n_system_qubits,
        "n_estimation_wires": circuit_bundle.n_estimation_wires,
        "total_qubits": circuit_bundle.n_system_qubits + circuit_bundle.n_estimation_wires,
        "n_hamiltonian_terms": len(circuit_bundle.base_coeffs),
        "n_trotter_steps": circuit_bundle.n_trotter_steps,
        "n_trotter_steps_requested": n_trotter_requested,
        "base_time": circuit_bundle.base_time,
        "energy_formula": energy_formula_map[config.hamiltonian_mode],
        "energy_shift": circuit_bundle.energy_shift,
    }
    return circuit_bundle, cache_stats, circuit_metadata


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

    # === Step 3: Build QPE Circuit (mode-dependent) ===
    circuit_bundle = None
    cache_stats: dict = {"is_cache_hit": False}
    energy_formula_map = {
        "hf_corrected": "E_HF(R)+E_MM",
        "fixed": "E_QPE(H_vac)+E_MM",
        "dynamic": "E_QPE(H_eff)+E_MM",
    }

    if mode == "hf_corrected":
        # Deferred compilation: skip circuit build at startup.
        # The hf_corrected step callback will lazily build on first QPE step.
        cache_stats["deferred"] = True
        circuit_metadata = {
            "hamiltonian_mode": mode,
            "energy_formula": energy_formula_map[mode],
            "deferred": True,
        }
        if config.verbose:
            console.print("  [dim]QPE circuit build deferred to first QPE interval[/dim]")
    else:
        circuit_bundle, cache_stats, circuit_metadata = _build_runtime_qpe_assets(
            config,
            qm_coords,
            e_vacuum,
            verbose=config.verbose,
        )

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
        step_callback = create_hf_corrected_step_callback(
            config,
            vacuum_cache,
            qm_coords,
            e_vacuum,
        )
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

    # === Step 6.5: Update hf_corrected circuit_metadata from deferred build ===
    if mode == "hf_corrected":
        post_bundle = getattr(step_callback, "_state", {}).get("bundle")
        if post_bundle is not None:
            n_trotter_requested = config.qpe_config.n_trotter_steps
            circuit_metadata.update(
                {
                    "n_system_qubits": post_bundle.n_system_qubits,
                    "n_estimation_wires": post_bundle.n_estimation_wires,
                    "total_qubits": post_bundle.n_system_qubits + post_bundle.n_estimation_wires,
                    "n_hamiltonian_terms": len(post_bundle.base_coeffs),
                    "n_trotter_steps": post_bundle.n_trotter_steps,
                    "n_trotter_steps_requested": n_trotter_requested,
                    "base_time": post_bundle.base_time,
                    "energy_shift": post_bundle.energy_shift,
                }
            )
        if config.verbose and config.ir_cache_enabled:
            post_cs = getattr(step_callback, "_state", {}).get("cache_stats", {})
            if post_cs:
                if post_cs.get("is_cache_hit"):
                    t = post_cs.get("phase_b_time_s", 0)
                    console.print(f"  [green]IR cache hit[/green] ({t:.2f}s)")
                elif post_cs.get("fallback"):
                    console.print("  [yellow]IR cache miss (fallback to normal compile)[/yellow]")
                else:
                    ta = post_cs.get("phase_a_time_s", 0)
                    tb = post_cs.get("phase_b_time_s", 0)
                    console.print(
                        f"  [cyan]IR cache miss[/cyan]" f" (Phase A: {ta:.1f}s, Phase B: {tb:.2f}s)"
                    )

    if config.verbose:
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
        "trajectory_solvent_states": mc_result.trajectory_solvent_states,
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


def replay_quantum_trajectory(
    config: SolvationConfig,
    trajectory_solvent_states: np.ndarray,
) -> dict[str, Any]:
    """Replay fixed or dynamic quantum evaluations on a saved solvent trajectory."""
    config.validate()
    if config.hamiltonian_mode not in {"fixed", "dynamic"}:
        raise ValueError("replay_quantum_trajectory only supports fixed or dynamic modes")

    trajectory = np.asarray(trajectory_solvent_states, dtype=np.float64)
    if trajectory.ndim != 3:
        raise ValueError("trajectory_solvent_states must have shape (n_steps, n_waters, 6)")
    if trajectory.shape[1] != config.n_waters or trajectory.shape[2] != 6:
        raise ValueError(
            "trajectory_solvent_states must have shape "
            f"(n_steps, {config.n_waters}, 6), got {trajectory.shape}"
        )

    qm_coords = config.molecule.coords_array
    qm_coords_flat = qm_coords.flatten().astype(np.float64)
    e_vacuum = compute_hf_energy_vacuum(config.molecule)

    build_start = time.perf_counter()
    circuit_bundle, cache_stats, circuit_metadata = _build_runtime_qpe_assets(
        config,
        qm_coords,
        e_vacuum,
        verbose=False,
    )
    build_time = time.perf_counter() - build_start

    vacuum_cache = precompute_vacuum_cache(config)
    step_callback = create_step_callback(circuit_bundle, config, vacuum_cache)

    n_steps = trajectory.shape[0]
    quantum_energies = np.full(n_steps, np.nan)
    hf_energies = np.zeros(n_steps)
    callback_times = np.zeros(n_steps)
    quantum_times = np.zeros(n_steps)

    loop_start = time.perf_counter()
    for step_idx, solvent_states in enumerate(trajectory):
        step_result = step_callback(solvent_states, qm_coords_flat)
        quantum_energies[step_idx] = step_result.e_qpe
        hf_energies[step_idx] = step_result.e_hf_ref
        callback_times[step_idx] = step_result.callback_time
        quantum_times[step_idx] = step_result.qpe_time
    loop_time = time.perf_counter() - loop_start

    best_qpe_idx = int(np.nanargmin(quantum_energies)) if n_steps else 0
    result: dict[str, Any] = {
        "quantum_energies": quantum_energies,
        "hf_energies": hf_energies,
        "hf_times": callback_times,
        "quantum_times": quantum_times,
        "trajectory_solvent_states": trajectory.copy(),
        "final_solvent_states": trajectory[-1].copy(),
        "best_qpe_solvent_states": trajectory[best_qpe_idx].copy(),
        "best_qpe_energy": float(np.nanmin(quantum_energies)),
        "n_quantum_evaluations": int(np.sum(~np.isnan(quantum_energies))),
        "e_vacuum": e_vacuum,
        "circuit_metadata": circuit_metadata,
        "cache_stats": cache_stats,
    }
    result["timing"] = create_timing_data_from_result(
        result,
        compile_time=build_time,
        loop_time=loop_time,
        hamiltonian_mode=config.hamiltonian_mode,
    )
    return result
