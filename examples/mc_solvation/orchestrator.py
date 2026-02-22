# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
MC Solvation Orchestrator

Main workflow coordinator for QM/MM Monte Carlo solvation simulations
with quantum algorithm validation. Supports two QPE modes:

1. vacuum_correction: E_total = E_corr(vac) + E_HF(R)
   - QPE on H' = H_vac - E_HF·I measures correlation energy directly
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

import jax.numpy as jnp
import numpy as np
import pennylane as qml
from catalyst import qjit
from rich.console import Console
from rich.panel import Panel

from q2m3.core import QPEEngine
from q2m3.core.device_utils import select_device as _select_device
from q2m3.interfaces import PySCFPennyLaneConverter

from .config import SolvationConfig
from .constants import HARTREE_TO_KCAL_MOL
from .energy import (
    build_operator_index_map,
    compute_hf_energy_vacuum,
    compute_mm_correction,
    compute_mulliken_charges,
    create_coeff_callback,
    decompose_hamiltonian,
)
from .mc_loop import create_mc_loop, create_mm_embedded_mc_loop
from .solvent import TIP3P_WATER, initialize_solvent_ring, molecules_to_state_array
from .statistics import create_timing_data_from_result, print_time_statistics

console = Console()

# Runtime coefficient parameterization imposes a Catalyst compilation memory ceiling.
# The MLIR compiler cannot constant-fold JAX-traced coefficients, so the full symbolic
# computation graph must fit in memory. Benchmark-validated: n_estimation=2, n_trotter=3
# compiles in ~24s for H2 and ~96s for H3O+. Exceeding this limit risks OOM (SIGKILL).
_MAX_TROTTER_STEPS_RUNTIME = 20


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


def _create_energy_callback_mm_embedded(config: SolvationConfig, e_vacuum: float):
    """
    Create energy callback for mm_embedded mode.

    The MC acceptance criterion uses HF-level total energy (same as vacuum_correction).
    The QPE energy with MM embedding is computed separately via the coefficient callback.
    """
    return _create_energy_callback_vacuum_correction(config, e_vacuum)


# =============================================================================
# QPE Circuit Builder
# =============================================================================


def build_vacuum_qpe_circuit(
    config: SolvationConfig,
    qm_coords: np.ndarray,
    hf_energy_estimate: float,
):
    """
    Build pre-compiled QPE circuit for energy-shifted vacuum Hamiltonian.

    Constructs H' = H_vac - E_HF·I so QPE measures correlation energy
    directly (delta_e ≈ E_corr). This eliminates phase aliasing that occurs
    when the full eigenvalue (~-1.137 Ha for H2) wraps multiple times in
    the phase register designed for a ~0.2 Ha energy range.

    Args:
        config: Solvation configuration
        qm_coords: QM region coordinates in Angstrom, shape (n_atoms, 3)
        hf_energy_estimate: Estimated HF energy for energy shift and base_time

    Returns:
        Tuple of (compiled_circuit, base_time, qpe_engine, n_hamiltonian_terms,
                  energy_shift, n_estimation_wires)
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

    # Energy shift: H' = H - E_HF·I (same pattern as mm_embedded)
    coeffs, ops = decompose_hamiltonian(H)
    op_index_map, coeffs, ops = build_operator_index_map(ops, n_qubits, coeffs)
    energy_shift = hf_energy_estimate
    coeffs[op_index_map["identity_idx"]] -= energy_shift
    H_shifted = qml.dot(coeffs, ops)

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

    # Build @qjit circuit returning probability distribution over estimation register
    n_system = n_qubits
    n_trotter = qpe.n_trotter_steps
    system_wires = list(range(n_system))
    est_wires = list(range(n_system, n_system + n_estimation_wires))
    total_wires = n_system + n_estimation_wires

    dev = _select_device("lightning.qubit", total_wires, use_catalyst=True)

    @qjit
    def vacuum_qpe_probs():
        @qml.qnode(dev)
        def qnode():
            # HF state preparation via X gates (Catalyst-compatible)
            for wire, occ in zip(system_wires, hf_state, strict=True):
                if occ == 1:
                    qml.PauliX(wires=wire)
            # Hadamard on estimation qubits
            for w in est_wires:
                qml.Hadamard(wires=w)
            # Controlled time evolutions (MSB-first convention)
            for k, ew in enumerate(est_wires):
                t = (2 ** (n_estimation_wires - 1 - k)) * base_time
                qml.ctrl(
                    qml.adjoint(qml.TrotterProduct(H_shifted, time=t, n=n_trotter, order=2)),
                    control=ew,
                )
            # Inverse QFT
            qml.adjoint(qml.QFT)(wires=est_wires)
            # Return probability distribution over estimation register
            return qml.probs(wires=est_wires)

        return qnode()

    compiled_circuit = vacuum_qpe_probs

    return compiled_circuit, base_time, qpe_engine, len(ops), energy_shift, n_estimation_wires


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


def build_mm_embedded_qpe_circuit(
    config: SolvationConfig,
    qm_coords: np.ndarray,
    hf_energy_estimate: float,
):
    """
    Build parametrized QPE circuit for mm_embedded mode.

    The circuit accepts Hamiltonian coefficients as a runtime argument.
    Operators are captured in the closure (compile-time constants).
    Coefficients are JAX-traceable runtime values -> compile once, reuse.

    Requires check_hermitian=False for TrotterProduct (JAX tracers fail
    math.iscomplex() check).

    Args:
        config: Solvation configuration
        qm_coords: QM region coordinates in Angstrom, shape (n_atoms, 3)
        hf_energy_estimate: Estimated HF energy for shifted QPE parameters

    Returns:
        Tuple of (compiled_circuit, base_coeffs, ops, base_time,
                  op_index_map, n_estimation_wires, energy_shift, n_system_qubits,
                  active_orbitals, n_trotter_actual)
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

    # Decompose into coefficients and operators
    coeffs, ops = decompose_hamiltonian(H)

    # Build operator index map (may extend coeffs/ops with missing Z terms)
    op_index_map, coeffs, ops = build_operator_index_map(ops, n_qubits, coeffs)

    # Compute shifted QPE parameters
    params = QPEEngine.compute_shifted_qpe_params(
        target_resolution=qpe.target_resolution,
        energy_range=qpe.energy_range,
    )
    base_time = params["base_time"]
    energy_shift = hf_energy_estimate

    n_estimation_wires = qpe.n_estimation_wires
    if n_estimation_wires <= 0:
        n_estimation_wires = params["n_estimation_wires"]

    # Apply energy shift to Identity coefficient
    identity_idx = op_index_map["identity_idx"]
    coeffs[identity_idx] -= energy_shift

    # Save base coefficients (vacuum + shift applied)
    base_coeffs = np.array(coeffs, dtype=np.float64)

    # Build @qjit parametrized QPE circuit
    n_system = n_qubits
    n_trotter = qpe.n_trotter_steps

    # Guard: cap Trotter steps for runtime-parameterized circuits to avoid
    # Catalyst MLIR compilation OOM. With JAX-traced coefficients, each
    # TrotterProduct gate retains symbolic parameters that cannot be folded,
    # scaling IR size as n_estimation × n_trotter × n_hamiltonian_terms.
    if n_trotter > _MAX_TROTTER_STEPS_RUNTIME:
        console.print(
            f"  [yellow]Warning: n_trotter_steps={n_trotter} exceeds runtime-parameterized "
            f"circuit ceiling ({_MAX_TROTTER_STEPS_RUNTIME}). Capping to avoid Catalyst "
            f"compilation OOM.[/yellow]"
        )
        n_trotter = _MAX_TROTTER_STEPS_RUNTIME
    system_wires = list(range(n_system))
    est_wires = list(range(n_system, n_system + n_estimation_wires))
    total_wires = n_system + n_estimation_wires

    dev = _select_device("lightning.qubit", total_wires, use_catalyst=True)

    @qjit
    def qpe_with_coeffs(coeffs_arr):
        H_runtime = qml.dot(coeffs_arr, ops)

        @qml.qnode(dev)
        def qnode():
            # HF state preparation via X gates (Catalyst-compatible)
            for wire, occ in zip(system_wires, hf_state, strict=True):
                if occ == 1:
                    qml.PauliX(wires=wire)
            # Hadamard on estimation qubits
            for w in est_wires:
                qml.Hadamard(wires=w)
            # Controlled time evolutions (MSB-first convention)
            for k, ew in enumerate(est_wires):
                t = (2 ** (n_estimation_wires - 1 - k)) * base_time
                qml.ctrl(
                    qml.adjoint(
                        qml.TrotterProduct(
                            H_runtime, time=t, n=n_trotter, order=2, check_hermitian=False
                        )
                    ),
                    control=ew,
                )
            # Inverse QFT
            qml.adjoint(qml.QFT)(wires=est_wires)
            # Return probability distribution over estimation register
            return qml.probs(wires=est_wires)

        return qnode()

    return (
        qpe_with_coeffs,
        base_coeffs,
        ops,
        base_time,
        op_index_map,
        n_estimation_wires,
        energy_shift,
        n_system,
        mol.active_orbitals,
        n_trotter,
    )


# =============================================================================
# Main Orchestrator
# =============================================================================


def run_solvation(config: SolvationConfig, show_plots: bool = True) -> dict[str, Any]:
    """
    Run MC solvation simulation with quantum algorithm validation.

    This is the main entry point that orchestrates the entire workflow:
    1. Initialize system (QM region + solvent)
    2. Compute vacuum HF reference energy
    3. Build QPE circuit + run @qjit MC loop with periodic QPE evaluation
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
    # Step 3: Build QPE Circuit + Run MC Sampling
    # ==========================================================================
    console.print("\n[bold]Step 3:[/bold] Running MC Sampling")
    console.print(f"  Mode: [cyan]{config.qpe_mode}[/cyan]")

    # Build QPE circuit (no compilation triggered yet)
    qpe = config.qpe_config
    if config.qpe_mode == "vacuum_correction":
        (
            compiled_circuit,
            base_time,
            qpe_engine,
            n_hamiltonian_terms,
            energy_shift,
            n_estimation_wires_actual,
        ) = build_vacuum_qpe_circuit(config, qm_coords, e_vacuum)
        circuit_metadata = {
            "n_system_qubits": qpe_engine.n_qubits,
            "n_estimation_wires": n_estimation_wires_actual,
            "total_qubits": qpe_engine.n_qubits + n_estimation_wires_actual,
            "n_hamiltonian_terms": n_hamiltonian_terms,
            "n_trotter_steps": qpe.n_trotter_steps,
            "n_trotter_steps_requested": qpe.n_trotter_steps,
            "base_time": base_time,
            "energy_formula": "E_corr(vac) + E_HF(R)",
            "energy_shift": energy_shift,
        }
    elif config.qpe_mode == "mm_embedded":
        (
            compiled_circuit,
            base_coeffs,
            ops,
            base_time,
            op_index_map,
            n_estimation_wires_actual,
            energy_shift,
            n_system_qubits,
            active_orbitals,
            n_trotter_actual,
        ) = build_mm_embedded_qpe_circuit(config, qm_coords, e_vacuum)
        circuit_metadata = {
            "n_system_qubits": n_system_qubits,
            "n_estimation_wires": n_estimation_wires_actual,
            "total_qubits": n_system_qubits + n_estimation_wires_actual,
            "n_hamiltonian_terms": len(base_coeffs),
            "n_trotter_steps": n_trotter_actual,
            "n_trotter_steps_requested": qpe.n_trotter_steps,
            "base_time": base_time,
            "energy_formula": "E_QPE(H_eff with MM)",
            "energy_shift": energy_shift,
        }

    console.print(f"  Base time (shifted QPE): {base_time:.6f}")
    console.print(f"  Energy shift: {energy_shift:.6f} Ha")
    if config.qpe_mode == "mm_embedded":
        console.print(f"  Hamiltonian terms: {len(base_coeffs)}")

    # Create MC loop
    if config.qpe_mode == "vacuum_correction":
        energy_callback = _create_energy_callback_vacuum_correction(config, e_vacuum)
        mm_correction_callback = _create_mm_correction_callback(config, e_vacuum)

        mc_loop = create_mc_loop(
            config=config,
            compiled_circuit=compiled_circuit,
            compute_energy_impl=energy_callback,
            base_time=base_time,
            n_estimation_wires=n_estimation_wires_actual,
            energy_shift=energy_shift,
            e_vacuum=e_vacuum,
            compute_mm_correction_impl=mm_correction_callback,
        )
    elif config.qpe_mode == "mm_embedded":
        energy_callback = _create_energy_callback_mm_embedded(config, e_vacuum)
        coeff_callback = create_coeff_callback(config, base_coeffs, op_index_map, active_orbitals)

        mc_loop = create_mm_embedded_mc_loop(
            config=config,
            compiled_circuit=compiled_circuit,
            compute_energy_impl=energy_callback,
            compute_coeffs_impl=coeff_callback,
            base_time=base_time,
            base_coeffs=base_coeffs,
            n_estimation_wires=n_estimation_wires_actual,
            energy_shift=energy_shift,
        )

    console.print(f"  MC steps: {config.n_mc_steps}")
    console.print(f"  QPE interval: every {config.qpe_config.qpe_interval} steps")
    console.print(f"  Temperature: {config.temperature} K")

    # Run MC loop (first call triggers @qjit compilation, including QPE circuit)
    qm_coords_flat = qm_coords.flatten().astype(np.float64)
    solvent_states_np = solvent_states.astype(np.float64)

    console.print("  [dim]Compiling @qjit (first-run, includes QPE circuit)...[/dim]")

    loop_start = time.perf_counter()
    result = mc_loop(solvent_states_np, qm_coords_flat, config.random_seed)
    loop_time = time.perf_counter() - loop_start

    # Compute compilation overhead from timing breakdown
    hf_times_arr = np.array(result["hf_times"])
    q_times_arr = np.array(result["quantum_times"])
    mc_exec_time = float(np.sum(hf_times_arr) + np.sum(q_times_arr[q_times_arr > 0]))
    mc_compile_time = max(0.0, loop_time - mc_exec_time)

    console.print(f"\n  @qjit compilation: {mc_compile_time:.2f}s (first-run, includes QPE)")
    console.print(f"  MC sampling completed in {mc_exec_time:.2f}s")
    console.print(f"  Acceptance rate: {float(result['acceptance_rate']) * 100:.1f}%")

    # ==========================================================================
    # Step 4: Report Results
    # ==========================================================================
    console.print("\n[bold]Step 4:[/bold] Results")

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

        # Solvation stabilization: E(vacuum) - E(best_QPE) > 0 means solvent stabilizes
        stabilization_ha = e_vacuum - best_qpe_e
        stabilization_kcal = stabilization_ha * HARTREE_TO_KCAL_MOL
        console.print()
        console.print(
            "  [bold green]Solvation Stabilization (vacuum → optimal solvation shell):[/bold green]"
        )
        console.print(f"    dE = {stabilization_ha:.6f} Ha")
        console.print(f"       = {stabilization_kcal:.2f} kcal/mol")
        if stabilization_ha > 0:
            console.print("    [OK] Solvent stabilizes the molecule")
        else:
            console.print("    [WARNING] No net stabilization detected")

        # Mulliken charge redistribution (vacuum → best solvated config)
        console.print()
        console.print("  Mulliken Charge Redistribution (vacuum → best solvation shell):")
        charges_vacuum = compute_mulliken_charges(mol)
        charges_solvated = compute_mulliken_charges(mol, best_qpe_solvents)
        for atom, q_vac in charges_vacuum.items():
            q_sol = charges_solvated[atom]
            dq = q_sol - q_vac
            console.print(f"    {atom}: {q_vac:+.4f} → {q_sol:+.4f} (Δq = {dq:+.4f})")

        console.print(f"    Configuration (solvent O positions):")
        for i in range(config.n_waters):
            pos = best_qpe_solvents[i, :3]
            console.print(
                f"      Water {i + 1:2d}: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}] Å"
            )

    # Timing statistics
    timing = create_timing_data_from_result(result, 0.0, loop_time)
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

    # Add timing and metadata to result
    result["timing"] = timing
    result["e_vacuum"] = e_vacuum
    result["circuit_metadata"] = circuit_metadata

    console.print("\n" + "=" * 60)

    return result
