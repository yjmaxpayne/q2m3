# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
MC Solvation Module: Modular QJIT-Compiled Monte Carlo Solvation Framework

This module provides a clean, modular implementation of Monte Carlo solvation
simulations with quantum algorithm validation. Supports arbitrary molecular
systems and multiple QPE modes.

Module Structure:
    - constants: Physical constants and default parameters
    - config: Configuration dataclasses (MoleculeConfig, QPEConfig, SolvationConfig)
    - solvent: Data-driven solvent models (TIP3P, SPC/E) and transformations
    - energy: HF and MM energy computation via PySCF
    - quantum_solver: Abstract interface for quantum algorithms
    - qpe_solver: QPE implementation with Catalyst support
    - mc_loop: @qjit compiled MC loop factory
    - statistics: Time statistics formatting with Rich
    - plotting: Energy trajectory visualization
    - orchestrator: Main workflow with qpe_mode switching

QPE Modes:
    - vacuum_correction: E_total = E_QPE(vacuum) + ΔE_MM(HF)
      Fast, pre-compiled circuit reused for all evaluations
    - mm_embedded: E_total = E_QPE(with_MM_embedding)
      Diagonal MM embedding: includes diagonal one-electron corrections (delta_h1e[p,p]);
      neglects off-diagonal h1e terms and two-electron modifications

Usage:
    from mc_solvation import run_solvation, SolvationConfig, MoleculeConfig

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

# Configuration
from .config import MoleculeConfig, QPEConfig, SolvationConfig

# Constants
from .constants import (
    ANGSTROM_TO_BOHR,
    BOLTZMANN_CONSTANT,
    HARTREE_TO_KCAL_MOL,
    KCAL_TO_HARTREE,
)

# Energy computation
from .energy import (
    build_operator_index_map,
    compute_hf_energy_solvated,
    compute_hf_energy_vacuum,
    compute_mm_correction,
    compute_total_energy,
    create_coeff_callback,
    create_fused_qpe_callback,
    create_qpe_step_callback,
    decompose_hamiltonian,
    precompute_vacuum_cache,
)

# MC loop
from .mc_loop import (
    create_classical_mc_loop,
    create_mc_loop,
    create_mm_embedded_mc_loop,
    create_qpe_driven_mc_loop,
)

# Main orchestrator
from .orchestrator import run_solvation
from .plotting import plot_energy_comparison, plot_energy_trajectory

# QPE implementation
from .qpe_solver import QPESolver, QPESolverConfig

# Quantum solver interface
from .quantum_solver import QuantumSolver, SolverResult, create_solver

# Solvent models
from .solvent import (
    SOLVENT_MODELS,
    SPC_E_WATER,
    TIP3P_WATER,
    SolventAtom,
    SolventModel,
    SolventMolecule,
    initialize_solvent_ring,
    molecules_to_state_array,
    state_array_to_molecules,
)

# Statistics and plotting
from .statistics import (
    TimingData,
    create_timing_data_from_result,
    print_time_statistics,
)

__all__ = [
    # Main entry point
    "run_solvation",
    # Configuration
    "SolvationConfig",
    "MoleculeConfig",
    "QPEConfig",
    # Solvent
    "SolventAtom",
    "SolventModel",
    "SolventMolecule",
    "TIP3P_WATER",
    "SPC_E_WATER",
    "SOLVENT_MODELS",
    "initialize_solvent_ring",
    "molecules_to_state_array",
    "state_array_to_molecules",
    # Energy
    "compute_hf_energy_vacuum",
    "compute_hf_energy_solvated",
    "compute_total_energy",
    "compute_mm_correction",
    "decompose_hamiltonian",
    "build_operator_index_map",
    "create_coeff_callback",
    "create_fused_qpe_callback",
    "create_qpe_step_callback",
    "precompute_vacuum_cache",
    # Quantum solver
    "QuantumSolver",
    "SolverResult",
    "create_solver",
    "QPESolver",
    "QPESolverConfig",
    # MC loop
    "create_mc_loop",
    "create_classical_mc_loop",
    "create_mm_embedded_mc_loop",
    "create_qpe_driven_mc_loop",
    # Statistics
    "TimingData",
    "print_time_statistics",
    "create_timing_data_from_result",
    # Plotting
    "plot_energy_trajectory",
    "plot_energy_comparison",
    # Constants
    "HARTREE_TO_KCAL_MOL",
    "ANGSTROM_TO_BOHR",
    "BOLTZMANN_CONSTANT",
    "KCAL_TO_HARTREE",
]
