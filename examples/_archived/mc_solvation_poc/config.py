# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Configuration Classes for MC Solvation Simulations

This module provides generic dataclasses for configuring molecular systems,
QPE parameters, and MC solvation simulations. Designed to support any
molecular system, not limited to specific molecules.
"""

from dataclasses import dataclass, field
from typing import Literal

from q2m3.molecule import MoleculeConfig  # re-export: definition moved to src/q2m3/molecule.py

from .constants import (
    BOLTZMANN_CONSTANT,
    DEFAULT_INITIAL_WATER_DISTANCE,
    DEFAULT_N_ESTIMATION_WIRES,
    DEFAULT_N_MC_STEPS,
    DEFAULT_N_QPE_SHOTS,
    DEFAULT_N_TROTTER_STEPS,
    DEFAULT_N_WATERS,
    DEFAULT_QPE_INTERVAL,
    DEFAULT_ROTATION_STEP,
    DEFAULT_TEMPERATURE,
    DEFAULT_TRANSLATION_STEP,
)


@dataclass
class QPEConfig:
    """
    QPE (Quantum Phase Estimation) configuration.

    Generic QPE parameters that can be tuned for different molecular systems
    and desired precision levels.

    Attributes:
        n_estimation_wires: Number of qubits for phase estimation register
        n_trotter_steps: Number of Trotter steps for time evolution
        n_shots: Number of measurement shots per QPE evaluation
        qpe_interval: MC step interval between QPE evaluations
        target_resolution: Target energy resolution in Hartree
        energy_range: Energy range for shifted QPE in Hartree
        use_catalyst: Whether to use Catalyst @qjit compilation
    """

    n_estimation_wires: int = DEFAULT_N_ESTIMATION_WIRES
    n_trotter_steps: int = DEFAULT_N_TROTTER_STEPS
    n_shots: int = DEFAULT_N_QPE_SHOTS
    qpe_interval: int = DEFAULT_QPE_INTERVAL
    target_resolution: float = 0.003  # ~2 kcal/mol
    energy_range: float = 0.2  # ±0.1 Ha
    use_catalyst: bool = True


@dataclass
class SolvationConfig:
    """
    Complete MC solvation simulation configuration.

    This is the main configuration class that combines molecule, QPE, and MC
    parameters. The qpe_mode field enables switching between different energy
    evaluation strategies.

    Attributes:
        molecule: Molecular system configuration (any molecule)
        qpe_config: QPE parameters
        qpe_mode: Energy evaluation mode
            - "vacuum_correction": E_QPE(vacuum) + ΔE_MM(HF) [fast, approximate]
            - "mm_embedded": E_QPE(with_MM_embedding) [slow, more complete]
            - "qpe_driven": E_QPE(H_eff) + E_MM(sol-sol) [QPE drives Metropolis every step]
        n_waters: Number of TIP3P water molecules
        n_mc_steps: Total MC sampling steps
        temperature: Simulation temperature in Kelvin
        translation_step: Max translation per MC move in Angstrom
        rotation_step: Max rotation per MC move in radians
        initial_water_distance: Initial water ring distance from solute center
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress information

    Example:
        config = SolvationConfig(
            molecule=my_molecule,
            qpe_mode="vacuum_correction",
            n_waters=10,
            n_mc_steps=1000,
        )
    """

    molecule: MoleculeConfig
    qpe_config: QPEConfig = field(default_factory=QPEConfig)
    qpe_mode: Literal["vacuum_correction", "mm_embedded", "qpe_driven"] = "vacuum_correction"
    n_waters: int = DEFAULT_N_WATERS
    n_mc_steps: int = DEFAULT_N_MC_STEPS
    temperature: float = DEFAULT_TEMPERATURE
    translation_step: float = DEFAULT_TRANSLATION_STEP
    rotation_step: float = DEFAULT_ROTATION_STEP
    initial_water_distance: float = DEFAULT_INITIAL_WATER_DISTANCE
    random_seed: int = 42
    verbose: bool = True

    @property
    def n_qpe_evaluations(self) -> int:
        """Calculate total number of QPE evaluations."""
        if self.qpe_mode == "qpe_driven":
            return self.n_mc_steps
        return self.n_mc_steps // self.qpe_config.qpe_interval

    @property
    def kt(self) -> float:
        """Thermal energy kT in Hartree."""
        return BOLTZMANN_CONSTANT * self.temperature

    def validate(self) -> None:
        """Validate configuration consistency."""
        self.molecule.validate()
        if self.n_waters < 1:
            raise ValueError("n_waters must be at least 1")
        if self.n_mc_steps < 1:
            raise ValueError("n_mc_steps must be at least 1")
        if self.qpe_config.qpe_interval > self.n_mc_steps:
            raise ValueError("qpe_interval cannot exceed n_mc_steps")
        if self.qpe_mode not in ("vacuum_correction", "mm_embedded", "qpe_driven"):
            raise ValueError(f"Invalid qpe_mode: {self.qpe_mode}")
