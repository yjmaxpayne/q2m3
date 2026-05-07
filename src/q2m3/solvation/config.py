# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Configuration dataclasses for MC solvation simulations.

Provides frozen (immutable) configuration for molecule, QPE parameters,
and solvation simulation settings with the three-mode Hamiltonian framework
(hf_corrected / fixed / dynamic).
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from q2m3.constants import BOLTZMANN_CONSTANT


@dataclass(frozen=True)
class MoleculeConfig:
    """
    Molecular system configuration for solvation simulations.

    Attributes:
        name: Human-readable identifier
        symbols: Atomic symbols (e.g., ["H", "H"])
        coords: Atomic coordinates in Angstrom, shape (n_atoms, 3)
        charge: Total molecular charge (REQUIRED — no default)
        active_electrons: Active electrons for QPE active space
        active_orbitals: Active orbitals for QPE active space
        basis: Basis set name
    """

    name: str
    symbols: list[str]
    coords: list[list[float]]
    charge: int  # REQUIRED: no default — prevents silent physics errors
    active_electrons: int = 2
    active_orbitals: int = 2
    basis: str = "sto-3g"

    @property
    def coords_array(self) -> np.ndarray:
        """Coordinates as numpy array of shape (n_atoms, 3)."""
        return np.array(self.coords)

    def validate(self) -> None:
        """Validate configuration consistency."""
        if len(self.symbols) != len(self.coords):
            raise ValueError(
                f"Number of symbols ({len(self.symbols)}) must match "
                f"number of coordinate sets ({len(self.coords)})"
            )
        for i, coord in enumerate(self.coords):
            if len(coord) != 3:
                raise ValueError(f"Coordinate {i} must have 3 components, got {len(coord)}")


@dataclass(frozen=True)
class QPEConfig:
    """
    QPE (Quantum Phase Estimation) parameters.

    Attributes:
        n_estimation_wires: Qubits for phase estimation register
        n_trotter_steps: Trotter steps for time evolution
        n_shots: Measurement shots (0 = analytical, >0 = shots)
        device_seed: Optional seed for shot-based device sampling reproducibility
        target_resolution: Target energy resolution in Hartree
        energy_range: Energy range for shifted QPE in Hartree
        qpe_interval: MC step interval between QPE evaluations (hf_corrected mode)
    """

    n_estimation_wires: int = 4
    n_trotter_steps: int = 10
    n_shots: int = 0  # 0 = analytical (precision-first); shots mode is optional diagnostic
    device_seed: int | None = None
    target_resolution: float = 0.003
    energy_range: float = 0.2
    qpe_interval: int = 10
    # use_catalyst removed — ADR-004: Catalyst is a hard dependency, always enabled


@dataclass(frozen=True)
class SolvationConfig:
    """
    Complete MC solvation simulation configuration.

    The hamiltonian_mode field selects the energy evaluation strategy:
        - "hf_corrected": E_HF + delta_corr_pol (fast, approximate)
        - "fixed": Compile-once vacuum Hamiltonian (medium)
        - "dynamic": Per-step MM-embedded Hamiltonian (slow, most complete)

    Attributes:
        molecule: Molecular system configuration
        qpe_config: QPE parameters
        hamiltonian_mode: Energy evaluation strategy
        n_waters: Number of TIP3P water molecules
        n_mc_steps: Total MC sampling steps
        temperature: Simulation temperature in Kelvin
        translation_step: Max translation per MC move in Angstrom
        rotation_step: Max rotation per MC move in radians
        initial_water_distance: Initial water ring radius in Angstrom
        random_seed: Random seed for reproducibility
        verbose: Print progress information
        ir_cache_dir: Directory for Catalyst LLVM IR cache, or None for the default cache path
        ir_cache_enabled: Enable reuse of cached Catalyst LLVM IR
        ir_cache_force_recompile: Ignore cached IR and force recompilation
    """

    molecule: MoleculeConfig
    qpe_config: QPEConfig = field(default_factory=QPEConfig)
    hamiltonian_mode: Literal["hf_corrected", "fixed", "dynamic"] = "dynamic"
    n_waters: int = 10
    n_mc_steps: int = 500
    temperature: float = 300.0
    translation_step: float = 0.3
    rotation_step: float = 0.2618
    initial_water_distance: float = 4.0
    random_seed: int = 42
    verbose: bool = True
    # IR cache settings (Catalyst LLVM IR caching for compilation speedup)
    ir_cache_dir: str | None = None  # None = {project_root}/tmp/qpe_ir_cache/
    ir_cache_enabled: bool = True
    ir_cache_force_recompile: bool = False

    @property
    def n_qpe_evaluations(self) -> int:
        """Compute expected QPE evaluation count based on mode."""
        if self.hamiltonian_mode == "hf_corrected":
            return self.n_mc_steps // self.qpe_config.qpe_interval
        return self.n_mc_steps

    @property
    def kt(self) -> float:
        """Thermal energy kT in Hartree."""
        return BOLTZMANN_CONSTANT * self.temperature

    def validate(self) -> "SolvationConfig":
        """
        Validate configuration consistency.

        Returns self to support chaining: config.validate()
        """
        self.molecule.validate()

        if self.n_waters < 1:
            raise ValueError(f"n_waters must be >= 1, got {self.n_waters}")

        if self.n_mc_steps < 1:
            raise ValueError(f"n_mc_steps must be >= 1, got {self.n_mc_steps}")

        _valid_modes = ("hf_corrected", "fixed", "dynamic")
        if self.hamiltonian_mode not in _valid_modes:
            raise ValueError(
                f"hamiltonian_mode must be one of {_valid_modes}, " f"got {self.hamiltonian_mode!r}"
            )

        if (
            self.hamiltonian_mode == "hf_corrected"
            and self.qpe_config.qpe_interval > self.n_mc_steps
        ):
            raise ValueError(
                f"qpe_interval ({self.qpe_config.qpe_interval}) "
                f"must be <= n_mc_steps ({self.n_mc_steps}) in hf_corrected mode"
            )

        return self
