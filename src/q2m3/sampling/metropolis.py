# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Metropolis-Hastings sampler for solvation structure optimization.

This module provides a Metropolis-Hastings algorithm for sampling
water configurations around a fixed solute molecule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mc_moves import DEFAULT_ROTATION_STEP, DEFAULT_TRANSLATION_STEP, MCMoveGenerator
from .water_molecule import WaterMolecule

# Physical constants
BOLTZMANN_CONSTANT = 3.1668114e-6  # Hartree/K


@dataclass
class MetropolisSampler:
    """
    Metropolis-Hastings sampler for water configuration optimization.

    Uses MC moves to sample configurations and accepts/rejects based on
    the Metropolis criterion with the given energy function.

    Attributes:
        waters: List of WaterMolecule objects (MM region)
        energy_function: Function that takes waters and returns energy in Hartree
        temperature: Temperature in Kelvin for Boltzmann acceptance
        translation_step: Maximum translation per move (Angstrom)
        rotation_step: Maximum rotation per move (radians)
    """

    waters: list[WaterMolecule]
    energy_function: Callable[[list[WaterMolecule]], float]
    temperature: float = 300.0
    translation_step: float = DEFAULT_TRANSLATION_STEP
    rotation_step: float = DEFAULT_ROTATION_STEP
    _move_generator: MCMoveGenerator = field(init=False)

    def __post_init__(self):
        """Initialize move generator."""
        self._move_generator = MCMoveGenerator(
            translation_step=self.translation_step,
            rotation_step=self.rotation_step,
        )

    def _accept_move(self, old_energy: float, new_energy: float) -> bool:
        """
        Decide whether to accept a move using Metropolis criterion.

        Args:
            old_energy: Energy before move (Hartree)
            new_energy: Energy after move (Hartree)

        Returns:
            True if move should be accepted
        """
        delta_e = new_energy - old_energy

        # Always accept lower energy
        if delta_e <= 0:
            return True

        # Probabilistic acceptance for higher energy
        # P(accept) = exp(-delta_E / (k_B * T))
        kT = BOLTZMANN_CONSTANT * self.temperature
        acceptance_prob = np.exp(-delta_e / kT)

        return np.random.random() < acceptance_prob

    def run(self, n_steps: int) -> dict[str, Any]:
        """
        Run Metropolis-Hastings sampling for n_steps.

        Args:
            n_steps: Number of MC steps to run

        Returns:
            Dictionary containing:
                - energies: List of energies at each step
                - acceptance_rate: Fraction of accepted moves
                - best_energy: Lowest energy found
                - best_config: Water configuration with lowest energy
        """
        # Initialize
        current_waters = [w.copy() for w in self.waters]
        current_energy = self.energy_function(current_waters)

        energies = []
        n_accepted = 0

        best_energy = current_energy
        best_config = [w.copy() for w in current_waters]

        for _ in range(n_steps):
            # Propose move
            new_waters, _ = self._move_generator.propose_move_for_system(current_waters)
            new_energy = self.energy_function(new_waters)

            # Accept/reject
            if self._accept_move(current_energy, new_energy):
                current_waters = new_waters
                current_energy = new_energy
                n_accepted += 1

                # Update best
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_config = [w.copy() for w in current_waters]

            energies.append(current_energy)

        # Update sampler state with final configuration
        self.waters = current_waters

        return {
            "energies": energies,
            "acceptance_rate": n_accepted / n_steps if n_steps > 0 else 0.0,
            "best_energy": best_energy,
            "best_config": best_config,
        }
