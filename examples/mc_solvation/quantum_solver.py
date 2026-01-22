# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Quantum Solver Interface for Ground State Energy Estimation

This module defines the abstract interface for quantum solvers used in
QM/MM simulations. Concrete implementations (QPE, VQE, etc.) are in
separate modules.

Design Philosophy:
    - Interface segregation: Only define what's needed for MC solvation
    - Open for extension: Easy to add new solver implementations
    - Configuration-driven: Each solver has its own config dataclass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# =============================================================================
# Result Container
# =============================================================================


@dataclass
class SolverResult:
    """
    Universal result container for quantum solvers.

    Attributes:
        energy: Estimated ground state energy in Hartree
        converged: Whether the algorithm converged
        n_evaluations: Number of circuit evaluations/iterations
        method: Name of the algorithm used
        metadata: Algorithm-specific additional data
    """

    energy: float
    converged: bool
    n_evaluations: int
    method: str
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Abstract Solver Interface
# =============================================================================


class QuantumSolver(ABC):
    """
    Abstract base class for quantum ground state solvers.

    Implementations should provide:
    - solve(): Execute the algorithm and return energy estimate
    - get_compiled_circuit(): Return pre-compiled circuit if available

    Supported implementations:
    - QPESolver (qpe_solver.py): Standard Quantum Phase Estimation
    - Future: VQESolver, IterativeQPESolver, etc.
    """

    @abstractmethod
    def solve(
        self,
        hamiltonian: Any,  # PennyLane Operator
        hf_state: np.ndarray,
        n_qubits: int,
        e_ref: float = 0.0,
    ) -> SolverResult:
        """
        Estimate ground state energy of the Hamiltonian.

        Args:
            hamiltonian: PennyLane Hamiltonian operator
            hf_state: Hartree-Fock reference state (binary occupation array)
            n_qubits: Number of system qubits
            e_ref: Reference energy for shifted algorithms (e.g., vacuum HF)

        Returns:
            SolverResult containing energy and convergence info
        """
        pass

    @abstractmethod
    def get_compiled_circuit(self) -> Any | None:
        """
        Get pre-compiled circuit if available.

        Some solvers (like QPE) support circuit pre-compilation for
        repeated execution with the same Hamiltonian. This is useful
        for the vacuum_correction mode where the circuit can be reused.

        Returns:
            Compiled circuit callable, or None if not supported/available
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the solver algorithm name."""
        pass


# =============================================================================
# Factory Function
# =============================================================================


def create_solver(method: str, config: Any = None) -> QuantumSolver:
    """
    Factory function to create quantum solvers.

    Args:
        method: Algorithm name (case-insensitive)
            - "qpe": Standard Quantum Phase Estimation
            - Future: "vqe", "iterative_qpe", etc.
        config: Solver-specific configuration object

    Returns:
        Configured QuantumSolver instance

    Raises:
        ValueError: If method is not recognized

    Example:
        from qpe_solver import QPESolverConfig
        solver = create_solver("qpe", QPESolverConfig(n_estimation_wires=6))
    """
    method_lower = method.lower()

    if method_lower == "qpe":
        from .qpe_solver import QPESolver, QPESolverConfig

        cfg = config if isinstance(config, QPESolverConfig) else QPESolverConfig()
        return QPESolver(cfg)

    # Future implementations can be added here:
    # elif method_lower == "vqe":
    #     from .vqe_solver import VQESolver, VQESolverConfig
    #     ...

    else:
        raise ValueError(f"Unknown solver method: {method}. " f"Available methods: qpe")
