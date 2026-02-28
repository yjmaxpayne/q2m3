"""
Quantum Solver Interface for Ground State Energy Estimation.

Abstract base class for quantum solvers used in QM/MM simulations.
Concrete implementations (QPE, VQE, etc.) are in separate modules.

This is a self-contained module — not exported from core/__init__.py.
Future VQE/QAOA solvers can import directly:
    from q2m3.core.quantum_solver import QuantumSolver, SolverResult
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SolverResult:
    """Universal result container for quantum solvers.

    Attributes:
        energy: Estimated ground state energy in Hartree.
        phase: Estimated phase value from QPE.
        raw_measurements: Algorithm-specific measurement data.
        metadata: Additional algorithm-specific information.
    """

    energy: float
    phase: float
    raw_measurements: Any
    metadata: dict = field(default_factory=dict)


class QuantumSolver(ABC):
    """Abstract base class for quantum ground state solvers.

    Implementations should provide:
    - solve(): Execute the algorithm and return energy estimate
    - compile(): Pre-compile circuit for repeated execution
    """

    @abstractmethod
    def solve(self, hamiltonian_data: dict) -> SolverResult:
        """Estimate ground state energy of the Hamiltonian.

        Args:
            hamiltonian_data: Dict containing Hamiltonian operator,
                HF state, qubit count, and reference energy.

        Returns:
            SolverResult with energy and measurement data.
        """
        ...

    @abstractmethod
    def compile(self, **kwargs: Any) -> None:
        """Pre-compile circuit for repeated execution.

        Some solvers (like QPE) support circuit pre-compilation
        for reuse across MC steps with the same Hamiltonian.
        """
        ...
