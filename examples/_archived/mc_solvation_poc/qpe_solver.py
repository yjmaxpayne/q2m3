# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
QPE (Quantum Phase Estimation) Solver Implementation

This module provides the QPE solver for ground state energy estimation,
wrapping the q2m3.core.QPEEngine with a configuration-driven interface.

QPE Algorithm Overview:
    1. Prepare system in HF reference state
    2. Apply controlled time evolution U = exp(-iHt)
    3. Inverse QFT on ancilla register
    4. Measure ancilla qubits to extract phase
    5. Convert phase to energy: E = -2πφ/t

Supports Catalyst @qjit compilation for repeated executions.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from q2m3.core import QPEEngine

from .quantum_solver import QuantumSolver, SolverResult

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class QPESolverConfig:
    """
    Configuration for QPE solver.

    Attributes:
        n_estimation_wires: Number of ancilla qubits for phase estimation.
            More qubits = higher precision but deeper circuit.
            If 0, computed automatically from target_resolution.
        n_trotter_steps: Number of Trotter decomposition steps for U=exp(-iHt).
            More steps = better approximation but longer circuit.
        n_shots: Number of measurement shots per QPE execution.
        use_catalyst: Whether to use Catalyst @qjit compilation.
            Enable for repeated executions (vacuum_correction mode).
        target_resolution: Target energy resolution in Hartree.
            Used to compute optimal parameters if not specified.
        energy_range: Expected energy range around reference (Hartree).
            Used for shifted QPE parameter optimization.
    """

    n_estimation_wires: int = 4
    n_trotter_steps: int = 10
    n_shots: int = 50
    use_catalyst: bool = True
    target_resolution: float = 0.003  # ~2 kcal/mol
    energy_range: float = 0.2  # ±0.1 Ha


# =============================================================================
# QPE Solver Implementation
# =============================================================================


class QPESolver(QuantumSolver):
    """
    Standard Quantum Phase Estimation solver.

    This solver uses multiple ancilla qubits to estimate all phase bits
    in a single circuit execution. Suitable for early fault-tolerant
    quantum computers (EFTQC).

    Features:
        - Pre-compilation support via Catalyst @qjit
        - Energy-shifted QPE for high-precision measurements
        - Automatic parameter optimization

    Usage:
        config = QPESolverConfig(n_estimation_wires=6)
        solver = QPESolver(config)

        # Build Hamiltonian (see quantum_solver.build_hamiltonian)
        H, n_qubits, hf_state = build_hamiltonian(molecule)

        result = solver.solve(H, hf_state, n_qubits, e_ref=vacuum_energy)
        print(f"Energy: {result.energy:.6f} Ha")
    """

    def __init__(self, config: QPESolverConfig):
        """
        Initialize QPE solver.

        Args:
            config: QPE configuration parameters
        """
        self.config = config
        self._engine: QPEEngine | None = None
        self._compiled_circuit = None
        self._base_time: float = 0.0
        self._n_estimation_wires: int = 0

    @property
    def name(self) -> str:
        return "QPE"

    def solve(
        self,
        hamiltonian: Any,
        hf_state: np.ndarray,
        n_qubits: int,
        e_ref: float = 0.0,
    ) -> SolverResult:
        """
        Execute QPE and return energy estimate.

        Args:
            hamiltonian: PennyLane Hamiltonian (should be energy-shifted)
            hf_state: HF reference state as binary array
            n_qubits: Number of system qubits
            e_ref: Reference energy (added back to measured delta)

        Returns:
            SolverResult with estimated energy
        """
        # Create or reuse engine
        if self._engine is None or self._engine.n_qubits != n_qubits:
            self._engine = QPEEngine(
                n_qubits=n_qubits,
                n_iterations=8,
                mapping="jordan_wigner",
                use_catalyst=self.config.use_catalyst,
            )

        # Compute optimal QPE parameters
        params = QPEEngine.compute_shifted_qpe_params(
            target_resolution=self.config.target_resolution,
            energy_range=self.config.energy_range,
        )
        self._base_time = params["base_time"]

        # Use configured or computed estimation wires
        self._n_estimation_wires = self.config.n_estimation_wires
        if self._n_estimation_wires <= 0:
            self._n_estimation_wires = params["n_estimation_wires"]

        # Build and compile circuit
        circuit = self._engine._build_standard_qpe_circuit(
            hamiltonian,
            hf_state,
            n_estimation_wires=self._n_estimation_wires,
            base_time=self._base_time,
            n_trotter_steps=self.config.n_trotter_steps,
            n_shots=self.config.n_shots,
        )
        self._compiled_circuit = circuit

        # Execute circuit
        samples = circuit()

        # Extract energy from samples
        energy = self._extract_energy(samples, e_ref)

        return SolverResult(
            energy=energy,
            converged=True,  # QPE always "converges"
            n_evaluations=1,
            method=self.name,
            metadata={
                "n_estimation_wires": self._n_estimation_wires,
                "base_time": self._base_time,
                "n_shots": self.config.n_shots,
                "n_trotter_steps": self.config.n_trotter_steps,
                "samples": np.asarray(samples),
                "e_ref": e_ref,
            },
        )

    def _extract_energy(self, samples: np.ndarray, e_ref: float) -> float:
        """
        Extract energy from QPE measurement samples.

        Physics:
            QPE measures phase φ where U|ψ⟩ = exp(i2πφ)|ψ⟩
            For U = exp(-iHt), we have φ = -Et/(2π)
            Therefore: E = -2πφ/t

        Args:
            samples: Binary samples from estimation register
            e_ref: Reference energy to add back

        Returns:
            Estimated energy in Hartree
        """
        samples = np.asarray(samples, dtype=np.int64)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        n_bits = samples.shape[1]

        # Convert binary samples to phase indices
        phase_indices = []
        for sample in samples:
            idx = sum(int(bit) * (2 ** (n_bits - 1 - k)) for k, bit in enumerate(sample))
            phase_indices.append(idx)

        # Find mode (most frequent measurement)
        counter = Counter(phase_indices)
        mode_idx, _ = counter.most_common(1)[0]

        # Convert to energy
        mode_phase = mode_idx / (2**n_bits)
        delta_e = -2 * np.pi * mode_phase / self._base_time

        return delta_e + e_ref

    def get_compiled_circuit(self) -> Any | None:
        """Return the compiled QPE circuit if available."""
        return self._compiled_circuit

    def precompile(
        self,
        hamiltonian: Any,
        hf_state: np.ndarray,
        n_qubits: int,
    ) -> None:
        """
        Pre-compile QPE circuit for repeated execution.

        Call this method once before MC loop to compile the circuit.
        Subsequent calls to solve() will reuse this compiled circuit
        if the Hamiltonian structure hasn't changed.

        Args:
            hamiltonian: PennyLane Hamiltonian
            hf_state: HF reference state
            n_qubits: Number of system qubits
        """
        # Create engine
        self._engine = QPEEngine(
            n_qubits=n_qubits,
            n_iterations=8,
            mapping="jordan_wigner",
            use_catalyst=self.config.use_catalyst,
        )

        # Compute parameters
        params = QPEEngine.compute_shifted_qpe_params(
            target_resolution=self.config.target_resolution,
            energy_range=self.config.energy_range,
        )
        self._base_time = params["base_time"]

        self._n_estimation_wires = self.config.n_estimation_wires
        if self._n_estimation_wires <= 0:
            self._n_estimation_wires = params["n_estimation_wires"]

        # Build and store compiled circuit
        self._compiled_circuit = self._engine._build_standard_qpe_circuit(
            hamiltonian,
            hf_state,
            n_estimation_wires=self._n_estimation_wires,
            base_time=self._base_time,
            n_trotter_steps=self.config.n_trotter_steps,
            n_shots=self.config.n_shots,
        )

        # Trigger compilation by running once
        _ = self._compiled_circuit()

    def run_precompiled(self, e_ref: float = 0.0) -> SolverResult:
        """
        Run pre-compiled circuit.

        Must call precompile() first.

        Args:
            e_ref: Reference energy to add to measured delta

        Returns:
            SolverResult with estimated energy
        """
        if self._compiled_circuit is None:
            raise RuntimeError("Circuit not pre-compiled. Call precompile() first.")

        samples = self._compiled_circuit()
        energy = self._extract_energy(samples, e_ref)

        return SolverResult(
            energy=energy,
            converged=True,
            n_evaluations=1,
            method=self.name,
            metadata={
                "n_estimation_wires": self._n_estimation_wires,
                "base_time": self._base_time,
                "samples": np.asarray(samples),
                "e_ref": e_ref,
                "precompiled": True,
            },
        )


# =============================================================================
# Callback-Compatible Function for @qjit
# =============================================================================


def extract_energy_from_samples_impl(
    samples: np.ndarray,
    base_time: float,
    e_ref: float,
) -> float:
    """
    Pure callback implementation for energy extraction.

    This function is designed for use with Catalyst pure_callback
    inside @qjit compiled code.

    Args:
        samples: QPE measurement samples
        base_time: Base evolution time
        e_ref: Reference energy

    Returns:
        Energy in Hartree as np.float64
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
    delta_e = -2 * np.pi * mode_phase / base_time

    return np.float64(delta_e + e_ref)
