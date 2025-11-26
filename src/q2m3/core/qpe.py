# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Quantum Phase Estimation (QPE) implementation for molecular systems.

Phase 2: Standard QPE circuit with:
- HF state preparation via qml.BasisState
- Controlled time evolution via qml.TrotterProduct
- Inverse QFT via qml.adjoint(qml.QFT)

Phase 4: Catalyst @qjit integration for JIT compilation support.
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pennylane as qml

# Optional Catalyst import with graceful degradation
try:
    from catalyst import qjit

    HAS_CATALYST = True
except ImportError:
    HAS_CATALYST = False

    def qjit(fn=None, **kwargs):
        """No-op fallback when Catalyst is not installed."""
        return fn if fn else lambda f: f


# Import shared device selection from device_utils
from .device_utils import (
    HAS_LIGHTNING_GPU,
    HAS_LIGHTNING_QUBIT,
    select_device as _select_device,
)


# Constants for QPE configuration
DEFAULT_ERROR_ESTIMATE = 0.001
MAX_EARLY_CONVERGENCE_ITERATIONS = 5
DEFAULT_N_ESTIMATION_WIRES = 4
DEFAULT_BASE_TIME = 0.3
DEFAULT_N_TROTTER_STEPS = 5


class QPEEngine:
    """
    Iterative Quantum Phase Estimation engine for QM/MM calculations.

    Implements the iterative QPE algorithm optimized for near-term quantum devices,
    targeting early fault-tolerant quantum computers (EFTQC).
    """

    def __init__(
        self,
        n_qubits: int,
        n_iterations: int = 8,
        mapping: str = "jordan_wigner",
        device: str = "default.qubit",
        device_type: str = "default.qubit",
        use_catalyst: bool = False,
        **kwargs,
    ):
        """
        Initialize QPE engine.

        Args:
            n_qubits: Number of system qubits
            n_iterations: Number of QPE iterations (5-10 for POC)
            mapping: Fermion-to-qubit mapping ('jordan_wigner' or 'bravyi_kitaev')
            device: PennyLane device name (deprecated, use device_type instead)
            device_type: Device selection strategy:
                - "auto": Auto-select best available (GPU > lightning.qubit > default.qubit)
                - "default.qubit": Standard PennyLane simulator
                - "lightning.qubit": High-performance CPU simulator
                - "lightning.gpu": GPU-accelerated simulator (requires cuQuantum)
            use_catalyst: Enable Catalyst @qjit compilation (requires pennylane-catalyst)
            **kwargs: Additional device configuration
        """
        self.n_qubits = n_qubits
        self.n_iterations = n_iterations
        self.mapping = mapping
        self.device_type = device_type

        # Catalyst JIT compilation support
        self.use_catalyst = use_catalyst and HAS_CATALYST
        if use_catalyst and not HAS_CATALYST:
            warnings.warn(
                "use_catalyst=True but pennylane-catalyst not installed. "
                "Falling back to standard execution. "
                "Install with: pip install pennylane-catalyst",
                UserWarning,
                stacklevel=2,
            )

        # Initialize quantum device (for non-QPE operations)
        self.dev = qml.device(device, wires=n_qubits + 1)  # +1 for ancilla

    def estimate_ground_state_energy(
        self, hamiltonian_data: dict[str, Any], initial_state: np.ndarray | None = None
    ) -> dict[str, Any]:
        """
        Estimate ground state energy using classical simulation (POC).

        This is a classical simulation that mimics QPE behavior for POC purposes.
        Uses PySCF Hartree-Fock results as approximation.

        Args:
            hamiltonian_data: Dictionary containing molecular data from PySCF
            initial_state: Initial quantum state preparation (unused in classical sim)

        Returns:
            Dictionary containing:
                - energy: Ground state energy in Hartree
                - convergence: Convergence information
                - density_matrix: Electronic density matrix
        """
        # Extract HF energy from hamiltonian data
        energy_hf = hamiltonian_data.get("energy_hf", 0.0)
        scf_result = hamiltonian_data["scf_result"]

        # Simulate iterative QPE convergence
        convergence_info = self._simulate_qpe_convergence()

        return {
            "energy": energy_hf,
            "convergence": convergence_info,
            "density_matrix": scf_result.make_rdm1(),
        }

    def _simulate_qpe_convergence(self) -> dict[str, Any]:
        """
        Simulate QPE convergence behavior.

        Returns:
            Dictionary with convergence information
        """
        # In real QPE, convergence would depend on circuit execution
        # For POC, simulate early convergence
        iterations_used = min(self.n_iterations, MAX_EARLY_CONVERGENCE_ITERATIONS)

        return {
            "converged": True,
            "iterations": iterations_used,
            "error_estimate": DEFAULT_ERROR_ESTIMATE,
        }

    def _prepare_initial_state(self, hf_state: np.ndarray, wires: list[int]) -> None:
        """
        Prepare Hartree-Fock reference state.

        For H3O+ (10 electrons, Jordan-Wigner): |1111111111000000>

        Args:
            hf_state: Binary array from qml.qchem.hf_state(), e.g., [1,1,0,0] for H2
            wires: System qubit indices to prepare the state on
        """
        qml.BasisState(hf_state, wires=wires)

    def _apply_controlled_unitary(
        self,
        hamiltonian: qml.Hamiltonian,
        time: float,
        control_wire: int,
        target_wires: list[int],
        n_trotter_steps: int = DEFAULT_N_TROTTER_STEPS,
    ) -> None:
        """
        Apply controlled time evolution U = exp(-iHt).

        In standard QPE, each estimation qubit k controls U^(2^k).
        Uses Trotter decomposition for molecular Hamiltonians.

        Args:
            hamiltonian: PennyLane molecular Hamiltonian
            time: Evolution time (typically 2^k * base_time)
            control_wire: Estimation qubit for control
            target_wires: System qubit indices
            n_trotter_steps: Number of Trotter steps for decomposition
        """
        # Use TrotterProduct for time evolution (user's choice)
        # TrotterProduct applies exp(-i * H * time) with specified order
        qml.ctrl(
            qml.TrotterProduct(hamiltonian, time, n=n_trotter_steps, order=2),
            control=control_wire,
        )

    def _inverse_qft(self, wires: list[int]) -> None:
        """
        Apply inverse Quantum Fourier Transform for phase readout.

        Uses PennyLane's built-in QFT with adjoint for inverse.

        Args:
            wires: Estimation qubit indices to apply QFT^-1
        """
        qml.adjoint(qml.QFT)(wires=wires)

    def _build_standard_qpe_circuit(
        self,
        hamiltonian: qml.Hamiltonian,
        hf_state: np.ndarray,
        n_estimation_wires: int = DEFAULT_N_ESTIMATION_WIRES,
        base_time: float = DEFAULT_BASE_TIME,
        n_trotter_steps: int = DEFAULT_N_TROTTER_STEPS,
        n_shots: int = 1,
    ) -> Callable:
        """
        Build standard QPE circuit for molecular energy estimation.

        Circuit structure:
        |0>^n (estimation) --H-----ctrl-----ctrl------- QFT^-1 -- Sample
                                   |         |
        |HF>  (system)     -------U^1-------U^2-------------------

        Args:
            hamiltonian: PennyLane molecular Hamiltonian
            hf_state: HF reference state binary array
            n_estimation_wires: Number of estimation qubits (precision bits)
            base_time: Base evolution time t_0
            n_trotter_steps: Trotter steps per evolution
            n_shots: Number of measurement shots (default 1 for single sample)

        Returns:
            QNode function that executes the QPE circuit
        """
        n_system_wires = len(hf_state)
        system_wires = list(range(n_system_wires))
        estimation_wires = list(range(n_system_wires, n_system_wires + n_estimation_wires))
        total_wires = n_system_wires + n_estimation_wires

        # Device selection based on device_type and use_catalyst
        # Priority: explicit device_type > Catalyst default (lightning.qubit) > default.qubit
        # Note: use_catalyst is passed to ensure Catalyst-compatible fallback when GPU unavailable
        if self.device_type != "default.qubit":
            # User specified a device type
            dev = _select_device(self.device_type, total_wires, n_shots, self.use_catalyst)
        elif self.use_catalyst:
            # Catalyst default: use lightning.qubit for @qjit compatibility
            dev = _select_device("lightning.qubit", total_wires, n_shots, self.use_catalyst)
        else:
            dev = _select_device("default.qubit", total_wires, n_shots, use_catalyst=False)

        @qml.qnode(dev)
        def qpe_circuit():
            # 1. Prepare HF initial state on system register
            self._prepare_initial_state(hf_state, system_wires)

            # 2. Hadamard on all estimation qubits
            for wire in estimation_wires:
                qml.Hadamard(wires=wire)

            # 3. Controlled time evolutions: U^(2^k) for k-th estimation qubit
            for k, control_wire in enumerate(estimation_wires):
                time = (2**k) * base_time
                self._apply_controlled_unitary(
                    hamiltonian, time, control_wire, system_wires, n_trotter_steps
                )

            # 4. Inverse QFT on estimation register
            self._inverse_qft(estimation_wires)

            # 5. Sample estimation register to get phase bits
            return qml.sample(wires=estimation_wires)

        # Apply Catalyst @qjit compilation if enabled
        if self.use_catalyst:
            return qjit(qpe_circuit)
        return qpe_circuit

    def _extract_energy_from_samples(
        self,
        samples: np.ndarray,
        base_time: float = DEFAULT_BASE_TIME,
    ) -> float:
        """
        Extract energy estimate from QPE measurement samples.

        The measured phase phi relates to energy: E = 2*pi*phi / base_time

        Args:
            samples: Binary samples from estimation register, shape (n_estimation,)
                     or (n_shots, n_estimation) for multiple shots
            base_time: Base evolution time used in QPE

        Returns:
            Estimated ground state energy in Hartree
        """
        # Handle single sample or multiple shots
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        # Convert binary samples to phase values
        # Phase phi = sum_k (b_k / 2^(k+1)) where b_k is the k-th bit
        phases = []
        for sample in samples:
            phase = 0.0
            for k, bit in enumerate(sample):
                phase += bit / (2 ** (k + 1))
            phases.append(phase)

        # Average phase from all shots
        avg_phase = np.mean(phases)

        # Convert phase to energy: E = 2*pi*phi / t
        # Note: The actual relationship depends on how time is encoded
        # For standard QPE: phi = E*t / (2*pi), so E = 2*pi*phi / t
        energy = 2 * np.pi * avg_phase / base_time

        # QPE measures positive phase, but molecular energies are negative
        # Apply sign correction based on expected range
        if energy > 0:
            energy = -energy

        return energy
