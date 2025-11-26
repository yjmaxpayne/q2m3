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
from .device_utils import HAS_LIGHTNING_GPU, HAS_LIGHTNING_QUBIT
from .device_utils import select_device as _select_device

# Constants for QPE configuration
DEFAULT_ERROR_ESTIMATE = 0.001
MAX_EARLY_CONVERGENCE_ITERATIONS = 5
DEFAULT_N_ESTIMATION_WIRES = 4
DEFAULT_BASE_TIME = 0.3
DEFAULT_N_TROTTER_STEPS = 5

# Phase overflow safety margin (use 80% of max allowed time to avoid edge cases)
PHASE_SAFETY_MARGIN = 0.8


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
        device_type: str = "lightning.qubit",
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
                - "lightning.qubit": High-performance CPU simulator (default)
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

        WORKAROUND: Uses explicit X gates instead of qml.BasisState due to
        Catalyst @qjit compatibility issue. When qml.BasisState is used with
        qml.ctrl(qml.adjoint(TrotterProduct)), Catalyst produces incorrect
        results (uniform distribution instead of peaked phase estimation).

        Root cause: Catalyst's set_basis_state_p primitive lacks control wire
        support, causing circuit compilation errors when combined with controlled
        operations in the same circuit.

        See: https://github.com/PennyLaneAI/catalyst/blob/main/frontend/catalyst/jax_primitives.py

        Args:
            hf_state: Binary array from qml.qchem.hf_state(), e.g., [1,1,0,0] for H2
            wires: System qubit indices to prepare the state on
        """
        # Apply X gates for occupied orbitals (state=1)
        # This is equivalent to BasisState but compatible with Catalyst @qjit
        for wire, state in zip(wires, hf_state):
            if state == 1:
                qml.PauliX(wires=wire)

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

        IMPORTANT: PennyLane's TrotterProduct implements exp(+iHt), not exp(-iHt).
        We use qml.adjoint() to get the correct sign convention for QPE:
            adjoint(TrotterProduct) = exp(-iHt)

        Args:
            hamiltonian: PennyLane molecular Hamiltonian
            time: Evolution time (typically 2^k * base_time)
            control_wire: Estimation qubit for control
            target_wires: System qubit indices
            n_trotter_steps: Number of Trotter steps for decomposition
        """
        # TrotterProduct implements exp(+iHt), but QPE needs exp(-iHt)
        # Use adjoint to flip the sign: adjoint(exp(+iHt)) = exp(-iHt)
        qml.ctrl(
            qml.adjoint(qml.TrotterProduct(hamiltonian, time, n=n_trotter_steps, order=2)),
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

            # 3. Controlled time evolutions
            # PennyLane QFT convention: qubit k should have phase 2^(n-1-k) * phi
            # This means qubit 0 (first estimation wire) controls U^(2^(n-1))
            # and qubit n-1 (last estimation wire) controls U^(2^0)
            for k, control_wire in enumerate(estimation_wires):
                time = (2 ** (n_estimation_wires - 1 - k)) * base_time
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

    @staticmethod
    def compute_optimal_base_time(energy_estimate: float, safety_margin: float = None) -> float:
        """
        Compute optimal base_time to avoid phase overflow in QPE.

        For QPE, the relationship is: phi = |E| * t / (2*pi)
        To avoid overflow, we need phi < 1, i.e., t < 2*pi / |E|

        Args:
            energy_estimate: Estimated ground state energy (e.g., HF energy)
            safety_margin: Safety factor (default: PHASE_SAFETY_MARGIN = 0.8)

        Returns:
            Optimal base_time that avoids phase overflow
        """
        if safety_margin is None:
            safety_margin = PHASE_SAFETY_MARGIN

        if abs(energy_estimate) < 1e-10:
            # Avoid division by zero for near-zero energies
            return DEFAULT_BASE_TIME

        # t_max = 2*pi / |E|, apply safety margin
        t_max = 2 * np.pi / abs(energy_estimate)
        return t_max * safety_margin

    def _extract_energy_from_samples(
        self,
        samples: np.ndarray,
        base_time: float = DEFAULT_BASE_TIME,
    ) -> float:
        """
        Extract energy estimate from QPE measurement samples.

        Physics derivation:
        - QPE measures eigenphase of U = exp(-iHt)
        - If H|ψ> = E|ψ>, then U|ψ> = exp(-iEt)|ψ> = exp(i*2π*φ)|ψ>
        - Therefore: -Et = 2πφ (mod 2π), which gives φ = -Et/(2π) mod 1
        - Inverting: E = -2πφ/t

        For molecular systems with negative energies (E < 0):
        - φ = -Et/(2π) = |E|t/(2π) > 0 (positive phase)
        - E = -2πφ/t (gives negative energy as expected)

        Note: QPE uses MODE (most frequent sample) not average, because
        the eigenphase should appear most frequently in repeated measurements.

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

        # Convert to numpy array to handle both regular arrays and JAX arrays
        # (Catalyst @qjit returns JAX arrays which need special handling)
        samples = np.asarray(samples, dtype=np.int64)

        # Convert binary samples to integer phase indices for mode calculation
        # In _build_standard_qpe_circuit:
        #   estimation_wires[0] controls U^(2^(n-1)) → MSB
        #   estimation_wires[n-1] controls U^(2^0) → LSB
        # qml.sample returns samples in wire order, so:
        #   sample[0] = MSB (2^(n-1)), sample[n-1] = LSB (2^0)
        n_bits = samples.shape[1]
        phase_indices = []
        for sample in samples:
            # Convert binary to integer with MSB-first ordering:
            # [1,1,0,1] -> 1*8 + 1*4 + 0*2 + 1*1 = 13
            idx = sum(int(bit) * (2 ** (n_bits - 1 - k)) for k, bit in enumerate(sample))
            phase_indices.append(idx)

        # Find the mode (most frequent phase index)
        from collections import Counter

        counter = Counter(phase_indices)
        mode_idx, mode_count = counter.most_common(1)[0]

        # Convert mode index to phase: phi = m / 2^n
        # This is the standard QPE formula where m is the measured integer
        # and 2^n is the number of estimation qubits.
        mode_phase = mode_idx / (2**n_bits)

        # Convert phase to energy: E = -2*pi*phi/t
        # Physics: U|psi> = exp(-iEt)|psi> = exp(2*pi*i*phi)|psi>
        # So -Et = 2*pi*phi => phi = -Et/(2*pi) => E = -2*pi*phi/t
        # For bound states (E < 0), phi > 0
        energy = -2 * np.pi * mode_phase / base_time

        return energy

    def draw_qpe_circuit(
        self,
        hamiltonian: qml.Hamiltonian,
        hf_state: np.ndarray,
        n_estimation_wires: int = DEFAULT_N_ESTIMATION_WIRES,
        base_time: float = DEFAULT_BASE_TIME,
        n_trotter_steps: int = DEFAULT_N_TROTTER_STEPS,
        max_display_wires: int = 8,
        use_pennylane_draw: bool = True,
    ) -> str:
        """
        Generate text visualization of QPE circuit structure.

        Uses qml.draw() with decimals=None and level=0 for clean output,
        showing circuit structure without parameter clutter.

        Args:
            hamiltonian: PennyLane molecular Hamiltonian
            hf_state: HF reference state binary array
            n_estimation_wires: Number of estimation qubits
            base_time: Base evolution time
            n_trotter_steps: Trotter steps per evolution
            max_display_wires: Max wires for full circuit display (default: 8)
            use_pennylane_draw: Use qml.draw() for visualization (default: True)

        Returns:
            String representation of the circuit
        """
        n_system_wires = len(hf_state)
        system_wires = list(range(n_system_wires))
        estimation_wires = list(range(n_system_wires, n_system_wires + n_estimation_wires))
        total_wires = n_system_wires + n_estimation_wires

        if not use_pennylane_draw:
            return self._generate_qpe_structure_diagram(
                n_system_wires, n_estimation_wires, base_time, n_trotter_steps
            )

        # Create QNode for visualization with decimals=None to hide parameters
        dev = qml.device("default.qubit", wires=total_wires)

        @qml.qnode(dev)
        def qpe_circuit_for_draw():
            # 1. Prepare HF initial state
            qml.BasisState(hf_state, wires=system_wires)

            # 2. Hadamard on estimation qubits
            for wire in estimation_wires:
                qml.Hadamard(wires=wire)

            # 3. Controlled time evolutions (show first 2 for clarity)
            # Use adjoint(TrotterProduct) for exp(-iHt) as in actual QPE circuit
            n_show = min(2, n_estimation_wires)
            for k in range(n_show):
                control_wire = estimation_wires[k]
                time = (2**k) * base_time
                qml.ctrl(
                    qml.adjoint(qml.TrotterProduct(hamiltonian, time, n=n_trotter_steps, order=2)),
                    control=control_wire,
                )

            # 4. Inverse QFT
            qml.adjoint(qml.QFT)(wires=estimation_wires)

            return qml.sample(wires=estimation_wires)

        # Generate clean circuit drawing with key parameters:
        # - decimals=None: omit all parameter values for cleaner display
        # - level=0 ("top"): show high-level circuit without decomposition
        # - max_length=80: reasonable terminal width
        try:
            circuit_str = qml.draw(
                qpe_circuit_for_draw,
                decimals=None,  # Hide parameter values
                level=0,  # No decomposition, show high-level ops
                max_length=80,
                show_matrices=False,
            )()
            # Add structure summary
            summary = self._generate_qpe_structure_diagram(
                n_system_wires, n_estimation_wires, base_time, n_trotter_steps
            )
            return f"PennyLane Circuit (decimals=None, level=0):\n{circuit_str}\n\n{summary}"
        except Exception:
            return self._generate_qpe_structure_diagram(
                n_system_wires, n_estimation_wires, base_time, n_trotter_steps
            )

    def _generate_qpe_structure_diagram(
        self,
        n_system_wires: int,
        n_estimation_wires: int,
        base_time: float,
        n_trotter_steps: int,
    ) -> str:
        """
        Generate ASCII structure diagram for QPE circuit.

        Returns:
            ASCII art representation of QPE structure
        """
        lines = []
        lines.append("QPE Circuit Structure:")
        lines.append("")
        lines.append("Estimation Register (phase readout):")
        for k in range(n_estimation_wires):
            ctrl_symbol = f"●──U^{2**k}"
            lines.append(f"  |0⟩ ──H──{ctrl_symbol:>10}──┤QFT†├── Sample")
        lines.append("")
        lines.append("System Register (molecular state):")
        lines.append(f"  |HF⟩ ({n_system_wires} qubits) ────────────────────────")
        lines.append("")
        lines.append("Where:")
        lines.append(f"  U = exp(-iHt), t₀ = {base_time}")
        lines.append(f"  Trotter steps: {n_trotter_steps}")
        lines.append(f"  Total qubits: {n_system_wires + n_estimation_wires}")
        return "\n".join(lines)
