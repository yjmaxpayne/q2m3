# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Reduced Density Matrix (RDM) estimation from quantum states.

Implements 1-RDM measurement using Pauli expectation values after
Jordan-Wigner transformation of fermionic operators.

Key formulas:
- Diagonal: γ_pp = <a_p†a_p> = (1 - <Z_p>) / 2
- Off-diagonal (p < q): γ_pq = 1/4 * [<X_p Z... X_q> + <Y_p Z... Y_q>
                                       + i(<X_p Z... Y_q> - <Y_p Z... X_q>)]

Performance optimization (Phase 7):
- Batched measurement: single QNode returns all observables (120x → 1x execution)
- GPU acceleration via device_utils
- Catalyst @qjit support
"""

from typing import Any

import numpy as np
import pennylane as qml

# Import shared device selection
from .device_utils import select_device

# Optional Catalyst import with graceful degradation
try:
    from catalyst import qjit

    HAS_CATALYST = True
except ImportError:
    HAS_CATALYST = False

    def qjit(fn=None, **kwargs):
        """No-op fallback when Catalyst is not installed."""
        return fn if fn else lambda f: f


# Default configuration for RDM measurement
DEFAULT_RDM_CONFIG = {
    "n_shots": 1000,
    "include_off_diagonal": True,
    "symmetrize": True,
    "output_basis": "spin",  # "spin" or "spatial"
}


class RDMEstimator:
    """
    1-RDM measurement from quantum states using Pauli expectation values.

    Measures the one-particle reduced density matrix (1-RDM) from a quantum
    state prepared by Trotter time evolution. Uses Jordan-Wigner transformation
    to convert fermionic operators to Pauli observables.

    Performance optimization: Uses batched measurement (single QNode) instead
    of individual QNodes per observable for 10-50x speedup.
    """

    def __init__(
        self,
        n_qubits: int,
        n_electrons: int,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize RDM estimator.

        Args:
            n_qubits: Number of spin orbitals (qubits in Jordan-Wigner)
            n_electrons: Number of electrons in the system
            config: Optional configuration dictionary with keys:
                - n_shots: Number of measurement shots (default: 1000)
                - include_off_diagonal: Measure off-diagonal elements (default: True)
                - symmetrize: Enforce Hermitian symmetry (default: True)
                - output_basis: "spin" or "spatial" (default: "spin")
        """
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.config = {**DEFAULT_RDM_CONFIG, **(config or {})}

    def _build_all_observables(self) -> list:
        """
        Pre-build all observables for batched measurement.

        Returns:
            List of PennyLane observables in order:
            - Diagonal elements: Z_0, Z_1, ..., Z_{n-1}
            - Off-diagonal elements (if enabled): XX, YY, XY, YX for each pair
        """
        observables = []

        # Diagonal elements: Z_p for occupation numbers
        for p in range(self.n_qubits):
            observables.append(qml.Z(p))

        # Off-diagonal elements: XX, YY, XY, YX Pauli strings
        if self.config.get("include_off_diagonal", True):
            for p in range(self.n_qubits):
                for q in range(p + 1, self.n_qubits):
                    # Jordan-Wigner Z-string between p and q
                    z_string_wires = list(range(p + 1, q))
                    observables.extend(
                        self._build_offdiag_observables(p, q, z_string_wires)
                    )

        return observables

    def _build_offdiag_observables(
        self, p: int, q: int, z_string_wires: list[int]
    ) -> list:
        """
        Build 4 Pauli observables for off-diagonal 1-RDM element γ_pq.

        Args:
            p: First orbital index (p < q)
            q: Second orbital index
            z_string_wires: Wire indices for Z-string between p and q

        Returns:
            List of [XX_term, YY_term, XY_term, YX_term]
        """

        def build_pauli_string(op_p, op_q):
            """Build tensor product: op_p @ Z_{p+1} @ ... @ Z_{q-1} @ op_q"""
            ops = [op_p(p)]
            for wire in z_string_wires:
                ops.append(qml.Z(wire))
            ops.append(op_q(q))

            if len(ops) == 1:
                return ops[0]
            return qml.prod(*ops)

        return [
            build_pauli_string(qml.X, qml.X),  # XX term (real part)
            build_pauli_string(qml.Y, qml.Y),  # YY term (real part)
            build_pauli_string(qml.X, qml.Y),  # XY term (imag part)
            build_pauli_string(qml.Y, qml.X),  # YX term (imag part)
        ]

    def _reconstruct_rdm_from_results(self, results: tuple | list) -> np.ndarray:
        """
        Reconstruct RDM matrix from batched measurement results.

        Args:
            results: Tuple/list of expectation values from batched measurement

        Returns:
            1-RDM as numpy array of shape (n_qubits, n_qubits)
        """
        rdm = np.zeros((self.n_qubits, self.n_qubits), dtype=complex)

        idx = 0
        # Diagonal elements: γ_pp = (1 - <Z_p>) / 2
        for p in range(self.n_qubits):
            rdm[p, p] = (1 - results[idx]) / 2
            idx += 1

        # Off-diagonal elements
        if self.config.get("include_off_diagonal", True):
            for p in range(self.n_qubits):
                for q in range(p + 1, self.n_qubits):
                    # Extract 4 Pauli expectation values
                    xx_val = results[idx]
                    yy_val = results[idx + 1]
                    xy_val = results[idx + 2]
                    yx_val = results[idx + 3]
                    idx += 4

                    # γ_pq = 1/4 * [<XX> + <YY> + i(<XY> - <YX>)]
                    real_part = (xx_val + yy_val) / 4
                    imag_part = (xy_val - yx_val) / 4
                    rdm[p, q] = real_part + 1j * imag_part

                    # Hermitian conjugate: γ_qp = γ_pq*
                    rdm[q, p] = real_part - 1j * imag_part

        return rdm

    def measure_1rdm(
        self,
        hamiltonian: qml.Hamiltonian,
        hf_state: np.ndarray,
        base_time: float,
        n_trotter_steps: int,
        device_type: str = "default.qubit",
        use_catalyst: bool = False,
    ) -> np.ndarray:
        """
        Measure 1-RDM from Trotter-evolved state using batched measurement.

        Prepares |HF⟩, applies TrotterProduct time evolution, then measures
        all 1-RDM elements as Pauli expectation values in a single QNode.

        Args:
            hamiltonian: PennyLane molecular Hamiltonian
            hf_state: Hartree-Fock reference state binary array
            base_time: Evolution time for Trotter
            n_trotter_steps: Number of Trotter steps
            device_type: Device selection strategy:
                - "auto": Auto-select best (GPU > lightning.qubit > default.qubit)
                - "default.qubit": Standard PennyLane simulator
                - "lightning.qubit": High-performance CPU simulator
                - "lightning.gpu": GPU-accelerated simulator
            use_catalyst: Enable Catalyst @qjit compilation

        Returns:
            1-RDM as numpy array of shape (n_qubits, n_qubits)
        """
        # Pre-build all observables for batched measurement
        all_observables = self._build_all_observables()

        # Resolve Catalyst flag
        actual_use_catalyst = use_catalyst and HAS_CATALYST

        # Select device using shared device_utils
        dev = select_device(
            device_type, n_wires=self.n_qubits, shots=None, use_catalyst=actual_use_catalyst
        )

        # Build batched measurement QNode (single circuit, multiple returns)
        @qml.qnode(dev)
        def measure_all_observables():
            # Prepare HF state
            qml.BasisState(hf_state, wires=range(self.n_qubits))
            # Time evolution
            qml.TrotterProduct(hamiltonian, base_time, n=n_trotter_steps, order=2)
            # Return all expectation values as tuple
            return tuple(qml.expval(obs) for obs in all_observables)

        # Apply Catalyst @qjit compilation if enabled
        if actual_use_catalyst:
            measure_all_observables = qjit(measure_all_observables)

        # Single execution for all observables
        results = measure_all_observables()

        # Reconstruct RDM matrix from results
        rdm = self._reconstruct_rdm_from_results(results)

        # Apply physical constraints if requested
        if self.config.get("symmetrize", True):
            rdm = self.enforce_physical_constraints(rdm)

        return rdm

    def build_rdm_observables(self) -> dict[tuple[int, int], Any]:
        """
        Build Pauli observables for all 1-RDM elements (legacy interface).

        Returns:
            Dictionary mapping (p, q) indices to PennyLane observables.
            For p == q: number operator observable
            For p < q: tuple of 4 Pauli observables for real/imag parts
        """
        observables = {}

        for p in range(self.n_qubits):
            observables[(p, p)] = qml.Z(p)

            if self.config["include_off_diagonal"]:
                for q in range(p + 1, self.n_qubits):
                    z_string_wires = list(range(p + 1, q))
                    obs_list = self._build_offdiag_observables(p, q, z_string_wires)
                    observables[(p, q)] = tuple(obs_list)

        return observables

    def enforce_physical_constraints(self, rdm_raw: np.ndarray) -> np.ndarray:
        """
        Enforce physical constraints on 1-RDM.

        Constraints:
        1. Hermiticity: γ_pq = γ_qp*
        2. Positive semi-definiteness: all eigenvalues >= 0
        3. Trace normalization: Tr(γ) = N_electrons

        Args:
            rdm_raw: Raw 1-RDM from measurement

        Returns:
            Physically valid 1-RDM
        """
        # 1. Enforce Hermiticity
        rdm = (rdm_raw + rdm_raw.conj().T) / 2

        # 2. Enforce positive semi-definiteness
        eigenvalues, eigenvectors = np.linalg.eigh(rdm)
        eigenvalues = np.maximum(eigenvalues, 0)  # Truncate negative eigenvalues

        # 3. Enforce trace normalization
        current_trace = np.sum(eigenvalues)
        if current_trace > 1e-10:
            eigenvalues = eigenvalues * self.n_electrons / current_trace

        # Reconstruct RDM
        rdm_physical = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T

        return rdm_physical.real  # 1-RDM should be real for real Hamiltonians

    def spin_to_spatial_rdm(self, spin_rdm: np.ndarray) -> np.ndarray:
        """
        Convert spin-orbital 1-RDM to spatial-orbital 1-RDM.

        For restricted systems, spatial RDM = sum of alpha and beta spin blocks.

        Args:
            spin_rdm: 1-RDM in spin-orbital basis, shape (2*n_spatial, 2*n_spatial)

        Returns:
            1-RDM in spatial-orbital basis, shape (n_spatial, n_spatial)
        """
        n_spin = spin_rdm.shape[0]
        n_spatial = n_spin // 2

        # Alpha block: indices 0 to n_spatial-1
        # Beta block: indices n_spatial to 2*n_spatial-1
        alpha_rdm = spin_rdm[:n_spatial, :n_spatial]
        beta_rdm = spin_rdm[n_spatial:, n_spatial:]

        # Spatial RDM = alpha + beta (for restricted systems)
        spatial_rdm = alpha_rdm + beta_rdm

        return spatial_rdm.real


def measure_rdm_from_qpe_state(
    hamiltonian: qml.Hamiltonian,
    hf_state: np.ndarray,
    n_electrons: int,
    base_time: float,
    n_trotter_steps: int,
    config: dict[str, Any] | None = None,
    device_type: str = "default.qubit",
    use_catalyst: bool = False,
) -> np.ndarray:
    """
    Convenience function to measure 1-RDM from QPE-like state.

    Args:
        hamiltonian: PennyLane molecular Hamiltonian
        hf_state: Hartree-Fock reference state
        n_electrons: Number of electrons
        base_time: Evolution time
        n_trotter_steps: Trotter steps
        config: Optional RDM configuration
        device_type: PennyLane device selection
        use_catalyst: Enable Catalyst @qjit compilation

    Returns:
        1-RDM numpy array
    """
    n_qubits = len(hf_state)
    estimator = RDMEstimator(n_qubits, n_electrons, config)
    return estimator.measure_1rdm(
        hamiltonian, hf_state, base_time, n_trotter_steps, device_type, use_catalyst
    )
