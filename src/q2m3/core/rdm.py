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
"""

from typing import Any

import numpy as np
import pennylane as qml


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

    def build_rdm_observables(self) -> dict[tuple[int, int], Any]:
        """
        Build Pauli observables for all 1-RDM elements.

        Returns:
            Dictionary mapping (p, q) indices to PennyLane observables.
            For p == q: number operator observable
            For p < q: tuple of 4 Pauli observables for real/imag parts
        """
        observables = {}

        for p in range(self.n_qubits):
            # Diagonal elements: γ_pp = <a_p†a_p> = (1 - <Z_p>) / 2
            # We just need to measure Z_p, then compute (1 - <Z_p>) / 2
            observables[(p, p)] = qml.Z(p)

            if self.config["include_off_diagonal"]:
                for q in range(p + 1, self.n_qubits):
                    # Off-diagonal: need 4 Pauli string measurements
                    # γ_pq = 1/4 * [<X_p Z... X_q> + <Y_p Z... Y_q>
                    #               + i(<X_p Z... Y_q> - <Y_p Z... X_q>)]
                    observables[(p, q)] = self._build_offdiag_observable(p, q)

        return observables

    def _build_offdiag_observable(self, p: int, q: int) -> tuple:
        """
        Build Pauli observables for off-diagonal 1-RDM element γ_pq.

        For p < q, need to measure 4 Pauli strings:
        1. X_p @ Z_{p+1} @ ... @ Z_{q-1} @ X_q  (real part)
        2. Y_p @ Z_{p+1} @ ... @ Z_{q-1} @ Y_q  (real part)
        3. X_p @ Z_{p+1} @ ... @ Z_{q-1} @ Y_q  (imaginary part)
        4. Y_p @ Z_{p+1} @ ... @ Z_{q-1} @ X_q  (imaginary part)

        Args:
            p: First orbital index (p < q)
            q: Second orbital index

        Returns:
            Tuple of 4 PennyLane observables (XX_term, YY_term, XY_term, YX_term)
        """
        # Build Jordan-Wigner Z-string between p and q
        z_string_wires = list(range(p + 1, q))

        def build_pauli_string(op_p, op_q):
            """Build tensor product: op_p @ Z_{p+1} @ ... @ Z_{q-1} @ op_q"""
            ops = [op_p(p)]
            for wire in z_string_wires:
                ops.append(qml.Z(wire))
            ops.append(op_q(q))

            # Combine into single observable using prod
            if len(ops) == 1:
                return ops[0]
            return qml.prod(*ops)

        # Four Pauli strings needed for off-diagonal element
        xx_term = build_pauli_string(qml.X, qml.X)
        yy_term = build_pauli_string(qml.Y, qml.Y)
        xy_term = build_pauli_string(qml.X, qml.Y)
        yx_term = build_pauli_string(qml.Y, qml.X)

        return (xx_term, yy_term, xy_term, yx_term)

    def measure_1rdm(
        self,
        hamiltonian: qml.Hamiltonian,
        hf_state: np.ndarray,
        base_time: float,
        n_trotter_steps: int,
        device_type: str = "default.qubit",
    ) -> np.ndarray:
        """
        Measure 1-RDM from Trotter-evolved state.

        Prepares |HF⟩, applies TrotterProduct time evolution, then measures
        all 1-RDM elements as Pauli expectation values.

        Args:
            hamiltonian: PennyLane molecular Hamiltonian
            hf_state: Hartree-Fock reference state binary array
            base_time: Evolution time for Trotter
            n_trotter_steps: Number of Trotter steps
            device_type: PennyLane device to use

        Returns:
            1-RDM as numpy array of shape (n_qubits, n_qubits)
        """
        observables = self.build_rdm_observables()
        n_shots = self.config["n_shots"]

        # Select device
        dev = qml.device(device_type, wires=self.n_qubits)

        # Build measurement results dictionary
        rdm = np.zeros((self.n_qubits, self.n_qubits), dtype=complex)

        # Measure diagonal elements
        for p in range(self.n_qubits):
            z_observable = observables[(p, p)]

            @qml.qnode(dev)
            def measure_diagonal():
                # Prepare HF state
                qml.BasisState(hf_state, wires=range(self.n_qubits))
                # Time evolution
                qml.TrotterProduct(hamiltonian, base_time, n=n_trotter_steps, order=2)
                return qml.expval(z_observable)

            z_expval = measure_diagonal()
            # γ_pp = (1 - <Z_p>) / 2
            rdm[p, p] = (1 - z_expval) / 2

        # Measure off-diagonal elements
        if self.config["include_off_diagonal"]:
            for p in range(self.n_qubits):
                for q in range(p + 1, self.n_qubits):
                    xx_term, yy_term, xy_term, yx_term = observables[(p, q)]

                    # Measure all 4 Pauli strings
                    @qml.qnode(dev)
                    def measure_xx():
                        qml.BasisState(hf_state, wires=range(self.n_qubits))
                        qml.TrotterProduct(hamiltonian, base_time, n=n_trotter_steps, order=2)
                        return qml.expval(xx_term)

                    @qml.qnode(dev)
                    def measure_yy():
                        qml.BasisState(hf_state, wires=range(self.n_qubits))
                        qml.TrotterProduct(hamiltonian, base_time, n=n_trotter_steps, order=2)
                        return qml.expval(yy_term)

                    @qml.qnode(dev)
                    def measure_xy():
                        qml.BasisState(hf_state, wires=range(self.n_qubits))
                        qml.TrotterProduct(hamiltonian, base_time, n=n_trotter_steps, order=2)
                        return qml.expval(xy_term)

                    @qml.qnode(dev)
                    def measure_yx():
                        qml.BasisState(hf_state, wires=range(self.n_qubits))
                        qml.TrotterProduct(hamiltonian, base_time, n=n_trotter_steps, order=2)
                        return qml.expval(yx_term)

                    xx_val = measure_xx()
                    yy_val = measure_yy()
                    xy_val = measure_xy()
                    yx_val = measure_yx()

                    # γ_pq = 1/4 * [<XX> + <YY> + i(<XY> - <YX>)]
                    real_part = (xx_val + yy_val) / 4
                    imag_part = (xy_val - yx_val) / 4
                    rdm[p, q] = real_part + 1j * imag_part

                    # Hermitian conjugate: γ_qp = γ_pq*
                    rdm[q, p] = real_part - 1j * imag_part

        # Apply physical constraints if requested
        if self.config["symmetrize"]:
            rdm = self.enforce_physical_constraints(rdm)

        return rdm

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

        # In Jordan-Wigner, spin orbitals are typically ordered as:
        # [alpha_0, beta_0, alpha_1, beta_1, ...] or
        # [alpha_0, alpha_1, ..., beta_0, beta_1, ...]
        # Assuming the second convention (PennyLane default):

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
        device_type: PennyLane device

    Returns:
        1-RDM numpy array
    """
    n_qubits = len(hf_state)
    estimator = RDMEstimator(n_qubits, n_electrons, config)
    return estimator.measure_1rdm(
        hamiltonian, hf_state, base_time, n_trotter_steps, device_type
    )
