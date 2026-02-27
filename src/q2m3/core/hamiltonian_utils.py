# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
PennyLane Hamiltonian operator utilities.

Provides helper functions for decomposing PennyLane Sum/Hamiltonian objects
(PennyLane 0.44+ compatible) and building operator index maps for efficient
coefficient updates in QM/MM simulations.
"""

import pennylane as qml


def decompose_hamiltonian(H) -> tuple[list[float], list]:
    """
    Extract (coeffs, ops) from PennyLane Sum/Hamiltonian.

    PennyLane 0.44+ returns Sum of SProd from qml.qchem.molecular_hamiltonian.
    We need separate coefficients and operators for qml.dot(coeffs, ops).

    Args:
        H: PennyLane Hamiltonian (Sum of SProd operators)

    Returns:
        Tuple of (coefficients, operators) where coefficients are floats
        and operators are PennyLane operator instances.
    """
    coeffs = []
    ops = []
    for op in H.operands:
        if isinstance(op, qml.ops.SProd):
            coeffs.append(float(op.scalar))
            ops.append(op.base)
        else:
            coeffs.append(1.0)
            ops.append(op)
    return coeffs, ops


def build_operator_index_map(
    ops: list,
    n_system_qubits: int,
    coeffs: list[float],
) -> tuple[dict, list[float], list]:
    """
    Scan ops to find Identity and single-Z(wire) indices for MM coefficient updates.

    If any Z(wire) for wire in range(n_system_qubits) is missing, appends a
    coeff=0.0 placeholder to ensure all spin orbitals can receive MM corrections.

    Args:
        ops: List of PennyLane operators (from decompose_hamiltonian)
        n_system_qubits: Number of system qubits (= 2 * active_orbitals)
        coeffs: Corresponding coefficient list (may be extended)

    Returns:
        Tuple of (index_map, extended_coeffs, extended_ops) where:
            - index_map: {"identity_idx": int, "z_wire_idx": {wire: idx, ...}}
            - extended_coeffs: coefficients with any missing Z terms appended
            - extended_ops: operators with any missing Z terms appended
    """
    coeffs = list(coeffs)
    ops = list(ops)

    identity_idx = None
    z_wire_idx = {}

    for i, op in enumerate(ops):
        # Detect multi-wire Identity (matches qml.Identity(wires=[0,1,...]))
        if isinstance(op, qml.Identity):
            identity_idx = i
        # Detect single-qubit PauliZ
        elif isinstance(op, qml.PauliZ) and len(op.wires) == 1:
            wire = op.wires[0]
            z_wire_idx[wire] = i

    # Ensure all spin-orbital wires have a Z entry
    for wire in range(n_system_qubits):
        if wire not in z_wire_idx:
            z_wire_idx[wire] = len(ops)
            coeffs.append(0.0)
            ops.append(qml.PauliZ(wires=wire))

    if identity_idx is None:
        raise ValueError("Hamiltonian has no Identity term — cannot apply MM corrections")

    index_map = {"identity_idx": identity_idx, "z_wire_idx": z_wire_idx}
    return index_map, coeffs, ops
