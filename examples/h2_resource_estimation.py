#!/usr/bin/env python3
# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
H2 EFTQC Resource Estimation: Vacuum vs Solvated

Demonstrates how MM electrostatic embedding affects quantum hardware
requirements using the q2m3.core resource estimation API.

Key insight: MM embedding only modifies 1-electron integrals (QM-MM
electrostatics). Resource estimates are dominated by 2-electron integrals,
which remain unchanged → minimal impact on EFTQC resources.

Usage:
    uv run python examples/h2_resource_estimation.py
"""

import numpy as np

from q2m3.core import compare_vacuum_solvated, estimate_resources

# H2 geometry (bond length 0.74 Angstrom)
H2_SYMBOLS = ["H", "H"]
H2_COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

# 2 TIP3P water molecules as MM environment (~3 Angstrom from H2)
MM_CHARGES = np.array([-0.834, 0.417, 0.417, -0.834, 0.417, 0.417])
MM_COORDS = np.array(
    [
        [3.0, 0.0, 0.0],
        [3.5, 0.8, 0.0],
        [3.5, -0.8, 0.0],  # Water 1
        [-3.0, 0.0, 0.0],
        [-3.5, 0.8, 0.0],
        [-3.5, -0.8, 0.0],  # Water 2
    ]
)


def main():
    print("=" * 60)
    print("  H2 EFTQC Resource Estimation: Vacuum vs Solvated")
    print("=" * 60)

    # Single system estimation (vacuum)
    print("\n[Step 1] Vacuum Resource Estimation")
    print("-" * 60)
    vacuum = estimate_resources(symbols=H2_SYMBOLS, coords=H2_COORDS, basis="sto-3g")
    print(f"  Logical qubits:    {vacuum.logical_qubits}")
    print(f"  Toffoli gates:     {vacuum.toffoli_gates:,}")
    print(f"  Hamiltonian λ:     {vacuum.hamiltonian_1norm:.4f} Ha")
    print(f"  System qubits:     {vacuum.n_system_qubits} (JW: 2 per spatial orbital)")
    print(
        f"  Target error:      {vacuum.target_error:.4f} Ha "
        f"({vacuum.target_error * 627.5:.2f} kcal/mol)"
    )

    # Comparative analysis (vacuum vs solvated)
    print("\n[Step 2] Vacuum vs Solvated Comparison")
    print("-" * 60)
    comparison = compare_vacuum_solvated(
        symbols=H2_SYMBOLS,
        coords=H2_COORDS,
        mm_charges=MM_CHARGES,
        mm_coords=MM_COORDS,
    )
    vac = comparison.vacuum
    sol = comparison.solvated

    print(f"  {'Metric':<25} {'Vacuum':<15} {'Solvated':<15} {'Delta':<10}")
    print(f"  {'-' * 65}")
    print(
        f"  {'Logical qubits':<25} {vac.logical_qubits:<15} "
        f"{sol.logical_qubits:<15} {'same':<10}"
    )
    print(
        f"  {'Toffoli gates':<25} {vac.toffoli_gates:<15,} "
        f"{sol.toffoli_gates:<15,} {comparison.delta_gates_percent:+.1f}%"
    )
    print(
        f"  {'Hamiltonian λ (Ha)':<25} {vac.hamiltonian_1norm:<15.4f} "
        f"{sol.hamiltonian_1norm:<15.4f} {comparison.delta_lambda_percent:+.1f}%"
    )
    print()
    print(f"  MM charges: {sol.n_mm_charges} point charges (2 TIP3P waters)")
    print()
    print("  Conclusion: MM embedding has minimal impact on EFTQC resources.")
    print("  Resource requirements are dominated by 2-electron integrals,")
    print("  which are unchanged by QM-MM electrostatic embedding.")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
