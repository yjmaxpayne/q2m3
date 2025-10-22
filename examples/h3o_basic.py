#!/usr/bin/env python
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Basic example: H3O+ ground state calculation with QPE.
"""

import numpy as np

from q2m3.core import QuantumQMMM
from q2m3.core.qmmm_system import Atom
from q2m3.utils import save_json_results


def main():
    """Run basic H3O+ QM/MM calculation."""

    # Define H3O+ geometry (optimized structure)
    h3o_atoms = [
        Atom("O", np.array([0.000000, 0.000000, 0.000000]), charge=-2.0),
        Atom("H", np.array([0.960000, 0.000000, 0.000000]), charge=1.0),
        Atom("H", np.array([-0.480000, 0.831384, 0.000000]), charge=1.0),
        Atom("H", np.array([-0.480000, -0.831384, 0.000000]), charge=1.0),
    ]

    # Configure QPE parameters
    qpe_config = {
        "algorithm": "iterative",
        "iterations": 8,
        "mapping": "jordan_wigner",
        "system_qubits": 12,
        "error_tolerance": 0.005,
    }

    print("Initializing Quantum-QM/MM system...")
    print(f"QM region: H3O+ ({len(h3o_atoms)} atoms)")
    print("MM region: 8 TIP3P water molecules")
    print(
        f"QPE configuration: {qpe_config['iterations']} iterations, {qpe_config['system_qubits']} qubits"
    )

    # Initialize calculator
    qmmm = QuantumQMMM(qm_atoms=h3o_atoms, mm_waters=8, qpe_config=qpe_config)

    print("\nStarting QPE calculation...")
    print("Note: This is a POC implementation - actual QPE execution not yet implemented")

    # Run calculation (placeholder for now)
    results = qmmm.compute_ground_state()

    # Display results
    print("\n" + "=" * 50)
    print("RESULTS (Placeholder)")
    print("=" * 50)
    print(f"Ground State Energy: {results['energy']:.6f} Hartree")
    print(f"Convergence: {'Yes' if results['convergence']['converged'] else 'No'}")
    print(f"Iterations used: {results['convergence']['iterations']}")

    # Save results
    output_file = "data/output/h3o_qpe_results.json"
    save_json_results(results, output_file)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
