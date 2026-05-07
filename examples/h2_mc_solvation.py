#!/usr/bin/env python3
# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
H2 + TIP3P Water: MC Solvation with QPE Validation

Demonstrates the q2m3.solvation module for hybrid quantum-classical
QM/MM Monte Carlo solvation simulations.

Features:
    - QJIT-compiled MC loop with Catalyst
    - Pre-compiled QPE circuit (fixed mode)
    - LLVM IR caching (Phase A/B: first run ~24s, subsequent ~5s)
    - Rich console output with timing statistics
    - Energy trajectory plotting
"""

import warnings

warnings.filterwarnings("ignore")

from q2m3.solvation import (
    MoleculeConfig,
    QPEConfig,
    SolvationConfig,
    run_solvation,
)

# =============================================================================
# H2 Molecule Configuration
# =============================================================================

H2_MOLECULE = MoleculeConfig(
    name="H2",
    symbols=["H", "H"],
    coords=[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.74],  # 0.74 Angstrom bond length
    ],
    charge=0,
    active_electrons=2,
    active_orbitals=2,
    basis="sto-3g",
)


# =============================================================================
# Main
# =============================================================================


def main():
    """Run H2 MC solvation simulation with QPE validation."""
    # Configure simulation
    config = SolvationConfig(
        molecule=H2_MOLECULE,
        qpe_config=QPEConfig(
            n_estimation_wires=4,
            n_trotter_steps=10,
            n_shots=50,
            qpe_interval=10,
            target_resolution=0.003,
            energy_range=0.2,
        ),
        hamiltonian_mode="fixed",
        n_waters=10,
        n_mc_steps=100,
        temperature=300.0,
        translation_step=0.3,
        rotation_step=0.2618,  # ~15 degrees
        initial_water_distance=4.0,
        random_seed=42,
        verbose=True,
    )

    # Run simulation (IR cache auto-enabled: first run compiles, subsequent use cache)
    result = run_solvation(config, show_plots=False)

    # Print cache status
    cache = result.get("cache_stats", {})
    if cache.get("is_cache_hit"):
        print(f"\nIR Cache: HIT (Phase B: {cache.get('phase_b_time_s', 0):.2f}s)")
    else:
        print(f"\nIR Cache: MISS (Phase A: {cache.get('phase_a_time_s', 0):.1f}s)")

    return result


if __name__ == "__main__":
    main()
