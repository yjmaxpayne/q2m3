#!/usr/bin/env python3
# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Full one-electron fixed-MO embedding tutorial.

This script demonstrates the public q2m3 resource-estimation API for an H2
active space in vacuum and near a single TIP3P water shell. It compares:

- vacuum: no MM point charges
- diagonal: only the diagonal active-space Delta h_pp terms
- full_oneelectron: the full fixed-MO active-space Delta h_pq matrix

The output is an EFTQC resource-planning table. It is not a relaxed solvation
energy calculation.

Usage:
    uv run python examples/full_oneelectron_embedding.py
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from q2m3.core import estimate_resources

H2_SYMBOLS = ["H", "H"]
H2_COORDS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.74],
    ]
)

# One TIP3P water placed outside the H2 bonding region. Coordinates are in
# Angstrom and charges are in elementary-charge units.
TIP3P_CHARGES = np.array([-0.834, 0.417, 0.417])
TIP3P_COORDS = np.array(
    [
        [3.0, 0.0, 0.0],
        [3.5, 0.8, 0.0],
        [3.5, -0.8, 0.0],
    ]
)


def main() -> None:
    """Run the full one-electron embedding resource-estimation tutorial."""
    common = dict(
        symbols=H2_SYMBOLS,
        coords=H2_COORDS,
        basis="sto-3g",
        active_electrons=2,
        active_orbitals=2,
    )

    vacuum = estimate_resources(**common)
    diagonal = estimate_resources(
        **common,
        mm_charges=TIP3P_CHARGES,
        mm_coords=TIP3P_COORDS,
        embedding_mode="diagonal",
    )
    full_oneelectron = estimate_resources(
        **common,
        mm_charges=TIP3P_CHARGES,
        mm_coords=TIP3P_COORDS,
        embedding_mode="full_oneelectron",
    )

    print("=" * 78)
    print("  H2 Full One-Electron Fixed-MO Embedding")
    print("=" * 78)
    print("  Active space: 2 electrons, 2 spatial orbitals -> 4 JW system qubits")
    print("  MM shell:     one TIP3P water represented as three point charges")
    print()
    print_resource_table(
        [
            ("vacuum", vacuum),
            ("diagonal", diagonal),
            ("full_oneelectron", full_oneelectron),
        ]
    )
    print()
    print("  Diagnostics are fixed-MO one-electron perturbation norms in Hartree.")
    print("  This is not a relaxed solvation energy calculation.")
    print("  The MO frame and two-electron tensor stay fixed at their vacuum values.")
    print("  Dynamic runtime coefficient workflows remain diagonal-update only.")
    print("=" * 78)


def print_resource_table(rows: Iterable[tuple[str, object]]) -> None:
    """Print selected resource fields and fixed-MO diagnostics."""
    print(
        f"  {'row':<18} {'effective_mode':<18} {'lambda_ha':>10} "
        f"{'Toffoli':>14} {'logical':>8} {'delta_h_offdiag_fro':>22}"
    )
    print(f"  {'-' * 94}")
    for label, resources in rows:
        print(
            f"  {label:<18} "
            f"{resources.embedding_mode:<18} "
            f"{resources.hamiltonian_1norm:>10.6f} "
            f"{resources.toffoli_gates:>14,} "
            f"{resources.logical_qubits:>8} "
            f"{_delta_h_offdiag_fro(resources):>22.6e}"
        )


def _delta_h_offdiag_fro(resources: object) -> float:
    diagnostics = getattr(resources, "embedding_diagnostics", None)
    if diagnostics is None:
        return 0.0
    return float(getattr(diagnostics, "delta_h_offdiag_fro", 0.0))


if __name__ == "__main__":
    main()
