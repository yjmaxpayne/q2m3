# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Phase extraction utilities for QPE measurement results.

Pure NumPy implementation — no PennyLane, Catalyst, or JAX dependencies.

Two extraction modes:
    1. Analytical (probs): probability-weighted expected bin, preserving sub-bin
       sensitivity (~0.1 mHa) compared to argmax (~12 mHa for 4-bit QPE).
    2. Shots-based (samples): decode each bitstring to a bin index, then compute
       energy statistics across shots.

Core formula (shared):
    phase = expected_bin / n_bins
    energy = -2π · phase / base_time + energy_shift
"""

from __future__ import annotations

import numpy as np


def extract_energy_from_probs(
    probs: np.ndarray,
    base_time: float,
    energy_shift: float,
    n_estimation_wires: int,
) -> float:
    """
    Extract energy from analytical probability distribution.

    Uses probability-weighted expected bin index for sub-bin sensitivity.

    Args:
        probs: Probability array of shape (2^n_estimation_wires,).
        base_time: Base evolution time for phase-to-energy conversion.
        energy_shift: Energy offset applied to the Hamiltonian.
        n_estimation_wires: Number of QPE estimation qubits.

    Returns:
        Estimated energy in Hartree.
    """
    n_bins = 2**n_estimation_wires
    bin_indices = np.arange(n_bins, dtype=np.float64)
    expected_bin = np.dot(probs, bin_indices)
    phase = expected_bin / n_bins
    delta_e = -2.0 * np.pi * phase / base_time
    return float(delta_e + energy_shift)


def extract_energy_from_shots(
    samples: np.ndarray,
    base_time: float,
    energy_shift: float,
    n_estimation_wires: int,
    return_statistics: bool = False,
) -> float | dict:
    """
    Extract energy from shots-based measurement samples.

    Each row of *samples* is a bitstring (MSB-first) that encodes a bin index.
    All per-shot energies are computed, then either the mean or a statistics
    dict is returned.

    Args:
        samples: Binary array of shape (n_shots, n_estimation_wires).
        base_time: Base evolution time for phase-to-energy conversion.
        energy_shift: Energy offset applied to the Hamiltonian.
        n_estimation_wires: Number of QPE estimation qubits.
        return_statistics: If True, return dict with mean, std, sem, n_shots.

    Returns:
        Mean energy (float) or statistics dict.

    Raises:
        ValueError: If samples is empty.
    """
    if samples.shape[0] == 0:
        raise ValueError("samples must contain at least one shot")

    n_bins = 2**n_estimation_wires

    # Decode bitstrings to bin indices: MSB-first → multiply by [2^(n-1), ..., 1]
    powers = 2 ** np.arange(n_estimation_wires - 1, -1, -1)
    bin_indices = samples @ powers  # shape (n_shots,)

    # Phase and energy per shot
    phases = bin_indices.astype(np.float64) / n_bins
    energies = -2.0 * np.pi * phases / base_time + energy_shift

    if return_statistics:
        n_shots = len(energies)
        std = float(np.std(energies, ddof=1)) if n_shots > 1 else 0.0
        return {
            "mean": float(np.mean(energies)),
            "std": std,
            "sem": std / np.sqrt(n_shots),
            "n_shots": n_shots,
        }

    return float(np.mean(energies))
