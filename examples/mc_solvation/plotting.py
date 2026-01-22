# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Plotting Utilities for MC Solvation Simulations

Provides energy trajectory visualization and analysis plots.
"""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .constants import HARTREE_TO_KCAL_MOL


def plot_energy_trajectory(
    mc_steps: Sequence[int],
    hf_energies: Sequence[float],
    quantum_steps: Sequence[int] | None = None,
    quantum_energies: Sequence[float] | None = None,
    reference_energy: float | None = None,
    title: str = "Energy Trajectory",
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot energy trajectory over MC steps.

    Args:
        mc_steps: MC step indices for HF energies
        hf_energies: HF (or total QM/MM) energies at each step
        quantum_steps: MC step indices where quantum evaluation was performed
        quantum_energies: Quantum algorithm energy estimates
        reference_energy: Reference energy for relative plotting (e.g., vacuum energy)
        title: Plot title
        output_path: Path to save figure (None = don't save)
        show: Whether to display the plot

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to relative energies if reference provided
    if reference_energy is not None:
        hf_plot = (np.array(hf_energies) - reference_energy) * HARTREE_TO_KCAL_MOL
        ylabel = "Relative Energy (kcal/mol)"
        if quantum_energies is not None:
            quantum_plot = (np.array(quantum_energies) - reference_energy) * HARTREE_TO_KCAL_MOL
    else:
        hf_plot = np.array(hf_energies)
        ylabel = "Energy (Hartree)"
        if quantum_energies is not None:
            quantum_plot = np.array(quantum_energies)

    # Plot HF trajectory
    ax.plot(mc_steps, hf_plot, "b-", alpha=0.7, linewidth=0.8, label="HF")

    # Plot quantum evaluations
    if quantum_steps is not None and quantum_energies is not None:
        ax.scatter(
            quantum_steps,
            quantum_plot,
            c="red",
            s=50,
            marker="o",
            label="Quantum",
            zorder=5,
        )

    ax.set_xlabel("MC Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_energy_comparison(
    hf_energies: Sequence[float],
    quantum_energies: Sequence[float],
    title: str = "HF vs Quantum Energy Comparison",
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot correlation between HF and quantum energies.

    Args:
        hf_energies: HF energies at quantum evaluation steps
        quantum_energies: Quantum algorithm energy estimates
        title: Plot title
        output_path: Path to save figure
        show: Whether to display

    Returns:
        Matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    hf = np.array(hf_energies)
    quantum = np.array(quantum_energies)
    diff = (quantum - hf) * HARTREE_TO_KCAL_MOL

    # Scatter plot
    ax1.scatter(hf, quantum, alpha=0.6)
    ax1.plot([hf.min(), hf.max()], [hf.min(), hf.max()], "r--", label="y=x")
    ax1.set_xlabel("HF Energy (Hartree)")
    ax1.set_ylabel("Quantum Energy (Hartree)")
    ax1.set_title("Correlation")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Error histogram
    ax2.hist(diff, bins=20, edgecolor="black", alpha=0.7)
    ax2.axvline(np.mean(diff), color="r", linestyle="--", label=f"Mean: {np.mean(diff):.2f}")
    ax2.axvline(0, color="k", linestyle="-", alpha=0.5)
    ax2.set_xlabel("Quantum - HF (kcal/mol)")
    ax2.set_ylabel("Count")
    ax2.set_title("Error Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_acceptance_rate(
    mc_steps: Sequence[int],
    cumulative_accepted: Sequence[int],
    window_size: int = 50,
    title: str = "MC Acceptance Rate",
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot cumulative and windowed acceptance rate.

    Args:
        mc_steps: MC step indices
        cumulative_accepted: Cumulative number of accepted moves
        window_size: Window size for rolling acceptance rate
        title: Plot title
        output_path: Path to save figure
        show: Whether to display

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    steps = np.array(mc_steps)
    accepted = np.array(cumulative_accepted)

    # Cumulative rate
    cumulative_rate = accepted / (steps + 1)
    ax.plot(steps, cumulative_rate * 100, "b-", label="Cumulative")

    # Rolling rate
    if len(steps) > window_size:
        rolling_accepted = np.diff(accepted, prepend=0)
        rolling_rate = np.convolve(
            rolling_accepted, np.ones(window_size) / window_size, mode="valid"
        )
        ax.plot(
            steps[window_size - 1 :],
            rolling_rate * 100,
            "r-",
            alpha=0.7,
            label=f"Rolling ({window_size} steps)",
        )

    ax.axhline(50, color="gray", linestyle="--", alpha=0.5, label="50% target")
    ax.set_xlabel("MC Step")
    ax.set_ylabel("Acceptance Rate (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
