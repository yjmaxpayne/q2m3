"""
Plotting utilities for MC solvation simulations.

Provides energy trajectory visualization and acceptance rate analysis plots.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from q2m3.constants import HARTREE_TO_KCAL_MOL


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
