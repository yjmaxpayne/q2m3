# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
General plotting utilities for energy analysis.

Provides cross-module visualization tools such as HF vs quantum energy comparison.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from q2m3.constants import HARTREE_TO_KCAL_MOL


def plot_energy_comparison(
    hf_energies: Sequence[float],
    quantum_energies: Sequence[float],
    title: str = "HF vs Quantum Energy Comparison",
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot correlation between HF and quantum energies.

    Creates a two-panel figure: (1) scatter plot of HF vs quantum energies
    with y=x reference line, (2) histogram of energy differences.

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
