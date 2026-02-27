"""
Time statistics formatting for MC solvation simulations.

Uses Rich library for console output with tables and panels.
Supports three Hamiltonian modes: hf_corrected / fixed / dynamic.
"""

from dataclasses import dataclass, field

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class TimingData:
    """Container for simulation timing data."""

    quantum_compile_time: float = 0.0
    mc_loop_time: float = 0.0
    hf_times: np.ndarray = field(default_factory=lambda: np.array([]))
    quantum_times: np.ndarray = field(default_factory=lambda: np.array([]))
    n_mc_steps: int = 0
    n_quantum_evals: int = 0
    hamiltonian_mode: str = ""


def create_timing_table(timing: TimingData) -> Table:
    """Create Rich table for execution phase timing."""
    table = Table(title="Execution Phase", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="dim")
    table.add_column("Total", justify="right")
    table.add_column("Avg", justify="right")
    table.add_column("StdDev", justify="right")

    # HF / Callback label depends on mode
    mode = timing.hamiltonian_mode
    if mode == "hf_corrected":
        hf_label = "HF+MM"
    elif mode in ("fixed", "dynamic"):
        hf_label = "Callback"
    else:
        hf_label = "HF"

    hf_total = np.sum(timing.hf_times)
    hf_avg = np.mean(timing.hf_times) * 1000 if len(timing.hf_times) > 0 else 0
    hf_std = np.std(timing.hf_times) * 1000 if len(timing.hf_times) > 0 else 0
    table.add_row(
        f"{hf_label} ({timing.n_mc_steps}x)",
        f"{hf_total:.1f}s",
        f"{hf_avg:.1f}ms",
        f"{hf_std:.1f}ms",
    )

    # QPE statistics — label by mode
    if mode == "hf_corrected":
        interval = max(1, timing.n_mc_steps // max(1, timing.n_quantum_evals))
        q_label = f"QPE (diagnostic, interval={interval})"
    elif mode == "fixed":
        q_label = "QPE (every step, frozen coeffs)"
    elif mode == "dynamic":
        q_label = "QPE (every step, dynamic coeffs)"
    else:
        q_label = "QPE"

    q_valid = timing.quantum_times[timing.quantum_times > 0]
    n_q = len(q_valid)
    if n_q > 0:
        q_total = np.sum(q_valid)
        q_avg = np.mean(q_valid) * 1000
        q_std = np.std(q_valid) * 1000
        table.add_row(
            f"{q_label} ({n_q}x)",
            f"{q_total:.1f}s",
            f"{q_avg:.1f}ms",
            f"{q_std:.1f}ms",
        )

        # First vs subsequent breakdown
        if n_q > 1:
            table.add_row(
                "  └ First",
                f"{q_valid[0] * 1000:.1f}ms",
                "-",
                "-",
                style="dim",
            )
            subsequent = q_valid[1:]
            table.add_row(
                "  └ Subsequent",
                f"{np.sum(subsequent):.2f}s",
                f"{np.mean(subsequent) * 1000:.1f}ms",
                f"{np.std(subsequent) * 1000:.1f}ms",
                style="dim",
            )

    return table


def print_time_statistics(timing: TimingData, console: Console | None = None) -> None:
    """
    Print formatted timing statistics to console.

    Args:
        timing: TimingData instance
        console: Rich Console (creates new if None)
    """
    console = console or Console()

    # Compilation phase overhead
    mc_jit_overhead = max(
        0,
        timing.mc_loop_time
        - np.sum(timing.hf_times)
        - np.sum(timing.quantum_times[timing.quantum_times > 0]),
    )

    mode = timing.hamiltonian_mode
    if mode in ("fixed", "dynamic"):
        compile_info = (
            f"[bold]Compilation Phase[/bold]\n"
            f"  • QPE @qjit precompilation: {timing.quantum_compile_time:.2f}s "
            f"(parameterized TrotterProduct)\n"
            f"  • MC loop: Python (calls precompiled QPE via JAX arrays)"
        )
    elif mode == "hf_corrected":
        compile_info = (
            f"[bold]Compilation Phase[/bold]\n"
            f"  • QPE @qjit: {timing.quantum_compile_time:.2f}s (one-time)\n"
            f"  • MC Loop overhead: ~{mc_jit_overhead:.2f}s"
        )
    else:
        compile_info = (
            f"[bold]Compilation Phase[/bold]\n" f"  • @qjit compilation: ~{mc_jit_overhead:.2f}s"
        )
    console.print(Panel(compile_info, title="Time Statistics", border_style="blue"))

    # Execution phase table
    console.print(create_timing_table(timing))

    # Total
    total = timing.quantum_compile_time + timing.mc_loop_time
    console.print(f"\n[bold]Total wall time:[/bold] {total:.2f}s")


def create_timing_data_from_result(
    result: dict,
    compile_time: float,
    loop_time: float,
    hamiltonian_mode: str = "",
) -> TimingData:
    """Create TimingData from MC loop result dictionary."""
    hf_times = np.array(result.get("hf_times", []))
    quantum_times = np.array(result.get("quantum_times", result.get("qpe_times", [])))

    return TimingData(
        quantum_compile_time=compile_time,
        mc_loop_time=loop_time,
        hf_times=hf_times,
        quantum_times=quantum_times,
        n_mc_steps=len(hf_times),
        n_quantum_evals=int(
            result.get("n_quantum_evaluations", result.get("n_qpe_evaluations", 0))
        ),
        hamiltonian_mode=hamiltonian_mode,
    )
