#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
QPE Circuit Compilation Memory Profiler

Measures real peak memory consumption during Catalyst @qjit compilation of
QPE circuits in two Hamiltonian modes:

  - H_fixed:   Coefficients are Python floats, built OUTSIDE @qjit.
                Catalyst can constant-fold them → smaller IR, less memory.
  - H_dynamic: Coefficients are JAX-traced runtime parameters INSIDE @qjit.
                Full symbolic graph preserved → larger IR, more memory.

Designed to answer Xanadu's question:

    "4.2k IR operations is relatively small and isn't expected to drive
     such large memory requirements."

Key fact: The ~16GB figure in energy.py:660-661 describes an OLD architecture
bug (QPE IR double-inlined into MC loop MLIR). Current code pre-compiles QPE
separately (orchestrator.py:656-673). This script measures ACTUAL memory.

Three measurement layers:
  - resource.getrusage(RUSAGE_SELF).ru_maxrss → process lifetime peak RSS
    (captures C++ MLIR compiler allocations)
  - /proc/self/status VmRSS/VmPeak → current/peak RSS (Linux kernel level)
  - tracemalloc → Python heap only (PennyLane IR construction, JAX tracing)

Core profiling logic has been migrated to ``q2m3.profiling`` subpackage.
This script retains only the CLI interface and Rich output formatting.

Usage:
    uv run python examples/qpe_memory_profile.py                          # both modes
    uv run python examples/qpe_memory_profile.py --mode fixed             # H_fixed only
    uv run python examples/qpe_memory_profile.py --mode dynamic           # H_dynamic only
    uv run python examples/qpe_memory_profile.py --mode both --n-est 2    # compare modes
    uv run python examples/qpe_memory_profile.py --sweep                  # parameter sweep
    uv run python examples/qpe_memory_profile.py --sweep --mode fixed     # sweep H_fixed
    uv run python examples/qpe_memory_profile.py --ir-dir ./tmp           # preserve IR files
"""

import argparse
import os
import sys
import tracemalloc
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pennylane as qml  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from q2m3.core.device_utils import CATALYST_VERSION  # noqa: E402
from q2m3.profiling import (  # noqa: E402
    MOLECULES,
    ProfileResult,
    run_both_modes,
    run_single_profile,
    run_sweep,
)

console = Console()


# =============================================================================
# Output Formatting (Rich tables/panels)
# =============================================================================


def print_mode_comparison(
    fixed: ProfileResult,
    dynamic: ProfileResult,
    parent_fixed: dict | None = None,
    parent_dynamic: dict | None = None,
):
    """Print side-by-side comparison table of H_fixed vs H_dynamic."""
    table = Table(title="H_fixed vs H_dynamic Comparison", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("H_fixed", justify="right", style="cyan")
    table.add_column("H_dynamic", justify="right", style="magenta")
    table.add_column("Ratio (dyn/fix)", justify="right", style="yellow")

    def _ratio(a: float, b: float) -> str:
        if a > 0:
            return f"{b / a:.2f}×"
        return "—"

    # Compilation time
    fix_t = fixed.phase_b.elapsed_s if fixed.phase_b else 0
    dyn_t = dynamic.phase_b.elapsed_s if dynamic.phase_b else 0
    table.add_row("Compile Time (s)", f"{fix_t:.1f}", f"{dyn_t:.1f}", _ratio(fix_t, dyn_t))

    # Peak RSS (Python process — self-reported)
    fix_rss = fixed.phase_b.maxrss_mb if fixed.phase_b else 0
    dyn_rss = dynamic.phase_b.maxrss_mb if dynamic.phase_b else 0
    table.add_row(
        "Python RSS — self (MB)", f"{fix_rss:.0f}", f"{dyn_rss:.0f}", _ratio(fix_rss, dyn_rss)
    )

    # Peak RSS of compiler subprocess (RUSAGE_CHILDREN — the real consumer!)
    fix_child = fixed.phase_b.maxrss_children_mb if fixed.phase_b else 0
    dyn_child = dynamic.phase_b.maxrss_children_mb if dynamic.phase_b else 0
    table.add_row(
        "[red]Compiler RSS (MB)[/red]",
        f"[red]{fix_child:.0f}[/red]",
        f"[red]{dyn_child:.0f}[/red]",
        _ratio(fix_child, dyn_child),
    )

    # Total (self + children)
    fix_total = fix_rss + fix_child
    dyn_total = dyn_rss + dyn_child
    table.add_row(
        "[bold]Total Peak (MB)[/bold]",
        f"[bold]{fix_total:.0f}[/bold]",
        f"[bold]{dyn_total:.0f}[/bold]",
        _ratio(fix_total, dyn_total),
    )

    # Peak RSS (parent-observed)
    if parent_fixed and parent_dynamic:
        p_fix = parent_fixed["peak_rss_mb"]
        p_dyn = parent_dynamic["peak_rss_mb"]
        table.add_row("Parent-observed (MB)", f"{p_fix:.0f}", f"{p_dyn:.0f}", _ratio(p_fix, p_dyn))
        # VmHWM (kernel high water mark)
        hwm_fix = parent_fixed["peak_hwm_mb"]
        hwm_dyn = parent_dynamic["peak_hwm_mb"]
        table.add_row(
            "VmHWM — kernel (MB)", f"{hwm_fix:.0f}", f"{hwm_dyn:.0f}", _ratio(hwm_fix, hwm_dyn)
        )

    # Timeline peak
    table.add_row(
        "Timeline Peak (MB)",
        f"{fixed.timeline_peak_mb:.0f}",
        f"{dynamic.timeline_peak_mb:.0f}",
        _ratio(fixed.timeline_peak_mb, dynamic.timeline_peak_mb),
    )

    # Python heap peak
    fix_heap = fixed.phase_b.tracemalloc_peak_mb if fixed.phase_b else 0
    dyn_heap = dynamic.phase_b.tracemalloc_peak_mb if dynamic.phase_b else 0
    table.add_row("Python Heap Peak (MB)", f"{fix_heap:.1f}", f"{dyn_heap:.1f}", "")

    # Execution time
    fix_exec = fixed.phase_c.elapsed_s if fixed.phase_c else 0
    dyn_exec = dynamic.phase_c.elapsed_s if dynamic.phase_c else 0
    table.add_row("Exec Time (5x, s)", f"{fix_exec:.3f}", f"{dyn_exec:.3f}", "")

    # IR sizes (largest stage)
    def _largest_ir(analysis):
        if not analysis:
            return 0.0, "N/A"
        stage, size, _ = max(analysis, key=lambda x: x[1])
        return size, stage

    fix_ir, fix_stage = _largest_ir(fixed.ir_analysis)
    dyn_ir, dyn_stage = _largest_ir(dynamic.ir_analysis)
    table.add_row(
        "Largest IR (KB)",
        f"{fix_ir:.0f} ({fix_stage})",
        f"{dyn_ir:.0f} ({dyn_stage})",
        _ratio(fix_ir, dyn_ir),
    )

    # Prob sum sanity
    table.add_row("Prob Sum", f"{fixed.prob_sum:.6f}", f"{dynamic.prob_sum:.6f}", "")

    console.print(table)

    # Print smaps breakdown at peak (memory categorization)
    if parent_fixed and parent_dynamic:
        _print_smaps_comparison(parent_fixed, parent_dynamic)


def _print_smaps_comparison(parent_fixed: dict, parent_dynamic: dict):
    """Print memory categorization from smaps_rollup at peak RSS."""
    smaps_keys = [
        "Rss",
        "Pss",
        "Anonymous",
        "LazyFree",
        "AnonHugePages",
        "Shared_Clean",
        "Shared_Dirty",
        "Private_Clean",
        "Private_Dirty",
    ]
    fix_smaps = parent_fixed.get("peak_smaps", {})
    dyn_smaps = parent_dynamic.get("peak_smaps", {})

    if not fix_smaps and not dyn_smaps:
        return

    table = Table(title="Memory Categorization at Peak (smaps_rollup)", show_lines=True)
    table.add_column("Category", style="bold")
    table.add_column("H_fixed (MB)", justify="right", style="cyan")
    table.add_column("H_dynamic (MB)", justify="right", style="magenta")
    table.add_column("Δ (dyn-fix)", justify="right", style="yellow")

    for key in smaps_keys:
        fix_val = fix_smaps.get(key, 0.0)
        dyn_val = dyn_smaps.get(key, 0.0)
        if fix_val > 0 or dyn_val > 0:
            delta = dyn_val - fix_val
            table.add_row(key, f"{fix_val:.1f}", f"{dyn_val:.1f}", f"{delta:+.1f}")

    console.print(table)


def print_system_info():
    """Print system information panel."""
    import platform

    total_ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)

    info_lines = [
        f"Python:     {platform.python_version()}",
        f"PennyLane:  {qml.__version__}",
        f"Catalyst:   {CATALYST_VERSION}",
        f"Platform:   {platform.system()} {platform.machine()}",
        f"Total RAM:  {total_ram_gb:.1f} GB",
    ]
    try:
        import jax

        info_lines.append(f"JAX:        {jax.__version__} (backend: {jax.default_backend()})")
    except ImportError:
        info_lines.append("JAX:        not installed")

    console.print(Panel("\n".join(info_lines), title="System Information", border_style="blue"))


def print_circuit_params(result: ProfileResult):
    """Print circuit parameters panel."""
    mode_label = (
        "H_fixed (compile-time constants)"
        if result.mode == "fixed"
        else "H_dynamic (runtime coefficients)"
    )
    lines = [
        f"Molecule:          {result.molecule}",
        f"Hamiltonian mode:  {mode_label}",
        f"System qubits:     {result.n_system_qubits}",
        f"Estimation wires:  {result.n_estimation_wires}",
        f"Total qubits:      {result.n_system_qubits + result.n_estimation_wires}",
        f"Trotter steps:     {result.n_trotter}",
        f"Hamiltonian terms: {result.n_terms}",
        f"IR scale estimate: {result.ir_scale} "
        f"({result.n_estimation_wires}×{result.n_trotter}×{result.n_terms})",
    ]
    console.print(Panel("\n".join(lines), title="Circuit Parameters", border_style="cyan"))


def print_memory_table(result: ProfileResult):
    """Print Phase A/B/C memory comparison table."""
    table = Table(title="Memory Profile by Phase", show_lines=True)
    table.add_column("Phase", style="bold")
    table.add_column("RSS Δ (MB)", justify="right")
    table.add_column("ru_maxrss\nself (MB)", justify="right")
    table.add_column("ru_maxrss\nchildren (MB)", justify="right", style="red")
    table.add_column("Python Heap\nPeak (MB)", justify="right")
    table.add_column("Wall Time (s)", justify="right")

    for snap in [result.phase_a, result.phase_b, result.phase_c]:
        if snap is None:
            continue
        table.add_row(
            snap.label,
            f"{snap.rss_mb:+.1f}",
            f"{snap.maxrss_mb:.1f}",
            f"{snap.maxrss_children_mb:.1f}",
            f"{snap.tracemalloc_peak_mb:.1f}",
            f"{snap.elapsed_s:.2f}",
        )

    console.print(table)


def print_ir_analysis(ir_analysis: list[tuple[str, float, int]]):
    """Print Catalyst IR stage analysis table."""
    if not ir_analysis:
        console.print("[yellow]No IR analysis data available.[/yellow]")
        return

    table = Table(title="Catalyst IR Stage Analysis", show_lines=True)
    table.add_column("Compilation Stage", style="bold")
    table.add_column("IR Size (KB)", justify="right")
    table.add_column("IR Lines", justify="right")
    table.add_column("Amplification", justify="right", style="yellow")

    prev_size = None
    for stage, size_kb, n_lines in ir_analysis:
        amp = f"{size_kb / prev_size:.1f}×" if prev_size and prev_size > 0 else "—"
        table.add_row(stage, f"{size_kb:.1f}", f"{n_lines:,}", amp)
        prev_size = size_kb

    console.print(table)

    # Total amplification
    if len(ir_analysis) >= 2:
        first_size = ir_analysis[0][1]
        last_size = ir_analysis[-1][1]
        if first_size > 0:
            console.print(
                f"  Total IR amplification: {first_size:.1f} KB → {last_size:.1f} KB "
                f"([bold]{last_size / first_size:.1f}×[/bold])"
            )


def print_memory_timeline(samples: list[tuple[float, float]]):
    """Print ASCII timeline of RSS during compilation."""
    if len(samples) < 3:
        return

    console.print("\n[bold]Memory Timeline (RSS during compilation):[/bold]")
    chart_width = 60
    rss_values = [s[1] for s in samples]
    min_rss = min(rss_values)
    max_rss = max(rss_values)
    rss_range = max_rss - min_rss

    if rss_range < 1:
        console.print("  (RSS variation < 1 MB — flat)")
        return

    # Downsample to ~20 rows
    n_rows = min(20, len(samples))
    step = max(1, len(samples) // n_rows)

    for i in range(0, len(samples), step):
        t, rss = samples[i]
        bar_len = int((rss - min_rss) / rss_range * chart_width)
        bar = "█" * bar_len
        console.print(f"  {t:6.1f}s │{bar} {rss:.0f} MB")

    console.print(f"  {'':6s} └{'─' * chart_width}")
    console.print(f"  Peak RSS: {max_rss:.0f} MB  |  Range: {min_rss:.0f}–{max_rss:.0f} MB")


def print_sweep_table(results: list[ProfileResult]):
    """Print parameter sweep results table."""
    table = Table(title="Parameter Sweep: Memory Scaling", show_lines=True)
    table.add_column("Mode", justify="center")
    table.add_column("n_est", justify="center")
    table.add_column("n_trotter", justify="center")
    table.add_column("IR Scale", justify="right")
    table.add_column("Peak RSS (MB)", justify="right", style="bold")
    table.add_column("Compile Time (s)", justify="right")
    table.add_column("Prob Sum", justify="right")
    table.add_column("Status", justify="center")

    for r in results:
        mode_tag = "fixed" if r.mode == "fixed" else "dyn"
        if r.error:
            table.add_row(
                mode_tag,
                str(r.n_estimation_wires),
                str(r.n_trotter),
                str(r.ir_scale),
                "—",
                "—",
                "—",
                f"[red]{r.error}[/red]",
            )
        else:
            table.add_row(
                mode_tag,
                str(r.n_estimation_wires),
                str(r.n_trotter),
                str(r.ir_scale),
                f"{r.phase_b.maxrss_mb:.0f}" if r.phase_b else "—",
                f"{r.phase_b.elapsed_s:.1f}" if r.phase_b else "—",
                f"{r.prob_sum:.6f}",
                "[green]OK[/green]" if abs(r.prob_sum - 1.0) < 0.01 else "[red]BAD[/red]",
            )

    console.print(table)


def print_summary(result: ProfileResult):
    """Print summary panel answering Xanadu's question."""
    if result.phase_b is None:
        return

    peak_rss = result.phase_b.maxrss_mb
    compile_time = result.phase_b.elapsed_s
    n_terms = result.n_terms
    ir_scale = result.ir_scale
    mode_label = "H_fixed" if result.mode == "fixed" else "H_dynamic"

    # Find largest IR stage
    largest_stage = "N/A"
    largest_size = 0
    for stage, size_kb, _ in result.ir_analysis:
        if size_kb > largest_size:
            largest_size = size_kb
            largest_stage = stage

    # Peak compiler subprocess RSS (the REAL memory consumer)
    children_rss = result.phase_b.maxrss_children_mb

    lines = [
        f"[bold]Molecule:[/bold] {result.molecule}",
        f"[bold]Hamiltonian Mode:[/bold] {mode_label}",
        f"[bold]Python Process RSS:[/bold] {peak_rss:.0f} MB ({peak_rss / 1024:.2f} GB)",
        f"[bold]Compiler Subprocess RSS:[/bold] {children_rss:.0f} MB ({children_rss / 1024:.2f} GB)",
        f"[bold]Total Peak (self+children):[/bold] {peak_rss + children_rss:.0f} MB "
        f"({(peak_rss + children_rss) / 1024:.2f} GB)",
        f"[bold]Compilation Time:[/bold] {compile_time:.1f}s",
        f"[bold]Hamiltonian Terms:[/bold] {n_terms}",
        f"[bold]IR Scale:[/bold] {ir_scale} ({result.n_estimation_wires}×{result.n_trotter}×{n_terms})",
        f"[bold]Largest IR Stage:[/bold] {largest_stage} ({largest_size:.0f} KB)",
        f"[bold]Prob Sum (sanity):[/bold] {result.prob_sum:.6f}",
        "",
        "[bold]Key Finding:[/bold]",
    ]

    if children_rss > peak_rss:
        lines.append(
            f"  Compiler subprocess ({children_rss:.0f} MB) dominates over Python "
            f"process ({peak_rss:.0f} MB). The `catalyst` MLIR→LLVM compiler is "
            "the true memory consumer — not the Python process."
        )
    elif peak_rss < 2048:
        lines.append(
            f"  Total peak = {peak_rss + children_rss:.0f} MB — well below the 16 GB "
            "figure. Pre-compilation fix is effective."
        )
    else:
        lines.append(
            f"  Total peak = {peak_rss + children_rss:.0f} MB — significant memory. "
            "Investigate IR amplification stages above."
        )

    console.print(Panel("\n".join(lines), title=f"Summary ({mode_label})", border_style="green"))


# =============================================================================
# Main + CLI
# =============================================================================


def _print_single_result(result: ProfileResult):
    """Print full profiling report for a single mode result."""
    print_circuit_params(result)
    print_memory_table(result)
    print_ir_analysis(result.ir_analysis)
    print_memory_timeline(result.timeline_samples)
    print_summary(result)


def main():
    parser = argparse.ArgumentParser(
        description="QPE Circuit Compilation Memory Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--molecule",
        choices=["h2", "h3o"],
        default="h2",
        help="Target molecule (default: h2)",
    )
    parser.add_argument(
        "--mode",
        choices=["fixed", "dynamic", "both"],
        default="both",
        help="Hamiltonian mode: fixed (compile-time), dynamic (runtime), both (default: both)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run parameter scaling sweep (H2 only)",
    )
    parser.add_argument("--n-est", type=int, default=4, help="Estimation wires (default: 4)")
    parser.add_argument("--n-trotter", type=int, default=10, help="Trotter steps (default: 10)")
    parser.add_argument(
        "--ir-dir",
        type=str,
        default=None,
        help="Directory to save Catalyst IR stages (default: auto-cleanup after analysis)",
    )
    args = parser.parse_args()

    console.print(
        Panel(
            "[bold]QPE Circuit Compilation Memory Profiler[/bold]\n"
            "Measuring real peak memory during Catalyst @qjit compilation\n"
            "of QPE circuits (H_fixed and/or H_dynamic modes).",
            border_style="magenta",
        )
    )

    print_system_info()

    if args.sweep:
        console.print(f"\n[bold]Running parameter sweep (mode={args.mode})...[/bold]")
        if args.mode == "both":
            results_fixed = run_sweep(args.molecule, mode="fixed", on_progress=console.print)
            results_dynamic = run_sweep(args.molecule, mode="dynamic", on_progress=console.print)
            print_sweep_table(list(results_fixed.values()) + list(results_dynamic.values()))
        else:
            results = run_sweep(args.molecule, mode=args.mode, on_progress=console.print)
            results_list = list(results.values())
            print_sweep_table(results_list)
            for r in reversed(results_list):
                if not r.error:
                    print_ir_analysis(r.ir_analysis)
                    print_summary(r)
                    break
    elif args.mode == "both":
        console.print(
            f"\n[bold]Profiling {args.molecule.upper()}: "
            f"n_est={args.n_est}, n_trotter={args.n_trotter} (both modes)[/bold]"
        )
        result_fixed, result_dynamic, parent_fixed, parent_dynamic = run_both_modes(
            args.molecule,
            args.n_est,
            args.n_trotter,
            ir_dir=args.ir_dir,
            on_progress=console.print,
        )

        # Print detailed reports for each mode
        for result in [result_fixed, result_dynamic]:
            if not result.error:
                console.print(f"\n{'=' * 70}")
                _print_single_result(result)

        # Print comparison if both succeeded
        if not result_fixed.error and not result_dynamic.error:
            console.print(f"\n{'=' * 70}")
            print_mode_comparison(
                result_fixed,
                result_dynamic,
                parent_fixed=parent_fixed,
                parent_dynamic=parent_dynamic,
            )
    else:
        # Single mode (in-process)
        console.print(
            f"\n[bold]Profiling {args.molecule.upper()}: "
            f"n_est={args.n_est}, n_trotter={args.n_trotter} "
            f"(H_{args.mode})[/bold]"
        )

        tracemalloc.start()
        result = run_single_profile(
            MOLECULES[args.molecule],
            args.n_est,
            args.n_trotter,
            mode=args.mode,
            ir_dir=args.ir_dir,
            on_progress=console.print,
        )
        tracemalloc.stop()

        if result.error:
            console.print(f"[red]ERROR: {result.error}[/red]")
            sys.exit(1)

        _print_single_result(result)


if __name__ == "__main__":
    main()
