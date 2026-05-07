#!/usr/bin/env python3
# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
H2 Three-Mode QPE Comparison: MC Solvation Experiment

Precision-ordered comparison of three QPE architectures:
  Mode 1 (fixed):        E = E_QPE(H_vac) + E_MM           [approximate]
  Mode 2 (hf_corrected): E = E_HF(R) + E_MM                [intermediate]
  Mode 3 (dynamic):      E = E_QPE(H_eff with MM) + E_MM   [most rigorous]

Analysis delegates to q2m3.solvation.analysis.run_mode_comparison().
"""

import warnings

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from q2m3.constants import HARTREE_TO_KCAL_MOL
from q2m3.solvation import MoleculeConfig, QPEConfig, SolvationConfig, run_solvation
from q2m3.solvation.analysis import ModeComparisonResult, run_mode_comparison

console = Console()


def _effective_compile_time(timing) -> float:
    """Total compilation overhead (QPE inlined into MC loop @qjit compilation)."""
    if timing is None:
        return 0.0
    exec_time = float(
        np.sum(timing.hf_times) + np.sum(timing.quantum_times[timing.quantum_times > 0])
    )
    return timing.quantum_compile_time + max(0.0, timing.mc_loop_time - exec_time)


# --- Molecule ---

H2_MOLECULE = MoleculeConfig(
    name="H2",
    symbols=["H", "H"],
    coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    charge=0,
    active_electrons=2,
    active_orbitals=2,
    basis="sto-3g",
)

# --- QPE Configurations (fixed → hf_corrected → dynamic, precision ascending) ---

QPE_FIXED = QPEConfig(  # static H_vac, drives acceptance every step
    n_estimation_wires=4,
    n_trotter_steps=10,
    n_shots=100,
    qpe_interval=1,
    target_resolution=0.003,
    energy_range=0.2,
)
QPE_HF_CORRECTED = QPEConfig(  # HF acceptance, interval QPE diagnostics
    n_estimation_wires=4,
    n_trotter_steps=10,
    n_shots=100,
    qpe_interval=10,
    target_resolution=0.003,
    energy_range=0.2,
)
QPE_DYNAMIC = QPEConfig(  # runtime H_eff, expected_value mode
    n_estimation_wires=4,
    n_trotter_steps=10,
    n_shots=0,
    qpe_interval=1,
    target_resolution=0.003,
    energy_range=0.2,
)

N_WATERS = 10
N_MC_STEPS = 500
RANDOM_SEED = 42
TEMPERATURE = 300.0
TRANSLATION_STEP = 0.3
ROTATION_STEP = 0.2618
INITIAL_WATER_DISTANCE = 4.0


# --- Report Functions ---


def print_mode_comparison(
    result_fixed: dict, result_hf: dict, result_dynamic: dict, e_vacuum: float
):
    """Print three-mode energy comparison table (fixed → hf_corrected → dynamic)."""
    console.print()
    console.print(
        Panel("[bold]Mode Comparison: Three QPE Architectures[/bold]", border_style="magenta")
    )

    modes = [("fixed", result_fixed), ("hf_corrected", result_hf), ("dynamic", result_dynamic)]
    metas = [r.get("circuit_metadata", {}) for _, r in modes]

    if all(metas):
        cfg = Table(title="Circuit Configuration")
        cfg.add_column("Parameter", style="bold")
        for name, _ in modes:
            cfg.add_column(name, justify="right")
        for param, label in [
            ("n_hamiltonian_terms", "Hamiltonian Terms"),
            ("n_system_qubits", "System Qubits"),
            ("n_estimation_wires", "Estimation Qubits"),
            ("total_qubits", "Total Qubits"),
        ]:
            cfg.add_row(label, *[str(m.get(param, "?")) for m in metas])

        trotter_vals = []
        for m in metas:
            actual, req = m.get("n_trotter_steps", "?"), m.get("n_trotter_steps_requested", "?")
            s = str(actual) + (f" (req={req}, capped)" if actual != req else "")
            trotter_vals.append(s)
        cfg.add_row("Trotter Steps", *trotter_vals)
        cfg.add_row("Energy Formula", *[m.get("energy_formula", "?") for m in metas])
        console.print(cfg)
        console.print()

    table = Table(title="QPE Energy Comparison")
    table.add_column("Metric", style="bold")
    for name, _ in modes:
        table.add_column(name, justify="right")

    qpe_vals = [float(r["best_qpe_energy"]) for _, r in modes]
    n_evals = [int(r["n_quantum_evaluations"]) for _, r in modes]
    table.add_row(
        "Best QPE Energy (Ha)†",
        *[
            f"{v:.6f} (n={n})" if v < 1e9 else "N/A"
            for v, n in zip(qpe_vals, n_evals, strict=False)
        ],
    )
    stab_vals = [e_vacuum - v if v < 1e9 else float("nan") for v in qpe_vals]
    table.add_row(
        "dE vs HF(vac) (Ha)**",
        *[f"{v:.6f}" if not np.isnan(v) else "N/A" for v in stab_vals],
    )
    table.add_row(
        "dE vs HF(vac) (kcal/mol)**",
        *[f"{v * HARTREE_TO_KCAL_MOL:.2f}" if not np.isnan(v) else "N/A" for v in stab_vals],
    )
    hf_vals = [float(r["best_energy"]) for _, r in modes]
    hf_labels = [
        f"{v:.6f}{'*' if name == 'dynamic' else ''}"
        for (name, _), v in zip(modes, hf_vals, strict=False)
    ]
    table.add_row("Best MC Energy (Ha)", *hf_labels)
    table.add_row(
        "Acceptance Rate (%)",
        *[f"{float(r['acceptance_rate']) * 100:.1f}" for _, r in modes],
    )

    timings = [r.get("timing") for _, r in modes]
    if all(t is not None for t in timings):
        table.add_row("Compile Time (s)", *[f"{_effective_compile_time(t):.1f}" for t in timings])
        table.add_row(
            "Total Time (s)",
            *[f"{t.quantum_compile_time + t.mc_loop_time:.1f}" for t in timings],
        )
    console.print(table)

    console.print(
        "\n  [dim]†Best QPE Energy: min across all evaluations — not directly comparable across modes.\n"
        "  **dE vs HF(vac) mixes theory levels (QPE includes correlation).\n"
        "  *dynamic Best MC Energy = E_QPE(H_eff) + E_MM(sol-sol).[/dim]"
    )

    if all(t is not None for t in timings):
        timing_fixed, timing_hf, timing_dyn = timings
        compile_fixed = _effective_compile_time(timing_fixed)
        compile_hf = _effective_compile_time(timing_hf)
        compile_dyn = _effective_compile_time(timing_dyn)

        def _avg_ms(t):
            q = t.quantum_times[t.quantum_times > 0]
            return float(q[1:].mean() * 1000) if len(q) > 1 else 0.0

        summary = (
            f"[bold]Compile-Once Architecture[/bold]\n"
            f"  fixed:        Fixed H_vac  -> compile {compile_fixed:.1f}s, per-eval ~{_avg_ms(timing_fixed):.1f}ms\n"
            f"  hf_corrected: Fixed H_vac  -> compile {compile_hf:.1f}s, per-eval ~{_avg_ms(timing_hf):.1f}ms\n"
            f"  dynamic:      Runtime H_eff -> compile {compile_dyn:.1f}s, per-eval ~{_avg_ms(timing_dyn):.1f}ms"
        )
        console.print()
        console.print(Panel(summary, border_style="green"))


def print_performance_analysis(result_fixed: dict, result_hf: dict, result_dynamic: dict):
    """Print compile-once architecture analysis with speedup evidence."""
    console.print()
    console.print(
        Panel("[bold]Runtime Coefficient Architecture Analysis[/bold]", border_style="cyan")
    )

    timing_fixed = result_fixed.get("timing")
    timing_dyn = result_dynamic.get("timing")
    meta_fixed = result_fixed.get("circuit_metadata", {})
    meta_hf = result_hf.get("circuit_metadata", {})
    meta_dyn = result_dynamic.get("circuit_metadata", {})

    console.print("\n[bold]1. Compile-Once Evidence[/bold]")
    for label, timing in [
        ("fixed (compiled QPE, every step)", timing_fixed),
        ("dynamic (runtime coefficients + QPE)", timing_dyn),
    ]:
        if not timing:
            continue
        compile_time = _effective_compile_time(timing)
        q = timing.quantum_times[timing.quantum_times > 0]
        console.print(f"  [bold]{label}[/bold]")
        if len(q) > 0:
            console.print(
                f"    Compilation time:    {compile_time:.2f}s  First exec: {q[0]*1000:.1f}ms"
            )
        if len(q) > 1:
            avg_ms = q[1:].mean() * 1000
            ratio = compile_time / q[1:].mean() if q[1:].mean() > 0 else 0
            console.print(
                f"    Subsequent avg:      {avg_ms:.1f}ms  Compile/exec ratio: {ratio:.0f}x"
            )

    console.print("\n[bold]2. Speedup vs Recompilation[/bold]")
    for label, timing in [("fixed", timing_fixed), ("dynamic", timing_dyn)]:
        if not timing:
            continue
        compile_time = _effective_compile_time(timing)
        q = timing.quantum_times[timing.quantum_times > 0]
        if len(q) > 1:
            avg_exec = q[1:].mean()
            n = len(q)
            old_total = n * compile_time
            new_total = compile_time + n * avg_exec
            speedup = old_total / new_total if new_total > 0 else float("inf")
            proj_1k = (1000 * compile_time) / (compile_time + 1000 * avg_exec)
            console.print(
                f"  [bold]{label}[/bold]: {speedup:.1f}x speedup ({n} evals),"
                f" projected 1k evals: {proj_1k:.0f}x"
            )

    console.print("\n[bold]3. Technical Details & Energy Formulas[/bold]")
    for meta, name in [(meta_fixed, "fixed"), (meta_dyn, "dynamic")]:
        if not meta:
            continue
        n_t = meta.get("n_trotter_steps", "?")
        n_req = meta.get("n_trotter_steps_requested", "?")
        cap = " (CAPPED)" if n_t != n_req else ""
        console.print(f"  {name}: Trotter {n_t} (req={n_req}){cap}")
    if meta_dyn:
        n_est = meta_dyn.get("n_estimation_wires", "?")
        n_t = meta_dyn.get("n_trotter_steps", "?")
        n_terms = meta_dyn.get("n_hamiltonian_terms", "?")
        if all(isinstance(x, int) for x in [n_est, n_t, n_terms]):
            console.print(f"  MLIR IR scale: {n_est} x {n_t} x {n_terms} = {n_est*n_t*n_terms} ops")

    ftable = Table(show_header=False, box=None, padding=(0, 2))
    ftable.add_column("Mode", style="bold")
    ftable.add_column("Formula")
    ftable.add_column("Physics")
    ftable.add_row(
        "fixed", meta_fixed.get("energy_formula", "N/A"), "Static vacuum QPE, every step"
    )
    ftable.add_row(
        "hf_corrected", meta_hf.get("energy_formula", "N/A"), "HF acceptance + QPE diagnostics"
    )
    ftable.add_row("dynamic", meta_dyn.get("energy_formula", "N/A"), "Runtime H_eff QPE")
    console.print(ftable)


def print_analysis_report(analysis: ModeComparisonResult):
    """Print delta_corr-pol analysis from structured ModeComparisonResult."""
    console.print()
    console.print(
        Panel(
            "[bold]delta_corr-pol: Correlation-Polarization Analysis[/bold]", border_style="yellow"
        )
    )

    dc = analysis.delta_corr_pol
    n_eval = dc.n_samples
    delta = dc.per_step_delta
    delta_kcal = delta * HARTREE_TO_KCAL_MOL

    # Summary statistics
    console.print(f"\n[bold]1. Per-step Statistics (n={n_eval}, fixed <-> dynamic)[/bold]")
    console.print(
        f"  <delta_corr-pol> = {dc.mean_ha:+.6f} Ha"
        f" = {dc.mean_ha * HARTREE_TO_KCAL_MOL:+.2f} kcal/mol"
    )
    console.print(f"  sigma = {dc.std_ha:.6f} Ha   SEM = {dc.sem_ha:.6f} Ha")
    console.print(f"  t-statistic = {dc.t_statistic:.2f}  (|t|>2.0 for p<0.05, df={n_eval - 1})")
    sig_color = "green" if dc.is_significant else "yellow"
    sig_text = "Statistically significant" if dc.is_significant else "NOT statistically significant"
    console.print(f"  [{sig_color}]{sig_text} at 95% confidence[/{sig_color}]")

    # Per-step delta table
    n_show = min(n_eval, 10)
    console.print(f"\n[bold]2. Per-step delta (first {n_show} steps)[/bold]")
    t = Table(show_header=True, header_style="bold")
    t.add_column("Step", justify="right")
    t.add_column("delta (Ha)", justify="right")
    t.add_column("delta (kcal/mol)", justify="right")
    for i in range(n_show):
        t.add_row(str(i + 1), f"{delta[i]:+.6f}", f"{delta_kcal[i]:+.2f}")
    console.print(t)

    # Energy distributions
    console.print("\n[bold]3. QPE Energy Distributions[/bold]")
    dist = analysis.energy_distributions
    display_modes = [m for m in ("fixed", "hf_corrected", "dynamic") if m in dist]
    dt = Table(show_header=True, header_style="bold")
    dt.add_column("Statistic", style="bold")
    for m in display_modes:
        dt.add_column(m, justify="right")

    def _row(key, label, scale=1.0, fmt=".6f"):
        vals = []
        for m in display_modes:
            v = dist[m].get(key, "N/A")
            vals.append(f"{v * scale:{fmt}}" if isinstance(v, float) else str(v))
        dt.add_row(label, *vals)

    _row("mean", "<E_QPE> (Ha)")
    _row("std", "sigma (mHa)", scale=1000, fmt=".3f")
    _row("min", "Min (Ha)")
    _row("max", "Max (Ha)")
    _row("n_evals", "n_evals")
    console.print(dt)

    # Energy phases
    console.print("\n[bold]4. Energy Trend Analysis[/bold]")
    for mode, phase in analysis.energy_phases.items():
        n_ph = phase.n_per_phase
        console.print(
            f"  {mode}: early (n={n_ph}) <E>={phase.early_mean:+.6f} Ha,"
            f" late <E>={phase.late_mean:+.6f} Ha"
        )

    # QPE-HF consistency
    if analysis.qpe_hf_consistency is not None:
        qhf = analysis.qpe_hf_consistency
        console.print(f"\n[bold]5. dynamic QPE-HF Consistency (n={qhf.n_samples})[/bold]")
        console.print(
            f"  Offset: {qhf.mean_offset_mha:+.2f} mHa  r={qhf.pearson_correlation:.4f}"
            f"  drift: {qhf.offset_drift_mha:+.2f} mHa"
        )
        if qhf.pearson_correlation > 0.5:
            console.print("  [green]Positively correlated (consistent)[/green]")
        elif qhf.pearson_correlation > 0:
            console.print("  [yellow]Weak positive correlation[/yellow]")
        else:
            console.print("  [red]No positive correlation[/red]")

    # Equilibration
    if analysis.equilibration is not None:
        eq = analysis.equilibration
        console.print(f"\n[bold]6. Equilibration Diagnostic ({eq.n_windows} windows)[/bold]")
        console.print(f"  {eq.frac_decreasing:.0%} decreasing, drift={eq.qpe_drift_mha:+.2f} mHa")
        if eq.is_monotonic:
            console.print(
                "  [yellow]WARNING: Monotonically decreasing — chain not equilibrated."
                " (Expected at POC scale.)[/yellow]"
            )
        else:
            console.print(
                "  [green]No monotonic trend — consistent with near-equilibration.[/green]"
            )

    # Physical interpretation
    console.print("\n[bold]7. Physical Interpretation[/bold]")
    console.print(f"  Trotter+phase bias: {analysis.trotter_bias_mha:+.1f} mHa (fixed QPE data)")
    console.print(
        f"  delta_corr-pol = {dc.mean_ha * HARTREE_TO_KCAL_MOL:+.2f}"
        f" +/- {dc.sem_ha * HARTREE_TO_KCAL_MOL:.2f} kcal/mol"
    )
    if dc.is_significant:
        direction = "OVERESTIMATES" if dc.mean_ha > 0 else "UNDERESTIMATES"
        console.print(f"  -> fixed mode {direction} solvation stabilization")
    else:
        console.print(
            f"  -> NOT significant (|t|={abs(dc.t_statistic):.2f} < 2.0). Direction inconclusive.\n"
            "  [dim]Signal (~0.1 kcal/mol) buried in 4-bit QPE noise (~7 kcal/mol sigma).[/dim]"
        )


# --- Main ---


def main():
    """Run H2 MC solvation with three QPE modes and compare."""
    console.print(
        Panel(
            "[bold]H2 Three-Mode QPE Comparison[/bold]\n"
            "fixed | hf_corrected | dynamic  (precision ascending)\n"
            "Runtime Hamiltonian coefficient parameterization",
            title="Experiment",
            border_style="green",
        )
    )
    console.print(
        Panel(
            "[bold]POC Scope[/bold]\n\n"
            "[green]Demonstrates:[/green] Three QPE architectures, runtime coefficient "
            "parameterization, fused PySCF+QPE callback, shared-trajectory delta_corr-pol.\n\n"
            "[yellow]Does NOT demonstrate:[/yellow] Numerical accuracy (4-bit QPE, STO-3G, "
            "500 steps), statistical convergence, or equilibration.\n\n"
            "[dim]Accuracy targets: more estimation qubits, larger basis, longer MC chains.[/dim]",
            border_style="dim",
        )
    )

    mc_params = dict(
        molecule=H2_MOLECULE,
        n_waters=N_WATERS,
        n_mc_steps=N_MC_STEPS,
        temperature=TEMPERATURE,
        translation_step=TRANSLATION_STEP,
        rotation_step=ROTATION_STEP,
        initial_water_distance=INITIAL_WATER_DISTANCE,
        random_seed=RANDOM_SEED,
        verbose=True,
    )

    # Mode 1: fixed (approximate)
    console.print("\n" + "=" * 60)
    console.print("[bold green]  MODE 1: fixed[/bold green]")
    console.print("=" * 60)
    config_fixed = SolvationConfig(**mc_params, qpe_config=QPE_FIXED, hamiltonian_mode="fixed")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="catalyst")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
        result_fixed = run_solvation(config_fixed, show_plots=False)

    # Mode 2: hf_corrected (intermediate)
    console.print("\n" + "=" * 60)
    console.print("[bold blue]  MODE 2: hf_corrected[/bold blue]")
    console.print("=" * 60)
    config_hf = SolvationConfig(
        **mc_params, qpe_config=QPE_HF_CORRECTED, hamiltonian_mode="hf_corrected"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="catalyst")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
        result_hf = run_solvation(config_hf, show_plots=False)

    # Mode 3: dynamic (most rigorous)
    console.print("\n" + "=" * 60)
    console.print("[bold yellow]  MODE 3: dynamic[/bold yellow]")
    console.print("=" * 60)
    config_dynamic = SolvationConfig(
        **mc_params, qpe_config=QPE_DYNAMIC, hamiltonian_mode="dynamic"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="catalyst")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
        result_dynamic = run_solvation(config_dynamic, show_plots=False)

    # Analysis via production API
    e_vacuum = result_fixed["e_vacuum"]
    analysis = run_mode_comparison(
        result_fixed=result_fixed,
        result_dynamic=result_dynamic,
        result_hf_corrected=result_hf,
        e_vacuum=e_vacuum,
    )

    # Reports
    print_mode_comparison(result_fixed, result_hf, result_dynamic, e_vacuum)
    print_performance_analysis(result_fixed, result_hf, result_dynamic)
    print_analysis_report(analysis)

    console.print()
    console.print(
        Panel(
            "[bold]POC Summary[/bold]\n\n"
            "Validated: Runtime coefficient parameterization, QPE-driven MC acceptance, "
            "shared-trajectory delta_corr-pol framework.\n\n"
            "Limitations: 4-bit QPE, STO-3G basis, 500 MC steps.\n\n"
            "Next: 8+ qubits, cc-pVDZ, 10k+ steps with burn-in diagnostics.",
            title="What This POC Proves",
            border_style="green",
        )
    )

    return result_fixed, result_hf, result_dynamic


if __name__ == "__main__":
    main()
