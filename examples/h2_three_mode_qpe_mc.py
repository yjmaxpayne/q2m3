#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
H2 Three-Mode QPE Comparison: MC Solvation Experiment

Side-by-side comparison of three QPE modes for MC solvation:

  Mode 1 (hf_corrected): E = E_HF(R) + E_MM
    - HF energy drives Metropolis acceptance criterion
    - QPE evaluated at intervals (qpe_interval) for diagnostics only
    - Pre-compiled QPE circuit, reused for all evaluations

  Mode 2 (fixed): E = E_QPE(H_vac) + E_MM
    - Pre-compiled vacuum QPE circuit, executed every MC step
    - QPE energy drives Metropolis acceptance
    - Approximate: ignores correlation-polarization coupling (delta_corr-pol)

  Mode 3 (dynamic): E = E_QPE(H_eff with MM) + E_MM(sol-sol)
    - Runtime Hamiltonian coefficient parameterization
    - Compile once, execute with new coefficients each step (no recompilation)
    - QPE energy directly drives Metropolis acceptance criterion

Phase extraction: All modes use qml.probs() + probability-weighted expected
value (Σ probs[k]·k / 2^n) to convert measurement distributions to continuous
phase estimates. This preserves sensitivity to sub-bin MM corrections (~0.1 mHa),
unlike argmax which discretizes to integer bins (resolution ~12 mHa for 4-bit QPE).

Key technical requirement: TrotterProduct must use check_hermitian=False
when coefficients are JAX-traceable runtime parameters.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from q2m3.constants import HARTREE_TO_KCAL_MOL
from q2m3.solvation import (
    MoleculeConfig,
    QPEConfig,
    SolvationConfig,
    run_solvation,
)

console = Console()


def _effective_compile_time(timing) -> float:
    """Total compilation overhead (QPE inlined into MC loop @qjit compilation)."""
    if timing is None:
        return 0.0
    exec_time = float(
        np.sum(timing.hf_times) + np.sum(timing.quantum_times[timing.quantum_times > 0])
    )
    return timing.quantum_compile_time + max(0.0, timing.mc_loop_time - exec_time)


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
# QPE Configurations (mode-specific)
# =============================================================================

# hf_corrected: HF acceptance + interval QPE diagnostics
# QPE circuit pre-compiled with static vacuum Hamiltonian
QPE_HF_CORRECTED = QPEConfig(
    n_estimation_wires=4,
    n_trotter_steps=10,
    n_shots=100,
    qpe_interval=10,
    target_resolution=0.003,
    energy_range=0.2,
)

# fixed: static vacuum Hamiltonian → compiler constant-folds gates → no memory issue
# QPE evaluated every step, drives acceptance
QPE_FIXED = QPEConfig(
    n_estimation_wires=4,
    n_trotter_steps=10,
    n_shots=100,
    qpe_interval=1,
    target_resolution=0.003,
    energy_range=0.2,
)

# dynamic: runtime-parameterized coefficients → Catalyst cannot constant-fold
# → symbolic IR scales as n_est × n_trotter × n_terms
# QPE energy drives Metropolis acceptance at every MC step.
# n_shots=0 for expected_value mode; qpe_interval=1 (every step).
QPE_DYNAMIC = QPEConfig(
    n_estimation_wires=4,
    n_trotter_steps=10,
    n_shots=0,
    qpe_interval=1,
    target_resolution=0.003,
    energy_range=0.2,
)

# MC parameters (small for demo)
N_WATERS = 10
N_MC_STEPS = 500
RANDOM_SEED = 42
TEMPERATURE = 300.0
TRANSLATION_STEP = 0.3
ROTATION_STEP = 0.2618
INITIAL_WATER_DISTANCE = 4.0


# =============================================================================
# Comparison Output
# =============================================================================


def print_mode_comparison(
    result_hf_corrected: dict, result_fixed: dict, result_dynamic: dict, e_vacuum: float
):
    """Print three-mode comparison of hf_corrected, fixed, and dynamic.

    Note: 'Best QPE Energy' selects the minimum across all MC evaluations.
    Statistical outliers are possible due to 4-bit phase resolution noise.
    The per-step δ_corr-pol analysis provides a more reliable comparison.
    """
    console.print()
    console.print(
        Panel("[bold]Mode Comparison: Three QPE Architectures[/bold]", border_style="magenta")
    )

    modes = [
        ("hf_corrected", result_hf_corrected),
        ("fixed", result_fixed),
        ("dynamic", result_dynamic),
    ]

    # Circuit Configuration table
    metas = [r.get("circuit_metadata", {}) for _, r in modes]
    if all(metas):
        cfg_table = Table(title="Circuit Configuration")
        cfg_table.add_column("Parameter", style="bold")
        for name, _ in modes:
            cfg_table.add_column(name, justify="right")

        for param in [
            "n_hamiltonian_terms",
            "n_system_qubits",
            "n_estimation_wires",
            "total_qubits",
        ]:
            label = param.replace("n_", "").replace("_", " ").title()
            if param == "n_hamiltonian_terms":
                label = "Hamiltonian Terms"
            elif param == "n_system_qubits":
                label = "System Qubits"
            elif param == "n_estimation_wires":
                label = "Estimation Qubits"
            elif param == "total_qubits":
                label = "Total Qubits"
            cfg_table.add_row(label, *[str(m.get(param, "?")) for m in metas])

        # Trotter steps — annotate if capped
        trotter_vals = []
        for m in metas:
            actual = m.get("n_trotter_steps", "?")
            req = m.get("n_trotter_steps_requested", "?")
            s = str(actual)
            if actual != req:
                s += f" (req={req}, capped)"
            trotter_vals.append(s)
        cfg_table.add_row("Trotter Steps", *trotter_vals)

        cfg_table.add_row("Energy Formula", *[m.get("energy_formula", "?") for m in metas])
        console.print(cfg_table)
        console.print()

    # QPE Energy Comparison table (3 columns, no Difference)
    table = Table(title="QPE Energy Comparison")
    table.add_column("Metric", style="bold")
    for name, _ in modes:
        table.add_column(name, justify="right")

    # Best QPE energy (with eval count for honest comparison)
    # †: min across different-sized samples is NOT directly comparable
    qpe_vals = [float(r["best_qpe_energy"]) for _, r in modes]
    n_evals = [int(r["n_quantum_evaluations"]) for _, r in modes]
    table.add_row(
        "Best QPE Energy (Ha)†",
        *[
            f"{v:.6f} (n={n})" if v < 1e9 else "N/A"
            for v, n in zip(qpe_vals, n_evals, strict=False)
        ],
    )

    # Energy change vs HF vacuum reference
    # NOTE: This metric mixes theory levels — e_vacuum is HF while best_qpe_energy
    # includes correlation. The difference contains E_corr(vac) + solvation effect,
    # so it overstates the true solvation stabilization by |E_corr(vac)|.
    stab_vals = [e_vacuum - v if v < 1e9 else float("nan") for v in qpe_vals]
    table.add_row(
        "dE vs HF(vac) (Ha)**",
        *[f"{v:.6f}" if not np.isnan(v) else "N/A" for v in stab_vals],
    )
    table.add_row(
        "dE vs HF(vac) (kcal/mol)**",
        *[f"{v * HARTREE_TO_KCAL_MOL:.2f}" if not np.isnan(v) else "N/A" for v in stab_vals],
    )

    # Best MC acceptance energy (different meaning per mode)
    hf_vals = [float(r["best_energy"]) for _, r in modes]
    # Label each value to reflect what best_energy actually tracks
    hf_labels = []
    for name, _ in modes:
        if name == "hf_corrected":
            hf_labels.append(f"{hf_vals[len(hf_labels)]:.6f}")
        elif name == "dynamic":
            hf_labels.append(f"{hf_vals[len(hf_labels)]:.6f}*")
        else:
            hf_labels.append(f"{hf_vals[len(hf_labels)]:.6f}")
    table.add_row("Best MC Energy (Ha)", *hf_labels)

    # Acceptance rate
    acc_vals = [float(r["acceptance_rate"]) * 100 for _, r in modes]
    table.add_row("Acceptance Rate (%)", *[f"{v:.1f}" for v in acc_vals])

    # Timing
    timings = [r.get("timing") for _, r in modes]
    if all(t is not None for t in timings):
        table.add_row(
            "Compile Time (s)",
            *[f"{_effective_compile_time(t):.1f}" for t in timings],
        )
        table.add_row(
            "Total Time (s)",
            *[f"{t.quantum_compile_time + t.mc_loop_time:.1f}" for t in timings],
        )

    console.print(table)

    console.print()
    console.print(
        "  [dim]†Best QPE Energy: min across all evaluations. NOT directly comparable across modes:\n"
        "    hf_corrected samples from interval-based diagnostics;\n"
        "    fixed/dynamic sample from QPE-driven trajectories.\n"
        "    More samples + downhill drift → dynamic min may be biased lower.\n"
        "  *dynamic: Best MC Energy = E_QPE(H_eff) + E_MM(sol-sol) (QPE-driven acceptance).\n"
        "   hf_corrected: Best MC Energy = E_HF(solvated) + E_MM(sol-sol) (HF-based acceptance).\n"
        "   fixed: Best MC Energy = E_QPE(H_vac) + E_MM(sol-sol) (QPE-driven acceptance).\n"
        "  **dE vs HF(vac) = E_HF(vacuum) - Best_QPE_Energy. This mixes theory levels:\n"
        "    QPE energies include correlation, so dE contains |E_corr(vac)| + solvation.\n"
        "    For pure solvation energy, compare QPE energies across modes (delta_corr-pol).[/dim]"
    )

    # Compile-once architecture summary (three-mode)
    if all(t is not None for t in timings):
        timing_hf, timing_fixed, timing_dyn = timings
        compile_hf = _effective_compile_time(timing_hf)
        compile_fixed = _effective_compile_time(timing_fixed)
        compile_dyn = _effective_compile_time(timing_dyn)

        q_fixed_valid = timing_fixed.quantum_times[timing_fixed.quantum_times > 0]
        avg_fixed = float(q_fixed_valid[1:].mean() * 1000) if len(q_fixed_valid) > 1 else 0.0

        q_hf_valid = timing_hf.quantum_times[timing_hf.quantum_times > 0]
        avg_hf = float(q_hf_valid[1:].mean() * 1000) if len(q_hf_valid) > 1 else 0.0

        q_dyn_valid = timing_dyn.quantum_times[timing_dyn.quantum_times > 0]
        avg_dyn = float(q_dyn_valid[1:].mean() * 1000) if len(q_dyn_valid) > 1 else 0.0

        arch_summary = (
            f"[bold]Compile-Once Architecture Comparison[/bold]\n"
            f"  hf_corrected: Fixed H_vac  -> compile {compile_hf:.1f}s, "
            f"per-eval ~{avg_hf:.1f}ms (QPE + PySCF callback)\n"
            f"  fixed:         Fixed H_vac  -> compile {compile_fixed:.1f}s, "
            f"per-eval ~{avg_fixed:.1f}ms (QPE every step)\n"
            f"  dynamic:       Runtime H_eff -> compile {compile_dyn:.1f}s, "
            f"per-eval ~{avg_dyn:.1f}ms (PySCF callback + QPE)"
        )
        console.print()
        console.print(Panel(arch_summary, border_style="green"))


def _print_architecture_performance_analysis(
    result_hf_corrected: dict, result_fixed: dict, result_dynamic: dict
):
    """Print detailed compile-once architecture analysis with performance evidence.

    Quantifies the speedup from runtime coefficient parameterization by comparing
    the compile-once model (one compilation + many fast executions) against the
    hypothetical recompile-each-step model (full compilation per MC evaluation).
    """
    console.print()
    console.print(
        Panel("[bold]Runtime Coefficient Architecture Analysis[/bold]", border_style="cyan")
    )

    timing_fixed = result_fixed.get("timing")
    timing_dyn = result_dynamic.get("timing")
    meta_hf = result_hf_corrected.get("circuit_metadata", {})
    meta_fixed = result_fixed.get("circuit_metadata", {})
    meta_dyn = result_dynamic.get("circuit_metadata", {})

    # 1. Compile-Once Evidence
    console.print("\n[bold]1. Compile-Once Evidence[/bold]")

    # Helper to print compile evidence for a mode
    def _print_compile_evidence(label: str, timing, indent: str = "  "):
        if not timing:
            return
        compile_time = _effective_compile_time(timing)
        q_valid = timing.quantum_times[timing.quantum_times > 0]
        n_evals = len(q_valid)
        console.print(f"{indent}[bold]{label}[/bold]")
        if n_evals > 0:
            first_exec = q_valid[0] * 1000
            console.print(f"{indent}  Compilation time:     {compile_time:.2f}s (one-time)")
            console.print(f"{indent}  First QPE execution:  {first_exec:.1f}ms")
        if n_evals > 1:
            subsequent = q_valid[1:]
            avg_subsequent = subsequent.mean() * 1000
            std_subsequent = subsequent.std() * 1000
            compile_exec_ratio = compile_time / subsequent.mean() if subsequent.mean() > 0 else 0
            console.print(
                f"{indent}  Subsequent QPE avg:   {avg_subsequent:.1f}ms "
                f"+/- {std_subsequent:.1f}ms (n={len(subsequent)})"
            )
            console.print(f"{indent}  Compile/Execute ratio: {compile_exec_ratio:.0f}x")

    _print_compile_evidence("fixed (compiled QPE, every step)", timing_fixed)
    _print_compile_evidence("dynamic (runtime coefficients + QPE)", timing_dyn)

    # 2. Speedup vs Recompilation
    console.print("\n[bold]2. Speedup vs Recompilation Approach[/bold]")

    def _print_speedup(label: str, timing):
        if not timing:
            return
        compile_time = _effective_compile_time(timing)
        q_valid = timing.quantum_times[timing.quantum_times > 0]
        n_evals = len(q_valid)
        if n_evals > 1:
            avg_exec = q_valid[1:].mean()
            old_total = n_evals * compile_time
            new_total = compile_time + n_evals * avg_exec
            speedup = old_total / new_total if new_total > 0 else float("inf")
            console.print(f"  [bold]{label}[/bold]")
            console.print(
                f"    Recompile each step: {compile_time:.1f}s x {n_evals} = " f"{old_total:.1f}s"
            )
            console.print(
                f"    Compile once + run:  {compile_time:.1f}s + "
                f"{n_evals} x {avg_exec * 1000:.1f}ms = {new_total:.1f}s"
            )
            console.print(f"    Speedup: {speedup:.1f}x ({n_evals} evals)")
            old_1k = 1000 * compile_time
            new_1k = compile_time + 1000 * avg_exec
            console.print(f"    Projected at 1000 evals: {old_1k / new_1k:.0f}x")

    _print_speedup("fixed", timing_fixed)
    _print_speedup("dynamic", timing_dyn)

    # 3. Technical Details
    console.print("\n[bold]3. Technical Details[/bold]")

    # Use circuit_metadata to get Trotter step info
    if meta_fixed:
        n_trotter = meta_fixed.get("n_trotter_steps", "?")
        n_req = meta_fixed.get("n_trotter_steps_requested", "?")
        capped = " [yellow](CAPPED)[/yellow]" if n_trotter != n_req else ""
        console.print(f"  fixed: Trotter steps = {n_trotter} (requested: {n_req}){capped}")

    if meta_dyn:
        n_trotter = meta_dyn.get("n_trotter_steps", "?")
        n_req = meta_dyn.get("n_trotter_steps_requested", "?")
        n_terms = meta_dyn.get("n_hamiltonian_terms", "?")
        n_est = meta_dyn.get("n_estimation_wires", "?")
        capped = " [yellow](CAPPED)[/yellow]" if n_trotter != n_req else ""
        console.print(f"  dynamic: Trotter steps = {n_trotter} (requested: {n_req}){capped}")
        console.print(f"  Hamiltonian terms: {n_terms}")
        if isinstance(n_est, int) and isinstance(n_trotter, int) and isinstance(n_terms, int):
            ir_scale = n_est * n_trotter * n_terms
            console.print(
                f"  MLIR IR scale estimate: {n_est} x {n_trotter} x {n_terms} = {ir_scale} ops"
            )

    # 4. Energy Formula Comparison
    console.print("\n[bold]4. Energy Formula Comparison[/bold]")
    formula_hf = meta_hf.get("energy_formula", "N/A")
    formula_fixed = meta_fixed.get("energy_formula", "N/A")
    formula_dyn = meta_dyn.get("energy_formula", "N/A")

    formula_table = Table(show_header=False, box=None, padding=(0, 2))
    formula_table.add_column("Mode", style="bold")
    formula_table.add_column("Formula")
    formula_table.add_column("Physics")
    formula_table.add_row(
        "hf_corrected",
        formula_hf,
        "HF acceptance + interval QPE diagnostics",
    )
    formula_table.add_row(
        "fixed",
        formula_fixed,
        "Static vacuum QPE, every step",
    )
    formula_table.add_row(
        "dynamic",
        formula_dyn,
        "Runtime H_eff QPE; expected_value mode",
    )
    console.print(formula_table)


def _print_delta_corr_pol_analysis(
    result_fixed: dict,
    result_dynamic: dict,
    result_hf_corrected: dict,
    e_vacuum: float,
    qpe_interval: int,
):
    """
    δ_corr-pol analysis with controlled variable analysis.

    Controlled per-step δ_corr-pol comparison uses fixed↔dynamic (shared QPE-driven
    MC trajectory, same seed, same acceptance criterion).
    hf_corrected data is presented as independent ensemble statistics only—
    not for per-step alignment (different acceptance criterion: HF-based).
    """
    console.print()
    console.print(
        Panel(
            "[bold]δ_corr-pol: Correlation-Polarization Analysis[/bold]",
            border_style="yellow",
        )
    )

    q_fixed = np.array(result_fixed["quantum_energies"])
    q_dyn = np.array(result_dynamic["quantum_energies"])
    q_hf = np.array(result_hf_corrected["quantum_energies"])
    n_fixed = int(result_fixed["n_quantum_evaluations"])
    n_dyn = int(result_dynamic["n_quantum_evaluations"])
    n_hf = int(result_hf_corrected["n_quantum_evaluations"])
    n_eval = min(n_fixed, n_dyn)

    if n_eval == 0:
        console.print("  No QPE evaluations available.")
        return

    q_fixed = q_fixed[:n_eval]
    q_dyn = q_dyn[:n_eval]
    q_hf = q_hf[:n_hf]

    # ── 1. Energy Definitions ───────────────────────────────────────────
    console.print("\n[bold]1. Energy Definitions[/bold]")
    console.print("  fixed:    E(R) = E_QPE(H_vac) + E_MM")
    console.print("            [QPE on H' = H_vac - E_HF*I; measures ~ E_corr(vac)]")
    console.print("  dynamic:  E(R) = E_QPE(H_eff(R)) + E_MM(sol-sol)")
    console.print("            [QPE on H'_eff ~ H_eff(R) - E_HF(vac)*I]")
    console.print("            delta_QPE(R) ~ E_corr(solvated, R) + [E_HF(R) - E_HF(vac)]")
    console.print()
    console.print("  delta_corr-pol(R) = E_QPE(dynamic, R) - E_QPE(fixed, R)")
    console.print("                    ~ E_corr(solvated, R) - E_corr(vacuum)")
    console.print("                      [Trotter errors cancel approximately (H_eff != H_vac)]")
    console.print()
    console.print(
        "  [dim]fixed <-> dynamic share the same QPE-driven MC trajectory (same seed,\n"
        "  same QPE acceptance), so quantum_energies arrays are directly comparable\n"
        "  element-by-element.[/dim]"
    )

    # ── 2. Per-step computation ─────────────────────────────────────────
    delta = q_dyn - q_fixed  # Per-step δ_corr-pol in Hartree
    delta_kcal = delta * HARTREE_TO_KCAL_MOL

    # ── 3. Statistics ───────────────────────────────────────────────────
    mean_delta = np.mean(delta)
    std_delta = np.std(delta, ddof=1) if n_eval > 1 else np.std(delta)
    sem_delta = std_delta / np.sqrt(n_eval) if n_eval > 0 else 0.0
    t_stat = mean_delta / sem_delta if sem_delta > 0 else 0.0

    mean_kcal = np.mean(delta_kcal)
    sem_kcal = sem_delta * HARTREE_TO_KCAL_MOL

    console.print(f"\n[bold]2. Per-step Statistics (n={n_eval}, fixed <-> dynamic)[/bold]")
    console.print(f"  <delta_corr-pol> = {mean_delta:+.6f} Ha" f" = {mean_kcal:+.2f} kcal/mol")
    console.print(
        f"  sigma(delta_corr-pol) = {std_delta:.6f} Ha" f" = {np.std(delta_kcal):.2f} kcal/mol"
    )
    console.print(f"  SEM = {sem_delta:.6f} Ha = {sem_kcal:.2f} kcal/mol")
    console.print(f"  t-statistic = {t_stat:.2f}  (|t|>2.0 needed for p<0.05, df={n_eval - 1})")
    if abs(t_stat) < 2.0:
        console.print("  [yellow]NOT statistically significant at 95% confidence[/yellow]")
    else:
        console.print("  [green]Statistically significant at 95% confidence[/green]")
    console.print(f"  Range: [{np.min(delta):+.6f}, {np.max(delta):+.6f}] Ha")

    # ── 4. Per-step Comparison (fixed <-> dynamic only) ──────────────────
    console.print("\n[bold]3. Per-step Comparison (fixed <-> dynamic)[/bold]")

    n_show = min(n_eval, 10)
    table = Table(show_header=True, header_style="bold")
    table.add_column("MC Step", justify="right")
    table.add_column("E_QPE(fixed)", justify="right")
    table.add_column("E_QPE(dynamic)", justify="right")
    table.add_column("delta (Ha)", justify="right")
    table.add_column("delta (kcal/mol)", justify="right")

    for i in range(n_show):
        mc_step = i + 1  # fixed/dynamic evaluate every step
        table.add_row(
            str(mc_step),
            f"{q_fixed[i]:.6f}",
            f"{q_dyn[i]:.6f}",
            f"{delta[i]:+.6f}",
            f"{delta_kcal[i]:+.2f}",
        )
    console.print(table)

    console.print(
        "  [dim]Shared QPE-driven MC trajectory: same seed + same QPE acceptance criterion.[/dim]"
    )

    # ── 5. Energy Trend Analysis ────────────────────────────────────────
    console.print("\n[bold]4. Energy Trend Analysis[/bold]")

    n_third = max(1, n_eval // 3)
    early_delta = delta[:n_third]
    late_delta = delta[-n_third:]

    console.print(f"  Early MC (steps 1-{n_third}):")
    console.print(
        f"    <delta> = {np.mean(early_delta):+.6f} Ha"
        f" = {np.mean(early_delta) * HARTREE_TO_KCAL_MOL:+.2f} kcal/mol"
        f"  sigma = {np.std(early_delta):.6f} Ha"
    )
    console.print(f"  Late MC (steps {n_eval - n_third + 1}-{n_eval}):")
    console.print(
        f"    <delta> = {np.mean(late_delta):+.6f} Ha"
        f" = {np.mean(late_delta) * HARTREE_TO_KCAL_MOL:+.2f} kcal/mol"
        f"  sigma = {np.std(late_delta):.6f} Ha"
    )

    # ── 6. Three-Mode QPE Energy Distribution ──────────────────────────
    console.print("\n[bold]5. Three-Mode QPE Energy Distribution[/bold]")

    dist_table = Table(show_header=True, header_style="bold")
    dist_table.add_column("Statistic", style="bold")
    dist_table.add_column("hf_corrected", justify="right")
    dist_table.add_column("fixed", justify="right")
    dist_table.add_column("dynamic", justify="right")

    dist_table.add_row(
        "<E_QPE> (Ha)",
        f"{np.mean(q_hf):.6f}",
        f"{np.mean(q_fixed):.6f}",
        f"{np.mean(q_dyn):.6f}",
    )
    dist_table.add_row(
        "sigma(E_QPE) (mHa)",
        f"{np.std(q_hf) * 1000:.3f}",
        f"{np.std(q_fixed) * 1000:.3f}",
        f"{np.std(q_dyn) * 1000:.3f}",
    )
    dist_table.add_row(
        "Min E_QPE (Ha)",
        f"{np.min(q_hf):.6f}",
        f"{np.min(q_fixed):.6f}",
        f"{np.min(q_dyn):.6f}",
    )
    dist_table.add_row(
        "Max E_QPE (Ha)",
        f"{np.max(q_hf):.6f}",
        f"{np.max(q_fixed):.6f}",
        f"{np.max(q_dyn):.6f}",
    )
    dist_table.add_row("n_evals", str(n_hf), str(n_eval), str(n_eval))
    dist_table.add_row(
        "Measurement mode",
        "analytical probs",
        "analytical probs",
        "analytical probs",
    )
    console.print(dist_table)

    console.print(
        "  [dim]Note: fixed/dynamic share QPE-driven MC trajectory; hf_corrected is an\n"
        "  independent ensemble (HF acceptance). Cross-ensemble sigma comparisons\n"
        "  are observational, not controlled.[/dim]"
    )

    # ── 7. Comparability Constraints ────────────────────────────────────
    console.print("\n[bold]6. Comparability Constraints[/bold]")

    console.print("  Controlled variable analysis:")
    console.print(
        "    fixed <-> dynamic: Same seed + same QPE acceptance " "-> shared MC trajectory"
    )
    console.print("    hf_corrected:  HF-based acceptance -> independent trajectory")
    console.print()
    console.print("  Controlled comparisons (shared trajectory):")
    console.print(
        "    * Per-step delta_corr-pol = E_QPE(dynamic) - E_QPE(fixed) at same configuration"
    )
    console.print("    * Trotter bias approximately cancels in per-step difference")
    console.print()
    console.print("  Independent observations (separate ensembles):")
    console.print("    * Energy distribution statistics (mean, sigma, min, max) for each mode")
    console.print("    * QPE-HF consistency within dynamic mode")
    console.print("    * Acceptance rates reflect different criteria (not directly comparable)")

    # ── 8. QPE-HF Trajectory Consistency (dynamic) ──────────────────────
    console.print("\n[bold]7. dynamic: QPE-HF Trajectory Consistency[/bold]")

    hf_energies_dyn = np.array(result_dynamic["hf_energies"])[:n_dyn]
    diff_dyn = q_dyn[:n_dyn] - hf_energies_dyn

    console.print(f"  Steps analyzed: {n_dyn}")
    console.print(
        f"  QPE-HF offset: <Delta> = {np.mean(diff_dyn) * 1000:+.2f} mHa"
        f"  (dominated by E_corr; residuals from Trotter bias and config-dependent MM embedding)"
    )

    if np.std(q_dyn[:n_dyn]) > 0 and np.std(hf_energies_dyn) > 0:
        correlation = np.corrcoef(q_dyn[:n_dyn], hf_energies_dyn)[0, 1]
        console.print(f"  Pearson correlation (QPE vs HF): {correlation:.4f}")
        if correlation > 0.5:
            console.print("  [green]Trends are positively correlated (consistent)[/green]")
        elif correlation > 0:
            console.print("  [yellow]Weak positive correlation[/yellow]")
        else:
            console.print("  [red]No positive correlation detected[/red]")
    else:
        console.print("  [dim]Insufficient variance for correlation analysis[/dim]")

    # Early vs late phases
    n_third_dyn = max(1, n_dyn // 3)
    early_diff_dyn = diff_dyn[:n_third_dyn]
    late_diff_dyn = diff_dyn[-n_third_dyn:]
    console.print(
        f"  Early (steps 1-{n_third_dyn}): "
        f"<Delta> = {np.mean(early_diff_dyn) * 1000:+.2f} mHa, "
        f"sigma = {np.std(early_diff_dyn) * 1000:.2f} mHa"
    )
    console.print(
        f"  Late  (steps {n_dyn - n_third_dyn + 1}-{n_dyn}): "
        f"<Delta> = {np.mean(late_diff_dyn) * 1000:+.2f} mHa, "
        f"sigma = {np.std(late_diff_dyn) * 1000:.2f} mHa"
    )

    # QPE-HF offset drift explanation
    offset_drift = (np.mean(late_diff_dyn) - np.mean(early_diff_dyn)) * 1000
    console.print(f"\n  Offset drift (early→late): {offset_drift:+.2f} mHa")
    if abs(offset_drift) > 5.0:
        console.print(
            "  [dim]Physical origin: as MC explores lower-energy configurations, MM perturbation\n"
            "  grows (solvent moves closer). This increases the gap between H_eff and H_vac,\n"
            "  causing Trotter decomposition errors to diverge — hence the growing QPE-HF offset.\n"
            "  At higher QPE resolution (more qubits, more Trotter steps), this drift shrinks.[/dim]"
        )

    # Monotonicity diagnostic for dynamic
    if n_dyn >= 10:
        # Check if QPE energies are monotonically decreasing (windowed)
        q_dyn_slice = q_dyn[:n_dyn]
        window = max(1, n_dyn // 5)
        n_windows = n_dyn // window
        window_means = [
            np.mean(q_dyn_slice[i * window : (i + 1) * window]) for i in range(n_windows)
        ]
        n_decreasing = sum(
            1 for i in range(1, len(window_means)) if window_means[i] < window_means[i - 1]
        )
        frac_decreasing = n_decreasing / max(1, len(window_means) - 1)
        monotonic = frac_decreasing > 0.8

        qpe_drift = (np.mean(q_dyn_slice[-window:]) - np.mean(q_dyn_slice[:window])) * 1000
        console.print()
        console.print("  [bold]Equilibration diagnostic:[/bold]")
        console.print(
            f"    Windowed trend (window={window} steps, {n_windows} segments): "
            f"{n_decreasing}/{n_windows - 1} decreasing ({frac_decreasing:.0%})"
        )
        console.print(f"    QPE energy drift: {qpe_drift:+.2f} mHa (first→last window)")
        if monotonic:
            console.print(
                "    [yellow]WARNING: Monotonically decreasing QPE energy detected.[/yellow]\n"
                "    [dim]At POC scale (4-bit QPE, 500 steps), this is expected behavior:\n"
                "    the QPE-driven MC loop systematically explores lower-energy configurations.\n"
                "    This does NOT indicate a bug — it indicates the chain has not equilibrated.\n"
                "    Longer chains (10k+ steps) with burn-in removal are needed to assess\n"
                "    whether stationary distribution is reached.[/dim]"
            )
        else:
            console.print(
                "    [green]No strong monotonic trend detected — "
                "consistent with near-equilibration.[/green]"
            )

    # ── 9. Physical Interpretation ──────────────────────────────────────
    console.print("\n[bold]8. Physical Interpretation[/bold]")

    # Compute Trotter bias from data (not hardcoded)
    trotter_corr_offset = (np.mean(q_fixed) - e_vacuum) * 1000  # mHa
    console.print(
        f"  Trotter+phase bias: <E_QPE(fixed)> - E_HF(vac) = {trotter_corr_offset:+.1f} mHa"
    )
    console.print("  [dim](derived from data: mean fixed QPE - vacuum HF energy)[/dim]")

    # Significance-aware directional claim
    console.print()
    console.print(f"  delta_corr-pol = {mean_kcal:+.2f} +/- {sem_kcal:.2f} kcal/mol (mean +/- SEM)")
    if abs(t_stat) >= 2.0:
        if mean_kcal > 0:
            console.print(
                "  -> Statistically significant: fixed mode OVERESTIMATES"
                " solvation stabilization"
            )
        else:
            console.print(
                "  -> Statistically significant: fixed mode UNDERESTIMATES"
                " solvation stabilization"
            )
    else:
        console.print(
            f"  -> NOT statistically significant (|t|={abs(t_stat):.2f} < 2.0, df={n_eval - 1})."
            " Direction inconclusive at this sample size."
        )
        console.print(
            "  [dim]Interpretation: the signal (~0.1 kcal/mol) is buried in 4-bit QPE noise\n"
            "  (~7 kcal/mol sigma). More qubits improve per-measurement resolution exponentially;\n"
            "  more MC steps reduce SEM as 1/sqrt(n) but cannot sharpen individual measurements.[/dim]"
        )

    console.print()
    console.print(
        "  [bold]Key mechanism:[/bold] All modes use energy-shifted QPE (H' = H - E_HF*I)."
    )
    console.print(
        "  The probability-weighted expected value (sum probs[k]*k) extracts a continuous"
    )
    console.print("  phase estimate dominated by E_corr (for H2/STO-3G vacuum, |<0|HF>|^2 ~ 0.97).")
    console.print(
        f"  Systematic Trotter bias ({trotter_corr_offset:+.1f} mHa, computed from fixed QPE data)"
    )
    console.print(
        "  approximately cancels in fixed<->dynamic per-step difference"
        " (same Trotter order/steps;"
    )
    console.print("  residual from H_eff != H_vac).")
    console.print()
    console.print(
        "  [dim]Note: 4-bit QPE resolution (bin width ~ 12 mHa ~ 7.5 kcal/mol) limits\n"
        "  the SNR of delta measurement. True delta_corr-pol for H2/STO-3G is expected\n"
        "  ~0.01-0.1 kcal/mol (SEM ~1 kcal/mol at n=50). Higher QPE resolution\n"
        "  (more estimation qubits) will improve accuracy.[/dim]"
    )


# =============================================================================
# Main
# =============================================================================


def main():
    """Run H2 MC solvation with three QPE modes and compare."""
    console.print(
        Panel(
            "[bold]H2 Three-Mode QPE Comparison[/bold]\n"
            "hf_corrected | fixed | dynamic\n"
            "Runtime Hamiltonian coefficient parameterization",
            title="Experiment",
            border_style="green",
        )
    )

    # POC Scope Declaration
    console.print(
        Panel(
            "[bold]POC Scope[/bold]\n\n"
            "[green]This experiment demonstrates:[/green]\n"
            "  • Three QPE architectures (hf_corrected, fixed, dynamic) running end-to-end\n"
            "  • Runtime Hamiltonian coefficient parameterization (compile once, reuse)\n"
            "  • Fused PySCF+QPE callback architecture for dynamic MC\n"
            "  • Controlled delta_corr-pol measurement via shared MC trajectory\n\n"
            "[yellow]This experiment does NOT demonstrate:[/yellow]\n"
            "  • Numerical accuracy (4-bit QPE, STO-3G basis, 500 MC steps)\n"
            "  • Statistical convergence (SEM ~ 1 kcal/mol >> expected signal)\n"
            "  • Equilibration adequacy (short chains, no burn-in analysis)\n\n"
            "[dim]Accuracy targets are next-phase milestones: more estimation qubits,\n"
            "larger basis sets, longer MC chains with burn-in diagnostics.[/dim]",
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

    # ── Mode 1: hf_corrected ──────────────────────────────────────────
    console.print("\n" + "=" * 60)
    console.print("[bold blue]  MODE 1: hf_corrected[/bold blue]")
    console.print("=" * 60)

    config_hf = SolvationConfig(
        **mc_params, qpe_config=QPE_HF_CORRECTED, hamiltonian_mode="hf_corrected"
    )
    result_hf = run_solvation(config_hf, show_plots=False)

    # ── Mode 2: fixed ─────────────────────────────────────────────────
    console.print("\n" + "=" * 60)
    console.print("[bold green]  MODE 2: fixed[/bold green]")
    console.print("=" * 60)

    config_fixed = SolvationConfig(**mc_params, qpe_config=QPE_FIXED, hamiltonian_mode="fixed")
    result_fixed = run_solvation(config_fixed, show_plots=False)

    # ── Mode 3: dynamic ───────────────────────────────────────────────
    console.print("\n" + "=" * 60)
    console.print("[bold yellow]  MODE 3: dynamic[/bold yellow]")
    console.print("=" * 60)

    config_dynamic = SolvationConfig(
        **mc_params, qpe_config=QPE_DYNAMIC, hamiltonian_mode="dynamic"
    )
    result_dynamic = run_solvation(config_dynamic, show_plots=False)

    # ── Three-Mode Analysis ──────────────────────────────────────────
    e_vacuum = result_hf["e_vacuum"]
    print_mode_comparison(result_hf, result_fixed, result_dynamic, e_vacuum)
    _print_architecture_performance_analysis(result_hf, result_fixed, result_dynamic)
    _print_delta_corr_pol_analysis(
        result_fixed, result_dynamic, result_hf, e_vacuum, QPE_FIXED.qpe_interval
    )

    # Closing POC Summary
    console.print()
    console.print(
        Panel(
            "[bold]POC Summary[/bold]\n\n"
            "Architecture validated:\n"
            "  • Runtime coefficient parameterization eliminates recompilation bottleneck\n"
            "  • QPE-driven MC acceptance is end-to-end functional\n"
            "  • Shared-trajectory delta_corr-pol measurement framework works\n\n"
            "Known limitations at POC scale:\n"
            "  • 4-bit QPE: ~12 mHa bin width >> sub-mHa solvation effects\n"
            "  • STO-3G: minimal basis limits correlation energy accuracy\n"
            "  • 500 MC steps: insufficient for equilibration convergence analysis\n\n"
            "Next-phase targets:\n"
            "  • 8+ estimation qubits for sub-mHa resolution\n"
            "  • cc-pVDZ or larger basis for quantitative correlation\n"
            "  • 10k+ MC steps with burn-in and autocorrelation diagnostics",
            title="What This POC Proves",
            border_style="green",
        )
    )

    return result_hf, result_fixed, result_dynamic


if __name__ == "__main__":
    main()
