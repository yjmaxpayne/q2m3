#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
H2 Three-Mode QPE Comparison: MC Solvation Experiment

Side-by-side comparison of three QPE modes for MC solvation:

  Mode 1 (vacuum_correction): E = E_QPE(vacuum) + delta_E_HF(MM)
    - Pre-compiled QPE circuit, reused for all evaluations
    - Approximate: ignores correlation-polarization coupling (delta_corr-pol)

  Mode 2 (mm_embedded): E = E_QPE(H_eff with MM embedding)
    - Runtime Hamiltonian coefficient parameterization (benchmark-validated)
    - Compile once (~219s for H2), execute with new coeffs (~45ms, no recompile)
    - More complete: QPE Hamiltonian includes diagonal MM corrections at every configuration

  Mode 3 (qpe_driven): E = E_QPE(H_eff) + E_MM(sol-sol)
    - QPE energy directly drives Metropolis acceptance criterion
    - Every MC step runs QPE (no interval-based scheduling)
    - Fused callback consolidates coefficient + energy in single PySCF call

Phase extraction: All modes use qml.probs() + probability-weighted expected
value (Σ probs[k]·k / 2^n) to convert measurement distributions to continuous
phase estimates. This preserves sensitivity to sub-bin MM corrections (~0.1 mHa),
unlike argmax which discretizes to integer bins (resolution ~12 mHa for 4-bit QPE).

Key technical requirement: TrotterProduct must use check_hermitian=False
when coefficients are JAX-traceable runtime parameters.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from examples.mc_solvation import (
    MoleculeConfig,
    QPEConfig,
    SolvationConfig,
    run_solvation,
)
from examples.mc_solvation.constants import HARTREE_TO_KCAL_MOL

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

# vacuum_correction: static Hamiltonian → compiler constant-folds gates → no memory issue
QPE_VACUUM = QPEConfig(
    n_estimation_wires=4,
    n_trotter_steps=10,
    n_shots=100,
    qpe_interval=10,
    target_resolution=0.003,
    energy_range=0.2,
    use_catalyst=True,
)

# mm_embedded: runtime-parameterized coefficients → Catalyst cannot constant-fold
# → symbolic IR scales as n_est × n_trotter × n_terms
# Validated: n_trotter=10 compiles in ~219s for H2 (4 estimation wires, 15 terms).
QPE_MM_EMBEDDED = QPEConfig(
    n_estimation_wires=4,
    n_trotter_steps=10,
    n_shots=100,
    qpe_interval=10,
    target_resolution=0.003,
    energy_range=0.2,
    use_catalyst=True,
)

# qpe_driven: same runtime-coefficient architecture as mm_embedded,
# but QPE energy drives Metropolis acceptance at every MC step.
# n_shots=0 for expected_value mode; qpe_interval is ignored (every step).
QPE_QPE_DRIVEN = QPEConfig(
    n_estimation_wires=4,
    n_trotter_steps=10,
    n_shots=0,
    qpe_interval=1,
    target_resolution=0.003,
    energy_range=0.2,
    use_catalyst=True,
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


def print_mode_comparison(result_vac: dict, result_emb: dict, result_driven: dict, e_vacuum: float):
    """Print three-mode comparison of vacuum_correction, mm_embedded, and qpe_driven.

    Note: 'Best QPE Energy' selects the minimum across all MC evaluations.
    Statistical outliers are possible due to 4-bit phase resolution noise.
    The per-step δ_corr-pol analysis provides a more reliable comparison.
    """
    console.print()
    console.print(
        Panel("[bold]Mode Comparison: Three QPE Architectures[/bold]", border_style="magenta")
    )

    modes = [
        ("vacuum_correction", result_vac),
        ("mm_embedded", result_emb),
        ("qpe_driven", result_driven),
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
    qpe_vals = [float(r["best_qpe_energy"]) for _, r in modes]
    n_evals = [int(r["n_quantum_evaluations"]) for _, r in modes]
    table.add_row(
        "Best QPE Energy (Ha)",
        *[f"{v:.6f} (n={n})" if v < 1e9 else "N/A" for v, n in zip(qpe_vals, n_evals)],
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
        if name == "qpe_driven":
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
        "  [dim]*qpe_driven: Best MC Energy = E_QPE + E_MM(sol-sol) (QPE-driven acceptance).\n"
        "   vac/emb: Best MC Energy = E_HF(solvated) + E_MM(sol-sol) (HF-based acceptance).\n"
        "  **dE vs HF(vac) = E_HF(vacuum) - Best_QPE_Energy. This mixes theory levels:\n"
        "    QPE energies include correlation, so dE contains |E_corr(vac)| + solvation.\n"
        "    For pure solvation energy, compare QPE energies across modes (delta_corr-pol).\n"
        "  Note: qpe_driven runs QPE every step (vs interval for vac/emb). Monotonically\n"
        "    decreasing QPE energy may indicate incomplete equilibration.[/dim]"
    )

    # Compile-once architecture summary (three-mode)
    if all(t is not None for t in timings):
        timing_vac, timing_emb, timing_drv = timings
        compile_vac = _effective_compile_time(timing_vac)
        compile_emb = _effective_compile_time(timing_emb)
        compile_drv = _effective_compile_time(timing_drv)

        q_emb_valid = timing_emb.quantum_times[timing_emb.quantum_times > 0]
        avg_emb = float(q_emb_valid[1:].mean() * 1000) if len(q_emb_valid) > 1 else 0.0

        q_vac_valid = timing_vac.quantum_times[timing_vac.quantum_times > 0]
        avg_vac = float(q_vac_valid[1:].mean() * 1000) if len(q_vac_valid) > 1 else 0.0

        q_drv_valid = timing_drv.quantum_times[timing_drv.quantum_times > 0]
        avg_drv = float(q_drv_valid[1:].mean() * 1000) if len(q_drv_valid) > 1 else 0.0

        arch_summary = (
            f"[bold]Compile-Once Architecture Comparison[/bold]\n"
            f"  vacuum_correction: Fixed H_vac  -> compile {compile_vac:.1f}s, "
            f"per-eval ~{avg_vac:.1f}ms (QPE + PySCF callback)\n"
            f"  mm_embedded:       Runtime H_eff -> compile {compile_emb:.1f}s, "
            f"per-eval ~{avg_emb:.1f}ms (PySCF callback + QPE)\n"
            f"  qpe_driven:        Runtime H_eff -> QPE precompile {compile_drv:.1f}s, "
            f"QPE only ~{avg_drv:.1f}ms/step (PySCF tracked separately)"
        )
        console.print()
        console.print(Panel(arch_summary, border_style="green"))


def _print_architecture_performance_analysis(
    result_vac: dict, result_emb: dict, result_driven: dict
):
    """Print detailed compile-once architecture analysis with performance evidence.

    Quantifies the speedup from runtime coefficient parameterization by comparing
    the compile-once model (one compilation + many fast executions) against the
    hypothetical recompile-each-step model (full compilation per MC evaluation).
    Includes qpe_driven mode which uses precompiled QPE + pure Python MC loop.
    """
    from examples.mc_solvation.orchestrator import _MAX_TROTTER_STEPS_RUNTIME

    console.print()
    console.print(
        Panel("[bold]Runtime Coefficient Architecture Analysis[/bold]", border_style="cyan")
    )

    timing_emb = result_emb.get("timing")
    timing_vac = result_vac.get("timing")
    timing_drv = result_driven.get("timing")
    meta_emb = result_emb.get("circuit_metadata", {})
    meta_vac = result_vac.get("circuit_metadata", {})
    meta_drv = result_driven.get("circuit_metadata", {})

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

    _print_compile_evidence("mm_embedded (@qjit MC loop)", timing_emb)
    _print_compile_evidence("qpe_driven (precompiled QPE + Python MC)", timing_drv)

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

    _print_speedup("mm_embedded", timing_emb)
    _print_speedup("qpe_driven", timing_drv)

    # 3. Technical Details
    console.print("\n[bold]3. Technical Details[/bold]")
    console.print(f"  _MAX_TROTTER_STEPS_RUNTIME = {_MAX_TROTTER_STEPS_RUNTIME}")

    if meta_emb:
        n_trotter = meta_emb.get("n_trotter_steps", "?")
        n_req = meta_emb.get("n_trotter_steps_requested", "?")
        n_terms = meta_emb.get("n_hamiltonian_terms", "?")
        n_est = meta_emb.get("n_estimation_wires", "?")
        capped = " [yellow](CAPPED)[/yellow]" if n_trotter != n_req else ""
        console.print(f"  Trotter steps: {n_trotter} (requested: {n_req}){capped}")
        console.print(f"  Hamiltonian terms: {n_terms}")
        if isinstance(n_est, int) and isinstance(n_trotter, int) and isinstance(n_terms, int):
            ir_scale = n_est * n_trotter * n_terms
            console.print(
                f"  MLIR IR scale estimate: {n_est} x {n_trotter} x {n_terms} = {ir_scale} ops"
            )

    # 4. Energy Formula Comparison
    console.print("\n[bold]4. Energy Formula Comparison[/bold]")
    formula_vac = meta_vac.get("energy_formula", "N/A")
    formula_emb = meta_emb.get("energy_formula", "N/A")
    formula_drv = meta_drv.get("energy_formula", "N/A")

    formula_table = Table(show_header=False, box=None, padding=(0, 2))
    formula_table.add_column("Mode", style="bold")
    formula_table.add_column("Formula")
    formula_table.add_column("Physics")
    formula_table.add_row(
        "vacuum_correction",
        formula_vac,
        "Approximate: ignores delta_corr-pol",
    )
    formula_table.add_row(
        "mm_embedded",
        formula_emb,
        "Diagonal MM embedding in QPE Hamiltonian",
    )
    formula_table.add_row(
        "qpe_driven",
        formula_drv,
        "QPE directly drives MC; expected_value mode",
    )
    console.print(formula_table)


def _print_delta_corr_pol_analysis(
    result_vac: dict,
    result_emb: dict,
    result_driven: dict,
    e_vacuum: float,
    qpe_interval: int,
):
    """
    δ_corr-pol analysis with controlled variable analysis.

    Controlled per-step δ_corr-pol comparison uses vac↔emb (shared MC trajectory).
    qpe_driven data is presented as independent ensemble statistics only—
    not for per-step alignment (different RNG + QPE-driven acceptance).
    """
    console.print()
    console.print(
        Panel(
            "[bold]δ_corr-pol: Correlation-Polarization Analysis[/bold]",
            border_style="yellow",
        )
    )

    q_vac = np.array(result_vac["quantum_energies"])
    q_emb = np.array(result_emb["quantum_energies"])
    q_drv = np.array(result_driven["quantum_energies"])
    n_vac = int(result_vac["n_quantum_evaluations"])
    n_emb = int(result_emb["n_quantum_evaluations"])
    n_drv = int(result_driven["n_quantum_evaluations"])
    n_eval = min(n_vac, n_emb)

    if n_eval == 0:
        console.print("  No QPE evaluations available.")
        return

    q_vac = q_vac[:n_eval]
    q_emb = q_emb[:n_eval]
    q_drv = q_drv[:n_drv]

    # ── 1. Energy Definitions ───────────────────────────────────────────
    console.print("\n[bold]1. Energy Definitions[/bold]")
    console.print("  vacuum_correction:  E(R) = E_corr(vac) + E_HF(R)")
    console.print("                      [QPE on H' = H_vac - E_HF*I; measures ~ E_corr(vac)]")
    console.print("  mm_embedded:        E(R) = delta_QPE(R) + E_HF(vac)")
    console.print("                      [QPE on H'_eff = H_eff(R) - E_HF(vac)*I]")
    console.print("                      delta_QPE(R) = E_exact(H_eff) - E_HF(vac)")
    console.print(
        "                                   = E_corr(solvated, R) + [E_HF(R) - E_HF(vac)]"
    )
    console.print()
    console.print("  delta_corr-pol(R) = E(mm_emb, R) - E(vac_corr, R)")
    console.print("                    = E_corr(solvated, R) - E_corr(vacuum)")
    console.print("                      [E_HF(R) terms cancel in the subtraction]")
    console.print()
    console.print(
        "  [dim]vac <-> emb share the same MC trajectory (same seed, same HF acceptance),\n"
        "  so quantum_energies arrays are directly comparable element-by-element.[/dim]"
    )

    # ── 2. Per-step computation ─────────────────────────────────────────
    delta = q_emb - q_vac  # Per-step δ_corr-pol in Hartree
    delta_kcal = delta * HARTREE_TO_KCAL_MOL

    # ── 3. Statistics ───────────────────────────────────────────────────
    console.print(f"\n[bold]2. Per-step Statistics (n={n_eval}, vac <-> emb)[/bold]")
    console.print(
        f"  <delta_corr-pol> = {np.mean(delta):+.6f} Ha" f" = {np.mean(delta_kcal):+.2f} kcal/mol"
    )
    console.print(
        f"  sigma(delta_corr-pol) = {np.std(delta):.6f} Ha" f" = {np.std(delta_kcal):.2f} kcal/mol"
    )
    console.print(f"  Range: [{np.min(delta):+.6f}, {np.max(delta):+.6f}] Ha")

    # ── 4. Per-step Comparison (vac <-> emb only) ───────────────────────
    console.print(f"\n[bold]3. Per-step Comparison (vac <-> emb)[/bold]")

    n_show = min(n_eval, 10)
    table = Table(show_header=True, header_style="bold")
    table.add_column("MC Step", justify="right")
    table.add_column("E_QPE(vac)", justify="right")
    table.add_column("E_QPE(emb)", justify="right")
    table.add_column("delta (Ha)", justify="right")
    table.add_column("delta (kcal/mol)", justify="right")

    for i in range(n_show):
        mc_step = (i + 1) * qpe_interval
        table.add_row(
            str(mc_step),
            f"{q_vac[i]:.6f}",
            f"{q_emb[i]:.6f}",
            f"{delta[i]:+.6f}",
            f"{delta_kcal[i]:+.2f}",
        )
    console.print(table)

    console.print(
        "  [dim]Shared MC trajectory: same LCG RNG (seed=42) + same HF acceptance criterion.[/dim]"
    )

    # ── 5. Energy Trend Analysis ────────────────────────────────────────
    console.print(f"\n[bold]4. Energy Trend Analysis[/bold]")

    n_third = max(1, n_eval // 3)
    early_delta = delta[:n_third]
    late_delta = delta[-n_third:]

    console.print(f"  Early MC (steps {qpe_interval}-{n_third * qpe_interval}):")
    console.print(
        f"    <delta> = {np.mean(early_delta):+.6f} Ha"
        f" = {np.mean(early_delta) * HARTREE_TO_KCAL_MOL:+.2f} kcal/mol"
        f"  sigma = {np.std(early_delta):.6f} Ha"
    )
    console.print(
        f"  Late MC (steps "
        f"{(n_eval - n_third) * qpe_interval + qpe_interval}-{n_eval * qpe_interval}):"
    )
    console.print(
        f"    <delta> = {np.mean(late_delta):+.6f} Ha"
        f" = {np.mean(late_delta) * HARTREE_TO_KCAL_MOL:+.2f} kcal/mol"
        f"  sigma = {np.std(late_delta):.6f} Ha"
    )

    # ── 6. Three-Mode QPE Energy Distribution ──────────────────────────
    console.print(f"\n[bold]5. Three-Mode QPE Energy Distribution[/bold]")

    dist_table = Table(show_header=True, header_style="bold")
    dist_table.add_column("Statistic", style="bold")
    dist_table.add_column("vacuum_correction", justify="right")
    dist_table.add_column("mm_embedded", justify="right")
    dist_table.add_column("qpe_driven", justify="right")

    dist_table.add_row(
        "<E_QPE> (Ha)",
        f"{np.mean(q_vac):.6f}",
        f"{np.mean(q_emb):.6f}",
        f"{np.mean(q_drv):.6f}",
    )
    dist_table.add_row(
        "sigma(E_QPE) (mHa)",
        f"{np.std(q_vac) * 1000:.3f}",
        f"{np.std(q_emb) * 1000:.3f}",
        f"{np.std(q_drv) * 1000:.3f}",
    )
    dist_table.add_row(
        "Min E_QPE (Ha)",
        f"{np.min(q_vac):.6f}",
        f"{np.min(q_emb):.6f}",
        f"{np.min(q_drv):.6f}",
    )
    dist_table.add_row(
        "Max E_QPE (Ha)",
        f"{np.max(q_vac):.6f}",
        f"{np.max(q_emb):.6f}",
        f"{np.max(q_drv):.6f}",
    )
    dist_table.add_row("n_evals", str(n_eval), str(n_eval), str(n_drv))
    dist_table.add_row(
        "Measurement mode",
        "analytical probs",
        "analytical probs",
        "analytical probs",
    )
    console.print(dist_table)

    console.print(
        "  [dim]Note: vac/emb share MC trajectory; qpe_driven is an independent ensemble.\n"
        "  Cross-ensemble sigma comparisons are observational, not controlled.[/dim]"
    )

    # ── 7. Comparability Constraints ────────────────────────────────────
    console.print(f"\n[bold]6. Comparability Constraints[/bold]")

    console.print("  Controlled variable analysis:")
    console.print(
        "    vac <-> emb: Same LCG RNG (seed=42) + same HF acceptance " "-> shared MC trajectory"
    )
    console.print("    qpe_driven:  NumPy RNG + QPE-driven acceptance " "-> independent trajectory")
    console.print()
    console.print("  Controlled comparisons (shared trajectory):")
    console.print("    * Per-step delta_corr-pol = E_QPE(emb) - E_QPE(vac) at same configuration")
    console.print("    * Trotter bias approximately cancels in per-step difference")
    console.print()
    console.print("  Independent observations (separate ensembles):")
    console.print("    * Energy distribution statistics (mean, sigma, min, max) for each mode")
    console.print("    * QPE-HF consistency within qpe_driven mode")
    console.print("    * Acceptance rates reflect different criteria (not directly comparable)")

    # ── 8. QPE-HF Trajectory Consistency (qpe_driven) ──────────────────
    console.print(f"\n[bold]7. qpe_driven: QPE-HF Trajectory Consistency[/bold]")

    hf_energies_drv = np.array(result_driven["hf_energies"])[:n_drv]
    diff_drv = q_drv - hf_energies_drv

    console.print(f"  Steps analyzed: {n_drv}")
    console.print(
        f"  QPE-HF offset: <Delta> = {np.mean(diff_drv) * 1000:+.2f} mHa"
        f"  (Trotter bias + correlation + configuration-dependent MM embedding)"
    )

    if np.std(q_drv) > 0 and np.std(hf_energies_drv) > 0:
        correlation = np.corrcoef(q_drv, hf_energies_drv)[0, 1]
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
    n_third_drv = max(1, n_drv // 3)
    early_diff_drv = diff_drv[:n_third_drv]
    late_diff_drv = diff_drv[-n_third_drv:]
    console.print(
        f"  Early (steps 1-{n_third_drv}): "
        f"<Delta> = {np.mean(early_diff_drv) * 1000:+.2f} mHa, "
        f"sigma = {np.std(early_diff_drv) * 1000:.2f} mHa"
    )
    console.print(
        f"  Late  (steps {n_drv - n_third_drv + 1}-{n_drv}): "
        f"<Delta> = {np.mean(late_diff_drv) * 1000:+.2f} mHa, "
        f"sigma = {np.std(late_diff_drv) * 1000:.2f} mHa"
    )

    # ── 9. Physical Interpretation ──────────────────────────────────────
    console.print(f"\n[bold]8. Physical Interpretation[/bold]")
    avg_kcal = np.mean(delta_kcal)

    if avg_kcal > 0:
        console.print(
            f"  delta_corr-pol = {avg_kcal:+.2f} kcal/mol " f"(positive, mm_embedded energy higher)"
        )
        console.print("  -> vacuum_correction OVERESTIMATES solvation stabilization")
    else:
        console.print(
            f"  delta_corr-pol = {avg_kcal:+.2f} kcal/mol " f"(negative, mm_embedded energy lower)"
        )
        console.print("  -> vacuum_correction UNDERESTIMATES solvation stabilization")

    console.print()
    console.print(
        "  [bold]Key mechanism:[/bold] All modes use energy-shifted QPE (H' = H - E_HF*I)."
    )
    console.print(
        "  The probability-weighted expected value (sum probs[k]*k) extracts a continuous"
    )
    console.print("  phase estimate dominated by E_corr (for H2/STO-3G, |<0|HF>|^2 ~ 0.97).")
    console.print(
        "  Systematic Trotter bias (~28.6 mHa, vacuum reference) approximately cancels in vac<->emb"
    )
    console.print("  per-step difference (same Trotter order/steps; residual from H_eff != H_vac).")
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
            "vacuum_correction | mm_embedded | qpe_driven\n"
            "Runtime Hamiltonian coefficient parameterization",
            title="Experiment",
            border_style="green",
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

    # ── Mode 1: vacuum_correction ──────────────────────────────────────
    console.print("\n" + "=" * 60)
    console.print("[bold blue]  MODE 1: vacuum_correction[/bold blue]")
    console.print("=" * 60)

    config_vac = SolvationConfig(**mc_params, qpe_config=QPE_VACUUM, qpe_mode="vacuum_correction")
    result_vac = run_solvation(config_vac, show_plots=False)

    # ── Mode 2: mm_embedded ────────────────────────────────────────────
    console.print("\n" + "=" * 60)
    console.print("[bold green]  MODE 2: mm_embedded[/bold green]")
    console.print("=" * 60)

    config_emb = SolvationConfig(**mc_params, qpe_config=QPE_MM_EMBEDDED, qpe_mode="mm_embedded")
    result_emb = run_solvation(config_emb, show_plots=False)

    # ── Mode 3: qpe_driven ─────────────────────────────────────────────
    console.print("\n" + "=" * 60)
    console.print("[bold yellow]  MODE 3: qpe_driven[/bold yellow]")
    console.print("=" * 60)

    config_driven = SolvationConfig(**mc_params, qpe_config=QPE_QPE_DRIVEN, qpe_mode="qpe_driven")
    result_driven = run_solvation(config_driven, show_plots=False)

    # ── Three-Mode Analysis ──────────────────────────────────────────
    e_vacuum = result_vac["e_vacuum"]
    print_mode_comparison(result_vac, result_emb, result_driven, e_vacuum)
    _print_architecture_performance_analysis(result_vac, result_emb, result_driven)
    _print_delta_corr_pol_analysis(
        result_vac, result_emb, result_driven, e_vacuum, QPE_VACUUM.qpe_interval
    )

    console.print("\n" + "=" * 60)
    console.print("  Experiment complete.")
    console.print("=" * 60)

    return result_vac, result_emb, result_driven


if __name__ == "__main__":
    main()
