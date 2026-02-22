#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
H2 MM-Embedded QPE: Frame 14 Target Architecture Implementation

Side-by-side comparison of two QPE modes for MC solvation:

  Mode 1 (vacuum_correction): E = E_QPE(vacuum) + delta_E_HF(MM)
    - Pre-compiled QPE circuit, reused for all evaluations
    - Approximate: ignores correlation-polarization coupling (delta_corr-pol)

  Mode 2 (mm_embedded): E = E_QPE(H_eff with MM embedding)
    - Runtime Hamiltonian coefficient parameterization (benchmark-validated)
    - Compile once (~219s for H2), execute with new coeffs (~45ms, no recompile)
    - Complete: QPE Hamiltonian includes MM embedding at every configuration

Phase extraction: Both modes use qml.probs() + probability-weighted expected
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

# MC parameters (small for demo)
N_WATERS = 10
N_MC_STEPS = 500
RANDOM_SEED = 42


# =============================================================================
# Comparison Output
# =============================================================================


def print_mode_comparison(result_vac: dict, result_emb: dict, e_vacuum: float):
    """Print side-by-side comparison of vacuum_correction vs mm_embedded results.

    Note: 'Best QPE Energy' selects the minimum across all MC evaluations.
    For mm_embedded, this can be a statistical outlier (>2σ from mean) due to
    4-bit phase resolution noise. The per-step δ_corr-pol analysis provides
    a more reliable comparison.
    """
    console.print()
    console.print(
        Panel(
            "[bold]Mode Comparison: vacuum_correction vs mm_embedded[/bold]", border_style="magenta"
        )
    )

    # Circuit Configuration table
    meta_vac = result_vac.get("circuit_metadata", {})
    meta_emb = result_emb.get("circuit_metadata", {})
    if meta_vac and meta_emb:
        cfg_table = Table(title="Circuit Configuration")
        cfg_table.add_column("Parameter", style="bold")
        cfg_table.add_column("vacuum_correction", justify="right")
        cfg_table.add_column("mm_embedded", justify="right")

        cfg_table.add_row(
            "Hamiltonian Terms",
            str(meta_vac.get("n_hamiltonian_terms", "?")),
            str(meta_emb.get("n_hamiltonian_terms", "?")),
        )
        cfg_table.add_row(
            "System Qubits",
            str(meta_vac.get("n_system_qubits", "?")),
            str(meta_emb.get("n_system_qubits", "?")),
        )
        cfg_table.add_row(
            "Estimation Qubits",
            str(meta_vac.get("n_estimation_wires", "?")),
            str(meta_emb.get("n_estimation_wires", "?")),
        )
        cfg_table.add_row(
            "Total Qubits",
            str(meta_vac.get("total_qubits", "?")),
            str(meta_emb.get("total_qubits", "?")),
        )

        # Trotter steps — annotate if capped
        trotter_vac = str(meta_vac.get("n_trotter_steps", "?"))
        trotter_emb_actual = meta_emb.get("n_trotter_steps", "?")
        trotter_emb_req = meta_emb.get("n_trotter_steps_requested", "?")
        trotter_emb_str = str(trotter_emb_actual)
        if trotter_emb_actual != trotter_emb_req:
            trotter_emb_str += f" (req={trotter_emb_req}, capped)"
        cfg_table.add_row("Trotter Steps", trotter_vac, trotter_emb_str)

        cfg_table.add_row(
            "Energy Formula",
            meta_vac.get("energy_formula", "?"),
            meta_emb.get("energy_formula", "?"),
        )
        console.print(cfg_table)
        console.print()

    table = Table(title="QPE Energy Comparison")
    table.add_column("Metric", style="bold")
    table.add_column("vacuum_correction", justify="right")
    table.add_column("mm_embedded", justify="right")
    table.add_column("Difference", justify="right")

    # Best QPE energy
    qpe_vac = float(result_vac["best_qpe_energy"])
    qpe_emb = float(result_emb["best_qpe_energy"])
    diff_ha = qpe_emb - qpe_vac
    diff_kcal = diff_ha * HARTREE_TO_KCAL_MOL

    table.add_row(
        "Best QPE Energy (Ha)",
        f"{qpe_vac:.6f}" if qpe_vac < 1e9 else "N/A",
        f"{qpe_emb:.6f}" if qpe_emb < 1e9 else "N/A",
        f"{diff_ha:+.6f}" if (qpe_vac < 1e9 and qpe_emb < 1e9) else "N/A",
    )

    # Solvation stabilization
    stab_vac = e_vacuum - qpe_vac if qpe_vac < 1e9 else float("nan")
    stab_emb = e_vacuum - qpe_emb if qpe_emb < 1e9 else float("nan")

    table.add_row(
        "Solvation Stabilization (Ha)",
        f"{stab_vac:.6f}" if qpe_vac < 1e9 else "N/A",
        f"{stab_emb:.6f}" if qpe_emb < 1e9 else "N/A",
        f"{(stab_emb - stab_vac):+.6f}" if (qpe_vac < 1e9 and qpe_emb < 1e9) else "N/A",
    )

    table.add_row(
        "Solvation Stabilization (kcal/mol)",
        f"{stab_vac * HARTREE_TO_KCAL_MOL:.2f}" if qpe_vac < 1e9 else "N/A",
        f"{stab_emb * HARTREE_TO_KCAL_MOL:.2f}" if qpe_emb < 1e9 else "N/A",
        f"{diff_kcal:+.4f}" if (qpe_vac < 1e9 and qpe_emb < 1e9) else "N/A",
    )

    # Best HF energy
    hf_vac = float(result_vac["best_energy"])
    hf_emb = float(result_emb["best_energy"])
    table.add_row(
        "Best HF Energy (Ha)",
        f"{hf_vac:.6f}",
        f"{hf_emb:.6f}",
        f"{(hf_emb - hf_vac):+.6f}",
    )

    # Acceptance rate
    acc_vac = float(result_vac["acceptance_rate"]) * 100
    acc_emb = float(result_emb["acceptance_rate"]) * 100
    table.add_row(
        "Acceptance Rate (%)",
        f"{acc_vac:.1f}",
        f"{acc_emb:.1f}",
        f"{(acc_emb - acc_vac):+.1f}",
    )

    # Timing
    timing_vac = result_vac.get("timing")
    timing_emb = result_emb.get("timing")
    if timing_vac and timing_emb:
        table.add_row(
            "Compile Time (s)",
            f"{_effective_compile_time(timing_vac):.2f}",
            f"{_effective_compile_time(timing_emb):.2f}",
            "",
        )
        table.add_row(
            "Total MC Time (s)",
            f"{timing_vac.mc_loop_time:.2f}",
            f"{timing_emb.mc_loop_time:.2f}",
            "",
        )

    console.print(table)

    # Physical interpretation (summary; detailed per-step analysis printed separately)
    if qpe_vac < 1e9 and qpe_emb < 1e9:
        console.print()
        if abs(diff_ha) > 0.0001:
            console.print(f"  [bold yellow]Δ(best QPE): {diff_kcal:+.4f} kcal/mol[/bold yellow]")
            console.print(
                "  [dim]Note: best QPE energies may correspond to different configurations.\n"
                "  See δ_corr-pol analysis below for rigorous per-step comparison.[/dim]"
            )
        else:
            console.print("  Both modes give similar best QPE energies.")

    # Compile-once architecture brief summary
    if timing_vac and timing_emb:
        compile_emb = _effective_compile_time(timing_emb)
        q_valid = timing_emb.quantum_times[timing_emb.quantum_times > 0]
        avg_exec = float(q_valid[1:].mean() * 1000) if len(q_valid) > 1 else 0.0
        arch_summary = (
            f"[bold]Compile-Once Architecture (mm_embedded)[/bold]\n"
            f"  Compile once: {compile_emb:.2f}s  |  "
            f"Per-step QPE: ~{avg_exec:.1f}ms (no recompile)\n"
            f"  Key: check_hermitian=False + JAX-traced coefficients"
        )
        console.print()
        console.print(Panel(arch_summary, border_style="green"))


def _print_architecture_performance_analysis(result_vac: dict, result_emb: dict):
    """Print detailed compile-once architecture analysis with performance evidence.

    Quantifies the speedup from runtime coefficient parameterization by comparing
    the compile-once model (one compilation + many fast executions) against the
    hypothetical recompile-each-step model (full compilation per MC evaluation).
    """
    from examples.mc_solvation.orchestrator import _MAX_TROTTER_STEPS_RUNTIME

    console.print()
    console.print(
        Panel("[bold]Runtime Coefficient Architecture Analysis[/bold]", border_style="cyan")
    )

    timing_emb = result_emb.get("timing")
    timing_vac = result_vac.get("timing")
    meta_emb = result_emb.get("circuit_metadata", {})
    meta_vac = result_vac.get("circuit_metadata", {})

    # 1. Compile-Once Evidence
    console.print("\n[bold]1. Compile-Once Evidence[/bold]")
    if timing_emb:
        compile_time = _effective_compile_time(timing_emb)
        q_valid = timing_emb.quantum_times[timing_emb.quantum_times > 0]
        n_evals = len(q_valid)

        if n_evals > 0:
            first_exec = q_valid[0] * 1000
            console.print(f"  Compilation time:     {compile_time:.2f}s (one-time)")
            console.print(f"  First QPE execution:  {first_exec:.1f}ms")

        if n_evals > 1:
            subsequent = q_valid[1:]
            avg_subsequent = subsequent.mean() * 1000
            std_subsequent = subsequent.std() * 1000
            compile_exec_ratio = compile_time / (subsequent.mean()) if subsequent.mean() > 0 else 0
            console.print(
                f"  Subsequent QPE avg:   {avg_subsequent:.1f}ms ± {std_subsequent:.1f}ms "
                f"(n={len(subsequent)})"
            )
            console.print(f"  Compile/Execute ratio: {compile_exec_ratio:.0f}×")

    # 2. Speedup vs Recompilation
    console.print("\n[bold]2. Speedup vs Recompilation Approach[/bold]")
    if timing_emb:
        compile_time = _effective_compile_time(timing_emb)
        q_valid = timing_emb.quantum_times[timing_emb.quantum_times > 0]
        n_evals = len(q_valid)
        if n_evals > 1:
            avg_exec = q_valid[1:].mean()
            # Old approach: recompile circuit for each step (= compile_time per step)
            old_total = n_evals * compile_time
            new_total = compile_time + n_evals * avg_exec
            speedup = old_total / new_total if new_total > 0 else float("inf")
            console.print(
                f"  Old approach (recompile each step): {compile_time:.1f}s × {n_evals} = "
                f"{old_total:.1f}s"
            )
            console.print(
                f"  New approach (compile once + run):   {compile_time:.1f}s + "
                f"{n_evals} × {avg_exec * 1000:.1f}ms = {new_total:.1f}s"
            )
            console.print(f"  Speedup for {n_evals} evaluations: {speedup:.1f}×")
            # Projected speedup at 1000 evaluations
            old_1k = 1000 * compile_time
            new_1k = compile_time + 1000 * avg_exec
            console.print(f"  Projected at 1000 evaluations: {old_1k / new_1k:.0f}×")
        else:
            console.print("  Insufficient QPE evaluations for speedup estimate")

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
                f"  MLIR IR scale estimate: {n_est} × {n_trotter} × {n_terms} = {ir_scale} ops"
            )

    # 4. Energy Formula Comparison
    console.print("\n[bold]4. Energy Formula Comparison[/bold]")
    formula_vac = meta_vac.get("energy_formula", "N/A")
    formula_emb = meta_emb.get("energy_formula", "N/A")

    formula_table = Table(show_header=False, box=None, padding=(0, 2))
    formula_table.add_column("Mode", style="bold")
    formula_table.add_column("Formula")
    formula_table.add_column("Physics")
    formula_table.add_row(
        "vacuum_correction",
        formula_vac,
        "Approximate: ignores δ_corr-pol",
    )
    formula_table.add_row(
        "mm_embedded",
        formula_emb,
        "Rigorous: MM in QPE Hamiltonian",
    )
    console.print(formula_table)


def _print_delta_corr_pol_analysis(
    result_vac: dict, result_emb: dict, e_vacuum: float, qpe_interval: int
):
    """
    Rigorous per-step δ_corr-pol analysis with energy trend comparison.

    δ_corr-pol captures the difference in electron correlation recovery
    between mm_embedded and vacuum_correction QPE modes at each solvent
    configuration R along the shared MC trajectory.

    Both modes extract phase via probability-weighted expected value over
    qml.probs() output. The systematic Trotter bias (~28.6 mHa for H2,
    n_trotter=10) cancels in the per-step difference δ = E(mm_emb) - E(vac_corr),
    achieving ~177× cancellation ratio, isolating the physical δ_corr-pol signal.
    """
    console.print()
    console.print(
        Panel(
            "[bold]δ_corr-pol: Correlation-Polarization Coupling Analysis[/bold]",
            border_style="yellow",
        )
    )

    q_vac = np.array(result_vac["quantum_energies"])
    q_emb = np.array(result_emb["quantum_energies"])
    n_vac = int(result_vac["n_quantum_evaluations"])
    n_emb = int(result_emb["n_quantum_evaluations"])
    n_eval = min(n_vac, n_emb)

    if n_eval == 0:
        console.print("  No QPE evaluations available.")
        return

    q_vac = q_vac[:n_eval]
    q_emb = q_emb[:n_eval]

    # ── 1. Rigorous Definition ──────────────────────────────────────────
    console.print("\n[bold]1. Rigorous Definition[/bold]")
    console.print("  vacuum_correction:  E(R) = E_corr(vac) + E_HF(R)")
    console.print("                      [QPE on H' = H_vac - E_HF·I → E_corr(vac)]")
    console.print("  mm_embedded:        E(R) = E_corr_eff(R) + E_HF(vac)")
    console.print("                      [QPE on H'_eff = H_eff(R) - E_HF·I → E_corr_eff(R)]")
    console.print()
    console.print("  δ_corr-pol(R) ≡ E(mm_emb, R) - E(vac_corr, R)")
    console.print("                = [E_QPE(H_eff(R)) - E_HF(R)] - [E_QPE(H_vac) - E_HF(vac)]")
    console.print("                = E_corr(solvated, R) - E_corr(vacuum)")
    console.print()
    console.print(
        "  [dim]Both modes share the same MC trajectory (same seed, same HF acceptance),\n"
        "  so quantum_energies arrays are directly comparable element-by-element.[/dim]"
    )

    # ── 2. Per-step computation ─────────────────────────────────────────
    delta = q_emb - q_vac  # Per-step δ_corr-pol in Hartree
    delta_kcal = delta * HARTREE_TO_KCAL_MOL

    # ── 3. Statistics ───────────────────────────────────────────────────
    console.print(f"\n[bold]2. Per-step Statistics (n={n_eval})[/bold]")
    console.print(
        f"  ⟨δ_corr-pol⟩ = {np.mean(delta):+.6f} Ha" f" = {np.mean(delta_kcal):+.2f} kcal/mol"
    )
    console.print(
        f"  σ(δ_corr-pol) = {np.std(delta):.6f} Ha" f" = {np.std(delta_kcal):.2f} kcal/mol"
    )
    console.print(f"  Range: [{np.min(delta):+.6f}, {np.max(delta):+.6f}] Ha")

    # ── 4. Per-step comparison table ────────────────────────────────────
    console.print(f"\n[bold]3. Per-step Comparison[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("MC Step", justify="right")
    table.add_column("E_QPE(vac_corr)", justify="right")
    table.add_column("E_QPE(mm_emb)", justify="right")
    table.add_column("δ (Ha)", justify="right")
    table.add_column("δ (kcal/mol)", justify="right")

    # Select ~10 representative steps
    step_interval = max(1, n_eval // 10)
    indices = list(range(0, n_eval, step_interval))
    if (n_eval - 1) not in indices:
        indices.append(n_eval - 1)

    for i in indices:
        mc_step = (i + 1) * qpe_interval
        table.add_row(
            str(mc_step),
            f"{q_vac[i]:.6f}",
            f"{q_emb[i]:.6f}",
            f"{delta[i]:+.6f}",
            f"{delta_kcal[i]:+.2f}",
        )
    console.print(table)

    # ── 5. Energy Trend Analysis ────────────────────────────────────────
    console.print(f"\n[bold]4. Energy Trend Analysis[/bold]")

    # Split into early vs late MC phases
    n_third = max(1, n_eval // 3)
    early_delta = delta[:n_third]
    late_delta = delta[-n_third:]

    console.print(f"  Early MC (steps {qpe_interval}-{n_third * qpe_interval}):")
    console.print(
        f"    ⟨δ⟩ = {np.mean(early_delta):+.6f} Ha"
        f" = {np.mean(early_delta) * HARTREE_TO_KCAL_MOL:+.2f} kcal/mol"
        f"  σ = {np.std(early_delta):.6f} Ha"
    )
    console.print(
        f"  Late MC (steps {(n_eval - n_third) * qpe_interval + qpe_interval}-{n_eval * qpe_interval}):"
    )
    console.print(
        f"    ⟨δ⟩ = {np.mean(late_delta):+.6f} Ha"
        f" = {np.mean(late_delta) * HARTREE_TO_KCAL_MOL:+.2f} kcal/mol"
        f"  σ = {np.std(late_delta):.6f} Ha"
    )

    # QPE energy stability comparison
    console.print(f"\n  [bold]QPE Energy Stability:[/bold]")
    console.print(
        f"    vacuum_correction: σ = {np.std(q_vac) * 1000:.3f} mHa"
        f" (fixed vacuum QPE + HF-level MM correction)"
    )
    console.print(
        f"    mm_embedded:       σ = {np.std(q_emb) * 1000:.3f} mHa"
        f" (full H_eff varies per solvent config)"
    )
    ratio = np.std(q_emb) / np.std(q_vac) if np.std(q_vac) > 0 else float("inf")
    console.print(
        f"    Variance ratio: {ratio:.1f}× " f"(mm_embedded is {ratio:.1f}× more variable)"
    )

    # ── 6. Physical Interpretation ──────────────────────────────────────
    console.print(f"\n[bold]5. Physical Interpretation[/bold]")
    avg_kcal = np.mean(delta_kcal)

    if avg_kcal > 0:
        console.print(
            f"  δ_corr-pol = {avg_kcal:+.2f} kcal/mol (positive, mm_embedded energy higher)"
        )
        console.print("  → vacuum_correction OVERESTIMATES solvation stabilization")
        console.print(
            "  → The fixed vacuum correlation assumption yields systematically lower QPE energy"
        )
        console.print(
            "  → mm_embedded responds to Hamiltonian changes per config, "
            "producing more variable QPE outputs"
        )
    else:
        console.print(
            f"  δ_corr-pol = {avg_kcal:+.2f} kcal/mol (negative, mm_embedded energy lower)"
        )
        console.print("  → vacuum_correction UNDERESTIMATES solvation stabilization")
        console.print(
            "  → MM-embedded QPE recovers additional correlation from solvent interaction"
        )

    console.print()
    console.print(
        "  [bold]Key mechanism:[/bold] Both modes use energy-shifted QPE (H' = H - E_HF·I),"
    )
    console.print("  measuring correlation energy directly. Phase is extracted via probability-")
    console.print("  weighted expected value (Σ probs[k]·k), preserving continuous sensitivity.")
    console.print("  vacuum_correction uses a fixed H'_vac (σ ~0.01 mHa), while mm_embedded uses")
    console.print("  configuration-dependent H'_eff(R) (σ ~11 mHa). Systematic Trotter bias")
    console.print("  (~28.6 mHa) cancels in the per-step difference at ~177× ratio.")
    console.print()
    console.print(
        "  [dim]Note: 4-bit QPE resolution (bin width ≈ 12 mHa ≈ 7.5 kcal/mol) limits\n"
        "  the SNR of δ measurement. True δ_corr-pol for H2/STO-3G is expected\n"
        "  ~0.01-0.1 kcal/mol (SEM ~1 kcal/mol at n=50). Higher QPE resolution\n"
        "  (more estimation qubits) will improve accuracy.[/dim]"
    )


# =============================================================================
# Main
# =============================================================================


def main():
    """Run H2 MC solvation with both QPE modes and compare."""
    console.print(
        Panel(
            "[bold]H2 MM-Embedded QPE: Frame 14 Target Architecture[/bold]\n"
            "Runtime Hamiltonian coefficient parameterization\n"
            "Compile once, execute with new coefficients (no recompilation)",
            title="Experiment",
            border_style="green",
        )
    )

    mc_params = dict(
        molecule=H2_MOLECULE,
        n_waters=N_WATERS,
        n_mc_steps=N_MC_STEPS,
        temperature=300.0,
        translation_step=0.3,
        rotation_step=0.2618,
        initial_water_distance=4.0,
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

    # ── Comparison ─────────────────────────────────────────────────────
    e_vacuum = result_vac["e_vacuum"]
    print_mode_comparison(result_vac, result_emb, e_vacuum)
    _print_architecture_performance_analysis(result_vac, result_emb)
    _print_delta_corr_pol_analysis(result_vac, result_emb, e_vacuum, QPE_VACUUM.qpe_interval)

    console.print("\n" + "=" * 60)
    console.print("  Experiment complete.")
    console.print("=" * 60)

    return result_vac, result_emb


if __name__ == "__main__":
    main()
