#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
H2 Cached QPE-Driven MC Solvation

End-to-end H2 QPE-driven Monte Carlo solvation using Catalyst IR cache
to mitigate the MLIR→LLVM amplification cascade.

Two-path compilation:
  Phase A (cache miss): spawn subprocess → full @qjit(keep_intermediate=True) →
                        save LLVM IR to disk
                        NOTE: Uses H_dynamic (runtime coefficients) because qpe_driven MC
                        requires updating Hamiltonian coefficients each step. H_fixed IR
                        (zero-arg) is incompatible with H_dynamic shell via replace_ir
                        due to LLVM symbol name mismatch (verified empirically).
  Phase B (cache hit):  replace_ir + jit_compile → skip MLIR pipeline (~5× faster,
                        ~79% memory reduction)

MC solvation reuses mc_solvation/ components (Mode 3 / qpe_driven architecture).

Usage:
    uv run python examples/h2_cached_qpe_driven_mc.py
    uv run python examples/h2_cached_qpe_driven_mc.py --n-est 2 --n-trotter 1 --n-mc-steps 10
    uv run python examples/h2_cached_qpe_driven_mc.py --force-recompile
"""

import argparse
import multiprocessing
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from catalyst import qjit  # noqa: E402
from catalyst.debug import get_compilation_stage, replace_ir  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from examples.mc_solvation import (  # noqa: E402
    TIP3P_WATER,
    MoleculeConfig,
    QPEConfig,
    SolvationConfig,
    create_fused_qpe_callback,
    create_qpe_driven_mc_loop,
    create_qpe_step_callback,
    initialize_solvent_ring,
    molecules_to_state_array,
    precompute_vacuum_cache,
)
from examples.mc_solvation.energy import compute_hf_energy_vacuum  # noqa: E402
from examples.qpe_compile_cache_verify import build_hamiltonian_data, make_qpe_circuit  # noqa: E402
from examples.qpe_memory_profile import ir_output_dir, take_snapshot  # noqa: E402

console = Console()

# =============================================================================
# Constants
# =============================================================================

N_WATERS = 5
N_MC_STEPS = 20
TEMPERATURE = 300.0
TRANSLATION_STEP = 0.3
ROTATION_STEP = 0.2618
INITIAL_WATER_DISTANCE = 4.0
RANDOM_SEED = 42
DEFAULT_CACHE_DIR = "/tmp/qpe_ir_cache"

H2_MOLECULE = MoleculeConfig(
    name="H2",
    symbols=["H", "H"],
    coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    charge=0,
    active_electrons=2,
    active_orbitals=2,
    basis="sto-3g",
)


def _cache_ir_path(cache_dir: str, n_est: int, n_trotter: int) -> Path:
    """Cache key encodes circuit params to prevent IR/param mismatch."""
    return Path(cache_dir) / f"h2_n{n_est}e_{n_trotter}t.ll"


# =============================================================================
# Phase A: spawn subprocess for full compilation + IR persistence
# =============================================================================


def _phase_a_worker(
    n_est: int,
    n_trotter: int,
    ir_dir: str,
    cache_path_str: str,
    queue: multiprocessing.Queue,
) -> None:
    """
    Subprocess worker: full @qjit(keep_intermediate=True) compilation + save LLVM IR.

    Uses H_dynamic mode (coeffs_arr as runtime parameter) because:
      1. qpe_driven MC requires runtime coefficients (energy.py:698 calls
         compiled_circuit(jnp.array(coeffs)) each step)
      2. H_fixed LLVM IR (zero-arg) is incompatible with H_dynamic shell —
         replace_ir cannot bridge the LLVM symbol name mismatch
         (qpe_fixed vs qpe_profiled entry points)
      3. Subprocess isolation (spawn context) confines ~19GB peak memory to
         child process, preventing parent RSS pollution

    Runs in spawn context so ru_maxrss baseline is clean (no inheritance from parent).
    """
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    try:
        tracemalloc.start()
        data = build_hamiltonian_data("h2", n_est, n_trotter)
        qpe_fn = make_qpe_circuit(
            data["ops"],
            data["hf_state"],
            data["n_qubits"],
            n_est,
            n_trotter,
            data["base_time"],
        )
        t0 = time.monotonic()
        with ir_output_dir(ir_dir):
            compiled = qjit(keep_intermediate=True)(qpe_fn)
            compiled(data["coeffs"])  # trigger compilation
            llvm_ir = get_compilation_stage(compiled, "LLVMIRTranslation")
        compile_time = time.monotonic() - t0

        Path(cache_path_str).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path_str).write_text(llvm_ir)
        snap = take_snapshot("phase_a")
        queue.put(
            {
                "compile_time_s": compile_time,
                "maxrss_self_mb": snap.maxrss_mb,
                "maxrss_children_mb": snap.maxrss_children_mb,
                "error": None,
            }
        )
    except Exception as exc:
        import traceback as tb

        queue.put(
            {
                "compile_time_s": 0.0,
                "maxrss_self_mb": 0.0,
                "maxrss_children_mb": 0.0,
                "error": f"{type(exc).__name__}: {exc}\n{tb.format_exc()}",
            }
        )


# =============================================================================
# Phase A/B orchestration
# =============================================================================


def setup_compiled_qpe(
    data: dict,
    n_est: int,
    n_trotter: int,
    cache_dir: str,
    force_recompile: bool = False,
) -> tuple:
    """
    Return (compiled_qpe_fn, is_cache_hit, stats_dict).

    Cache miss / force_recompile:
        spawn subprocess → Phase A (populates disk cache) → Phase B in main process
    Cache hit:
        Phase B only (replace_ir + jit_compile, skips MLIR pipeline)
    """
    cache_path = _cache_ir_path(cache_dir, n_est, n_trotter)
    is_cache_hit = cache_path.exists() and not force_recompile
    stats: dict = {"is_cache_hit": is_cache_hit}

    if not is_cache_hit:
        console.print(
            "[bold cyan]Phase A:[/bold cyan] cache miss — running full @qjit compilation "
            "in subprocess..."
        )
        ctx = multiprocessing.get_context("spawn")
        q: multiprocessing.Queue = ctx.Queue()
        proc = ctx.Process(
            target=_phase_a_worker,
            args=(n_est, n_trotter, str(cache_path.parent), str(cache_path), q),
        )
        proc.start()
        proc.join()
        phase_a = q.get()
        if phase_a["error"]:
            raise RuntimeError(f"Phase A failed:\n{phase_a['error']}")
        stats["phase_a"] = phase_a
        console.print(
            f"  Phase A done: {phase_a['compile_time_s']:.1f}s | "
            f"SELF {phase_a['maxrss_self_mb']:.0f} MB | "
            f"CHILDREN {phase_a['maxrss_children_mb']:.0f} MB"
        )
    else:
        console.print("[bold green]Phase B:[/bold green] cache hit — loading LLVM IR from disk...")

    # Phase B: always runs in main process (fast — LLVM→machine code only)
    qpe_fn = make_qpe_circuit(
        data["ops"],
        data["hf_state"],
        data["n_qubits"],
        n_est,
        n_trotter,
        data["base_time"],
    )
    compiled = qjit(qpe_fn)
    llvm_ir = cache_path.read_text()
    replace_ir(compiled, "LLVMIRTranslation", llvm_ir)
    t0 = time.monotonic()
    compiled.jit_compile((data["coeffs"],))  # tuple syntax required
    phase_b_time = time.monotonic() - t0
    stats["phase_b_compile_s"] = phase_b_time
    console.print(f"  Phase B done: {phase_b_time:.2f}s (LLVM→machine code only)")

    return compiled, is_cache_hit, stats


# =============================================================================
# MC solvation integration (Mode 3: qpe_driven)
# =============================================================================


def run_cached_mc(
    compiled_qpe,
    data: dict,
    n_est: int,
    n_mc_steps: int,
    n_waters: int,
) -> dict:
    """
    Mode 3 (qpe_driven) MC solvation using pre-compiled (cached) QPE function.
    Reuses mc_solvation/ components — no reimplementation.
    """
    qpe_cfg = QPEConfig(
        n_estimation_wires=n_est,
        n_trotter_steps=data["n_trotter"],
        n_shots=0,
        qpe_interval=1,
        target_resolution=0.003,
        energy_range=0.2,
        use_catalyst=True,
    )
    config = SolvationConfig(
        molecule=H2_MOLECULE,
        qpe_config=qpe_cfg,
        qpe_mode="qpe_driven",
        n_waters=n_waters,
        n_mc_steps=n_mc_steps,
        temperature=TEMPERATURE,
        translation_step=TRANSLATION_STEP,
        rotation_step=ROTATION_STEP,
        initial_water_distance=INITIAL_WATER_DISTANCE,
        random_seed=RANDOM_SEED,
        verbose=False,
    )

    # Setup solvent system (initialize_solvent_ring requires explicit args)
    qm_center = H2_MOLECULE.center
    solvent_molecules = initialize_solvent_ring(
        model=TIP3P_WATER,
        n_molecules=n_waters,
        center=qm_center,
        radius=INITIAL_WATER_DISTANCE,
        random_seed=RANDOM_SEED,
    )
    solvent_states = molecules_to_state_array(solvent_molecules)
    qm_coords_flat = H2_MOLECULE.coords_array.flatten()

    # Vacuum reference energy
    e_vacuum = compute_hf_energy_vacuum(H2_MOLECULE)

    # Precompute vacuum cache (avoids redundant SCF calls in MC loop)
    vacuum_cache = precompute_vacuum_cache(config)

    # Build callbacks using mc_solvation components
    fused_cb = create_fused_qpe_callback(
        config,
        data["coeffs"],
        data["op_index_map"],
        H2_MOLECULE.active_orbitals,
        vacuum_cache,
    )
    step_cb = create_qpe_step_callback(
        fused_cb,
        compiled_qpe,
        data["base_time"],
        n_est,
        data["energy_shift"],
        data["n_terms"],
    )

    # Warmup: first call verifies compiled circuit works
    console.print("[dim]Warmup QPE call (verifies compiled circuit works)...[/dim]")
    init_result = step_cb(solvent_states, qm_coords_flat)
    init_energy = float(init_result[0]) + float(init_result[1])

    # MC loop
    mc_loop = create_qpe_driven_mc_loop(config, step_cb)
    t0 = time.monotonic()
    result = mc_loop(solvent_states, qm_coords_flat, RANDOM_SEED, init_energy)
    result["mc_wall_time_s"] = time.monotonic() - t0
    result["e_vacuum"] = e_vacuum

    return result


# =============================================================================
# Output
# =============================================================================


def print_results(
    n_est: int,
    n_trotter: int,
    is_cache_hit: bool,
    stats: dict,
    mc_result: dict,
) -> None:
    """Rich output: compilation stats + MC simulation summary."""
    console.print()
    if is_cache_hit:
        mode_label = "[green]Phase B (cache hit)[/green]"
    else:
        mode_label = "[yellow]Phase A + Phase B (cache miss)[/yellow]"

    lines = [
        f"[bold]Molecule:[/bold] H2 (STO-3G, 2e/2o)  n_est={n_est}  n_trotter={n_trotter}",
        f"[bold]Compilation:[/bold] {mode_label}  Phase B: {stats['phase_b_compile_s']:.2f}s",
    ]
    if "phase_a" in stats:
        pa = stats["phase_a"]
        lines.append(
            f"  Phase A: {pa['compile_time_s']:.1f}s | "
            f"SELF {pa['maxrss_self_mb']:.0f} MB | "
            f"CHILDREN {pa['maxrss_children_mb']:.0f} MB"
        )
    console.print(
        Panel(
            "\n".join(lines),
            title="[cyan]Cached QPE-Driven MC Solvation[/cyan]",
            border_style="cyan",
        )
    )

    t = Table(title="MC Simulation Results")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("Best QPE energy (Ha)", f"{mc_result['best_qpe_energy']:.6f}")
    t.add_row("E_vacuum (Ha)", f"{mc_result['e_vacuum']:.6f}")
    t.add_row("Acceptance rate", f"{mc_result['acceptance_rate'] * 100:.1f}%")
    t.add_row("n QPE evaluations", str(mc_result.get("n_quantum_evaluations", "?")))
    t.add_row("MC wall time (s)", f"{mc_result['mc_wall_time_s']:.1f}")
    console.print(t)


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H2 QPE-driven MC solvation with Catalyst IR cache",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-est", type=int, default=2)
    parser.add_argument("--n-trotter", type=int, default=1)
    parser.add_argument("--n-mc-steps", type=int, default=N_MC_STEPS)
    parser.add_argument("--n-waters", type=int, default=N_WATERS)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--force-recompile", action="store_true")
    args = parser.parse_args()

    # Build Hamiltonian data (shared by Phase A/B and MC callbacks)
    console.print("[dim]Building H2 Hamiltonian...[/dim]")
    data = build_hamiltonian_data("h2", args.n_est, args.n_trotter)

    # Phase A or Phase B: get compiled QPE function
    compiled_qpe, is_cache_hit, stats = setup_compiled_qpe(
        data,
        args.n_est,
        args.n_trotter,
        args.cache_dir,
        force_recompile=args.force_recompile,
    )

    # MC solvation using pre-compiled QPE
    mc_result = run_cached_mc(
        compiled_qpe,
        data,
        args.n_est,
        n_mc_steps=args.n_mc_steps,
        n_waters=args.n_waters,
    )

    print_results(args.n_est, args.n_trotter, is_cache_hit, stats, mc_result)


if __name__ == "__main__":
    main()
