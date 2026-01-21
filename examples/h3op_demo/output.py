# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Output module for H3O+ QPE Demo.

Contains all print/display functions for demo output.
"""

from datetime import datetime

from examples.h3op_demo.config import (
    CATALYST_VERSION,
    ENERGY_CONSISTENCY_THRESHOLD,
    HAS_CATALYST,
    HAS_JAX_CUDA,
    HAS_LIGHTNING_GPU,
    MM_STABILIZATION_THRESHOLD,
    get_best_available_device,
    get_catalyst_effective_backend,
)


def print_header():
    """Print demo header."""
    print("=" * 80)
    print("                    H3O+ Quantum Phase Estimation (QPE) Demo")
    print("                    q2m3 MVP - Catalyst Technical Validation")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Catalyst Available: {'Yes (v' + CATALYST_VERSION + ')' if HAS_CATALYST else 'No'}")
    print(f"Lightning GPU Available: {'Yes' if HAS_LIGHTNING_GPU else 'No'}")

    # Show JAX/Catalyst backend info (IMPORTANT for understanding performance)
    if HAS_CATALYST:
        jax_backend_display = get_catalyst_effective_backend()
        print(f"Catalyst Execution Backend: {jax_backend_display}")
        if not HAS_JAX_CUDA and HAS_LIGHTNING_GPU:
            print("  WARNING: Catalyst @qjit runs on CPU (JAX lacks CUDA support)")
            print("  To enable Catalyst GPU: pip install 'jax[cuda12]'")
    print()


def print_section(title: str, step: int | float | None = None) -> None:
    """Print section header."""
    if step is not None:
        print(f"\n[Step {step}] {title}")
    else:
        print(f"\n{title}")
    print("-" * 80)


def print_system_info(qpe_config: dict, mm_waters: int):
    """Print system configuration details."""
    active_e = qpe_config["active_electrons"]
    active_o = qpe_config["active_orbitals"]
    system_qubits = active_o * 2  # 2 spin orbitals per spatial orbital
    estimation_qubits = qpe_config["n_estimation_wires"]
    total_qubits = system_qubits + estimation_qubits

    print("QM Region: H3O+ (4 atoms, total charge +1)")
    print(f"MM Region: {mm_waters} TIP3P water molecules")
    print()
    print("Quantum Resource Requirements:")
    print(f"  Active Space: {active_e} electrons, {active_o} orbitals")
    print(f"  System Qubits: {system_qubits} (spin orbitals)")
    print(f"  Estimation Qubits: {estimation_qubits} (precision bits)")
    print(f"  Total Qubits: {total_qubits}")
    print()
    print("QPE Circuit Parameters:")
    print(f"  Base Evolution Time: {qpe_config['base_time']}")
    print(f"  Trotter Steps: {qpe_config['n_trotter_steps']}")
    print(f"  Measurement Shots: {qpe_config['n_shots']}")
    print(f"  Qubit Mapping: {qpe_config['mapping']}")


def print_hf_solvation_effect(solvation_data: dict) -> None:
    """Print HF solvation effect analysis results."""
    print(
        f"MM Environment: {solvation_data['n_mm_waters']} TIP3P waters "
        f"({solvation_data['n_mm_atoms']} point charges)"
    )
    print()
    print("Hartree-Fock Energy Comparison:")
    print(f"  Vacuum (no MM):     {solvation_data['energy_vacuum']:.6f} Hartree")
    print(f"  Solvated (with MM): {solvation_data['energy_solvated']:.6f} Hartree")
    print()
    print("Solvation Stabilization:")
    print(f"  dE = {solvation_data['stabilization_hartree']:.6f} Hartree")
    print(f"     = {solvation_data['stabilization_kcal_mol']:.2f} kcal/mol")
    print()
    if solvation_data["stabilization_hartree"] > MM_STABILIZATION_THRESHOLD:
        print("  [OK] MM embedding is working: explicit solvent stabilizes H3O+")
    else:
        print("  [WARNING] Unexpected: no significant stabilization detected")


def print_resource_estimation(eftqc_data: dict, mm_waters: int) -> None:
    """Print EFTQC resource estimation results."""
    n_mm_charges = eftqc_data["n_mm_charges"]
    eftqc_vac_chem = eftqc_data["vacuum_chemical"]
    eftqc_vac_relax = eftqc_data["vacuum_relaxed"]
    eftqc_sol_chem = eftqc_data["solvated_chemical"]
    eftqc_sol_relax = eftqc_data["solvated_relaxed"]

    print(f"Resource Comparison (H3O+ with {mm_waters} TIP3P waters, {n_mm_charges} MM charges)")
    print()
    print("  " + "-" * 70)
    print(f"  {'Configuration':<25} {'1-norm (Ha)':<15} {'Toffoli Gates':<18} {'Qubits':<10}")
    print("  " + "-" * 70)
    print(
        f"  {'Vacuum (chemical)':<25} {eftqc_vac_chem['hamiltonian_1norm']:<15.2f} "
        f"{eftqc_vac_chem['toffoli_gates']:<18,} {eftqc_vac_chem['logical_qubits']:<10}"
    )
    print(
        f"  {'Vacuum (relaxed)':<25} {eftqc_vac_relax['hamiltonian_1norm']:<15.2f} "
        f"{eftqc_vac_relax['toffoli_gates']:<18,} {eftqc_vac_relax['logical_qubits']:<10}"
    )
    print(
        f"  {'Solvated (chemical)':<25} {eftqc_sol_chem['hamiltonian_1norm']:<15.2f} "
        f"{eftqc_sol_chem['toffoli_gates']:<18,} {eftqc_sol_chem['logical_qubits']:<10}"
    )
    print(
        f"  {'Solvated (relaxed)':<25} {eftqc_sol_relax['hamiltonian_1norm']:<15.2f} "
        f"{eftqc_sol_relax['toffoli_gates']:<18,} {eftqc_sol_relax['logical_qubits']:<10}"
    )
    print("  " + "-" * 70)
    print()

    print("  Analysis:")
    print(f"    MM embedding effect: dL = {eftqc_data['delta_lambda']:+.1f}% (1-norm increase)")
    print(
        f"    Error relaxation:    {eftqc_data['gate_reduction']:.1f}% fewer gates "
        "(10x error tolerance)"
    )
    print()
    print("  Note: MM embedding only modifies 1-electron integrals (QM-MM electrostatics).")
    print("        Resource estimates are dominated by 2-electron integrals, which remain")
    print("        unchanged. Thus, vacuum vs solvated show similar resource requirements.")


def print_qpe_solvation_effect(
    qpe_solvation_data: dict,
    hf_solvation_data: dict,
    label: str = "QPE",
) -> None:
    """
    Print QPE solvation effect analysis results.

    Args:
        qpe_solvation_data: QPE solvation effect data
        hf_solvation_data: HF solvation effect data for comparison
        label: Label for the QPE method (e.g., "Standard QPE", "Catalyst QPE")
    """
    # Execution time
    print("Execution Time:")
    print(f"  Vacuum QPE:   {qpe_solvation_data['time_vacuum_s']:.3f} s")
    print(f"  Solvated QPE: {qpe_solvation_data['time_solvated_s']:.3f} s")
    print(f"  Total:        {qpe_solvation_data['time_total_s']:.3f} s")
    print()

    # Energy results
    print(f"{label} Energy Comparison:")
    print(f"  Vacuum (no MM):     {qpe_solvation_data['energy_vacuum']:.6f} Hartree")
    print(f"  Solvated (with MM): {qpe_solvation_data['energy_solvated']:.6f} Hartree")
    print()
    print(f"{label} Solvation Stabilization:")
    print(f"  dE = {qpe_solvation_data['stabilization_hartree']:.6f} Hartree")
    print(f"     = {qpe_solvation_data['stabilization_kcal_mol']:.2f} kcal/mol")
    print()

    # Convergence info for solvated calculation
    conv = qpe_solvation_data.get("convergence_solvated", {})
    rdm_source = qpe_solvation_data.get("rdm_source_solvated", "unknown")
    if conv:
        print("Convergence Status (Solvated):")
        print(f"  Converged: {'Yes' if conv.get('converged') else 'No'}")
        print(f"  Method: {conv.get('method', 'N/A')}")
        print(f"  RDM Source: {rdm_source}")
        print()

    # Compare with HF solvation effect
    print("Comparison with Classical HF:")
    print(f"  HF Stabilization:  {hf_solvation_data['stabilization_kcal_mol']:.2f} kcal/mol")
    print(f"  {label} Stabilization: {qpe_solvation_data['stabilization_kcal_mol']:.2f} kcal/mol")
    diff_kcal = abs(
        hf_solvation_data["stabilization_kcal_mol"] - qpe_solvation_data["stabilization_kcal_mol"]
    )
    print(f"  Difference: {diff_kcal:.2f} kcal/mol")
    print()

    if qpe_solvation_data["stabilization_hartree"] > MM_STABILIZATION_THRESHOLD:
        print(f"  [OK] {label} correctly captures MM solvation effect")
    else:
        print(f"  [WARNING] {label} solvation effect not detected")

    # Mulliken charge comparison
    print()
    print("Mulliken Charge Redistribution (Vacuum -> Solvated):")
    for atom in qpe_solvation_data["charges_vacuum"].keys():
        q_vac = qpe_solvation_data["charges_vacuum"][atom]
        q_sol = qpe_solvation_data["charges_solvated"][atom]
        delta_q = q_sol - q_vac
        print(f"  {atom}: {q_vac:+.4f} -> {q_sol:+.4f} (dq = {delta_q:+.4f})")


def print_comparison(
    standard_solvation_data: dict,
    catalyst_solvation_data: dict | None,
):
    """Print comparison between standard and Catalyst execution."""
    std_device = get_best_available_device()
    # IMPORTANT: Catalyst backend is determined by JAX, not PennyLane device
    catalyst_backend = get_catalyst_effective_backend()

    # Use solvated QPE time for comparison (more representative with MM embedding)
    time_standard = standard_solvation_data["time_solvated_s"]
    e_std = standard_solvation_data["energy_solvated"]

    print("Execution Time Comparison (Solvated QPE):")
    print(f"  Standard QPE ({std_device}): {time_standard:.3f} s")

    if catalyst_solvation_data is not None:
        time_catalyst = catalyst_solvation_data["time_solvated_s"]
        e_cat = catalyst_solvation_data["energy_solvated"]

        # Show Catalyst with ACTUAL execution backend, not device name
        print(f"  Catalyst QPE ({catalyst_backend}): {time_catalyst:.3f} s")

        if time_catalyst > 0:
            speedup = time_standard / time_catalyst
            if speedup > 1:
                print(f"  Speedup: {speedup:.2f}x faster with Catalyst")
            else:
                print(f"  Ratio: {1 / speedup:.2f}x slower with Catalyst")
                # Show accurate performance analysis based on actual backend
                if not HAS_JAX_CUDA:
                    print("  Note: Catalyst is running on CPU (JAX lacks CUDA support)")
                    print("        To enable Catalyst GPU: pip install 'jax[cuda12]'")
                    if HAS_LIGHTNING_GPU:
                        print("        (Standard QPE already uses GPU via lightning.gpu)")
        print()

        print("Energy Comparison (Solvated):")
        e_diff = abs(e_std - e_cat)
        print(f"  Standard QPE: {e_std:.6f} Hartree")
        print(f"  Catalyst QPE: {e_cat:.6f} Hartree")
        print(f"  Difference: {e_diff:.6f} Hartree")

        if e_diff < ENERGY_CONSISTENCY_THRESHOLD:
            print("  Status: Results consistent (diff < 0.01 Ha)")
        else:
            print("  Status: Results differ (stochastic QPE sampling)")
    else:
        print("  Catalyst QPE: Not available")
        print()
        print("Energy (Solvated):")
        print(f"  Standard QPE: {e_std:.6f} Hartree")


def print_summary(
    solvation_data: dict,
    qpe_solvation_data: dict,
    catalyst_solvation_data: dict | None,
    eftqc_data: dict,
) -> None:
    """Print demo summary."""
    eftqc_vac_chem = eftqc_data["vacuum_chemical"]
    eftqc_sol_chem = eftqc_data["solvated_chemical"]

    print("q2m3 MVP Capabilities Demonstrated:")
    print("  [OK] PySCF -> PennyLane Hamiltonian conversion")
    print("  [OK] Standard QPE circuit implementation")
    print("  [OK] HF state preparation (qml.BasisState)")
    print("  [OK] Trotter time evolution (qml.TrotterProduct)")
    print("  [OK] Inverse QFT (qml.adjoint(qml.QFT))")
    print("  [OK] Phase-to-energy extraction")
    print("  [OK] QM/MM system with TIP3P solvation")
    print(
        f"  [OK] EFTQC resource estimation (vacuum: {eftqc_vac_chem['toffoli_gates']:,}, "
        f"solvated: {eftqc_sol_chem['toffoli_gates']:,} Toffoli)"
    )

    if solvation_data["stabilization_hartree"] > MM_STABILIZATION_THRESHOLD:
        print(
            f"  [OK] HF MM embedding (dE = {solvation_data['stabilization_kcal_mol']:.1f} kcal/mol)"
        )
    else:
        print("  [--] HF MM embedding (no stabilization detected)")

    if qpe_solvation_data["stabilization_hartree"] > MM_STABILIZATION_THRESHOLD:
        print(
            f"  [OK] Standard QPE MM embedding "
            f"(dE = {qpe_solvation_data['stabilization_kcal_mol']:.1f} kcal/mol)"
        )
    else:
        print("  [--] Standard QPE MM embedding (no stabilization detected)")

    if catalyst_solvation_data is not None:
        if catalyst_solvation_data["stabilization_hartree"] > MM_STABILIZATION_THRESHOLD:
            print(
                f"  [OK] Catalyst QPE MM embedding "
                f"(dE = {catalyst_solvation_data['stabilization_kcal_mol']:.1f} kcal/mol)"
            )
        else:
            print("  [--] Catalyst QPE MM embedding (no stabilization detected)")

    print("  [OK] Quantum RDM measurement (Pauli expectation values)")
    print("  [OK] Mulliken population analysis (from quantum RDM)")
    print("  [OK] Circuit visualization (qml.draw)")

    # Accurately report Catalyst status with actual execution backend
    if HAS_CATALYST:
        catalyst_backend = get_catalyst_effective_backend()
        if HAS_JAX_CUDA:
            print(f"  [OK] Catalyst @qjit JIT compilation ({catalyst_backend})")
        else:
            print(f"  [OK] Catalyst @qjit JIT compilation ({catalyst_backend})")
            if HAS_LIGHTNING_GPU:
                print("       -> Note: JAX lacks CUDA support, Catalyst runs on CPU")
                print("       -> To enable Catalyst GPU: pip install 'jax[cuda12]'")
    else:
        print("  [--] Catalyst @qjit (not installed)")

    # GPU acceleration status for standard QPE (separate from Catalyst)
    if HAS_LIGHTNING_GPU:
        print("  [OK] GPU acceleration for standard QPE (lightning.gpu)")
    else:
        print("  [--] GPU acceleration (lightning.gpu not available)")

    print()
    print("=" * 80)
    print("                           Demo Completed Successfully")
    print("=" * 80)


def print_profiling_report(
    profiling_data: dict,
    standard_qpe_data: dict | None = None,
    catalyst_qpe_data: dict | None = None,
) -> None:
    """Print profiling report with detailed performance analysis.

    Args:
        profiling_data: Dictionary containing timing data for each step.
            Expected keys: 'resource_estimation', 'hf_solvation', 'standard_qpe',
            'catalyst_qpe', 'total'. Each value should be a timing dict
            with 'elapsed' key.
        standard_qpe_data: Optional QPE solvation data from analyze_qpe_solvation_effect
            with use_catalyst=False. Contains 'time_vacuum_s', 'time_solvated_s'.
        catalyst_qpe_data: Optional QPE solvation data from analyze_qpe_solvation_effect
            with use_catalyst=True. Contains 'time_vacuum_s', 'time_solvated_s'.
    """
    print()
    print("=" * 80)
    print("                         Performance Profiling Report")
    print("=" * 80)
    print()

    # Extract timing values
    t_resource = profiling_data.get("resource_estimation", {}).get("elapsed", 0.0)
    t_hf = profiling_data.get("hf_solvation", {}).get("elapsed", 0.0)
    t_standard = profiling_data.get("standard_qpe", {}).get("elapsed", 0.0)
    t_catalyst = profiling_data.get("catalyst_qpe", {}).get("elapsed", 0.0)
    t_total = profiling_data.get("total", {}).get("elapsed", 0.0)

    # Detailed timing breakdown
    print("Detailed Timing Breakdown:")
    print(f"  {'Step':<35} {'Time (s)':<12} {'Percentage':<10}")
    print("  " + "-" * 60)

    steps = [
        ("Resource Estimation (EFTQC)", t_resource),
        ("HF Solvation Analysis", t_hf),
        ("Standard QPE (vacuum + solvated)", t_standard),
        ("Catalyst QPE (vacuum + solvated)", t_catalyst),
    ]

    for name, t in steps:
        if t > 0:
            pct = (t / t_total * 100) if t_total > 0 else 0
            print(f"  {name:<35} {t:<12.3f} {pct:.1f}%")

    print("  " + "-" * 60)
    print(f"  {'Total Demo Time':<35} {t_total:<12.3f}")
    print()

    # Fine-grained QPE breakdown (key for jit + lightning.gpu bottleneck analysis)
    if standard_qpe_data or catalyst_qpe_data:
        print("=" * 80)
        print("          Fine-Grained QPE Timing Breakdown (Bottleneck Analysis)")
        print("=" * 80)
        print()

        # Table header
        print(f"  {'Phase':<25} {'Standard QPE':<15} {'Catalyst QPE':<15} {'Ratio':<10}")
        print("  " + "-" * 70)

        # Extract fine-grained timings
        std_vacuum = standard_qpe_data.get("time_vacuum_s", 0.0) if standard_qpe_data else 0.0
        std_solvated = standard_qpe_data.get("time_solvated_s", 0.0) if standard_qpe_data else 0.0
        std_total = standard_qpe_data.get("time_total_s", 0.0) if standard_qpe_data else 0.0

        cat_vacuum = catalyst_qpe_data.get("time_vacuum_s", 0.0) if catalyst_qpe_data else 0.0
        cat_solvated = catalyst_qpe_data.get("time_solvated_s", 0.0) if catalyst_qpe_data else 0.0
        cat_total = catalyst_qpe_data.get("time_total_s", 0.0) if catalyst_qpe_data else 0.0

        # Print comparison table
        def ratio_str(a: float, b: float) -> str:
            if a > 0 and b > 0:
                return f"{b / a:.2f}x"
            return "N/A"

        rows = [
            ("Vacuum QPE", std_vacuum, cat_vacuum),
            ("Solvated QPE", std_solvated, cat_solvated),
            ("Total", std_total, cat_total),
        ]

        for name, std_t, cat_t in rows:
            std_str = f"{std_t:.3f}s" if std_t > 0 else "N/A"
            cat_str = f"{cat_t:.3f}s" if cat_t > 0 else "N/A"
            r_str = ratio_str(std_t, cat_t)
            print(f"  {name:<25} {std_str:<15} {cat_str:<15} {r_str:<10}")

        print("  " + "-" * 70)
        print()

        # Internal stage breakdown (if timing data available)
        cat_timing_vacuum = catalyst_qpe_data.get("timing_vacuum", {}) if catalyst_qpe_data else {}
        cat_timing_solvated = (
            catalyst_qpe_data.get("timing_solvated", {}) if catalyst_qpe_data else {}
        )
        std_timing_vacuum = standard_qpe_data.get("timing_vacuum", {}) if standard_qpe_data else {}
        std_timing_solvated = (
            standard_qpe_data.get("timing_solvated", {}) if standard_qpe_data else {}
        )

        if cat_timing_vacuum or cat_timing_solvated:
            print("  Internal Stage Breakdown (Catalyst QPE):")
            print(f"  {'Stage':<25} {'Vacuum':<12} {'Solvated':<12} {'Delta':<12}")
            print("  " + "-" * 55)

            stages = [
                ("Hamiltonian Build", "hamiltonian_build_s"),
                ("QPE Circuit Build", "qpe_circuit_build_s"),
                ("QPE Circuit Exec", "qpe_circuit_exec_s"),
                ("RDM Measurement", "rdm_measurement_s"),
                ("RDM Postprocess", "rdm_postprocess_s"),
                ("Mulliken Analysis", "mulliken_analysis_s"),
            ]

            for label, key in stages:
                v = cat_timing_vacuum.get(key, 0.0)
                s = cat_timing_solvated.get(key, 0.0)
                if v > 0 or s > 0:
                    delta = s - v
                    delta_str = f"+{delta:.3f}s" if delta >= 0 else f"{delta:.3f}s"
                    print(f"  {label:<25} {v:.3f}s       {s:.3f}s       {delta_str}")

            print("  " + "-" * 55)
            print()

            # Compare with Standard QPE internal stages if available
            if std_timing_vacuum or std_timing_solvated:
                print("  Stage-wise Catalyst/Standard Ratio (Vacuum):")
                print(f"  {'Stage':<25} {'Standard':<12} {'Catalyst':<12} {'Ratio':<10}")
                print("  " + "-" * 55)

                for label, key in stages:
                    std_v = std_timing_vacuum.get(key, 0.0)
                    cat_v = cat_timing_vacuum.get(key, 0.0)
                    if std_v > 0 and cat_v > 0:
                        ratio = cat_v / std_v
                        print(f"  {label:<25} {std_v:.3f}s       {cat_v:.3f}s       {ratio:.2f}x")

                print("  " + "-" * 55)
                print()

        # Bottleneck analysis
        if cat_vacuum > 0 and cat_solvated > 0:
            print("  [BOTTLENECK ANALYSIS]")
            print()

            # First call vs second call analysis
            # First call (vacuum) includes JIT compilation overhead
            # Second call (solvated) should benefit from cached compilation
            if cat_vacuum > cat_solvated * 1.2:
                overhead = cat_vacuum - cat_solvated
                overhead_pct = (overhead / cat_vacuum) * 100
                print(f"    JIT Compilation Overhead (1st call - 2nd call):")
                print(f"      Estimated: {overhead:.3f}s ({overhead_pct:.1f}% of vacuum QPE)")
                print(
                    f"      Evidence: Vacuum ({cat_vacuum:.3f}s) >> Solvated ({cat_solvated:.3f}s)"
                )
                print()
            elif cat_solvated > cat_vacuum * 1.2:
                print(
                    f"    Observation: Solvated ({cat_solvated:.3f}s) > Vacuum ({cat_vacuum:.3f}s)"
                )
                print(
                    "      This suggests MM embedding adds significant computation, not JIT overhead."
                )
                print()

            # Compare with Standard QPE to assess device mismatch
            if std_vacuum > 0 and std_solvated > 0:
                vacuum_ratio = cat_vacuum / std_vacuum if std_vacuum > 0 else 0
                solvated_ratio = cat_solvated / std_solvated if std_solvated > 0 else 0

                print(f"    Per-Phase Slowdown (Catalyst / Standard):")
                print(f"      Vacuum:   {vacuum_ratio:.2f}x slower")
                print(f"      Solvated: {solvated_ratio:.2f}x slower")
                print()

                if vacuum_ratio > 2.0 and solvated_ratio < 1.5:
                    print("    [DIAGNOSIS] JIT compilation is the PRIMARY bottleneck")
                    print("      - First call (vacuum) is significantly slower due to compilation")
                    print("      - Second call (solvated) shows better performance (cached)")
                elif solvated_ratio > vacuum_ratio:
                    print("    [DIAGNOSIS] Device mismatch or backend inefficiency")
                    print("      - Both calls are slow, suggesting systematic overhead")
                    if not HAS_JAX_CUDA:
                        print("      - Likely cause: Catalyst running on CPU, Standard on GPU")
                        print("      - Solution: pip install 'jax[cuda12]'")
                else:
                    print("    [DIAGNOSIS] Consistent overhead across both phases")
                    print("      - JIT compilation adds fixed overhead to first call")
                    print("      - @qjit benefits realized in multi-iteration scenarios")

    # Legacy comparison (if no fine-grained data)
    elif t_standard > 0 and t_catalyst > 0:
        print("QPE Performance Comparison (Standard vs Catalyst):")
        print("  " + "-" * 60)

        ratio = t_catalyst / t_standard if t_standard > 0 else 0
        print(f"  Standard QPE:  {t_standard:.3f}s")
        print(f"  Catalyst QPE:  {t_catalyst:.3f}s")
        print(f"  Ratio:         {ratio:.2f}x")
        print()

        if ratio > 1.5:
            print("  [DIAGNOSIS] Catalyst is significantly SLOWER than Standard QPE")
            print()
            print("  Potential bottlenecks:")
            print("    1. JIT Compilation Overhead: First execution compiles the circuit")
            print("    2. Device Mismatch: Catalyst may be running on CPU while")
            print("       Standard QPE uses lightning.gpu")
            print("    3. Single Execution: @qjit benefits are realized in multi-run")
            print("       scenarios (VQE loops), not single QPE calls")
            print()
            if not HAS_JAX_CUDA:
                print("  [ACTION] Enable JAX CUDA for Catalyst GPU support:")
                print("           pip install 'jax[cuda12]'")
        elif ratio < 0.8:
            print("  [DIAGNOSIS] Catalyst shows speedup over Standard QPE")
            print("  JIT compilation benefits are being realized.")
        else:
            print("  [DIAGNOSIS] Similar performance between Standard and Catalyst QPE")
            print("  This is expected for single-run scenarios.")

    elif t_standard > 0 and t_catalyst == 0:
        print("QPE Performance (Standard only - Catalyst not available):")
        print(f"  Standard QPE: {t_standard:.3f}s")

    print()
    print("=" * 80)
