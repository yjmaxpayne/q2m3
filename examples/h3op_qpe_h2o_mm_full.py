#!/usr/bin/env python
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
H3O+ Quantum Phase Estimation (QPE) Demo

This script demonstrates q2m3's core capabilities for hybrid quantum-classical
QM/MM calculations targeting early fault-tolerant quantum computers (EFTQC).

Key features demonstrated:
1. PySCF to PennyLane Hamiltonian conversion
2. Standard QPE circuit (HF state prep -> Trotter evolution -> inverse QFT)
3. Catalyst @qjit JIT compilation for circuit optimization
4. QM/MM system setup with TIP3P water solvation

IMPORTANT: GPU Support Architecture
------------------------------------
q2m3 has TWO separate GPU support systems:

1. PennyLane Lightning GPU (for standard QPE):
   - Device: lightning.gpu
   - Requirements: pennylane-lightning[gpu], cuQuantum, CUDA
   - Status checked by: HAS_LIGHTNING_GPU

2. JAX/Catalyst GPU (for @qjit compiled circuits):
   - Backend: JAX with CUDA support
   - Requirements: jax[cuda12] or jax[cuda11]
   - Status checked by: HAS_JAX_CUDA

When Catalyst @qjit is enabled, execution backend is determined by JAX,
NOT PennyLane device selection. If JAX lacks CUDA support, Catalyst
runs on CPU regardless of lightning.gpu availability.
"""

import time
import warnings
from datetime import datetime

import numpy as np
from pyscf import gto

from q2m3.core import QuantumQMMM
from q2m3.core.qmmm_system import Atom, QMMMSystem
from q2m3.interfaces import PySCFPennyLaneConverter
from q2m3.utils import save_json_results

# Check Catalyst availability
try:
    import catalyst

    HAS_CATALYST = True
    CATALYST_VERSION = catalyst.__version__
except ImportError:
    HAS_CATALYST = False
    CATALYST_VERSION = "N/A"

# Check lightning devices availability
import pennylane as qml

# Physical constants and thresholds
HARTREE_TO_KCAL_MOL = 627.5094
MM_STABILIZATION_THRESHOLD = (
    0.001  # Hartree - minimum stabilization to consider MM embedding active
)
ENERGY_CONSISTENCY_THRESHOLD = 0.01  # Hartree - acceptable difference between methods
CHEMICAL_ACCURACY_ERROR = 0.0016  # Hartree (~1 kcal/mol)
RELAXED_ACCURACY_ERROR = 0.016  # Hartree (~10 kcal/mol)

# =============================================================================
# Device Detection: PennyLane Lightning (for standard QPE)
# =============================================================================
HAS_LIGHTNING_GPU = False
try:
    _test_dev = qml.device("lightning.gpu", wires=1)
    del _test_dev
    HAS_LIGHTNING_GPU = True
except Exception:
    pass

HAS_LIGHTNING_QUBIT = False
try:
    _test_dev = qml.device("lightning.qubit", wires=1)
    del _test_dev
    HAS_LIGHTNING_QUBIT = True
except Exception as e:
    warnings.warn(
        f"Could not initialize PennyLane 'lightning.qubit' device: {e}. Falling back to other devices.",
        RuntimeWarning,
        stacklevel=2,
    )

# =============================================================================
# Device Detection: JAX/Catalyst GPU (for @qjit compiled circuits)
# IMPORTANT: This is SEPARATE from PennyLane Lightning GPU!
# =============================================================================
HAS_JAX_CUDA = False
JAX_DEFAULT_BACKEND = "cpu"
try:
    import jax

    JAX_DEFAULT_BACKEND = jax.default_backend()
    HAS_JAX_CUDA = JAX_DEFAULT_BACKEND in ("cuda", "gpu")
except ImportError:
    pass
except Exception:
    pass


def get_best_available_device() -> str:
    """Return the best available PennyLane device name.

    Priority: lightning.gpu > lightning.qubit > default.qubit
    """
    if HAS_LIGHTNING_GPU:
        return "lightning.gpu"
    elif HAS_LIGHTNING_QUBIT:
        return "lightning.qubit"
    return "default.qubit"


def get_catalyst_effective_backend() -> str:
    """Get the actual execution backend for Catalyst @qjit.

    IMPORTANT: Catalyst uses JAX as its backend, which is SEPARATE from
    PennyLane device selection. Even with lightning.gpu device, Catalyst
    runs on CPU if JAX lacks CUDA support.

    Returns:
        Human-readable backend string like "GPU (JAX CUDA)" or "CPU (JAX)"
    """
    if HAS_JAX_CUDA:
        return "GPU (JAX CUDA)"
    return "CPU (JAX)"


def analyze_solvation_effect(h3o_atoms: list[Atom], mm_waters: int) -> dict:
    """
    Analyze the solvation effect on H3O+ energy at the classical HF level.

    Compares vacuum vs solvated H3O+ to demonstrate MM embedding is working.

    Args:
        h3o_atoms: H3O+ atomic geometry
        mm_waters: Number of TIP3P water molecules

    Returns:
        Dictionary with vacuum energy, solvated energy, and stabilization energy
    """
    # Build QM/MM system (MM environment set up automatically in __init__)
    qmmm_system = QMMMSystem(qm_atoms=h3o_atoms, num_waters=mm_waters)
    mol_dict = qmmm_system.to_pyscf_mol()

    # Create PySCF molecule
    mol = gto.M(**mol_dict)

    # Get MM embedding potential
    mm_charges, mm_coords = qmmm_system.get_embedding_potential()

    converter = PySCFPennyLaneConverter()

    # Calculate energy in vacuum (no MM)
    result_vacuum = converter.build_qmmm_hamiltonian(mol, np.array([]), np.array([]).reshape(0, 3))
    energy_vacuum = result_vacuum["energy_hf"]

    # Calculate energy with MM embedding
    result_solvated = converter.build_qmmm_hamiltonian(mol, mm_charges, mm_coords)
    energy_solvated = result_solvated["energy_hf"]

    # Stabilization energy (positive = MM stabilizes the system)
    stabilization = energy_vacuum - energy_solvated
    stabilization_kcal = stabilization * HARTREE_TO_KCAL_MOL

    return {
        "energy_vacuum": energy_vacuum,
        "energy_solvated": energy_solvated,
        "stabilization_hartree": stabilization,
        "stabilization_kcal_mol": stabilization_kcal,
        "n_mm_atoms": len(mm_charges),
        "n_mm_waters": mm_waters,
    }


def analyze_qpe_solvation_effect(
    h3o_atoms: list[Atom], mm_waters: int, qpe_config: dict, use_catalyst: bool = False
) -> dict:
    """
    Analyze solvation effect at the QPE quantum level.

    Compares vacuum vs solvated QPE energies to verify MM embedding
    is correctly included in the quantum Hamiltonian.

    Args:
        h3o_atoms: H3O+ atomic geometry
        mm_waters: Number of TIP3P water molecules
        qpe_config: QPE configuration dictionary
        use_catalyst: Enable Catalyst @qjit compilation

    Returns:
        Dictionary with vacuum QPE, solvated QPE, comparison metrics, and timing
    """
    # QPE in vacuum (no MM waters)
    start_vacuum = time.perf_counter()
    qmmm_vacuum = QuantumQMMM(
        qm_atoms=h3o_atoms,
        mm_waters=0,
        qpe_config=qpe_config,
        use_catalyst=use_catalyst,
    )
    result_vacuum = qmmm_vacuum.compute_ground_state()
    time_vacuum = time.perf_counter() - start_vacuum

    # QPE with MM solvation
    start_solvated = time.perf_counter()
    qmmm_solvated = QuantumQMMM(
        qm_atoms=h3o_atoms,
        mm_waters=mm_waters,
        qpe_config=qpe_config,
        use_catalyst=use_catalyst,
    )
    result_solvated = qmmm_solvated.compute_ground_state()
    time_solvated = time.perf_counter() - start_solvated

    # Compute stabilization
    stabilization = result_vacuum["energy"] - result_solvated["energy"]
    stabilization_kcal = stabilization * HARTREE_TO_KCAL_MOL

    return {
        "energy_vacuum": result_vacuum["energy"],
        "energy_solvated": result_solvated["energy"],
        "energy_hf_vacuum": result_vacuum.get("energy_hf", None),
        "energy_hf_solvated": result_solvated.get("energy_hf", None),
        "stabilization_hartree": stabilization,
        "stabilization_kcal_mol": stabilization_kcal,
        "charges_vacuum": result_vacuum["atomic_charges"],
        "charges_solvated": result_solvated["atomic_charges"],
        "convergence_solvated": result_solvated["convergence"],
        "rdm_source_solvated": result_solvated.get("rdm_source", "unknown"),
        "time_vacuum_s": time_vacuum,
        "time_solvated_s": time_solvated,
        "time_total_s": time_vacuum + time_solvated,
    }


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


def create_h3o_geometry() -> list[Atom]:
    """
    Create H3O+ (hydronium) molecular geometry.

    Returns optimized structure with formal charges for QM/MM calculation.
    O: -2.0 formal charge, H: +1.0 each -> total charge +1
    """
    return [
        Atom("O", np.array([0.000000, 0.000000, 0.000000]), charge=-2.0),
        Atom("H", np.array([0.960000, 0.000000, 0.000000]), charge=1.0),
        Atom("H", np.array([-0.480000, 0.831384, 0.000000]), charge=1.0),
        Atom("H", np.array([-0.480000, -0.831384, 0.000000]), charge=1.0),
    ]


def get_qpe_config(device_type: str = "auto") -> dict:
    """
    Get standard QPE configuration for H3O+ system.

    Active space: 4 electrons, 4 orbitals -> 8 system qubits
    Estimation qubits: 4 (precision bits)
    Total qubits: 12

    Args:
        device_type: Device selection strategy
            - "auto": Auto-select best (GPU > lightning.qubit > default.qubit)
            - "lightning.gpu": Force GPU device
            - "lightning.qubit": Force CPU lightning device
            - "default.qubit": Force standard PennyLane device
    """
    return {
        "use_real_qpe": True,
        "n_estimation_wires": 4,
        "base_time": "auto",  # Auto-compute to avoid phase overflow
        "n_trotter_steps": 10,
        "n_shots": 100,
        "active_electrons": 4,
        "active_orbitals": 4,
        "energy_warning_threshold": 1.0,
        "algorithm": "standard",
        "mapping": "jordan_wigner",
        "device_type": device_type,
    }


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
    print(f"  ΔE = {qpe_solvation_data['stabilization_hartree']:.6f} Hartree")
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
        print(f"  {atom}: {q_vac:+.4f} -> {q_sol:+.4f} (Δq = {delta_q:+.4f})")


def run_resource_estimation(h3o_atoms: list[Atom], mm_waters: int) -> dict:
    """Run EFTQC resource estimation for vacuum and solvated systems.

    Returns:
        Dictionary containing vacuum and solvated resource estimates,
        and derived analysis metrics (delta_lambda, gate_reduction).
    """
    h3o_symbols = [atom.symbol for atom in h3o_atoms]
    h3o_coords = np.array([atom.position for atom in h3o_atoms])

    qmmm_system = QMMMSystem(qm_atoms=h3o_atoms, num_waters=mm_waters)
    mm_charges, mm_coords = qmmm_system.get_embedding_potential()

    converter = PySCFPennyLaneConverter(basis="sto-3g", mapping="jordan_wigner")

    eftqc_vac_chem = converter.estimate_qpe_resources(
        symbols=h3o_symbols, coords=h3o_coords, charge=1, target_error=CHEMICAL_ACCURACY_ERROR
    )
    eftqc_vac_relax = converter.estimate_qpe_resources(
        symbols=h3o_symbols, coords=h3o_coords, charge=1, target_error=RELAXED_ACCURACY_ERROR
    )
    eftqc_sol_chem = converter.estimate_qpe_resources(
        symbols=h3o_symbols,
        coords=h3o_coords,
        charge=1,
        mm_charges=mm_charges,
        mm_coords=mm_coords,
        target_error=CHEMICAL_ACCURACY_ERROR,
    )
    eftqc_sol_relax = converter.estimate_qpe_resources(
        symbols=h3o_symbols,
        coords=h3o_coords,
        charge=1,
        mm_charges=mm_charges,
        mm_coords=mm_coords,
        target_error=RELAXED_ACCURACY_ERROR,
    )

    delta_lambda = (
        (eftqc_sol_chem["hamiltonian_1norm"] - eftqc_vac_chem["hamiltonian_1norm"])
        / eftqc_vac_chem["hamiltonian_1norm"]
        * 100
    )
    gate_reduction = (1 - eftqc_vac_relax["toffoli_gates"] / eftqc_vac_chem["toffoli_gates"]) * 100

    return {
        "vacuum_chemical": eftqc_vac_chem,
        "vacuum_relaxed": eftqc_vac_relax,
        "solvated_chemical": eftqc_sol_chem,
        "solvated_relaxed": eftqc_sol_relax,
        "delta_lambda": delta_lambda,
        "gate_reduction": gate_reduction,
        "n_mm_charges": len(mm_charges),
    }


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
        f"  {'Vacuum (chemical)':<25} {eftqc_vac_chem['hamiltonian_1norm']:<15.2f} {eftqc_vac_chem['toffoli_gates']:<18,} {eftqc_vac_chem['logical_qubits']:<10}"
    )
    print(
        f"  {'Vacuum (relaxed)':<25} {eftqc_vac_relax['hamiltonian_1norm']:<15.2f} {eftqc_vac_relax['toffoli_gates']:<18,} {eftqc_vac_relax['logical_qubits']:<10}"
    )
    print(
        f"  {'Solvated (chemical)':<25} {eftqc_sol_chem['hamiltonian_1norm']:<15.2f} {eftqc_sol_chem['toffoli_gates']:<18,} {eftqc_sol_chem['logical_qubits']:<10}"
    )
    print(
        f"  {'Solvated (relaxed)':<25} {eftqc_sol_relax['hamiltonian_1norm']:<15.2f} {eftqc_sol_relax['toffoli_gates']:<18,} {eftqc_sol_relax['logical_qubits']:<10}"
    )
    print("  " + "-" * 70)
    print()

    print("  Analysis:")
    print(f"    MM embedding effect: Δλ = {eftqc_data['delta_lambda']:+.1f}% (1-norm increase)")
    print(
        f"    Error relaxation:    {eftqc_data['gate_reduction']:.1f}% fewer gates (10× error tolerance)"
    )
    print()
    print("  Note: MM embedding only modifies 1-electron integrals (QM-MM electrostatics).")
    print("        Resource estimates are dominated by 2-electron integrals, which remain")
    print("        unchanged. Thus, vacuum vs solvated show similar resource requirements.")


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
    print(f"  ΔE = {solvation_data['stabilization_hartree']:.6f} Hartree")
    print(f"     = {solvation_data['stabilization_kcal_mol']:.2f} kcal/mol")
    print()
    if solvation_data["stabilization_hartree"] > MM_STABILIZATION_THRESHOLD:
        print("  [OK] MM embedding is working: explicit solvent stabilizes H3O+")
    else:
        print("  [WARNING] Unexpected: no significant stabilization detected")


def run_catalyst_analysis(
    h3o_atoms: list[Atom], mm_waters: int, solvation_data: dict
) -> dict | None:
    """Run Catalyst @qjit QPE solvation effect analysis.

    Returns:
        Catalyst solvation data dict if Catalyst is available, None otherwise.
    """
    if not HAS_CATALYST:
        print("WARNING: pennylane-catalyst is not installed.")
        print("To enable Catalyst support, install with:")
        print("  pip install pennylane-catalyst")
        print()
        print("Skipping Catalyst solvation effect analysis...")
        return None

    print("Comparing Catalyst QPE energies: vacuum vs. explicit TIP3P solvation...")
    print("This validates MM embedding works correctly with Catalyst JIT compilation.")
    print()

    # Use auto device selection (now supports lightning.gpu as of PennyLane 0.44.0)
    qpe_config_catalyst = get_qpe_config(device_type="auto")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        catalyst_solvation_data = analyze_qpe_solvation_effect(
            h3o_atoms, mm_waters, qpe_config_catalyst, use_catalyst=True
        )

    print_qpe_solvation_effect(catalyst_solvation_data, solvation_data, label="Catalyst QPE")
    return catalyst_solvation_data


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


def build_output_data(
    h3o_atoms: list[Atom],
    mm_waters: int,
    qpe_config: dict,
    solvation_data: dict,
    qpe_solvation_data: dict,
    catalyst_solvation_data: dict | None,
    eftqc_data: dict,
) -> dict:
    """Build output data dictionary for JSON export."""
    actual_device = get_best_available_device()
    eftqc_vac_chem = eftqc_data["vacuum_chemical"]
    eftqc_vac_relax = eftqc_data["vacuum_relaxed"]
    eftqc_sol_chem = eftqc_data["solvated_chemical"]
    eftqc_sol_relax = eftqc_data["solvated_relaxed"]

    return {
        "timestamp": datetime.now().isoformat(),
        "catalyst_available": HAS_CATALYST,
        "catalyst_version": CATALYST_VERSION if HAS_CATALYST else None,
        "lightning_gpu_available": HAS_LIGHTNING_GPU,
        "jax_cuda_available": HAS_JAX_CUDA,  # Separate from Lightning GPU!
        "jax_backend": JAX_DEFAULT_BACKEND,  # Actual JAX execution backend
        "system": {
            "qm_region": "H3O+",
            "n_atoms": len(h3o_atoms),
            "total_charge": 1,
            "mm_waters": mm_waters,
        },
        "qpe_config": qpe_config,
        "quantum_resources": {
            "active_electrons": qpe_config["active_electrons"],
            "active_orbitals": qpe_config["active_orbitals"],
            "system_qubits": qpe_config["active_orbitals"] * 2,
            "estimation_qubits": qpe_config["n_estimation_wires"],
            "total_qubits": qpe_config["active_orbitals"] * 2 + qpe_config["n_estimation_wires"],
        },
        "solvation_effect_hf": {
            "energy_vacuum_hartree": solvation_data["energy_vacuum"],
            "energy_solvated_hartree": solvation_data["energy_solvated"],
            "stabilization_hartree": solvation_data["stabilization_hartree"],
            "stabilization_kcal_mol": solvation_data["stabilization_kcal_mol"],
            "mm_embedding_active": bool(
                solvation_data["stabilization_hartree"] > MM_STABILIZATION_THRESHOLD
            ),
        },
        "solvation_effect_standard_qpe": {
            "device": actual_device,
            "energy_vacuum_hartree": qpe_solvation_data["energy_vacuum"],
            "energy_solvated_hartree": qpe_solvation_data["energy_solvated"],
            "energy_hf_solvated": qpe_solvation_data["energy_hf_solvated"],
            "stabilization_hartree": qpe_solvation_data["stabilization_hartree"],
            "stabilization_kcal_mol": qpe_solvation_data["stabilization_kcal_mol"],
            "mm_embedding_active": bool(
                qpe_solvation_data["stabilization_hartree"] > MM_STABILIZATION_THRESHOLD
            ),
            "charges_vacuum": qpe_solvation_data["charges_vacuum"],
            "charges_solvated": qpe_solvation_data["charges_solvated"],
            "execution_time_vacuum_s": qpe_solvation_data["time_vacuum_s"],
            "execution_time_solvated_s": qpe_solvation_data["time_solvated_s"],
            "execution_time_total_s": qpe_solvation_data["time_total_s"],
        },
        "solvation_effect_catalyst_qpe": (
            {
                "pennylane_device": get_best_available_device(),  # PennyLane device name
                "jax_backend": JAX_DEFAULT_BACKEND,  # Actual JAX execution backend
                "effective_backend": get_catalyst_effective_backend(),  # Human-readable
                "has_jax_cuda": HAS_JAX_CUDA,  # Whether Catalyst uses GPU
                "energy_vacuum_hartree": catalyst_solvation_data["energy_vacuum"],
                "energy_solvated_hartree": catalyst_solvation_data["energy_solvated"],
                "energy_hf_solvated": catalyst_solvation_data["energy_hf_solvated"],
                "stabilization_hartree": catalyst_solvation_data["stabilization_hartree"],
                "stabilization_kcal_mol": catalyst_solvation_data["stabilization_kcal_mol"],
                "mm_embedding_active": bool(
                    catalyst_solvation_data["stabilization_hartree"] > MM_STABILIZATION_THRESHOLD
                ),
                "charges_vacuum": catalyst_solvation_data["charges_vacuum"],
                "charges_solvated": catalyst_solvation_data["charges_solvated"],
                "execution_time_vacuum_s": catalyst_solvation_data["time_vacuum_s"],
                "execution_time_solvated_s": catalyst_solvation_data["time_solvated_s"],
                "execution_time_total_s": catalyst_solvation_data["time_total_s"],
            }
            if catalyst_solvation_data is not None
            else None
        ),
        "eftqc_resources": {
            "vacuum": {
                "chemical_accuracy": {
                    "target_error_hartree": eftqc_vac_chem["target_error"],
                    "hamiltonian_1norm": eftqc_vac_chem["hamiltonian_1norm"],
                    "logical_qubits": eftqc_vac_chem["logical_qubits"],
                    "toffoli_gates": eftqc_vac_chem["toffoli_gates"],
                    "qpe_iterations": eftqc_vac_chem["qpe_iterations"],
                    "trotter_steps": eftqc_vac_chem["trotter_steps"],
                },
                "relaxed_accuracy": {
                    "target_error_hartree": eftqc_vac_relax["target_error"],
                    "hamiltonian_1norm": eftqc_vac_relax["hamiltonian_1norm"],
                    "logical_qubits": eftqc_vac_relax["logical_qubits"],
                    "toffoli_gates": eftqc_vac_relax["toffoli_gates"],
                    "qpe_iterations": eftqc_vac_relax["qpe_iterations"],
                    "trotter_steps": eftqc_vac_relax["trotter_steps"],
                },
            },
            "solvated": {
                "n_mm_charges": eftqc_sol_chem["n_mm_charges"],
                "chemical_accuracy": {
                    "target_error_hartree": eftqc_sol_chem["target_error"],
                    "hamiltonian_1norm": eftqc_sol_chem["hamiltonian_1norm"],
                    "logical_qubits": eftqc_sol_chem["logical_qubits"],
                    "toffoli_gates": eftqc_sol_chem["toffoli_gates"],
                    "qpe_iterations": eftqc_sol_chem["qpe_iterations"],
                    "trotter_steps": eftqc_sol_chem["trotter_steps"],
                },
                "relaxed_accuracy": {
                    "target_error_hartree": eftqc_sol_relax["target_error"],
                    "hamiltonian_1norm": eftqc_sol_relax["hamiltonian_1norm"],
                    "logical_qubits": eftqc_sol_relax["logical_qubits"],
                    "toffoli_gates": eftqc_sol_relax["toffoli_gates"],
                    "qpe_iterations": eftqc_sol_relax["qpe_iterations"],
                    "trotter_steps": eftqc_sol_relax["trotter_steps"],
                },
            },
            "analysis": {
                "mm_embedding_1norm_increase_percent": eftqc_data["delta_lambda"],
                "error_relaxation_gate_reduction_percent": eftqc_data["gate_reduction"],
            },
        },
    }


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
            f"  [OK] HF MM embedding (ΔE = {solvation_data['stabilization_kcal_mol']:.1f} kcal/mol)"
        )
    else:
        print("  [--] HF MM embedding (no stabilization detected)")

    if qpe_solvation_data["stabilization_hartree"] > MM_STABILIZATION_THRESHOLD:
        print(
            f"  [OK] Standard QPE MM embedding "
            f"(ΔE = {qpe_solvation_data['stabilization_kcal_mol']:.1f} kcal/mol)"
        )
    else:
        print("  [--] Standard QPE MM embedding (no stabilization detected)")

    if catalyst_solvation_data is not None:
        if catalyst_solvation_data["stabilization_hartree"] > MM_STABILIZATION_THRESHOLD:
            print(
                f"  [OK] Catalyst QPE MM embedding "
                f"(ΔE = {catalyst_solvation_data['stabilization_kcal_mol']:.1f} kcal/mol)"
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


def main():
    """Main demo execution."""
    print_header()

    # Step 1: System configuration
    print_section("System Configuration", step=1)
    h3o_atoms = create_h3o_geometry()
    mm_waters = 8
    qpe_config = get_qpe_config(device_type="auto")

    print_system_info(qpe_config, mm_waters)
    print()
    selected_device = get_best_available_device()
    device_note = " (GPU detected)" if HAS_LIGHTNING_GPU else ""
    print(f"Device Selection: auto -> {selected_device}{device_note}")

    # Step 1.5: Circuit visualization (show circuit structure before calculations)
    print_section("Circuit Visualization (PennyLane)", step=1.5)
    print("Generating QPE + RDM circuit diagrams...")
    print()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        viz_qmmm = QuantumQMMM(
            qm_atoms=h3o_atoms,
            mm_waters=mm_waters,
            qpe_config=qpe_config,
            use_catalyst=False,
        )
        circuits = viz_qmmm.draw_circuits()

    print("QPE Circuit (Standard Phase Estimation):")
    print("-" * 60)
    print(circuits["qpe"])
    print()
    print("RDM Measurement Circuit (Pauli Expectation Values):")
    print("-" * 60)
    print(circuits["rdm"])
    print()

    # Step 2: EFTQC Resource Estimation
    print_section("EFTQC Resource Estimation (Vacuum vs Solvated)", step=2)
    eftqc_data = run_resource_estimation(h3o_atoms, mm_waters)
    print_resource_estimation(eftqc_data, mm_waters)

    # Step 3: Solvation effect analysis (classical HF level)
    print_section("Solvation Effect Analysis (Classical HF)", step=3)
    print("Comparing H3O+ energy in vacuum vs. explicit TIP3P water environment...")
    print("This validates that MM embedding correctly polarizes the QM electron density.")
    print()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solvation_data = analyze_solvation_effect(h3o_atoms, mm_waters)

    print_hf_solvation_effect(solvation_data)

    # Step 4: QPE solvation effect analysis (Standard QPE)
    print_section("Standard QPE Solvation Effect Analysis (Quantum Level)", step=4)
    print("Comparing QPE energies: vacuum vs. explicit TIP3P solvation...")
    print("This validates MM embedding is correctly included in the quantum Hamiltonian.")
    print()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qpe_solvation_data = analyze_qpe_solvation_effect(
            h3o_atoms, mm_waters, qpe_config, use_catalyst=False
        )

    print_qpe_solvation_effect(qpe_solvation_data, solvation_data, label="Standard QPE")

    # Step 5: Catalyst @qjit QPE Solvation Effect Analysis
    print_section("Catalyst @qjit QPE Solvation Effect Analysis", step=5)
    catalyst_solvation_data = run_catalyst_analysis(h3o_atoms, mm_waters, solvation_data)

    # Step 6: Results comparison
    print_section("Results Comparison", step=6)
    print_comparison(qpe_solvation_data, catalyst_solvation_data)

    # Step 7: Save results
    print_section("Save Results", step=7)
    output_data = build_output_data(
        h3o_atoms,
        mm_waters,
        qpe_config,
        solvation_data,
        qpe_solvation_data,
        catalyst_solvation_data,
        eftqc_data,
    )

    output_file = "data/output/h3o_quantum_qpe_results.json"
    save_json_results(output_data, output_file)
    print(f"Results saved to: {output_file}")

    # Summary
    print_section("Demo Summary")
    print_summary(solvation_data, qpe_solvation_data, catalyst_solvation_data, eftqc_data)


if __name__ == "__main__":
    main()
