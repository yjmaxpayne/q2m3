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
"""

import time
import warnings
from datetime import datetime

import numpy as np

from q2m3.core import QuantumQMMM
from q2m3.core.qmmm_system import Atom
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
except Exception:
    pass


def print_header():
    """Print demo header."""
    print("=" * 80)
    print("                    H3O+ Quantum Phase Estimation (QPE) Demo")
    print("                    q2m3 MVP - Catalyst Technical Validation")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Catalyst Available: {'Yes (v' + CATALYST_VERSION + ')' if HAS_CATALYST else 'No'}")
    print(f"Lightning GPU Available: {'Yes' if HAS_LIGHTNING_GPU else 'No'}")
    print()


def print_section(title: str, step: int = None):
    """Print section header."""
    if step:
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


def run_qpe_calculation(
    h3o_atoms: list[Atom],
    qpe_config: dict,
    mm_waters: int,
    use_catalyst: bool,
    label: str,
) -> tuple[dict, float]:
    """
    Execute QPE calculation with timing.

    Args:
        h3o_atoms: H3O+ atomic geometry
        qpe_config: QPE configuration dictionary
        mm_waters: Number of TIP3P water molecules
        use_catalyst: Enable Catalyst @qjit compilation
        label: Description for logging

    Returns:
        Tuple of (results dict, execution time in seconds)
    """
    print(f"Executing: {label}...")

    # Suppress PySCF output for cleaner demo
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        start_time = time.perf_counter()

        qmmm = QuantumQMMM(
            qm_atoms=h3o_atoms,
            mm_waters=mm_waters,
            qpe_config=qpe_config,
            use_catalyst=use_catalyst,
        )
        results = qmmm.compute_ground_state()

        end_time = time.perf_counter()

    execution_time = end_time - start_time
    return results, execution_time


def print_system_info(qpe_config: dict, mm_waters: int):
    """Print system configuration details."""
    active_e = qpe_config["active_electrons"]
    active_o = qpe_config["active_orbitals"]
    system_qubits = active_o * 2  # 2 spin orbitals per spatial orbital
    estimation_qubits = qpe_config["n_estimation_wires"]
    total_qubits = system_qubits + estimation_qubits

    print(f"QM Region: H3O+ (4 atoms, total charge +1)")
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


def print_results(results: dict, execution_time: float, label: str):
    """Print calculation results."""
    energy_qpe = results["energy"]
    energy_hf = results.get("energy_hf", "N/A")
    energy_diff = results.get("energy_difference", "N/A")
    rdm_source = results.get("rdm_source", "unknown")

    print(f"Execution Time: {execution_time:.3f} s")
    print()
    print("Energy Results:")
    if isinstance(energy_hf, float):
        print(f"  HF Reference Energy: {energy_hf:.6f} Hartree")
    print(f"  QPE Estimated Energy: {energy_qpe:.6f} Hartree")
    if isinstance(energy_diff, float):
        print(f"  Energy Difference: {energy_diff:.6f} Hartree")
    print()

    # Convergence info
    conv = results["convergence"]
    print("Convergence Status:")
    print(f"  Converged: {'Yes' if conv['converged'] else 'No'}")
    print(f"  Method: {conv.get('method', 'N/A')}")
    if conv.get("rdm_enabled"):
        print(f"  RDM Measurement: Enabled")

    # Mulliken charges with RDM source indication
    charges = results["atomic_charges"]
    total_charge = sum(charges.values())
    print()
    print(f"Mulliken Population Analysis (RDM source: {rdm_source}):")
    for atom_label, charge in charges.items():
        print(f"  {atom_label}: {charge:+.4f}")
    print(f"  Total Charge: {total_charge:+.4f}")


def print_comparison(
    results_standard: dict,
    time_standard: float,
    results_catalyst: dict,
    time_catalyst: float,
):
    """Print comparison between standard and Catalyst execution."""
    print("Execution Time Comparison:")
    std_device = (
        "lightning.gpu"
        if HAS_LIGHTNING_GPU
        else "lightning.qubit" if HAS_LIGHTNING_QUBIT else "default.qubit"
    )
    print(f"  Standard QPE ({std_device}): {time_standard:.3f} s")
    print(f"  Catalyst QPE (lightning.qubit): {time_catalyst:.3f} s")

    if time_catalyst > 0:
        speedup = time_standard / time_catalyst
        if speedup > 1:
            print(f"  Speedup: {speedup:.2f}x faster with Catalyst")
        else:
            print(f"  Ratio: {1/speedup:.2f}x slower with Catalyst")
            if HAS_LIGHTNING_GPU:
                print("  Note: Catalyst cannot use lightning.gpu for qml.ctrl(TrotterProduct),")
                print("        so it falls back to lightning.qubit (CPU). This explains the")
                print("        performance difference when GPU is available.")
    print()

    print("Energy Comparison:")
    e_std = results_standard["energy"]
    e_cat = results_catalyst["energy"]
    e_diff = abs(e_std - e_cat)
    print(f"  Standard QPE: {e_std:.6f} Hartree")
    print(f"  Catalyst QPE: {e_cat:.6f} Hartree")
    print(f"  Difference: {e_diff:.6f} Hartree")

    if e_diff < 0.01:
        print("  Status: Results consistent (diff < 0.01 Ha)")
    else:
        print("  Status: Results differ (stochastic QPE sampling)")


def main():
    """Main demo execution."""
    print_header()

    # Step 1: System configuration
    print_section("System Configuration", step=1)
    h3o_atoms = create_h3o_geometry()
    mm_waters = 8

    # Auto device selection: GPU > lightning.qubit > default.qubit
    qpe_config = get_qpe_config(device_type="auto")

    print_system_info(qpe_config, mm_waters)
    print()
    print(f"Device Selection: auto -> ", end="")
    if HAS_LIGHTNING_GPU:
        print("lightning.gpu (GPU detected)")
    elif HAS_LIGHTNING_QUBIT:
        print("lightning.qubit")
    else:
        print("default.qubit")

    # Step 2: Circuit visualization
    print_section("Circuit Visualization (PennyLane)", step=2)
    print("Generating QPE + RDM circuit diagrams...")
    print()

    # Create a temporary QuantumQMMM instance for visualization
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

    # Step 3: QPE execution with best available device
    print_section("QPE Execution (auto device selection)", step=3)
    results_standard, time_standard = run_qpe_calculation(
        h3o_atoms=h3o_atoms,
        qpe_config=qpe_config,
        mm_waters=mm_waters,
        use_catalyst=False,
        label=f"QPE with {'lightning.gpu' if HAS_LIGHTNING_GPU else 'lightning.qubit' if HAS_LIGHTNING_QUBIT else 'default.qubit'}",
    )
    print_results(results_standard, time_standard, "Standard")

    # Step 4: Catalyst @qjit QPE execution
    print_section("Catalyst @qjit QPE Execution", step=4)
    print("NOTE: Catalyst @qjit uses lightning.qubit instead of lightning.gpu due to")
    print("      compatibility issues with qml.ctrl(TrotterProduct) gate (custatevec error).")
    print("      See: examples/README.md - 'Catalyst @qjit + lightning.gpu Incompatibility'")
    print()
    if HAS_CATALYST:
        qpe_config_catalyst = get_qpe_config(device_type="lightning.qubit")
        results_catalyst, time_catalyst = run_qpe_calculation(
            h3o_atoms=h3o_atoms,
            qpe_config=qpe_config_catalyst,
            mm_waters=mm_waters,
            use_catalyst=True,
            label="Catalyst QPE with lightning.qubit + @qjit",
        )
        print_results(results_catalyst, time_catalyst, "Catalyst")
    else:
        print("WARNING: pennylane-catalyst is not installed.")
        print("To enable Catalyst support, install with:")
        print("  pip install pennylane-catalyst")
        print()
        print("Using standard execution as fallback...")
        results_catalyst, time_catalyst = results_standard, time_standard

    # Step 5: Results comparison
    print_section("Results Comparison", step=5)
    print_comparison(results_standard, time_standard, results_catalyst, time_catalyst)

    # Step 6: Save results
    print_section("Save Results", step=6)
    # Determine actual device used
    actual_device = (
        "lightning.gpu"
        if HAS_LIGHTNING_GPU
        else "lightning.qubit" if HAS_LIGHTNING_QUBIT else "default.qubit"
    )
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "catalyst_available": HAS_CATALYST,
        "catalyst_version": CATALYST_VERSION if HAS_CATALYST else None,
        "lightning_gpu_available": HAS_LIGHTNING_GPU,
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
        "results_standard": {
            "device": actual_device,
            "energy": results_standard["energy"],
            "energy_hf": results_standard.get("energy_hf"),
            "execution_time_s": time_standard,
            "atomic_charges": results_standard["atomic_charges"],
        },
        "results_catalyst": {
            # Catalyst always uses lightning.qubit due to qml.ctrl(TrotterProduct) incompatibility
            "device": "lightning.qubit" if HAS_CATALYST else actual_device,
            "energy": results_catalyst["energy"],
            "energy_hf": results_catalyst.get("energy_hf"),
            "execution_time_s": time_catalyst,
            "atomic_charges": results_catalyst["atomic_charges"],
            "catalyst_enabled": HAS_CATALYST,
        },
    }

    output_file = "data/output/h3o_quantum_qpe_results.json"
    save_json_results(output_data, output_file)
    print(f"Results saved to: {output_file}")

    # Summary
    print_section("Demo Summary")
    print("q2m3 MVP Capabilities Demonstrated:")
    print("  [OK] PySCF -> PennyLane Hamiltonian conversion")
    print("  [OK] Standard QPE circuit implementation")
    print("  [OK] HF state preparation (qml.BasisState)")
    print("  [OK] Trotter time evolution (qml.TrotterProduct)")
    print("  [OK] Inverse QFT (qml.adjoint(qml.QFT))")
    print("  [OK] Phase-to-energy extraction")
    print("  [OK] QM/MM system with TIP3P solvation")
    print("  [OK] Quantum RDM measurement (Pauli expectation values)")
    print("  [OK] Mulliken population analysis (from quantum RDM)")
    print("  [OK] Circuit visualization (qml.draw)")
    if HAS_CATALYST:
        print("  [OK] Catalyst @qjit JIT compilation (lightning.qubit only)")
        print("       -> Note: qml.ctrl(TrotterProduct) incompatible with lightning.gpu")
    else:
        print("  [--] Catalyst @qjit (not installed)")
    if HAS_LIGHTNING_GPU:
        print("  [OK] GPU acceleration (lightning.gpu, standard QPE only)")
    else:
        print("  [--] GPU acceleration (lightning.gpu not available)")
    print()
    print("=" * 80)
    print("                           Demo Completed Successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()
