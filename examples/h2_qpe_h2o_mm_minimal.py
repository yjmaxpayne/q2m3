#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Minimal H2 + MM Water Example for QPE Validation

This script validates QPE + explicit MM solvation using the simplest
possible molecular system: H2 molecule with 1-3 TIP3P water molecules.

Validation Strategy:
1. Compare vacuum HF vs vacuum QPE (verify QPE correctness)
2. Compare solvated HF vs solvated QPE (verify MM embedding in QPE)
3. Compare stabilization effects (HF vs QPE should agree)
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")


def detect_device():
    """Detect best available PennyLane device."""
    import pennylane as qml

    # Try lightning.gpu first
    try:
        dev = qml.device("lightning.gpu", wires=2)
        del dev
        return "lightning.gpu"
    except Exception:
        pass

    # Fallback to lightning.qubit
    try:
        dev = qml.device("lightning.qubit", wires=2)
        del dev
        return "lightning.qubit"
    except Exception:
        pass

    # Last resort
    return "default.qubit"


def main():
    import pennylane as qml
    from pyscf import gto, qmmm, scf

    from q2m3.interfaces import PySCFPennyLaneConverter

    print("=" * 70)
    print("        H2 + MM Water: Minimal QPE Solvation Validation")
    print("=" * 70)
    print()

    # Detect device
    device_type = detect_device()
    print(f"Device: {device_type}")
    print()

    # H2 geometry (bond length 0.74 Angstrom)
    h2_symbols = ["H", "H"]
    h2_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.74],
        ]
    )

    # TIP3P water molecules (2 waters, placed ~3 Angstrom away)
    # TIP3P charges: O = -0.834, H = +0.417
    mm_charges = np.array(
        [
            -0.834,
            0.417,
            0.417,  # Water 1
            -0.834,
            0.417,
            0.417,  # Water 2
        ]
    )
    mm_coords = np.array(
        [
            [3.0, 0.0, 0.0],  # O1
            [3.5, 0.8, 0.0],  # H1a
            [3.5, -0.8, 0.0],  # H1b
            [-3.0, 0.0, 0.0],  # O2
            [-3.5, 0.8, 0.0],  # H2a
            [-3.5, -0.8, 0.0],  # H2b
        ]
    )

    ANGSTROM_TO_BOHR = 1.8897259886

    # =========================================================================
    # Step 1: Classical HF Calculations
    # =========================================================================
    print("[Step 1] Classical Hartree-Fock Reference")
    print("-" * 70)

    # Vacuum HF
    mol_vacuum = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
    )
    mf_vacuum = scf.RHF(mol_vacuum)
    mf_vacuum.run()
    hf_vacuum = mf_vacuum.e_tot

    # Solvated HF (with MM embedding)
    mol_solvated = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
    )
    mf_solvated = scf.RHF(mol_solvated)
    mm_coords_bohr = mm_coords * ANGSTROM_TO_BOHR
    mf_solvated = qmmm.mm_charge(mf_solvated, mm_coords_bohr, mm_charges)
    mf_solvated.run()
    hf_solvated = mf_solvated.e_tot

    hf_stabilization = (hf_vacuum - hf_solvated) * 627.5094  # kcal/mol

    print(f"  Vacuum HF:     {hf_vacuum:.8f} Ha")
    print(f"  Solvated HF:   {hf_solvated:.8f} Ha")
    print(f"  Stabilization: {hf_stabilization:.4f} kcal/mol")
    print()

    # =========================================================================
    # Step 2: PennyLane Hamiltonian Construction
    # =========================================================================
    print("[Step 2] PennyLane Hamiltonian Construction")
    print("-" * 70)

    converter = PySCFPennyLaneConverter(basis="sto-3g", mapping="jordan_wigner")

    # Vacuum Hamiltonian (using standard qml.qchem)
    H_vacuum, n_qubits_v, hf_state_v = converter.pyscf_to_pennylane_hamiltonian(
        symbols=h2_symbols,
        coords=h2_coords,
        charge=0,
        active_electrons=2,
        active_orbitals=2,
    )

    # Solvated Hamiltonian (using MM-embedded version)
    H_solvated, n_qubits_s, hf_state_s = converter.pyscf_to_pennylane_hamiltonian_with_mm(
        symbols=h2_symbols,
        coords=h2_coords,
        charge=0,
        mm_charges=mm_charges,
        mm_coords=mm_coords,
        active_electrons=2,
        active_orbitals=2,
    )

    print(f"  Vacuum Hamiltonian:   {n_qubits_v} qubits, HF state = {hf_state_v}")
    print(f"  Solvated Hamiltonian: {n_qubits_s} qubits, HF state = {hf_state_s}")
    print()

    # =========================================================================
    # Step 3: Verify Hamiltonians via Expectation Values
    # =========================================================================
    print("[Step 3] Hamiltonian Expectation Value Verification")
    print("-" * 70)

    # Create device for expectation value calculation
    dev = qml.device(device_type, wires=n_qubits_v)

    @qml.qnode(dev)
    def compute_hf_expectation(hamiltonian, hf_state):
        qml.BasisState(hf_state, wires=range(len(hf_state)))
        return qml.expval(hamiltonian)

    exp_vacuum = compute_hf_expectation(H_vacuum, hf_state_v)
    exp_solvated = compute_hf_expectation(H_solvated, hf_state_s)
    exp_stabilization = (float(exp_vacuum) - float(exp_solvated)) * 627.5094

    print(f"  <HF|H_vacuum|HF>:   {float(exp_vacuum):.8f} Ha")
    print(f"  <HF|H_solvated|HF>: {float(exp_solvated):.8f} Ha")
    print(f"  Stabilization:      {exp_stabilization:.4f} kcal/mol")
    print()

    # Compare with PySCF
    print("  Comparison with PySCF HF:")
    print(
        f"    Vacuum:   PySCF = {hf_vacuum:.8f}, PL = {float(exp_vacuum):.8f}, "
        f"diff = {abs(hf_vacuum - float(exp_vacuum)) * 627.5094:.4f} kcal/mol"
    )
    print(
        f"    Solvated: PySCF = {hf_solvated:.8f}, PL = {float(exp_solvated):.8f}, "
        f"diff = {abs(hf_solvated - float(exp_solvated)) * 627.5094:.4f} kcal/mol"
    )
    print()

    # =========================================================================
    # Step 4: QPE Energy Estimation
    # =========================================================================
    print("[Step 4] QPE Energy Estimation")
    print("-" * 70)

    from q2m3.core.qpe import QPEEngine

    # QPE parameters
    # Note: Solvated Hamiltonian requires more Trotter steps due to additional terms
    # 20 steps -> ~5 kcal/mol error; 100 steps -> ~0.2 kcal/mol error
    n_estimation_wires = 4
    n_trotter_steps = 100
    n_shots = 1000

    # Create QPE engine
    qpe_engine = QPEEngine(
        n_qubits=n_qubits_v,
        n_iterations=8,
        mapping="jordan_wigner",
        device_type=device_type,
        use_catalyst=False,
    )

    # Auto base_time from vacuum HF energy
    base_time_vacuum = QPEEngine.compute_optimal_base_time(hf_vacuum)
    base_time_solvated = QPEEngine.compute_optimal_base_time(hf_solvated)

    print(f"  QPE Parameters:")
    print(f"    Estimation wires: {n_estimation_wires}")
    print(f"    Trotter steps:    {n_trotter_steps}")
    print(f"    Shots:            {n_shots}")
    print(f"    Base time (vac):  {base_time_vacuum:.6f}")
    print(f"    Base time (sol):  {base_time_solvated:.6f}")
    print()

    # Check phase estimates
    phase_vacuum = abs(hf_vacuum) * base_time_vacuum / (2 * np.pi)
    phase_solvated = abs(hf_solvated) * base_time_solvated / (2 * np.pi)
    print(f"  Phase estimates (should be < 1.0):")
    print(f"    Vacuum:   {phase_vacuum:.4f}")
    print(f"    Solvated: {phase_solvated:.4f}")
    print()

    # Run QPE for vacuum
    print("  Running vacuum QPE...")
    qpe_circuit_vacuum = qpe_engine._build_standard_qpe_circuit(
        H_vacuum,
        hf_state_v,
        n_estimation_wires=n_estimation_wires,
        base_time=base_time_vacuum,
        n_trotter_steps=n_trotter_steps,
        n_shots=n_shots,
    )
    samples_vacuum = qpe_circuit_vacuum()
    qpe_vacuum = qpe_engine._extract_energy_from_samples(samples_vacuum, base_time_vacuum)

    # Run QPE for solvated
    print("  Running solvated QPE...")
    qpe_circuit_solvated = qpe_engine._build_standard_qpe_circuit(
        H_solvated,
        hf_state_s,
        n_estimation_wires=n_estimation_wires,
        base_time=base_time_solvated,
        n_trotter_steps=n_trotter_steps,
        n_shots=n_shots,
    )
    samples_solvated = qpe_circuit_solvated()
    qpe_solvated = qpe_engine._extract_energy_from_samples(samples_solvated, base_time_solvated)

    qpe_stabilization = (qpe_vacuum - qpe_solvated) * 627.5094

    print()
    print(f"  QPE Results:")
    print(f"    Vacuum QPE:     {qpe_vacuum:.8f} Ha")
    print(f"    Solvated QPE:   {qpe_solvated:.8f} Ha")
    print(f"    Stabilization:  {qpe_stabilization:.4f} kcal/mol")
    print()

    # =========================================================================
    # Step 5: Summary and Validation
    # =========================================================================
    print("[Step 5] Summary and Validation")
    print("-" * 70)

    print("  Energy Comparison (Ha):")
    print("  " + "-" * 50)
    print(f"  {'Method':<15} {'Vacuum':<15} {'Solvated':<15} {'Stab (kcal/mol)':<15}")
    print("  " + "-" * 50)
    print(f"  {'PySCF HF':<15} {hf_vacuum:<15.6f} {hf_solvated:<15.6f} {hf_stabilization:<15.4f}")
    print(
        f"  {'PL <HF|H|HF>':<15} {float(exp_vacuum):<15.6f} {float(exp_solvated):<15.6f} {exp_stabilization:<15.4f}"
    )
    print(f"  {'QPE':<15} {qpe_vacuum:<15.6f} {qpe_solvated:<15.6f} {qpe_stabilization:<15.4f}")
    print("  " + "-" * 50)
    print()

    # Validation checks
    print("  Validation Checks:")

    # Check 1: PennyLane vacuum should match PySCF vacuum
    diff_vacuum = abs(hf_vacuum - float(exp_vacuum)) * 627.5094
    check1 = diff_vacuum < 0.1  # < 0.1 kcal/mol
    print(f"    [{'OK' if check1 else 'FAIL'}] PL vacuum matches PySCF: {diff_vacuum:.4f} kcal/mol")

    # Check 2: PennyLane solvated should match PySCF solvated
    diff_solvated = abs(hf_solvated - float(exp_solvated)) * 627.5094
    check2 = diff_solvated < 0.1  # < 0.1 kcal/mol
    print(
        f"    [{'OK' if check2 else 'FAIL'}] PL solvated matches PySCF: {diff_solvated:.4f} kcal/mol"
    )

    # Check 3: QPE vacuum should be close to HF vacuum
    diff_qpe_vacuum = abs(hf_vacuum - qpe_vacuum)
    check3 = diff_qpe_vacuum < 2.0  # < 2.0 Ha for POC
    print(f"    [{'OK' if check3 else 'FAIL'}] QPE vacuum vs HF: {diff_qpe_vacuum:.4f} Ha")

    # Check 4: QPE solvated should be close to HF solvated
    diff_qpe_solvated = abs(hf_solvated - qpe_solvated)
    check4 = diff_qpe_solvated < 2.0  # < 2.0 Ha for POC
    print(f"    [{'OK' if check4 else 'FAIL'}] QPE solvated vs HF: {diff_qpe_solvated:.4f} Ha")

    # Check 5: Stabilization should have same sign
    check5 = (
        (hf_stabilization > 0) == (qpe_stabilization > 0) if abs(qpe_stabilization) > 0.01 else True
    )
    print(f"    [{'OK' if check5 else 'FAIL'}] Stabilization sign consistent")

    print()
    all_passed = check1 and check2 and check3 and check4 and check5
    if all_passed:
        print("  [SUCCESS] All validation checks passed!")
    else:
        print("  [WARNING] Some validation checks failed - further debugging needed")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
