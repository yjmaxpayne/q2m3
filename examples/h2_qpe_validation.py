#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Minimal H2 + MM Water Example for QPE Validation + Catalyst Benchmark

This script validates QPE + explicit MM solvation using the simplest
possible molecular system: H2 molecule with 1-3 TIP3P water molecules.

Validation Strategy:
1. Compare vacuum HF vs vacuum QPE (verify QPE correctness)
2. Compare solvated HF vs solvated QPE (verify MM embedding in QPE)
3. Compare stabilization effects (HF vs QPE should agree)

Catalyst Benchmark:
4. Compare lightning.gpu vs qjit+lightning.gpu performance
"""

import time
import warnings
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore")

# =============================================================================
# Device Detection: PennyLane Lightning (for standard QPE)
# =============================================================================
HAS_LIGHTNING_GPU = False
try:
    import pennylane as qml

    _test_dev = qml.device("lightning.gpu", wires=1)
    del _test_dev
    HAS_LIGHTNING_GPU = True
except Exception:
    pass

HAS_LIGHTNING_QUBIT = False
try:
    import pennylane as qml

    _test_dev = qml.device("lightning.qubit", wires=1)
    del _test_dev
    HAS_LIGHTNING_QUBIT = True
except Exception:
    pass

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

# Check Catalyst availability
HAS_CATALYST = False
CATALYST_VERSION = "N/A"
try:
    import catalyst

    HAS_CATALYST = True
    CATALYST_VERSION = catalyst.__version__
except ImportError:
    pass

# =============================================================================
# Constants
# =============================================================================
HARTREE_TO_KCAL_MOL = 627.5094
ANGSTROM_TO_BOHR = 1.8897259886
CHEMICAL_ACCURACY_KCAL = 0.1  # kcal/mol
QPE_ENERGY_TOLERANCE_HA = 2.0  # Hartree (POC validation threshold)

# Default QPE parameters
DEFAULT_N_ESTIMATION_WIRES = 4
DEFAULT_N_TROTTER_STEPS = 20
DEFAULT_N_SHOTS = 100

# TIP3P water model charges
TIP3P_OXYGEN_CHARGE = -0.834
TIP3P_HYDROGEN_CHARGE = 0.417


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class MolecularGeometry:
    """QM region molecular geometry."""

    symbols: list[str]
    coords: np.ndarray


@dataclass
class MMEnvironment:
    """MM region point charges."""

    charges: np.ndarray
    coords: np.ndarray


@dataclass
class QPEParams:
    """QPE algorithm parameters."""

    n_estimation_wires: int = DEFAULT_N_ESTIMATION_WIRES
    n_trotter_steps: int = DEFAULT_N_TROTTER_STEPS
    n_shots: int = DEFAULT_N_SHOTS


@dataclass
class ValidationResult:
    """Single validation check result."""

    name: str
    passed: bool
    detail: str


# =============================================================================
# Utility Functions
# =============================================================================
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


def compute_stabilization_kcal(e_vacuum: float, e_solvated: float) -> float:
    """Compute stabilization energy in kcal/mol."""
    return (e_vacuum - e_solvated) * HARTREE_TO_KCAL_MOL


def run_qpe_for_system(
    qpe_engine,
    hamiltonian,
    hf_state: np.ndarray,
    base_time: float,
    params: QPEParams,
) -> float:
    """Run QPE and extract energy for a single system."""
    circuit = qpe_engine._build_standard_qpe_circuit(
        hamiltonian,
        hf_state,
        n_estimation_wires=params.n_estimation_wires,
        base_time=base_time,
        n_trotter_steps=params.n_trotter_steps,
        n_shots=params.n_shots,
    )
    samples = circuit()
    return qpe_engine._extract_energy_from_samples(samples, base_time)


def run_catalyst_benchmark(
    qm_atoms: list,
    mm_waters: int,
    hf_results: dict[str, float],
) -> dict | None:
    """Run Catalyst @qjit QPE benchmark and compare with standard QPE.

    Returns:
        Catalyst benchmark data if Catalyst is available, None otherwise.
    """
    from q2m3.core import QuantumQMMM

    if not HAS_CATALYST:
        print("WARNING: pennylane-catalyst is not installed.")
        print("To enable Catalyst support, install with:")
        print("  pip install pennylane-catalyst")
        print()
        print("Skipping Catalyst benchmark...")
        return None

    print()
    print("=" * 70)
    print("Catalyst @qjit Benchmark: lightning.gpu vs qjit+lightning.gpu")
    print("=" * 70)
    print()

    # Step 1: Run standard QPE (baseline)
    print("[Step 1] Standard QPE (lightning.gpu)")
    print("-" * 70)
    device_type = get_best_available_device()
    print(f"Device: {device_type}")

    start_standard = time.perf_counter()
    qmmm_standard = QuantumQMMM(
        qm_atoms=qm_atoms,
        mm_waters=mm_waters,
        qpe_config={
            "use_real_qpe": True,
            "n_estimation_wires": DEFAULT_N_ESTIMATION_WIRES,
            "base_time": "auto",
            "n_trotter_steps": DEFAULT_N_TROTTER_STEPS,
            "n_shots": DEFAULT_N_SHOTS,
            "active_electrons": 2,
            "active_orbitals": 2,
            "device_type": device_type,
            "use_catalyst": False,
        },
    )
    result_standard = qmmm_standard.compute_ground_state()
    time_standard = time.perf_counter() - start_standard

    print(f"Energy: {result_standard['energy']:.6f} Ha")
    print(f"Time:   {time_standard:.3f} s")
    print()

    # Step 2: Run Catalyst QPE
    print("[Step 2] Catalyst @qjit QPE")
    print("-" * 70)
    catalyst_backend = get_catalyst_effective_backend()
    print(f"Effective Backend: {catalyst_backend}")
    print("  (Note: JAX backend is SEPARATE from PennyLane device)")

    start_catalyst = time.perf_counter()
    qmmm_catalyst = QuantumQMMM(
        qm_atoms=qm_atoms,
        mm_waters=mm_waters,
        qpe_config={
            "use_real_qpe": True,
            "n_estimation_wires": DEFAULT_N_ESTIMATION_WIRES,
            "base_time": "auto",
            "n_trotter_steps": DEFAULT_N_TROTTER_STEPS,
            "n_shots": DEFAULT_N_SHOTS,
            "active_electrons": 2,
            "active_orbitals": 2,
            "device_type": device_type,
            "use_catalyst": True,
        },
    )
    result_catalyst = qmmm_catalyst.compute_ground_state()
    time_catalyst = time.perf_counter() - start_catalyst

    print(f"Energy: {result_catalyst['energy']:.6f} Ha")
    print(f"Time:   {time_catalyst:.3f} s")
    print()

    # Step 3: Performance comparison
    print("[Step 3] Performance Comparison")
    print("-" * 70)
    print(f"Standard QPE ({device_type}):")
    print(f"  Time:  {time_standard:.3f} s")
    print()
    print(f"Catalyst QPE ({catalyst_backend}):")
    print(f"  Time:  {time_catalyst:.3f} s")
    print()

    if time_catalyst > 0:
        speedup = time_standard / time_catalyst
        if speedup > 1:
            print(f"Speedup: {speedup:.2f}x faster with Catalyst")
        else:
            print(f"Ratio: {1 / speedup:.2f}x slower with Catalyst")
            if not HAS_JAX_CUDA:
                print("Note: Catalyst is running on CPU (JAX lacks CUDA support)")
                if HAS_LIGHTNING_GPU:
                    print("      (Standard QPE uses GPU via lightning.gpu)")
    print()

    # Energy comparison
    e_diff = abs(result_standard["energy"] - result_catalyst["energy"])
    print("Energy Comparison:")
    print(f"  Standard QPE:  {result_standard['energy']:.6f} Ha")
    print(f"  Catalyst QPE:  {result_catalyst['energy']:.6f} Ha")
    print(f"  Difference:   {e_diff:.6f} Ha")

    if e_diff < 0.01:
        print("  Status: Results consistent (diff < 0.01 Ha)")
    else:
        print("  Status: Results differ (stochastic QPE sampling)")
    print()

    return {
        "device_type": device_type,
        "catalyst_backend": catalyst_backend,
        "has_lightning_gpu": HAS_LIGHTNING_GPU,
        "has_jax_cuda": HAS_JAX_CUDA,
        "time_standard": time_standard,
        "time_catalyst": time_catalyst,
        "energy_standard": result_standard["energy"],
        "energy_catalyst": result_catalyst["energy"],
        "speedup": time_standard / time_catalyst if time_catalyst > 0 else None,
    }


# =============================================================================
# Geometry and Environment Setup
# =============================================================================
def create_h2_geometry() -> MolecularGeometry:
    """Create H2 molecular geometry (bond length 0.74 Angstrom)."""
    return MolecularGeometry(
        symbols=["H", "H"],
        coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
    )


def create_tip3p_environment() -> MMEnvironment:
    """Create 2 TIP3P water molecules as MM environment (~3 Angstrom away)."""
    charges = np.array(
        [
            TIP3P_OXYGEN_CHARGE,
            TIP3P_HYDROGEN_CHARGE,
            TIP3P_HYDROGEN_CHARGE,  # Water 1
            TIP3P_OXYGEN_CHARGE,
            TIP3P_HYDROGEN_CHARGE,
            TIP3P_HYDROGEN_CHARGE,  # Water 2
        ]
    )
    coords = np.array(
        [
            [3.0, 0.0, 0.0],  # O1
            [3.5, 0.8, 0.0],  # H1a
            [3.5, -0.8, 0.0],  # H1b
            [-3.0, 0.0, 0.0],  # O2
            [-3.5, 0.8, 0.0],  # H2a
            [-3.5, -0.8, 0.0],  # H2b
        ]
    )
    return MMEnvironment(charges=charges, coords=coords)


# =============================================================================
# Core Computation Functions
# =============================================================================
def run_classical_hf(geometry: MolecularGeometry, mm_env: MMEnvironment) -> dict[str, float]:
    """Run classical Hartree-Fock calculations for vacuum and solvated systems."""
    from pyscf import gto, qmmm, scf

    print("[Step 3] Classical Hartree-Fock Reference")
    print("-" * 70)

    # Build atom string from geometry
    atom_str = "; ".join(
        f"{s} {c[0]} {c[1]} {c[2]}" for s, c in zip(geometry.symbols, geometry.coords, strict=True)
    )

    # Vacuum HF
    mol_vacuum = gto.M(atom=atom_str, basis="sto-3g", unit="Angstrom")
    mf_vacuum = scf.RHF(mol_vacuum)
    mf_vacuum.verbose = 0
    mf_vacuum.run()
    hf_vacuum = mf_vacuum.e_tot

    # Solvated HF (with MM embedding)
    mol_solvated = gto.M(atom=atom_str, basis="sto-3g", unit="Angstrom")
    mf_solvated = scf.RHF(mol_solvated)
    mf_solvated.verbose = 0
    mm_coords_bohr = mm_env.coords * ANGSTROM_TO_BOHR
    mf_solvated = qmmm.mm_charge(mf_solvated, mm_coords_bohr, mm_env.charges)
    mf_solvated.run()
    hf_solvated = mf_solvated.e_tot

    hf_stabilization = compute_stabilization_kcal(hf_vacuum, hf_solvated)

    print(f"  Vacuum HF:     {hf_vacuum:.8f} Ha")
    print(f"  Solvated HF:   {hf_solvated:.8f} Ha")
    print(f"  Stabilization: {hf_stabilization:.4f} kcal/mol")
    print()

    return {
        "vacuum": hf_vacuum,
        "solvated": hf_solvated,
        "stabilization": hf_stabilization,
    }


def run_resource_estimation(geometry: MolecularGeometry, mm_env: MMEnvironment) -> dict[str, dict]:
    """Run EFTQC resource estimation for vacuum and solvated Hamiltonians."""
    from q2m3.interfaces import PySCFPennyLaneConverter

    print("[Step 2] EFTQC Resource Estimation (Vacuum vs Solvated)")
    print("-" * 70)

    converter = PySCFPennyLaneConverter(basis="sto-3g", mapping="jordan_wigner")

    # Vacuum Hamiltonian resource estimation
    eftqc_vacuum = converter.estimate_qpe_resources(
        symbols=geometry.symbols,
        coords=geometry.coords,
        charge=0,
        target_error=0.0016,  # Chemical accuracy (1 kcal/mol)
    )

    # Solvated Hamiltonian resource estimation
    eftqc_solvated = converter.estimate_qpe_resources(
        symbols=geometry.symbols,
        coords=geometry.coords,
        charge=0,
        mm_charges=mm_env.charges,
        mm_coords=mm_env.coords,
        target_error=0.0016,
    )

    # Print results
    print("  Vacuum Hamiltonian (no MM):")
    print(f"    Hamiltonian 1-norm:  {eftqc_vacuum['hamiltonian_1norm']:.2f} Ha")
    print(f"    Logical Qubits:      {eftqc_vacuum['logical_qubits']}")
    print(f"    Toffoli Gates:       {eftqc_vacuum['toffoli_gates']:,}")
    print()

    delta_lambda = (
        (eftqc_solvated["hamiltonian_1norm"] - eftqc_vacuum["hamiltonian_1norm"])
        / eftqc_vacuum["hamiltonian_1norm"]
        * 100
    )
    delta_gates = (
        (eftqc_solvated["toffoli_gates"] - eftqc_vacuum["toffoli_gates"])
        / eftqc_vacuum["toffoli_gates"]
        * 100
    )

    print(f"  Solvated Hamiltonian (2 TIP3P waters, {eftqc_solvated['n_mm_charges']} MM charges):")
    print(
        f"    Hamiltonian 1-norm:  {eftqc_solvated['hamiltonian_1norm']:.2f} Ha (Δλ = {delta_lambda:+.1f}%)"
    )
    print(f"    Logical Qubits:      {eftqc_solvated['logical_qubits']}")
    print(
        f"    Toffoli Gates:       {eftqc_solvated['toffoli_gates']:,} (ΔG = {delta_gates:+.1f}%)"
    )
    print()
    print("  Note: MM embedding only modifies 1-electron integrals (QM-MM electrostatics).")
    print("        Resource estimates are dominated by 2-electron integrals, which remain")
    print("        unchanged. Thus, MM embedding has minimal impact on EFTQC resources.")
    print()

    return {"vacuum": eftqc_vacuum, "solvated": eftqc_solvated}


def build_hamiltonians(geometry: MolecularGeometry, mm_env: MMEnvironment) -> dict:
    """Build PennyLane Hamiltonians for vacuum and solvated systems."""
    from q2m3.interfaces import PySCFPennyLaneConverter

    print("[Step 1] System Setup & Hamiltonian Construction")
    print("-" * 70)

    converter = PySCFPennyLaneConverter(basis="sto-3g", mapping="jordan_wigner")

    # Vacuum Hamiltonian
    H_vacuum, n_qubits_v, hf_state_v = converter.pyscf_to_pennylane_hamiltonian(
        symbols=geometry.symbols,
        coords=geometry.coords,
        charge=0,
        active_electrons=2,
        active_orbitals=2,
    )

    # Solvated Hamiltonian
    H_solvated, n_qubits_s, hf_state_s = converter.pyscf_to_pennylane_hamiltonian_with_mm(
        symbols=geometry.symbols,
        coords=geometry.coords,
        charge=0,
        mm_charges=mm_env.charges,
        mm_coords=mm_env.coords,
        active_electrons=2,
        active_orbitals=2,
    )

    print(f"  Vacuum Hamiltonian:   {n_qubits_v} qubits, HF state = {hf_state_v}")
    print(f"  Solvated Hamiltonian: {n_qubits_s} qubits, HF state = {hf_state_s}")
    print()

    return {
        "H_vacuum": H_vacuum,
        "H_solvated": H_solvated,
        "n_qubits": n_qubits_v,
        "hf_state_vacuum": hf_state_v,
        "hf_state_solvated": hf_state_s,
    }


def verify_expectation_values(
    hamiltonians: dict, hf_results: dict[str, float], device_type: str
) -> dict[str, float]:
    """Verify Hamiltonians via HF expectation values."""
    import pennylane as qml

    print("[Step 4] Hamiltonian Expectation Value Verification")
    print("-" * 70)

    dev = qml.device(device_type, wires=hamiltonians["n_qubits"])

    @qml.qnode(dev)
    def compute_hf_expectation(hamiltonian, hf_state):
        qml.BasisState(hf_state, wires=range(len(hf_state)))
        return qml.expval(hamiltonian)

    exp_vacuum = float(
        compute_hf_expectation(hamiltonians["H_vacuum"], hamiltonians["hf_state_vacuum"])
    )
    exp_solvated = float(
        compute_hf_expectation(hamiltonians["H_solvated"], hamiltonians["hf_state_solvated"])
    )
    exp_stabilization = compute_stabilization_kcal(exp_vacuum, exp_solvated)

    print(f"  <HF|H_vacuum|HF>:   {exp_vacuum:.8f} Ha")
    print(f"  <HF|H_solvated|HF>: {exp_solvated:.8f} Ha")
    print(f"  Stabilization:      {exp_stabilization:.4f} kcal/mol")
    print()

    # Compare with PySCF
    diff_vacuum = abs(hf_results["vacuum"] - exp_vacuum) * HARTREE_TO_KCAL_MOL
    diff_solvated = abs(hf_results["solvated"] - exp_solvated) * HARTREE_TO_KCAL_MOL

    print("  Comparison with PySCF HF:")
    print(
        f"    Vacuum:   PySCF = {hf_results['vacuum']:.8f}, PL = {exp_vacuum:.8f}, "
        f"diff = {diff_vacuum:.4f} kcal/mol"
    )
    print(
        f"    Solvated: PySCF = {hf_results['solvated']:.8f}, PL = {exp_solvated:.8f}, "
        f"diff = {diff_solvated:.4f} kcal/mol"
    )
    print()

    return {
        "vacuum": exp_vacuum,
        "solvated": exp_solvated,
        "stabilization": exp_stabilization,
    }


def print_circuit_visualization(hamiltonians: dict, device_type: str) -> None:
    """Print QPE circuit visualization using QPEEngine.draw_qpe_circuit()."""
    from q2m3.core.qpe import QPEEngine

    print("[Step 1.5] Circuit Visualization")
    print("-" * 70)

    n_qubits = hamiltonians["n_qubits"]
    n_estimation_wires = DEFAULT_N_ESTIMATION_WIRES

    qpe_engine = QPEEngine(
        n_qubits=n_qubits,
        n_iterations=8,
        mapping="jordan_wigner",
        device_type=device_type,
        use_catalyst=False,
    )

    # Generate QPE circuit diagram using the dedicated method
    base_time = 0.1  # Just for visualization
    qpe_diagram = qpe_engine.draw_qpe_circuit(
        hamiltonian=hamiltonians["H_vacuum"],
        hf_state=hamiltonians["hf_state_vacuum"],
        n_estimation_wires=n_estimation_wires,
        base_time=base_time,
        n_trotter_steps=2,  # Simplified for visualization
    )

    print("QPE Circuit (Standard Phase Estimation):")
    print("-" * 60)
    print(qpe_diagram)
    print()


def run_qpe_estimation(
    hamiltonians: dict,
    hf_results: dict[str, float],
    device_type: str,
    params: QPEParams | None = None,
) -> dict[str, float]:
    """Run QPE energy estimation for vacuum and solvated systems."""
    from q2m3.core.qpe import QPEEngine

    print("[Step 5] QPE Energy Estimation")
    print("-" * 70)

    if params is None:
        params = QPEParams()

    qpe_engine = QPEEngine(
        n_qubits=hamiltonians["n_qubits"],
        n_iterations=8,
        mapping="jordan_wigner",
        device_type=device_type,
        use_catalyst=False,
    )

    base_time_vacuum = QPEEngine.compute_optimal_base_time(hf_results["vacuum"])
    base_time_solvated = QPEEngine.compute_optimal_base_time(hf_results["solvated"])

    print("  QPE Parameters:")
    print(f"    Estimation wires: {params.n_estimation_wires}")
    print(f"    Trotter steps:    {params.n_trotter_steps}")
    print(f"    Shots:            {params.n_shots}")
    print(f"    Base time (vac):  {base_time_vacuum:.6f}")
    print(f"    Base time (sol):  {base_time_solvated:.6f}")
    print()

    # Check phase estimates
    phase_vacuum = abs(hf_results["vacuum"]) * base_time_vacuum / (2 * np.pi)
    phase_solvated = abs(hf_results["solvated"]) * base_time_solvated / (2 * np.pi)
    print("  Phase estimates (should be < 1.0):")
    print(f"    Vacuum:   {phase_vacuum:.4f}")
    print(f"    Solvated: {phase_solvated:.4f}")
    print()

    # Run QPE for vacuum
    print("  Running vacuum QPE...")
    qpe_vacuum = run_qpe_for_system(
        qpe_engine,
        hamiltonians["H_vacuum"],
        hamiltonians["hf_state_vacuum"],
        base_time_vacuum,
        params,
    )

    # Run QPE for solvated
    print("  Running solvated QPE...")
    qpe_solvated = run_qpe_for_system(
        qpe_engine,
        hamiltonians["H_solvated"],
        hamiltonians["hf_state_solvated"],
        base_time_solvated,
        params,
    )

    qpe_stabilization = compute_stabilization_kcal(qpe_vacuum, qpe_solvated)

    print()
    print("  QPE Results:")
    print(f"    Vacuum QPE:     {qpe_vacuum:.8f} Ha")
    print(f"    Solvated QPE:   {qpe_solvated:.8f} Ha")
    print(f"    Stabilization:  {qpe_stabilization:.4f} kcal/mol")
    print()

    return {
        "vacuum": qpe_vacuum,
        "solvated": qpe_solvated,
        "stabilization": qpe_stabilization,
    }


def print_summary(
    hf_results: dict[str, float],
    exp_results: dict[str, float],
    qpe_results: dict[str, float],
) -> None:
    """Print energy comparison summary table."""
    print("  Energy Comparison (Ha):")
    print("  " + "-" * 50)
    print(f"  {'Method':<15} {'Vacuum':<15} {'Solvated':<15} {'Stab (kcal/mol)':<15}")
    print("  " + "-" * 50)
    print(
        f"  {'PySCF HF':<15} {hf_results['vacuum']:<15.6f} "
        f"{hf_results['solvated']:<15.6f} {hf_results['stabilization']:<15.4f}"
    )
    print(
        f"  {'PL <HF|H|HF>':<15} {exp_results['vacuum']:<15.6f} "
        f"{exp_results['solvated']:<15.6f} {exp_results['stabilization']:<15.4f}"
    )
    print(
        f"  {'QPE':<15} {qpe_results['vacuum']:<15.6f} "
        f"{qpe_results['solvated']:<15.6f} {qpe_results['stabilization']:<15.4f}"
    )
    print("  " + "-" * 50)
    print()


def run_validation_checks(
    hf_results: dict[str, float],
    exp_results: dict[str, float],
    qpe_results: dict[str, float],
) -> bool:
    """Run all validation checks and return overall pass/fail status."""
    print("  Validation Checks:")

    checks = [
        ValidationResult(
            name="PL vacuum matches PySCF",
            passed=abs(hf_results["vacuum"] - exp_results["vacuum"]) * HARTREE_TO_KCAL_MOL
            < CHEMICAL_ACCURACY_KCAL,
            detail=f"{abs(hf_results['vacuum'] - exp_results['vacuum']) * HARTREE_TO_KCAL_MOL:.4f} kcal/mol",
        ),
        ValidationResult(
            name="PL solvated matches PySCF",
            passed=abs(hf_results["solvated"] - exp_results["solvated"]) * HARTREE_TO_KCAL_MOL
            < CHEMICAL_ACCURACY_KCAL,
            detail=f"{abs(hf_results['solvated'] - exp_results['solvated']) * HARTREE_TO_KCAL_MOL:.4f} kcal/mol",
        ),
        ValidationResult(
            name="QPE vacuum vs HF",
            passed=abs(hf_results["vacuum"] - qpe_results["vacuum"]) < QPE_ENERGY_TOLERANCE_HA,
            detail=f"{abs(hf_results['vacuum'] - qpe_results['vacuum']):.4f} Ha",
        ),
        ValidationResult(
            name="QPE solvated vs HF",
            passed=abs(hf_results["solvated"] - qpe_results["solvated"]) < QPE_ENERGY_TOLERANCE_HA,
            detail=f"{abs(hf_results['solvated'] - qpe_results['solvated']):.4f} Ha",
        ),
        ValidationResult(
            name="Stabilization sign consistent",
            passed=(
                (hf_results["stabilization"] > 0) == (qpe_results["stabilization"] > 0)
                if abs(qpe_results["stabilization"]) > 0.01
                else True
            ),
            detail="",
        ),
    ]

    for check in checks:
        status = "OK" if check.passed else "FAIL"
        detail_str = f": {check.detail}" if check.detail else ""
        print(f"    [{status}] {check.name}{detail_str}")

    all_passed = all(c.passed for c in checks)
    print()
    if all_passed:
        print("  [SUCCESS] All validation checks passed!")
    else:
        print("  [WARNING] Some validation checks failed - further debugging needed")

    return all_passed


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    print("=" * 70)
    print("        H2 + MM Water: Minimal QPE Solvation Validation")
    print("=" * 70)
    print()

    # Device detection
    device_type = get_best_available_device()
    print(f"Device: {device_type}")
    if HAS_CATALYST:
        catalyst_backend = get_catalyst_effective_backend()
        print(f"Catalyst: Available (backend: {catalyst_backend})")
        if not HAS_JAX_CUDA and HAS_LIGHTNING_GPU:
            print("  Warning: Catalyst runs on CPU (JAX lacks CUDA)")
    else:
        print("Catalyst: Not available")
    print()

    # Setup molecular system
    geometry = create_h2_geometry()
    mm_env = create_tip3p_environment()

    # Step 1: Build Hamiltonians (before any calculations)
    hamiltonians = build_hamiltonians(geometry, mm_env)

    # Step 1.5: Circuit visualization (show circuit structure first)
    print_circuit_visualization(hamiltonians, device_type)

    # Step 2: Resource estimation
    run_resource_estimation(geometry, mm_env)

    # Step 3: Classical HF reference
    hf_results = run_classical_hf(geometry, mm_env)

    # Step 4: Verify expectation values
    exp_results = verify_expectation_values(hamiltonians, hf_results, device_type)

    # Step 5: QPE estimation
    qpe_results = run_qpe_estimation(hamiltonians, hf_results, device_type)

    # Step 5.5: Catalyst benchmark (optional)
    from q2m3.core.qmmm_system import Atom

    h2_atoms = [Atom("H", geometry.coords[0]), Atom("H", geometry.coords[1])]
    catalyst_results = run_catalyst_benchmark(h2_atoms, 2, hf_results)

    # Step 6: Summary and validation
    print("[Step 6] Summary and Validation")
    print("-" * 70)
    print_summary(hf_results, exp_results, qpe_results)
    run_validation_checks(hf_results, exp_results, qpe_results)

    # Add Catalyst performance summary if benchmark was run
    if catalyst_results is not None:
        print()
        print("Catalyst Performance Summary:")
        print(f"  Device: {catalyst_results['device_type']}")
        print(f"  Catalyst Backend: {catalyst_results['catalyst_backend']}")
        print(
            f"  Speedup: {catalyst_results['speedup']:.2f}x"
            if catalyst_results["speedup"]
            else "  Speedup: N/A"
        )
        print()

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
