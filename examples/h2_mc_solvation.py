#!/usr/bin/env python3
# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
H2 + TIP3P Water: Monte Carlo Solvation Structure Optimization

This example demonstrates Monte Carlo sampling of water configurations around
a fixed H2 solute molecule, with the goal of finding optimal solvation structures.

Workflow:
1. Initialize H2 solute + 2 TIP3P waters in random positions
2. Run MC sampling with HF energy evaluation (fast, for screening)
3. Validate best configuration with QPE energy (quantum verification)

Energy function:
    E_total = E_QM(HF with MM embedding) + E_MM-MM(LJ + Coulomb)

Key features:
- Solute (H2) geometry is FIXED
- Solvent (water) positions and orientations are sampled
- HF is used for fast MC screening
- QPE is used for final quantum verification
"""

import time
import warnings
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore")

from pyscf import gto, qmmm, scf  # noqa: E402

from q2m3.sampling import (  # noqa: E402
    MetropolisSampler,
    TIP3PForceField,
    WaterMolecule,
)

# =============================================================================
# Constants
# =============================================================================
HARTREE_TO_KCAL_MOL = 627.5094
ANGSTROM_TO_BOHR = 1.8897259886

# H2 geometry (fixed solute)
H2_BOND_LENGTH = 0.74  # Angstrom
H2_SYMBOLS = ["H", "H"]
H2_COORDS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, H2_BOND_LENGTH]])

# MC parameters
DEFAULT_N_MC_STEPS = 10
DEFAULT_TEMPERATURE = 300.0  # Kelvin
DEFAULT_TRANSLATION_STEP = 0.3  # Angstrom
DEFAULT_ROTATION_STEP = np.radians(15.0)  # radians

# Solvation constraints
MIN_SOLUTE_SOLVENT_DISTANCE = 2.5  # Angstrom
MAX_SOLUTE_SOLVENT_DISTANCE = 6.0  # Angstrom
INITIAL_WATER_DISTANCE = 4.0  # Angstrom


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class SolvationConfig:
    """Configuration for solvation optimization."""

    n_waters: int = 2
    n_mc_steps: int = DEFAULT_N_MC_STEPS
    temperature: float = DEFAULT_TEMPERATURE
    translation_step: float = DEFAULT_TRANSLATION_STEP
    rotation_step: float = DEFAULT_ROTATION_STEP
    use_qpe_validation: bool = True
    random_seed: int | None = 42


@dataclass
class SolvationResult:
    """Result of solvation optimization."""

    initial_energy: float
    best_energy: float
    best_waters: list[WaterMolecule]
    mc_energies: list[float]
    acceptance_rate: float
    qpe_energy: float | None = None


# =============================================================================
# Energy Evaluation Functions
# =============================================================================
def compute_hf_energy_with_mm(
    qm_symbols: list[str],
    qm_coords: np.ndarray,
    waters: list[WaterMolecule],
) -> float:
    """
    Compute HF energy of QM region with MM embedding from waters.

    Args:
        qm_symbols: QM atom symbols (e.g., ["H", "H"])
        qm_coords: QM atom coordinates (Angstrom), shape (n_atoms, 3)
        waters: List of WaterMolecule objects for MM region

    Returns:
        Total HF energy in Hartree (includes QM-MM electrostatics)
    """
    # Build PySCF molecule
    atom_str = "; ".join(
        f"{s} {c[0]} {c[1]} {c[2]}" for s, c in zip(qm_symbols, qm_coords, strict=True)
    )
    mol = gto.M(atom=atom_str, basis="sto-3g", unit="Angstrom")

    # Setup HF with MM embedding
    mf = scf.RHF(mol)
    mf.verbose = 0

    if waters:
        # Collect MM charges and coordinates
        mm_charges = []
        mm_coords = []
        for water in waters:
            coords = water.get_atom_coords()
            charges = water.get_charges()
            for i in range(3):
                mm_charges.append(charges[i])
                mm_coords.append(coords[i])

        mm_charges = np.array(mm_charges)
        mm_coords = np.array(mm_coords) * ANGSTROM_TO_BOHR  # Convert to Bohr

        mf = qmmm.mm_charge(mf, mm_coords, mm_charges)

    mf.run()
    return mf.e_tot


def compute_total_energy(
    qm_symbols: list[str],
    qm_coords: np.ndarray,
    waters: list[WaterMolecule],
    forcefield: TIP3PForceField,
) -> float:
    """
    Compute total energy: E_QM(HF) + E_MM-MM.

    Args:
        qm_symbols: QM atom symbols
        qm_coords: QM atom coordinates (Angstrom)
        waters: List of WaterMolecule objects
        forcefield: TIP3P force field for MM-MM interactions

    Returns:
        Total energy in Hartree
    """
    e_qm = compute_hf_energy_with_mm(qm_symbols, qm_coords, waters)
    e_mm = forcefield.compute_mm_energy(waters)
    return e_qm + e_mm


def check_distance_constraints(
    qm_coords: np.ndarray,
    waters: list[WaterMolecule],
    min_dist: float = MIN_SOLUTE_SOLVENT_DISTANCE,
    max_dist: float = MAX_SOLUTE_SOLVENT_DISTANCE,
) -> bool:
    """
    Check if water molecules satisfy distance constraints.

    Args:
        qm_coords: QM atom coordinates (Angstrom)
        waters: List of WaterMolecule objects
        min_dist: Minimum allowed QM-MM distance
        max_dist: Maximum allowed QM-MM distance

    Returns:
        True if all constraints are satisfied
    """
    qm_center = np.mean(qm_coords, axis=0)

    for water in waters:
        o_pos = water.oxygen_position
        dist = np.linalg.norm(o_pos - qm_center)

        if dist < min_dist or dist > max_dist:
            return False

    # Check water-water distances
    for i, w1 in enumerate(waters):
        for j, w2 in enumerate(waters):
            if i < j:
                dist = np.linalg.norm(w1.oxygen_position - w2.oxygen_position)
                if dist < min_dist:
                    return False

    return True


# =============================================================================
# Initialization Functions
# =============================================================================
def initialize_waters(
    n_waters: int,
    qm_center: np.ndarray,
    initial_distance: float = INITIAL_WATER_DISTANCE,
) -> list[WaterMolecule]:
    """
    Initialize water molecules in a ring around QM center.

    Args:
        n_waters: Number of water molecules
        qm_center: Center of QM region
        initial_distance: Distance from center for oxygen atoms

    Returns:
        List of WaterMolecule objects
    """
    waters = []

    for i in range(n_waters):
        # Place waters evenly around the z-axis
        angle = 2 * np.pi * i / n_waters
        x = qm_center[0] + initial_distance * np.cos(angle)
        y = qm_center[1] + initial_distance * np.sin(angle)
        z = qm_center[2]

        position = np.array([x, y, z])

        # Random initial orientation
        euler_angles = np.random.uniform(-np.pi, np.pi, size=3)

        water = WaterMolecule(position=position, euler_angles=euler_angles)
        waters.append(water)

    return waters


# =============================================================================
# Main MC Optimization
# =============================================================================
def run_mc_solvation_optimization(config: SolvationConfig) -> SolvationResult:
    """
    Run Monte Carlo optimization of solvation structure.

    Args:
        config: Solvation configuration

    Returns:
        SolvationResult with optimized configuration
    """
    if config.random_seed is not None:
        np.random.seed(config.random_seed)

    print("=" * 70)
    print("      H2 + TIP3P Water: MC Solvation Structure Optimization")
    print("=" * 70)
    print()

    # Initialize system
    qm_center = np.mean(H2_COORDS, axis=0)
    waters = initialize_waters(config.n_waters, qm_center)
    forcefield = TIP3PForceField()

    print("[Step 1] System Initialization")
    print("-" * 70)
    print("  Solute: H2 (fixed geometry)")
    print(f"  Solvent: {config.n_waters} TIP3P water molecules")
    print("  Initial water positions:")
    for i, water in enumerate(waters):
        print(f"    Water {i + 1}: O at {water.oxygen_position}")
    print()

    # Define energy function for MC
    def energy_fn(ws: list[WaterMolecule]) -> float:
        return compute_total_energy(H2_SYMBOLS, H2_COORDS, ws, forcefield)

    # Compute initial energy
    initial_energy = energy_fn(waters)

    print("[Step 2] Initial Energy Evaluation")
    print("-" * 70)
    e_qm = compute_hf_energy_with_mm(H2_SYMBOLS, H2_COORDS, waters)
    e_mm = forcefield.compute_mm_energy(waters)
    print(f"  E_QM (HF with MM embedding): {e_qm:.6f} Ha")
    print(f"  E_MM-MM (LJ + Coulomb):      {e_mm:.6f} Ha")
    print(f"  E_total:                     {initial_energy:.6f} Ha")
    print()

    # Run MC sampling
    print("[Step 3] Monte Carlo Sampling")
    print("-" * 70)
    print(f"  MC steps:          {config.n_mc_steps}")
    print(f"  Temperature:       {config.temperature} K")
    print(f"  Translation step:  {config.translation_step:.2f} Angstrom")
    print(f"  Rotation step:     {np.degrees(config.rotation_step):.1f} degrees")
    print()

    sampler = MetropolisSampler(
        waters=waters,
        energy_function=energy_fn,
        temperature=config.temperature,
        translation_step=config.translation_step,
        rotation_step=config.rotation_step,
    )

    start_time = time.perf_counter()
    mc_result = sampler.run(n_steps=config.n_mc_steps)
    mc_time = time.perf_counter() - start_time

    print(f"  MC sampling completed in {mc_time:.2f} s")
    print(f"  Acceptance rate: {mc_result['acceptance_rate'] * 100:.1f}%")
    print()

    # Report best configuration
    print("[Step 4] Best Configuration Found")
    print("-" * 70)
    best_waters = mc_result["best_config"]
    best_energy = mc_result["best_energy"]

    e_qm_best = compute_hf_energy_with_mm(H2_SYMBOLS, H2_COORDS, best_waters)
    e_mm_best = forcefield.compute_mm_energy(best_waters)

    print("  Best water positions:")
    for i, water in enumerate(best_waters):
        dist = np.linalg.norm(water.oxygen_position - qm_center)
        print(f"    Water {i + 1}: O at {water.oxygen_position}, dist = {dist:.2f} A")
    print()
    print(f"  E_QM (HF with MM embedding): {e_qm_best:.6f} Ha")
    print(f"  E_MM-MM (LJ + Coulomb):      {e_mm_best:.6f} Ha")
    print(f"  E_total (best):              {best_energy:.6f} Ha")
    print()

    energy_change = (best_energy - initial_energy) * HARTREE_TO_KCAL_MOL
    print(f"  Energy change: {energy_change:+.4f} kcal/mol")
    if energy_change < 0:
        print("  [SUCCESS] Found lower energy configuration!")
    else:
        print("  [INFO] No lower energy found (may need more MC steps)")
    print()

    # QPE validation (optional)
    qpe_energy = None
    if config.use_qpe_validation:
        print("[Step 5] QPE Validation of Best Configuration")
        print("-" * 70)
        try:
            qpe_energy = run_qpe_validation(best_waters)
            print(f"  QPE energy: {qpe_energy:.6f} Ha")
            qpe_diff = abs(e_qm_best - qpe_energy) * HARTREE_TO_KCAL_MOL
            print(f"  HF vs QPE difference: {qpe_diff:.4f} kcal/mol")
        except Exception as e:
            print(f"  QPE validation skipped: {e}")
        print()

    # Summary
    print("[Step 6] Summary")
    print("-" * 70)
    print(f"  Initial energy:  {initial_energy:.6f} Ha")
    print(f"  Best energy:     {best_energy:.6f} Ha")
    print(f"  Energy change:   {energy_change:+.4f} kcal/mol")
    print(f"  Acceptance rate: {mc_result['acceptance_rate'] * 100:.1f}%")
    if qpe_energy is not None:
        print(f"  QPE energy:      {qpe_energy:.6f} Ha")
    print()
    print("=" * 70)

    return SolvationResult(
        initial_energy=initial_energy,
        best_energy=best_energy,
        best_waters=best_waters,
        mc_energies=mc_result["energies"],
        acceptance_rate=mc_result["acceptance_rate"],
        qpe_energy=qpe_energy,
    )


def run_qpe_validation(waters: list[WaterMolecule]) -> float:
    """
    Run QPE energy estimation for validation.

    Args:
        waters: Water configuration to validate

    Returns:
        QPE energy in Hartree
    """
    from q2m3.core import QPEEngine
    from q2m3.interfaces import PySCFPennyLaneConverter

    # Collect MM charges and coordinates
    mm_charges = []
    mm_coords = []
    for water in waters:
        coords = water.get_atom_coords()
        charges = water.get_charges()
        for i in range(3):
            mm_charges.append(charges[i])
            mm_coords.append(coords[i])

    mm_charges = np.array(mm_charges)
    mm_coords = np.array(mm_coords)

    # Build Hamiltonian with MM embedding
    converter = PySCFPennyLaneConverter(basis="sto-3g", mapping="jordan_wigner")
    H, n_qubits, hf_state = converter.pyscf_to_pennylane_hamiltonian_with_mm(
        symbols=H2_SYMBOLS,
        coords=H2_COORDS,
        charge=0,
        mm_charges=mm_charges,
        mm_coords=mm_coords,
        active_electrons=2,
        active_orbitals=2,
    )

    # Compute HF energy for base time estimation
    e_hf = compute_hf_energy_with_mm(H2_SYMBOLS, H2_COORDS, waters)

    # Run QPE
    qpe_engine = QPEEngine(
        n_qubits=n_qubits,
        n_iterations=8,
        mapping="jordan_wigner",
    )

    circuit = qpe_engine._build_standard_qpe_circuit(
        H,
        hf_state,
        n_estimation_wires=4,
        base_time=QPEEngine.compute_optimal_base_time(e_hf),
        n_trotter_steps=20,
        n_shots=100,
    )
    samples = circuit()
    qpe_energy = qpe_engine._extract_energy_from_samples(
        samples, QPEEngine.compute_optimal_base_time(e_hf)
    )

    return qpe_energy


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Run the MC solvation optimization example."""
    config = SolvationConfig(
        n_waters=2,
        n_mc_steps=10,
        temperature=300.0,
        use_qpe_validation=True,
        random_seed=42,
    )

    result = run_mc_solvation_optimization(config)

    return result


if __name__ == "__main__":
    main()
