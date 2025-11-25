# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Main Quantum-QM/MM interface combining all components.
"""

import logging
from typing import Any

import numpy as np
from pyscf import gto

from ..interfaces import PySCFPennyLaneConverter
from .qmmm_system import Atom, QMMMSystem
from .qpe import QPEEngine

logger = logging.getLogger(__name__)


class QuantumQMMM:
    """
    Main interface for Quantum-QM/MM calculations.

    Coordinates the workflow between PySCF classical calculations,
    PennyLane quantum circuits, and QM/MM embedding.
    """

    def __init__(
        self,
        qm_atoms: list[Atom],
        mm_waters: int = 8,
        qpe_config: dict[str, Any] = None,
        use_catalyst: bool = False,
    ):
        """
        Initialize Quantum-QM/MM calculator.

        Args:
            qm_atoms: Atoms in the QM region
            mm_waters: Number of water molecules in MM region
            qpe_config: Configuration for QPE algorithm
            use_catalyst: Enable Catalyst @qjit compilation for QPE circuits
        """
        # Set default QPE configuration
        default_config = {
            "algorithm": "standard",
            "iterations": 8,
            "mapping": "jordan_wigner",
            "system_qubits": 12,
            "error_tolerance": 0.005,
            # QPE parameters
            "use_real_qpe": True,  # Default to real QPE
            "n_estimation_wires": 4,
            "base_time": 0.1,
            "n_trotter_steps": 10,
            "n_shots": 100,
            # Active space (optional)
            "active_electrons": None,
            "active_orbitals": None,
            # Energy validation threshold
            "energy_warning_threshold": 1.0,  # Warn if |E_QPE - E_HF| > 1.0 Ha
        }
        # Merge user config with defaults
        if qpe_config:
            default_config.update(qpe_config)
        self.qpe_config = default_config
        self.use_catalyst = use_catalyst

        # Initialize components
        self.qmmm_system = QMMMSystem(qm_atoms=qm_atoms, num_waters=mm_waters)
        self.converter = PySCFPennyLaneConverter()

        self.qpe_engine = QPEEngine(
            n_qubits=self.qpe_config["system_qubits"],
            n_iterations=self.qpe_config["iterations"],
            mapping=self.qpe_config["mapping"],
            use_catalyst=use_catalyst,
        )

    def compute_ground_state(self) -> dict[str, Any]:
        """
        Compute ground state properties using QPE.

        Returns:
            Dictionary containing:
                - energy: Ground state energy (Hartree)
                - energy_hf: HF reference energy (Hartree)
                - energy_difference: |E_QPE - E_HF| (Hartree)
                - density_matrix: Electronic density matrix
                - atomic_charges: Mulliken charges
                - convergence: Convergence information
        """
        # Build QM/MM Hamiltonian (includes PennyLane Hamiltonian)
        hamiltonian_data = self._build_qmmm_hamiltonian()

        # Select QPE mode based on configuration
        use_real_qpe = self.qpe_config.get("use_real_qpe", True)

        if use_real_qpe and "pennylane_hamiltonian" in hamiltonian_data:
            qpe_result = self._run_real_qpe(hamiltonian_data)
        else:
            # Fallback to classical HF simulation
            qpe_result = self.qpe_engine.estimate_ground_state_energy(hamiltonian_data)

        # Extract density matrix
        density_matrix = qpe_result["density_matrix"]

        # Perform Mulliken charge analysis
        atomic_charges = self._mulliken_analysis(density_matrix, hamiltonian_data["mol"])

        # Build result dictionary
        result = {
            "energy": qpe_result["energy"],
            "density_matrix": density_matrix,
            "atomic_charges": atomic_charges,
            "convergence": qpe_result["convergence"],
        }

        # Add QPE-specific fields if available
        if "energy_hf" in qpe_result:
            result["energy_hf"] = qpe_result["energy_hf"]
        if "energy_difference" in qpe_result:
            result["energy_difference"] = qpe_result["energy_difference"]

        return result

    def _build_qmmm_hamiltonian(self) -> dict[str, Any]:
        """
        Build QM/MM Hamiltonian with embedding.

        Returns:
            Dictionary containing molecular data from PySCF and PennyLane Hamiltonian
        """
        # Convert QM/MM system to PySCF molecule
        mol_dict = self.qmmm_system.to_pyscf_mol()
        mol = gto.M(**mol_dict)

        # Get MM embedding potential
        mm_charges, mm_coords = self.qmmm_system.get_embedding_potential()

        # Build PySCF Hamiltonian data
        result = self.converter.build_qmmm_hamiltonian(mol, mm_charges, mm_coords)

        # Build PennyLane Hamiltonian for QPE
        symbols = [atom.symbol for atom in self.qmmm_system.qm_atoms]
        coords = self.qmmm_system.get_qm_coords()
        charge = int(self.qmmm_system.get_total_charge())

        # Get active space parameters from config
        active_electrons = self.qpe_config.get("active_electrons")
        active_orbitals = self.qpe_config.get("active_orbitals")

        pl_hamiltonian, n_qubits, hf_state = self.converter.pyscf_to_pennylane_hamiltonian(
            symbols,
            coords,
            charge,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
        )

        # Add PennyLane data to result
        result["pennylane_hamiltonian"] = pl_hamiltonian
        result["n_qubits"] = n_qubits
        result["hf_state"] = hf_state

        return result

    def _run_real_qpe(self, hamiltonian_data: dict[str, Any]) -> dict[str, Any]:
        """
        Execute real QPE circuit for energy estimation.

        Args:
            hamiltonian_data: Dictionary containing PennyLane Hamiltonian and HF state

        Returns:
            Dictionary containing QPE results
        """
        H = hamiltonian_data["pennylane_hamiltonian"]
        hf_state = hamiltonian_data["hf_state"]
        n_qubits = hamiltonian_data["n_qubits"]

        # Update QPE engine's qubit count
        self.qpe_engine.n_qubits = n_qubits

        # Get QPE parameters from config
        n_estimation_wires = self.qpe_config.get("n_estimation_wires", 4)
        base_time = self.qpe_config.get("base_time", 0.1)
        n_trotter_steps = self.qpe_config.get("n_trotter_steps", 10)
        n_shots = self.qpe_config.get("n_shots", 100)

        # Build and execute QPE circuit
        qpe_circuit = self.qpe_engine._build_standard_qpe_circuit(
            H,
            hf_state,
            n_estimation_wires=n_estimation_wires,
            base_time=base_time,
            n_trotter_steps=n_trotter_steps,
            n_shots=n_shots,
        )

        samples = qpe_circuit()
        energy = self.qpe_engine._extract_energy_from_samples(samples, base_time)

        # HF energy as reference
        energy_hf = hamiltonian_data.get("energy_hf", 0.0)

        # Energy validation warning
        energy_diff = abs(energy - energy_hf)
        threshold = self.qpe_config.get("energy_warning_threshold", 1.0)
        if energy_diff > threshold:
            logger.warning(
                f"QPE energy ({energy:.4f} Ha) differs from HF ({energy_hf:.4f} Ha) "
                f"by {energy_diff:.4f} Ha, exceeding threshold {threshold} Ha"
            )

        return {
            "energy": energy,
            "energy_hf": energy_hf,
            "energy_difference": energy_diff,
            "density_matrix": hamiltonian_data["scf_result"].make_rdm1(),
            "convergence": {
                "converged": True,
                "method": "real_qpe",
                "n_estimation_wires": n_estimation_wires,
                "n_shots": n_shots,
            },
        }

    def _mulliken_analysis(self, density_matrix: np.ndarray, mol: Any) -> dict[str, float]:
        """
        Perform Mulliken population analysis.

        Args:
            density_matrix: Electronic density matrix
            mol: PySCF molecule object

        Returns:
            Dictionary mapping atom labels to Mulliken charges
        """
        overlap = mol.intor("int1e_ovlp")
        pop_matrix = self._compute_population_matrix(density_matrix, overlap)

        return self._extract_atomic_charges(pop_matrix, mol)

    def _compute_population_matrix(
        self, density_matrix: np.ndarray, overlap: np.ndarray
    ) -> np.ndarray:
        """
        Compute Mulliken population matrix.

        Args:
            density_matrix: Electronic density matrix
            overlap: Overlap matrix

        Returns:
            Population matrix P * S
        """
        return np.dot(density_matrix, overlap)

    def _extract_atomic_charges(self, pop_matrix: np.ndarray, mol: Any) -> dict[str, float]:
        """
        Extract atomic charges from population matrix.

        Args:
            pop_matrix: Mulliken population matrix
            mol: PySCF molecule object

        Returns:
            Dictionary of atomic charges
        """
        atomic_charges = {}
        ao_offset = 0

        for i, atom in enumerate(mol._atom):
            symbol = atom[0]
            nbas = mol.nao_nr(atom[0])

            # Calculate charge for this atom
            electron_pop = np.trace(pop_matrix[ao_offset : ao_offset + nbas, :])
            nuclear_charge = gto.charge(symbol)
            charge = nuclear_charge - electron_pop

            atomic_charges[f"{symbol}{i}"] = float(charge)
            ao_offset += nbas

        return atomic_charges
