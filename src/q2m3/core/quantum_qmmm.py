# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Main Quantum-QM/MM interface combining all components.
"""

from typing import Any

import numpy as np
from pyscf import gto

from ..interfaces import PySCFPennyLaneConverter
from .qmmm_system import Atom, QMMMSystem
from .qpe import QPEEngine


class QuantumQMMM:
    """
    Main interface for Quantum-QM/MM calculations.

    Coordinates the workflow between PySCF classical calculations,
    PennyLane quantum circuits, and QM/MM embedding.
    """

    def __init__(self, qm_atoms: list[Atom], mm_waters: int = 8, qpe_config: dict[str, Any] = None):
        """
        Initialize Quantum-QM/MM calculator.

        Args:
            qm_atoms: Atoms in the QM region
            mm_waters: Number of water molecules in MM region
            qpe_config: Configuration for QPE algorithm
        """
        # Set default QPE configuration
        self.qpe_config = qpe_config or {
            "algorithm": "iterative",
            "iterations": 8,
            "mapping": "jordan_wigner",
            "system_qubits": 12,
            "error_tolerance": 0.005,
        }

        # Initialize components
        self.qmmm_system = QMMMSystem(qm_atoms=qm_atoms, num_waters=mm_waters)

        self.qpe_engine = QPEEngine(
            n_qubits=self.qpe_config["system_qubits"],
            n_iterations=self.qpe_config["iterations"],
            mapping=self.qpe_config["mapping"],
        )

    def compute_ground_state(self) -> dict[str, Any]:
        """
        Compute ground state properties using QPE with classical simulation.

        Returns:
            Dictionary containing:
                - energy: Ground state energy (Hartree)
                - density_matrix: Electronic density matrix
                - atomic_charges: Mulliken charges
                - convergence: Convergence information
        """
        # Build QM/MM Hamiltonian
        hamiltonian_data = self._build_qmmm_hamiltonian()

        # Run QPE to estimate ground state energy
        qpe_result = self.qpe_engine.estimate_ground_state_energy(hamiltonian_data)

        # Extract density matrix
        density_matrix = qpe_result["density_matrix"]

        # Perform Mulliken charge analysis
        atomic_charges = self._mulliken_analysis(density_matrix, hamiltonian_data["mol"])

        return {
            "energy": qpe_result["energy"],
            "density_matrix": density_matrix,
            "atomic_charges": atomic_charges,
            "convergence": qpe_result["convergence"],
        }

    def _build_qmmm_hamiltonian(self) -> dict[str, Any]:
        """
        Build QM/MM Hamiltonian with embedding.

        Returns:
            Dictionary containing molecular data from PySCF
        """
        # Convert QM/MM system to PySCF molecule
        mol_dict = self.qmmm_system.to_pyscf_mol()
        mol = gto.M(**mol_dict)

        # Get MM embedding potential
        mm_charges, mm_coords = self.qmmm_system.get_embedding_potential()

        # Build Hamiltonian using converter
        converter = PySCFPennyLaneConverter()
        return converter.build_qmmm_hamiltonian(mol, mm_charges, mm_coords)

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
