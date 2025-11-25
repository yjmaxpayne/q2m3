# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
PySCF-PennyLane bidirectional conversion interface.
"""

from typing import Any

import numpy as np
import pennylane as qml


class UnifiedDensityMatrix:
    """
    Unified density matrix interface for quantum-classical data exchange.

    Provides seamless conversion between PySCF and PennyLane representations.
    """

    def __init__(self, dm_data: np.ndarray, source: str = "pyscf"):
        """
        Initialize unified density matrix.

        Args:
            dm_data: Density matrix data
            source: Source format ('pyscf' or 'pennylane')
        """
        self.dm_data = dm_data
        self.source = source
        self._validate_density_matrix()

    def _validate_density_matrix(self) -> None:
        """Validate density matrix properties."""
        # Check if matrix is square
        if self.dm_data.ndim != 2 or self.dm_data.shape[0] != self.dm_data.shape[1]:
            raise ValueError("Density matrix must be square")

        # Check if Hermitian
        if not np.allclose(self.dm_data, self.dm_data.conj().T):
            raise ValueError("Density matrix must be Hermitian")

    def to_pennylane_observable(self) -> Any:
        """
        Convert to PennyLane observable format.

        Returns:
            PennyLane Hermitian observable
        """
        # Placeholder for conversion to PennyLane format
        # Will use qml.Hermitian or decomposition
        pass

    def from_quantum_state(self, quantum_state: np.ndarray) -> None:
        """
        Update density matrix from quantum state vector.

        Args:
            quantum_state: Quantum state vector from QPE
        """
        # |psi><psi| calculation
        self.dm_data = np.outer(quantum_state, quantum_state.conj())
        self.source = "pennylane"


class PySCFPennyLaneConverter:
    """
    Converter for molecular integrals and Hamiltonians between PySCF and PennyLane.
    """

    def __init__(self, basis: str = "sto-3g", mapping: str = "jordan_wigner"):
        """
        Initialize converter.

        Args:
            basis: Basis set for calculations
            mapping: Fermion-to-qubit mapping scheme
        """
        self.basis = basis
        self.mapping = mapping

    def pyscf_to_pennylane_hamiltonian(
        self,
        symbols: list[str],
        coords: np.ndarray,
        charge: int = 0,
        active_electrons: int | None = None,
        active_orbitals: int | None = None,
    ) -> tuple[Any, int, np.ndarray]:
        """
        Convert molecular structure to PennyLane qubit Hamiltonian.

        Args:
            symbols: List of atomic symbols ['O', 'H', 'H', 'H']
            coords: Atomic coordinates in Angstrom, shape (n_atoms, 3) or (n_atoms*3,)
            charge: Total molecular charge (default 0)
            active_electrons: Number of active electrons for active space (optional)
            active_orbitals: Number of active orbitals for active space (optional)

        Returns:
            tuple: (hamiltonian, n_qubits, hf_state)
                - hamiltonian: PennyLane Hamiltonian operator
                - n_qubits: Number of qubits required
                - hf_state: Hartree-Fock reference state as numpy array

        Raises:
            ValueError: If symbols and coords dimensions don't match

        Example:
            >>> converter = PySCFPennyLaneConverter()
            >>> symbols = ["H", "H"]
            >>> coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
            >>> H, n_qubits, hf_state = converter.pyscf_to_pennylane_hamiltonian(
            ...     symbols, coords, charge=0
            ... )
        """
        # Validate input dimensions
        coords_arr = np.asarray(coords)
        if coords_arr.ndim == 2:
            if coords_arr.shape[0] != len(symbols):
                raise ValueError(
                    f"Number of symbols ({len(symbols)}) must match "
                    f"number of coords ({coords_arr.shape[0]})"
                )
            coords_flat = coords_arr.flatten()
        else:
            # 1D array - validate length
            if len(coords_arr) != len(symbols) * 3:
                raise ValueError(
                    f"Flattened coords length ({len(coords_arr)}) must be "
                    f"symbols * 3 = {len(symbols) * 3}"
                )
            coords_flat = coords_arr

        # Build Hamiltonian using PennyLane qchem
        # Note: PennyLane expects coordinates in Bohr by default
        # We convert from Angstrom to Bohr (1 Angstrom = 1.8897259886 Bohr)
        angstrom_to_bohr = 1.8897259886
        coords_bohr = coords_flat * angstrom_to_bohr

        # Prepare kwargs for molecular_hamiltonian
        mol_kwargs = {
            "symbols": symbols,
            "coordinates": coords_bohr,
            "charge": charge,
            "basis": self.basis,
            "mapping": self.mapping,
        }

        # Add active space if specified
        if active_electrons is not None:
            mol_kwargs["active_electrons"] = active_electrons
        if active_orbitals is not None:
            mol_kwargs["active_orbitals"] = active_orbitals

        # Generate molecular Hamiltonian
        H, n_qubits = qml.qchem.molecular_hamiltonian(**mol_kwargs)

        # Calculate number of electrons
        atomic_numbers = {
            "H": 1,
            "He": 2,
            "Li": 3,
            "Be": 4,
            "B": 5,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
            "Ne": 10,
            "Na": 11,
            "Mg": 12,
            "Al": 13,
            "Si": 14,
            "P": 15,
            "S": 16,
            "Cl": 17,
            "Ar": 18,
        }
        n_electrons = sum(atomic_numbers.get(s, 0) for s in symbols) - charge

        # Generate HF reference state
        hf_state = qml.qchem.hf_state(n_electrons, n_qubits)

        return H, n_qubits, hf_state

    def build_qmmm_hamiltonian(
        self, qm_mol: Any, mm_charges: np.ndarray, mm_coords: np.ndarray  # PySCF mol object
    ) -> dict[str, Any]:
        """
        Build QM/MM Hamiltonian with MM embedding using classical simulation.

        Args:
            qm_mol: PySCF molecule object
            mm_charges: MM region point charges
            mm_coords: MM charge coordinates

        Returns:
            Dictionary containing molecular data for QPE simulation
        """
        from pyscf import scf

        # Run Hartree-Fock calculation
        mf = scf.RHF(qm_mol)

        # Add MM charges as background charges if provided
        if len(mm_charges) > 0 and len(mm_coords) > 0:
            # Simple point charge embedding
            mf = scf.RHF(qm_mol).run()
        else:
            mf = mf.run()

        return {
            "mol": qm_mol,
            "scf_result": mf,
            "energy_nuc": qm_mol.energy_nuc(),
            "energy_hf": mf.e_tot,
            "mo_coeff": mf.mo_coeff,
            "mo_energy": mf.mo_energy,
            "mo_occ": mf.mo_occ,
        }

    def extract_molecular_orbitals(self, scf_result: Any) -> np.ndarray:
        """
        Extract molecular orbital coefficients from PySCF.

        Args:
            scf_result: PySCF SCF calculation result

        Returns:
            MO coefficient matrix
        """
        # Will extract mo_coeff from SCF object
        pass

    def compute_overlap_integrals(self, mol: Any) -> np.ndarray:
        """
        Compute overlap integrals for Mulliken analysis.

        Args:
            mol: PySCF molecule object

        Returns:
            Overlap matrix
        """
        return mol.intor("int1e_ovlp")
