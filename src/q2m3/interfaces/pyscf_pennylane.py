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
        total_electrons = sum(atomic_numbers.get(s, 0) for s in symbols) - charge

        # For active space, use active_electrons for HF state
        # Otherwise use total electrons
        if active_electrons is not None:
            n_electrons_for_hf = active_electrons
        else:
            n_electrons_for_hf = total_electrons

        # Generate HF reference state
        hf_state = qml.qchem.hf_state(n_electrons_for_hf, n_qubits)

        return H, n_qubits, hf_state

    def pyscf_to_pennylane_hamiltonian_with_mm(
        self,
        symbols: list[str],
        coords: np.ndarray,
        charge: int = 0,
        mm_charges: np.ndarray | None = None,
        mm_coords: np.ndarray | None = None,
        active_electrons: int | None = None,
        active_orbitals: int | None = None,
    ) -> tuple[Any, int, np.ndarray]:
        """
        Build PennyLane Hamiltonian with MM point charge embedding.

        Uses a perturbative approach:
        1. Get vacuum Hamiltonian from molecular_hamiltonian (correct eigenspectrum)
        2. Compute MM corrections to single-electron integrals
        3. Add MM correction as Pauli operators to the vacuum Hamiltonian

        This approach preserves the correct eigenspectrum structure from
        molecular_hamiltonian while adding MM electrostatic effects.

        Args:
            symbols: List of atomic symbols ['O', 'H', 'H', 'H']
            coords: Atomic coordinates in Angstrom, shape (n_atoms, 3)
            charge: Total molecular charge
            mm_charges: MM point charges array
            mm_coords: MM charge coordinates in Angstrom, shape (n_mm, 3)
            active_electrons: Number of active electrons
            active_orbitals: Number of active orbitals

        Returns:
            tuple: (hamiltonian, n_qubits, hf_state)
        """
        from pyscf import gto, qmmm, scf

        ANGSTROM_TO_BOHR = 1.8897259886

        # Prepare coordinates in Bohr for PennyLane
        coords_arr = np.asarray(coords).reshape(-1, 3)
        coords_bohr = (coords_arr * ANGSTROM_TO_BOHR).flatten()

        # Step 1: Get vacuum Hamiltonian from molecular_hamiltonian
        # This ensures correct eigenspectrum structure
        mol_kwargs = {
            "symbols": symbols,
            "coordinates": coords_bohr,
            "charge": charge,
            "basis": self.basis,
            "mapping": self.mapping,
        }
        if active_electrons is not None:
            mol_kwargs["active_electrons"] = active_electrons
        if active_orbitals is not None:
            mol_kwargs["active_orbitals"] = active_orbitals

        H_vacuum, n_qubits = qml.qchem.molecular_hamiltonian(**mol_kwargs)

        # If no MM charges, return vacuum Hamiltonian
        if mm_charges is None or len(mm_charges) == 0:
            hf_state = qml.qchem.hf_state(
                (
                    active_electrons
                    if active_electrons
                    else sum(self._get_atomic_numbers(symbols)) - charge
                ),
                n_qubits,
            )
            return H_vacuum, n_qubits, hf_state

        # Step 2: Compute MM corrections using PySCF
        atom_str = "\n".join(
            f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" for s, c in zip(symbols, coords_arr)
        )
        mol = gto.M(atom=atom_str, basis=self.basis, charge=charge, unit="Angstrom")

        # Vacuum SCF (for MO coefficients)
        mf_vac = scf.RHF(mol)
        mf_vac.verbose = 0
        mf_vac.run()

        # Solvated SCF (for MM-modified integrals)
        mf_sol = scf.RHF(mol)
        mf_sol.verbose = 0
        mm_coords_bohr = np.asarray(mm_coords) * ANGSTROM_TO_BOHR
        mf_sol = qmmm.mm_charge(mf_sol, mm_coords_bohr, mm_charges)
        mf_sol.run()

        # Determine active space
        n_electrons = mol.nelectron
        n_orbitals = mf_vac.mo_coeff.shape[1]
        if active_electrons is None:
            active_electrons = n_electrons
        if active_orbitals is None:
            active_orbitals = n_orbitals

        n_core = (n_electrons - active_electrons) // 2
        active_idx = list(range(n_core, n_core + active_orbitals))

        # MM effect on single-electron integrals (transform to MO basis)
        mo_coeff_active = mf_vac.mo_coeff[:, active_idx]
        delta_h1e_ao = mf_sol.get_hcore() - mf_vac.get_hcore()
        delta_h1e_mo = mo_coeff_active.T @ delta_h1e_ao @ mo_coeff_active

        # MM effect on nuclear energy
        delta_nuc = mf_sol.energy_nuc() - mol.energy_nuc()

        # Step 3: Build MM correction using qml.s_prod for lightning device compatibility
        # H_mm = delta_nuc + sum_p delta_h1e[p,p] * n_p
        # where n_p = (1 - Z_p)/2 for each spin orbital
        #
        # IMPORTANT: Use qml.s_prod instead of qml.Hamiltonian to maintain Sum type.
        # qml.Hamiltonian returns LinearCombination which breaks TrotterProduct
        # decomposition on lightning devices.

        # Identity term (nuclear + diagonal h1e contribution)
        identity_coeff = delta_nuc
        for p in range(active_orbitals):
            identity_coeff += delta_h1e_mo[p, p]  # 2 * (1/2) for alpha + beta

        # Use multi-wire Identity to match H_vacuum structure
        mm_terms = [qml.s_prod(identity_coeff, qml.Identity(wires=list(range(n_qubits))))]

        # Z terms from diagonal h1e
        for p in range(active_orbitals):
            for spin in [0, 1]:  # alpha, beta
                wire = 2 * p + spin
                mm_terms.append(qml.s_prod(-delta_h1e_mo[p, p] / 2, qml.Z(wire)))

        # Step 4: Combine using qml.sum to preserve Sum type for lightning compatibility
        all_operands = list(H_vacuum.operands) + mm_terms
        H_solvated = qml.sum(*all_operands)

        # Generate HF reference state
        hf_state = qml.qchem.hf_state(active_electrons, n_qubits)

        return H_solvated, n_qubits, hf_state

    def _get_atomic_numbers(self, symbols: list[str]) -> list[int]:
        """Get atomic numbers for a list of element symbols."""
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
        return [atomic_numbers.get(s, 0) for s in symbols]

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
        from pyscf import qmmm, scf

        # Run Hartree-Fock calculation
        mf = scf.RHF(qm_mol)
        mf.verbose = 0  # Suppress SCF output

        # Add MM charges as background charges if provided
        if len(mm_charges) > 0 and len(mm_coords) > 0:
            # Convert MM coords from Angstrom to Bohr (PySCF internal unit)
            ANGSTROM_TO_BOHR = 1.8897259886
            mm_coords_bohr = mm_coords * ANGSTROM_TO_BOHR

            # Add MM point charges to modify single-electron Hamiltonian
            mf = qmmm.mm_charge(mf, mm_coords_bohr, mm_charges)

        # Run SCF calculation
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
