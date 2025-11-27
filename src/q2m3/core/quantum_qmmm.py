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
from .rdm import RDMEstimator

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
        rdm_config: dict[str, Any] | None = None,
    ):
        """
        Initialize Quantum-QM/MM calculator.

        Args:
            qm_atoms: Atoms in the QM region
            mm_waters: Number of water molecules in MM region
            qpe_config: Configuration for QPE algorithm
            use_catalyst: Enable Catalyst @qjit compilation for QPE circuits
            rdm_config: Configuration for RDM measurement (default: enabled with defaults)
                - enabled: Enable RDM measurement (default: True)
                - n_shots: Measurement shots (default: 1000)
                - include_off_diagonal: Measure off-diagonal elements (default: True)
                - symmetrize: Enforce Hermitian symmetry (default: True)
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
            # Device selection: "auto", "default.qubit", "lightning.qubit", "lightning.gpu"
            "device_type": "default.qubit",
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
            device_type=self.qpe_config.get("device_type", "default.qubit"),
            use_catalyst=use_catalyst,
        )

        # RDM configuration: default enabled
        default_rdm_config = {
            "enabled": True,
            "n_shots": 1000,
            "include_off_diagonal": True,
            "symmetrize": True,
        }
        if rdm_config is not None:
            default_rdm_config.update(rdm_config)
        self.rdm_config = default_rdm_config

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
            "rdm_source": qpe_result.get("rdm_source", "hartree_fock"),
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

        # Choose Hamiltonian builder based on MM presence
        if len(mm_charges) > 0 and len(mm_coords) > 0:
            # Use new MM-embedded version for solvent effects in QPE
            pl_hamiltonian, n_qubits, hf_state = (
                self.converter.pyscf_to_pennylane_hamiltonian_with_mm(
                    symbols,
                    coords,
                    charge,
                    mm_charges=mm_charges,
                    mm_coords=mm_coords,
                    active_electrons=active_electrons,
                    active_orbitals=active_orbitals,
                )
            )
        else:
            # Use original version for vacuum (no MM) calculations
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
        n_trotter_steps = self.qpe_config.get("n_trotter_steps", 10)
        n_shots = self.qpe_config.get("n_shots", 100)

        # HF energy as reference (used for auto base_time calculation)
        energy_hf = hamiltonian_data.get("energy_hf", 0.0)

        # Auto-compute optimal base_time to avoid phase overflow
        # If base_time="auto" or not specified, compute from HF energy estimate
        base_time_config = self.qpe_config.get("base_time", "auto")
        if base_time_config == "auto" or base_time_config is None:
            base_time = QPEEngine.compute_optimal_base_time(energy_hf)
            logger.info(
                f"Auto-computed base_time={base_time:.4f} from HF energy={energy_hf:.4f} Ha"
            )
        else:
            base_time = float(base_time_config)
            # Warn if configured base_time might cause phase overflow
            phase_estimate = abs(energy_hf) * base_time / (2 * np.pi)
            if phase_estimate > 1.0:
                logger.warning(
                    f"Configured base_time={base_time} may cause phase overflow "
                    f"(estimated phase={phase_estimate:.2f} > 1). Consider using "
                    f"base_time='auto' or a smaller value."
                )

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

        # Determine density matrix source based on RDM config
        if self.rdm_config.get("enabled", True):
            # Measure RDM from Trotter-evolved state
            n_electrons = int(np.sum(hf_state))  # Count electrons from HF state
            rdm_estimator = RDMEstimator(
                n_qubits=n_qubits,
                n_electrons=n_electrons,
                config=self.rdm_config,
            )

            # Get device type from QPE config - pass directly, let device_utils handle "auto"
            device_type = self.qpe_config.get("device_type", "default.qubit")

            # Measure 1-RDM in spin-orbital basis (now supports GPU + batched measurement)
            spin_rdm = rdm_estimator.measure_1rdm(
                hamiltonian=H,
                hf_state=hf_state,
                base_time=base_time,
                n_trotter_steps=n_trotter_steps,
                device_type=device_type,
                use_catalyst=self.use_catalyst,
            )

            # Convert spin-orbital RDM to spatial-orbital RDM
            spatial_rdm = rdm_estimator.spin_to_spatial_rdm(spin_rdm)

            # Check if active space was used - need to convert to AO basis
            active_electrons = self.qpe_config.get("active_electrons")
            active_orbitals = self.qpe_config.get("active_orbitals")

            if active_electrons is not None and active_orbitals is not None:
                # Active space used: convert from active MO basis to AO basis
                mo_coeff = hamiltonian_data["mo_coeff"]
                mo_occ = hamiltonian_data["mo_occ"]
                density_matrix = rdm_estimator.active_mo_to_ao_rdm(
                    active_spatial_rdm=spatial_rdm,
                    mo_coeff=mo_coeff,
                    mo_occ=mo_occ,
                    active_electrons=active_electrons,
                    active_orbitals=active_orbitals,
                )
            else:
                # Full space: spatial RDM is already in MO basis matching AO dimensions
                density_matrix = spatial_rdm
            rdm_source = "quantum_measurement"
        else:
            # Fallback to HF density matrix
            density_matrix = hamiltonian_data["scf_result"].make_rdm1()
            rdm_source = "hartree_fock"

        return {
            "energy": energy,
            "energy_hf": energy_hf,
            "energy_difference": energy_diff,
            "density_matrix": density_matrix,
            "rdm_source": rdm_source,
            "convergence": {
                "converged": True,
                "method": "real_qpe",
                "n_estimation_wires": n_estimation_wires,
                "n_shots": n_shots,
                "rdm_enabled": self.rdm_config.get("enabled", True),
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
        ao_slices = mol.aoslice_by_atom()

        for i, atom in enumerate(mol._atom):
            symbol = atom[0]
            ao_start = ao_slices[i, 2]
            ao_end = ao_slices[i, 3]

            # Mulliken: electron population = sum of diagonal elements for this atom
            electron_pop = np.sum(np.diag(pop_matrix)[ao_start:ao_end])
            nuclear_charge = gto.charge(symbol)
            charge = nuclear_charge - electron_pop

            atomic_charges[f"{symbol}{i}"] = float(charge)

        return atomic_charges

    def draw_circuits(self) -> dict[str, str]:
        """
        Generate text visualizations of QPE and RDM circuits.

        Returns:
            Dictionary with keys:
                - "qpe": QPE circuit visualization
                - "rdm": RDM measurement circuit visualization
        """
        # Build Hamiltonian data to get PennyLane Hamiltonian and HF state
        hamiltonian_data = self._build_qmmm_hamiltonian()
        H = hamiltonian_data["pennylane_hamiltonian"]
        hf_state = hamiltonian_data["hf_state"]
        n_qubits = hamiltonian_data["n_qubits"]

        # Get QPE parameters
        n_estimation_wires = self.qpe_config.get("n_estimation_wires", 4)
        base_time = self.qpe_config.get("base_time", 0.1)
        n_trotter_steps = self.qpe_config.get("n_trotter_steps", 10)

        # Generate QPE circuit visualization
        qpe_diagram = self.qpe_engine.draw_qpe_circuit(
            hamiltonian=H,
            hf_state=hf_state,
            n_estimation_wires=n_estimation_wires,
            base_time=base_time,
            n_trotter_steps=n_trotter_steps,
        )

        # Generate RDM circuit visualization
        n_electrons = int(np.sum(hf_state))
        rdm_estimator = RDMEstimator(
            n_qubits=n_qubits,
            n_electrons=n_electrons,
            config=self.rdm_config,
        )
        rdm_diagram = rdm_estimator.draw_rdm_circuit(
            hamiltonian=H,
            hf_state=hf_state,
            base_time=base_time,
            n_trotter_steps=n_trotter_steps,
        )

        return {
            "qpe": qpe_diagram,
            "rdm": rdm_diagram,
        }
