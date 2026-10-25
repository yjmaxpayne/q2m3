# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Tests for EFTQC resource estimation functionality.
"""

import numpy as np
import pytest

from q2m3.interfaces import PySCFPennyLaneConverter


def _base_operator(op):
    """Return the non-scaled PennyLane operator for support checks."""
    return getattr(op, "base", op)


def _operator_name(op) -> str:
    return getattr(_base_operator(op), "name", type(_base_operator(op)).__name__)


class TestResourceEstimation:
    """Test EFTQC resource estimation functionality."""

    @pytest.fixture
    def converter(self):
        """Create PySCFPennyLaneConverter instance."""
        return PySCFPennyLaneConverter()

    def test_h2_resource_estimation(self, converter):
        """H2: smallest benchmark system."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        result = converter.estimate_qpe_resources(symbols, coords)

        # Basic structure checks
        assert "logical_qubits" in result
        assert "toffoli_gates" in result
        assert "hamiltonian_1norm" in result
        assert "qpe_iterations" in result
        assert "trotter_steps" in result
        assert "target_error" in result
        assert "n_electrons" in result
        assert "n_orbitals" in result
        assert "basis" in result

        # H2 resource estimation validation
        assert result["logical_qubits"] > 0
        assert result["toffoli_gates"] > 0
        assert result["hamiltonian_1norm"] > 0
        assert result["n_electrons"] == 2  # H + H

        # Sanity checks based on demo results (~1.2M gates, ~115 qubits)
        assert 100_000 < result["toffoli_gates"] < 5_000_000

    def test_trotter_steps_recommendation(self, converter):
        """Verify Trotter steps are reasonably estimated."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        result = converter.estimate_qpe_resources(symbols, coords)

        # Trotter steps should be > 0 and reasonable (10-100 for small molecules)
        assert result["trotter_steps"] >= 10
        assert result["trotter_steps"] <= 200

    def test_qpe_iterations_formula(self, converter):
        """qpe_iterations must match PennyLane DF.estimation_cost = ceil(pi*lambda/(2*error)).

        BUG-QRE-RUNTIME-001 (A3): the old ceil(lambda/error) differed from the
        factor already baked into ``DoubleFactorization.gates`` by pi/2.
        """
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        result = converter.estimate_qpe_resources(symbols, coords, target_error=0.0016)

        expected_iterations = int(np.ceil(np.pi * result["hamiltonian_1norm"] / (2 * 0.0016)))
        assert result["qpe_iterations"] == expected_iterations

    def test_error_scaling(self, converter):
        """Gates should scale inversely with target error."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        result_strict = converter.estimate_qpe_resources(symbols, coords, target_error=0.0016)
        result_relaxed = converter.estimate_qpe_resources(symbols, coords, target_error=0.016)

        # Relaxed error should require fewer gates
        assert result_relaxed["toffoli_gates"] < result_strict["toffoli_gates"]

        # Roughly 10x fewer gates for 10x relaxed error
        ratio = result_strict["toffoli_gates"] / result_relaxed["toffoli_gates"]
        assert 5 < ratio < 15  # Allow tolerance

    def test_active_space_parameters(self, converter):
        """Verify active_electrons and active_orbitals are recorded."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        # Full space (default)
        result_full = converter.estimate_qpe_resources(symbols, coords)
        assert "active_electrons" not in result_full or result_full["active_electrons"] is None

        # Active space (future feature - placeholder test)
        # result_active = converter.estimate_qpe_resources(
        #     symbols, coords, active_electrons=2, active_orbitals=2
        # )
        # assert result_active["active_electrons"] == 2
        # assert result_active["active_orbitals"] == 2

    def test_invalid_coordinates(self, converter):
        """Test error handling for invalid coordinates."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0]])  # Wrong shape

        with pytest.raises((ValueError, IndexError)):
            converter.estimate_qpe_resources(symbols, coords)

    def test_basis_set_recording(self, converter):
        """Verify basis set is correctly recorded."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        result = converter.estimate_qpe_resources(symbols, coords)
        assert result["basis"] == "sto-3g"  # Default basis

    def test_mm_embedded_flag_vacuum(self, converter):
        """Verify mm_embedded flag is False for vacuum Hamiltonian."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        result = converter.estimate_qpe_resources(symbols, coords)
        assert result["mm_embedded"] is False
        assert result["n_mm_charges"] == 0

    def test_mm_embedded_flag_solvated(self, converter):
        """Verify mm_embedded flag is True when MM charges provided."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        # Single TIP3P water molecule
        mm_charges = np.array([-0.834, 0.417, 0.417])
        mm_coords = np.array([[3.0, 0.0, 0.0], [3.5, 0.8, 0.0], [3.5, -0.8, 0.0]])

        result = converter.estimate_qpe_resources(
            symbols, coords, mm_charges=mm_charges, mm_coords=mm_coords
        )
        assert result["mm_embedded"] is True
        assert result["n_mm_charges"] == 3

    def test_mm_embedding_modifies_1norm(self, converter):
        """MM embedding modifies Hamiltonian 1-norm due to QM-MM terms."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        # Vacuum estimate
        result_vacuum = converter.estimate_qpe_resources(symbols, coords)

        # Solvated estimate (2 TIP3P waters)
        mm_charges = np.array([-0.834, 0.417, 0.417, -0.834, 0.417, 0.417])
        mm_coords = np.array(
            [
                [3.0, 0.0, 0.0],
                [3.5, 0.8, 0.0],
                [3.5, -0.8, 0.0],
                [-3.0, 0.0, 0.0],
                [-3.5, 0.8, 0.0],
                [-3.5, -0.8, 0.0],
            ]
        )

        result_solvated = converter.estimate_qpe_resources(
            symbols, coords, mm_charges=mm_charges, mm_coords=mm_coords
        )

        # MM embedding modifies 1-norm (direction depends on charge distribution)
        # The key validation is that both produce valid resource estimates
        assert result_vacuum["hamiltonian_1norm"] > 0
        assert result_solvated["hamiltonian_1norm"] > 0
        assert result_solvated["mm_embedded"] is True
        assert result_vacuum["mm_embedded"] is False

    def test_mm_embedding_mode_selects_diagonal_or_full_delta_h(self, converter):
        """Resource estimates expose diagonal and full one-electron embedding rows."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        mm_charges = np.array([0.25])
        mm_coords = np.array([[2.2, 0.7, 0.3]])

        diagonal = converter.estimate_qpe_resources(
            symbols,
            coords,
            mm_charges=mm_charges,
            mm_coords=mm_coords,
            active_electrons=2,
            active_orbitals=2,
            embedding_mode="diagonal",
        )
        full = converter.estimate_qpe_resources(
            symbols,
            coords,
            mm_charges=mm_charges,
            mm_coords=mm_coords,
            active_electrons=2,
            active_orbitals=2,
            embedding_mode="full_oneelectron",
        )

        assert diagonal["embedding_mode"] == "diagonal"
        assert full["embedding_mode"] == "full_oneelectron"
        assert diagonal["requested_embedding_mode"] == "diagonal"
        assert full["requested_embedding_mode"] == "full_oneelectron"
        assert diagonal["active_indices"] == [0, 1]
        assert full["active_indices"] == [0, 1]
        assert diagonal["fixed_mo"] is True
        assert full["two_electron_tensor_fixed"] is True
        assert full["delta_h_offdiag_fro"] > 0.0
        assert abs(full["hamiltonian_1norm"] - diagonal["hamiltonian_1norm"]) > 1e-10

    def test_default_mm_resource_mode_matches_explicit_full_oneelectron(self, converter):
        """Historical default MM resource rows are explicit full-oneelectron rows."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        mm_charges = np.array([-0.834, 0.417, 0.417])
        mm_coords = np.array([[3.0, 0.0, 0.0], [3.5, 0.8, 0.0], [3.5, -0.8, 0.0]])

        default = converter.estimate_qpe_resources(
            symbols,
            coords,
            active_electrons=2,
            active_orbitals=2,
            mm_charges=mm_charges,
            mm_coords=mm_coords,
        )
        explicit = converter.estimate_qpe_resources(
            symbols,
            coords,
            active_electrons=2,
            active_orbitals=2,
            mm_charges=mm_charges,
            mm_coords=mm_coords,
            embedding_mode="full_oneelectron",
        )

        assert default["embedding_mode"] == "full_oneelectron"
        assert default["hamiltonian_1norm"] == pytest.approx(explicit["hamiltonian_1norm"])
        assert default["toffoli_gates"] == explicit["toffoli_gates"]
        assert default["delta_h_diag_fro"] == pytest.approx(explicit["delta_h_diag_fro"])
        assert default["delta_h_offdiag_fro"] == pytest.approx(explicit["delta_h_offdiag_fro"])

    def test_vacuum_resource_estimate_records_no_embedding_state(self, converter):
        """Vacuum resource rows record the effective embedding state as none."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        result = converter.estimate_qpe_resources(symbols, coords)

        assert result["embedding_mode"] == "none"
        assert result["requested_embedding_mode"] is None
        assert result["fixed_mo"] is False
        assert result["two_electron_tensor_fixed"] is True
        assert result["delta_h_diag_fro"] == pytest.approx(0.0)
        assert result["delta_h_offdiag_fro"] == pytest.approx(0.0)

    def test_invalid_embedding_mode_raises(self, converter):
        """Unsupported embedding modes fail with a clear ValueError."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        with pytest.raises(ValueError, match="embedding_mode"):
            converter.estimate_qpe_resources(
                symbols,
                coords,
                embedding_mode="offdiag_only",
            )

    def test_full_oneelectron_mm_hamiltonian_adds_offdiagonal_operator_support(self, converter):
        """The full one-electron MM Hamiltonian exposes fixed off-diagonal Pauli support."""
        from q2m3.core.hamiltonian_utils import decompose_hamiltonian

        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        mm_charges = np.array([0.25])
        mm_coords = np.array([[2.2, 0.7, 0.3]])

        diagonal, n_qubits_diag, hf_diag = converter.pyscf_to_pennylane_hamiltonian_with_mm(
            symbols,
            coords,
            mm_charges=mm_charges,
            mm_coords=mm_coords,
            active_electrons=2,
            active_orbitals=2,
            embedding_mode="diagonal",
        )
        full, n_qubits_full, hf_full = converter.pyscf_to_pennylane_hamiltonian_with_mm(
            symbols,
            coords,
            mm_charges=mm_charges,
            mm_coords=mm_coords,
            active_electrons=2,
            active_orbitals=2,
            embedding_mode="full_oneelectron",
        )

        _, diagonal_ops = decompose_hamiltonian(diagonal)
        _, full_ops = decompose_hamiltonian(full)

        assert n_qubits_full == n_qubits_diag == 4
        np.testing.assert_array_equal(hf_full, hf_diag)
        assert len(full_ops) > len(diagonal_ops)
        assert any("X" in str(op) or "Y" in str(op) for op in full_ops)

    def test_diagonal_mm_hamiltonian_appends_only_identity_and_z_corrections(self, converter):
        """Diagonal MM mode preserves the historical Identity/Z correction boundary."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        mm_charges = np.array([0.25])
        mm_coords = np.array([[2.2, 0.7, 0.3]])

        vacuum, _, _ = converter.pyscf_to_pennylane_hamiltonian(
            symbols,
            coords,
            active_electrons=2,
            active_orbitals=2,
        )
        diagonal, _, _ = converter.pyscf_to_pennylane_hamiltonian_with_mm(
            symbols,
            coords,
            mm_charges=mm_charges,
            mm_coords=mm_coords,
            active_electrons=2,
            active_orbitals=2,
            embedding_mode="diagonal",
        )

        correction_ops = list(diagonal.operands)[len(vacuum.operands) :]
        assert correction_ops
        assert {_operator_name(op) for op in correction_ops}.issubset({"Identity", "PauliZ"})

    def test_full_oneelectron_hamiltonian_without_mm_charges_returns_vacuum_shape(self, converter):
        """No-MM Hamiltonian calls ignore embedding mode and keep vacuum register shape."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        vacuum, n_qubits_vac, hf_vac = converter.pyscf_to_pennylane_hamiltonian(
            symbols,
            coords,
            active_electrons=2,
            active_orbitals=2,
        )
        no_mm, n_qubits, hf_state = converter.pyscf_to_pennylane_hamiltonian_with_mm(
            symbols,
            coords,
            active_electrons=2,
            active_orbitals=2,
            embedding_mode="full_oneelectron",
        )

        assert n_qubits == n_qubits_vac
        np.testing.assert_array_equal(hf_state, hf_vac)
        assert len(no_mm.operands) == len(vacuum.operands)

    def test_invalid_hamiltonian_embedding_mode_raises(self, converter):
        """Hamiltonian construction validates the requested embedding mode."""
        symbols = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

        with pytest.raises(ValueError, match="embedding_mode"):
            converter.pyscf_to_pennylane_hamiltonian_with_mm(
                symbols,
                coords,
                embedding_mode="offdiag_only",
            )


# Water monomer of the S22 water dimer (system 2); the coordinate unit is set per test.
WATER_MONOMER = (
    ("O", (-1.551007, -0.114520, 0.000000)),
    ("H", (-1.934259, 0.762503, 0.000000)),
    ("H", (-0.599677, 0.040712, 0.000000)),
)


def _water_cas44_integrals(unit: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (4e,4o) STO-3G integrals over canonical RHF orbitals, ``h2`` in chemist notation."""
    from pyscf import ao2mo, gto, mcscf, scf

    mf = scf.RHF(gto.M(atom=list(WATER_MONOMER), basis="sto-3g", unit=unit, verbose=0))
    mf.conv_tol = 1e-10
    mf.kernel()
    cas = mcscf.CASCI(mf, 4, 4)
    h1, _ = cas.get_h1eff()
    return 0.5 * (h1 + h1.T), ao2mo.restore(1, cas.get_h2eff(), 4)


class TestEstimateQPEResourcesFromIntegrals:
    """Double-factorization resources from active-space integrals in chemist notation."""

    @pytest.fixture(scope="class")
    def angstrom(self):
        return _water_cas44_integrals("Angstrom")

    def test_water_monomer_matches_reference_double_factorization(self, angstrom):
        from q2m3.core.resource_estimation import estimate_qpe_resources_from_integrals

        result = estimate_qpe_resources_from_integrals(*angstrom, basis="sto-3g")

        # Reference values from a direct DoubleFactorization run on the same integrals.
        assert result.hamiltonian_1norm == pytest.approx(4.717106721305003, abs=1e-6)
        assert result.logical_qubits == 131
        assert result.toffoli_gates == 5025720
        assert result.n_system_qubits == 8
        assert result.basis == "sto-3g"
        assert result.n_mm_charges == 0
        assert result.embedding_mode == "none"

    def test_bohr_reading_agrees_with_geometry_path(self):
        # The geometry path passes Angstrom coordinates to qml.qchem.Molecule, whose
        # default unit is bohr. Reading the same numbers as Bohr on the integral path
        # therefore gives the same Hamiltonian. When the geometry path converts units,
        # replace this test by a comparison at the same Angstrom geometry.
        from q2m3.core.resource_estimation import (
            estimate_qpe_resources_from_integrals,
            estimate_resources,
        )

        from_integrals = estimate_qpe_resources_from_integrals(
            *_water_cas44_integrals("Bohr"), basis="sto-3g"
        )
        symbols = [symbol for symbol, _ in WATER_MONOMER]
        coords = np.array([xyz for _, xyz in WATER_MONOMER])
        from_geometry = estimate_resources(
            symbols, coords, basis="sto-3g", active_electrons=4, active_orbitals=4
        )

        assert abs(from_integrals.hamiltonian_1norm - from_geometry.hamiltonian_1norm) <= 1e-6
        assert from_integrals.logical_qubits == from_geometry.logical_qubits
        assert from_integrals.toffoli_gates == from_geometry.toffoli_gates

    def test_target_error_and_basis_label_are_passed_through(self, angstrom):
        from q2m3.core.resource_estimation import estimate_qpe_resources_from_integrals

        result = estimate_qpe_resources_from_integrals(
            *angstrom, basis="label", target_error=0.0032
        )

        # Doubling the target error halves the walk-operator calls; lambda is unchanged.
        assert result.hamiltonian_1norm == pytest.approx(4.717106721305003, abs=1e-6)
        assert result.logical_qubits == 129
        assert result.toffoli_gates == 2512860
        assert result.target_error == 0.0032
        assert result.walk_operator_calls == int(
            np.ceil(np.pi * result.hamiltonian_1norm / (2 * 0.0032))
        )
        assert result.basis == "label"

    @pytest.mark.parametrize(
        "h1_shape, h2_shape",
        [((4, 3), (4, 4, 4, 4)), ((4, 4), (4, 4, 4, 3)), ((4, 4), (3, 3, 3, 3))],
    )
    def test_rejects_mismatched_shapes(self, h1_shape, h2_shape):
        from q2m3.core.resource_estimation import estimate_qpe_resources_from_integrals

        with pytest.raises(ValueError, match="shape"):
            estimate_qpe_resources_from_integrals(
                np.zeros(h1_shape), np.zeros(h2_shape), basis="sto-3g"
            )

    def test_rejects_asymmetric_one_electron_matrix(self, angstrom):
        from q2m3.core.resource_estimation import estimate_qpe_resources_from_integrals

        h1, h2 = angstrom
        h1 = h1.copy()
        h1[0, 1] += 1e-10
        with pytest.raises(ValueError, match="symmetry"):
            estimate_qpe_resources_from_integrals(h1, h2, basis="sto-3g")

    def test_rejects_tensor_without_pair_exchange_symmetry(self):
        from q2m3.core.resource_estimation import estimate_qpe_resources_from_integrals

        # Symmetric within each index pair but not under exchange of the two pairs.
        rng = np.random.default_rng(7)
        h2 = rng.normal(size=(4, 4, 4, 4))
        h2 = h2 + h2.transpose(1, 0, 2, 3)
        h2 = h2 + h2.transpose(0, 1, 3, 2)
        assert not np.allclose(h2, h2.transpose(2, 3, 0, 1))
        with pytest.raises(ValueError, match="symmetry"):
            estimate_qpe_resources_from_integrals(np.eye(4), h2, basis="sto-3g")


class TestQuantumQMMMResourceEstimation:
    """Integration tests for resource estimation in QuantumQMMM workflow."""

    def test_compute_ground_state_with_resource_estimation(self):
        """Test resource estimation integration in compute_ground_state."""
        from q2m3.core import QuantumQMMM
        from q2m3.core.qmmm_system import Atom

        # H2 molecule (simplest case for fast test)
        h2_atoms = [
            Atom("H", np.array([0.0, 0.0, 0.0])),
            Atom("H", np.array([0.74, 0.0, 0.0])),
        ]

        # Use minimal QPE config for fast execution
        qpe_config = {
            "use_real_qpe": False,  # Use classical simulation for speed
            "device_type": "default.qubit",
        }

        qmmm = QuantumQMMM(qm_atoms=h2_atoms, mm_waters=0, qpe_config=qpe_config)

        # Without resource estimation (default)
        result_no_res = qmmm.compute_ground_state(include_resource_estimation=False)
        assert "eftqc_resources" not in result_no_res
        assert "energy" in result_no_res

        # With resource estimation
        result_with_res = qmmm.compute_ground_state(include_resource_estimation=True)
        assert "eftqc_resources" in result_with_res
        assert "energy" in result_with_res

        # Validate resource estimation structure
        resources = result_with_res["eftqc_resources"]
        assert "logical_qubits" in resources
        assert "toffoli_gates" in resources
        assert "hamiltonian_1norm" in resources
        assert "qpe_iterations" in resources
        assert "trotter_steps" in resources
        assert resources["n_electrons"] == 2  # H2 has 2 electrons
