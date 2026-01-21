# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Protective tests for h3op_qpe_h2o_mm_full.py demo script.

These tests ensure the refactoring of the 918-line demo script
maintains functional equivalence. Tests are designed to run quickly
(< 5 seconds) by mocking expensive QPE/Catalyst computations.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Test that demo module imports correctly."""

    def test_import_demo_module(self):
        """Demo module should import without errors."""
        import examples.h3op_qpe_h2o_mm_full as demo

        assert hasattr(demo, "main")
        assert hasattr(demo, "create_h3o_geometry")
        assert hasattr(demo, "get_qpe_config")
        assert hasattr(demo, "get_best_available_device")

    def test_import_constants(self):
        """Demo constants should be defined."""
        from examples.h3op_qpe_h2o_mm_full import (
            CHEMICAL_ACCURACY_ERROR,
            ENERGY_CONSISTENCY_THRESHOLD,
            HARTREE_TO_KCAL_MOL,
            MM_STABILIZATION_THRESHOLD,
            RELAXED_ACCURACY_ERROR,
        )

        assert HARTREE_TO_KCAL_MOL == pytest.approx(627.5094, rel=1e-4)
        assert MM_STABILIZATION_THRESHOLD == 0.001
        assert ENERGY_CONSISTENCY_THRESHOLD == 0.01
        assert CHEMICAL_ACCURACY_ERROR == pytest.approx(0.0016, rel=1e-2)
        assert RELAXED_ACCURACY_ERROR == pytest.approx(0.016, rel=1e-2)


# =============================================================================
# Configuration Function Tests
# =============================================================================


class TestCreateH3OGeometry:
    """Test create_h3o_geometry function."""

    def test_returns_list_of_atoms(self):
        """Should return a list of Atom objects."""
        from examples.h3op_qpe_h2o_mm_full import create_h3o_geometry
        from q2m3.core.qmmm_system import Atom

        atoms = create_h3o_geometry()

        assert isinstance(atoms, list)
        assert len(atoms) == 4
        for atom in atoms:
            assert isinstance(atom, Atom)

    def test_contains_correct_elements(self):
        """Should contain 1 O and 3 H atoms."""
        from examples.h3op_qpe_h2o_mm_full import create_h3o_geometry

        atoms = create_h3o_geometry()
        symbols = [atom.symbol for atom in atoms]

        assert symbols.count("O") == 1
        assert symbols.count("H") == 3

    def test_has_correct_total_charge(self):
        """Formal charges should sum to +1 (H3O+ cation)."""
        from examples.h3op_qpe_h2o_mm_full import create_h3o_geometry

        atoms = create_h3o_geometry()
        total_charge = sum(atom.charge for atom in atoms)

        assert total_charge == pytest.approx(1.0, rel=1e-6)

    def test_oxygen_is_first_atom(self):
        """Oxygen should be the first atom in the list."""
        from examples.h3op_qpe_h2o_mm_full import create_h3o_geometry

        atoms = create_h3o_geometry()

        assert atoms[0].symbol == "O"

    def test_positions_are_valid_arrays(self):
        """All positions should be 3D numpy arrays."""
        from examples.h3op_qpe_h2o_mm_full import create_h3o_geometry

        atoms = create_h3o_geometry()

        for atom in atoms:
            assert isinstance(atom.position, np.ndarray)
            assert atom.position.shape == (3,)


class TestGetQPEConfig:
    """Test get_qpe_config function."""

    def test_returns_valid_config_dict(self):
        """Should return a dictionary with required keys."""
        from examples.h3op_qpe_h2o_mm_full import get_qpe_config

        config = get_qpe_config()

        required_keys = [
            "use_real_qpe",
            "n_estimation_wires",
            "base_time",
            "n_trotter_steps",
            "n_shots",
            "active_electrons",
            "active_orbitals",
            "algorithm",
            "mapping",
            "device_type",
        ]

        for key in required_keys:
            assert key in config, f"Missing required key: {key}"

    def test_default_device_type_is_auto(self):
        """Default device_type should be 'auto'."""
        from examples.h3op_qpe_h2o_mm_full import get_qpe_config

        config = get_qpe_config()

        assert config["device_type"] == "auto"

    def test_custom_device_type(self):
        """Should accept custom device_type parameter."""
        from examples.h3op_qpe_h2o_mm_full import get_qpe_config

        config_gpu = get_qpe_config(device_type="lightning.gpu")
        config_cpu = get_qpe_config(device_type="default.qubit")

        assert config_gpu["device_type"] == "lightning.gpu"
        assert config_cpu["device_type"] == "default.qubit"

    def test_uses_real_qpe(self):
        """use_real_qpe should be True."""
        from examples.h3op_qpe_h2o_mm_full import get_qpe_config

        config = get_qpe_config()

        assert config["use_real_qpe"] is True

    def test_active_space_is_4e4o(self):
        """Active space should be 4 electrons, 4 orbitals."""
        from examples.h3op_qpe_h2o_mm_full import get_qpe_config

        config = get_qpe_config()

        assert config["active_electrons"] == 4
        assert config["active_orbitals"] == 4


class TestGetBestAvailableDevice:
    """Test get_best_available_device function."""

    def test_returns_string(self):
        """Should return a device name string."""
        from examples.h3op_qpe_h2o_mm_full import get_best_available_device

        device = get_best_available_device()

        assert isinstance(device, str)
        assert device in ["lightning.gpu", "lightning.qubit", "default.qubit"]

    def test_device_priority_when_no_gpu(self):
        """Without GPU, should return lightning.qubit or default.qubit."""
        from examples.h3op_qpe_h2o_mm_full import (
            HAS_LIGHTNING_GPU,
            HAS_LIGHTNING_QUBIT,
            get_best_available_device,
        )

        device = get_best_available_device()

        if HAS_LIGHTNING_GPU:
            assert device == "lightning.gpu"
        elif HAS_LIGHTNING_QUBIT:
            assert device == "lightning.qubit"
        else:
            assert device == "default.qubit"


class TestGetCatalystEffectiveBackend:
    """Test get_catalyst_effective_backend function."""

    def test_returns_human_readable_string(self):
        """Should return a human-readable backend description."""
        from examples.h3op_qpe_h2o_mm_full import get_catalyst_effective_backend

        backend = get_catalyst_effective_backend()

        assert isinstance(backend, str)
        assert backend in ["GPU (JAX CUDA)", "CPU (JAX)"]


# =============================================================================
# Data Builder Tests
# =============================================================================


class TestBuildOutputData:
    """Test build_output_data function."""

    @pytest.fixture
    def mock_input_data(self):
        """Create mock input data for build_output_data."""
        from examples.h3op_qpe_h2o_mm_full import create_h3o_geometry

        h3o_atoms = create_h3o_geometry()
        mm_waters = 8

        qpe_config = {
            "use_real_qpe": True,
            "n_estimation_wires": 4,
            "base_time": "auto",
            "n_trotter_steps": 10,
            "n_shots": 100,
            "active_electrons": 4,
            "active_orbitals": 4,
            "energy_warning_threshold": 1.0,
            "algorithm": "standard",
            "mapping": "jordan_wigner",
            "device_type": "auto",
        }

        solvation_data = {
            "energy_vacuum": -75.0,
            "energy_solvated": -75.1,
            "stabilization_hartree": 0.1,
            "stabilization_kcal_mol": 62.75,
            "n_mm_atoms": 24,
            "n_mm_waters": 8,
        }

        qpe_solvation_data = {
            "energy_vacuum": -75.0,
            "energy_solvated": -75.1,
            "energy_hf_vacuum": -75.0,
            "energy_hf_solvated": -75.1,
            "stabilization_hartree": 0.1,
            "stabilization_kcal_mol": 62.75,
            "charges_vacuum": {"O": -0.5, "H1": 0.5, "H2": 0.5, "H3": 0.5},
            "charges_solvated": {"O": -0.5, "H1": 0.5, "H2": 0.5, "H3": 0.5},
            "convergence_solvated": {"converged": True, "method": "QPE"},
            "rdm_source_solvated": "quantum",
            "time_vacuum_s": 1.0,
            "time_solvated_s": 1.5,
            "time_total_s": 2.5,
        }

        eftqc_data = {
            "vacuum_chemical": {
                "hamiltonian_1norm": 10.0,
                "toffoli_gates": 1000,
                "logical_qubits": 12,
                "target_error": 0.0016,
                "qpe_iterations": 100,
                "trotter_steps": 10,
            },
            "vacuum_relaxed": {
                "hamiltonian_1norm": 10.0,
                "toffoli_gates": 500,
                "logical_qubits": 12,
                "target_error": 0.016,
                "qpe_iterations": 50,
                "trotter_steps": 5,
            },
            "solvated_chemical": {
                "hamiltonian_1norm": 11.0,
                "toffoli_gates": 1100,
                "logical_qubits": 12,
                "target_error": 0.0016,
                "qpe_iterations": 100,
                "trotter_steps": 10,
                "n_mm_charges": 24,
            },
            "solvated_relaxed": {
                "hamiltonian_1norm": 11.0,
                "toffoli_gates": 550,
                "logical_qubits": 12,
                "target_error": 0.016,
                "qpe_iterations": 50,
                "trotter_steps": 5,
                "n_mm_charges": 24,
            },
            "delta_lambda": 10.0,
            "gate_reduction": 50.0,
            "n_mm_charges": 24,
        }

        return {
            "h3o_atoms": h3o_atoms,
            "mm_waters": mm_waters,
            "qpe_config": qpe_config,
            "solvation_data": solvation_data,
            "qpe_solvation_data": qpe_solvation_data,
            "catalyst_solvation_data": None,
            "eftqc_data": eftqc_data,
        }

    def test_returns_dict(self, mock_input_data):
        """Should return a dictionary."""
        from examples.h3op_qpe_h2o_mm_full import build_output_data

        result = build_output_data(**mock_input_data)

        assert isinstance(result, dict)

    def test_contains_required_top_level_keys(self, mock_input_data):
        """Output should contain all required top-level keys."""
        from examples.h3op_qpe_h2o_mm_full import build_output_data

        result = build_output_data(**mock_input_data)

        required_keys = [
            "timestamp",
            "catalyst_available",
            "catalyst_version",
            "lightning_gpu_available",
            "jax_cuda_available",
            "jax_backend",
            "system",
            "qpe_config",
            "quantum_resources",
            "solvation_effect_hf",
            "solvation_effect_standard_qpe",
            "solvation_effect_catalyst_qpe",
            "eftqc_resources",
        ]

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    def test_system_section_structure(self, mock_input_data):
        """System section should have correct structure."""
        from examples.h3op_qpe_h2o_mm_full import build_output_data

        result = build_output_data(**mock_input_data)

        assert result["system"]["qm_region"] == "H3O+"
        assert result["system"]["n_atoms"] == 4
        assert result["system"]["total_charge"] == 1
        assert result["system"]["mm_waters"] == 8

    def test_quantum_resources_qubit_calculation(self, mock_input_data):
        """Quantum resources should correctly calculate qubit counts."""
        from examples.h3op_qpe_h2o_mm_full import build_output_data

        result = build_output_data(**mock_input_data)
        qr = result["quantum_resources"]

        # 4 orbitals * 2 spin orbitals = 8 system qubits
        assert qr["system_qubits"] == 8
        # 4 estimation qubits
        assert qr["estimation_qubits"] == 4
        # Total = 8 + 4 = 12
        assert qr["total_qubits"] == 12

    def test_catalyst_section_is_none_when_not_provided(self, mock_input_data):
        """Catalyst section should be None when catalyst data is None."""
        from examples.h3op_qpe_h2o_mm_full import build_output_data

        mock_input_data["catalyst_solvation_data"] = None
        result = build_output_data(**mock_input_data)

        assert result["solvation_effect_catalyst_qpe"] is None

    def test_timestamp_is_iso_format(self, mock_input_data):
        """Timestamp should be in ISO format."""
        from datetime import datetime

        from examples.h3op_qpe_h2o_mm_full import build_output_data

        result = build_output_data(**mock_input_data)

        # Should not raise an exception
        datetime.fromisoformat(result["timestamp"])


# =============================================================================
# Print Function Tests (Smoke Tests)
# =============================================================================


class TestPrintFunctions:
    """Smoke tests for print functions - ensure they don't raise exceptions."""

    def test_print_header_no_exception(self, capsys):
        """print_header should execute without error."""
        from examples.h3op_qpe_h2o_mm_full import print_header

        # Should not raise
        print_header()

        captured = capsys.readouterr()
        assert "H3O+" in captured.out or "QPE" in captured.out

    def test_print_section_no_exception(self, capsys):
        """print_section should execute without error."""
        from examples.h3op_qpe_h2o_mm_full import print_section

        print_section("Test Section", step=1)
        print_section("No Step Section")

        captured = capsys.readouterr()
        assert "Test Section" in captured.out
        assert "No Step Section" in captured.out

    def test_print_system_info_no_exception(self, capsys):
        """print_system_info should execute without error."""
        from examples.h3op_qpe_h2o_mm_full import get_qpe_config, print_system_info

        config = get_qpe_config()
        print_system_info(config, mm_waters=8)

        captured = capsys.readouterr()
        assert "H3O+" in captured.out
        assert "8" in captured.out  # mm_waters

    def test_print_hf_solvation_effect_no_exception(self, capsys):
        """print_hf_solvation_effect should execute without error."""
        from examples.h3op_qpe_h2o_mm_full import print_hf_solvation_effect

        solvation_data = {
            "energy_vacuum": -75.0,
            "energy_solvated": -75.1,
            "stabilization_hartree": 0.1,
            "stabilization_kcal_mol": 62.75,
            "n_mm_atoms": 24,
            "n_mm_waters": 8,
        }

        print_hf_solvation_effect(solvation_data)

        captured = capsys.readouterr()
        assert "Hartree" in captured.out or "kcal" in captured.out


# =============================================================================
# Smoke Test for main function
# =============================================================================


class TestMainFunction:
    """Test main function can be called (with mocked computations)."""

    @pytest.mark.slow
    def test_main_function_is_callable(self):
        """main function should exist and be callable."""
        from examples.h3op_qpe_h2o_mm_full import main

        assert callable(main)

    @pytest.mark.slow
    def test_main_with_mocked_computation(self):
        """main should run with mocked QPE computation.

        This test validates the main function's structure without
        running expensive QPE calculations.
        """
        # Mock the expensive computations
        mock_qmmm_result = {
            "energy": -75.3,
            "energy_hf": -75.2,
            "atomic_charges": {"O": -0.5, "H1": 0.5, "H2": 0.5, "H3": 0.5},
            "convergence": {"converged": True, "method": "QPE"},
            "rdm_source": "quantum",
        }

        mock_circuits = {"qpe": "mock_qpe_circuit", "rdm": "mock_rdm_circuit"}

        mock_eftqc = {
            "hamiltonian_1norm": 10.0,
            "toffoli_gates": 1000,
            "logical_qubits": 12,
            "target_error": 0.0016,
            "qpe_iterations": 100,
            "trotter_steps": 10,
            "n_mm_charges": 0,
        }

        with (
            patch("examples.h3op_qpe_h2o_mm_full.QuantumQMMM") as mock_qmmm_class,
            patch("examples.h3op_qpe_h2o_mm_full.PySCFPennyLaneConverter") as mock_converter_class,
            patch("examples.h3op_qpe_h2o_mm_full.save_json_results") as mock_save,
        ):
            # Setup mock QuantumQMMM
            mock_qmmm = MagicMock()
            mock_qmmm.compute_ground_state.return_value = mock_qmmm_result
            mock_qmmm.draw_circuits.return_value = mock_circuits
            mock_qmmm_class.return_value = mock_qmmm

            # Setup mock converter
            mock_converter = MagicMock()
            mock_converter.build_qmmm_hamiltonian.return_value = {
                "energy_hf": -75.2,
                "mol": MagicMock(),
            }
            mock_converter.estimate_qpe_resources.return_value = mock_eftqc
            mock_converter_class.return_value = mock_converter

            # Import and call main
            from examples.h3op_qpe_h2o_mm_full import main

            # This should complete without error
            main()

            # Verify save was called
            mock_save.assert_called_once()
            saved_data = mock_save.call_args[0][0]
            assert "timestamp" in saved_data
            assert "system" in saved_data


# =============================================================================
# Analysis Function Tests
# =============================================================================


class TestAnalysisFunctions:
    """Test analysis functions with mocked dependencies."""

    @pytest.mark.slow
    def test_analyze_solvation_effect_structure(self):
        """analyze_solvation_effect should return expected structure."""
        from examples.h3op_qpe_h2o_mm_full import (
            analyze_solvation_effect,
            create_h3o_geometry,
        )

        h3o_atoms = create_h3o_geometry()

        # Use 0 waters for faster test
        result = analyze_solvation_effect(h3o_atoms, mm_waters=0)

        assert "energy_vacuum" in result
        assert "energy_solvated" in result
        assert "stabilization_hartree" in result
        assert "stabilization_kcal_mol" in result
        assert isinstance(result["energy_vacuum"], float)

    @pytest.mark.slow
    def test_run_resource_estimation_structure(self):
        """run_resource_estimation should return expected structure."""
        from examples.h3op_qpe_h2o_mm_full import (
            create_h3o_geometry,
            run_resource_estimation,
        )

        h3o_atoms = create_h3o_geometry()

        # Use 0 waters for faster test
        result = run_resource_estimation(h3o_atoms, mm_waters=0)

        assert "vacuum_chemical" in result
        assert "vacuum_relaxed" in result
        assert "solvated_chemical" in result
        assert "solvated_relaxed" in result
        assert "delta_lambda" in result
        assert "gate_reduction" in result

        # Check nested structure
        assert "hamiltonian_1norm" in result["vacuum_chemical"]
        assert "toffoli_gates" in result["vacuum_chemical"]
        assert "logical_qubits" in result["vacuum_chemical"]
