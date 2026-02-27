"""Tests for orchestrator, statistics, and plotting modules."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Headless backend for CI

from q2m3.solvation.config import QPEConfig, SolvationConfig
from q2m3.solvation.statistics import TimingData, create_timing_data_from_result

# ============================================================================
# statistics.py tests
# ============================================================================


class TestTimingDataCreation:
    """Test create_timing_data_from_result field mapping."""

    def test_basic_field_mapping(self):
        """TimingData fields correctly populated from result dict."""
        result = {
            "hf_times": [0.01, 0.02, 0.015],
            "quantum_times": [0.5, 0.0, 0.6],
            "n_quantum_evaluations": 2,
        }
        timing = create_timing_data_from_result(result, compile_time=1.5, loop_time=10.0)

        assert timing.quantum_compile_time == 1.5
        assert timing.mc_loop_time == 10.0
        assert timing.n_mc_steps == 3
        assert timing.n_quantum_evals == 2
        np.testing.assert_array_equal(timing.hf_times, [0.01, 0.02, 0.015])
        np.testing.assert_array_equal(timing.quantum_times, [0.5, 0.0, 0.6])

    def test_hamiltonian_mode_field(self):
        """hamiltonian_mode field correctly stored."""
        result = {"hf_times": [0.01], "quantum_times": [0.5], "n_quantum_evaluations": 1}
        timing = create_timing_data_from_result(
            result, compile_time=0.0, loop_time=1.0, hamiltonian_mode="dynamic"
        )
        assert timing.hamiltonian_mode == "dynamic"

    def test_empty_result(self):
        """Handles empty/missing arrays gracefully."""
        result = {}
        timing = create_timing_data_from_result(result, compile_time=0.0, loop_time=0.0)
        assert timing.n_mc_steps == 0
        assert timing.n_quantum_evals == 0


class TestPrintTimeStatistics:
    """Smoke test: print_time_statistics with three modes."""

    @pytest.mark.parametrize("mode", ["hf_corrected", "fixed", "dynamic"])
    def test_smoke_three_modes(self, mode):
        """print_time_statistics runs without error for each mode."""
        from rich.console import Console

        from q2m3.solvation.statistics import print_time_statistics

        timing = TimingData(
            quantum_compile_time=2.0,
            mc_loop_time=15.0,
            hf_times=np.array([0.01, 0.02, 0.015, 0.01, 0.02]),
            quantum_times=np.array([0.5, 0.0, 0.6, 0.0, 0.55]),
            n_mc_steps=5,
            n_quantum_evals=3,
            hamiltonian_mode=mode,
        )
        console = Console(file=None, stderr=False, force_terminal=False)
        # Should not raise
        print_time_statistics(timing, console)


class TestStatisticsLabelAdaptation:
    """Test create_timing_table three-mode label adaptation."""

    def test_hf_corrected_label(self):
        """hf_corrected mode: QPE label includes 'diagnostic, interval'."""
        from q2m3.solvation.statistics import create_timing_table

        timing = TimingData(
            hf_times=np.array([0.01] * 10),
            quantum_times=np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
            n_mc_steps=10,
            n_quantum_evals=2,
            hamiltonian_mode="hf_corrected",
        )
        table = create_timing_table(timing)
        # Table rows should be populated without error
        assert table.row_count >= 1

    def test_fixed_mode_label(self):
        """fixed mode: all steps run QPE."""
        from q2m3.solvation.statistics import create_timing_table

        timing = TimingData(
            hf_times=np.array([0.01] * 5),
            quantum_times=np.array([0.5] * 5),
            n_mc_steps=5,
            n_quantum_evals=5,
            hamiltonian_mode="fixed",
        )
        table = create_timing_table(timing)
        assert table.row_count >= 1

    def test_dynamic_mode_label(self):
        """dynamic mode: all steps run QPE."""
        from q2m3.solvation.statistics import create_timing_table

        timing = TimingData(
            hf_times=np.array([0.05] * 5),
            quantum_times=np.array([0.5] * 5),
            n_mc_steps=5,
            n_quantum_evals=5,
            hamiltonian_mode="dynamic",
        )
        table = create_timing_table(timing)
        assert table.row_count >= 1


# ============================================================================
# plotting.py tests
# ============================================================================


class TestPlotEnergyTrajectory:
    """Smoke test: plot_energy_trajectory completes without error."""

    def test_plot_energy_trajectory(self):
        """plot_energy_trajectory on synthetic data."""
        import matplotlib.pyplot as plt

        from q2m3.solvation.plotting import plot_energy_trajectory

        energies = np.array([-1.12, -1.15, -1.10, -1.18, -1.14])
        result = plot_energy_trajectory(
            mc_steps=list(range(1, 6)),
            hf_energies=energies,
            show=False,
        )
        assert result is not None
        plt.close("all")


class TestPlotAcceptanceRate:
    """Smoke test: plot_acceptance_rate completes without error."""

    def test_plot_acceptance_rate(self):
        """plot_acceptance_rate on synthetic data."""
        import matplotlib.pyplot as plt

        from q2m3.solvation.plotting import plot_acceptance_rate

        steps = list(range(6))
        cumulative = [0, 1, 2, 2, 3, 4]
        result = plot_acceptance_rate(steps, cumulative, show=False)
        assert result is not None
        plt.close("all")


# ============================================================================
# orchestrator.py tests
# ============================================================================


class TestPublicAPIExports:
    """Test __all__ exports exactly 7 symbols."""

    def test_public_api_symbols(self):
        """Only 7 public symbols are exported."""
        from q2m3 import solvation

        expected = {
            "run_solvation",
            "MoleculeConfig",
            "QPEConfig",
            "SolvationConfig",
            "SolventModel",
            "TIP3P_WATER",
            "SPC_E_WATER",
        }
        assert set(solvation.__all__) == expected

    def test_run_solvation_importable(self):
        """run_solvation is importable from q2m3.solvation."""
        from q2m3.solvation import run_solvation

        assert callable(run_solvation)


class TestResultDictCompleteness:
    """Test returned dict contains all Layer 1 + Layer 2 fields."""

    # Required fields per the task spec
    LAYER_1_FIELDS = {
        "initial_energy",
        "final_energy",
        "best_energy",
        "best_qpe_energy",
        "acceptance_rate",
        "avg_energy",
        "quantum_energies",
        "hf_energies",
        "hf_times",
        "quantum_times",
        "final_solvent_states",
        "best_solvent_states",
        "best_qpe_solvent_states",
        "n_quantum_evaluations",
        "n_accepted",
    }
    LAYER_2_FIELDS = {
        "timing",
        "e_vacuum",
        "circuit_metadata",
        "mulliken_charges",
    }
    CIRCUIT_META_FIELDS = {
        "hamiltonian_mode",
        "n_system_qubits",
        "n_estimation_wires",
        "total_qubits",
        "n_hamiltonian_terms",
        "n_trotter_steps",
        "n_trotter_steps_requested",
        "base_time",
        "energy_formula",
        "energy_shift",
    }

    @pytest.mark.solvation
    def test_result_dict_fixed_h2(self, h2_molecule_config):
        """Result dict contains all required fields (fixed mode, minimal H2)."""
        from q2m3.solvation import run_solvation

        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=QPEConfig(n_estimation_wires=3, n_trotter_steps=2),
            hamiltonian_mode="fixed",
            n_waters=3,
            n_mc_steps=3,
            verbose=False,
        )
        result = run_solvation(config, show_plots=False)

        # Layer 1
        for field in self.LAYER_1_FIELDS:
            assert field in result, f"Missing Layer 1 field: {field}"
        # Layer 2
        for field in self.LAYER_2_FIELDS:
            assert field in result, f"Missing Layer 2 field: {field}"
        # circuit_metadata sub-fields
        meta = result["circuit_metadata"]
        for field in self.CIRCUIT_META_FIELDS:
            assert field in meta, f"Missing circuit_metadata field: {field}"

        # Type checks
        assert isinstance(result["timing"], TimingData)
        assert isinstance(result["e_vacuum"], float)
        assert isinstance(result["mulliken_charges"], dict)
        assert isinstance(result["quantum_energies"], np.ndarray)
        assert result["quantum_energies"].shape == (3,)
        assert 0.0 <= result["acceptance_rate"] <= 1.0


# ============================================================================
# End-to-end tests (require Catalyst @qjit compilation)
# ============================================================================


@pytest.mark.solvation
class TestEndToEndHfCorrected:
    """End-to-end test: hf_corrected mode."""

    def test_run_solvation_hf_corrected_h2(self, h2_molecule_config):
        """Complete hf_corrected workflow on H2."""
        from q2m3.solvation import run_solvation

        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=QPEConfig(n_estimation_wires=3, n_trotter_steps=2, qpe_interval=2),
            hamiltonian_mode="hf_corrected",
            n_waters=3,
            n_mc_steps=5,
            verbose=False,
        )
        result = run_solvation(config, show_plots=False)

        assert "best_energy" in result
        assert "acceptance_rate" in result
        assert 0.0 <= result["acceptance_rate"] <= 1.0
        assert result["quantum_energies"].shape == (5,)
        # hf_corrected: orchestrator initial call consumes step_counter=0,
        # MC loop steps use counter 1..5, QPE at counter 2,4 → 2 evals in loop
        assert result["n_quantum_evaluations"] == 2


@pytest.mark.solvation
class TestEndToEndFixed:
    """End-to-end test: fixed mode."""

    def test_run_solvation_fixed_h2(self, h2_molecule_config):
        """Complete fixed workflow on H2."""
        from q2m3.solvation import run_solvation

        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=QPEConfig(n_estimation_wires=3, n_trotter_steps=2),
            hamiltonian_mode="fixed",
            n_waters=3,
            n_mc_steps=5,
            verbose=False,
        )
        result = run_solvation(config, show_plots=False)

        assert "best_energy" in result
        assert 0.0 <= result["acceptance_rate"] <= 1.0
        assert result["quantum_energies"].shape == (5,)


@pytest.mark.solvation
class TestEndToEndDynamic:
    """End-to-end test: dynamic mode."""

    def test_run_solvation_dynamic_h2(self, h2_molecule_config):
        """Complete dynamic workflow on H2."""
        from q2m3.solvation import run_solvation

        config = SolvationConfig(
            molecule=h2_molecule_config,
            qpe_config=QPEConfig(n_estimation_wires=3, n_trotter_steps=2),
            hamiltonian_mode="dynamic",
            n_waters=3,
            n_mc_steps=5,
            verbose=False,
        )
        result = run_solvation(config, show_plots=False)

        assert "best_energy" in result
        assert 0.0 <= result["acceptance_rate"] <= 1.0
        assert result["quantum_energies"].shape == (5,)
