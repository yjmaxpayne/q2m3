# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""QA tests for examples/qpe_memory_profile.py after refactoring.

Four-layer test strategy:
  Layer 1: Import & re-export verification (pure Python, no deps)
  Layer 2: Output formatting functions (construct data, capture Rich output)
  Layer 3: main() flow paths (mock profiling functions, verify routing)
  Layer 4: End-to-end integration (requires Catalyst, marked slow)
"""

import importlib.util
from dataclasses import replace
from unittest.mock import patch

import pytest

from q2m3.profiling.memory import ProfileResult

HAS_CATALYST = importlib.util.find_spec("catalyst") is not None


# =============================================================================
# Layer 1: Import & Re-export Integrity
# =============================================================================


class TestImportIntegrity:
    """Verify script importability and re-export chain completeness."""

    def test_script_importable(self):
        """examples/qpe_memory_profile.py can be imported as a module."""
        import examples.qpe_memory_profile as mod

        assert hasattr(mod, "main")
        assert hasattr(mod, "console")

    def test_reexported_symbols_match_source(self):
        """Three re-exported symbols have object identity with source module."""
        from examples.qpe_memory_profile import (
            analyze_ir_stages,
            ir_output_dir,
            take_snapshot,
        )
        from q2m3.profiling import (
            analyze_ir_stages as src_analyze,
        )
        from q2m3.profiling import (
            ir_output_dir as src_ir_dir,
        )
        from q2m3.profiling import (
            take_snapshot as src_snap,
        )

        assert analyze_ir_stages is src_analyze
        assert ir_output_dir is src_ir_dir
        assert take_snapshot is src_snap

    def test_downstream_imports_work(self):
        """Downstream scripts' import statements succeed without error."""
        from examples.qpe_memory_profile import (
            analyze_ir_stages,
            ir_output_dir,
            take_snapshot,
        )

        assert callable(analyze_ir_stages)
        assert callable(take_snapshot)
        assert callable(ir_output_dir)


# =============================================================================
# Layer 2: Output Formatting Functions
# =============================================================================


class TestOutputFormatting:
    """Verify print_* functions produce expected output without crashing."""

    def test_print_circuit_params(self, mock_result, capture_console):
        """Circuit parameters panel includes molecule and qubit count."""
        _, buf = capture_console
        from examples.qpe_memory_profile import print_circuit_params

        print_circuit_params(mock_result)
        output = buf.getvalue()
        assert "H2" in output
        assert "4" in output  # n_system_qubits

    def test_print_memory_table(self, mock_result, capture_console):
        """Memory table renders all three phase rows."""
        _, buf = capture_console
        from examples.qpe_memory_profile import print_memory_table

        print_memory_table(mock_result)
        output = buf.getvalue()
        assert "test" in output  # snapshot label appears in each row

    def test_print_ir_analysis_with_data(self, mock_result, capture_console):
        """IR analysis table shows stage names from ir_analysis list."""
        _, buf = capture_console
        from examples.qpe_memory_profile import print_ir_analysis

        print_ir_analysis(mock_result.ir_analysis)
        output = buf.getvalue()
        assert "mlir" in output
        assert "LLVMIRTranslation" in output

    def test_print_ir_analysis_empty(self, capture_console):
        """Empty ir_analysis triggers 'No IR analysis' message."""
        _, buf = capture_console
        from examples.qpe_memory_profile import print_ir_analysis

        print_ir_analysis([])
        output = buf.getvalue()
        assert "No IR analysis" in output

    def test_print_memory_timeline_with_samples(self, capture_console):
        """4+ samples render ASCII timeline with Peak RSS summary."""
        _, buf = capture_console
        from examples.qpe_memory_profile import print_memory_timeline

        samples = [(0.0, 100.0), (1.0, 200.0), (2.0, 250.0), (3.0, 180.0)]
        print_memory_timeline(samples)
        output = buf.getvalue()
        assert "Peak RSS" in output

    def test_print_memory_timeline_too_few_samples(self, capture_console):
        """Fewer than 3 samples silently skipped (no output)."""
        _, buf = capture_console
        from examples.qpe_memory_profile import print_memory_timeline

        print_memory_timeline([(0.0, 100.0), (1.0, 200.0)])
        output = buf.getvalue()
        assert "Peak" not in output

    def test_print_mode_comparison(self, mock_result, capture_console):
        """Mode comparison table includes H_fixed and H_dynamic columns."""
        _, buf = capture_console
        from examples.qpe_memory_profile import print_mode_comparison

        fixed = replace(mock_result, mode="fixed")
        dynamic = replace(mock_result, mode="dynamic")
        print_mode_comparison(fixed, dynamic)
        output = buf.getvalue()
        assert "H_fixed" in output
        assert "H_dynamic" in output

    def test_print_sweep_table(self, mock_result, capture_console):
        """Sweep table renders OK status and error message for mixed results."""
        _, buf = capture_console
        from examples.qpe_memory_profile import print_sweep_table

        ok_result = replace(mock_result, mode="fixed")
        err_result = replace(mock_result, error="OOM", phase_b=None, mode="dynamic")
        print_sweep_table([ok_result, err_result])
        output = buf.getvalue()
        assert "OK" in output
        assert "OOM" in output

    def test_print_summary(self, mock_result, capture_console):
        """Summary panel includes molecule name and 'Summary' in title."""
        _, buf = capture_console
        from examples.qpe_memory_profile import print_summary

        print_summary(mock_result)
        output = buf.getvalue()
        assert "H2" in output
        assert "Summary" in output


# =============================================================================
# Layer 3: main() Flow Routing
# =============================================================================


class TestMainFlowRouting:
    """Verify main() routes to correct profiling functions with correct args."""

    def test_main_both_mode_default(self, monkeypatch, mock_result, capture_console, parent_data):
        """Default args (no flags) → run_both_modes with h2, n_est=4, n_trotter=10."""
        monkeypatch.setattr("sys.argv", ["prog"])
        fixed = replace(mock_result, mode="fixed")
        dynamic = replace(mock_result, mode="dynamic")

        with patch("examples.qpe_memory_profile.run_both_modes") as m:
            m.return_value = (fixed, dynamic, parent_data, parent_data)
            from examples.qpe_memory_profile import main

            main()

        m.assert_called_once()
        call_args, call_kwargs = m.call_args
        assert call_args[0] == "h2"
        assert call_args[1] == 4  # n_est default
        assert call_args[2] == 10  # n_trotter default
        assert call_kwargs["ir_dir"] is None
        assert call_kwargs["on_progress"] is not None

    def test_main_single_fixed(self, monkeypatch, mock_result, capture_console):
        """--mode fixed --n-trotter 3 → run_single_profile with correct params."""
        monkeypatch.setattr("sys.argv", ["prog", "--mode", "fixed", "--n-trotter", "3"])
        result = replace(mock_result, mode="fixed")

        with patch("examples.qpe_memory_profile.run_single_profile") as m:
            m.return_value = result
            from examples.qpe_memory_profile import main

            main()

        m.assert_called_once()
        call_args, call_kwargs = m.call_args
        assert call_args[0].name == "H2"  # MOLECULES["h2"]
        assert call_args[1] == 4  # n_est default
        assert call_args[2] == 3  # n_trotter from CLI
        assert call_kwargs["mode"] == "fixed"

    def test_main_single_dynamic(self, monkeypatch, mock_result, capture_console):
        """--mode dynamic → run_single_profile with mode='dynamic'."""
        monkeypatch.setattr("sys.argv", ["prog", "--mode", "dynamic"])
        result = replace(mock_result, mode="dynamic")

        with patch("examples.qpe_memory_profile.run_single_profile") as m:
            m.return_value = result
            from examples.qpe_memory_profile import main

            main()

        m.assert_called_once()
        _, call_kwargs = m.call_args
        assert call_kwargs["mode"] == "dynamic"

    def test_main_sweep_both(self, monkeypatch, mock_result, capture_console):
        """--sweep (default mode=both) → run_sweep called twice (fixed + dynamic)."""
        monkeypatch.setattr("sys.argv", ["prog", "--sweep"])

        with patch("examples.qpe_memory_profile.run_sweep") as m:
            m.return_value = {(2, 1): mock_result}
            from examples.qpe_memory_profile import main

            main()

        assert m.call_count == 2

    def test_main_sweep_single_mode(self, monkeypatch, mock_result, capture_console):
        """--sweep --mode fixed → run_sweep called once."""
        monkeypatch.setattr("sys.argv", ["prog", "--sweep", "--mode", "fixed"])

        with patch("examples.qpe_memory_profile.run_sweep") as m:
            m.return_value = {(2, 1): mock_result}
            from examples.qpe_memory_profile import main

            main()

        assert m.call_count == 1

    def test_main_molecule_h3o(self, monkeypatch, mock_result, capture_console):
        """--molecule h3o --mode fixed → run_single_profile receives H3O+ molecule."""
        monkeypatch.setattr("sys.argv", ["prog", "--molecule", "h3o", "--mode", "fixed"])
        result = replace(mock_result, mode="fixed", molecule="H3O+")

        with patch("examples.qpe_memory_profile.run_single_profile") as m:
            m.return_value = result
            from examples.qpe_memory_profile import main

            main()

        m.assert_called_once()
        call_args, _ = m.call_args
        assert call_args[0].name == "H3O+"


# =============================================================================
# Layer 4: End-to-End Integration (slow, requires Catalyst)
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not HAS_CATALYST, reason="requires Catalyst")
class TestEndToEnd:
    """Integration smoke tests with real Catalyst compilation."""

    def test_e2e_single_mode_fixed(self):
        """H2 fixed mode, n_est=2, n_trotter=1 — full pipeline validation."""
        import tracemalloc

        from q2m3.profiling import MOLECULES, run_single_profile

        tracemalloc.start()
        result = run_single_profile(
            MOLECULES["h2"],
            n_est=2,
            n_trotter=1,
            mode="fixed",
            on_progress=print,
        )
        tracemalloc.stop()

        # Structural completeness
        assert isinstance(result, ProfileResult)
        assert result.error is None
        assert result.phase_a is not None
        assert result.phase_b is not None
        assert result.phase_c is not None

        # Scientific sanity
        assert result.prob_sum > 0.99
        assert result.phase_b.elapsed_s > 0
        assert result.n_terms > 0

        # Output formatting doesn't crash
        from examples.qpe_memory_profile import _print_single_result

        _print_single_result(result)

    def test_e2e_sweep_minimal_grid(self):
        """Single-point sweep with subprocess isolation — validates full chain."""
        from q2m3.profiling import run_sweep

        results = run_sweep(
            mol_key="h2",
            mode="dynamic",
            grid=[(2, 1)],
            on_progress=print,
        )

        assert len(results) == 1
        result = results[(2, 1)]
        assert result.error is None
        assert result.prob_sum > 0.99
