# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Tests for mc_solvation.statistics module."""

import numpy as np

from examples.mc_solvation.statistics import TimingData, create_timing_table


class TestTimingTableLabels:
    """Tests for QPE-driven mode label adaptation in timing table."""

    def test_standard_mode_labels(self):
        """Standard mode should use HF and Quantum labels."""
        timing = TimingData(
            hf_times=np.array([0.01, 0.02, 0.01]),
            quantum_times=np.array([0.0, 0.0, 0.5]),
            n_mc_steps=3,
            n_quantum_evals=1,
            mc_loop_time=1.0,
        )
        table = create_timing_table(timing)
        # Standard mode: n_quantum_evals (1) != n_mc_steps (3) -> HF/Quantum labels
        assert any("HF" in str(cell) for cell in table.columns[0]._cells)
        assert any("Quantum" in str(cell) for cell in table.columns[0]._cells)

    def test_qpe_driven_mode_labels(self):
        """QPE-driven mode should use Callback and QPE labels."""
        # QPE-driven: n_quantum_evals == n_mc_steps
        timing = TimingData(
            hf_times=np.array([0.01, 0.02, 0.01]),
            quantum_times=np.array([0.5, 0.3, 0.4]),
            n_mc_steps=3,
            n_quantum_evals=3,
            mc_loop_time=1.0,
        )
        table = create_timing_table(timing)
        cells = [str(cell) for cell in table.columns[0]._cells]
        assert any("Callback" in c for c in cells)
        assert any("QPE" in c for c in cells)
        # Should NOT have HF or Quantum labels
        assert not any("HF" in c for c in cells)
        assert not any("Quantum" in c for c in cells)
