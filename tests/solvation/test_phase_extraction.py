# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Tests for phase extraction module (pure NumPy, no quantum dependencies)."""

import numpy as np
import pytest

# ============================================================================
# Tests for extract_energy_from_probs
# ============================================================================


class TestExtractEnergyFromProbs:
    """Tests for analytical probability-based energy extraction."""

    # --- Roundtrip tests: known energy → construct probs → recover energy ---

    def test_known_phase_roundtrip(self):
        """Known energy produces a delta-peak probability → extract recovers it."""
        from q2m3.solvation.phase_extraction import extract_energy_from_probs

        n_estimation_wires = 4
        n_bins = 2**n_estimation_wires  # 16
        base_time = 1.0
        energy_shift = -1.0

        # Target: energy = -2π * phase / base_time + energy_shift
        # Choose bin_idx = 5 → phase = 5/16
        target_bin = 5
        target_phase = target_bin / n_bins
        expected_energy = -2.0 * np.pi * target_phase / base_time + energy_shift

        # Delta-peak at target bin
        probs = np.zeros(n_bins)
        probs[target_bin] = 1.0

        result = extract_energy_from_probs(probs, base_time, energy_shift, n_estimation_wires)
        assert abs(result - expected_energy) < 1e-12

    def test_single_bin_equals_argmax(self):
        """All probability in one bin → result equals argmax approach."""
        from q2m3.solvation.phase_extraction import extract_energy_from_probs

        n_estimation_wires = 3
        n_bins = 2**n_estimation_wires  # 8
        base_time = 2.0
        energy_shift = -0.5

        for bin_idx in range(n_bins):
            probs = np.zeros(n_bins)
            probs[bin_idx] = 1.0

            phase = bin_idx / n_bins
            expected = -2.0 * np.pi * phase / base_time + energy_shift

            result = extract_energy_from_probs(probs, base_time, energy_shift, n_estimation_wires)
            assert abs(result - expected) < 1e-12, f"Failed for bin_idx={bin_idx}"

    def test_uniform_distribution_gives_midpoint(self):
        """Uniform distribution → expected bin = (n_bins-1)/2 → midpoint energy."""
        from q2m3.solvation.phase_extraction import extract_energy_from_probs

        n_estimation_wires = 4
        n_bins = 2**n_estimation_wires
        base_time = 1.0
        energy_shift = 0.0

        probs = np.ones(n_bins) / n_bins

        # expected_bin = sum(k/n_bins for k in range(n_bins)) = (n_bins-1)/2
        expected_bin = (n_bins - 1) / 2.0
        expected_phase = expected_bin / n_bins
        expected_energy = -2.0 * np.pi * expected_phase / base_time + energy_shift

        result = extract_energy_from_probs(probs, base_time, energy_shift, n_estimation_wires)
        assert abs(result - expected_energy) < 1e-12

    def test_base_time_linear_scaling(self):
        """Doubling base_time halves the phase-to-energy contribution."""
        from q2m3.solvation.phase_extraction import extract_energy_from_probs

        n_estimation_wires = 4
        n_bins = 2**n_estimation_wires
        energy_shift = 0.0

        probs = np.zeros(n_bins)
        probs[4] = 1.0  # fixed bin

        e1 = extract_energy_from_probs(
            probs, base_time=1.0, energy_shift=energy_shift, n_estimation_wires=n_estimation_wires
        )
        e2 = extract_energy_from_probs(
            probs, base_time=2.0, energy_shift=energy_shift, n_estimation_wires=n_estimation_wires
        )

        # delta_e = -2π * phase / base_time (shift=0), so e1/e2 = 2
        assert abs(e1 / e2 - 2.0) < 1e-12

    def test_energy_shift_additive(self):
        """Energy shift is purely additive."""
        from q2m3.solvation.phase_extraction import extract_energy_from_probs

        n_estimation_wires = 4
        n_bins = 2**n_estimation_wires

        probs = np.zeros(n_bins)
        probs[3] = 1.0

        e0 = extract_energy_from_probs(
            probs, base_time=1.0, energy_shift=0.0, n_estimation_wires=n_estimation_wires
        )
        e1 = extract_energy_from_probs(
            probs, base_time=1.0, energy_shift=5.0, n_estimation_wires=n_estimation_wires
        )

        assert abs((e1 - e0) - 5.0) < 1e-12

    def test_sub_bin_sensitivity(self):
        """Probability spread across adjacent bins → sub-bin energy resolution."""
        from q2m3.solvation.phase_extraction import extract_energy_from_probs

        n_estimation_wires = 4
        n_bins = 2**n_estimation_wires
        base_time = 1.0
        energy_shift = 0.0

        # 70% on bin 5, 30% on bin 6 → expected_bin = 5.3
        probs = np.zeros(n_bins)
        probs[5] = 0.7
        probs[6] = 0.3

        expected_bin = 5.3
        expected_phase = expected_bin / n_bins
        expected_energy = -2.0 * np.pi * expected_phase / base_time + energy_shift

        result = extract_energy_from_probs(probs, base_time, energy_shift, n_estimation_wires)
        assert abs(result - expected_energy) < 1e-12

    def test_zero_phase_returns_energy_shift(self):
        """All probability on bin 0 → phase=0 → energy=energy_shift."""
        from q2m3.solvation.phase_extraction import extract_energy_from_probs

        n_estimation_wires = 4
        n_bins = 2**n_estimation_wires
        energy_shift = -1.5

        probs = np.zeros(n_bins)
        probs[0] = 1.0

        result = extract_energy_from_probs(
            probs, base_time=1.0, energy_shift=energy_shift, n_estimation_wires=n_estimation_wires
        )
        assert abs(result - energy_shift) < 1e-12


# ============================================================================
# Tests for extract_energy_from_shots
# ============================================================================


class TestExtractEnergyFromShots:
    """Tests for shots-based energy extraction."""

    def test_identical_shots_matches_probs(self):
        """All samples identical → result equals analytical delta-peak."""
        from q2m3.solvation.phase_extraction import (
            extract_energy_from_probs,
            extract_energy_from_shots,
        )

        n_estimation_wires = 4
        n_bins = 2**n_estimation_wires
        base_time = 1.0
        energy_shift = -1.0

        # All shots yield bin_idx = 5 → bitstring [0, 1, 0, 1] (MSB first)
        # bin 5 = 0b0101 → bits [0, 1, 0, 1]
        bitstring = [0, 1, 0, 1]
        n_shots = 100
        samples = np.tile(bitstring, (n_shots, 1))

        # Analytical equivalent
        probs = np.zeros(n_bins)
        probs[5] = 1.0

        e_shots = extract_energy_from_shots(samples, base_time, energy_shift, n_estimation_wires)
        e_probs = extract_energy_from_probs(probs, base_time, energy_shift, n_estimation_wires)

        assert abs(e_shots - e_probs) < 1e-12

    def test_large_sample_convergence(self):
        """Large samples converge to analytical result within 3σ."""
        from q2m3.solvation.phase_extraction import (
            extract_energy_from_probs,
            extract_energy_from_shots,
        )

        n_estimation_wires = 4
        n_bins = 2**n_estimation_wires
        base_time = 1.0
        energy_shift = -1.0

        # Construct probability: 70% bin 5, 30% bin 6
        probs = np.zeros(n_bins)
        probs[5] = 0.7
        probs[6] = 0.3

        e_probs = extract_energy_from_probs(probs, base_time, energy_shift, n_estimation_wires)

        # Generate 10000 samples from this distribution
        rng = np.random.default_rng(42)
        n_shots = 10000
        bin_indices = rng.choice(n_bins, size=n_shots, p=probs)

        # Convert bin indices to bitstrings (MSB first)
        samples = np.zeros((n_shots, n_estimation_wires), dtype=int)
        for i, b in enumerate(bin_indices):
            for j in range(n_estimation_wires):
                samples[i, j] = (b >> (n_estimation_wires - 1 - j)) & 1

        result = extract_energy_from_shots(
            samples,
            base_time,
            energy_shift,
            n_estimation_wires,
            return_statistics=True,
        )

        assert abs(result["mean"] - e_probs) < 3 * result["sem"]

    def test_return_statistics_dict_structure(self):
        """return_statistics=True returns correct dict with required keys."""
        from q2m3.solvation.phase_extraction import extract_energy_from_shots

        n_estimation_wires = 3
        base_time = 1.0
        energy_shift = 0.0
        n_shots = 50

        rng = np.random.default_rng(123)
        samples = rng.integers(0, 2, size=(n_shots, n_estimation_wires))

        result = extract_energy_from_shots(
            samples,
            base_time,
            energy_shift,
            n_estimation_wires,
            return_statistics=True,
        )

        assert isinstance(result, dict)
        assert "mean" in result
        assert "std" in result
        assert "sem" in result
        assert "n_shots" in result
        assert result["n_shots"] == n_shots
        assert result["sem"] == pytest.approx(result["std"] / np.sqrt(n_shots))

    def test_single_shot_returns_that_energy(self):
        """Single shot → energy equals that shot's decoded energy."""
        from q2m3.solvation.phase_extraction import extract_energy_from_shots

        n_estimation_wires = 4
        base_time = 1.0
        energy_shift = -1.0

        # Single shot: bin 10 = 0b1010 → [1, 0, 1, 0]
        samples = np.array([[1, 0, 1, 0]])

        expected_phase = 10 / 16.0
        expected_energy = -2.0 * np.pi * expected_phase / base_time + energy_shift

        result = extract_energy_from_shots(samples, base_time, energy_shift, n_estimation_wires)
        assert abs(result - expected_energy) < 1e-12

    def test_return_float_by_default(self):
        """Default return_statistics=False → returns float."""
        from q2m3.solvation.phase_extraction import extract_energy_from_shots

        n_estimation_wires = 3
        samples = np.array([[0, 1, 1], [1, 0, 0]])

        result = extract_energy_from_shots(
            samples,
            base_time=1.0,
            energy_shift=0.0,
            n_estimation_wires=n_estimation_wires,
        )
        assert isinstance(result, float)

    def test_empty_samples_raises(self):
        """Empty samples array raises ValueError."""
        from q2m3.solvation.phase_extraction import extract_energy_from_shots

        samples = np.empty((0, 4), dtype=int)

        with pytest.raises(ValueError, match="samples"):
            extract_energy_from_shots(
                samples,
                base_time=1.0,
                energy_shift=0.0,
                n_estimation_wires=4,
            )
