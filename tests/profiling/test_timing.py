# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Tests for profiling/timing.py - general-purpose timing utilities."""


from q2m3.profiling.timing import ProfilingStats, profile_function, profile_section


class TestProfileSection:
    def test_profile_section_returns_timing(self):
        """Context manager should return elapsed > 0 after block completes."""
        with profile_section("test", verbose=False) as timing:
            _ = sum(range(1000))

        assert timing["elapsed"] > 0
        assert timing["name"] == "test"

    def test_profile_section_verbose_off(self, capsys):
        """verbose=False should produce no printed output."""
        with profile_section("silent", verbose=False):
            pass

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_profile_section_verbose_on(self, capsys):
        """verbose=True (default) should print timing information."""
        with profile_section("loud"):
            pass

        captured = capsys.readouterr()
        assert "loud" in captured.out
        assert "Profile" in captured.out


class TestProfileFunction:
    def test_profile_function_decorator(self):
        """Decorator should not change the function's return value."""

        @profile_function(verbose=False)
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_profile_function_with_kwargs(self):
        """@profile_function(verbose=False) syntax should work correctly."""

        @profile_function(verbose=False)
        def compute():
            return 42

        assert compute() == 42

    def test_profile_function_bare_decorator(self, capsys):
        """@profile_function (without parentheses) should print and not change return value."""

        @profile_function
        def greet():
            return "hello"

        result = greet()
        assert result == "hello"
        captured = capsys.readouterr()
        assert "greet" in captured.out


class TestProfilingStats:
    def test_profiling_stats_empty(self):
        """Empty stats should have mean=0 and count=0."""
        stats = ProfilingStats("empty")
        assert stats.mean == 0.0
        assert stats.count == 0
        assert stats.min == 0.0
        assert stats.max == 0.0

    def test_profiling_stats_accumulate(self):
        """Multiple records should produce correct mean/min/max."""
        stats = ProfilingStats("work")
        stats.record(0.1)
        stats.record(0.3)
        stats.record(0.2)

        assert stats.count == 3
        assert abs(stats.mean - 0.2) < 1e-9
        assert abs(stats.min - 0.1) < 1e-9
        assert abs(stats.max - 0.3) < 1e-9

    def test_profiling_stats_summary(self):
        """summary() should return a formatted string with key metrics."""
        stats = ProfilingStats("bench")
        stats.record(1.0)
        stats.record(2.0)

        summary = stats.summary()
        assert "bench" in summary
        assert "count=2" in summary
        assert "mean=" in summary

    def test_profiling_stats_summary_empty(self):
        """summary() with no timings should report 'No timings recorded'."""
        stats = ProfilingStats("nothing")
        summary = stats.summary()
        assert "No timings recorded" in summary
