# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
General-purpose timing utilities for q2m3.

Provides lightweight profiling tools for measuring code section and function
execution times. These utilities are independent of QPE-specific profiling
and can be used anywhere in the codebase.
"""

import functools
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any


@contextmanager
def profile_section(name: str, verbose: bool = True):
    """Context manager for profiling a code section.

    Args:
        name: Name of the section being profiled
        verbose: Whether to print timing information

    Yields:
        dict: A timing info dictionary that gets updated with elapsed time

    Example:
        with profile_section("QPE calculation") as timing:
            result = run_qpe()
        print(f"Elapsed: {timing['elapsed']:.3f}s")
    """
    timing_info = {"start": 0.0, "end": 0.0, "elapsed": 0.0, "name": name}
    timing_info["start"] = time.perf_counter()

    try:
        yield timing_info
    finally:
        timing_info["end"] = time.perf_counter()
        timing_info["elapsed"] = timing_info["end"] - timing_info["start"]

        if verbose:
            print(f"[Profile] {name}: {timing_info['elapsed']:.3f}s")


def profile_function(func: Callable = None, *, verbose: bool = True) -> Callable:
    """Decorator for profiling a function's execution time.

    Args:
        func: The function to profile
        verbose: Whether to print timing information

    Returns:
        Wrapped function that tracks execution time

    Example:
        @profile_function
        def compute_energy():
            ...

        @profile_function(verbose=False)
        def silent_compute():
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                if verbose:
                    print(f"[Profile] {fn.__name__}: {elapsed:.3f}s")

        # Store timing info on the wrapper for inspection
        wrapper._last_elapsed = 0.0
        return wrapper

    # Handle both @profile_function and @profile_function() syntax
    if func is not None:
        return decorator(func)
    return decorator


class ProfilingStats:
    """Accumulator for profiling statistics across multiple runs."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.timings: list[float] = []

    def record(self, elapsed: float) -> None:
        """Record a timing measurement."""
        self.timings.append(elapsed)

    @property
    def total(self) -> float:
        """Total elapsed time."""
        return sum(self.timings)

    @property
    def count(self) -> int:
        """Number of recorded timings."""
        return len(self.timings)

    @property
    def mean(self) -> float:
        """Mean elapsed time."""
        if not self.timings:
            return 0.0
        return self.total / self.count

    @property
    def min(self) -> float:
        """Minimum elapsed time."""
        if not self.timings:
            return 0.0
        return min(self.timings)

    @property
    def max(self) -> float:
        """Maximum elapsed time."""
        if not self.timings:
            return 0.0
        return max(self.timings)

    def summary(self) -> str:
        """Return a summary string of profiling stats."""
        if not self.timings:
            return f"{self.name}: No timings recorded"
        return (
            f"{self.name}: "
            f"count={self.count}, total={self.total:.3f}s, "
            f"mean={self.mean:.3f}s, min={self.min:.3f}s, max={self.max:.3f}s"
        )

    def __repr__(self) -> str:
        return self.summary()
