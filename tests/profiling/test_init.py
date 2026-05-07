# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""Tests for q2m3.profiling public API accessibility."""


def test_profiling_public_api():
    """Verify all expected symbols are accessible from q2m3.profiling."""
    from q2m3.profiling import (
        COMPILATION_STAGES,
        MemorySnapshot,
        MemoryTimeline,
        ParentSideMonitor,
        ProfileResult,
        analyze_ir_stages,
        ir_output_dir,
        profile_execution,
        profile_hamiltonian_build,
        profile_qjit_compilation,
        profile_qjit_compilation_fixed,
        run_both_modes,
        run_single_profile,
        run_single_profile_in_subprocess,
        run_sweep,
        take_snapshot,
    )

    assert ProfileResult is not None
    assert MemorySnapshot is not None
    assert MemoryTimeline is not None
    assert ParentSideMonitor is not None
    assert COMPILATION_STAGES is not None
    assert callable(take_snapshot)
    assert callable(ir_output_dir)
    assert callable(analyze_ir_stages)
    assert callable(profile_hamiltonian_build)
    assert callable(profile_qjit_compilation)
    assert callable(profile_qjit_compilation_fixed)
    assert callable(profile_execution)
    assert callable(run_single_profile)
    assert callable(run_single_profile_in_subprocess)
    assert callable(run_sweep)
    assert callable(run_both_modes)


def test_profiling_constants():
    """Verify module-level constants are accessible."""
    from q2m3.profiling import H2_SWEEP_GRID, MOLECULES

    assert isinstance(MOLECULES, dict)
    assert len(MOLECULES) > 0
    assert isinstance(H2_SWEEP_GRID, list)
    assert len(H2_SWEEP_GRID) > 0


def test_profiling_all_list():
    """Verify __all__ is defined and comprehensive."""
    import q2m3.profiling as profiling

    assert hasattr(profiling, "__all__")
    expected = {
        "MemorySnapshot",
        "ProfileResult",
        "MemoryTimeline",
        "ParentSideMonitor",
        "take_snapshot",
        "COMPILATION_STAGES",
        "ir_output_dir",
        "analyze_ir_stages",
        "profile_hamiltonian_build",
        "profile_qjit_compilation",
        "profile_qjit_compilation_fixed",
        "profile_execution",
        "MOLECULES",
        "H2_SWEEP_GRID",
        "run_single_profile",
        "run_single_profile_in_subprocess",
        "run_sweep",
        "run_both_modes",
    }
    assert expected.issubset(set(profiling.__all__))
