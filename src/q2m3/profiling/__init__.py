# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
q2m3 profiling subpackage.

Provides memory measurement and QPE compilation profiling utilities.
"""

from .catalyst_ir import COMPILATION_STAGES, analyze_ir_stages, ir_output_dir
from .memory import (
    MemorySnapshot,
    MemoryTimeline,
    ParentSideMonitor,
    ProfileResult,
    take_snapshot,
)
from .orchestrator import (
    H2_SWEEP_GRID,
    MOLECULES,
    run_both_modes,
    run_single_profile,
    run_single_profile_in_subprocess,
    run_sweep,
)
from .qpe_profiler import (
    profile_execution,
    profile_hamiltonian_build,
    profile_qjit_compilation,
    profile_qjit_compilation_fixed,
)

__all__ = [
    # memory.py
    "MemorySnapshot",
    "ProfileResult",
    "MemoryTimeline",
    "ParentSideMonitor",
    "take_snapshot",
    # catalyst_ir.py
    "COMPILATION_STAGES",
    "ir_output_dir",
    "analyze_ir_stages",
    # qpe_profiler.py
    "profile_hamiltonian_build",
    "profile_qjit_compilation",
    "profile_qjit_compilation_fixed",
    "profile_execution",
    # orchestrator.py
    "MOLECULES",
    "H2_SWEEP_GRID",
    "run_single_profile",
    "run_single_profile_in_subprocess",
    "run_sweep",
    "run_both_modes",
]
