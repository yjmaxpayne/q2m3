# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT
"""
Catalyst IR stage analysis utilities.

WARNING: ir_output_dir() uses os.chdir() which modifies global process state.
It is NOT thread-safe. Do not call concurrently from multiple threads.
Must be used as the outermost context manager wrapping @qjit decoration.
"""

import os
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

try:
    from catalyst.debug import get_compilation_stage
except ImportError:
    get_compilation_stage = None  # type: ignore[assignment]


COMPILATION_STAGES = [
    "mlir",
    "QuantumCompilationStage",
    "HLOLoweringStage",
    "BufferizationStage",
    "MLIRToLLVMDialectConversion",
    "LLVMIRTranslation",
]


@contextmanager
def ir_output_dir(path: str | None = None) -> Generator[str, None, None]:
    """Manage Catalyst IR workspace directory.

    Must wrap @qjit decoration (not just the call), because Catalyst captures
    os.getcwd() at decoration time to determine the IR output location.

    Args:
        path: If provided, IR files are written here and preserved after exit.
              If None, a temporary directory is used and auto-cleaned on exit.

    Yields:
        The workspace directory path (user-specified or auto-created tempdir).
    """
    original_cwd = os.getcwd()
    if path is not None:
        os.makedirs(path, exist_ok=True)
        workspace = path
    else:
        workspace = tempfile.mkdtemp(prefix="qpe_ir_")
    os.chdir(workspace)
    try:
        yield workspace
    finally:
        os.chdir(original_cwd)
        if path is None:
            shutil.rmtree(workspace, ignore_errors=True)


def analyze_ir_stages(
    compiled_fn: Any,
    stages: list[str] | None = None,
) -> list[tuple[str, float, int]]:
    """Export IR text from each compilation stage and measure size/lines.

    Args:
        compiled_fn: A Catalyst @qjit compiled function.
        stages: Override list of stage names. If None, uses COMPILATION_STAGES.

    Returns:
        List of (stage_name, size_kb, n_lines). Empty list if Catalyst
        is unavailable or all stages fail.
    """
    if stages is None:
        stages = COMPILATION_STAGES
    results: list[tuple[str, float, int]] = []
    for stage in stages:
        try:
            ir_text = get_compilation_stage(compiled_fn, stage)
            size_kb = len(ir_text.encode("utf-8")) / 1024.0
            n_lines = ir_text.count("\n")
            results.append((stage, size_kb, n_lines))
        except Exception:
            # Stage may not exist for this compilation
            pass
    return results
