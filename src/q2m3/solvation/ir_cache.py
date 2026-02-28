"""
Catalyst LLVM IR cache for QPE circuit compilation.

Implements two-phase compilation caching:
  Phase A (cache miss): Subprocess full @qjit(keep_intermediate=True) -> save LLVM IR
  Phase B (cache hit):  replace_ir + jit_compile -> skip MLIR pipeline (~5x faster)

Cache key = circuit topology parameters only. Hamiltonian coefficients are JAX
runtime parameters that don't affect LLVM IR structure, so IR compiled for
n_waters=5 is valid for n_waters=20.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from .config import SolvationConfig

if TYPE_CHECKING:
    from .circuit_builder import QPECircuitBundle

logger = logging.getLogger(__name__)


def default_cache_dir() -> Path:
    """Find project root (containing pyproject.toml) and return cache dir."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent / "tmp" / "qpe_ir_cache"
    # Fallback to temp directory
    return Path(tempfile.gettempdir()) / "qpe_ir_cache"


def compute_cache_key(config: SolvationConfig) -> str:
    """Generate deterministic cache key from circuit topology parameters.

    The key encodes parameters that affect LLVM IR structure:
    - Molecule identity (name, basis, active space)
    - QPE circuit parameters (estimation wires, Trotter steps, shots)

    Parameters that only affect runtime coefficient values (n_waters,
    temperature, coordinates) are NOT included.
    """
    mol = config.molecule
    qpe = config.qpe_config

    # Map hamiltonian_mode to circuit style for cache key.
    # fixed → zero-arg circuit (different IR); hf_corrected/dynamic → parameterized (same IR).
    circuit_style = "fixed" if config.hamiltonian_mode == "fixed" else "dynamic"

    key_parts = [
        mol.name,
        mol.basis,
        f"{mol.active_electrons}e",
        f"{mol.active_orbitals}o",
        f"{qpe.n_estimation_wires}est",
        f"{qpe.n_trotter_steps}t",
        f"{qpe.n_shots}sh",
        circuit_style,
    ]

    # base_time depends on target_resolution and energy_range — hash them
    bt_str = f"{qpe.target_resolution:.6f}_{qpe.energy_range:.6f}"
    bt_hash = hashlib.sha256(bt_str.encode()).hexdigest()[:8]
    key_parts.append(bt_hash)

    return "_".join(key_parts)


def cache_path_for_config(config: SolvationConfig) -> Path:
    """Return the full path for the cached IR file."""
    cache_dir = Path(config.ir_cache_dir) if config.ir_cache_dir else default_cache_dir()
    key = compute_cache_key(config)
    return cache_dir / f"{key}.ll"


def is_cache_available(config: SolvationConfig) -> bool:
    """Check if a valid cache file exists for this config."""
    if not config.ir_cache_enabled:
        return False
    if config.ir_cache_force_recompile:
        return False
    return cache_path_for_config(config).exists()


def _serialize_config_for_subprocess(config: SolvationConfig) -> dict[str, Any]:
    """Serialize SolvationConfig for cross-process transfer.

    Disables caching in the subprocess config to prevent infinite recursion.
    """
    mol = config.molecule
    return {
        "molecule": {
            "name": mol.name,
            "symbols": mol.symbols,
            "coords": mol.coords,
            "charge": mol.charge,
            "active_electrons": mol.active_electrons,
            "active_orbitals": mol.active_orbitals,
            "basis": mol.basis,
        },
        "qpe_config": {
            "n_estimation_wires": config.qpe_config.n_estimation_wires,
            "n_trotter_steps": config.qpe_config.n_trotter_steps,
            "n_shots": config.qpe_config.n_shots,
            "target_resolution": config.qpe_config.target_resolution,
            "energy_range": config.qpe_config.energy_range,
            "qpe_interval": config.qpe_config.qpe_interval,
        },
        "hamiltonian_mode": config.hamiltonian_mode,
        "n_waters": config.n_waters,
        "n_mc_steps": config.n_mc_steps,
        "temperature": config.temperature,
        "translation_step": config.translation_step,
        "rotation_step": config.rotation_step,
        "initial_water_distance": config.initial_water_distance,
        "random_seed": config.random_seed,
        "verbose": False,  # suppress output in subprocess
        "ir_cache_enabled": False,  # prevent recursion
    }


def _reconstruct_config(d: dict[str, Any]) -> SolvationConfig:
    """Reconstruct SolvationConfig from serialized dict."""
    from .config import MoleculeConfig, QPEConfig

    return SolvationConfig(
        molecule=MoleculeConfig(**d["molecule"]),
        qpe_config=QPEConfig(**d["qpe_config"]),
        hamiltonian_mode=d["hamiltonian_mode"],
        n_waters=d["n_waters"],
        n_mc_steps=d["n_mc_steps"],
        temperature=d["temperature"],
        translation_step=d["translation_step"],
        rotation_step=d["rotation_step"],
        initial_water_distance=d["initial_water_distance"],
        random_seed=d["random_seed"],
        verbose=d["verbose"],
        ir_cache_enabled=d["ir_cache_enabled"],
    )


def _phase_a_worker(
    config_dict: dict[str, Any],
    cache_path_str: str,
    queue: multiprocessing.Queue,
) -> None:
    """Subprocess worker: full @qjit(keep_intermediate=True) compilation.

    Builds the QPE circuit with keep_intermediate=True, triggers compilation,
    extracts LLVM IR via get_compilation_stage, and writes to disk.

    Catalyst captures os.getcwd() at @qjit decoration time to determine
    the compilation artifact output directory. We chdir to the cache directory
    so that compiled_circuit/ artifacts land alongside the .ll cache files.
    """
    try:
        import warnings

        warnings.filterwarnings("ignore")

        from catalyst.debug import get_compilation_stage

        from .circuit_builder import build_qpe_circuit
        from .energy import compute_hf_energy_vacuum

        config = _reconstruct_config(config_dict)
        mol = config.molecule
        qm_coords = np.array(mol.coords)

        # Compute vacuum HF energy (needed for energy shift)
        hf_energy = compute_hf_energy_vacuum(mol)

        # Redirect compilation artifacts to cache directory
        cache_path = Path(cache_path_str)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        os.chdir(str(cache_path.parent))

        # Build circuit with keep_intermediate=True
        t0 = time.monotonic()
        bundle = build_qpe_circuit(config, qm_coords, hf_energy, _keep_intermediate=True)

        # Trigger compilation by calling the circuit
        if bundle.is_fixed_circuit:
            bundle.compiled_circuit()  # Zero-arg
        else:
            bundle.compiled_circuit(jnp.array(bundle.base_coeffs))

        # Extract LLVM IR
        llvm_ir = get_compilation_stage(bundle.compiled_circuit, "LLVMIRTranslation")
        compile_time = time.monotonic() - t0

        # Atomic write: write to temp file then rename
        tmp_path = cache_path.with_suffix(".ll.tmp")
        tmp_path.write_text(llvm_ir)
        os.replace(str(tmp_path), str(cache_path))

        queue.put(
            {
                "compile_time_s": compile_time,
                "ir_size_bytes": len(llvm_ir),
                "error": None,
            }
        )
    except Exception as exc:
        import traceback as tb

        queue.put(
            {
                "compile_time_s": 0.0,
                "ir_size_bytes": 0,
                "error": f"{type(exc).__name__}: {exc}\n{tb.format_exc()}",
            }
        )


def resolve_compiled_circuit(
    config: SolvationConfig,
    bundle: QPECircuitBundle,
) -> tuple[QPECircuitBundle, dict[str, Any]]:
    """Cache-aware QPE circuit compilation resolution.

    On cache hit:  replace_ir + jit_compile (seconds)
    On cache miss: subprocess full compile -> save IR -> replace_ir + jit_compile

    The bundle's compiled_circuit is modified in-place by replace_ir.

    Args:
        config: Solvation configuration with cache settings
        bundle: QPECircuitBundle from build_qpe_circuit()

    Returns:
        (bundle, cache_stats) — bundle is the same object (replace_ir modifies in-place)
    """
    cache_stats: dict[str, Any] = {"is_cache_hit": False}
    cp = cache_path_for_config(config)

    if is_cache_available(config):
        # Phase B only: cache hit
        cache_stats["is_cache_hit"] = True
        try:
            llvm_ir = cp.read_text()
            t0 = time.monotonic()
            _apply_cached_ir(bundle, llvm_ir, work_dir=cp.parent)
            cache_stats["phase_b_time_s"] = time.monotonic() - t0
            cache_stats["ir_size_bytes"] = len(llvm_ir)
            logger.info("IR cache hit: %s (%.2fs)", cp.name, cache_stats["phase_b_time_s"])
            return bundle, cache_stats
        except Exception:
            # Corrupted cache — delete and fall through to Phase A
            logger.warning("Corrupted IR cache %s, falling back to full compile", cp)
            cache_stats["is_cache_hit"] = False
            try:
                cp.unlink(missing_ok=True)
            except OSError:
                pass

    # Phase A + B: cache miss — subprocess compile + save + apply
    config_dict = _serialize_config_for_subprocess(config)
    ctx = multiprocessing.get_context("spawn")
    q: multiprocessing.Queue = ctx.Queue()

    proc = ctx.Process(target=_phase_a_worker, args=(config_dict, str(cp), q))
    proc.start()
    proc.join()

    if q.empty():
        logger.warning("Phase A subprocess produced no result, falling back to normal compile")
        cache_stats["fallback"] = True
        return bundle, cache_stats

    phase_a_result = q.get()
    if phase_a_result["error"]:
        logger.warning("Phase A failed: %s", phase_a_result["error"])
        cache_stats["phase_a_error"] = phase_a_result["error"]
        cache_stats["fallback"] = True
        return bundle, cache_stats

    cache_stats["phase_a_time_s"] = phase_a_result["compile_time_s"]
    cache_stats["ir_size_bytes"] = phase_a_result["ir_size_bytes"]

    # Phase B: apply cached IR
    try:
        llvm_ir = cp.read_text()
        t0 = time.monotonic()
        _apply_cached_ir(bundle, llvm_ir, work_dir=cp.parent)
        cache_stats["phase_b_time_s"] = time.monotonic() - t0
    except Exception:
        logger.warning("Phase B failed after Phase A, falling back to normal compile")
        try:
            cp.unlink(missing_ok=True)
        except OSError:
            pass
        cache_stats["fallback"] = True

    return bundle, cache_stats


def _apply_cached_ir(bundle: QPECircuitBundle, llvm_ir: str, work_dir: Path | None = None) -> None:
    """Apply cached LLVM IR to the bundle's compiled circuit.

    Uses Catalyst's replace_ir to inject cached LLVM IR and jit_compile
    to compile only from LLVM -> machine code (skipping MLIR pipeline).

    Args:
        bundle: QPECircuitBundle to modify in-place
        llvm_ir: LLVM IR text to inject
        work_dir: Directory for compilation artifacts. If provided, chdir
                  there during jit_compile so artifacts land in cache dir.
    """
    from catalyst.debug import replace_ir

    replace_ir(bundle.compiled_circuit, "LLVMIRTranslation", llvm_ir)

    original_cwd = os.getcwd()
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(str(work_dir))
    try:
        if bundle.is_fixed_circuit:
            bundle.compiled_circuit.jit_compile(())  # Zero-arg
        else:
            bundle.compiled_circuit.jit_compile((jnp.array(bundle.base_coeffs),))
    finally:
        if work_dir is not None:
            os.chdir(original_cwd)
