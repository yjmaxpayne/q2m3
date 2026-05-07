# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Quantum device selection and management utilities.

Provides unified device selection logic for QPE and RDM modules,
with GPU acceleration support and Catalyst compatibility.

IMPORTANT: There are TWO separate GPU support systems in q2m3:

1. PennyLane Lightning GPU (cuQuantum/CUDA):
   - Used by standard QPE circuits (without @qjit)
   - Detected by HAS_LIGHTNING_GPU flag
   - Requires: pennylane-lightning[gpu], cuQuantum, CUDA

2. JAX/Catalyst GPU (jaxlib[cuda]):
   - Used by Catalyst @qjit compiled circuits
   - Detected by HAS_JAX_CUDA flag
   - Requires: jax[cuda11_pip] or jax[cuda12_pip]

When Catalyst @qjit is enabled, even with lightning.gpu device,
the actual execution backend is determined by JAX, NOT PennyLane.
If JAX lacks CUDA support, Catalyst runs on CPU regardless of device name.
"""

import warnings
from typing import Literal

import pennylane as qml

# =============================================================================
# Device Availability Detection (executed once at module import)
# =============================================================================

# 1. PennyLane Lightning GPU availability (for standard QPE)
HAS_LIGHTNING_GPU = False
try:
    _test_dev = qml.device("lightning.gpu", wires=1)
    del _test_dev
    HAS_LIGHTNING_GPU = True
except Exception as e:
    # Device "lightning.gpu" not available or failed to initialize; ignore and continue.
    warnings.warn(f'Could not initialize "lightning.gpu" device: {e}', stacklevel=2)

HAS_LIGHTNING_QUBIT = False
try:
    _test_dev = qml.device("lightning.qubit", wires=1)
    del _test_dev
    HAS_LIGHTNING_QUBIT = True
except Exception as e:
    # Device "lightning.qubit" not available or failed to initialize; ignore and continue.
    warnings.warn(f'Could not initialize "lightning.qubit" device: {e}', stacklevel=2)

# 2. JAX/Catalyst GPU availability (for @qjit compiled circuits)
# This is SEPARATE from PennyLane Lightning GPU!
HAS_JAX_CUDA = False
JAX_DEFAULT_BACKEND = "cpu"
try:
    import jax

    JAX_DEFAULT_BACKEND = jax.default_backend()
    # JAX backend is "cuda" or "gpu" when GPU is available
    HAS_JAX_CUDA = JAX_DEFAULT_BACKEND in ("cuda", "gpu")
except ImportError:
    # JAX not installed, which is fine for non-Catalyst usage
    pass
except Exception:
    # Other JAX initialization errors
    pass

# 3. Catalyst availability (for reporting/display purposes)
# NOTE: qpe.py and rdm.py maintain their own HAS_CATALYST + no-op fallback
# for graceful degradation. This detection is for display/reporting only.
HAS_CATALYST = False
CATALYST_VERSION = "N/A"
try:
    import catalyst

    HAS_CATALYST = True
    CATALYST_VERSION = catalyst.__version__
except ImportError:
    pass

# Type alias for supported device types
DeviceType = Literal["auto", "default.qubit", "lightning.qubit", "lightning.gpu"]


def get_best_available_device() -> str:
    """
    Return the name of the best available device.

    Priority: lightning.gpu > lightning.qubit > default.qubit

    Returns:
        Device name string
    """
    if HAS_LIGHTNING_GPU:
        return "lightning.gpu"
    elif HAS_LIGHTNING_QUBIT:
        return "lightning.qubit"
    else:
        return "default.qubit"


def get_catalyst_backend_info() -> dict:
    """
    Get detailed information about Catalyst execution backend.

    IMPORTANT: Catalyst @qjit uses JAX as its execution backend, which is
    SEPARATE from PennyLane device selection. Even if device="lightning.gpu",
    Catalyst will run on CPU if JAX lacks CUDA support.

    Returns:
        Dictionary with:
            - jax_backend: JAX default backend ("cpu", "cuda", "gpu")
            - has_jax_cuda: Whether JAX has CUDA support
            - effective_device: Actual execution device for Catalyst
            - warning: Optional warning message if GPU expected but unavailable
    """
    result = {
        "jax_backend": JAX_DEFAULT_BACKEND,
        "has_jax_cuda": HAS_JAX_CUDA,
        "effective_device": "GPU" if HAS_JAX_CUDA else "CPU",
        "warning": None,
    }

    # Add warning if Lightning GPU is available but JAX GPU is not
    if HAS_LIGHTNING_GPU and not HAS_JAX_CUDA:
        result["warning"] = (
            "PennyLane lightning.gpu is available, but Catalyst @qjit will run on CPU "
            "because JAX lacks CUDA support. To enable Catalyst GPU, install: "
            "pip install 'jax[cuda12]' (or cuda11 for older CUDA versions)"
        )

    return result


def get_effective_catalyst_device_label() -> str:
    """
    Get the effective device label for Catalyst execution.

    Unlike get_best_available_device() which returns the PennyLane device name,
    this returns what Catalyst actually runs on (CPU/GPU based on JAX backend).

    Returns:
        Human-readable label like "lightning.gpu (JAX: CPU)" or "lightning.gpu (JAX: GPU)"
    """
    pennylane_device = get_best_available_device()
    jax_backend = "GPU" if HAS_JAX_CUDA else "CPU"
    return f"{pennylane_device} (JAX: {jax_backend})"


def get_catalyst_effective_backend() -> str:
    """Get human-readable Catalyst execution backend label.

    Returns:
        String like "GPU (JAX CUDA)" or "CPU (JAX)"
    """
    if HAS_JAX_CUDA:
        return "GPU (JAX CUDA)"
    return "CPU (JAX)"


def select_device(
    device_type: str,
    n_wires: int,
    use_catalyst: bool = False,
    seed: int | None = None,
):
    """
    Select quantum device based on device_type parameter.

    NOTE: As of PennyLane v0.43, setting shots on device is deprecated.
    Use qml.set_shots() transform on QNode instead.

    Args:
        device_type: Device selection strategy:
            - "auto": Auto-select best available (GPU > lightning.qubit > default.qubit)
            - "default.qubit": Standard PennyLane simulator
            - "lightning.qubit": High-performance CPU simulator
            - "lightning.gpu": GPU-accelerated simulator (requires cuQuantum)
        n_wires: Number of qubits
        use_catalyst: If True, ensure Catalyst-compatible fallback
                      (lightning.qubit instead of default.qubit)
        seed: Optional device seed for reproducible shot-based sampling

    Returns:
        PennyLane device instance
    """
    device_kwargs = {"wires": n_wires}
    if seed is not None:
        device_kwargs["seed"] = seed

    # Determine fallback device based on Catalyst compatibility
    # Catalyst only supports lightning.qubit and lightning.gpu, not default.qubit
    fallback_device = (
        "lightning.qubit" if (use_catalyst and HAS_LIGHTNING_QUBIT) else "default.qubit"
    )

    if device_type == "auto":
        # Auto-select: GPU > lightning.qubit > default.qubit
        if HAS_LIGHTNING_GPU:
            return qml.device("lightning.gpu", **device_kwargs)
        elif HAS_LIGHTNING_QUBIT:
            return qml.device("lightning.qubit", **device_kwargs)
        else:
            return qml.device("default.qubit", **device_kwargs)

    elif device_type == "lightning.gpu":
        if not HAS_LIGHTNING_GPU:
            warnings.warn(
                f"lightning.gpu not available, falling back to {fallback_device}",
                UserWarning,
                stacklevel=3,
            )
            return qml.device(fallback_device, **device_kwargs)
        return qml.device("lightning.gpu", **device_kwargs)

    elif device_type == "lightning.qubit":
        if not HAS_LIGHTNING_QUBIT:
            warnings.warn(
                "lightning.qubit not available, falling back to default.qubit",
                UserWarning,
                stacklevel=3,
            )
            return qml.device("default.qubit", **device_kwargs)
        return qml.device("lightning.qubit", **device_kwargs)

    else:
        # default.qubit or any unrecognized value
        return qml.device("default.qubit", **device_kwargs)
