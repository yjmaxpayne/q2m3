# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Configuration module for H3O+ QPE Demo.

Contains:
- Physical constants and thresholds
- Device detection variables
- Configuration functions
"""

import warnings

import numpy as np
import pennylane as qml

from q2m3.core.qmmm_system import Atom

# =============================================================================
# Physical Constants and Thresholds
# =============================================================================

HARTREE_TO_KCAL_MOL = 627.5094
MM_STABILIZATION_THRESHOLD = (
    0.001  # Hartree - minimum stabilization to consider MM embedding active
)
ENERGY_CONSISTENCY_THRESHOLD = 0.01  # Hartree - acceptable difference between methods
CHEMICAL_ACCURACY_ERROR = 0.0016  # Hartree (~1 kcal/mol)
RELAXED_ACCURACY_ERROR = 0.016  # Hartree (~10 kcal/mol)

# =============================================================================
# Catalyst Availability Detection
# =============================================================================

try:
    import catalyst

    HAS_CATALYST = True
    CATALYST_VERSION = catalyst.__version__
except ImportError:
    HAS_CATALYST = False
    CATALYST_VERSION = "N/A"

# =============================================================================
# Device Detection: PennyLane Lightning (for standard QPE)
# =============================================================================

HAS_LIGHTNING_GPU = False
try:
    _test_dev = qml.device("lightning.gpu", wires=1)
    del _test_dev
    HAS_LIGHTNING_GPU = True
except Exception:
    pass

HAS_LIGHTNING_QUBIT = False
try:
    _test_dev = qml.device("lightning.qubit", wires=1)
    del _test_dev
    HAS_LIGHTNING_QUBIT = True
except Exception as e:
    warnings.warn(
        f"Could not initialize PennyLane 'lightning.qubit' device: {e}. "
        "Falling back to other devices.",
        RuntimeWarning,
        stacklevel=2,
    )

# =============================================================================
# Device Detection: JAX/Catalyst GPU (for @qjit compiled circuits)
# IMPORTANT: This is SEPARATE from PennyLane Lightning GPU!
# =============================================================================

HAS_JAX_CUDA = False
JAX_DEFAULT_BACKEND = "cpu"
try:
    import jax

    JAX_DEFAULT_BACKEND = jax.default_backend()
    HAS_JAX_CUDA = JAX_DEFAULT_BACKEND in ("cuda", "gpu")
except ImportError:
    pass
except Exception:
    pass


# =============================================================================
# Configuration Functions
# =============================================================================


def get_best_available_device() -> str:
    """Return the best available PennyLane device name.

    Priority: lightning.gpu > lightning.qubit > default.qubit
    """
    if HAS_LIGHTNING_GPU:
        return "lightning.gpu"
    elif HAS_LIGHTNING_QUBIT:
        return "lightning.qubit"
    return "default.qubit"


def get_catalyst_effective_backend() -> str:
    """Get the actual execution backend for Catalyst @qjit.

    IMPORTANT: Catalyst uses JAX as its backend, which is SEPARATE from
    PennyLane device selection. Even with lightning.gpu device, Catalyst
    runs on CPU if JAX lacks CUDA support.

    Returns:
        Human-readable backend string like "GPU (JAX CUDA)" or "CPU (JAX)"
    """
    if HAS_JAX_CUDA:
        return "GPU (JAX CUDA)"
    return "CPU (JAX)"


def create_h3o_geometry() -> list[Atom]:
    """
    Create H3O+ (hydronium) molecular geometry.

    Returns optimized structure with formal charges for QM/MM calculation.
    O: -2.0 formal charge, H: +1.0 each -> total charge +1
    """
    return [
        Atom("O", np.array([0.000000, 0.000000, 0.000000]), charge=-2.0),
        Atom("H", np.array([0.960000, 0.000000, 0.000000]), charge=1.0),
        Atom("H", np.array([-0.480000, 0.831384, 0.000000]), charge=1.0),
        Atom("H", np.array([-0.480000, -0.831384, 0.000000]), charge=1.0),
    ]


def get_qpe_config(device_type: str = "auto") -> dict:
    """
    Get standard QPE configuration for H3O+ system.

    Active space: 4 electrons, 4 orbitals -> 8 system qubits
    Estimation qubits: 4 (precision bits)
    Total qubits: 12

    Args:
        device_type: Device selection strategy
            - "auto": Auto-select best (GPU > lightning.qubit > default.qubit)
            - "lightning.gpu": Force GPU device
            - "lightning.qubit": Force CPU lightning device
            - "default.qubit": Force standard PennyLane device
    """
    return {
        "use_real_qpe": True,
        "n_estimation_wires": 4,
        "base_time": "auto",  # Auto-compute to avoid phase overflow
        "n_trotter_steps": 10,
        "n_shots": 100,
        "active_electrons": 4,
        "active_orbitals": 4,
        "energy_warning_threshold": 1.0,
        "algorithm": "standard",
        "mapping": "jordan_wigner",
        "device_type": device_type,
    }
