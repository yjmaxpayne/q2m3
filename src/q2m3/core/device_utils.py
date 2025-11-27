# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Quantum device selection and management utilities.

Provides unified device selection logic for QPE and RDM modules,
with GPU acceleration support and Catalyst compatibility.
"""

import warnings
from typing import Literal

import pennylane as qml

# Device availability detection (executed once at module import)
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


def select_device(
    device_type: str,
    n_wires: int,
    use_catalyst: bool = False,
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

    Returns:
        PennyLane device instance
    """
    # Determine fallback device based on Catalyst compatibility
    # Catalyst only supports lightning.qubit and lightning.gpu, not default.qubit
    fallback_device = (
        "lightning.qubit" if (use_catalyst and HAS_LIGHTNING_QUBIT) else "default.qubit"
    )

    if device_type == "auto":
        # Auto-select: GPU > lightning.qubit > default.qubit
        if HAS_LIGHTNING_GPU:
            return qml.device("lightning.gpu", wires=n_wires)
        elif HAS_LIGHTNING_QUBIT:
            return qml.device("lightning.qubit", wires=n_wires)
        else:
            return qml.device("default.qubit", wires=n_wires)

    elif device_type == "lightning.gpu":
        if not HAS_LIGHTNING_GPU:
            warnings.warn(
                f"lightning.gpu not available, falling back to {fallback_device}",
                UserWarning,
                stacklevel=3,
            )
            return qml.device(fallback_device, wires=n_wires)
        return qml.device("lightning.gpu", wires=n_wires)

    elif device_type == "lightning.qubit":
        if not HAS_LIGHTNING_QUBIT:
            warnings.warn(
                "lightning.qubit not available, falling back to default.qubit",
                UserWarning,
                stacklevel=3,
            )
            return qml.device("default.qubit", wires=n_wires)
        return qml.device("lightning.qubit", wires=n_wires)

    else:
        # default.qubit or any unrecognized value
        return qml.device("default.qubit", wires=n_wires)
