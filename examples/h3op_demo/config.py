# Copyright (c) 2025 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Configuration module for H3O+ QPE Demo.

Contains:
- Demo-specific thresholds
- Configuration functions
"""

import numpy as np

from q2m3.constants import (  # noqa: F401 — re-exported for submodules
    HARTREE_TO_KCAL_MOL,
)
from q2m3.core.device_utils import (  # noqa: F401 — re-exported for submodules
    CATALYST_VERSION,
    HAS_CATALYST,
    HAS_JAX_CUDA,
    HAS_LIGHTNING_GPU,
    HAS_LIGHTNING_QUBIT,
    JAX_DEFAULT_BACKEND,
    get_best_available_device,
    get_catalyst_effective_backend,
)
from q2m3.core.qmmm_system import Atom

# =============================================================================
# Demo-specific Thresholds
# =============================================================================

MM_STABILIZATION_THRESHOLD = (
    0.001  # Hartree - minimum stabilization to consider MM embedding active
)
ENERGY_CONSISTENCY_THRESHOLD = 0.01  # Hartree - acceptable difference between methods
CHEMICAL_ACCURACY_ERROR = 0.0016  # Hartree (~1 kcal/mol)
RELAXED_ACCURACY_ERROR = 0.016  # Hartree (~10 kcal/mol)


# =============================================================================
# Configuration Functions
# =============================================================================


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
