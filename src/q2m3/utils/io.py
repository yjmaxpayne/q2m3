# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""
Input/Output utilities for molecular data and results.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..core.qmmm_system import Atom


def load_xyz(filepath: str) -> list[Atom]:
    """
    Load molecular structure from XYZ file.

    Args:
        filepath: Path to XYZ file

    Returns:
        List of Atom objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"XYZ file not found: {filepath}")

    atoms = []
    with open(path) as f:
        lines = f.readlines()

        if len(lines) < 2:
            raise ValueError("Invalid XYZ file format")

        try:
            n_atoms = int(lines[0].strip())
        except ValueError:
            raise ValueError("First line must contain number of atoms")

        # Skip comment line
        for i in range(2, min(2 + n_atoms, len(lines))):
            parts = lines[i].strip().split()
            if len(parts) < 4:
                continue

            symbol = parts[0]
            position = np.array([float(parts[j]) for j in range(1, 4)])

            # Determine charge for H3O+
            charge = 0.0
            if symbol == "O" and n_atoms == 4:  # H3O+ system
                charge = -2.0  # Formal charge on oxygen in H3O+
            elif symbol == "H" and n_atoms == 4:
                charge = 1.0  # Distribute positive charge

            atoms.append(Atom(symbol=symbol, position=position, charge=charge, is_qm=True))

    return atoms


def save_json_results(results: dict[str, Any], filepath: str, indent: int = 2) -> None:
    """
    Save calculation results to JSON file.

    Args:
        results: Dictionary of results
        filepath: Output file path
        indent: JSON indentation level
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    serializable_results = _make_json_serializable(results)

    with open(path, "w") as f:
        json.dump(serializable_results, f, indent=indent)


def load_config(filepath: str) -> dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        filepath: Path to config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    suffix = path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        with open(path) as f:
            return yaml.safe_load(f)
    elif suffix == ".json":
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")


def _make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy arrays and other non-serializable objects.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle numpy boolean types (compatible with NumPy 1.x and 2.x)
    # Note: np.bool_ is the scalar type, check it first before generic bool
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bool):
        return obj
    # Handle numpy integer types
    elif isinstance(obj, np.integer):
        return int(obj)
    # Handle numpy floating point types
    elif isinstance(obj, np.floating):
        return float(obj)
    # Handle other numpy number types
    elif isinstance(obj, np.number):
        return float(obj)
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list | tuple):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj
