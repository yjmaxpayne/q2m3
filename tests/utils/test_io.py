"""Strict JSON serialization of calculation results and immutable snapshots."""

import json
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pytest

from q2m3.utils.io import save_json_results


@dataclass(frozen=True)
class Snapshot:
    diagnostics: object


def read_strict(path):
    def reject_constant(value):
        raise AssertionError(f"Non-standard JSON constant: {value}")

    return json.loads(path.read_text(), parse_constant=reject_constant)


def test_dataclass_nested_immutable_mapping_and_numpy_round_trip(tmp_path):
    snapshot = Snapshot(
        MappingProxyType({"nested": Snapshot((np.array([1, 2]), np.float64(0.5), np.bool_(True)))})
    )
    path = tmp_path / "nested" / "results.json"
    save_json_results(snapshot, path)
    assert read_strict(path) == {"diagnostics": {"nested": {"diagnostics": [[1, 2], 0.5, True]}}}


def test_mapping_and_complex_array_preserve_complex_convention(tmp_path):
    path = tmp_path / "results.json"
    save_json_results(MappingProxyType({"values": np.array([1 + 2j, 3 - 4j])}), path)
    assert read_strict(path) == {
        "values": [{"real": 1.0, "imag": 2.0}, {"real": 3.0, "imag": -4.0}]
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), np.float64("nan")])
def test_nonfinite_numbers_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="Out of range float"):
        save_json_results({"diagnostics": (np.array([value]),)}, tmp_path / "bad.json")


def test_finite_dictionary_preserves_existing_representation(tmp_path):
    path = tmp_path / "results.json"
    save_json_results(
        {"count": np.int64(3), "amplitude": 1 - 2j, "unknown": None, "flags": (True, False)},
        str(path),
    )
    assert read_strict(path) == {
        "count": 3,
        "amplitude": {"real": 1.0, "imag": -2.0},
        "unknown": None,
        "flags": [True, False],
    }


def test_dataclass_class_is_not_serialized_as_instance(tmp_path):
    with pytest.raises(TypeError):
        save_json_results(Snapshot, tmp_path / "bad.json")


def test_resource_dataclass_round_trip(tmp_path):
    from q2m3.core.resource_estimation import EFTQCResources

    result = EFTQCResources(1.5, 8, 100, None, 0.001, 4, "sto-3g", 0)
    path = tmp_path / "resources.json"
    save_json_results({"resources": result}, path)
    assert read_strict(path) == {
        "resources": {
            "hamiltonian_1norm": 1.5,
            "logical_qubits": 8,
            "toffoli_gates": 100,
            "n_terms": None,
            "target_error": 0.001,
            "n_system_qubits": 4,
            "basis": "sto-3g",
            "n_mm_charges": 0,
            "embedding_mode": "none",
            "embedding_diagnostics": None,
            "walk_operator_calls": None,
        }
    }
