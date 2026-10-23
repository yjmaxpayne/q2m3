"""Cross-process encoding of SQD records."""

import pickle
import threading
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import MappingProxyType

import numpy as np
import pytest

from q2m3.molecule import MoleculeConfig
from q2m3.sqd.config import IntegralContext, LUCJConfig, ReferenceConfig, SQDConfig
from q2m3.sqd.exceptions import ReferenceTimeoutError
from q2m3.sqd.result import SQDResult
from q2m3.sqd.transport import decode_result, encode_result
from tests.sqd.test_result import result_data


def molecule():
    return MoleculeConfig(
        name="H2",
        symbols=["H", "H"],
        coords=[[0, 0, 0], [0, 0, 0.7]],
        charge=0,
        active_electrons=2,
        active_orbitals=2,
    )


def context():
    return IntegralContext(
        frame_id="rhf",
        hamiltonian_id="a" * 64,
        active_indices=(0, 1),
        n_core_orbitals=0,
        vacuum_core_constant=0.5,
        delta_core_constant=0.0,
        embedding_mode="vacuum",
        fixed_mo=True,
        two_electron_tensor_fixed=True,
        hf_reference_kind="canonical_rhf",
        source={"nested": np.array([1.0, 2.0]), "labels": ("a", "b")},
    )


def assert_same(actual, expected):
    """Compare values and container types, so tuple/list swaps are visible."""
    if is_dataclass(expected):
        assert type(actual) is type(expected)
        for f in fields(expected):
            assert_same(getattr(actual, f.name), getattr(expected, f.name))
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping) and actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, tuple | list):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert_same(a, e)
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    else:
        assert type(actual) is type(expected) and actual == expected


class FlipsAfter:
    """Cancellation flag that turns on after the first ``n`` checks."""

    def __init__(self, n):
        self.n = n

    def is_set(self):
        self.n -= 1
        return self.n < 0


def test_sqd_result_round_trip_keeps_all_forty_fields():
    result = SQDResult(**result_data())
    decoded = decode_result(encode_result(result))
    assert type(decoded) is SQDResult
    assert len(fields(decoded)) == 40
    assert_same(decoded, result)


@pytest.mark.parametrize(
    "record", [context(), LUCJConfig(), ReferenceConfig()], ids=lambda r: type(r).__name__
)
def test_config_records_round_trip_through_pickle(record):
    payload = pickle.loads(pickle.dumps(encode_result(record)))
    decoded = decode_result(payload)
    assert_same(decoded, record)
    if isinstance(record, IntegralContext):
        # Encoding thaws the snapshot; the record constructor must freeze it again.
        assert isinstance(payload[3]["source"][1], dict)
        assert isinstance(decoded.source, MappingProxyType)


def test_plain_containers_keep_their_types():
    # Record constructors normalize sequences, so only bare containers expose the tags.
    value = {"pair": (1, [2.0, "x"]), "empty": []}
    assert_same(decode_result(encode_result(value)), value)


def test_cancelled_reconstruction_raises_timeout():
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(ReferenceTimeoutError, match="SQD result reconstruction cancelled"):
        decode_result(encode_result(LUCJConfig()), cancelled=cancelled)


def test_cancellation_is_checked_at_nested_nodes():
    # The root check passes; the flag turns on before the first field is rebuilt.
    with pytest.raises(ReferenceTimeoutError):
        decode_result(encode_result(SQDResult(**result_data())), cancelled=FlipsAfter(1))


def test_records_outside_sqd_package_are_rejected():
    payload = encode_result(molecule())
    assert payload[:3] == ("record", "q2m3.molecule", "MoleculeConfig")
    with pytest.raises(ValueError, match="q2m3.molecule"):
        decode_result(payload)


def test_sqd_config_cannot_be_rebuilt_because_molecule_is_outside_sqd():
    payload = encode_result(SQDConfig(molecule()))
    assert payload[3]["molecule"][1] == "q2m3.molecule"
    with pytest.raises(ValueError, match="q2m3.molecule"):
        decode_result(payload)


def test_orchestrator_reader_forwards_its_cancellation_flag():
    from q2m3.sqd import orchestrator

    cancelled = threading.Event()
    cancelled.set()
    orchestrator._transport_state.cancelled = cancelled
    try:
        with pytest.raises(ReferenceTimeoutError):
            orchestrator._unwire(orchestrator._wire(LUCJConfig()))
    finally:
        del orchestrator._transport_state.cancelled
