"""Picklable encoding of frozen SQD records for cross-process transport."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib import import_module
from typing import Any

from q2m3.sqd.exceptions import ReferenceTimeoutError

_RECORD_PREFIX = "q2m3.sqd."


def encode_result(value: Any) -> tuple:
    """Encode a value as nested tagged tuples that pickle without custom reducers.

    Dataclass records become ``("record", module, name, fields)``. Mappings, including
    read-only snapshots, become plain dictionaries, so the record constructor must
    freeze them again on decode.

    Args:
        value: A dataclass record, mapping, tuple, list or plain value.

    Returns:
        The tagged encoding of ``value``.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return (
            "record",
            type(value).__module__,
            type(value).__name__,
            {f.name: encode_result(getattr(value, f.name)) for f in fields(value)},
        )
    if isinstance(value, Mapping):
        return ("mapping", {k: encode_result(v) for k, v in value.items()})
    if isinstance(value, tuple):
        return ("tuple", tuple(encode_result(v) for v in value))
    if isinstance(value, list):
        return ("list", [encode_result(v) for v in value])
    return ("value", value)


def decode_result(value: tuple, *, cancelled: threading.Event | None = None) -> Any:
    """Rebuild a value encoded by :func:`encode_result`.

    Records are rebuilt through their constructors, so their validation runs again.
    Only record types defined under ``q2m3.sqd.`` are rebuilt. This limits which
    types can be constructed; it does not make untrusted payloads safe. Callers must
    only unpickle payloads that the parent process wrote to its own private
    directory or pipe.

    ``SQDConfig`` cannot be decoded: its ``molecule`` field is a
    ``q2m3.molecule.MoleculeConfig``.

    Args:
        value: The tagged encoding.
        cancelled: Flag checked before each node is rebuilt.

    Returns:
        The rebuilt value.

    Raises:
        ReferenceTimeoutError: If ``cancelled`` is set during reconstruction.
        ValueError: If a record type is defined outside ``q2m3.sqd.``.
    """
    if cancelled is not None and cancelled.is_set():
        raise ReferenceTimeoutError("SQD result reconstruction cancelled")
    kind, *parts = value
    if kind == "record":
        module, name, values = parts
        if not module.startswith(_RECORD_PREFIX):
            raise ValueError(f"Refusing to rebuild record {module}.{name} outside q2m3.sqd")
        return getattr(import_module(module), name)(
            **{k: decode_result(v, cancelled=cancelled) for k, v in values.items()}
        )
    if kind == "mapping":
        return {k: decode_result(v, cancelled=cancelled) for k, v in parts[0].items()}
    if kind in ("tuple", "list"):
        values = [decode_result(v, cancelled=cancelled) for v in parts[0]]
        return tuple(values) if kind == "tuple" else values
    return parts[0]
