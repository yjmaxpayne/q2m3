# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Shared PEP 562 exports for optional backends, probed without importing them."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Final

EXTRA_PROBES: Final[dict[str, tuple[str, ...]]] = {
    "sqd": ("ffsim", "qiskit_addon_sqd"),
    "catalyst": ("catalyst", "jax"),
}


def extra_available(extra: str) -> bool:
    """Check whether every module required by an extra can be located.

    Args:
        extra: Registered optional extra name.

    Returns:
        Whether all probe modules have import specifications. This does not
        guarantee that an installed backend can initialize successfully.

    Raises:
        KeyError: The extra is not registered.
    """
    for module_name in EXTRA_PROBES[extra]:
        try:
            if importlib.util.find_spec(module_name) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def lazy_getattr(
    package: str,
    namespace: dict[str, Any],
    lazy: dict[str, tuple[str, str]],
    name: str,
) -> Any:
    """Resolve an optional export and cache successful lookups in its package.

    Args:
        package: Name of the exporting package.
        namespace: Package globals used to cache the resolved value.
        lazy: Mapping from export name to (module path, required extra).
        name: Requested attribute.

    Returns:
        The named object from the backing module.

    Raises:
        AttributeError: The export is unknown or absent from its backing module.
        ImportError: An extra is missing (with an installation command), or an
            installed backend fails to import (with its original exception).
    """
    entry = lazy.get(name)
    if entry is None:
        raise AttributeError(f"module {package!r} has no attribute {name!r}")
    module_path, extra = entry
    if not extra_available(extra):
        raise ImportError(
            f"{package}.{name} requires the optional {extra!r} backend. "
            f"Install it with: uv sync --extra {extra}"
        )
    value = getattr(importlib.import_module(module_path), name)
    namespace[name] = value
    return value


def available_exports(lazy: dict[str, tuple[str, str]]) -> list[str]:
    """Filter optional exports without loading their backends.

    Args:
        lazy: Mapping from export name to (module path, required extra).

    Returns:
        Names whose extras can be located, for appending to a literal __all__.
    """
    return [name for name, (_, extra) in lazy.items() if extra_available(extra)]
