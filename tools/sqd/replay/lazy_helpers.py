# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""PEP 562 lazy-export helpers for optional backends.

``q2m3`` ships optional extras (``sqd`` -> ffsim + qiskit-addon-sqd,
``catalyst`` -> catalyst + jax). Packages that re-export symbols from those
backends must stay importable when the extra is absent, while keeping
``__all__``, ``dir()`` and ``from ... import *`` mutually consistent.

This module provides the two primitives used by every such ``__init__.py``:
``available_exports`` (filter ``__all__`` without importing any backend) and
``lazy_getattr`` (import on first attribute access, raising an actionable
``ImportError`` when the extra is missing).
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Final

#: extra name -> modules that must be importable for the extra to be usable.
EXTRA_PROBES: Final[dict[str, tuple[str, ...]]] = {
    "sqd": ("ffsim", "qiskit_addon_sqd"),
    "catalyst": ("catalyst", "jax"),
}


def extra_available(extra: str) -> bool:
    """Report whether an optional extra is installed, without importing it.

    Args:
        extra: Extra name declared in ``pyproject.toml`` (e.g. ``"sqd"``).

    Returns:
        True if every probe module of the extra can be located on ``sys.path``.
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
    """Resolve a lazily exported name, caching it in the caller's namespace.

    Args:
        package: ``__name__`` of the calling package.
        namespace: ``globals()`` of the calling package (used as the cache).
        lazy: Mapping ``name -> (module_path, extra)``.
        name: Attribute being looked up.

    Returns:
        The resolved object.

    Raises:
        AttributeError: ``name`` is not a known export of ``package``.
        ImportError: The backing extra is not installed; the message carries
            the exact ``uv sync --extra <extra>`` command that fixes it.
    """
    entry = lazy.get(name)
    if entry is None:
        raise AttributeError(f"module {package!r} has no attribute {name!r}")
    module_path, extra = entry
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"{package}.{name} requires the optional {extra!r} backend, which is not "
            f"installed. Install it with:  uv sync --extra {extra}"
        ) from exc
    value = getattr(module, name)
    namespace[name] = value
    return value


def available_exports(lazy: dict[str, tuple[str, str]]) -> list[str]:
    """Return the lazy export names whose backing extra is installed.

    Appending the result to a literal ``__all__`` keeps ``__all__`` a subset of
    ``dir(module)`` in every install profile, so ``from q2m3 import *`` never
    raises while the literal part stays visible to ruff/mypy.

    Args:
        lazy: Mapping ``name -> (module_path, extra)``.

    Returns:
        Names that can actually be resolved in the current environment.
    """
    return [name for name, (_module, extra) in lazy.items() if extra_available(extra)]
