# Copyright (c) 2026 Ye Jun <yjmaxpayne@hotmail.com>
# SPDX-License-Identifier: MIT

"""Version resolution helpers for q2m3."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version
from pathlib import Path

_DISTRIBUTION_NAME = "q2m3"
_SCM_FALLBACK_VERSION = "0.1.0.dev0"
_UNKNOWN_VERSION = "0.0.0"


def _version_from_scm() -> str | None:
    try:
        from setuptools_scm import get_version
    except Exception:
        return None

    try:
        return get_version(
            root=Path(__file__).resolve().parents[2],
            fallback_version=_SCM_FALLBACK_VERSION,
        )
    except Exception:
        return None


def get_version() -> str:
    """Resolve the package version from installed metadata or SCM tags.

    Returns:
        The installed package version, SCM-derived version, or ``"0.0.0"`` if
        neither source is available.
    """

    try:
        return metadata_version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        pass

    scm_version = _version_from_scm()
    if scm_version is not None:
        return scm_version

    return _UNKNOWN_VERSION


__version__ = get_version()
