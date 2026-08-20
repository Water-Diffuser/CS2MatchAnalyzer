"""Locate bundled data files whether running from source or from a binary.

PyInstaller unpacks a one-file bundle into a temporary directory and points
`sys._MEIPASS` at it, so `Path(__file__).parents[1]` — correct in a source
checkout — resolves inside the extraction directory's package folder instead
of at its root. Every data lookup goes through here so the two cases cannot
drift apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.5.0"


def is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def resource_root() -> Path:
    """Directory that bundled data files sit under."""
    if is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def resource_path(*parts: str) -> Path:
    return resource_root().joinpath(*parts)


def profile_dir() -> Path:
    return resource_path("profiles")


def schema_path() -> Path:
    return resource_path("schemas", "clip_analysis.v1.schema.json")


def runtime_report() -> dict[str, str]:
    """What the binary is actually running, for bug reports and `doctor`."""
    import numpy
    try:
        import cv2
        cv_version = cv2.__version__
    except ImportError:                       # pragma: no cover
        cv_version = "MISSING"
    return {
        "version": __version__,
        "frozen": str(is_frozen()),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "opencv": cv_version,
        "resource_root": str(resource_root()),
        "platform": sys.platform,
    }
