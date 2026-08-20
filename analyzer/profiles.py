"""Game profiles: everything title-specific, as data rather than code.

HUD layouts change with patches. Keeping ROIs and detection parameters in YAML
means a restyle is an edit to a data file — shippable as a hot update, or
fixable by a user who can find their own ammo counter — instead of a release.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .resources import profile_dir

Roi = tuple[float, float, float, float]


@dataclass(frozen=True)
class GameProfile:
    game: str
    fov_degrees: float
    rois: dict[str, Roi]
    flow_exclude: list[Roi] = field(default_factory=list)
    shot_signal: str = "ammo_decrement"
    kill_feed_rows: int = 5
    feed_row_uniformity: float = 0.70
    target_hsv: dict[str, list[int]] | None = None

    @classmethod
    def load(cls, name_or_path: str) -> "GameProfile":
        path = Path(name_or_path)
        if not path.exists():
            path = profile_dir() / f"{name_or_path}.yaml"
        if not path.exists():
            available = sorted(p.stem for p in profile_dir().glob("*.yaml"))
            raise FileNotFoundError(f"no profile {name_or_path!r}; have {available}")

        raw = yaml.safe_load(path.read_text())
        rois = {k: tuple(v) for k, v in raw.get("rois", {}).items()}
        for name, roi in rois.items():
            if len(roi) != 4 or not all(0.0 <= v <= 1.0 for v in roi):
                raise ValueError(f"{path.name}: roi {name!r} must be 4 normalized values")
            if roi[0] + roi[2] > 1.0 or roi[1] + roi[3] > 1.0:
                raise ValueError(f"{path.name}: roi {name!r} extends past the frame")

        return cls(
            game=raw["game"],
            fov_degrees=float(raw["fov_degrees"]),
            rois=rois,
            flow_exclude=[tuple(r) for r in raw.get("flow_exclude", [])],
            shot_signal=raw.get("shot_signal", "ammo_decrement"),
            kill_feed_rows=int(raw.get("kill_feed_rows", 5)),
            feed_row_uniformity=float(raw.get("feed_row_uniformity", 0.70)),
            target_hsv=raw.get("target_hsv"),
        )

    @staticmethod
    def available() -> list[str]:
        return sorted(p.stem for p in profile_dir().glob("*.yaml"))
