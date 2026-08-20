"""Turn a timestamp inside a recording into one annotated frame-grid image.

This is the single biggest cost lever in the pipeline. Nine frames sent as nine
images cost roughly nine images' worth of tokens; tiled into one 3x3 JPEG they
cost roughly one. Burning the relative timestamp into each tile preserves the
model's ability to reason about ordering *and* gives it a coordinate system it
can cite, which makes its claims checkable instead of merely plausible.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass

# cv2 / PIL are heavy; import lazily so the module can be imported for typing
# and unit tests on machines without them.
try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in bare environments
    cv2 = None
    np = None


@dataclass(frozen=True)
class FrameGrid:
    """A tiled contact sheet plus the timestamps that were burned into it."""

    jpeg: bytes
    tile_timestamps_s: list[float]
    rows: int
    cols: int
    tile_px: int

    @property
    def mime_type(self) -> str:
        return "image/jpeg"


def build_frame_grid(
    video_path: str,
    start_ms: int,
    end_ms: int,
    *,
    rows: int = 3,
    cols: int = 3,
    tile_px: int = 512,
    jpeg_quality: int = 82,
) -> FrameGrid:
    """Sample rows*cols frames evenly across [start_ms, end_ms] and tile them.

    Frames are seeked by millisecond rather than by index: OBS captures are
    frequently variable-frame-rate, and frame indices on a VFR file do not map
    linearly to time.
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is required to build frame grids")
    if end_ms <= start_ms:
        raise ValueError(f"end_ms ({end_ms}) must be after start_ms ({start_ms})")

    n = rows * cols
    span_ms = end_ms - start_ms
    # Sample at tile centres, not edges: the first and last frame of a trimmed
    # clip are the least informative ones.
    sample_points_ms = [start_ms + span_ms * (i + 0.5) / n for i in range(n)]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    tiles: list["np.ndarray"] = []
    stamps: list[float] = []
    try:
        for ts_ms in sample_points_ms:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts_ms)
            ok, frame = cap.read()
            if not ok:
                # A failed seek near the tail is normal; repeat the last good
                # tile rather than shifting every subsequent timestamp.
                if not tiles:
                    raise RuntimeError(f"no readable frame at {ts_ms:.0f}ms")
                frame = tiles[-1].copy()
            tile = _fit_square(frame, tile_px)
            rel_s = (ts_ms - start_ms) / 1000.0
            _burn_timestamp(tile, rel_s)
            tiles.append(tile)
            stamps.append(round(rel_s, 2))
    finally:
        cap.release()

    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    ok, buf = cv2.imencode(".jpg", grid, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")

    return FrameGrid(
        jpeg=buf.tobytes(),
        tile_timestamps_s=stamps,
        rows=rows,
        cols=cols,
        tile_px=tile_px,
    )


def _fit_square(frame: "np.ndarray", size: int) -> "np.ndarray":
    """Letterbox to a square without distorting aspect ratio.

    Stretching a 16:9 frame into a square would skew every angular judgement the
    model makes about the crosshair, so pad instead of squash.
    """
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(
        frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
    )
    rh, rw = resized.shape[:2]
    canvas = np.zeros((size, size, 3), dtype=resized.dtype)
    y0, x0 = (size - rh) // 2, (size - rw) // 2
    canvas[y0:y0 + rh, x0:x0 + rw] = resized
    return canvas


def _burn_timestamp(tile: "np.ndarray", rel_s: float) -> None:
    """Draw '1.33s' into the tile corner, in place, with a legible backing."""
    label = f"{rel_s:.2f}s"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    cv2.rectangle(tile, (6, 6), (14 + tw, 18 + th), (0, 0, 0), -1)
    cv2.putText(tile, label, (10, 14 + th), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def estimate_image_tokens(grid: FrameGrid) -> int:
    """Rough shared-order-of-magnitude estimate for the pre-flight cost dialog.

    Vision models tile images into patches of roughly 28-32px, so token count
    scales with area. This is deliberately an over-estimate: showing a user a
    number slightly above what they are charged is a good failure direction.
    """
    w = grid.cols * grid.tile_px
    h = grid.rows * grid.tile_px
    return int(math.ceil(w / 28) * math.ceil(h / 28) * 1.15)
