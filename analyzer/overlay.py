"""Render a recovered motion trace over the footage it came from.

This is the artifact that makes the analysis legible. A table saying
"overshoot_count: 3" is a claim; the crosshair's actual path, drawn over the
frame with the reversals marked, is the evidence — and it shows the player a
mistake that was invisible to them at the time.
"""
from __future__ import annotations

import cv2
import numpy as np

from .metrics import ClipMetrics, Trace
from .motion import focal_length_px

TRAIL_COLOUR = (120, 235, 255)      # BGR amber
OVERSHOOT_COLOUR = (80, 80, 255)    # BGR red
SETTLED_COLOUR = (140, 255, 170)    # BGR green


def draw_trace(frame: np.ndarray, trace: Trace, h_fov_deg: float,
               *, metrics: ClipMetrics | None = None,
               up_to: int | None = None) -> np.ndarray:
    """Draw the crosshair path onto a copy of `frame`.

    The trace is in degrees, so it projects back to pixels through the same
    focal length the estimator used. Drawing it relative to screen centre is
    the whole point: the crosshair never moves on screen, so the only way to
    show where the player was aiming is to show where the world went.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    focal = focal_length_px(w, h_fov_deg)
    cx, cy = w // 2, h // 2

    n = len(trace.t_ms) if up_to is None else min(up_to, len(trace.t_ms))
    if n < 2:
        return out

    # Anchor on the final position so the path shows how the crosshair arrived
    # where it is now, rather than trailing off the frame.
    end_yaw, end_pitch = trace.yaw_deg[n - 1], trace.pitch_deg[n - 1]
    pts = []
    for i in range(n):
        dx = focal * np.tan(np.radians(trace.yaw_deg[i] - end_yaw))
        dy = focal * np.tan(np.radians(trace.pitch_deg[i] - end_pitch))
        pts.append((int(cx - dx), int(cy - dy)))

    # Dark casing first, then the coloured trail on top. Game frames are
    # visually busy — a thin bright line vanishes against light scenery, and
    # this overlay has to stay legible on whatever the player was looking at.
    for i in range(1, n):
        cv2.line(out, pts[i - 1], pts[i], (12, 12, 12), 7, cv2.LINE_AA)

    # Fading tail: recent motion reads brightest, so direction is obvious
    # without an arrowhead cluttering the path.
    for i in range(1, n):
        age = i / n
        colour = tuple(int(c * (0.45 + 0.55 * age)) for c in TRAIL_COLOUR)
        cv2.line(out, pts[i - 1], pts[i], colour, 3, cv2.LINE_AA)

    # Mark direction reversals — the overshoots the metric counts.
    yaw = trace.yaw_deg[:n]
    if yaw.size > 2:
        vel = np.diff(yaw)
        moving = np.abs(vel) > (np.abs(vel).max() * 0.08 if np.any(vel) else 0)
        for i in range(1, len(vel)):
            if moving[i] and moving[i - 1] and np.sign(vel[i]) != np.sign(vel[i - 1]):
                cv2.circle(out, pts[i], 11, (12, 12, 12), -1, cv2.LINE_AA)
                cv2.circle(out, pts[i], 10, OVERSHOOT_COLOUR, 3, cv2.LINE_AA)

    cv2.circle(out, pts[0], 8, (12, 12, 12), -1, cv2.LINE_AA)
    cv2.circle(out, pts[0], 6, (225, 225, 225), -1, cv2.LINE_AA)
    cv2.drawMarker(out, (cx, cy), (12, 12, 12), cv2.MARKER_CROSS, 26, 5, cv2.LINE_AA)
    cv2.drawMarker(out, (cx, cy), SETTLED_COLOUR, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)

    if metrics is not None:
        _draw_readout(out, metrics, trace)
        _draw_sparkline(out, trace, n)
    return out


def _draw_sparkline(frame: np.ndarray, trace: Trace, n: int,
                    *, w: int = 256, h: int = 76, x0: int = 16) -> None:
    """Yaw against time, with reversals marked.

    The spatial trace cannot show an overshoot that happens along a single
    axis: the crosshair flicks past the target and comes back along the same
    line, so the path draws on top of itself and a 45-degree overshoot looks
    identical to a clean 34-degree flick. Plotted against time the reversal is
    unmistakable, which is the whole point of showing it to a player who could
    not see it happen.
    """
    if n < 2:
        return
    y0 = 16 + 14 * 2 + 22 * 6 + 10
    roi = frame[y0:y0 + h, x0:x0 + w]
    if roi.shape[:2] != (h, w):
        return
    panel = np.full(roi.shape, (22, 18, 14), np.uint8)
    cv2.addWeighted(panel, 0.88, roi, 0.12, 0, roi)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (90, 120, 140), 1)

    yaw = trace.yaw_deg[:n]
    lo, hi = float(yaw.min()), float(yaw.max())
    span = max(hi - lo, 1e-6)
    pad = 12
    pts = [
        (int(x0 + pad + (w - 2 * pad) * i / max(1, n - 1)),
         int(y0 + h - pad - (h - 2 * pad) * (yaw[i] - lo) / span))
        for i in range(n)
    ]
    clip = (x0 + 1, y0 + 1, w - 2, h - 2)
    for i in range(1, n):
        ok, a, b = cv2.clipLine(clip, pts[i - 1], pts[i])
        if ok:
            cv2.line(frame, a, b, TRAIL_COLOUR, 2, cv2.LINE_AA)

    vel = np.diff(yaw)
    if vel.size > 1:
        moving = np.abs(vel) > (np.abs(vel).max() * 0.08 if np.any(vel) else 0)
        for i in range(1, len(vel)):
            if moving[i] and moving[i - 1] and np.sign(vel[i]) != np.sign(vel[i - 1]):
                cv2.circle(frame, pts[i], 6, (12, 12, 12), -1, cv2.LINE_AA)
                cv2.circle(frame, pts[i], 5, OVERSHOOT_COLOUR, 2, cv2.LINE_AA)

    cv2.putText(frame, "yaw vs time", (x0 + pad, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 165, 175), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{hi - lo:.0f} deg", (x0 + w - 58, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 165, 175), 1, cv2.LINE_AA)


def _draw_readout(frame: np.ndarray, m: ClipMetrics, trace: Trace) -> None:
    """A compact metric panel. Null values render as a dash rather than being
    hidden — an absent measurement is information the viewer needs."""
    rt = (f"{m.reaction_time_ms:.0f}+-{m.reaction_time_error_ms:.0f}ms"
          if m.reaction_time_ms is not None else "not measurable")
    rows = [
        ("reaction", rt),
        ("smoothness", f"{m.smoothness_sparc:.2f} SPARC" if m.smoothness_sparc else "-"),
        ("jitter", f"{m.jitter_ratio:.3f}" if m.jitter_ratio is not None else "-"),
        ("overshoot", str(m.overshoot_count)),
        ("path eff.", f"{m.path_efficiency:.2f}" if m.path_efficiency is not None else "-"),
        ("peak error", f"{m.peak_angular_error_deg:.1f} deg"
         if m.peak_angular_error_deg is not None else "-"),
    ]
    pad, lh = 14, 22
    box_h = pad * 2 + lh * len(rows)
    roi = frame[16:16 + box_h, 16:272]
    panel = np.full(roi.shape, (22, 18, 14), np.uint8)
    cv2.addWeighted(panel, 0.88, roi, 0.12, 0, roi)
    cv2.rectangle(frame, (16, 16), (272, 16 + box_h), (90, 120, 140), 1)

    for i, (label, value) in enumerate(rows):
        y = 16 + pad + lh * (i + 1) - 6
        cv2.putText(frame, label, (16 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (150, 165, 175), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (16 + pad + 108, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (235, 240, 245), 1, cv2.LINE_AA)


def render_clip_overlay(frames: list[np.ndarray], trace: Trace, h_fov_deg: float,
                        metrics: ClipMetrics | None = None) -> list[np.ndarray]:
    """Overlay a growing trace across a clip, for export or scrubbing."""
    return [draw_trace(f, trace, h_fov_deg, metrics=metrics, up_to=i + 1)
            for i, f in enumerate(frames)]
