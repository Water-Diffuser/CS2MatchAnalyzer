"""Synthetic gameplay footage with known ground-truth camera motion.

The motion estimator is the one component whose output cannot be eyeballed:
a trace that looks plausible and a trace that is correct are indistinguishable
by inspection. Rendering footage from a known angular path is the only way to
measure the estimator's actual error, so this generator is test infrastructure
rather than a demo — it is what the accuracy claims in the docs rest on.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def make_world(width: int = 5000, height: int = 2200, seed: int = 11) -> np.ndarray:
    """A large, richly-textured backdrop for a viewport to pan across.

    Needs enough corner features at varied scales that Shi-Tomasi has
    something to lock onto anywhere in the frame — real game environments are
    far more textured than a synthetic gradient, so an under-textured world
    would make the test easier than reality rather than harder.
    """
    rng = np.random.default_rng(seed)
    world = rng.integers(28, 58, (height, width, 3), dtype=np.uint8)

    for _ in range(1400):
        x, y = rng.integers(0, width), rng.integers(0, height)
        w, h = rng.integers(18, 190), rng.integers(18, 190)
        colour = tuple(int(c) for c in rng.integers(35, 215, 3))
        cv2.rectangle(world, (x, y), (x + w, y + h), colour, -1)
    for _ in range(700):
        x, y = rng.integers(0, width), rng.integers(0, height)
        colour = tuple(int(c) for c in rng.integers(60, 245, 3))
        cv2.circle(world, (x, y), int(rng.integers(6, 46)), colour, -1)
    for _ in range(500):
        p1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        p2 = (p1[0] + int(rng.integers(-160, 160)), p1[1] + int(rng.integers(-160, 160)))
        colour = tuple(int(c) for c in rng.integers(90, 255, 3))
        cv2.line(world, p1, p2, colour, int(rng.integers(1, 5)))

    return cv2.GaussianBlur(world, (3, 3), 0)


def _draw_hud(frame: np.ndarray) -> None:
    """Static HUD furniture — ammo, health, minimap, kill feed.

    These are pinned to the screen, so their corners are perfectly stationary
    no matter how the camera moves. Unmasked, they are a powerful signal
    pulling the estimate toward zero, which is precisely why the estimator
    masks them and why the test draws them.
    """
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (int(w * .86), int(h * .88)), (int(w * .97), int(h * .96)), (18, 18, 18), -1)
    cv2.putText(frame, "24/90", (int(w * .87), int(h * .94)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2)
    cv2.rectangle(frame, (int(w * .04), int(h * .88)), (int(w * .18), int(h * .95)), (18, 18, 18), -1)
    cv2.putText(frame, "100", (int(w * .05), int(h * .94)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 235, 180), 2)
    cv2.rectangle(frame, (int(w * .02), int(h * .02)), (int(w * .22), int(h * .30)), (24, 26, 30), -1)
    for i in range(6):
        cv2.line(frame, (int(w * .02), int(h * (.02 + i * .047))),
                 (int(w * .22), int(h * (.02 + i * .047))), (70, 78, 88), 1)
    cv2.rectangle(frame, (int(w * .62), int(h * .06)), (int(w * .98), int(h * .13)), (30, 22, 22), -1)
    cv2.putText(frame, "player_a  [AK]  player_b", (int(w * .63), int(h * .105)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 210, 200), 1)


def _draw_crosshair(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        cv2.line(frame, (cx + dx * 6, cy + dy * 6), (cx + dx * 16, cy + dy * 16), (90, 255, 130), 2)


def render_clip(yaw_path_deg: np.ndarray, pitch_path_deg: np.ndarray, *,
                width: int = 1280, height: int = 720, h_fov_deg: float = 103.0,
                world: np.ndarray | None = None, distractor: bool = True,
                hud: bool = True, noise_sigma: float = 2.0,
                seed: int = 3) -> list[np.ndarray]:
    """Render frames following an exact angular path.

    `yaw_path_deg[i]` is the camera's absolute yaw at frame i, so the caller
    holds the ground truth the estimator is scored against.

    With `distractor=True` a large bright block tracks across the frame moving
    *against* the camera. It is sized to cover a substantial share of the
    trackable area deliberately: if RANSAC is doing its job the block's
    coherent, wrong-direction flow is rejected as outlier motion; if it is not,
    the recovered trace is visibly dragged toward the block.
    """
    if len(yaw_path_deg) != len(pitch_path_deg):
        raise ValueError("yaw and pitch paths must match in length")
    world = make_world() if world is None else world
    wh, ww = world.shape[:2]
    focal = (width / 2.0) / math.tan(math.radians(h_fov_deg) / 2.0)
    rng = np.random.default_rng(seed)

    # Grain, drawn once and sampled per frame rather than regenerated.
    # Generating a full frame of Gaussian noise per frame costs ~2.7M draws and
    # dominated the whole test suite's runtime; cropping a larger precomputed
    # field gives per-frame variation for a fraction of the cost.
    noise_field = (rng.normal(0, noise_sigma, (height + 64, width + 64, 3))
                   if noise_sigma > 0 else None)

    # Anchor the viewport centrally so the full path stays inside the world.
    base_x = (ww - width) // 2
    base_y = (wh - height) // 2

    frames: list[np.ndarray] = []
    for i, (yaw, pitch) in enumerate(zip(yaw_path_deg, pitch_path_deg)):
        dx = int(round(focal * math.tan(math.radians(yaw))))
        dy = int(round(focal * math.tan(math.radians(pitch))))
        x0 = int(np.clip(base_x + dx, 0, ww - width))
        y0 = int(np.clip(base_y + dy, 0, wh - height))
        frame = world[y0:y0 + height, x0:x0 + width].copy()

        if distractor:
            # Moves opposite to the camera, at a different rate, so its flow
            # can never be confused with world motion.
            ex = int(width * 0.18 + (i * 9) % int(width * 0.5))
            ey = int(height * 0.30 + 40 * math.sin(i * 0.22))
            cv2.rectangle(frame, (ex, ey), (ex + int(width * 0.24), ey + int(height * 0.42)),
                          (40, 60, 220), -1)
            cv2.rectangle(frame, (ex + 14, ey + 14),
                          (ex + int(width * 0.24) - 14, ey + int(height * 0.42) - 14),
                          (250, 240, 60), 6)

        if hud:
            _draw_hud(frame)
        _draw_crosshair(frame)

        if noise_field is not None:  # sensor / compression grain
            ox, oy = int(rng.integers(0, 64)), int(rng.integers(0, 64))
            frame = np.clip(frame.astype(np.int16) +
                            noise_field[oy:oy + height, ox:ox + width],
                            0, 255).astype(np.uint8)
        frames.append(frame)

    return frames


def flick_path(n_frames: int, fps: float, amplitude_deg: float,
               duration_s: float, overshoot_deg: float = 0.0) -> np.ndarray:
    """A minimum-jerk flick, optionally with a corrective overshoot."""
    t = np.arange(n_frames) / fps

    def mj(tt: np.ndarray, amp: float, dur: float) -> np.ndarray:
        tau = np.clip(tt / dur, 0.0, 1.0)
        return amp * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)

    path = mj(t, amplitude_deg + overshoot_deg, duration_s)
    if overshoot_deg:
        path -= mj(np.clip(t - duration_s, 0, None), overshoot_deg, duration_s * 0.6)
    return path


DEFAULT_HUD_ROIS = [
    (0.62, 0.06, 0.38, 0.22),   # kill feed
    (0.86, 0.88, 0.11, 0.08),   # ammo
    (0.04, 0.88, 0.14, 0.07),   # health
    (0.02, 0.02, 0.20, 0.28),   # minimap
]


def write_video(frames: list[np.ndarray], path: str, fps: float) -> str:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()
    return path
