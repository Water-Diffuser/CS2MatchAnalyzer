"""Recover camera motion — and therefore aim motion — from raw frames.

In an FPS the crosshair is nailed to screen centre, so aim movement *is*
camera movement, and camera movement is recoverable from the pixels alone. No
game API, no memory reads, no packet capture: this reads a saved video file
and nothing else, which is the only approach that is unambiguously safe with
respect to anti-cheat.

The estimate is deliberately conservative. Where it cannot separate camera
motion from scene motion it reports low confidence rather than a plausible
number, because everything downstream treats these values as ground truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .metrics import Trace

# Sparse flow is tracked on strong corners only; these are the standard
# Shi-Tomasi / Lucas-Kanade parameters tuned for 1080p game footage.
FEATURE_PARAMS = dict(maxCorners=260, qualityLevel=0.02, minDistance=9, blockSize=7)
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Fraction of frame width/height around centre to exclude. Enemies, muzzle
# flash and hit effects all live here; without this mask the flow estimate is
# dragged toward whatever is being shot at rather than the world.
CENTRE_EXCLUSION = 0.15

# Gate thresholds. The absolute inlier count is the primary signal: 50 points
# agreeing on one translation is a strong estimate no matter how many
# candidates were dropped getting there.
#
# The ratio guard is deliberately loose. Fast camera motion sheds features
# (they leave the frame, they blur), so a good flick estimate can sit at a 0.23
# inlier ratio while being sub-pixel accurate — measured, not assumed, in
# tests/test_motion.py. A tight ratio gate rejects exactly the frames at the
# peak of a flick, silently dropping the largest real motion in the clip.
#
# What the ratio was being asked to catch is a different thing: a scene with
# two competing coherent motions, where RANSAC may lock onto a large moving
# object instead of the world. Spatial spread separates those two cases
# properly — world features are distributed across the whole frame, an object's
# are clustered — so that is gated on directly.
MIN_INLIERS = 24
MIN_INLIER_RATIO = 0.18
MIN_INLIER_SPREAD = 0.13


@dataclass(frozen=True)
class MotionEstimate:
    """Per-frame-pair camera displacement, in PIXELS on the projection plane.

    Deliberately not in degrees. A camera at angle theta projects a world point
    to focal*tan(theta), and tan is not linear, so a fixed angular step covers
    more pixels the further off-axis it happens. Converting each frame's pixel
    delta to degrees independently and then summing therefore over-counts
    exactly where it matters most: measured against rendered ground truth, a
    0.6 deg/frame pan reads 1.01x correct on-axis but 1.72x correct at 40 deg
    off-axis, which would inflate every large flick this tool exists to
    measure. Pixels accumulate linearly; degrees do not. So the trace sums
    pixels and converts the cumulative offset once, in trace_from_frames.
    """

    dx_px: float
    dy_px: float
    inliers: int
    tracked: int
    spread: float = 0.0
    """Normalized spatial dispersion of the inliers, ~0.29 for points spread
    evenly over the frame and well under 0.1 for a single clustered object."""

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.tracked if self.tracked else 0.0

    @property
    def ok(self) -> bool:
        return (self.inliers >= MIN_INLIERS
                and self.inlier_ratio >= MIN_INLIER_RATIO
                and self.spread >= MIN_INLIER_SPREAD)


def focal_length_px(width: int, h_fov_deg: float) -> float:
    """Pinhole focal length in pixels from horizontal field of view.

    Games render a rectilinear projection, so this conversion is exact rather
    than an approximation — which is what makes degrees, not pixels, the unit
    the metrics can be compared across users and resolutions in.
    """
    if not 1.0 < h_fov_deg < 179.0:
        raise ValueError(f"implausible FOV: {h_fov_deg}")
    return (width / 2.0) / math.tan(math.radians(h_fov_deg) / 2.0)


def px_to_deg(dx: float, dy: float, focal_px: float) -> tuple[float, float]:
    """Convert a screen displacement to an angular one.

    Uses arctan rather than the small-angle approximation: a 400 deg/s flick
    covers enough of the frame that the linear approximation is visibly wrong
    at the edges, and it is the fast flicks that matter most here.

    Apply this to a CUMULATIVE offset from the trace origin, never to a single
    frame's delta — see MotionEstimate for why that distinction is worth 70%
    of a large flick's measured amplitude.

    Modelling limit: a camera rotating about its own centre induces a
    homography, not a pure translation, so this inverts exactly only near the
    optical axis and degrades toward frame edges. The translation model is the
    standard MVP approximation and the residual is second-order next to the
    error it replaces; a later version wanting sub-0.1 deg accuracy at wide
    FOV should fit a rotation homography instead.
    """
    return math.degrees(math.atan(dx / focal_px)), math.degrees(math.atan(dy / focal_px))


def build_mask(shape: tuple[int, int], hud_rois: list[tuple[float, float, float, float]] | None = None,
               centre_exclusion: float = CENTRE_EXCLUSION) -> np.ndarray:
    """White where features may be tracked, black where they may not.

    Excludes the HUD (which is static and would anchor the estimate to zero),
    the minimap (which scrolls independently of the world), and the centre
    region (where the thing being shot at is moving on its own).
    """
    h, w = shape
    mask = np.full((h, w), 255, dtype=np.uint8)

    cw, ch = int(w * centre_exclusion), int(h * centre_exclusion)
    cv2.rectangle(mask, ((w - cw) // 2, (h - ch) // 2), ((w + cw) // 2, (h + ch) // 2), 0, -1)

    for rx, ry, rw, rh in (hud_rois or []):
        x0, y0 = int(rx * w), int(ry * h)
        cv2.rectangle(mask, (x0, y0), (x0 + int(rw * w), y0 + int(rh * h)), 0, -1)
    return mask


def estimate_pair(prev_gray: np.ndarray, gray: np.ndarray, mask: np.ndarray,
                  focal_px: float) -> MotionEstimate:
    """Camera displacement between two consecutive frames.

    RANSAC is not optional here. Masking removes the centre, but a player
    strafing across the left third of the screen still contributes a coherent
    block of flow vectors pointing the wrong way; a least-squares fit would
    average them in, while RANSAC rejects them as outliers to the dominant
    (world) motion.
    """
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **FEATURE_PARAMS)
    if prev_pts is None or len(prev_pts) < MIN_INLIERS:
        return MotionEstimate(0.0, 0.0, 0, 0)

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **LK_PARAMS)
    if next_pts is None:
        return MotionEstimate(0.0, 0.0, 0, len(prev_pts))

    good = status.ravel() == 1
    src, dst = prev_pts[good], next_pts[good]
    tracked = int(good.sum())
    if tracked < MIN_INLIERS:
        return MotionEstimate(0.0, 0.0, 0, tracked)

    # Partial affine (translation + rotation + uniform scale) rather than a
    # full homography: the extra degrees of freedom in a homography are not
    # constrained by a pure camera rotation and let it absorb outlier motion
    # that should have been rejected.
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=2000, confidence=0.995
    )
    if matrix is None or inlier_mask is None:
        return MotionEstimate(0.0, 0.0, 0, tracked)

    # Are the agreeing points spread across the frame, or clustered on one
    # object? This is what tells a genuine world estimate from RANSAC having
    # locked onto a large mover.
    keep = inlier_mask.ravel().astype(bool)
    pts = src[keep].reshape(-1, 2)
    h, w = prev_gray.shape[:2]
    spread = float(min(pts[:, 0].std() / w, pts[:, 1].std() / h)) if len(pts) > 1 else 0.0

    # Screen content moving right means the camera turned left, hence negation.
    return MotionEstimate(-float(matrix[0, 2]), -float(matrix[1, 2]),
                          int(keep.sum()), tracked, spread)


def trace_from_frames(frames: list[np.ndarray], fps: float, h_fov_deg: float,
                      hud_rois: list[tuple[float, float, float, float]] | None = None
                      ) -> tuple[Trace, float]:
    """Build a cumulative angular trace from a sequence of frames.

    Returns the trace and a confidence in [0, 1] — the fraction of frame pairs
    that produced a well-conditioned estimate. A clip full of smoke, flashes or
    a blank skybox has nothing to track and will score low; the caller should
    surface that rather than presenting the metrics as if they were solid.
    """
    if len(frames) < 2:
        raise ValueError("need at least 2 frames")

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]
    h, w = grays[0].shape[:2]
    mask = build_mask((h, w), hud_rois)
    focal = focal_length_px(w, h_fov_deg)

    # Accumulate on the projection plane, where displacement is linear.
    cum_x, cum_y = [0.0], [0.0]
    ok_count = 0
    for i in range(1, len(grays)):
        est = estimate_pair(grays[i - 1], grays[i], mask, focal)
        if est.ok:
            ok_count += 1
            cum_x.append(cum_x[-1] + est.dx_px)
            cum_y.append(cum_y[-1] + est.dy_px)
        else:
            # Hold position rather than integrating a bad estimate: a spurious
            # spike would propagate into every cumulative sample after it.
            cum_x.append(cum_x[-1])
            cum_y.append(cum_y[-1])

    # Convert to angle exactly once, from the cumulative offset.
    angles = [px_to_deg(x, y, focal) for x, y in zip(cum_x, cum_y)]

    trace = Trace(
        t_ms=np.arange(len(frames)) * (1000.0 / fps),
        yaw_deg=np.asarray([a[0] for a in angles]),
        pitch_deg=np.asarray([a[1] for a in angles]),
        fps=fps,
    )
    return trace, ok_count / (len(grays) - 1)


def trace_from_video(path: str, start_ms: float, end_ms: float, h_fov_deg: float,
                     hud_rois: list[tuple[float, float, float, float]] | None = None
                     ) -> tuple[Trace, float]:
    """Decode a window of video and trace the camera through it.

    Seeks by millisecond rather than frame index because OBS output is
    frequently variable-frame-rate, where indices do not map linearly to time.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
        frames = []
        while cap.get(cv2.CAP_PROP_POS_MSEC) <= end_ms:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) < 2:
        raise RuntimeError(f"only {len(frames)} frames decoded in [{start_ms}, {end_ms}]ms")
    return trace_from_frames(frames, fps, h_fov_deg, hud_rois)
