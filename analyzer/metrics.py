"""Mechanical aim metrics derived from an angular motion trace.

Everything here operates on the output of `analyzer.motion`: a per-frame yaw /
pitch trace in degrees. These are the numbers the AI layer is later told to
treat as ground truth and never recompute, so correctness here matters more
than anywhere else in the pipeline.

Each metric carries its own uncertainty where sampling rate bounds it. A
reaction time reported without its error bar invites the user to read frame
noise as a real difference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from scipy import signal

# Voluntary human aim adjustment lives below ~5 Hz. Energy above this band is
# tremor, mouse sensor noise, or grip tension — mechanically distinct from
# "bad aim", and worth separating because the corrective advice differs.
JITTER_BAND_HZ = (8.0, 30.0)

# Angular speed below which the crosshair counts as settled on a target.
SETTLE_THRESHOLD_DPS = 15.0
SETTLE_FRAMES = 3


@dataclass(frozen=True)
class Trace:
    """A resampled angular motion trace. Angles are absolute degrees from the
    trace origin, not per-frame deltas."""

    t_ms: np.ndarray
    yaw_deg: np.ndarray
    pitch_deg: np.ndarray
    fps: float

    def __post_init__(self) -> None:
        n = len(self.t_ms)
        if not (len(self.yaw_deg) == len(self.pitch_deg) == n):
            raise ValueError("trace arrays must share a length")
        if n < 2:
            raise ValueError("trace needs at least 2 samples")

    @property
    def speed_dps(self) -> np.ndarray:
        """Angular speed per sample, in degrees/second."""
        dt = np.diff(self.t_ms, prepend=self.t_ms[0] - 1000.0 / self.fps) / 1000.0
        dyaw = np.diff(self.yaw_deg, prepend=self.yaw_deg[0])
        dpitch = np.diff(self.pitch_deg, prepend=self.pitch_deg[0])
        return np.hypot(dyaw, dpitch) / np.maximum(dt, 1e-9)

    def to_json(self) -> dict:
        """Parallel arrays, matching clip_analysis/v1's `measured.trace`."""
        return {
            "t_ms": [round(float(v), 1) for v in self.t_ms],
            "yaw_deg": [round(float(v), 3) for v in self.yaw_deg],
            "pitch_deg": [round(float(v), 3) for v in self.pitch_deg],
            "speed_dps": [round(float(v), 2) for v in self.speed_dps],
        }


def sparc(speed: Sequence[float], fps: float, *, fc: float = 10.0,
          amp_threshold: float = 0.05, pad_level: int = 4) -> float:
    """Spectral arc length of a speed profile. Less negative means smoother.

    This is the Balasubramanian et al. (2015) formulation rather than a
    home-grown smoothness score, and the choice matters: SPARC is normalized
    for both amplitude and duration, so a 400 deg/s flick and a slow tracking
    pass produce directly comparable numbers. Normalized jerk is not, and would
    simply report that fast movements are worse — which is useless as coaching.

    Returns 0.0 for a movement with no energy (a perfectly still trace).
    """
    v = np.asarray(speed, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 4 or not np.any(v):
        return 0.0

    nfft = int(2 ** (math.ceil(math.log2(v.size)) + pad_level))
    freq = np.arange(0, fps, fps / nfft)
    mag = np.abs(np.fft.fft(v, nfft))
    peak = mag.max()
    if peak <= 0:
        return 0.0
    mag = mag / peak

    in_band = freq <= fc
    if not np.any(in_band):
        return 0.0
    f_sel, m_sel = freq[in_band], mag[in_band]

    # Adaptive cutoff: trim to the span that actually carries energy, so a
    # long tail of near-zero bins cannot inflate the arc length.
    above = np.nonzero(m_sel >= amp_threshold)[0]
    if above.size < 2:
        return 0.0
    f_sel, m_sel = f_sel[above[0]:above[-1] + 1], m_sel[above[0]:above[-1] + 1]

    span = f_sel[-1] - f_sel[0]
    if span <= 0:
        return 0.0
    return float(-np.sum(np.hypot(np.diff(f_sel) / span, np.diff(m_sel))))


def jitter_ratio(speed: Sequence[float], fps: float,
                 band: tuple[float, float] = JITTER_BAND_HZ) -> float:
    """Fraction of speed-profile power sitting in the tremor band.

    Separates "aims badly" from "holds the mouse too tightly / runs too much
    sensitivity" — two problems with completely different corrections, which a
    single smoothness number would conflate.

    Known coupling, measured in tests/test_metrics.py: corrective submovements
    inject genuine broadband energy, so a heavily-corrected flick reads as
    elevated jitter even with a rock-steady hand. Jitter and overshoot are
    therefore correlated, not independent axes. Until the thresholds are
    calibrated against known-skill players, do not present a high jitter_ratio
    as a grip/sensitivity diagnosis on its own — cross-check it against
    overshoot_count first.
    """
    v = np.asarray(speed, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 8:
        return 0.0
    nyquist = fps / 2.0
    if band[0] >= nyquist:
        # Sampling rate cannot see the tremor band at all; report nothing
        # rather than a number derived from aliased energy.
        return float("nan")

    nperseg = min(v.size, 256)
    freqs, psd = signal.welch(v - v.mean(), fs=fps, nperseg=nperseg)
    total = np.trapezoid(psd, freqs)
    if total <= 0:
        return 0.0
    hi, lo = min(band[1], nyquist), band[0]
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    return float(np.clip(np.trapezoid(psd[mask], freqs[mask]) / total, 0.0, 1.0))


def overshoot_count(trace: Trace, shot_idx: int, *, max_window_ms: float = 600.0,
                    quiet_dps: float = 20.0) -> int:
    """Corrective direction reversals in the approach to a shot.

    The classic high-sensitivity failure: the flick passes the target, comes
    back, passes it again. Counted as sign changes in the yaw velocity, which
    is where overshoot actually shows up — vertical correction is usually
    recoil, not aim error.

    The window runs from the movement's onset — the last moment the crosshair
    was at rest before the shot — rather than over a fixed interval. A fixed
    150ms window captures only the tail of the approach, so a player who
    flicks, overshoots by 11 degrees, corrects, settles, and only then fires
    scores a clean zero. That is the exact mistake this metric exists to
    catch, and it is invisible on any clip where the trigger pull is not
    immediate.
    """
    if shot_idx <= 0 or shot_idx >= len(trace.t_ms):
        return 0

    t_end = trace.t_ms[shot_idx]
    speed = trace.speed_dps

    def in_budget(i: int) -> bool:
        return i > 0 and t_end - trace.t_ms[i] < max_window_ms

    quiet = speed < quiet_dps
    # Rest means *sustained* quiet, not one slow sample. Velocity passes
    # through zero at every direction reversal, so a single-sample test treats
    # the overshoot itself as the start of the movement and clips the window to
    # just after the mistake — reporting zero reversals on a trace that
    # visibly has two.
    # 150ms, because the pause that matters is the one separating engagements,
    # not the one between submovements. Corrective submovements are themselves
    # separated by genuine 20-50ms lulls, so a short threshold finds "rest" in
    # the middle of the correction sequence and clips the window to the last
    # submovement — again reporting zero on a trace with two clear reversals.
    quiet_run = max(2, int(0.150 * trace.fps))

    # Step back over the settled tail first. A well-executed shot is fired once
    # the crosshair has stopped, so at shot_idx the speed is usually already
    # quiet; walking back only while moving would stop immediately and leave an
    # empty window.
    i = shot_idx
    while in_budget(i) and quiet[i]:
        i -= 1

    # Then back through the movement to the last sustained rest before it.
    run = 0
    while in_budget(i):
        run = run + 1 if quiet[i] else 0
        if run >= quiet_run:
            break
        i -= 1

    window = (trace.t_ms >= trace.t_ms[i]) & (trace.t_ms <= t_end)
    yaw = trace.yaw_deg[window]
    if yaw.size < 3:
        return 0

    vel = np.diff(yaw)
    # Ignore near-stationary samples: noise around zero would otherwise be
    # counted as dozens of reversals.
    moving = np.abs(vel) > (np.abs(vel).max() * 0.08 if np.any(vel) else 0)
    vel = vel[moving]
    if vel.size < 2:
        return 0
    return int(np.count_nonzero(np.diff(np.sign(vel)) != 0))


def path_efficiency(trace: Trace, start_idx: int = 0, end_idx: int | None = None) -> float:
    """Straight-line angular distance divided by distance actually travelled.

    1.0 is a perfect direct flick; an oscillating correction drags it down.
    """
    end_idx = len(trace.t_ms) - 1 if end_idx is None else end_idx
    if end_idx <= start_idx:
        return 0.0
    yaw = trace.yaw_deg[start_idx:end_idx + 1]
    pitch = trace.pitch_deg[start_idx:end_idx + 1]
    travelled = float(np.sum(np.hypot(np.diff(yaw), np.diff(pitch))))
    if travelled <= 1e-9:
        return 0.0
    direct = float(math.hypot(yaw[-1] - yaw[0], pitch[-1] - pitch[0]))
    return float(np.clip(direct / travelled, 0.0, 1.0))


def reaction_time_ms(stimulus_frame: int, shot_frame: int, fps: float) -> tuple[float, float]:
    """(reaction_time, uncertainty) in milliseconds.

    The uncertainty is half a frame interval and is not optional. At 60 fps it
    is +/-8.3 ms per boundary, so callers must render it: without it a user
    reads a 12 ms session-over-session change as real improvement when it is
    entirely sampling noise. At 30 fps the error swamps the differences that
    matter, which is why `analyze` refuses to report it there at all.
    """
    if shot_frame < stimulus_frame:
        raise ValueError("shot cannot precede the stimulus")
    frame_ms = 1000.0 / fps
    return (shot_frame - stimulus_frame) * frame_ms, frame_ms


def target_switch_ms(trace: Trace, from_idx: int,
                     *, threshold_dps: float = SETTLE_THRESHOLD_DPS,
                     settle_frames: int = SETTLE_FRAMES) -> float | None:
    """Time from one kill until the crosshair settles on the next target.

    "Settled" is defined explicitly — angular speed below threshold for N
    consecutive frames — because without a stated criterion the number is not
    comparable between two clips, let alone two players.
    """
    speed = trace.speed_dps
    if from_idx >= len(speed) - settle_frames:
        return None
    run = 0
    for i in range(from_idx + 1, len(speed)):
        run = run + 1 if speed[i] < threshold_dps else 0
        if run >= settle_frames:
            settled_at = i - settle_frames + 1
            return float(trace.t_ms[settled_at] - trace.t_ms[from_idx])
    return None


def crosshair_placement_error_deg(trace: Trace, horizon_deg: float = 0.0) -> float:
    """Mean absolute pitch deviation from the horizon.

    An explicit 2D proxy, not true head-level error: without scene depth you
    cannot know where a head would have been. The pipeline records
    `crosshair_placement_method` alongside this value so that rows produced by
    a later depth-aware version stay distinguishable in the trend chart
    instead of being silently averaged together.
    """
    return float(np.mean(np.abs(trace.pitch_deg - horizon_deg)))


@dataclass(frozen=True)
class ClipMetrics:
    """Populates `measured` in clip_analysis/v1."""

    reaction_time_ms: float | None
    reaction_time_error_ms: float | None
    time_to_first_shot_ms: float | None
    shots_fired: int
    shots_hit: int
    headshot_pct: float | None
    smoothness_sparc: float | None
    jitter_ratio: float | None
    jitter_band_hz: tuple[float, float]
    overshoot_count: int
    peak_angular_error_deg: float | None
    crosshair_placement_error_deg: float | None
    crosshair_placement_method: str
    target_switch_ms: float | None
    path_efficiency: float | None
    cv_confidence: float
    cv_warnings: tuple[str, ...]

    def to_json(self, trace: Trace) -> dict:
        d = asdict(self)
        d["jitter_band_hz"] = list(self.jitter_band_hz)
        d["cv_warnings"] = list(self.cv_warnings)
        d["trace"] = trace.to_json()
        # NaN is not valid JSON and silently becomes `null` in some encoders
        # and a parse error in others. Normalize once, here.
        for k, v in d.items():
            if isinstance(v, float) and not math.isfinite(v):
                d[k] = None
        return d


def analyze(trace: Trace, *, stimulus_frame: int | None = None,
            shot_frames: Sequence[int] = (), hits: int = 0, headshots: int = 0,
            horizon_deg: float = 0.0) -> ClipMetrics:
    """Compute every mechanical metric for one engagement."""
    speed = trace.speed_dps
    warnings: list[str] = []
    confidence = 1.0

    rt = rt_err = ttfs = None
    if stimulus_frame is not None and shot_frames:
        if trace.fps < 50:
            # Below ~50fps the quantization error is comparable to the
            # differences being measured, so a number here would imply
            # precision the footage cannot support.
            warnings.append(
                f"fps={trace.fps:.0f} too low for reaction time; "
                f"quantization +/-{1000.0 / trace.fps / 2:.1f}ms exceeds useful resolution"
            )
            confidence -= 0.25
        else:
            rt, rt_err = reaction_time_ms(stimulus_frame, shot_frames[0], trace.fps)
            ttfs = rt
            warnings.append(
                f"fps={trace.fps:.0f} limits reaction_time resolution to +/-{rt_err:.1f}ms"
            )

    jr = jitter_ratio(speed, trace.fps)
    if math.isnan(jr):
        warnings.append(
            f"fps={trace.fps:.0f} cannot resolve the {JITTER_BAND_HZ[0]:.0f}Hz "
            f"jitter band (Nyquist {trace.fps / 2:.0f}Hz)"
        )
        confidence -= 0.2

    shot_idx = shot_frames[0] if shot_frames else len(trace.t_ms) - 1
    peak_err = float(np.max(np.abs(trace.yaw_deg))) if trace.yaw_deg.size else None
    shots = len(shot_frames)

    return ClipMetrics(
        reaction_time_ms=rt,
        reaction_time_error_ms=rt_err,
        time_to_first_shot_ms=ttfs,
        shots_fired=shots,
        shots_hit=hits,
        headshot_pct=(headshots / hits) if hits else None,
        smoothness_sparc=sparc(speed, trace.fps),
        jitter_ratio=jr,
        jitter_band_hz=JITTER_BAND_HZ,
        overshoot_count=overshoot_count(trace, shot_idx),
        peak_angular_error_deg=peak_err,
        crosshair_placement_error_deg=crosshair_placement_error_deg(trace, horizon_deg),
        crosshair_placement_method="horizon_proxy_2d",
        target_switch_ms=target_switch_ms(trace, shot_idx),
        path_efficiency=path_efficiency(trace, 0, shot_idx),
        cv_confidence=round(max(0.0, confidence), 2),
        cv_warnings=tuple(warnings),
    )
