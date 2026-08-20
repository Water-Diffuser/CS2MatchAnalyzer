"""Accuracy of the camera-motion estimator against rendered ground truth.

Every test renders footage from a known angular path and scores the recovered
trace against it. These numbers are the basis for the pipeline's claim that its
angular metrics are absolute rather than relative.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.motion import build_mask, focal_length_px, px_to_deg, trace_from_frames
from analyzer.synthetic import DEFAULT_HUD_ROIS, flick_path, make_world, render_clip

FPS = 60.0
FOV = 103.0
WORLD = make_world()          # expensive; shared across tests


def recover(yaw_truth, pitch_truth=None, **kw):
    pitch_truth = np.zeros_like(yaw_truth) if pitch_truth is None else pitch_truth
    frames = render_clip(yaw_truth, pitch_truth, h_fov_deg=FOV, world=WORLD, **kw)
    trace, conf = trace_from_frames(frames, FPS, FOV, DEFAULT_HUD_ROIS)
    return trace, conf


def err(trace_vals, truth):
    """Peak absolute deviation in degrees over the whole path."""
    return float(np.max(np.abs(np.asarray(trace_vals) - np.asarray(truth))))


# ── conversion ───────────────────────────────────────────────────────────────

def test_focal_length_matches_known_fov():
    """At 90 degrees horizontal FOV the focal length is exactly half the width."""
    assert abs(focal_length_px(1920, 90.0) - 960.0) < 1e-6
    f = focal_length_px(1280, 103.0)
    assert abs(px_to_deg(f, 0, f)[0] - 45.0) < 1e-9   # 1 focal length = 45 degrees
    print(f"    focal(1280px @103deg) = {f:.1f}px")


def test_focal_length_rejects_impossible_fov():
    for bad in (0.0, 180.0, -30.0):
        try:
            focal_length_px(1920, bad)
            raise AssertionError(f"accepted FOV {bad}")
        except ValueError:
            pass


def test_mask_excludes_hud_and_centre():
    mask = build_mask((720, 1280), DEFAULT_HUD_ROIS)
    assert mask[360, 640] == 0, "screen centre must be excluded"
    assert mask[int(720 * .91), int(1280 * .90)] == 0, "ammo counter must be excluded"
    assert mask[int(720 * .10), int(1280 * .10)] == 0, "minimap must be excluded"
    assert mask[int(720 * .55), int(1280 * .08)] == 255, "open world must stay trackable"


# ── accuracy ─────────────────────────────────────────────────────────────────

def test_recovers_steady_pan():
    truth = np.linspace(0, 24, 40)          # 24 degrees over 40 frames
    trace, conf = recover(truth, distractor=False)
    e = err(trace.yaw_deg, truth)
    assert conf > 0.9, f"confidence {conf:.2f}"
    assert e < 0.25, f"peak error {e:.3f} deg"
    print(f"    24deg pan: peak error {e:.3f}deg, confidence {conf:.2f}")


def test_recovers_fast_flick():
    """A 45 degree flick in 120ms — near the top of what a player produces,
    and the case where large inter-frame displacement stresses LK tracking."""
    truth = flick_path(36, FPS, 45.0, 0.12)
    trace, conf = recover(truth, distractor=False)
    e = err(trace.yaw_deg, truth)
    assert conf > 0.85, f"confidence {conf:.2f}"
    assert e < 0.30, f"peak error {e:.3f} deg"
    print(f"    45deg/120ms flick: peak error {e:.3f}deg, confidence {conf:.2f}")


def test_recovers_yaw_and_pitch_together():
    yaw = np.linspace(0, 18, 36)
    pitch = np.linspace(0, -7, 36)
    trace, conf = recover(yaw, pitch, distractor=False)
    ey, ep = err(trace.yaw_deg, yaw), err(trace.pitch_deg, pitch)
    assert ey < 0.25 and ep < 0.7, f"yaw {ey:.3f}, pitch {ep:.3f}"
    print(f"    combined: yaw error {ey:.3f}deg, pitch error {ep:.3f}deg")


def test_ransac_rejects_a_large_moving_distractor():
    """The load-bearing test for RANSAC.

    A bright block covering roughly a quarter of the frame tracks against the
    camera. Without outlier rejection its coherent flow drags the estimate;
    with it, accuracy should be close to the distractor-free case.
    """
    truth = np.linspace(0, 20, 40)
    clean, _ = recover(truth, distractor=False)
    noisy, conf = recover(truth, distractor=True)

    e_clean, e_noisy = err(clean.yaw_deg, truth), err(noisy.yaw_deg, truth)
    assert e_noisy < 0.30, f"distractor dragged the estimate: {e_noisy:.3f} deg"
    assert e_noisy < e_clean + 0.25, f"distractor cost {e_noisy - e_clean:.3f} deg"
    print(f"    without distractor {e_clean:.3f}deg -> with {e_noisy:.3f}deg "
          f"(cost {e_noisy - e_clean:+.3f}deg, confidence {conf:.2f})")


def test_static_camera_reports_no_motion():
    """A stationary camera behind a moving HUD and a moving enemy must trace
    flat. Drift here would be integrated into every downstream metric."""
    truth = np.zeros(40)
    trace, conf = recover(truth, distractor=True)
    drift = float(np.max(np.abs(trace.yaw_deg)))
    assert drift < 0.35, f"drifted {drift:.3f} deg while stationary"
    print(f"    stationary drift over 40 frames: {drift:.3f}deg")


def test_direction_convention():
    """Content moving left means the camera turned right: positive yaw.
    A sign error here would invert every overshoot and placement reading."""
    trace, _ = recover(np.linspace(0, 15, 30), distractor=False)
    assert trace.yaw_deg[-1] > 10, f"expected positive yaw, got {trace.yaw_deg[-1]:.2f}"
    trace_neg, _ = recover(np.linspace(0, -15, 30), distractor=False)
    assert trace_neg.yaw_deg[-1] < -10, f"expected negative yaw, got {trace_neg.yaw_deg[-1]:.2f}"


def test_no_scale_bias_at_large_angles():
    """Regression guard: angular accumulation.

    A fixed angular step covers more pixels the further off-axis it happens,
    because a camera at angle theta projects to focal*tan(theta). Converting
    each frame's pixel delta to degrees and then summing over-counts by 1.72x
    at 40 degrees off-axis — which inflated exactly the large flicks this tool
    exists to measure. The trace must accumulate in pixel space and convert
    once, so measured amplitude stays proportional across the whole range.
    """
    for amplitude in (10.0, 25.0, 45.0):
        truth = np.linspace(0, amplitude, 40)
        trace, _ = recover(truth, distractor=False)
        scale = trace.yaw_deg[-1] / amplitude
        assert abs(scale - 1.0) < 0.02, \
            f"{amplitude}deg pan recovered at {scale:.4f}x — scale bias is back"
        print(f"    {amplitude:4.0f}deg -> {trace.yaw_deg[-1]:6.3f}deg ({scale:.4f}x)")


def test_gate_admits_fast_motion_but_rejects_clustered_agreement():
    """Regression guard: the confidence gate.

    Fast camera motion sheds trackable features, so a perfectly good flick
    estimate can sit at a 0.23 inlier ratio. Gating on that ratio discarded the
    peak-velocity frames and lost 21 degrees of real motion while the
    underlying measurement was sub-pixel accurate. The gate keys on absolute
    inlier count and spatial spread instead: spread is what actually
    distinguishes world motion from a large mover.
    """
    from analyzer.motion import MIN_INLIER_SPREAD, MotionEstimate

    fast_flick = MotionEstimate(dx_px=127.0, dy_px=0.0, inliers=53, tracked=230, spread=0.27)
    assert fast_flick.ok, "gate rejects an accurate fast-flick frame"
    assert fast_flick.inlier_ratio < 0.25, "fixture no longer represents the failing case"

    clustered = MotionEstimate(dx_px=40.0, dy_px=0.0, inliers=90, tracked=140, spread=0.06)
    assert not clustered.ok, "gate accepts agreement clustered on one moving object"
    print(f"    fast flick (ratio {fast_flick.inlier_ratio:.2f}, spread "
          f"{fast_flick.spread:.2f}) admitted; clustered (spread "
          f"{clustered.spread:.2f} < {MIN_INLIER_SPREAD}) rejected")


# ── degraded input ───────────────────────────────────────────────────────────

def test_untrackable_footage_reports_low_confidence():
    """A flat grey frame — a smoke, a flash, a blank skybox — has nothing to
    track. Reporting low confidence is correct; inventing motion is not."""
    blank = [np.full((720, 1280, 3), 128, np.uint8) for _ in range(20)]
    trace, conf = trace_from_frames(blank, FPS, FOV, DEFAULT_HUD_ROIS)
    assert conf < 0.2, f"confidence {conf:.2f} on untrackable footage"
    assert float(np.max(np.abs(trace.yaw_deg))) < 0.1, "invented motion from nothing"
    print(f"    blank footage: confidence {conf:.2f}, no invented motion")


def test_rejects_too_short_input():
    try:
        trace_from_frames([np.zeros((720, 1280, 3), np.uint8)], FPS, FOV)
        raise AssertionError("accepted a single frame")
    except ValueError:
        pass


def test_metrics_survive_a_real_recovered_trace():
    """End-to-end: rendered footage -> optical flow -> mechanical metrics."""
    from analyzer.metrics import analyze
    truth = flick_path(48, FPS, 32.0, 0.15, overshoot_deg=7.0)
    trace, conf = recover(truth, distractor=True)
    m = analyze(trace, stimulus_frame=4, shot_frames=[30], hits=1)

    assert m.reaction_time_ms is not None
    assert m.smoothness_sparc < 0
    assert 0.0 <= m.path_efficiency <= 1.0
    assert m.overshoot_count >= 1, "the rendered overshoot was not detected"
    print(f"    recovered: rt={m.reaction_time_ms:.0f}±{m.reaction_time_error_ms:.1f}ms "
          f"sparc={m.smoothness_sparc:.2f} overshoot={m.overshoot_count} "
          f"eff={m.path_efficiency:.2f} conf={conf:.2f}")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed.append(name)
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
