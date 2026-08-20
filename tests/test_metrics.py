"""Validation of the mechanical metrics against synthetic ground truth.

Each test constructs a trace whose correct answer is known analytically, which
is the only way to tell a working metric from a plausible-looking one. Where a
metric has no absolute reference (SPARC), the test asserts the ordering
property the metric exists to provide.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.metrics import (
    JITTER_BAND_HZ, Trace, analyze, crosshair_placement_error_deg, jitter_ratio,
    overshoot_count, path_efficiency, reaction_time_ms, sparc, target_switch_ms,
)

FPS = 240.0  # high rate so the tremor band is well inside Nyquist


def min_jerk(t: np.ndarray, amplitude: float, duration: float) -> np.ndarray:
    """Canonical minimum-jerk reach position profile — the smoothest possible
    point-to-point movement, and therefore the reference for 'smooth'."""
    tau = np.clip(t / duration, 0.0, 1.0)
    return amplitude * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)


def make_trace(yaw: np.ndarray, pitch: np.ndarray | None = None, fps: float = FPS) -> Trace:
    n = len(yaw)
    return Trace(
        t_ms=np.arange(n) * (1000.0 / fps),
        yaw_deg=yaw,
        pitch_deg=np.zeros(n) if pitch is None else pitch,
        fps=fps,
    )


# ── SPARC ────────────────────────────────────────────────────────────────────

def test_sparc_ranks_smooth_above_segmented():
    """A single min-jerk flick must score smoother than the same total
    displacement delivered as three corrective submovements."""
    t = np.linspace(0, 0.4, int(0.4 * FPS))
    smooth = make_trace(min_jerk(t, 30.0, 0.4))

    seg = np.zeros_like(t)
    for i, (amp, start) in enumerate([(20.0, 0.0), (7.0, 0.14), (3.0, 0.26)]):
        local = np.clip(t - start, 0, None)
        seg += min_jerk(local, amp, 0.12)
    segmented = make_trace(seg)

    s_smooth = sparc(smooth.speed_dps, FPS)
    s_seg = sparc(segmented.speed_dps, FPS)
    assert s_smooth > s_seg, f"smooth {s_smooth:.3f} should exceed segmented {s_seg:.3f}"
    print(f"    smooth={s_smooth:.3f}  segmented={s_seg:.3f}")


def test_sparc_is_amplitude_invariant():
    """The whole reason for choosing SPARC over normalized jerk: a big fast
    flick and a small slow one of the same shape must score the same, or the
    metric just reports that fast movements are worse."""
    t = np.linspace(0, 0.4, int(0.4 * FPS))
    small = sparc(make_trace(min_jerk(t, 5.0, 0.4)).speed_dps, FPS)
    large = sparc(make_trace(min_jerk(t, 90.0, 0.4)).speed_dps, FPS)
    assert abs(small - large) < 0.02, f"{small:.4f} vs {large:.4f} — not amplitude invariant"
    print(f"    5deg={small:.4f}  90deg={large:.4f}")


def test_sparc_handles_still_trace():
    assert sparc(np.zeros(64), FPS) == 0.0


# ── jitter ───────────────────────────────────────────────────────────────────

def test_jitter_separates_tremor_from_smooth_tracking():
    """A clean 2 Hz tracking motion has no business registering as jitter; the
    same motion with a 15 Hz tremor superimposed must."""
    t = np.arange(0, 2.0, 1 / FPS)
    clean = make_trace(8 * np.sin(2 * np.pi * 2.0 * t))
    shaky = make_trace(8 * np.sin(2 * np.pi * 2.0 * t) + 0.6 * np.sin(2 * np.pi * 15.0 * t))

    j_clean = jitter_ratio(clean.speed_dps, FPS)
    j_shaky = jitter_ratio(shaky.speed_dps, FPS)
    assert j_clean < 0.10, f"clean tracking flagged as jittery: {j_clean:.3f}"
    assert j_shaky > j_clean * 3, f"tremor not detected: {j_clean:.3f} -> {j_shaky:.3f}"
    print(f"    clean={j_clean:.4f}  with 15Hz tremor={j_shaky:.4f}")


def test_jitter_refuses_when_band_exceeds_nyquist():
    """At 12 fps the 8 Hz band is unobservable. Returning a number derived from
    aliased energy would be worse than returning nothing."""
    t = np.arange(0, 2.0, 1 / 12.0)
    assert math.isnan(jitter_ratio(make_trace(np.sin(t), fps=12.0).speed_dps, 12.0))


# ── overshoot ────────────────────────────────────────────────────────────────

def test_overshoot_counts_known_reversals():
    """Construct a flick that passes the target, comes back past it, and
    corrects once more: exactly 2 direction reversals."""
    t = np.linspace(0, 0.15, int(0.15 * FPS))
    yaw = min_jerk(t, 40.0, 0.06)                       # overshoot right
    yaw -= min_jerk(np.clip(t - 0.06, 0, None), 12.0, 0.04)   # back left, too far
    yaw += min_jerk(np.clip(t - 0.10, 0, None), 4.0, 0.04)    # correct right
    n = overshoot_count(make_trace(yaw), shot_idx=len(yaw) - 1)
    assert n == 2, f"expected 2 reversals, got {n}"
    print(f"    reversals detected={n}")


def test_overshoot_zero_on_direct_flick():
    t = np.linspace(0, 0.15, int(0.15 * FPS))
    n = overshoot_count(make_trace(min_jerk(t, 40.0, 0.15)), shot_idx=int(0.15 * FPS) - 1)
    assert n == 0, f"direct flick reported {n} overshoots"


def test_overshoot_ignores_noise_around_zero():
    """A stationary trace with sensor noise must not read as many reversals."""
    rng = np.random.default_rng(7)
    t = np.linspace(0, 0.15, int(0.15 * FPS))
    yaw = min_jerk(t, 40.0, 0.15) + rng.normal(0, 0.004, t.size)
    assert overshoot_count(make_trace(yaw), shot_idx=t.size - 1) <= 1


# ── path efficiency ──────────────────────────────────────────────────────────

def test_path_efficiency_unity_for_straight_line():
    yaw = np.linspace(0, 30, 100)
    assert abs(path_efficiency(make_trace(yaw)) - 1.0) < 1e-6


def test_path_efficiency_penalizes_oscillation():
    yaw = np.concatenate([np.linspace(0, 40, 50), np.linspace(40, 28, 25), np.linspace(28, 30, 25)])
    eff = path_efficiency(make_trace(yaw))
    assert 0.4 < eff < 0.75, f"oscillating path scored {eff:.3f}"
    print(f"    oscillating path efficiency={eff:.3f}")


# ── reaction time ────────────────────────────────────────────────────────────

def test_reaction_time_exact_and_carries_error_bar():
    rt, err = reaction_time_ms(stimulus_frame=100, shot_frame=113, fps=60.0)
    assert abs(rt - 13 * (1000 / 60)) < 1e-9
    assert abs(err - 1000 / 60) < 1e-9, "error bar must be one frame interval"
    print(f"    13 frames @60fps = {rt:.1f}ms +/-{err:.1f}ms")


def test_reaction_time_rejects_shot_before_stimulus():
    try:
        reaction_time_ms(120, 100, 60.0)
        raise AssertionError("should have rejected a negative reaction time")
    except ValueError:
        pass


def test_analyze_withholds_reaction_time_at_low_fps():
    """30fps footage cannot support the claim; the pipeline must decline rather
    than publish a number that is half quantization noise."""
    t = np.linspace(0, 0.5, 15)
    m = analyze(make_trace(min_jerk(t, 30, 0.5), fps=30.0), stimulus_frame=2, shot_frames=[9])
    assert m.reaction_time_ms is None
    assert any("too low" in w for w in m.cv_warnings)
    print(f"    30fps -> {m.cv_warnings[0][:64]}…")


# ── target switch ────────────────────────────────────────────────────────────

def test_target_switch_finds_settle_point():
    """Move fast for 0.2s, then hold still: the settle time must land at the
    transition, not at the end of the trace."""
    fast = np.linspace(0, 60, int(0.2 * FPS))
    held = np.full(int(0.3 * FPS), 60.0)
    ms = target_switch_ms(make_trace(np.concatenate([fast, held])), from_idx=0)
    assert ms is not None and 190 < ms < 225, f"settle detected at {ms}ms, expected ~200ms"
    print(f"    settle at {ms:.1f}ms (motion stopped at 200ms)")


def test_target_switch_none_when_never_settles():
    t = np.arange(0, 1.0, 1 / FPS)
    assert target_switch_ms(make_trace(200 * t), from_idx=0) is None


# ── placement + integration ──────────────────────────────────────────────────

def test_placement_error_measures_pitch_deviation():
    n = 100
    err = crosshair_placement_error_deg(make_trace(np.zeros(n), pitch=np.full(n, -4.0)))
    assert abs(err - 4.0) < 1e-9


def test_analyze_emits_json_safe_output():
    """NaN is not valid JSON. A metric that cannot be computed must serialize
    as null, not as a value that breaks the consumer's parser."""
    import json
    t = np.arange(0, 1.0, 1 / 12.0)
    tr = make_trace(np.sin(t), fps=12.0)
    payload = analyze(tr, stimulus_frame=1, shot_frames=[6]).to_json(tr)
    text = json.dumps(payload)
    assert "NaN" not in text and "Infinity" not in text
    assert payload["jitter_ratio"] is None
    assert len(payload["trace"]["t_ms"]) == len(payload["trace"]["speed_dps"])
    json.loads(text)
    print(f"    unresolvable jitter serialized as {payload['jitter_ratio']}")


def test_analyze_full_engagement():
    """End-to-end on a realistic overshooting flick at 240fps."""
    t = np.linspace(0, 0.5, int(0.5 * FPS))
    yaw = min_jerk(t, 35.0, 0.18)
    yaw -= min_jerk(np.clip(t - 0.18, 0, None), 9.0, 0.08)
    yaw += min_jerk(np.clip(t - 0.28, 0, None), 3.0, 0.06)
    tr = make_trace(yaw, pitch=np.full(t.size, -2.5))
    m = analyze(tr, stimulus_frame=10, shot_frames=[int(0.36 * FPS)], hits=1, headshots=0)

    assert m.reaction_time_ms is not None and m.reaction_time_ms > 0
    assert m.overshoot_count >= 1
    assert m.smoothness_sparc < 0
    assert 0.0 <= m.path_efficiency <= 1.0
    assert abs(m.crosshair_placement_error_deg - 2.5) < 1e-6
    assert m.crosshair_placement_method == "horizon_proxy_2d"
    print(f"    rt={m.reaction_time_ms:.1f}±{m.reaction_time_error_ms:.1f}ms "
          f"sparc={m.smoothness_sparc:.2f} jitter={m.jitter_ratio:.3f} "
          f"overshoot={m.overshoot_count} eff={m.path_efficiency:.2f}")


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
