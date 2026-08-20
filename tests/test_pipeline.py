"""End-to-end: rendered footage through every local stage to schema-valid JSON.

The unit tests prove each stage in isolation. This proves they compose — which
is where an integration usually breaks, because each stage's idea of a frame
index, a timestamp, or a confidence has to survive contact with the next one.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyzer.events import DigitClassifier
from analyzer.metrics import Trace, analyze
from analyzer.pipeline import (
    Candidate, analyze_video, build_record, coachability_scores, find_candidates,
    select_clips,
)
from analyzer.profiles import GameProfile
from analyzer.synthetic import flick_path, make_world, render_clip, write_video

SCHEMA = json.loads((ROOT / "schemas" / "clip_analysis.v1.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)
FPS = 60.0
WORLD = make_world()


# Pipeline tests exercise composition, not sub-degree accuracy — that is
# test_motion.py's job at full resolution. Rendering these fixtures at 640x360
# cuts the suite's runtime by roughly 4x with no loss of coverage here.
FIXTURE_W, FIXTURE_H = 640, 360
_MATCH_CACHE: dict[tuple[int, int], list[np.ndarray]] = {}


def _draw_ammo(frame: np.ndarray, value: int) -> None:
    """Draw the ammo counter into the profile's declared ROI, at any size."""
    h, w = frame.shape[:2]
    x0, y0 = int(w * 0.86), int(h * 0.88)
    x1, y1 = x0 + int(w * 0.11), y0 + int(h * 0.08)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (18, 18, 18), -1)
    scale = (y1 - y0) / 32.0
    cv2.putText(frame, str(value), (x0 + int(5 * scale), y1 - int((y1 - y0) * 0.22)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235),
                max(1, int(round(scale * 2))), cv2.LINE_AA)


def synth_match(n_engagements: int = 3, frames_per: int = 190) -> list[np.ndarray]:
    """A recording of several engagements, each a flick then a burst.

    Engagements are spaced seconds apart, as in a real match, so the extracted
    clip windows do not all overlap. The ammo counter decrements across the
    whole recording, so shot detection reads real pixels rather than being
    handed events directly.

    Memoized: several tests want the same recording, and re-rendering it each
    time dominated the suite's runtime.
    """
    key = (n_engagements, frames_per)
    if key in _MATCH_CACHE:
        return _MATCH_CACHE[key]

    frames: list[np.ndarray] = []
    ammo = 30
    burst_at = frames_per // 2
    for e in range(n_engagements):
        path = flick_path(frames_per, FPS, 20.0 + e * 9, 0.16, overshoot_deg=4.0 + e * 2)
        clip = render_clip(path, np.zeros_like(path), world=WORLD, h_fov_deg=103.0,
                           width=FIXTURE_W, height=FIXTURE_H, distractor=False, hud=False)
        for i, frame in enumerate(clip):
            if burst_at <= i < burst_at + 4:
                ammo = max(0, ammo - 1)
            _draw_ammo(frame, ammo)
            frames.append(frame)

    _MATCH_CACHE[key] = frames
    return frames


# ── profiles ─────────────────────────────────────────────────────────────────

def test_shipped_profiles_load_and_validate():
    for name in GameProfile.available():
        p = GameProfile.load(name)
        assert p.rois and 1.0 < p.fov_degrees < 179.0
    print(f"    loaded {GameProfile.available()}")


def test_profile_rejects_out_of_frame_roi():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write("game: bad\nfov_degrees: 100\nrois:\n  ammo: [0.9, 0.9, 0.5, 0.5]\n")
        path = fh.name
    try:
        GameProfile.load(path)
        raise AssertionError("accepted an ROI extending past the frame")
    except ValueError:
        pass
    finally:
        Path(path).unlink()


def test_missing_profile_names_the_alternatives():
    try:
        GameProfile.load("quake_champions")
        raise AssertionError("accepted an unknown profile")
    except FileNotFoundError as exc:
        assert "valorant" in str(exc), "error should list what is available"


# ── selection ────────────────────────────────────────────────────────────────

def _fake(reaction: float, overshoot: int, jitter: float, event_type: str = "whiff") -> Candidate:
    n = 30
    trace = Trace(np.arange(n) * (1000 / FPS), np.linspace(0, 10, n), np.zeros(n), FPS)
    m = analyze(trace, stimulus_frame=0, shot_frames=[20], hits=0)
    m = type(m)(**{**m.__dict__, "reaction_time_ms": reaction,
                   "overshoot_count": overshoot, "jitter_ratio": jitter})
    return Candidate(0, 500, event_type, [], trace, m, 0.9, [])


def test_selection_respects_the_budget():
    """Stage 5 is the cost valve: it must never return more than asked for."""
    cands = [_fake(200 + i * 7, i % 4, 0.1 + i * 0.01) for i in range(40)]
    assert len(select_clips(cands, 8)) == 8
    assert len(select_clips(cands, 3)) == 3
    assert len(select_clips(cands[:2], 8)) == 2, "must not invent clips it does not have"


def test_selection_prefers_diverse_failures_over_duplicates():
    """Ranking by score alone returns eight instances of one mistake, costing
    eight times as much to say one thing."""
    duplicates = [_fake(400, 5, 0.4) for _ in range(9)]
    distinct = [_fake(180, 0, 0.05), _fake(300, 2, 0.22)]
    picked = select_clips(duplicates + distinct, 4)

    from_distinct = sum(1 for i in picked if i >= len(duplicates))
    assert from_distinct >= 1, "selection returned only near-identical clips"
    print(f"    from 9 near-identical + 2 distinct, picked {sorted(picked)}")


def test_selection_includes_a_clean_rep():
    """Users need to see what their own correct mechanics look like."""
    bad = [_fake(420 + i * 5, 4, 0.4) for i in range(6)]
    good = _fake(185, 0, 0.03, "clean_rep")
    picked = select_clips(bad + [good], 4)
    assert len(bad) in picked, "no good rep survived selection"
    print(f"    clean rep (index {len(bad)}) retained in {sorted(picked)}")


def test_scores_reward_deviation_in_either_direction():
    cands = [_fake(200, 0, 0.1), _fake(205, 0, 0.1), _fake(600, 0, 0.1), _fake(198, 0, 0.1)]
    scores = coachability_scores(cands)
    assert scores[2] == max(scores), "the outlier engagement did not rank highest"


# ── end to end ───────────────────────────────────────────────────────────────

def test_pipeline_finds_engagements_in_rendered_footage():
    frames = synth_match(n_engagements=3)
    profile = GameProfile.load("valorant")
    cands = find_candidates(frames, FPS, profile)
    assert len(cands) >= 3, f"expected >=3 engagements, found {len(cands)}"
    for c in cands:
        assert c.metrics.shots_fired > 0, "engagement with no shots"
        assert c.trace_confidence > 0.5, f"low trace confidence {c.trace_confidence:.2f}"
    print(f"    {len(cands)} engagements, "
          f"shots={[c.metrics.shots_fired for c in cands]}, "
          f"trace confidence={[round(c.trace_confidence, 2) for c in cands]}")


def test_pipeline_output_validates_against_the_schema():
    """The contract between CV, storage, and the dashboard."""
    frames = synth_match(n_engagements=3)
    profile = GameProfile.load("valorant")
    cands = find_candidates(frames, FPS, profile)
    records = [build_record(c, "session-1", "synth.mp4", profile.game, FPS,
                            f"{FIXTURE_W}x{FIXTURE_H}")
               for c in cands]
    assert records
    for rec in records:
        errors = list(VALIDATOR.iter_errors(rec))
        assert not errors, [f"{list(e.path)}: {e.message}" for e in errors]
    print(f"    {len(records)} records validate against clip_analysis/v1")


def test_back_to_back_engagements_do_not_produce_identical_clips():
    """Regression guard: window anchoring.

    Clamping a near-start window with max(0, ...) and then adding the window
    length re-anchors every early engagement to frame 0, so distinct
    engagements extract byte-identical footage. The content-addressed AI cache
    keys on those bytes, so it would then serve one engagement's analysis for
    another. Windows must slide, and any that genuinely coincide must collapse
    into one clip rather than being analyzed and billed twice.
    """
    frames = synth_match(n_engagements=3, frames_per=95)   # bursts ~1.6s apart
    cands = find_candidates(frames, FPS, GameProfile.load("valorant"))
    hashes = [c.content_sha256 for c in cands]
    assert len(hashes) == len(set(hashes)), \
        f"{len(hashes)} clips collapsed to {len(set(hashes))} distinct hashes"
    spans = [(c.start_ms, c.end_ms) for c in cands]
    assert len(spans) == len(set(spans)), f"duplicate windows survived: {spans}"
    print(f"    {len(cands)} tightly-spaced engagements -> {len(set(hashes))} "
          f"distinct clips at {spans}")


def _trainer_clip(spawn: int = 40, shot: int = 53, n: int = 150) -> list[np.ndarray]:
    """Aim-trainer footage with a target spawning and a shot at known frames."""
    frames, ammo = [], 30
    for i in range(n):
        f = np.full((360, 640, 3), 35, np.uint8)
        cv2.rectangle(f, (60, 40), (580, 320), (48, 44, 40), 2)
        for x in range(80, 600, 60):
            cv2.line(f, (x, 45), (x, 315), (52, 48, 44), 1)
        if i >= spawn:
            cv2.circle(f, (420, 150), 22, (60, 60, 230), -1)
        if shot <= i < shot + 2:
            ammo = max(0, ammo - 1)
        _draw_ammo(f, ammo)
        frames.append(f)
    return frames


def test_reaction_time_measured_against_a_real_stimulus():
    """The full loop: target spawn detected, shot detected, interval measured.

    Aim trainers are the one place this tier can localize a stimulus onset
    reliably, which is exactly why the build order puts them first.
    """
    spawn, shot = 40, 53
    cands = find_candidates(_trainer_clip(spawn, shot), FPS, GameProfile.load("aimlabs"))
    assert cands, "no engagement found in trainer footage"
    m = cands[0].metrics
    expected = (shot - spawn) * 1000.0 / FPS
    assert m.reaction_time_ms is not None, "stimulus was detectable but no reaction time"
    assert abs(m.reaction_time_ms - expected) <= m.reaction_time_error_ms, \
        f"measured {m.reaction_time_ms:.1f}ms against ground truth {expected:.1f}ms"
    print(f"    ground truth {expected:.1f}ms -> measured "
          f"{m.reaction_time_ms:.1f}±{m.reaction_time_error_ms:.1f}ms")


def test_stimulus_must_precede_the_shot():
    """Regression guard: a clip window routinely spans more than one target.

    Taking the earliest spawn in the window picks up the next target appearing
    *after* the trigger pull, producing a negative interval — which the metrics
    layer rejects outright, crashing the run.
    """
    # Target spawns at 40, shot at 53, then a second target spawns at 70.
    frames = _trainer_clip(spawn=40, shot=53, n=150)
    for i in range(70, 150):
        cv2.circle(frames[i], (180, 250), 22, (60, 60, 230), -1)

    cands = find_candidates(frames, FPS, GameProfile.load("aimlabs"))
    assert cands, "no engagement found"
    m = cands[0].metrics
    assert m.reaction_time_ms is not None and m.reaction_time_ms > 0, \
        f"reaction time {m.reaction_time_ms} measured against a later stimulus"
    print(f"    two targets in one window -> {m.reaction_time_ms:.1f}ms "
          f"(measured from the earlier spawn, not the later one)")


def test_reaction_time_is_null_without_a_stimulus_detector():
    """Integrity guard.

    Falling back to the clip's own start frame re-reports the window offset as
    a reaction time: a constant that looks entirely plausible on a dashboard,
    reported identically for every clip in a match. The AI layer is told to
    treat `measured` as ground truth, so a fabricated value there is the worst
    failure this system can produce.
    """
    frames = synth_match(n_engagements=2)
    cands = find_candidates(frames, FPS, GameProfile.load("valorant"))
    with_shots = [c for c in cands if c.metrics.shots_fired > 0]
    assert with_shots, "fixture produced no shots"
    for c in with_shots:
        assert c.metrics.reaction_time_ms is None, \
            f"invented a reaction time of {c.metrics.reaction_time_ms}ms with no stimulus"
        assert any("no stimulus onset" in w for w in c.metrics.cv_warnings), \
            "null reaction time with no explanation of why"
    print(f"    {len(with_shots)} clips with shots, all reaction_time=None with a warning")


def test_no_whiff_claimed_without_hit_detection():
    """A shot is only a whiff if something could have registered a hit."""
    frames = synth_match(n_engagements=2)
    cands = find_candidates(frames, FPS, GameProfile.load("valorant"))
    for c in cands:
        assert c.event_type not in ("whiff", "whiff_then_death"), \
            f"claimed {c.event_type} with no hitmarker template configured"
        if c.metrics.shots_fired:
            assert any("hit detection not configured" in w for w in c.metrics.cv_warnings)
    print(f"    {len(cands)} clips labelled without inventing a hit outcome")


def test_records_are_json_serializable_without_nan():
    frames = synth_match(n_engagements=2)
    profile = GameProfile.load("valorant")
    cands = find_candidates(frames, FPS, profile)
    text = json.dumps([build_record(c, "s", "f.mp4", "valorant", FPS, f"{FIXTURE_W}x{FIXTURE_H}")
                       for c in cands])
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)


def test_skipped_assessment_is_well_formed_not_absent():
    """With no API key the AI block must still be present and schema-valid, so
    no consumer has to branch on its absence."""
    frames = synth_match(n_engagements=1)
    cands = find_candidates(frames, FPS, GameProfile.load("valorant"))
    rec = build_record(cands[0], "s", "f.mp4", "valorant", FPS,
                       f"{FIXTURE_W}x{FIXTURE_H}")
    assert rec["assessed"]["ai_status"] == "skipped"
    assert rec["assessed"]["evidence"] == []
    assert rec["assessed"]["not_determinable"]
    assert not list(VALIDATOR.iter_errors(rec))
    print(f"    no-key record still valid: ai_status={rec['assessed']['ai_status']!r}")


def test_content_hash_is_stable_and_content_addressed():
    """The AI cache keys on this: identical pixels must hash identically, and
    different footage must not collide, or re-analysis silently costs money."""
    frames = synth_match(n_engagements=2)
    profile = GameProfile.load("valorant")
    a = find_candidates(frames, FPS, profile)
    b = find_candidates(frames, FPS, profile)
    assert a[0].content_sha256 == b[0].content_sha256, "hash not reproducible"
    assert a[0].content_sha256 != a[1].content_sha256, "distinct clips collided"
    print(f"    stable: {a[0].content_sha256[:16]}…")


def test_cli_runs_on_a_real_file():
    """Decode from disk through the argparse entry point, as a user would."""
    frames = synth_match(n_engagements=2)
    with tempfile.TemporaryDirectory() as tmp:
        video = write_video(frames, str(Path(tmp) / "match.mp4"), FPS)
        out = Path(tmp) / "out.json"
        result = subprocess.run(
            [sys.executable, "-m", "analyzer.pipeline", video,
             "--profile", "valorant", "--max-clips", "4", "--out", str(out)],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        records = json.loads(out.read_text())
        assert 0 < len(records) <= 4
        for rec in records:
            assert not list(VALIDATOR.iter_errors(rec))
        print(f"    CLI produced {len(records)} valid records from "
              f"{len(frames)} frames ({len(frames) / FPS:.1f}s)")


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
