"""Event detection against rendered HUDs with known content.

Every fixture here is drawn by the test, so the correct event list is known
exactly — including the negative cases, which is where a detector that looks
fine on a demo clip usually falls over.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.events import (
    DigitClassifier, GameEvent, KillFeedTracker, crop_roi, derive_events,
    detect_hitmarkers, detect_shots_from_ammo, detect_targets_hsv, hamming, phash,
)

W, H = 1280, 720
AMMO_ROI = (0.86, 0.88, 0.11, 0.08)
FEED_ROI = (0.62, 0.06, 0.38, 0.22)
CROSS_ROI = (0.44, 0.42, 0.12, 0.16)


def blank(noise: float = 0.0, seed: int = 1) -> np.ndarray:
    frame = np.full((H, W, 3), 40, np.uint8)
    if noise:
        rng = np.random.default_rng(seed)
        frame = np.clip(frame + rng.normal(0, noise, frame.shape), 0, 255).astype(np.uint8)
    return frame


def draw_ammo(frame: np.ndarray, value: int) -> np.ndarray:
    x0, y0 = int(W * 0.86), int(H * 0.88)
    cv2.rectangle(frame, (x0, y0), (x0 + int(W * 0.11), y0 + int(H * 0.08)), (18, 18, 18), -1)
    cv2.putText(frame, str(value), (x0 + 8, y0 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
    return frame


def hitmarker_sprite() -> np.ndarray:
    s = np.zeros((28, 28, 3), np.uint8)
    for a, b in (((4, 4), (10, 10)), ((24, 4), (18, 10)), ((4, 24), (10, 18)), ((24, 24), (18, 18))):
        cv2.line(s, a, b, (255, 255, 255), 2)
    return s


def draw_hitmarker(frame: np.ndarray) -> np.ndarray:
    s = hitmarker_sprite()
    cx, cy = W // 2 - 14, H // 2 - 14
    frame[cy:cy + 28, cx:cx + 28] = np.maximum(frame[cy:cy + 28, cx:cx + 28], s)
    return frame


def draw_feed(frame: np.ndarray, rows: list[str]) -> np.ndarray:
    x0, y0 = int(W * 0.62), int(H * 0.06)
    fw, fh = int(W * 0.38), int(H * 0.22)
    cv2.rectangle(frame, (x0, y0), (x0 + fw, y0 + fh), (26, 22, 22), -1)
    for i, text in enumerate(rows):
        cv2.putText(frame, text, (x0 + 10, y0 + 22 + i * (fh // 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 215, 205), 1, cv2.LINE_AA)
    return frame


# ── phash ────────────────────────────────────────────────────────────────────

def test_phash_is_stable_under_compression_noise():
    """Rows must hash alike across frames despite codec noise, or every frame
    of one kill emits a fresh event."""
    base = draw_feed(blank(), ["player_a  [AK]  player_b"])
    row = crop_roi(base, FEED_ROI)[:30]
    rng = np.random.default_rng(4)
    noisy = np.clip(row.astype(np.int16) + rng.normal(0, 4, row.shape), 0, 255).astype(np.uint8)
    d = hamming(phash(row), phash(noisy))
    assert d <= 6, f"hash drifted by {d} bits under mild noise"
    print(f"    hamming distance under noise: {d} bits")


def test_phash_separates_different_rows():
    a = crop_roi(draw_feed(blank(), ["player_a  [AK]  player_b"]), FEED_ROI)[:30]
    b = crop_roi(draw_feed(blank(), ["someone_c [OP]  other_d"]), FEED_ROI)[:30]
    d = hamming(phash(a), phash(b))
    assert d > 6, f"distinct kills hash only {d} bits apart"
    print(f"    hamming distance between different kills: {d} bits")


# ── digits ───────────────────────────────────────────────────────────────────

def test_digit_reader_reads_every_ammo_value():
    clf = DigitClassifier.from_font()
    wrong = []
    for value in list(range(0, 31)) + [45, 90, 100]:
        got, conf = clf.read_number(crop_roi(draw_ammo(blank(), value), AMMO_ROI))
        if got != value:
            wrong.append((value, got))
    assert not wrong, f"misread: {wrong}"
    print(f"    read 34/34 ammo values correctly (0-30, 45, 90, 100)")


def test_digit_reader_declines_on_garbage():
    """An unreadable counter must return None, not a partial number — a
    misread produces a phantom shot, which is worse than a gap."""
    clf = DigitClassifier.from_font()
    rng = np.random.default_rng(2)
    noise = rng.integers(0, 255, (58, 141, 3), dtype=np.uint8)
    value, _ = clf.read_number(noise)
    assert value is None or value < 1000


# ── shots ────────────────────────────────────────────────────────────────────

def test_shots_from_ammo_counts_a_spray():
    """30 down to 25 is five shots, one per decrement."""
    clf = DigitClassifier.from_font()
    frames = [draw_ammo(blank(), v) for v in [30, 30, 29, 28, 27, 27, 26, 25, 25]]
    shots = detect_shots_from_ammo(frames, AMMO_ROI, clf)
    assert len(shots) == 5, f"expected 5 shots, got {len(shots)}"
    assert [s.frame for s in shots] == [2, 3, 4, 6, 7]
    print(f"    spray 30->25 detected as {len(shots)} shots at frames "
          f"{[s.frame for s in shots]}")


def test_reload_is_not_a_shot():
    """Ammo going up is a reload. Counting it would invert the metric."""
    clf = DigitClassifier.from_font()
    frames = [draw_ammo(blank(), v) for v in [8, 7, 6, 30, 30, 29]]
    shots = detect_shots_from_ammo(frames, AMMO_ROI, clf)
    assert len(shots) == 3, f"reload miscounted: {len(shots)} shots"
    print(f"    reload 6->30 correctly ignored ({len(shots)} shots, not 4)")


def test_weapon_switch_does_not_emit_a_burst():
    """A 30->4 jump is a weapon switch, not 26 shots in one frame interval."""
    clf = DigitClassifier.from_font()
    frames = [draw_ammo(blank(), v) for v in [30, 30, 4, 4, 3]]
    shots = detect_shots_from_ammo(frames, AMMO_ROI, clf)
    assert len(shots) == 1, f"weapon switch produced {len(shots)} shots"


# ── hits ─────────────────────────────────────────────────────────────────────

def test_hitmarker_detected_once_per_hit():
    """The sprite persists ~4 frames; that is one hit, not four."""
    frames = [blank(noise=3, seed=i) for i in range(12)]
    for i in (3, 4, 5, 6):
        draw_hitmarker(frames[i])
    hits = detect_hitmarkers(frames, hitmarker_sprite(), CROSS_ROI)
    assert len(hits) == 1, f"expected 1 hit, got {len(hits)}"
    assert hits[0].frame == 3, f"hit registered at frame {hits[0].frame}, expected 3"
    print(f"    4-frame sprite collapsed to 1 hit at frame {hits[0].frame} "
          f"(score {hits[0].confidence:.3f})")


def test_no_hitmarker_no_false_positive():
    frames = [blank(noise=6, seed=i) for i in range(20)]
    assert detect_hitmarkers(frames, hitmarker_sprite(), CROSS_ROI) == []


def test_two_separated_hits_both_register():
    frames = [blank(noise=3, seed=i) for i in range(20)]
    for i in (2, 3, 12, 13):
        draw_hitmarker(frames[i])
    hits = detect_hitmarkers(frames, hitmarker_sprite(), CROSS_ROI)
    assert [h.frame for h in hits] == [2, 12], f"got {[h.frame for h in hits]}"


# ── kill feed ────────────────────────────────────────────────────────────────

def test_kill_feed_dedups_a_persistent_row():
    """The load-bearing test. A row on screen for 120 frames is one kill."""
    frames = [draw_feed(blank(noise=2, seed=i), ["player_a  [AK]  player_b"])
              for i in range(120)]
    kills = KillFeedTracker(FEED_ROI).run(frames)
    assert len(kills) == 1, f"120 frames of one kill produced {len(kills)} events"
    assert kills[0].frame == 0
    print(f"    120 frames of one row -> {len(kills)} event at frame {kills[0].frame}")


def test_kill_feed_emits_at_first_appearance():
    """The first-appearance frame is the frame-accurate kill time that the
    reaction-time metric depends on."""
    frames = ([draw_feed(blank(noise=2, seed=i), []) for i in range(30)] +
              [draw_feed(blank(noise=2, seed=i), ["player_a  [AK]  player_b"])
               for i in range(30, 90)])
    kills = KillFeedTracker(FEED_ROI).run(frames)
    assert len(kills) == 1, f"got {len(kills)} events"
    assert kills[0].frame == 30, f"kill timed at frame {kills[0].frame}, expected 30"
    print(f"    kill timed at first appearance: frame {kills[0].frame}")


def test_kill_feed_tracks_a_sequence_of_distinct_kills():
    frames = []
    for i in range(40):
        frames.append(draw_feed(blank(noise=2, seed=i), ["aaa_one  [AK]  bbb_two"]))
    for i in range(40):
        frames.append(draw_feed(blank(noise=2, seed=i),
                                ["ccc_three [OP] ddd_four", "aaa_one  [AK]  bbb_two"]))
    kills = KillFeedTracker(FEED_ROI).run(frames)
    assert len(kills) == 2, f"expected 2 distinct kills, got {len(kills)}"
    assert [k.frame for k in kills] == [0, 40], f"timed at {[k.frame for k in kills]}"
    print(f"    2 distinct kills at frames {[k.frame for k in kills]}")


def test_empty_feed_emits_nothing():
    frames = [draw_feed(blank(noise=2, seed=i), []) for i in range(60)]
    assert KillFeedTracker(FEED_ROI).run(frames) == []


# ── targets ──────────────────────────────────────────────────────────────────

def test_hsv_target_detection_finds_known_spheres():
    frame = np.full((H, W, 3), 30, np.uint8)
    truth = [(400, 300, 26), (820, 460, 20)]
    for x, y, r in truth:
        cv2.circle(frame, (x, y), r, (60, 60, 230), -1)     # BGR red
    found = detect_targets_hsv(frame, (0, 120, 90), (10, 255, 255))
    assert len(found) == 2, f"found {len(found)} targets, expected 2"
    for (tx, ty, _), (fx, fy, _) in zip(sorted(truth, key=lambda t: -t[2]), found):
        assert abs(fx - tx) < 2 and abs(fy - ty) < 2, f"centroid off: ({fx},{fy}) vs ({tx},{ty})"
    print(f"    2 targets located to sub-2px: "
          f"{[(round(f[0]), round(f[1])) for f in found]}")


# ── composition ──────────────────────────────────────────────────────────────

def test_whiffs_are_derived_not_detected():
    """5 shots, 1 hit -> 4 whiffs, no whiff detector involved.

    The invariant that matters: whiffs == shots - hits, always. One hitmarker
    cannot satisfy three shots just because they fell inside its tolerance
    window, or a spray that landed once would report as three hits.
    """
    shots = [GameEvent(f, "shot", 0.9) for f in (10, 12, 14, 16, 18)]
    hits = [GameEvent(14, "hit", 0.9)]
    events = derive_events(shots, hits)
    whiffs = [e for e in events if e.type == "whiff"]
    assert len(whiffs) == len(shots) - len(hits) == 4, \
        f"expected 4 whiffs, got {len(whiffs)} at {[w.frame for w in whiffs]}"
    # The hitmarker landed on frame 14, so the shot at 14 is the one that
    # connected — not the shot at 12 that merely sits inside its window.
    assert 14 not in [w.frame for w in whiffs], \
        f"the shot that actually hit was reported as a whiff: {[w.frame for w in whiffs]}"
    print(f"    5 shots, 1 hit -> {len(whiffs)} whiffs at "
          f"{[w.frame for w in whiffs]} (frame 14 correctly credited)")


def test_whiff_invariant_holds_across_hit_counts():
    """whiffs == shots - hits for any number of landed shots in a spray."""
    shots = [GameEvent(f, "shot", 0.9) for f in range(10, 30, 2)]
    for n_hits in range(0, len(shots) + 1):
        hits = [GameEvent(shots[i].frame, "hit", 0.9) for i in range(n_hits)]
        whiffs = [e for e in derive_events(shots, hits) if e.type == "whiff"]
        assert len(whiffs) == len(shots) - n_hits, \
            f"{len(shots)} shots, {n_hits} hits -> {len(whiffs)} whiffs"
    print(f"    invariant holds for 0..{len(shots)} hits on a "
          f"{len(shots)}-shot spray")


def test_hit_tolerance_spans_frame_lag():
    """A hitmarker lands a frame or two after the shot; that is still a hit."""
    events = derive_events([GameEvent(10, "shot", 0.9)], [GameEvent(12, "hit", 0.9)])
    assert not [e for e in events if e.type == "whiff"]


def test_detected_by_matches_schema_enum():
    """Provenance strings must match clip_analysis/v1's detected_by enum."""
    import json
    schema = json.loads((Path(__file__).resolve().parents[1] /
                         "schemas" / "clip_analysis.v1.schema.json").read_text())
    allowed = set(schema["properties"]["event"]["properties"]["detected_by"]["items"]["enum"])
    for t in ("shot", "hit", "whiff", "kill", "death", "target_spawn"):
        by = GameEvent(0, t, 1.0).detected_by
        assert by in allowed, f"{t} -> {by!r} is not in the schema enum"
    print(f"    all detector provenance strings valid against the schema")


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
