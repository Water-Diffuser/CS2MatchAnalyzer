"""Stages 1-6 and 9: video in, schema-valid clip records out.

This is the half of the system that costs nothing to run. It works fully
offline with no API key, and the dashboard is expected to be usable on its
output alone — the AI tier layers explanation on top of these numbers, it does
not supply any of them.

    python -m analyzer.pipeline match.mp4 --profile valorant
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .events import (
    DigitClassifier, GameEvent, KillFeedTracker, derive_events, detect_hitmarkers,
    detect_shots_from_ammo,
)
from .metrics import ClipMetrics, Trace, analyze
from .motion import trace_from_frames
from .profiles import GameProfile

CV_VERSION = "0.4.0"
SCHEMA_VERSION = "clip_analysis/v1"

# Stage 5's budget. Everything upstream is free; everything downstream is not.
DEFAULT_CLIP_COUNT = 8
CLIP_DURATION_MS = 3000

# Shots further apart than this belong to different engagements. Roughly the
# gap between bursts in a fast game; well above a spray's inter-shot interval.
ENGAGEMENT_GAP_MS = 800


@dataclass
class Candidate:
    """One detected engagement, before it has earned an API call."""

    start_ms: int
    end_ms: int
    event_type: str
    frames: list[np.ndarray]
    trace: Trace
    metrics: ClipMetrics
    trace_confidence: float
    events: list[GameEvent]

    @property
    def content_sha256(self) -> str:
        """Hash of the actual decoded pixels, so the AI cache keys on content
        rather than on a filename and timestamp that a re-export would change."""
        h = hashlib.sha256()
        for f in self.frames[::4]:
            h.update(np.ascontiguousarray(f).tobytes())
        return h.hexdigest()


def _z(values: np.ndarray) -> np.ndarray:
    sd = values.std()
    return np.zeros_like(values) if sd < 1e-9 else (values - values.mean()) / sd


def coachability_scores(candidates: list[Candidate]) -> np.ndarray:
    """Rank engagements by how much a player would learn from reviewing them.

    Deviation in either direction counts: an unusually fast rep is as
    instructive as a slow one, and a session that only ever shows failures
    teaches nothing about what the player's own good mechanics look like.
    """
    if not candidates:
        return np.zeros(0)

    def col(attr: str) -> np.ndarray:
        return np.array([getattr(c.metrics, attr) or 0.0 for c in candidates], float)

    return (
        2.0 * np.abs(_z(col("reaction_time_ms")))
        + 1.5 * col("overshoot_count")
        + 1.5 * _z(col("jitter_ratio"))
        + 1.0 * np.array([
            (c.metrics.shots_fired - c.metrics.shots_hit) / max(1, c.metrics.shots_fired)
            for c in candidates
        ])
        + 1.0 * np.array([1.0 if c.event_type in ("death", "whiff_then_death") else 0.0
                          for c in candidates])
    )


def _feature_vector(c: Candidate) -> np.ndarray:
    m = c.metrics
    return np.array([
        (m.reaction_time_ms or 0.0) / 500.0,
        float(m.overshoot_count),
        (m.jitter_ratio or 0.0) * 3.0,
        (m.path_efficiency or 0.0),
        (m.smoothness_sparc or 0.0) / 3.0,
    ], float)


def select_clips(candidates: list[Candidate], n: int = DEFAULT_CLIP_COUNT
                 ) -> list[int]:
    """Pick the N engagements worth spending an API call on.

    The diversity penalty is the point. Ranking by score alone returns eight
    near-identical instances of one mistake, which costs eight times as much as
    it should and tells the player one thing. Each pick suppresses candidates
    that resemble it, so the set spans the session's distinct failure modes.

    At least one clean rep is forced into the selection: users need to see what
    their own correct mechanics look like, not only a list of failures.
    """
    if not candidates:
        return []
    n = min(n, len(candidates))

    scores = coachability_scores(candidates).astype(float)
    features = np.array([_feature_vector(c) for c in candidates])
    chosen: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(chosen) < n:
        pick = max(remaining, key=lambda i: scores[i])
        chosen.append(pick)
        remaining.discard(pick)
        # Similar candidates are worth progressively less once one is taken.
        for i in remaining:
            distance = float(np.linalg.norm(features[i] - features[pick]))
            scores[i] -= 2.0 * np.exp(-distance)

    if len(candidates) > len(chosen):
        cleanest = min(range(len(candidates)), key=lambda i: coachability_scores(candidates)[i])
        if cleanest not in chosen:
            chosen[-1] = cleanest      # trade the weakest pick for a good rep
    return chosen


def build_record(candidate: Candidate, session_id: str, source_file: str,
                 game: str, fps: float, resolution: str,
                 assessed: dict | None = None) -> dict:
    """Assemble a clip_analysis/v1 record.

    `measured` comes from CV and `assessed` from the model; they are written
    into sibling objects and never merged, so the AI's claims stay auditable
    against the numbers and a bad response cannot corrupt metrics history.
    When no key is configured, `assessed` is a well-formed skipped block rather
    than a missing one — consumers should never have to branch on its absence.
    """
    detected_by = sorted({e.detected_by for e in candidate.events}) or ["manual"]

    return {
        "schema_version": SCHEMA_VERSION,
        "clip_id": str(uuid.uuid4()),
        "session_id": session_id,
        "source": {
            "file": source_file,
            "game": game,
            "start_ms": candidate.start_ms,
            "end_ms": candidate.end_ms,
            "fps": fps,
            "resolution": resolution,
            "content_sha256": candidate.content_sha256,
        },
        "event": {
            "type": candidate.event_type,
            "weapon": None,
            "detection_confidence": round(candidate.trace_confidence, 3),
            "detected_by": detected_by,
        },
        "measured": candidate.metrics.to_json(candidate.trace),
        "assessed": assessed or {
            "ai_status": "skipped",
            "primary_weakness": None, "severity": None, "confidence": None,
            "summary": None, "evidence": [], "drill": None,
            "not_determinable": ["no AI provider configured; metrics only"],
        },
        "provenance": {
            "cv_version": CV_VERSION,
            "prompt_version": None if assessed is None else "clip_v1",
            "provider": None, "model": None,
            "input_tokens": None, "output_tokens": None,
            "estimated_cost_usd": None,
            "cached": False,
            "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }


def find_candidates(frames: list[np.ndarray], fps: float, profile: GameProfile,
                    classifier: DigitClassifier | None = None,
                    hitmarker: np.ndarray | None = None) -> list[Candidate]:
    """Detect engagements and measure each one.

    Windows are built around shots rather than around fixed intervals: an
    engagement's interesting part is the approach to the trigger pull, so the
    window is weighted before the shot rather than centred on it.
    """
    classifier = classifier or DigitClassifier.from_font()
    shots: list[GameEvent] = []
    if "ammo" in profile.rois:
        shots = detect_shots_from_ammo(frames, profile.rois["ammo"], classifier)

    hits: list[GameEvent] = []
    if hitmarker is not None and "crosshair" in profile.rois:
        hits = detect_hitmarkers(frames, hitmarker, profile.rois["crosshair"])

    kills: list[GameEvent] = []
    if "kill_feed" in profile.rois:
        kills = KillFeedTracker(profile.rois["kill_feed"], profile.kill_feed_rows).run(frames)

    all_events = derive_events(shots, hits) + kills
    if not all_events:
        return []

    # Group events into engagements. The gap is a property of what an
    # engagement *is* — one burst of fire and the aiming that led to it — not
    # of how long a clip happens to be. Tying it to CLIP_DURATION_MS merged
    # every engagement inside a 3s window into one, collapsing a whole match
    # into a single candidate. Two engagements closer together than the clip
    # window are still two engagements; the near-duplicate footage that
    # produces is what the selection stage's diversity penalty exists to
    # handle, at no cost, one stage later.
    gap_frames = max(2, int(ENGAGEMENT_GAP_MS / 1000.0 * fps))
    groups: list[list[GameEvent]] = []
    for event in sorted(all_events, key=lambda e: e.frame):
        if groups and event.frame - groups[-1][-1].frame <= gap_frames:
            groups[-1].append(event)
        else:
            groups.append([event])

    window = int(CLIP_DURATION_MS / 1000.0 * fps)
    candidates: list[Candidate] = []
    seen_windows: dict[tuple[int, int], int] = {}

    for group in groups:
        anchor = group[0].frame
        # Two thirds of the window sits before the anchor: what the crosshair
        # was doing on the approach is the coachable part, not the aftermath.
        # Slide rather than truncate when the anchor is near either end — a
        # naive max(0, ...) followed by lo + window silently re-anchors every
        # early engagement to frame 0, so several distinct engagements extract
        # byte-identical clips and the content-addressed AI cache then serves
        # one engagement's analysis for another.
        lo = anchor - int(window * 0.66)
        if len(frames) <= window:
            lo, hi = 0, len(frames)
        else:
            lo = max(0, min(lo, len(frames) - window))
            hi = lo + window
        if hi - lo < 4:
            continue

        # Two anchors can still land in one window once clamped. Identical
        # windows are one clip, so merge their events rather than paying twice
        # to analyze the same frames.
        if (lo, hi) in seen_windows:
            existing = candidates[seen_windows[(lo, hi)]]
            existing.events.extend(group)
            continue
        seen_windows[(lo, hi)] = len(candidates)

        clip_frames = frames[lo:hi]
        trace, confidence = trace_from_frames(
            clip_frames, fps, profile.fov_degrees, profile.flow_exclude
        )

        local_shots = [e.frame - lo for e in group if e.type == "shot" and lo <= e.frame < hi]
        n_hits = sum(1 for e in group if e.type == "hit")
        types = {e.type for e in group}
        event_type = ("whiff_then_death" if "whiff" in types and "death" in types
                      else "kill" if "kill" in types
                      else "whiff" if "whiff" in types
                      else "clean_rep")

        metrics = analyze(
            trace,
            stimulus_frame=0 if local_shots else None,
            shot_frames=local_shots,
            hits=n_hits,
        )
        candidates.append(Candidate(
            start_ms=int(lo * 1000 / fps),
            end_ms=int(hi * 1000 / fps),
            event_type=event_type,
            frames=clip_frames,
            trace=trace,
            metrics=metrics,
            trace_confidence=confidence,
            events=group,
        ))
    return candidates


def analyze_video(path: str, profile: GameProfile, *, max_clips: int = DEFAULT_CLIP_COUNT,
                  hitmarker: np.ndarray | None = None) -> list[dict]:
    """Full local pipeline for one recording."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) < 2:
        raise RuntimeError(f"{path}: decoded {len(frames)} frames")

    h, w = frames[0].shape[:2]
    session_id = str(uuid.uuid4())
    candidates = find_candidates(frames, fps, profile, hitmarker=hitmarker)
    selected = select_clips(candidates, max_clips)

    return [
        build_record(candidates[i], session_id, Path(path).name,
                     profile.game, fps, f"{w}x{h}")
        for i in sorted(selected, key=lambda i: candidates[i].start_ms)
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--profile", default="valorant",
                    help=f"one of {GameProfile.available()}, or a path to a yaml file")
    ap.add_argument("--max-clips", type=int, default=DEFAULT_CLIP_COUNT,
                    help="stage 5 budget: how many engagements may reach the AI tier")
    ap.add_argument("--out", default="-", help="write JSON here, or - for stdout")
    args = ap.parse_args(argv)

    records = analyze_video(args.video, GameProfile.load(args.profile),
                            max_clips=args.max_clips)
    payload = json.dumps(records, indent=2)
    if args.out == "-":
        print(payload)
    else:
        Path(args.out).write_text(payload)
        print(f"wrote {len(records)} clips to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
