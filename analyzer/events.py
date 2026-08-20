"""HUD-region event detection: shots, hits, whiffs, kills, targets.

Everything here reads fixed screen regions declared by a game profile. That
constraint is what makes the tier cheap enough to run over a whole match on a
CPU, and it is why detection is template- and hash-based rather than learned:
a HUD is drawn from a small fixed sprite set at a fixed position, which is the
easiest possible recognition problem if you don't throw a general model at it.

Note the composition in `derive_events`: whiffs are never detected. They fall
out of shots and hits, which means there is no whiff detector to train, tune,
or have go wrong.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

# A kill-feed row persists for several seconds, so a naive per-frame reader
# reports one kill hundreds of times. Rows are matched by perceptual hash
# within this Hamming distance to survive antialiasing and compression noise.
PHASH_MATCH_DISTANCE = 6
HITMARKER_THRESHOLD = 0.80

# A real kill-feed row holds still for seconds. World content behind a
# semi-transparent feed changes every frame as the camera moves. Requiring a
# hash to repeat across consecutive frames is what separates the two, and
# without it the tracker fires on every frame of any non-flat background.
KILL_ROW_PERSISTENCE_FRAMES = 4

# Rows expire from the feed after a few seconds, so remembering more than this
# buys nothing and makes every lookup linear in the length of the match.
SEEN_HISTORY = 64

# A feed ROI is sized to bound where rows *can* appear, so most of it is
# usually empty — showing whatever the world is doing behind a semi-transparent
# panel. Hashing those slices treats every camera movement as a fresh kill:
# on a 20s recording that produced 560 events against 5 real engagements, and
# collapsed the entire match into one candidate.
#
# Kill feeds render on a solid or heavily darkened band, so a genuine row has a
# dominant background colour while world content does not. Measured on rendered
# footage: feed rows sit at 0.91, world content at 0.28-0.60. The threshold is
# a profile parameter because how opaque a feed's backing is varies by title.
FEED_ROW_UNIFORMITY = 0.70


def phash(image: np.ndarray, hash_size: int = 8) -> int:
    """64-bit perceptual hash via DCT.

    Chosen over an exact hash because compression makes a kill-feed row differ
    by a few pixels frame to frame while being visually identical; an exact
    hash would emit a fresh event on every frame of the same kill.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low = dct[:hash_size, :hash_size]
    # Exclude the DC term: it tracks overall brightness, so including it makes
    # the hash change when the scene behind a translucent HUD changes.
    med = np.median(low.flatten()[1:])
    bits = (low.flatten() > med).astype(np.uint8)
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def crop_roi(frame: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    """Crop a normalized (x, y, w, h) region. Normalized so one profile works
    at every resolution the user might have recorded at."""
    h, w = frame.shape[:2]
    x0, y0 = int(roi[0] * w), int(roi[1] * h)
    x1, y1 = min(w, x0 + int(roi[2] * w)), min(h, y0 + int(roi[3] * h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty ROI {roi} at {w}x{h}")
    return frame[y0:y1, x0:x1]


# ── digits ───────────────────────────────────────────────────────────────────

class DigitClassifier:
    """Template-matching digit reader for fixed-font HUD counters.

    Deliberately not OCR. An ammo counter is ten glyphs in one font at one
    size, rendered identically every frame — a closed-set classification
    problem. General OCR solves a far harder problem and does it less
    reliably here, while adding a native binary dependency to the installer.
    """

    def __init__(self, templates: dict[int, np.ndarray]):
        if set(templates) - set(range(10)):
            raise ValueError("templates must be keyed 0-9")
        self.templates = {d: self._normalize(t) for d, t in templates.items()}

    @staticmethod
    def _tight_crop(glyph: np.ndarray) -> np.ndarray:
        """Trim to the glyph's ink bounding box.

        Both templates and segmented glyphs must be cropped the same way or
        they are never comparable: a segmented glyph arrives tight from
        connected components, so a template stored with its rendering padding
        still around it aligns against nothing once both are resized.
        """
        ys, xs = np.nonzero(glyph > glyph.max() * 0.3 if glyph.max() else glyph)
        if ys.size == 0:
            return glyph
        return glyph[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    @classmethod
    def _normalize(cls, glyph: np.ndarray, size: tuple[int, int] = (16, 24)) -> np.ndarray:
        if glyph.ndim == 3:
            glyph = cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY)
        glyph = cls._tight_crop(glyph)
        resized = cv2.resize(glyph, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        # Match contrast too: segmented glyphs come back hard-thresholded while
        # a rendered template is antialiased, so compare shape, not intensity.
        lo, hi = resized.min(), resized.max()
        return (resized - lo) / (hi - lo) if hi > lo else resized

    @classmethod
    def from_font(cls, font=cv2.FONT_HERSHEY_SIMPLEX, scale: float = 0.9,
                  thickness: int = 2) -> "DigitClassifier":
        """Build templates by rendering the glyphs.

        For a real title you would instead cut these from screenshots once, per
        game, and ship them in the profile — same code path, different source.
        """
        templates = {}
        for d in range(10):
            canvas = np.zeros((48, 36), np.uint8)
            (tw, th), _ = cv2.getTextSize(str(d), font, scale, thickness)
            cv2.putText(canvas, str(d), ((36 - tw) // 2, (48 + th) // 2),
                        font, scale, 255, thickness, cv2.LINE_AA)
            templates[d] = canvas
        return cls(templates)

    def classify(self, glyph: np.ndarray) -> tuple[int, float]:
        """Return (digit, confidence) for a single segmented glyph."""
        probe = self._normalize(glyph)
        best, best_score = 0, -1.0
        for digit, template in self.templates.items():
            score = float(cv2.matchTemplate(probe, template, cv2.TM_CCOEFF_NORMED)[0, 0])
            if score > best_score:
                best, best_score = digit, score
        return best, best_score

    def read_number(self, roi: np.ndarray, *, min_confidence: float = 0.45
                    ) -> tuple[int | None, float]:
        """Segment an ROI into glyphs and read them left to right.

        Returns (None, 0.0) rather than a partial number when any glyph is
        unreadable: a misread ammo count produces a phantom shot event, which
        is worse than a gap the caller can interpolate across.
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if binary.mean() > 127:      # dark glyphs on a light panel
            binary = cv2.bitwise_not(binary)

        n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        h_roi, w_roi = binary.shape[:2]
        comps = [
            (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
             stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] > 12
            # Reject the panel background, which is a component too.
            and stats[i, cv2.CC_STAT_WIDTH] < w_roi * 0.9
            and stats[i, cv2.CC_STAT_HEIGHT] < h_roi * 0.9
        ]
        if not comps:
            return None, 0.0

        # Size-filter against the tallest glyph rather than the ROI height.
        # How much vertical padding a profile's ROI carries varies by game and
        # is not knowable here; that the digits of one counter share a height
        # is true everywhere.
        tallest = max(c[3] for c in comps)
        if tallest < h_roi * 0.12:
            return None, 0.0           # nothing glyph-sized in this ROI
        boxes = [c for c in comps if c[3] >= tallest * 0.55]
        if not boxes:
            return None, 0.0

        digits, scores = [], []
        for x, y, w, h in sorted(boxes, key=lambda b: b[0]):
            digit, score = self.classify(binary[y:y + h, x:x + w])
            if score < min_confidence:
                return None, 0.0
            digits.append(digit)
            scores.append(score)
        return int("".join(map(str, digits))), float(np.mean(scores))


# ── detectors ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GameEvent:
    frame: int
    type: str          # shot | hit | whiff | kill | death | target_spawn
    confidence: float
    detail: str = ""

    @property
    def detected_by(self) -> str:
        return {"shot": "ammo_decrement", "hit": "hitmarker_template",
                "whiff": "ammo_decrement", "kill": "kill_feed_ocr",
                "death": "kill_feed_ocr", "target_spawn": "blob_target"}[self.type]


def detect_shots_from_ammo(frames: Sequence[np.ndarray], ammo_roi: tuple[float, float, float, float],
                           classifier: DigitClassifier) -> list[GameEvent]:
    """A shot is a decrement in the ammo counter.

    The most reliable shot proxy available without touching the game: it
    handles spray (27->26->25) for free, cannot be triggered by another
    player's muzzle flash, and survives smoke and flashbangs that defeat
    luminance-based detection. A reload (count increasing) is not a shot.
    """
    events: list[GameEvent] = []
    previous: int | None = None
    for i, frame in enumerate(frames):
        value, confidence = classifier.read_number(crop_roi(frame, ammo_roi))
        if value is None:
            continue            # unreadable frame: hold state, do not guess
        if previous is not None and 0 < previous - value <= 3:
            # Cap the step: a jump from 30 to 4 is a weapon switch or a misread,
            # not eight shots inside one frame interval.
            for _ in range(previous - value):
                events.append(GameEvent(i, "shot", confidence, f"ammo {previous}->{value}"))
        previous = value
    return events


def detect_hitmarkers(frames: Sequence[np.ndarray], template: np.ndarray,
                      roi: tuple[float, float, float, float],
                      threshold: float = HITMARKER_THRESHOLD) -> list[GameEvent]:
    """Template-match the hitmarker sprite in the crosshair region.

    A hitmarker persists for a few frames, so consecutive detections collapse
    into one hit — otherwise a single confirmed hit inflates the hit count and
    silently suppresses a whiff that really happened.
    """
    tmpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template
    events: list[GameEvent] = []
    active = False
    for i, frame in enumerate(frames):
        patch = crop_roi(frame, roi)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        if gray.shape[0] < tmpl_gray.shape[0] or gray.shape[1] < tmpl_gray.shape[1]:
            continue
        score = float(cv2.matchTemplate(gray, tmpl_gray, cv2.TM_CCOEFF_NORMED).max())
        if score >= threshold:
            if not active:
                events.append(GameEvent(i, "hit", score))
            active = True
        else:
            active = False
    return events


class KillFeedTracker:
    """Emits one event per kill-feed row, at the frame it first appeared.

    The dedup is the entire point. A row is on screen for roughly five seconds
    — 300 frames at 60fps — and every one of those frames reads as a kill to a
    stateless detector. Hashing rows and remembering what has been seen turns
    that into one event, and the first-appearance frame is also the
    frame-accurate kill time, which is what the reaction-time metric needs.
    """

    def __init__(self, roi: tuple[float, float, float, float], rows: int = 5,
                 match_distance: int = PHASH_MATCH_DISTANCE,
                 persistence_frames: int = KILL_ROW_PERSISTENCE_FRAMES,
                 uniformity_threshold: float = FEED_ROW_UNIFORMITY):
        self.roi = roi
        self.rows = rows
        self.match_distance = match_distance
        self.persistence_frames = persistence_frames
        self.uniformity_threshold = uniformity_threshold
        self._seen: deque[int] = deque(maxlen=SEEN_HISTORY)
        # Per row slot: the hash currently being observed, the frame it first
        # appeared on, and how many consecutive frames it has held.
        self._pending: dict[int, tuple[int, int, int]] = {}

    @staticmethod
    def background_uniformity(row: np.ndarray) -> float:
        """Fraction of pixels sitting near the row's modal intensity.

        High for a panel with text drawn on it, low for natural scenery. This
        is what tells a real feed row from the world showing through an empty
        slot — persistence alone cannot, because a player holding an angle
        produces a genuinely static world for seconds at a time.
        """
        gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY) if row.ndim == 3 else row
        hist = np.bincount(gray.ravel(), minlength=256)
        mode = int(hist.argmax())
        return float(hist[max(0, mode - 12):mode + 13].sum()) / gray.size

    def _row_hashes(self, frame: np.ndarray) -> list[int]:
        feed = crop_roi(frame, self.roi)
        row_h = max(1, feed.shape[0] // self.rows)
        out = []
        for r in range(self.rows):
            row = feed[r * row_h:(r + 1) * row_h]
            if row.size == 0:
                continue
            gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY) if row.ndim == 3 else row
            # An empty feed slot is flat; hashing it would produce a stable
            # "row" that fires once and then blocks a real kill hashing near it.
            if gray.std() < 6.0:
                continue
            # Scenery behind an empty slot: not a row, whatever it hashes to.
            if self.background_uniformity(row) < self.uniformity_threshold:
                continue
            out.append(phash(row))
        return out

    def _is_new(self, h: int) -> bool:
        return all(hamming(h, prev) > self.match_distance for prev in self._seen)

    def update(self, frame_idx: int, frame: np.ndarray) -> list[GameEvent]:
        """Emit a kill only once a row has held still long enough to be one.

        A hash is confirmed after `persistence_frames` consecutive sightings,
        but the event is timestamped to the frame it *first* appeared on —
        that first-appearance frame is the frame-accurate kill time the
        reaction-time metric depends on, so the confirmation delay must not
        leak into it.
        """
        events = []
        current = self._row_hashes(frame)

        for slot, h in enumerate(current):
            prev = self._pending.get(slot)
            if prev is not None and hamming(h, prev[0]) <= self.match_distance:
                first_frame, count = prev[1], prev[2] + 1
            else:
                first_frame, count = frame_idx, 1
            self._pending[slot] = (h, first_frame, count)

            if count == self.persistence_frames and self._is_new(h):
                self._seen.append(h)
                events.append(GameEvent(first_frame, "kill", 0.9, f"feed row {h:016x}"))

        # Drop state for slots that vanished, so a returning row is treated as
        # new rather than resuming a stale count.
        for slot in [k for k in self._pending if k >= len(current)]:
            del self._pending[slot]
        return events

    def run(self, frames: Sequence[np.ndarray]) -> list[GameEvent]:
        out = []
        for i, frame in enumerate(frames):
            out.extend(self.update(i, frame))
        return out


def detect_targets_hsv(frame: np.ndarray, lower: tuple[int, int, int],
                       upper: tuple[int, int, int], min_area: int = 60
                       ) -> list[tuple[float, float, float]]:
    """Find aim-trainer targets by colour. Returns (cx, cy, area) per target.

    Trainer targets are deliberately high-contrast against a flat background,
    which is why the build order puts trainers first: near-perfect detection
    means the metric maths can be validated before the genuinely hard
    kill-feed CV is attempted.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        out.append((m["m10"] / m["m00"], m["m01"] / m["m00"], area))
    return sorted(out, key=lambda t: -t[2])


def derive_events(shots: Iterable[GameEvent], hits: Iterable[GameEvent],
                  tolerance_frames: int = 2) -> list[GameEvent]:
    """Compose shots and hits into the full event list, whiffs included.

    A whiff is a shot with no hit registered within a couple of frames. It is
    derived rather than detected, so there is no whiff detector to train or
    tune — and the definition stays legible enough to argue with.
    """
    hits = list(hits)
    shots = sorted(shots, key=lambda e: e.frame)

    # Each hitmarker may be claimed by exactly one shot. Letting a single hit
    # satisfy every shot within tolerance would report a 5-shot spray that
    # landed once as three hits, quietly erasing whiffs that happened.
    #
    # Assignment is globally nearest-first rather than in shot order. Walking
    # the shots in sequence lets an earlier shot at the edge of the tolerance
    # window claim a hitmarker that lands exactly on a later one, which then
    # reports the shot that actually connected as a whiff.
    pairs = sorted(
        ((abs(s.frame - h.frame), si, hi)
         for si, s in enumerate(shots) for hi, h in enumerate(hits)
         if abs(s.frame - h.frame) <= tolerance_frames),
        key=lambda p: (p[0], p[1]),
    )
    matched_shots: set[int] = set()
    claimed_hits: set[int] = set()
    for _, si, hi in pairs:
        if si not in matched_shots and hi not in claimed_hits:
            matched_shots.add(si)
            claimed_hits.add(hi)

    out: list[GameEvent] = []
    for si, shot in enumerate(shots):
        out.append(shot)
        if si not in matched_shots:
            out.append(GameEvent(shot.frame, "whiff", shot.confidence, "no hitmarker"))
    return sorted(out + hits, key=lambda e: (e.frame, e.type))
