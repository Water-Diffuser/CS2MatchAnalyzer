#!/usr/bin/env python3
"""End-to-end example: one clip timestamp in, one structured analysis out.

    python reference/analyze_clip.py match.mp4 412300 --provider google

Shows the whole path a single engagement takes: measured CV telemetry, a frame
grid built from the timestamp, a pre-flight cost estimate the user confirms, the
provider call pinned to the JSON schema, and the assembled clip_analysis/v1
record ready for SQLite.

The CV numbers below are hard-coded so the example runs standalone; in the real
app they arrive from the sidecar's pipeline stage 4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_engine import BudgetExceeded, BudgetGuard, ClipRequest, build_result, get_provider

# Skill-tier expectations. Passing these turns "is 0.31 jitter bad?" — which the
# model cannot answer from priors and will guess at — into a factual comparison.
REFERENCE_BANDS = {
    "reaction_time_ms": [180, 250],
    "jitter_ratio": [0.05, 0.15],
    "overshoot_count": [0, 1],
    "crosshair_placement_error_deg": [0, 3],
    "path_efficiency": [0.85, 1.0],
}

MEASURED_FROM_CV = {
    "reaction_time_ms": 214,
    "reaction_time_error_ms": 16.7,
    "time_to_first_shot_ms": 231,
    "shots_fired": 5,
    "shots_hit": 1,
    "headshot_pct": 0.0,
    "smoothness_sparc": -2.41,
    "jitter_ratio": 0.31,
    "jitter_band_hz": [8, 30],
    "overshoot_count": 3,
    "peak_angular_error_deg": 8.2,
    "crosshair_placement_error_deg": 6.4,
    "crosshair_placement_method": "horizon_proxy_2d",
    "target_switch_ms": None,
    "path_efficiency": 0.62,
    "trace": {"t_ms": [0, 16, 33, 50], "yaw_deg": [0.0, -1.2, -4.8, -9.1],
              "pitch_deg": [0.0, 0.3, 0.9, 1.4], "speed_dps": [0.0, 72.1, 216.4, 258.0]},
    "cv_confidence": 0.88,
    "cv_warnings": ["fps=60 limits reaction_time resolution to +/-16.7ms"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("start_ms", type=int, help="engagement start within the recording")
    ap.add_argument("--duration-ms", type=int, default=3000)
    ap.add_argument("--provider", default="google", choices=["google", "openai", "anthropic"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--budget", type=float, default=0.50, help="hard session ceiling in USD")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = ap.parse_args()

    # In the app this is the hash of the trimmed clip bytes; it keys the cache,
    # so re-analyzing the same engagement is free.
    clip_sha = hashlib.sha256(
        f"{args.video}:{args.start_ms}:{args.duration_ms}".encode()
    ).hexdigest()

    req = ClipRequest(
        video_path=args.video,
        start_ms=args.start_ms,
        end_ms=args.start_ms + args.duration_ms,
        clip_sha256=clip_sha,
        game="valorant",
        event_type="whiff_then_death",
        measured=MEASURED_FROM_CV,
        reference_bands=REFERENCE_BANDS,
        clip_index=4,
        clip_total=8,
        sensitivity_note="0.42 @ 800 DPI (eDPI 336), FOV 103°",
    )

    provider = get_provider(args.provider, model=args.model,
                            budget=BudgetGuard(session_limit_usd=args.budget))

    # Pre-flight: never let a batch start without the user seeing the number.
    estimate = provider.estimate_cost(req)
    print(f"  estimate: {estimate.human(n_clips=1, model=provider.model)}", file=sys.stderr)
    if not args.yes and input("  proceed? [y/N] ").strip().lower() != "y":
        return 1

    try:
        assessed = provider.analyze_clip(req)
    except BudgetExceeded as exc:
        print(f"  refused: {exc}", file=sys.stderr)
        return 2

    result = build_result(req, assessed, provider=args.provider, model=provider.model)
    json.dump(result, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
