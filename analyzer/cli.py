"""Command-line front door for the analyzer.

Subcommands rather than one flag-laden entry point, because the tool does
genuinely different jobs: analyze a recording, render an overlay, inspect the
environment. `selftest` matters most in a frozen binary — it proves the
bundled OpenCV and NumPy actually work on the user's machine, which is the
first question when a download misbehaves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .profiles import GameProfile
from .resources import __version__, runtime_report


def _err(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


def cmd_analyze(args: argparse.Namespace) -> int:
    from .pipeline import analyze_video

    video = Path(args.video)
    if not video.exists():
        return _err(f"no such file: {video}")
    try:
        profile = GameProfile.load(args.profile)
    except (FileNotFoundError, ValueError) as exc:
        return _err(str(exc))

    if not args.quiet:
        print(f"analyzing {video.name} with profile {profile.game}…", file=sys.stderr)
    try:
        records = analyze_video(str(video), profile, max_clips=args.max_clips,
                                overlay_dir=args.overlay_dir)
    except RuntimeError as exc:
        return _err(str(exc))

    payload = json.dumps(records, indent=2)
    if args.out == "-":
        print(payload)
    else:
        Path(args.out).write_text(payload)
        print(f"wrote {len(records)} clips to {args.out}", file=sys.stderr)

    if not args.quiet and records:
        _summarize(records)
    return 0


def _summarize(records: list[dict]) -> None:
    """A readable table, because JSON on a terminal is not a result."""
    print(f"\n{'window':<16}{'event':<17}{'shots':<7}{'reaction':<17}"
          f"{'sparc':<8}{'over':<6}{'eff'}", file=sys.stderr)
    print("-" * 77, file=sys.stderr)
    for rec in records:
        m, s, e = rec["measured"], rec["source"], rec["event"]
        rt = (f"{m['reaction_time_ms']:.0f}+-{m['reaction_time_error_ms']:.0f}ms"
              if m["reaction_time_ms"] is not None else "not measurable")
        sparc = f"{m['smoothness_sparc']:.2f}" if m["smoothness_sparc"] is not None else "-"
        eff = f"{m['path_efficiency']:.2f}" if m["path_efficiency"] is not None else "-"
        window = f"{s['start_ms']}-{s['end_ms']}"
        print(f"{window:<16}{e['type']:<17}{m['shots_fired']:<7}"
              f"{rt:<17}{sparc:<8}{m['overshoot_count']:<6}{eff}", file=sys.stderr)

    if any("overlay_image" in r for r in records):
        print(f"\n{len(records)} trace images written alongside the results.",
              file=sys.stderr)

    warnings = {w for r in records for w in r["measured"]["cv_warnings"]}
    if warnings:
        print("\nnotes:", file=sys.stderr)
        for w in sorted(warnings):
            print(f"  - {w}", file=sys.stderr)


def cmd_overlay(args: argparse.Namespace) -> int:
    import cv2

    from .metrics import analyze
    from .motion import trace_from_frames
    from .overlay import draw_trace

    video = Path(args.video)
    if not video.exists():
        return _err(f"no such file: {video}")
    try:
        profile = GameProfile.load(args.profile)
    except (FileNotFoundError, ValueError) as exc:
        return _err(str(exc))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return _err(f"could not open {video}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_ms)
        frames = []
        while cap.get(cv2.CAP_PROP_POS_MSEC) <= args.start_ms + args.duration_ms:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) < 2:
        return _err(f"only {len(frames)} frames decoded at {args.start_ms}ms")

    trace, confidence = trace_from_frames(frames, fps, profile.fov_degrees,
                                          profile.flow_exclude)
    metrics = analyze(trace)
    cv2.imwrite(args.out, draw_trace(frames[-1], trace, profile.fov_degrees,
                                     metrics=metrics))
    print(f"wrote {args.out}  ({len(frames)} frames, trace confidence "
          f"{confidence:.2f}, {metrics.overshoot_count} reversals)", file=sys.stderr)
    return 0


def cmd_profiles(args: argparse.Namespace) -> int:
    names = GameProfile.available()
    if not names:
        return _err("no profiles bundled — this build is broken")
    for name in names:
        p = GameProfile.load(name)
        print(f"{p.game:<12} fov={p.fov_degrees:<7} rois={', '.join(sorted(p.rois))}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    for key, value in runtime_report().items():
        print(f"{key:<15} {value}")
    print(f"{'profiles':<15} {', '.join(GameProfile.available()) or 'NONE — broken build'}")
    from .resources import schema_path
    print(f"{'schema':<15} {'ok' if schema_path().exists() else 'MISSING — broken build'}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Render footage along a known path and check the CV recovers it.

    Worth its own subcommand in a shipped binary: it answers "is this download
    working on my machine" without needing a game recording to hand.
    """
    import numpy as np

    from .metrics import analyze
    from .motion import trace_from_frames
    from .synthetic import flick_path, make_world, render_clip

    fps, fov = 60.0, 103.0
    print("rendering synthetic footage along a known 45deg flick…", file=sys.stderr)
    truth = flick_path(36, fps, 45.0, 0.12)
    frames = render_clip(truth, np.zeros_like(truth), world=make_world(),
                         h_fov_deg=fov, width=640, height=360, distractor=False)
    trace, confidence = trace_from_frames(frames, fps, fov)
    error = float(np.max(np.abs(trace.yaw_deg - truth)))
    metrics = analyze(trace, shot_frames=[len(frames) - 1])

    print(f"  recovered peak error : {error:.3f} deg", file=sys.stderr)
    print(f"  trace confidence     : {confidence:.2f}", file=sys.stderr)
    print(f"  smoothness (SPARC)   : {metrics.smoothness_sparc:.3f}", file=sys.stderr)

    tolerance = 1.2      # looser than the test suite: 640x360 has fewer features
    if error > tolerance or confidence < 0.8:
        print(f"\nFAILED — expected error < {tolerance} deg and confidence > 0.8",
              file=sys.stderr)
        return 1
    print("\nOK — computer vision pipeline is working", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gameplay-analyzer",
        description="Analyze gameplay recordings for aim mechanics. Runs fully "
                    "offline; no account, no upload, no API key required.",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="analyze a recording and emit clip records")
    p.add_argument("video")
    p.add_argument("--profile", default="valorant",
                   help="game profile name or path to a yaml file (default: valorant)")
    p.add_argument("--max-clips", type=int, default=8,
                   help="how many engagements to keep (default: 8)")
    p.add_argument("--out", default="-", help="output JSON path, or - for stdout")
    p.add_argument("--overlay-dir", default=None, metavar="DIR",
                   help="also write a crosshair-trace image per clip into DIR")
    p.add_argument("--quiet", action="store_true", help="suppress the summary table")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("overlay", help="render the crosshair trace over a frame")
    p.add_argument("video")
    p.add_argument("--profile", default="valorant")
    p.add_argument("--start-ms", type=int, default=0)
    p.add_argument("--duration-ms", type=int, default=3000)
    p.add_argument("--out", default="overlay.png")
    p.set_defaults(func=cmd_overlay)

    p = sub.add_parser("profiles", help="list bundled game profiles")
    p.set_defaults(func=cmd_profiles)

    p = sub.add_parser("doctor", help="report versions and bundled resources")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("selftest", help="verify the CV pipeline against known ground truth")
    p.set_defaults(func=cmd_selftest)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
