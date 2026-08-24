"""Desktop-launch behavior: dropped files, double-clicks, terminal use.

The point of this module is that one executable can serve two audiences without
either noticing the other. Terminal use must be byte-for-byte what it always
was; a dropped video must Just Work for someone who has never seen a command
line. These tests pin both.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.cli import KNOWN_COMMANDS, build_parser
from analyzer.desktop import (
    launched_from_file_manager, looks_like_a_video, rewrite_dropped_file,
)


def _tmp_video(name: str = "match.mp4") -> Path:
    d = Path(tempfile.mkdtemp())
    v = d / name
    v.write_bytes(b"\x00" * 64)
    return v


# ── recognising a dropped file ───────────────────────────────────────────────

def test_recognises_real_video_files():
    for suffix in (".mp4", ".webm", ".mkv", ".mov", ".MP4"):
        v = _tmp_video(f"clip{suffix}")
        assert looks_like_a_video(str(v)), f"{suffix} not recognised"
    print("    .mp4 .webm .mkv .mov and uppercase all recognised")


def test_requires_the_file_to_exist():
    """A bare word ending in .mp4 is likelier a typo than a file, and guessing
    turns a clear 'unknown command' into a confusing 'no such file'."""
    assert not looks_like_a_video("notafile.mp4")
    assert not looks_like_a_video("analyze")


def test_flags_are_never_dropped_files():
    assert not looks_like_a_video("--version")
    assert not looks_like_a_video("-h")


# ── argv rewriting ───────────────────────────────────────────────────────────

def test_dropped_video_becomes_an_analyze_command():
    v = _tmp_video()
    out = rewrite_dropped_file([str(v)], KNOWN_COMMANDS)
    assert out is not None
    assert out[0] == "analyze" and out[1] == str(v)
    assert "--overlay-dir" in out and "--out" in out
    # Results land beside the video, in a folder named after it.
    assert out[out.index("--overlay-dir") + 1].endswith("match_analysis")
    print(f"    drop -> {out[0]} + overlay-dir named {Path(out[3]).name}")


def test_the_rewritten_command_actually_parses():
    """Rewriting to something argparse rejects would be worse than not
    rewriting at all."""
    v = _tmp_video()
    out = rewrite_dropped_file([str(v)], KNOWN_COMMANDS)
    args = build_parser().parse_args(out)
    assert args.command == "analyze"
    assert args.video == str(v)
    assert args.overlay_dir and args.out
    print(f"    parses cleanly: command={args.command!r}")


def test_terminal_invocations_are_untouched():
    """The load-bearing guarantee: existing CLI use must see exactly what it
    always saw, so adding desktop support cannot regress scripts."""
    for argv in (
        ["analyze", "x.mp4"],
        ["overlay", "x.mp4", "--out", "y.png"],
        ["profiles"], ["doctor"], ["selftest"],
        ["--version"], ["-h"], [],
    ):
        assert rewrite_dropped_file(argv, KNOWN_COMMANDS) is None, f"{argv} was rewritten"
    print("    8 terminal invocations pass through unchanged")


def test_extra_flags_survive_a_drop():
    v = _tmp_video()
    out = rewrite_dropped_file([str(v), "--profile", "aimlabs"], KNOWN_COMMANDS)
    assert out[-2:] == ["--profile", "aimlabs"]
    assert build_parser().parse_args(out).profile == "aimlabs"


# ── launch detection ─────────────────────────────────────────────────────────

def test_never_treats_a_test_run_as_a_desktop_launch():
    """If this ever returned True in CI, the runner would block forever on a
    keypress that is never coming."""
    assert launched_from_file_manager() is False


def test_launch_detection_is_windows_only(monkeypatch=None):
    """Double-clicking on macOS or Linux produces no console to reason about,
    so the check must not fire there."""
    assert sys.platform == "win32" or launched_from_file_manager() is False


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
