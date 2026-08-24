"""Behave like a desktop app when launched from a file manager.

A single executable that people drag a video onto beats an exe plus a launcher
script: two files that must stay together is a failure mode with no upside, and
it reached a real user as "Could not find gameplay-analyzer.exe".

Everything here is about the difference between being run from a shell — where
the user typed a command and wants terse output and a clean exit — and being
double-clicked, where the window vanishes the instant the process ends unless
something holds it open.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v", ".flv", ".wmv"}


def launched_from_file_manager() -> bool:
    """True when this process owns its console alone.

    Windows gives a double-clicked or drag-target program a brand new console
    with only that process attached; a program started from cmd or PowerShell
    shares the shell's console, so the list has at least two entries. That
    distinction is what tells a desktop launch from a terminal one, and it is
    more reliable than checking whether stdout is a tty (it is, in both cases).

    Always False off Windows: double-clicking on macOS or Linux does not
    produce a console for this to reason about.
    """
    if sys.platform != "win32":
        return False
    if os.environ.get("CI"):
        # Belt and braces: never hold a CI runner open waiting for a keypress.
        return False
    try:
        import ctypes

        buf = (ctypes.c_uint * 8)()
        count = ctypes.windll.kernel32.GetConsoleProcessList(buf, 8)
        return count == 1
    except Exception:
        # Any failure here must not stop the tool doing its actual job.
        return False


def looks_like_a_video(arg: str) -> bool:
    """Is this argument a dropped file rather than a subcommand?

    Requires the path to exist. A bare word that happens to end in .mp4 is more
    likely a typo than a file, and guessing wrong turns a clear "unknown
    command" into a confusing "no such file".
    """
    if arg.startswith("-"):
        return False
    p = Path(arg)
    return p.suffix.lower() in VIDEO_SUFFIXES and p.exists()


def rewrite_dropped_file(argv: list[str], known_commands: set[str]) -> list[str] | None:
    """Turn `gameplay-analyzer C:\\clips\\match.mp4` into an analyze invocation.

    Returns None when argv is already a normal command line, so the parser sees
    exactly what it always did and terminal use is untouched.
    """
    if not argv or argv[0] in known_commands or not looks_like_a_video(argv[0]):
        return None

    video = Path(argv[0])
    out_dir = video.parent / f"{video.stem}_analysis"
    return [
        "analyze", str(video),
        "--overlay-dir", str(out_dir),
        "--out", str(out_dir / "results.json"),
        *argv[1:],
    ]


def hold_window_open(message: str = "Press Enter to close this window...") -> None:
    """Stop a double-clicked window from vanishing before it can be read."""
    try:
        print(f"\n{message}", file=sys.stderr)
        sys.stdin.readline()
    except (EOFError, KeyboardInterrupt, OSError):
        pass


def open_folder(path: str | Path) -> None:
    """Show the results to someone who will not go looking for them."""
    try:
        if sys.platform == "win32":
            os.startfile(str(path))            # noqa: S606 - Windows-only API
        elif sys.platform == "darwin":
            import subprocess
            subprocess.run(["open", str(path)], check=False)
        else:
            import subprocess
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass          # Opening a folder is a convenience, never a failure.


WELCOME = """
  ==========================================================
    Gameplay Analyzer
  ==========================================================

  To analyze a recording: drag a video file ONTO this
  program's icon and let go.

  Works with .mp4 and .webm recordings - whatever your
  capture software saved, from OBS, ShadowPlay, or the
  Xbox Game Bar.

  Record at 60fps or higher if you can. Below that the
  timing measurements are not reliable enough to report,
  and this tool says so rather than guessing.

  ----------------------------------------------------------
  Running a self-check now to confirm everything works on
  this PC...
  ----------------------------------------------------------
"""
