#!/usr/bin/env python3
"""Run every test module. Exit non-zero if any fails."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ["test_schema.py", "test_metrics.py", "test_motion.py", "test_events.py"]

if __name__ == "__main__":
    failed = []
    for mod in MODULES:
        path = HERE / mod
        if not path.exists():
            continue
        print(f"\n\033[1m── {mod} {'─' * (58 - len(mod))}\033[0m")
        if subprocess.run([sys.executable, str(path)]).returncode:
            failed.append(mod)
    print(f"\n{'=' * 62}")
    print(f"FAILED: {', '.join(failed)}" if failed else "all modules passed")
    raise SystemExit(1 if failed else 0)
